"""KON-161: one shared device copy of the language weights, and the release policy.

On an accelerator both language graphs declare their weights device-side
(``declare_device_resident_weights``) and bind ONE registry of device
``Buffer``s (``UnlimitedOcrPipeline._resolved_language_weights``), instead of
each ``session.load`` materialising its own ~5.5 GiB copy. The release policy
-- one language graph at a time, ``releases_language_graphs`` -- is a separate
predicate on purpose: sharing the weights does not make both graphs fit.

Everything here is model-free except the registry-ownership check, which needs
a real accelerator for its tiny device buffers (still no model, no compile).
The graph checks stage only (``str(graph)``), the way ``test_decoder_int8.py``
does, and reuse its synthetic config and op census.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from max.graph import DeviceRef

from unlimited_ocr_max.decoder import UnlimitedOcrDecoder
from unlimited_ocr_max.graphs import build_decode_graph, build_language_graph
from unlimited_ocr_max.model_config import UnlimitedOCRConfig
from unlimited_ocr_max.pipeline import SHARE_LANGUAGE_WEIGHTS_DEFAULT, UnlimitedOcrPipeline

from test_decoder_int8 import _config, _named, _op_counts


def _accelerator_count() -> int:
    from max.driver import accelerator_count

    return int(accelerator_count())


gpu_only = pytest.mark.skipif(
    _accelerator_count() == 0, reason="needs an accelerator: the registry is device buffers"
)


class _Session:
    """Never used: the pipeline consumes its session lazily."""


def _pipeline(
    device: DeviceRef, *, language: dict | None = None, driver=None, config=None
) -> UnlimitedOcrPipeline:
    return UnlimitedOcrPipeline(
        _config(int8=False) if config is None else config,
        vision_state_dict={},
        language_state_dict={} if language is None else language,
        seq_len=277,
        device=device,
        driver_device=driver,
        session=_Session(),
    )


# --------------------------------------------------------------------------
# polarity: the constant, the accelerator gate, the override in both directions
# --------------------------------------------------------------------------


def test_language_weight_sharing_is_on_and_accelerator_only() -> None:
    """The shipped default is ON, and the predicate is ``on_accelerator``-gated.

    Two independent things that can silently flip apart: the constant carries
    the policy, the conjunction keeps it off CPU -- where a host array *is* the
    graph's memory, so there is no duplicate device copy to remove and the
    numerically gated path stays byte-identical by construction.
    """
    assert SHARE_LANGUAGE_WEIGHTS_DEFAULT is True

    cpu, gpu = _pipeline(DeviceRef.CPU()), _pipeline(DeviceRef.GPU(0))
    assert cpu.shares_language_weights is False
    assert gpu.shares_language_weights is True
    # On CPU the registry values are the host arrays, unchanged.
    assert cpu._resolved_language_weights() is cpu._language_state_dict

    # The attribute is an override a probe sets either way; the accelerator
    # gate wins in both directions.
    cpu._share_language_weights = True
    assert cpu.shares_language_weights is False
    assert cpu._resolved_language_weights() is cpu._language_state_dict
    gpu._share_language_weights = False
    assert gpu.shares_language_weights is False
    assert gpu._resolved_language_weights() is gpu._language_state_dict


def test_int8_gates_the_sharing_off_and_bf16_keeps_it() -> None:
    """KON-162/KON-161 round 2: the int8 variant serves with the sharing OFF.

    The served int8 identity gate at the registry commit read **0/12 in both
    request orders** against the pinned KON-149 transcripts -- deterministic
    across three server processes -- while every in-process value-level A/B is
    bitwise green for both variants: the bare MoE layer (synthetic, production
    stack shapes), the real-weight decode graph at the served shape, the served
    prefill -> release -> decode sequence, and full real pages generated to EOS
    (toc_dotted 646/646 tokens, byte-identical to the pinned transcript under
    BOTH flag settings). The divergence is therefore not localisable at the
    graph/pipeline level, so int8 forgoes the shared registry until a served
    gate clears it; bf16 keeps it (the mechanism served 12/12 in both request
    orders on the research port, KON-160).
    """
    bf16 = _pipeline(DeviceRef.GPU(0))
    int8 = _pipeline(DeviceRef.GPU(0), config=_config(int8=True))
    assert bf16.shares_language_weights is True
    assert int8.shares_language_weights is False
    # The probe override cannot force it back on for int8: the variant gate is
    # part of the conjunction, not a default the attribute can overwrite.
    int8._share_language_weights = True
    assert int8.shares_language_weights is False
    # And the registry path is never taken: an int8 pipeline binds the host
    # tensors exactly as the pinned-transcript builds did.
    assert int8._resolved_language_weights() is int8._language_state_dict
    # The release policy is the device, not the sharing flag (KON-113): int8
    # still holds one language graph at a time.
    assert int8.releases_language_graphs is True


def test_releases_language_graphs_is_the_device_not_the_sharing_flag() -> None:
    """One graph at a time is ``on_accelerator`` verbatim, NOT gated by sharing.

    KON-113 measured hold-both over the Metal budget *with* sharing on (18.629
    of 17.760 GiB, short from both load orders), so turning the sharing off or
    on must not move the release policy in either direction.
    """
    cpu, gpu = _pipeline(DeviceRef.CPU()), _pipeline(DeviceRef.GPU(0))
    assert cpu.releases_language_graphs is False
    assert gpu.releases_language_graphs is True
    gpu._share_language_weights = False
    assert gpu.releases_language_graphs is True
    cpu._share_language_weights = True
    assert cpu.releases_language_graphs is False


# --------------------------------------------------------------------------
# releases: idempotent, safe before build, and the registry survives them
# --------------------------------------------------------------------------


def test_releases_are_idempotent_safe_before_build_and_keep_the_registry() -> None:
    pipeline = _pipeline(DeviceRef.GPU(0))
    # Nothing built yet: both must be no-ops, not errors.
    pipeline.release_decode()
    pipeline.release_prefill()

    # Sentinels first: asserting `is None` on attributes nothing ever set
    # passes whether or not the methods do anything at all.
    pipeline._decode = "decode-sentinel"
    pipeline._prefill[277] = "prefill-sentinel"
    pipeline._language_device_weights = {"w": "registry-sentinel"}
    pipeline.release_decode()
    pipeline.release_decode()
    pipeline.release_prefill()
    pipeline.release_prefill()
    assert pipeline._decode is None
    assert pipeline._prefill == {}
    # The shared registry outlives every graph: both graphs bind it across the
    # release/reload cycle, and it cannot be rebuilt (the host arrays are gone
    # once it exists). Dropping it here would break the next reload outright.
    assert pipeline._language_device_weights == {"w": "registry-sentinel"}


def test_the_served_prefill_drops_the_decode_graph_first() -> None:
    """The structural invariant: ``_prefill`` releases decode before anything else.

    On an accelerator the decode release comes unconditionally first -- before
    the vision tower runs, before the prefill graph loads -- so no failure path
    can reach a prefill load with the previous request's decode graph resident.
    And it reads the NAMED predicate, so a CPU pipeline is untouched. Driven
    with a stub pipeline: what is under test is which calls ``_prefill`` makes
    and in what order, not a ~18 GiB graph load.
    """
    from unlimited_ocr_max.model import UnlimitedOCRModel

    class _Buffer:
        def to(self, _device):
            return self

        def to_numpy(self):
            return np.zeros((3, 8, 8), dtype=np.float32)

    class _Result:
        cache = "cache-sentinel"
        logits = np.zeros((1, 4), dtype=np.float32)

    class _Inputs:
        pixel_values = [_Buffer()]
        image_token_indices = None

    class _Pipeline:
        def __init__(self, transient: bool) -> None:
            self.releases_language_graphs = transient
            self.calls: list[str] = []

        def release_decode(self):
            self.calls.append("release_decode")

        def release_prefill(self):
            self.calls.append("release_prefill")

        def run_vision(self, _pixels):
            self.calls.append("run_vision")
            return {"image_embeds": np.zeros((273, 4), dtype=np.float32)}

        def drop_vision_weights(self):
            self.calls.append("drop_vision_weights")

        def run_prefill(self, _tokens, _embeds):
            self.calls.append("run_prefill")
            return _Result()

    for transient, expected in (
        (True, ["release_decode", "run_vision", "drop_vision_weights", "run_prefill", "release_prefill"]),
        (False, ["run_vision", "drop_vision_weights", "run_prefill"]),
    ):
        model = object.__new__(UnlimitedOCRModel)
        model._pipeline = _Pipeline(transient)
        model._served = {}
        logits = UnlimitedOCRModel._prefill(model, "r1", _Inputs(), np.array([1, 2, 3], dtype=np.int64))
        assert model._pipeline.calls == expected, transient
        # The request's state is recorded either way -- a release that also
        # dropped the host KV cache would break decoding entirely.
        assert model._served["r1"].cache == "cache-sentinel"
        assert logits.shape == (1, 4)


# --------------------------------------------------------------------------
# the registry: takes the host tensors, one Buffer per weight, every dtype
# --------------------------------------------------------------------------


@gpu_only
def test_the_registry_takes_the_host_tensors_and_holds_every_dtype() -> None:
    """``_resolved_language_weights`` pops the host entries as it converts.

    Ownership is the design, not a side effect: building the device dict
    beside a full host copy held 2x ~5.5 GiB at once in the research port, so
    each host entry is dropped the moment its buffer exists, and afterwards the
    host mapping is empty and ``_language_state_dict`` is gone. All three
    checkpoint dtypes -- bf16 dense, int8 stacks, fp32 scales -- arrive as host
    ``Buffer``s from the adapter and take the same single ``.to(device)`` step,
    and each round-trips bit for bit.
    """
    from max.driver import CPU, Accelerator, Buffer

    originals = {
        "dense": torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.bfloat16),
        "stack": torch.arange(-4, 4, dtype=torch.int8).reshape(2, 4),
        "scales": torch.linspace(0.5, 1.5, 8, dtype=torch.float32).reshape(2, 4),
    }
    host = {name: Buffer.from_dlpack(tensor) for name, tensor in originals.items()}
    pipeline = _pipeline(DeviceRef.GPU(0), language=host, driver=Accelerator())

    registry = pipeline._resolved_language_weights()
    assert set(registry) == set(originals)
    assert all(isinstance(value, Buffer) for value in registry.values())
    # Ownership taken: the host mapping was emptied entry by entry and dropped.
    assert host == {}
    assert pipeline._language_state_dict is None
    # Idempotent: the same dict, not a second set of device copies.
    assert pipeline._resolved_language_weights() is registry

    for name, tensor in originals.items():
        back = torch.from_dlpack(registry[name].to(CPU()))
        assert back.dtype == tensor.dtype, name
        assert torch.equal(back, tensor), name


@gpu_only
def test_the_registry_releases_each_host_entry_before_it_builds_the_next() -> None:
    """The host mapping shrinks *during* the loop, which is the whole reason it is one.

    ``assert host == {}`` above says only that the entries went, not when: a
    build that made a full second dict and cleared the first afterwards passes
    it while holding 2x ~5.5 GiB (17.83 GiB host peak, measured in the research
    port). So the liveness is recorded instead. A ``Buffer`` cannot be weakly
    referenced, but the numpy array it aliases can, and that array dies exactly
    with the last ``Buffer`` holding it -- which also shows the device copy does
    not keep its source alive. Entry ``k`` dying while ``n-1-k`` entries remain
    is one at a time; a bulk release records every entry at zero.
    """
    import weakref

    from max.driver import Accelerator, Buffer

    names = [f"w{index}" for index in range(4)]
    arrays = [np.full((256, 256), index, dtype=np.float32) for index in range(len(names))]
    host = {name: Buffer.from_numpy(array) for name, array in zip(names, arrays, strict=True)}
    released: list[tuple[str, int]] = []
    for name, array in zip(names, arrays, strict=True):
        weakref.finalize(array, lambda held=name: released.append((held, len(host))))
    del arrays, array, name  # only the host Buffers hold the arrays now

    pipeline = _pipeline(DeviceRef.GPU(0), language=host, driver=Accelerator())
    registry = pipeline._resolved_language_weights()

    assert released == [(name, len(names) - 1 - index) for index, name in enumerate(names)]
    assert host == {}
    assert len(registry) == len(names)


# --------------------------------------------------------------------------
# the declaration: same set of weights, the per-weight transfers disappear
# --------------------------------------------------------------------------


def _graph_text(config: UnlimitedOCRConfig, device: DeviceRef, *, decode: bool, resident: bool) -> str:
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=device))
    if decode:
        staged = build_decode_graph(config, decoder, max_seq_len=64, device=device, device_resident_weights=resident)
    else:
        staged = build_language_graph(
            config, decoder, seq_len=5, n_image_tokens=2, device=device, device_resident_weights=resident
        )
    return str(staged.graph)


#: ``rmo.mo.transfer`` is a multi-result op (``%result, %outChain = rmo.mo.transfer[...]``),
#: which ``test_decoder_int8._op_counts``'s single-result regex never matched --
#: so the int8 op-census test there is untouched by this declaration change,
#: and the transfers get their own count here.
_TRANSFERS = re.compile(r"^\s*%\S+(?:,\s*%\S+)*\s*=\s*\"?rmo\.mo\.transfer\b", re.MULTILINE)

#: A weight declaration and the device it is placed on.
_EXTERNAL_DEVICE = re.compile(r'mo\.constant\.external \{[^}]*device = #M\.device_ref<"(\w+)"')


@pytest.mark.parametrize("int8", [False, True], ids=["bf16", "int8"])
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "prefill"])
def test_declaring_resident_weights_changes_placement_not_the_declaration_set(int8: bool, decode: bool) -> None:
    """Both graphs, both weight variants: the declaration set is unchanged.

    One ``mo.constant.external`` per declared weight either way -- which is why
    ``check_against_declared`` (key/shape/dtype against ``raw_state_dict()``)
    is untouched by the registry path. What moves is placement only: every
    declaration lands on the device instead of on the host, exactly one
    ``rmo.mo.transfer`` per weight disappears with it (the prefill graph keeps
    its two ``ops.nonzero`` CPU round-trip transfers, which are not weight
    transfers), and no op kind the staged census sees changes count.
    """
    config = _config(int8=int8, num_hidden_layers=3)
    dref = DeviceRef.GPU(0)
    n_weights = len(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=dref).raw_state_dict())

    plain_text = _graph_text(config, dref, decode=decode, resident=False)
    resident_text = _graph_text(config, dref, decode=decode, resident=True)

    # The declaration set: one external constant per declared weight, either way.
    plain_devices = _EXTERNAL_DEVICE.findall(plain_text)
    resident_devices = _EXTERNAL_DEVICE.findall(resident_text)
    assert len(plain_devices) == n_weights
    assert len(resident_devices) == n_weights
    # Placement: host declarations become device declarations.
    assert set(plain_devices) == {"cpu"}
    assert set(resident_devices) == {"gpu"}
    # The per-weight host->device transfers disappear, and only those.
    assert len(_TRANSFERS.findall(plain_text)) - len(_TRANSFERS.findall(resident_text)) == n_weights
    # Everything else the census sees is unchanged in kind and count.
    assert _op_counts(plain_text) == _op_counts(resident_text)


# --------------------------------------------------------------------------
# the value-level A/B: a registry-bound graph computes the same BITS (KON-161 r2)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _accelerator_session():
    from max.driver import Accelerator
    from max.engine import InferenceSession

    driver = Accelerator()
    return driver, InferenceSession(devices=[driver])


@pytest.mark.slow
@gpu_only
@pytest.mark.parametrize("int8", [True, False], ids=["int8", "bf16"])
@pytest.mark.parametrize("seq", [1, 4], ids=["decode", "prefill"])
def test_device_resident_registry_is_bitwise_equal_to_plain_load(
    int8: bool, seq: int, _accelerator_session
) -> None:
    """A graph whose weights are pre-added device-side and bound from a device-
    ``Buffer`` registry computes bit-for-bit what the plain (host-declared,
    MAX-placed) load computes -- for the int8 kernel paths (``moe_int8_qmv`` at
    ``seq == 1``, ``int8_dequant_expert`` at ``seq > 1``) and the bf16 paths
    alike.

    This is the value-level net under the sharing feature: declaration-only
    changes must not move a single bit (KON-122's bar). At the registry commit
    it was verified to hold in-process all the way up to full real pages
    (KON-161 round 2), which is exactly why the served int8 falsification
    (KON-162) could not be localised here and int8 is variant-gated instead.
    """
    from max.driver import CPU, Buffer
    from max.dtype import DType
    from max.graph import Graph, TensorType

    from unlimited_ocr_max.decoder import MoE
    from unlimited_ocr_max.ngram import MOJO_KERNELS

    from test_decoder_int8 import _moe_weights, _small_decoder_config

    driver, session = _accelerator_session
    config = _small_decoder_config(int8=int8)
    original, quantized, _ = _moe_weights(config, seed=17)
    weights = quantized if int8 else original
    dref = DeviceRef.GPU(0)

    def load(resident: bool):
        moe = MoE(config, dtype=DType.bfloat16, device=dref)
        moe.load_state_dict(weights)
        extensions = {"custom_extensions": [MOJO_KERNELS]} if int8 else {}
        with Graph(
            f"registry_ab_{'int8' if int8 else 'bf16'}_{seq}_{resident}",
            input_types=[TensorType(DType.float32, [seq, config.hidden_size], device=dref)],
            **extensions,
        ) as graph:
            if resident:
                for weight in moe.raw_state_dict().values():
                    graph.add_weight(weight, force_initial_weight_on_host=False)
            graph.output(moe(graph.inputs[0].tensor))
        if resident:
            registry = {name: Buffer.from_dlpack(t).to(driver) for name, t in weights.items()}
            driver.synchronize()
        else:
            registry = moe.state_dict()
        return session.load(graph, weights_registry=registry)

    x = np.random.default_rng(23).standard_normal((seq, config.hidden_size)).astype(np.float32)

    def run(model) -> np.ndarray:
        return model.execute(Buffer.from_numpy(np.ascontiguousarray(x)).to(driver))[0].to(CPU()).to_numpy()

    plain = run(load(resident=False))
    resident = run(load(resident=True))
    assert np.array_equal(plain, resident), (
        f"registry-bound graph moved the bits: max abs diff "
        f"{float(np.max(np.abs(plain.astype(np.float64) - resident.astype(np.float64)))):.3e}"
    )
