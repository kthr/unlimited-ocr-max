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
from typing import Any

import numpy as np
import pytest
import torch
from max.graph import DeviceRef

from unlimited_ocr_max.decoder import UnlimitedOcrDecoder
from unlimited_ocr_max.pipeline import SHARE_LANGUAGE_WEIGHTS_DEFAULT, UnlimitedOcrPipeline

from test_decoder_int8 import (
    _SSA_ASSIGN,
    _config,
    _is_weight_only,
    _language_graph,
    _op_counts,
    _parse_ssa_defs,
    _top_level_boundary,
)


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


def test_int8_shares_the_registry_like_bf16() -> None:
    """KON-225: the variant gate is gone -- int8 shares the registry exactly like bf16.

    int8 was excluded here from KON-162 until KON-224 localised OQ-158-A (the
    served int8 identity gate's 0/12-in-both-orders falsification, while every
    in-process value-level A/B stayed bitwise green) to a load-time fp32 fold
    of the prefill dequant expert index, removed by commit ``82ab60f`` and
    pinned by ``test_int8_prefill_dequant_expert_index_is_not_load_foldable``.
    With that gate lifted int8 serves 12/12 byte-identical on both MAX 26.6.0
    and the 26.7 nightly, so it now shares the registry under the same
    accelerator-only conjunction as bf16, with no variant exception left to
    spell.
    """
    bf16 = _pipeline(DeviceRef.GPU(0))
    int8 = _pipeline(DeviceRef.GPU(0), config=_config(int8=True))
    assert bf16.shares_language_weights is True
    assert int8.shares_language_weights is True
    # The probe override works both ways for int8 now, exactly as for bf16:
    # no variant gate left to fight it.
    int8._share_language_weights = False
    assert int8.shares_language_weights is False
    # And with sharing off, the registry path is not taken: the pipeline binds
    # the host tensors directly.
    assert int8._resolved_language_weights() is int8._language_state_dict
    int8._share_language_weights = True
    assert int8.shares_language_weights is True
    # The release policy follows the registry, not the variant (KON-113):
    # int8 now holds both language graphs, like bf16.
    assert int8.releases_language_graphs is False


def test_releases_language_graphs_only_without_the_shared_registry() -> None:
    """One graph at a time exactly where the graphs carry their own weights: an accelerator without the registry.

    With the registry the graphs hold no weights of their own since 0.3.2
    (prefill 0.002 GiB, decode 0.000 GiB beyond it), so bf16 and int8 alike
    hold both on an accelerator (KON-225). Without it -- reachable today only
    by forcing the sharing flag off -- each graph places its own copy, which
    KON-113 measured over the Metal budget. CPU never releases.
    """
    cpu, gpu = _pipeline(DeviceRef.CPU()), _pipeline(DeviceRef.GPU(0))
    int8 = _pipeline(DeviceRef.GPU(0), config=_config(int8=True))
    assert cpu.releases_language_graphs is False
    assert gpu.releases_language_graphs is False
    assert int8.releases_language_graphs is False
    gpu._share_language_weights = False
    assert gpu.releases_language_graphs is True
    int8._share_language_weights = False
    assert int8.releases_language_graphs is True
    cpu._share_language_weights = True
    assert cpu.releases_language_graphs is False


def test_retain_only_prefill_keeps_just_the_current_length() -> None:
    pipeline = _pipeline(DeviceRef.GPU(0))
    pipeline.retain_only_prefill(282)  # nothing cached: a no-op
    pipeline._prefill.update({277: "p277", 282: "p282", 300: "p300"})
    pipeline.retain_only_prefill(282)
    assert pipeline._prefill == {282: "p282"}
    pipeline.retain_only_prefill(277)  # a length not cached drops the rest too
    assert pipeline._prefill == {}


# --------------------------------------------------------------------------
# releases: idempotent, safe before build, and the registry survives them
# --------------------------------------------------------------------------


def test_releases_are_idempotent_safe_before_build_and_keep_the_registry() -> None:
    pipeline = _pipeline(DeviceRef.GPU(0))
    # Nothing built yet: both must be no-ops, not errors.
    pipeline.release_decode()
    pipeline.release_prefill()

    # Sentinels first: asserting emptiness on caches nothing ever filled
    # passes whether or not the methods do anything at all.
    pipeline._decode[1] = "decode-sentinel"
    pipeline._prefill[277] = "prefill-sentinel"
    pipeline._language_device_weights = {"w": "registry-sentinel"}
    pipeline.release_decode()
    pipeline.release_decode()
    pipeline.release_prefill()
    pipeline.release_prefill()
    assert pipeline._decode == {}
    assert pipeline._prefill == {}
    # The shared registry outlives every graph: both graphs bind it across the
    # release/reload cycle, and it cannot be rebuilt (the host arrays are gone
    # once it exists). Dropping it here would break the next reload outright.
    assert pipeline._language_device_weights == {"w": "registry-sentinel"}


def test_the_served_prefill_drops_the_decode_graph_first() -> None:
    """The structural invariant: where it releases, ``_prefill`` releases decode before anything else.

    On an accelerator the decode release comes unconditionally first -- before
    the vision tower runs, before the prefill graph loads -- so no failure path
    can reach a prefill load with the previous request's decode graph resident.
    It reads the NAMED predicate; with both graphs resident the prefill cache
    is bounded instead (the model refuses a CPU device at construction, so
    there is no third case). Driven with a stub pipeline: what is under test is
    which calls ``_prefill`` makes and in what order, not a ~18 GiB graph load.
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

    class _Pipeline:
        def __init__(self, transient: bool) -> None:
            self.releases_language_graphs = transient
            self.calls: list[str] = []

        def release_decode(self):
            self.calls.append("release_decode")

        def retain_only_prefill(self, seq_len):
            self.calls.append(f"retain_only_prefill({seq_len})")

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
        # an accelerator without the registry (since KON-225 reachable only by
        # forcing the sharing flag off, bf16 and int8 alike): one language graph at a time
        (True, ["release_decode", "run_vision", "drop_vision_weights", "run_prefill", "release_prefill"]),
        # an accelerator with the registry (the default, bf16 and int8 alike):
        # both graphs stay, the prefill cache is bounded
        (False, ["retain_only_prefill(3)", "run_vision", "drop_vision_weights", "run_prefill"]),
    ):
        model = object.__new__(UnlimitedOCRModel)
        model._pipeline = _Pipeline(transient)
        model._served = {}
        logits = UnlimitedOCRModel._prefill(model, "r1", np.array([1, 2, 3], dtype=np.int64), _Buffer(), None)
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
    # Ownership taken: the host mapping is empty and dropped. This is the end
    # state only; *when* each entry goes is the loop's invariant, and it is
    # documented on `_resolved_language_weights` rather than pinned here.
    assert host == {}
    assert pipeline._language_state_dict is None
    # Idempotent: the same dict, not a second set of device copies.
    assert pipeline._resolved_language_weights() is registry

    for name, tensor in originals.items():
        back = torch.from_dlpack(registry[name].to(CPU()))
        assert back.dtype == tensor.dtype, name
        assert torch.equal(back, tensor), name


def test_a_cpu_registry_binds_a_stacked_buffer_straight_into_a_graph() -> None:
    """The CPU arm: no sharing, so the adapter's own values are what ``session.load`` binds.

    Everything above is about the accelerator registry, where the values are
    converted before MAX sees them. On CPU :attr:`shares_language_weights` is
    False and the host mapping goes through untouched -- so the one thing that
    has to hold is that a stack built by ``stack_expert_weights`` (a ``Buffer``
    re-viewed as bfloat16 over a uint16 ``np.stack``) is something MAX will load
    and read correctly. Deliberately not ``gpu_only``: this arm is the one that
    runs everywhere, CI included, and it is the one nothing else covers.
    """
    from max.driver import CPU, Buffer
    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import Graph, Weight, ops

    from unlimited_ocr_max.weight_adapters import stack_expert_weights

    experts, rows, cols = 3, 2, 4
    stem = "layers.1.mlp.experts"
    # Each expert carries its own index, so a misordered stack cannot pass.
    state = {
        f"{stem}.{index}.gate_proj.weight": Buffer.from_dlpack(
            torch.full((rows, cols), float(index) + 0.5, dtype=torch.bfloat16)
        )
        for index in range(experts)
    }
    name = f"{stem}.gate_proj"
    stack = stack_expert_weights(state, num_experts=experts)[name]

    weight = Weight(name, DType.bfloat16, [experts, rows, cols], device=DeviceRef.CPU())
    with Graph("cpu_registry", input_types=[]) as graph:
        graph.output(ops.cast(graph.add_weight(weight), DType.float32))
    model = InferenceSession(devices=[CPU()]).load(graph, weights_registry={name: stack})

    got = model.execute()[0].to_numpy()
    want = np.stack([np.full((rows, cols), float(index) + 0.5, dtype=np.float32) for index in range(experts)])
    assert got.dtype == np.float32
    assert np.array_equal(got, want), f"the graph read {got[:, 0, 0]}, expected {want[:, 0, 0]}"


# --------------------------------------------------------------------------
# the declaration: same set of weights, the per-weight transfers disappear
# --------------------------------------------------------------------------


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

    plain_text = str(_language_graph(config, dref, decode=decode).graph)
    resident_text = str(_language_graph(config, dref, decode=decode, resident=True).graph)

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
    (``moe_bf16_qmv`` at ``seq == 1``, MAX's native routing at ``seq > 1``)
    alike.

    This is the value-level net under the sharing feature: declaration-only
    changes must not move a single bit (KON-122's bar). At the registry commit
    it was verified to hold in-process all the way up to full real pages
    (KON-161 round 2) -- which is exactly why the served int8 falsification
    (KON-162) could not be localised here either: the real cause was a
    load-time fold the registry made reachable, traced by KON-224 and removed
    by commit ``82ab60f``. This check never moved; only the pipeline's
    variant gate did, and KON-225 dropped it now that the served gate is
    clear.
    """
    from max.driver import CPU, Buffer
    from max.dtype import DType
    from max.graph import Graph, TensorType

    from unlimited_ocr_max.decoder import MoE
    from unlimited_ocr_max.ngram import MOJO_KERNELS

    from test_decoder_int8 import _as_declared, _moe_weights, _small_decoder_config

    driver, session = _accelerator_session
    config = _small_decoder_config(int8=int8)
    original, quantized, _ = _moe_weights(config, seed=17)
    weights = _as_declared(quantized if int8 else original)
    dref = DeviceRef.GPU(0)

    def load(resident: bool):
        moe = MoE(config, dtype=DType.bfloat16, device=dref)
        moe.load_state_dict(weights)
        # Decode runs a Mojo qmv kernel in both dtypes; int8 prefill dequantizes through one.
        extensions = {"custom_extensions": [MOJO_KERNELS]} if int8 or seq == 1 else {}
        with Graph(
            f"registry_ab_{'int8' if int8 else 'bf16'}_{seq}_{resident}",
            input_types=[TensorType(DType.float32, [seq, config.hidden_size], device=dref)],
            **extensions,
        ) as graph:
            if resident:
                for weight in moe.raw_state_dict().values():
                    graph.add_weight(weight, force_initial_weight_on_host=False)
            xv = graph.inputs[0].tensor
            graph.output(moe.decode_rows(xv) if seq == 1 else moe(xv))
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


# --------------------------------------------------------------------------
# the norms and the router held as fp32 in the registry (OQ-113-A), the projections as bf16 (KON-238)
# --------------------------------------------------------------------------


def _is_fp32_resident(name: str) -> bool:
    """The RMSNorm gammas and the MoE router: the weights MAX's own fp32 ops read whole."""
    return name.endswith("norm.weight") or ".mlp.gate.gate_score." in name


def _is_bf16_projection(name: str) -> bool:
    """A projection KON-238 reads without an upcast: attention, the dense FFN, the shared experts, ``lm_head``."""
    return name == "lm_head.weight" or (name.endswith("_proj.weight") and ".mlp.experts." not in name)


@pytest.mark.parametrize("int8", [False, True], ids=["bf16", "int8"])
def test_norm_router_dtype_widens_exactly_the_norms_and_the_router(int8: bool) -> None:
    """``norm_router_dtype`` re-types the norms and the router and nothing else; the default changes nothing.

    The stacks are read by kernels that take their storage dtype (bf16 by
    ``moe_bf16_qmv`` and ``grouped_matmul_ragged``, int8 by the Mojo ops), the
    table is gathered before its cast, and every projection -- ``lm_head``
    included -- is read by ``dense_bf16_qmv`` or upcast at run time (KON-238),
    so none of those is ever compile-folded and widening them would only cost
    memory. The norms and the router are consumed by an fp32 norm or matmul,
    which is where the fold happens.
    """
    from max.dtype import DType

    config = _config(int8=int8, num_hidden_layers=3)
    plain = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict()
    widened = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, norm_router_dtype=DType.float32).raw_state_dict()
    assert set(plain) == set(widened)
    for name, weight in widened.items():
        if _is_fp32_resident(name):
            assert plain[name].dtype == DType.bfloat16, name
            assert weight.dtype == DType.float32, name
        else:
            assert weight.dtype == plain[name].dtype, name
            assert list(weight.shape) == list(plain[name].shape), name
    assert any(_is_fp32_resident(name) for name in widened)
    # 4 attention projections and 3 FFN projections (dense or shared) per layer, and lm_head
    assert sum(_is_bf16_projection(name) for name in widened) == 7 * config.decoder.num_hidden_layers + 1
    assert all(widened[name].dtype == DType.bfloat16 for name in widened if _is_bf16_projection(name))
    assert "embed_tokens.weight" in widened and any(".mlp.experts." in name for name in widened)


def test_norm_router_dtype_follows_the_registry() -> None:
    """fp32 exactly where the shared registry is: bf16 and int8 alike on an accelerator; ``None`` on CPU."""
    from max.dtype import DType

    assert _pipeline(DeviceRef.GPU(0)).norm_router_dtype == DType.float32
    assert _pipeline(DeviceRef.CPU()).norm_router_dtype is None
    assert _pipeline(DeviceRef.GPU(0), config=_config(int8=True)).norm_router_dtype == DType.float32
    unshared = _pipeline(DeviceRef.GPU(0))
    unshared._share_language_weights = False
    assert unshared.norm_router_dtype is None
    assert unshared._fp32_resident_names() == frozenset()


@pytest.mark.parametrize("int8", [False, True], ids=["bf16", "int8"])
def test_the_registry_widens_only_the_norms_and_the_router(int8: bool) -> None:
    """KON-238, model-free: the registry's fp32 set is the norms and the router; every projection stays bf16.

    (int8's expert scales are declared fp32 in either mode and are never widened: they are fp32 already.)
    """
    config = _config(int8=int8, num_hidden_layers=3)
    pipeline = _pipeline(DeviceRef.GPU(0), config=config)
    declared = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict()
    widened = frozenset(name for name in pipeline._fp32_resident_names() if not name.endswith("_scales"))
    assert widened == frozenset(name for name in declared if _is_fp32_resident(name))
    assert not any(_is_bf16_projection(name) for name in pipeline._fp32_resident_names())


#: One staged tensor type's dtype: ``!mo.tensor<[1280, 1280], bf16, gpu:0>`` -> ``bf16``.
_TENSOR_DTYPE = re.compile(r"!mo\.tensor<\[[^\]]*\],\s*(\w+)")
_WEIGHT_NAME = re.compile(r'mo\.constant\.external \{[^}]*name = "([^"]+)"')


def _result_dtypes(text: str) -> dict[str, str]:
    """Every SSA result's tensor dtype, read off its type signature (results after ``->``, or the declared type)."""
    dtypes: dict[str, str] = {}
    for line in text.splitlines():
        assign = _SSA_ASSIGN.match(line)
        if not assign:
            continue
        rhs = assign.group(2)
        if "->" in rhs:
            tail = rhs.rsplit("->", 1)[1]
        else:
            boundary = _top_level_boundary(rhs)
            tail = "" if boundary is None else rhs[boundary:]
        names = [name.strip() for name in assign.group(1).split(",")]
        dtypes.update(zip(names, _TENSOR_DTYPE.findall(tail)))
    return dtypes


def _foldable_projection_upcasts(text: str) -> tuple[list[str], list[str]]:
    """``(fp32 weight-only values, matmuls with a weight-only operand)`` derived from a bf16 projection weight.

    MAX folds a weight-only expression into a device constant at
    ``session.load`` (KON-142, OQ-113-A), so a projection weight must never
    reach an fp32 value through weight-only ops -- an explicit ``mo.cast`` --
    nor reach MAX's matmul weight-only: an ``rmo.matmul`` of an fp32
    activation and a bf16 weight stages no cast but lowers to one, and MAX
    folded it into an fp32 copy at load (``probe_b``: +0.617 GiB for
    ``lm_head``). Weight-only views (transpose, slice) of the bf16 weight
    stay bf16 and are fine, as is a custom op reading the weight itself.
    """
    defs = _parse_ssa_defs(text)
    dtypes = _result_dtypes(text)
    projections = {
        result: match.group(1)
        for result, line in ((line.split("=")[0].strip(), line) for line in text.splitlines())
        if (match := _WEIGHT_NAME.search(line)) and _is_bf16_projection(match.group(1))
    }
    assert projections and all(dtypes[result] == "bf16" for result in projections), "a projection is not declared bf16"
    weight_only: dict[str, bool] = {}
    derived: dict[str, bool] = {}

    def from_projection(name: str) -> bool:
        if name not in derived:
            entry = defs.get(name)
            derived[name] = name in projections or (
                entry is not None and _is_weight_only(name, defs, weight_only) and any(map(from_projection, entry[1]))
            )
        return derived[name]

    upcasts = [name for name in defs if from_projection(name) and dtypes.get(name) == "f32"]
    matmuls = [
        name
        for name, (op, operands) in defs.items()
        if op == "rmo.matmul" and any(from_projection(operand) for operand in operands)
    ]
    return upcasts, matmuls


def _unblocked(weight: Any) -> Any:
    """A K-blocked ``[K / KBLOCK, N, KBLOCK]`` weight value as ``[N, K]``."""
    blocks, rows, width = (int(dim) for dim in weight.shape)
    return weight.permute([1, 0, 2]).reshape((rows, blocks * width))


@pytest.mark.parametrize("int8", [False, True], ids=["bf16-gpu", "int8-gpu"])
@pytest.mark.parametrize(("decode", "batch"), [(False, 1), (True, 1), (True, 8)], ids=["prefill", "decode-b1", "decode-b8"])
def test_no_language_graph_upcasts_a_projection_weight_foldably(
    int8: bool, decode: bool, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KON-238's acceptance, model-free: every projection weight is declared bf16, and no staged graph -- prefill,
    decode at B = 1 and B = 8 -- upcasts one to fp32 in a weight-only (load-foldable) expression.

    The decode graphs read every projection through ``dense_bf16_qmv``, the
    prefill its ``lm_head`` row too, and its other projections through a
    run-time upcast (``decoder._runtime_one``). The detector has teeth: a
    constant ``one``, or MAX's mixed matmul in place of the kernel, must flip
    it.
    """
    import unlimited_ocr_max.decoder as decoder_module
    from max.dtype import DType
    from max.graph import ops

    config = _config(int8=int8, num_hidden_layers=3)
    dref = DeviceRef.GPU(0)
    def served_text() -> str:
        return str(_language_graph(config, dref, decode=decode, batch=batch, served=True).graph)

    upcasts, matmuls = _foldable_projection_upcasts(served_text())
    assert not upcasts and not matmuls, (upcasts[:5], matmuls[:5])

    with monkeypatch.context() as patch:
        if decode:
            # MAX's mixed matmul on the K-blocked weight put back in [N, K] order (weight-only views).
            patch.setattr(decoder_module, "dense_bf16_qmv", lambda x, weight, **_: x @ _unblocked(weight).T)
        else:
            patch.setattr(
                decoder_module,
                "_runtime_one",
                lambda x, dtype: ops.cast(ops.constant(np.ones((1, 1), np.float32), DType.float32, device=x.device), dtype),
            )
        upcasts, matmuls = _foldable_projection_upcasts(served_text())
    assert matmuls, "the detector missed a projection weight reaching MAX's matmul weight-only"
    assert decode or upcasts, "the detector missed a weight-only fp32 upcast of a projection weight"


@gpu_only
def test_the_registry_widens_the_norms_and_the_router_exactly() -> None:
    """Declared-fp32 names become their exact fp32 upcast on the device; projections, stacks and the table stay bf16, bit for bit."""
    from max.driver import CPU, Accelerator, Buffer

    config = _config(int8=False, num_hidden_layers=3)
    pipeline = _pipeline(DeviceRef.GPU(0), driver=Accelerator(), config=config)
    names = sorted(pipeline._fp32_resident_names())
    assert names and all(_is_fp32_resident(name) for name in names)
    picked = [
        names[0],
        "layers.1.mlp.gate.gate_score.weight",
        "lm_head.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.1.mlp.shared_experts.down_proj.weight",
        "embed_tokens.weight",
        "layers.1.mlp.experts.gate_proj",
    ]
    rng = np.random.default_rng(5)
    originals = {name: _bf16_tensor(rng.standard_normal((3, 4))) for name in picked}
    pipeline._language_state_dict = {name: Buffer.from_dlpack(tensor) for name, tensor in originals.items()}

    registry = pipeline._resolved_language_weights()
    for name, tensor in originals.items():
        back = torch.from_dlpack(registry[name].to(CPU()))
        if name in pipeline._fp32_resident_names():
            assert back.dtype == torch.float32, name
            assert torch.equal(back, tensor.to(torch.float32)), name
        else:
            assert back.dtype == torch.bfloat16, name
            assert torch.equal(back, tensor), name


def _bf16_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(torch.bfloat16)


@pytest.mark.slow
@gpu_only
@pytest.mark.parametrize("seq", [1, 4], ids=["decode", "prefill"])
def test_an_fp32_resident_router_matches_a_bf16_router(seq: int, _accelerator_session) -> None:
    """A bf16 MoE with its router held as the exact fp32 upcast against the same MoE with a bf16 router.

    Before KON-238 the bf16 arm was the folded path -- MAX upcast the weight
    for the fp32 matmul at load -- and the two were bitwise equal (KON-122's
    bar). A :class:`~unlimited_ocr_max.decoder.Projection` no longer stages a
    foldable upcast: a bf16 router is upcast at run time. At prefill (MAX's
    native expert kernels) that is still the fp32 matmul on the same values,
    bitwise. At one row (the experts through ``moe_bf16_qmv``) MAX's matmul on
    a run-time weight is not bitwise its matmul on a constant one (1.9e-7 abs
    on Metal), so the bar there is 1e-5 relative. The served router is the
    fp32-resident arm. The shared experts are bf16 in both arms
    (``dense_bf16_qmv`` at one row, a run-time upcast at prefill).
    """
    from max.driver import CPU, Buffer
    from max.dtype import DType
    from max.graph import Graph, TensorType

    from unlimited_ocr_max.decoder import MoE
    from unlimited_ocr_max.ngram import MOJO_KERNELS

    from test_decoder_int8 import _as_declared, _moe_weights, _small_decoder_config

    driver, session = _accelerator_session
    config = _small_decoder_config(int8=False)
    weights = _as_declared(_moe_weights(config, seed=29)[0])
    dref = DeviceRef.GPU(0)

    def load(widen: bool):
        moe = MoE(config, dtype=DType.bfloat16, device=dref, norm_router_dtype=DType.float32 if widen else None)
        declared = moe.raw_state_dict()
        values = {
            name: tensor.to(torch.float32) if declared[name].dtype == DType.float32 else tensor
            for name, tensor in weights.items()
        }
        moe.load_state_dict(values)
        with Graph(
            f"fp32_resident_ab_{seq}_{widen}",
            input_types=[TensorType(DType.float32, [seq, config.hidden_size], device=dref)],
            **({"custom_extensions": [MOJO_KERNELS]} if seq == 1 else {}),
        ) as graph:
            for weight in moe.raw_state_dict().values():
                graph.add_weight(weight, force_initial_weight_on_host=False)
            xv = graph.inputs[0].tensor
            graph.output(moe.decode_rows(xv) if seq == 1 else moe(xv))
        registry = {name: Buffer.from_dlpack(tensor.contiguous()).to(driver) for name, tensor in values.items()}
        driver.synchronize()
        return session.load(graph, weights_registry=registry)

    x = np.random.default_rng(31).standard_normal((seq, config.hidden_size)).astype(np.float32)

    def run(model) -> np.ndarray:
        return model.execute(Buffer.from_numpy(np.ascontiguousarray(x)).to(driver))[0].to(CPU()).to_numpy()

    bf16_router = run(load(widen=False))
    resident = run(load(widen=True))
    delta = float(np.max(np.abs(bf16_router.astype(np.float64) - resident.astype(np.float64))))
    print(f"[uocr] fp32-resident vs bf16 router, {seq} row(s): bitwise {bool(np.array_equal(bf16_router, resident))}, max |delta| {delta:.3e}")
    if seq > 1:
        assert np.array_equal(bf16_router, resident), f"the fp32-resident router moved the bits: max abs diff {delta:.3e}"
    assert delta <= 1e-5 * float(np.max(np.abs(resident)))
