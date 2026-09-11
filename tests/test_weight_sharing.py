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


def _pipeline(device: DeviceRef, *, language: dict | None = None, driver=None) -> UnlimitedOcrPipeline:
    return UnlimitedOcrPipeline(
        _config(int8=False),
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
    checkpoint dtypes -- bf16 dense, int8 stacks, fp32 scales -- go through
    ``Buffer.from_dlpack`` uniformly (numpy has no bf16), and each round-trips
    bit for bit.
    """
    from max.driver import CPU, Accelerator, Buffer

    host = {
        "dense": torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.bfloat16),
        "stack": torch.arange(-4, 4, dtype=torch.int8).reshape(2, 4),
        "scales": torch.linspace(0.5, 1.5, 8, dtype=torch.float32).reshape(2, 4),
    }
    originals = {name: tensor.clone() for name, tensor in host.items()}
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
