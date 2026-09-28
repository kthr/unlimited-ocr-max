"""KON-212: the batched decode graph (``B >= 2`` requests per execute) and ``pipeline.decode_rows``.

The batch-1 decode graph is untouched, and a golden hash of its staged text
proves it: bitwise identity is a property of the whole emitted graph (KON-122),
so "the batch-1 math did not change" is the graph's to prove, not a reviewer's.
Whether the *batched* graph is bitwise the batch-1 step is not asserted
anywhere: the slow accelerator test prints it, and the real-checkpoint spike
measures it.

Staging checks run weightless on the tiny config (``test_decoder_int8``'s
``_config`` / ``_named``) and reuse ``test_weight_sharing``'s graph text and
placement census. The ``decode_rows`` checks are model-free: a stub
``_execute`` / ``decode_step`` and host :class:`~unlimited_ocr_max.kv_cache.KvCache`
rows.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef

import unlimited_ocr_max.pipeline as pipeline_module
from unlimited_ocr_max.decoder import UnlimitedOcrDecoder
from unlimited_ocr_max.graphs import DecodeGraph, build_batched_decode_graph
from unlimited_ocr_max.kv_cache import DeviceKvCache, KvCache, allocate_kv_cache, to_host
from unlimited_ocr_max.model_config import UnlimitedOCRConfig
from unlimited_ocr_max.pipeline import UnlimitedOcrPipeline

from test_decoder_int8 import _config, _named, _op_counts, _small_decoder_config
from test_weight_sharing import _EXTERNAL_DEVICE, _TRANSFERS, _graph_text, _pipeline, gpu_only

DEVICES = {"cpu": DeviceRef.CPU(), "gpu": DeviceRef.GPU(0)}


def _batched(config: UnlimitedOCRConfig, device: DeviceRef, *, batch: int, resident: bool = False) -> DecodeGraph:
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=device))
    return build_batched_decode_graph(
        config, decoder, batch=batch, max_seq_len=64, device=device, device_resident_weights=resident
    )


def _output_names(num_layers: int) -> tuple[str, ...]:
    return ("logits", *(f"key_{i}" for i in range(num_layers)), *(f"value_{i}" for i in range(num_layers)))


# --------------------------------------------------------------------------
# the batch-1 graph did not move
# --------------------------------------------------------------------------

#: sha256 of ``str(build_decode_graph(...).graph)`` for ``_config(int8=False,
#: num_hidden_layers=3)`` at ``max_seq_len=64``, staged through ``_graph_text``.
#: Recorded on the untouched tree (``release/0.3.3`` @ b783c95) before KON-212
#: added the batched graph.
_BATCH1_DECODE_GOLDEN = {
    "cpu": "367f661016cd01e531f4e492a42a9a7439c473b2449a9571bd2a985d2f294454",
    "gpu-resident": "41595edb2b89970ed17605899787f9d4e83f65e78d498d8a0c72fc246c5ae072",
    "gpu-plain": "0cad396f903ecb623af4ebea4adada373c6e9b094224caf087a25b4ac27c0f52",
}


@pytest.mark.parametrize(
    ("variant", "device", "resident"),
    [("cpu", "cpu", False), ("gpu-resident", "gpu", True), ("gpu-plain", "gpu", False)],
)
@pytest.mark.skipif(
    (sys.platform, platform.machine()) != ("darwin", "arm64"),
    reason="goldens recorded on darwin/arm64: the text prints numpy float32 RoPE tables, whose cos/sin digits are platform-dependent",
)
def test_the_batch1_decode_graph_is_byte_identical(variant: str, device: str, resident: bool) -> None:
    """The batch-1 ``build_decode_graph`` text hashes to its pre-KON-212 value, in all three variants the port stages.

    The goldens are scoped to the exact ``max`` pin (``max[all]==26.6.0``):
    the text is MAX's own MLIR printing, so a ``max`` bump may change it
    without the graph meaning anything different. Re-record them on a bump
    only after proving the bump itself did not move the graph (KON-122's
    bar), never by pasting what the new build prints. The text also carries
    the RoPE tables as printed float32 constants, and numpy's float32
    ``cos``/``sin`` are not correctly rounded on every platform (628 of 8192
    cos entries differ from the correctly rounded value on darwin/arm64), so
    the goldens are only asserted on the platform they were recorded on; the
    macOS CI leg and every local run on Apple silicon carry the guard.
    """
    config = _config(int8=False, num_hidden_layers=3)
    text = _graph_text(config, DEVICES[device], decode=True, resident=resident)
    assert hashlib.sha256(text.encode()).hexdigest() == _BATCH1_DECODE_GOLDEN[variant]


# --------------------------------------------------------------------------
# the batched graph: signature, refusal, weight placement, int8 dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_batched_decode_graph_stages_its_signature(device: str, batch: int) -> None:
    """``2 + B * (1 + 2L)`` inputs in ``selector(), *views()`` order per row, and ``[B, ...]`` outputs."""
    config = _config(int8=False, num_hidden_layers=3)
    dec = config.decoder
    layers = dec.num_hidden_layers
    staged = _batched(config, DEVICES[device], batch=batch)
    graph = staged.graph

    assert graph.name == f"unlimited_ocr_decode_64_ring_b{batch}"
    assert (staged.batch, staged.num_layers) == (batch, layers)
    assert staged.output_names == _output_names(layers)

    types = [value.type for value in graph.inputs]
    assert len(types) == 2 + batch * (1 + 2 * layers)
    assert (types[0].dtype, list(types[0].shape)) == (DType.int64, [batch])
    assert (types[1].dtype, list(types[1].shape)) == (DType.int32, [batch])
    for b in range(batch):
        start = 2 + b * (1 + 2 * layers)
        past_len = f"past_len_{b}"
        assert (types[start].dtype, [str(d) for d in types[start].shape]) == (DType.bool, ["1", past_len, "1"])
        for cache in types[start + 1 : start + 1 + 2 * layers]:
            assert cache.dtype == DType.float32
            assert [str(d) for d in cache.shape] == [past_len, str(dec.num_key_value_heads), str(dec.head_dim)]

    outputs = graph.output_types
    assert len(outputs) == 1 + 2 * layers
    assert [int(d) for d in outputs[0].shape] == [batch, dec.vocab_size]
    for kv in outputs[1:]:
        assert [int(d) for d in kv.shape] == [batch, dec.num_key_value_heads, dec.head_dim]
    assert {out.dtype for out in outputs} == {DType.float32}


@pytest.mark.parametrize("batch", [1, 0, -1])
def test_batched_decode_graph_refuses_batch_below_two(batch: int) -> None:
    config = _config(int8=False)
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=DeviceRef.CPU()))
    with pytest.raises(ValueError, match="batch 1 is build_decode_graph"):
        build_batched_decode_graph(config, decoder, batch=batch, max_seq_len=64, device=DeviceRef.CPU())


def test_batched_decode_graph_declares_resident_weights_like_build_decode_graph() -> None:
    """``device_resident_weights`` has ``build_decode_graph``'s contract: placement moves, nothing else does."""
    config = _config(int8=False, num_hidden_layers=3)
    dref = DeviceRef.GPU(0)
    n_weights = len(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=dref).raw_state_dict())
    plain = str(_batched(config, dref, batch=2).graph)
    resident = str(_batched(config, dref, batch=2, resident=True).graph)

    assert len(_EXTERNAL_DEVICE.findall(plain)) == len(_EXTERNAL_DEVICE.findall(resident)) == n_weights
    assert set(_EXTERNAL_DEVICE.findall(plain)) == {"cpu"}
    assert set(_EXTERNAL_DEVICE.findall(resident)) == {"gpu"}
    assert len(_TRANSFERS.findall(plain)) - len(_TRANSFERS.findall(resident)) == n_weights
    assert _op_counts(plain) == _op_counts(resident)


def test_int8_batched_decode_takes_the_prefill_chain_and_refuses_cpu() -> None:
    """int8 at ``B >= 2`` is the MoE's multi-row path: ``int8_dequant_expert`` per expert, not ``moe_int8_qmv``.

    The MoE is called unchanged on the ``[B, hidden]`` rows, so its dispatch
    is keyed on the row count; this pins what that means for int8. And the
    builder refuses int8 on CPU like every other language builder.
    """
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace
    text = str(_batched(config, DeviceRef.GPU(0), batch=2).graph)
    assert _op_counts(text)["mo.custom"] == 3 * dec.n_routed_experts * n_moe
    assert "int8_dequant_expert" in text and "moe_int8_qmv" not in text
    with pytest.raises(ValueError, match="GPU-only"):
        _batched(config, DeviceRef.CPU(), batch=2)


# --------------------------------------------------------------------------
# the pipeline: lazy per-batch graphs, and release_decode drops them
# --------------------------------------------------------------------------


def test_batched_decode_graph_is_built_lazily_once_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built like ``decode_graph`` (fresh decoder, its registry, the shared-weights flag), cached per ``batch``."""
    pipeline = _pipeline(DeviceRef.GPU(0))
    builds: list[dict[str, Any]] = []
    loads: list[tuple[Any, Any]] = []

    def build(config, decoder, **kwargs):
        assert config is pipeline.config
        builds.append(kwargs)
        return DecodeGraph(graph=f"graph-b{kwargs['batch']}", output_names=(), num_layers=0, batch=kwargs["batch"])

    class _Decoder:
        def state_dict(self) -> dict[str, str]:
            return {"w": "registry-sentinel"}

    class _Session:
        def load(self, graph, *, weights_registry):
            loads.append((graph, weights_registry))
            return f"model-{graph}"

    monkeypatch.setattr(pipeline_module, "build_batched_decode_graph", build)
    pipeline._decoder = _Decoder
    pipeline.session = _Session()

    staged, model = pipeline.batched_decode_graph(2)
    assert (staged.graph, staged.batch, model) == ("graph-b2", 2, "model-graph-b2")
    assert pipeline.batched_decode_graph(2)[0] is staged  # cached: no second build
    assert builds == [
        {
            "batch": 2,
            "max_seq_len": pipeline.max_total_len,
            "device": pipeline.device,
            "device_resident_weights": pipeline.shares_language_weights,
        }
    ]
    assert loads == [("graph-b2", {"w": "registry-sentinel"})]

    pipeline.batched_decode_graph(3)
    assert [kwargs["batch"] for kwargs in builds] == [2, 3]
    pipeline.release_decode()
    pipeline.batched_decode_graph(2)
    assert [kwargs["batch"] for kwargs in builds] == [2, 3, 2]


def test_release_decode_drops_the_batched_graphs_and_is_idempotent() -> None:
    pipeline = _pipeline(DeviceRef.GPU(0))
    pipeline.release_decode()  # nothing built: a no-op, not an error
    pipeline._decode = "decode-sentinel"
    pipeline._batched_decode.update({2: "b2-sentinel", 3: "b3-sentinel"})
    pipeline._language_device_weights = {"w": "registry-sentinel"}
    pipeline.release_decode()
    pipeline.release_decode()
    assert pipeline._decode is None
    assert pipeline._batched_decode == {}
    # Graphs only: the shared registry outlives every release (see release_decode).
    assert pipeline._language_device_weights == {"w": "registry-sentinel"}


# --------------------------------------------------------------------------
# decode_rows, model-free
# --------------------------------------------------------------------------

LAYERS, HEADS, HEAD_DIM, VOCAB = 2, 2, 3, 8


def _host_cache(prefill_len: int, *, seed: int) -> KvCache:
    rng = np.random.default_rng(seed)
    cache = allocate_kv_cache(
        num_layers=LAYERS, prefill_len=prefill_len, window=4, num_kv_heads=HEADS, head_dim=HEAD_DIM, device=None
    )
    shape = (prefill_len, HEADS, HEAD_DIM)
    cache.seed(
        [rng.standard_normal(shape).astype(np.float32) for _ in range(LAYERS)],
        [rng.standard_normal(shape).astype(np.float32) for _ in range(LAYERS)],
    )
    return cache


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("must not be reached")


class _BatchedExecute:
    """A stub ``_execute`` for the batched graph: row ``b`` of every KV output is filled with a value naming it."""

    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.calls: list[tuple[Any, tuple[str, ...], tuple[Any, ...], Any]] = []
        self.outputs: dict[str, np.ndarray] = {
            "logits": np.arange(batch * VOCAB, dtype=np.float32).reshape(batch, VOCAB)
        }
        for i in range(LAYERS):
            for kind, sign in (("key", 1.0), ("value", -1.0)):
                rows = [np.full((HEADS, HEAD_DIM), sign * (100.0 * (i + 1) + b), dtype=np.float32) for b in range(batch)]
                self.outputs[f"{kind}_{i}"] = np.stack(rows)

    def __call__(self, model: Any, names: Any, *arrays: Any, host_outputs: Any = None) -> dict[str, np.ndarray]:
        self.calls.append((model, tuple(names), arrays, host_outputs))
        return self.outputs


def test_decode_rows_with_one_row_is_the_batch1_step() -> None:
    pipeline = _pipeline(DeviceRef.CPU())
    calls: list[tuple[KvCache, int]] = []

    def decode_step(cache: KvCache, token_id: int) -> np.ndarray:
        calls.append((cache, token_id))
        return np.arange(VOCAB, dtype=np.float32)

    pipeline.decode_step = decode_step
    pipeline.batched_decode_graph = _refuse
    pipeline._execute = _refuse
    cache = _host_cache(3, seed=0)
    logits = pipeline.decode_rows([cache], [5])
    assert calls == [(cache, 5)]
    assert logits.shape == (1, VOCAB)
    assert np.array_equal(logits[0], np.arange(VOCAB, dtype=np.float32))


@pytest.mark.parametrize("batch", [2, 3])
def test_decode_rows_writes_row_b_into_cache_b(batch: int) -> None:
    """One execute, inputs in the graph's order, and each cache appends exactly its own row of every output."""
    pipeline = _pipeline(DeviceRef.CPU())
    staged = DecodeGraph(graph=None, output_names=_output_names(LAYERS), num_layers=LAYERS, batch=batch)
    pipeline._batched_decode[batch] = (staged, "model-sentinel")
    pipeline.decode_step = _refuse
    execute = _BatchedExecute(batch)
    pipeline._execute = execute

    caches = [_host_cache(3 + 2 * b, seed=b) for b in range(batch)]  # a different length and position per row
    tokens = [11 + b for b in range(batch)]
    before = [
        {
            "row": cache.write_index,
            "attend": cache.attend_len,
            "length": cache.length,
            "position": cache.position,
            "keys": [key.copy() for key in cache.keys],
            "values": [value.copy() for value in cache.values],
        }
        for cache in caches
    ]

    logits = pipeline.decode_rows(caches, tokens)
    assert logits is execute.outputs["logits"]

    assert len(execute.calls) == 1
    model, names, arrays, host_outputs = execute.calls[0]
    assert (model, names, host_outputs) == ("model-sentinel", staged.output_names, KvCache.HOST_OUTPUTS)
    assert len(arrays) == 2 + batch * (1 + 2 * LAYERS)
    assert arrays[0].dtype == np.int64 and arrays[0].tolist() == tokens
    assert arrays[1].dtype == np.int32 and arrays[1].tolist() == [state["position"] for state in before]
    for b, (cache, state) in enumerate(zip(caches, before, strict=True)):
        start = 2 + b * (1 + 2 * LAYERS)
        selector = arrays[start]
        assert selector.shape == (1, state["attend"], 1)
        assert np.flatnonzero(selector).tolist() == [state["row"]]
        views = arrays[start + 1 : start + 1 + 2 * LAYERS]
        for view, rows in zip(views, [*cache.keys, *cache.values], strict=True):
            assert view.shape[0] == state["attend"] and np.shares_memory(view, rows)

    for b, (cache, state) in enumerate(zip(caches, before, strict=True)):
        assert (cache.length, cache.position) == (state["length"] + 1, state["position"] + 1)
        for kind, rows, old in (("key", cache.keys, state["keys"]), ("value", cache.values, state["values"])):
            for i in range(LAYERS):
                assert np.array_equal(rows[i][state["row"]], execute.outputs[f"{kind}_{i}"][b]), (b, kind, i)
                untouched = np.delete(np.arange(rows[i].shape[0]), state["row"])
                assert np.array_equal(rows[i][untouched], old[i][untouched]), (b, kind, i)


def test_decode_rows_refuses_mismatched_rows_and_mixed_caches() -> None:
    pipeline = _pipeline(DeviceRef.CPU())
    pipeline.decode_step = _refuse
    pipeline.batched_decode_graph = _refuse
    pipeline._execute = _refuse
    host = [_host_cache(3, seed=0), _host_cache(4, seed=1)]
    with pytest.raises(ValueError, match="2 caches for 1 tokens"):
        pipeline.decode_rows(host, [1])
    with pytest.raises(ValueError, match="1 caches for 2 tokens"):
        pipeline.decode_rows(host[:1], [1, 2])
    with pytest.raises(ValueError, match="at least one row"):
        pipeline.decode_rows([], [])
    # One execute brings back one set of host outputs, so the classes cannot mix.
    device = DeviceKvCache(keys=[], values=[], window=4)
    with pytest.raises(TypeError, match="one class"):
        pipeline.decode_rows([host[0], device], [1, 2])
    assert [cache.position for cache in host] == [3, 4]


# --------------------------------------------------------------------------
# on the accelerator: B = 2 through the pipeline vs two batch-1 steps
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _accelerator_session():
    from max.driver import Accelerator
    from max.engine import InferenceSession

    driver = Accelerator()
    return driver, InferenceSession(devices=[driver])


def _random_language_weights(config: Any, seed: int) -> dict[str, Buffer]:
    """Host bf16 ``Buffer``s for every declared language weight: gammas near 1, everything else small noise."""
    rng = np.random.default_rng(seed)
    declared = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict()
    weights: dict[str, Buffer] = {}
    for name, weight in declared.items():
        shape = tuple(int(d) for d in weight.shape)
        if len(shape) == 1:
            values = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name == "embed_tokens.weight":
            values = rng.standard_normal(shape)
        else:
            values = 0.05 * rng.standard_normal(shape)
        weights[name] = Buffer.from_dlpack(torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16))
    return weights


def _device_cache(driver, config: Any, rows, *, prefill_len: int, length: int, ring_pos: int, position: int):
    """A device cache holding ``rows`` with its ring state set outright (``rows`` are ``[capacity, heads, head_dim]``)."""
    keys, values = rows
    cache = DeviceKvCache(
        keys=[],
        values=[],
        window=config.decoder.sliding_window_size,
        length=length,
        prefill_len=prefill_len,
        ring_pos=ring_pos,
        position=position,
        device_keys=[Buffer.from_numpy(np.ascontiguousarray(k)).to(driver) for k in keys],
        device_values=[Buffer.from_numpy(np.ascontiguousarray(v)).to(driver) for v in values],
        capacity=int(keys[0].shape[0]),
    )
    driver.synchronize()  # the numpy sources must outlive the async copies (KON-125)
    return cache


@pytest.mark.slow
@gpu_only
def test_batched_decode_rows_against_two_batch1_steps(_accelerator_session) -> None:
    """B = 2 through ``decode_rows`` vs the two rows through ``decode_step``; bitwise-ness is printed, not asserted.

    The pipeline is the shipped accelerator configuration -- the shared device
    registry, fp32-resident non-expert weights, device KV caches -- on
    ``_small_decoder_config`` with random weights. Row 0 is warming up (an
    append at ``length``), row 1 has a full ring and overwrites slot 5, so the
    rows differ in cache length, position and write mode.

    What is asserted besides shape and finiteness is the copy, not the
    numerics: each batched cache must hold exactly its own row of the graph's
    device outputs at its write index, and nothing else may move.
    """
    driver, session = _accelerator_session
    small = _small_decoder_config(int8=False)
    # `UnlimitedOCRConfig` pins hidden 1280 through the projector; the language
    # path reads nothing of the config but these two.
    config = SimpleNamespace(decoder=small, dtype=DType.bfloat16)
    pipeline = UnlimitedOcrPipeline(
        config,
        vision_state_dict={},
        language_state_dict=_random_language_weights(config, seed=41),
        seq_len=32,
        max_new_tokens=32,
        device=DeviceRef.GPU(0),
        driver_device=driver,
        session=session,
    )
    assert pipeline.shares_language_weights

    window, layers = small.sliding_window_size, small.num_hidden_layers
    rng = np.random.default_rng(43)
    shapes = [(5 + window, small.num_key_value_heads, small.head_dim), (9 + window, small.num_key_value_heads, small.head_dim)]
    host_rows = [
        ([rng.standard_normal(s).astype(np.float32) for _ in range(layers)],
         [rng.standard_normal(s).astype(np.float32) for _ in range(layers)])
        for s in shapes
    ]
    states = [
        {"prefill_len": 5, "length": 7, "ring_pos": 0, "position": 7},  # warm-up: appends at row 7
        {"prefill_len": 9, "length": 9 + window, "ring_pos": 5, "position": 40},  # full ring: overwrites row 14
    ]
    tokens = [3, 17]

    def caches() -> list[DeviceKvCache]:
        return [_device_cache(driver, config, rows, **state) for rows, state in zip(host_rows, states, strict=True)]

    batched, single = caches(), caches()
    write_rows = [cache.write_index for cache in batched]
    assert write_rows == [7, 14]

    executed: list[dict[str, Any]] = []
    execute = pipeline._execute

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executed.append(execute(*args, **kwargs))
        return executed[-1]

    pipeline._execute = spy
    logits = pipeline.decode_rows(batched, tokens)
    graph_out = executed[-1]
    reference = np.stack([pipeline.decode_step(cache, token) for cache, token in zip(single, tokens, strict=True)])

    assert logits.shape == reference.shape == (2, small.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits)) and np.all(np.isfinite(reference))

    got: dict[str, np.ndarray] = {"logits": logits}
    want: dict[str, np.ndarray] = {"logits": reference}
    for kind, attr in (("key", "device_keys"), ("value", "device_values")):
        for i in range(layers):
            name = f"{kind}_{i}"
            out = to_host(graph_out[name])
            rows_got, rows_want = [], []
            for b in range(2):
                after = to_host(getattr(batched[b], attr)[i])
                before = host_rows[b][0 if kind == "key" else 1][i]
                row = write_rows[b]
                # The copy: exactly row b of the graph's output, at the write index, and nothing else moved.
                assert np.array_equal(after[row], out[b]), (name, b)
                assert np.array_equal(np.delete(after, row, axis=0), np.delete(before, row, axis=0)), (name, b)
                rows_got.append(after[row])
                rows_want.append(to_host(getattr(single[b], attr)[i])[row])
            got[name], want[name] = np.stack(rows_got), np.stack(rows_want)
            assert np.all(np.isfinite(got[name]))
    assert [(c.length, c.ring_pos, c.position) for c in batched] == [(8, 0, 8), (9 + window, 6, 41)]

    bitwise = {name: bool(np.array_equal(got[name], want[name])) for name in got}
    print(f"[uocr] batched decode B=2 vs two batch-1 steps: bitwise {all(bitwise.values())}")
    for name in got:
        delta = float(np.max(np.abs(got[name].astype(np.float64) - want[name].astype(np.float64))))
        print(f"[uocr]   {name}: bitwise {bitwise[name]}, max |delta| {delta:.3e}")
        # Not the numeric bar (the rows are not bitwise to batch 1 on Metal, ~1e-6 here), but a row
        # swapped, a RoPE row misaligned or a cache mixed up moves these by O(1); fail on that.
        assert delta <= 1e-3, (name, delta)
