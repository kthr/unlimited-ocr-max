"""KON-212/KON-237: the decode graph (``B >= 1`` requests per execute) and ``pipeline.decode_rows``.

A golden hash of the batch-1 decode graph's staged text pins it: bitwise
identity is a property of the whole emitted graph (KON-122), so "the batch-1
math did not change" is the graph's to prove, not a reviewer's -- and a change
that does move it re-records the hash and says why (KON-234, KON-235, KON-237).
Since KON-237 every ``B`` is the same builder: per layer the graph stores each
row's new k/v into the pipeline's KV page pool and runs ONE paged attention op
for all rows (no per-row cache inputs, no ``where`` ring write, no KV
outputs); ``test_kv_cache`` gates that op against the old chain's math.
KON-235 put the n-gram guard into the decode graph (one ``ngram_block`` op on
the logits, each row against its own history): the staging checks pin the one
op, the model-free checks the history each row is handed, and a slow check
that ``n = 0`` returns bitwise the logits of the same graph built without the
op, and ``n = 3`` exactly the guard applied to them.
Whether a ``B >= 2`` graph is bitwise the batch-1 step is not asserted
anywhere: the slow accelerator tests print it. What is asserted bitwise is a
row against itself across ``B >= 2`` (``B`` in 2, 4, 8, int8 and bf16), the
load-independence the served padding relies on.

Staging checks run weightless on the tiny config (``test_decoder_int8``'s
``_config`` and ``_language_graph``) and reuse ``test_weight_sharing``'s
placement census. The ``decode_rows`` checks are model-free: a stub
``_execute`` and caches in the pipeline's own CPU page pool.
"""

from __future__ import annotations

import hashlib
import platform
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef

import unlimited_ocr_max.graphs as graphs_module
import unlimited_ocr_max.pipeline as pipeline_module
from unlimited_ocr_max.decoder import UnlimitedOcrDecoder
from unlimited_ocr_max.graphs import DecodeGraph, build_decode_graph
from unlimited_ocr_max.kv_cache import PAGE_SIZE, KvCache, PagedStep
from unlimited_ocr_max.pipeline import UnlimitedOcrPipeline

from test_decoder_int8 import (
    _config,
    _custom_op_calls,
    _dense_qmv_calls,
    _is_weight_only,
    _language_graph,
    _named,
    _op_counts,
    _parse_ssa_defs,
    _quantize,
    _small_decoder_config,
)
from test_kernels_ngram import BLOCKED, _banned
from test_weight_sharing import _EXTERNAL_DEVICE, _TRANSFERS, _pipeline, gpu_only

DEVICES = {"cpu": DeviceRef.CPU(), "gpu": DeviceRef.GPU(0)}


def _symbol_count(text: str, symbol: str) -> int:
    """Staged ``mo.custom`` ops naming ``symbol`` (a multi-result op's line included, which ``_custom_op_calls`` skips)."""
    return sum(f'symbol = "{symbol}"' in line for line in text.splitlines())


#: MAX's paged ops: per layer two stores (k, v) and one attention.
_PAGED_STORE, _PAGED_ATTENTION = "mo.kv_cache.store.paged.ragged", "mo.mha.ragged.paged"


# --------------------------------------------------------------------------
# the batch-1 graph is pinned
# --------------------------------------------------------------------------

#: sha256 of ``str(build_decode_graph(...).graph)`` for ``_config(int8=...,
#: num_hidden_layers=3)`` at ``max_seq_len=64``, staged through ``_language_graph``.
#: Re-recorded for the K-blocked projections (TODO ID 45): the 22 projection
#: weights are declared ``[K / KBLOCK, N, KBLOCK]`` and the 22 ``dense_bf16_qmv``
#: calls (and, plain, the 22 weight transfers) carry that type; no other line
#: moved, and the values are bitwise the old graph's (prefill and decode logits,
#: 2 pages x 600 steps, bf16, Apple M4).
#: Before: the KON-231 quality pass: the MoE spreads its one token row
#: to ``k`` expert rows as every ``B`` does (unsqueeze, broadcast, reshape)
#: instead of a plain broadcast; the values are bitwise the old graph's
#: (prefill and decode logits at B = 1 and 8, CPU and GPU, bf16 and int8).
#: Before: KON-238 (MAX ``26.7.0.dev2026100105``): every projection --
#: q/k/v/o, the dense FFN, the shared experts and ``lm_head``, 22 here -- is a
#: ``dense_bf16_qmv`` call on its bf16 weight instead of a transpose and
#: ``rmo.matmul`` on a weight MAX upcasts, so a row is summed in the kernel's
#: order (gated against float64 and the fp32-weight path in
#: ``test_dense_projections``). The bf16 router this plain declaration stages
#: is upcast at run time (``decoder._runtime_one``): two ``abs``/``min``/
#: ``add``/``cast`` chains and a ``mul``. Before: KON-237 -- MAX's paged
#: store and attention ops over the pipeline's page pool, q/k/v/RoPE on
#: ``[B, H, D]``, the batch-1 MoE as :meth:`MoE.decode_rows` -- on top of
#: KON-234 (``moe_bf16_qmv``) and KON-235 (the in-graph n-gram guard). Only
#: the accelerator variants are pinned: nothing serves on CPU. Every entry hashes
#: :func:`_golden_text`, which pins the one checkout-dependent string.
_BATCH1_DECODE_GOLDEN = {
    "gpu-resident": "47d992eade7dc8d0f84daa4ea8f8879aecf4220578ed44e7a2a961cd78a1d992",
    "gpu-plain": "e645efac83a2a84956ec7e4d7d9897e9aa014bf9c400b2fc7691ee1603f8d58b",
    "int8-gpu-resident": "4d7d6d05c619973a13c09c956b99a8477d4a5a6b472fe0d7e6f50b7b5721853d",
    "int8-gpu-plain": "c8622e2883c6c887b48f0daf8130cba4b295b6a6f9a6786f2ccd2326e5d31195",
}


#: A decode graph's header (and an int8 prefill graph's) names its compiled
#: Mojo kernel package by absolute path -- ``<tempdir>/.modular_<uid>/mojo_pkg/
#: mojo_pkg_<md5 of the absolute kernels dir>.mojoc``
#: (``mojo/paths.py::_build_mojo_source_package``) -- so it differs per
#: checkout, user and machine. Only that list entry is replaced; a bf16 prefill
#: graph's ``_kernel_library_paths = []`` has none and hashes as is.
_MOJO_PACKAGE = re.compile(r'"[^"]*/mojo_pkg_[0-9a-f]{32}\.mojoc"')


def _golden_text(text: str) -> str:
    """The graph text with the kernel package path replaced by a fixed token (see :data:`_MOJO_PACKAGE`)."""
    return _MOJO_PACKAGE.sub('"<MOJO_KERNELS>"', text)


@pytest.mark.parametrize(
    ("variant", "device", "resident"),
    [
        ("gpu-resident", "gpu", True),
        ("gpu-plain", "gpu", False),
        ("int8-gpu-resident", "gpu", True),
        ("int8-gpu-plain", "gpu", False),
    ],
)
@pytest.mark.skipif(
    (sys.platform, platform.machine()) != ("darwin", "arm64"),
    reason="goldens recorded on darwin/arm64: the text prints numpy float32 RoPE tables, whose cos/sin digits are platform-dependent",
)
def test_the_batch1_decode_graph_is_byte_identical(variant: str, device: str, resident: bool) -> None:
    """The batch-1 ``build_decode_graph`` text hashes to its recorded value, in every variant the port serves.

    The goldens are scoped to the exact ``max`` pin (all recorded on ``26.7.0.dev2026100105``):
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
    config = _config(int8=variant.startswith("int8-"), num_hidden_layers=3)
    text = str(_language_graph(config, DEVICES[device], decode=True, resident=resident).graph)
    assert hashlib.sha256(_golden_text(text).encode()).hexdigest() == _BATCH1_DECODE_GOLDEN[variant]


# --------------------------------------------------------------------------
# the decode graph: signature, the paged ops, refusal, weight placement, MoE dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_the_decode_graph_stages_its_signature(batch: int) -> None:
    """Eleven inputs at every ``B`` -- tokens, positions, the page pool, the guard's two -- and one output, the logits."""
    config = _config(int8=False, num_hidden_layers=3)
    dec = config.decoder
    dref, cpu = DEVICES["gpu"], DeviceRef.CPU()
    staged = _language_graph(config, dref, decode=True, batch=batch)
    graph = staged.graph

    assert graph.name == f"unlimited_ocr_decode_64_paged_b{batch}"
    assert staged.batch == batch and staged.output_names == ("logits",)

    def described(value_type: Any) -> tuple[Any, ...]:
        kind = "buffer" if isinstance(value_type, BufferType) else "tensor"
        return kind, value_type.dtype, [str(d) for d in value_type.shape], value_type.device

    blocks = ["total_num_pages", "2", "3", str(PAGE_SIZE), str(dec.num_key_value_heads), str(dec.head_dim)]
    want = [
        ("tensor", DType.int64, [str(batch)], dref),  # tokens
        ("tensor", DType.int32, [str(batch)], dref),  # positions
        ("buffer", DType.float32, blocks, dref),  # kv_blocks
        ("tensor", DType.uint32, [str(batch), "lut_cols"], dref),  # lookup table
        ("tensor", DType.uint32, [str(batch)], dref),  # attention cache lengths: attend_len - 1
        ("tensor", DType.uint32, ["1"], cpu),  # their max
        ("tensor", DType.uint32, [str(batch)], dref),  # store cache lengths: write_index
        ("tensor", DType.uint32, ["1"], cpu),  # their max
        ("tensor", DType.int64, ["4"], cpu),  # MHA dispatch key
        ("tensor", DType.int32, [str(batch), str(dec.sliding_window_size)], dref),  # guard history
        ("tensor", DType.int32, ["1"], dref),  # n-gram size
    ]
    assert [described(value.type) for value in graph.inputs] == want

    (logits,) = graph.output_types
    assert (logits.dtype, [int(d) for d in logits.shape]) == (DType.float32, [batch, dec.vocab_size])


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize(("device", "int8"), [("gpu", False), ("gpu", True)], ids=["gpu", "int8-gpu"])
def test_each_layer_stores_and_attends_through_the_page_pool_with_no_ring_write(device: str, int8: bool, batch: int) -> None:
    """Per layer two paged stores and ONE paged attention for all ``B`` rows, causal, on ``kv_blocks``; no ``where`` anywhere.

    The old chain's fingerprints are gone: no ``rmo.select`` (its ring
    write), no attention softmax (only the MoE gate's, one per MoE layer) and
    no per-row unrolling -- the op counts do not grow with ``B``.
    """
    config = _config(int8=int8, num_hidden_layers=3)
    dec = config.decoder
    layers, n_moe = dec.num_hidden_layers, dec.num_hidden_layers - dec.first_k_dense_replace
    text = str(_language_graph(config, DEVICES[device], decode=True, batch=batch, resident=device == "gpu").graph)

    assert _symbol_count(text, _PAGED_STORE) == 2 * layers
    assert _symbol_count(text, _PAGED_ATTENTION) == layers
    attention = [line for line in text.splitlines() if f'symbol = "{_PAGED_ATTENTION}"' in line]
    assert all('mask_str = "causal"' in line and "%arg2," in line for line in attention)  # kv_blocks is input 2
    assert "rmo.select" not in text
    assert _op_counts(text)["rmo.mo.reduce.softmax"] == n_moe
    one_row = _op_counts(str(_language_graph(config, DEVICES[device], decode=True, resident=device == "gpu").graph))
    want = 3 * n_moe + 1 + 3 * layers + _dense_qmv_calls(dec, decode=True)
    assert _op_counts(text)["mo.custom"] == one_row["mo.custom"] == want


@pytest.mark.parametrize("batch", [0, -1])
def test_the_decode_graph_refuses_batch_below_one(batch: int) -> None:
    config = _config(int8=False)
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=DeviceRef.CPU()))
    with pytest.raises(ValueError, match="batch must be >= 1"):
        build_decode_graph(config, decoder, batch=batch, max_seq_len=64, device=DeviceRef.CPU())


def test_a_multi_row_decode_graph_declares_resident_weights_like_batch_one() -> None:
    """``device_resident_weights`` at ``B = 2`` has the batch-1 contract: placement moves, nothing else does."""
    config = _config(int8=False, num_hidden_layers=3)
    dref = DeviceRef.GPU(0)
    n_weights = len(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=dref).raw_state_dict())
    plain = str(_language_graph(config, dref, decode=True, batch=2).graph)
    resident = str(_language_graph(config, dref, decode=True, batch=2, resident=True).graph)

    assert len(_EXTERNAL_DEVICE.findall(plain)) == len(_EXTERNAL_DEVICE.findall(resident)) == n_weights
    assert set(_EXTERNAL_DEVICE.findall(plain)) == {"cpu"}
    assert set(_EXTERNAL_DEVICE.findall(resident)) == {"gpu"}
    assert len(_TRANSFERS.findall(plain)) - len(_TRANSFERS.findall(resident)) == n_weights
    assert _op_counts(plain) == _op_counts(resident)


@pytest.mark.parametrize("resident", [False, True], ids=["plain", "resident"])
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_int8_decode_runs_moe_int8_qmv_over_b_times_k_rows(batch: int, resident: bool) -> None:
    """int8 decode at every ``B`` is :meth:`MoE.decode_rows`: three ``moe_int8_qmv`` calls per MoE layer over ``B * k`` rows.

    No ``int8_dequant_expert`` at all -- the prefill chain dequantizes every
    expert to serve ``B * k`` (token, expert) pairs -- and every qmv call's
    expert-ids operand traces back to a graph input, not only to
    declarations, so MAX cannot fold it at load (KON-224's criterion, the
    same SSA walk as ``test_int8_decode_graph_has_no_dequant_call_and_moe_int8_qmv_stays_runtime``).
    """
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace
    text = str(_language_graph(config, DeviceRef.GPU(0), decode=True, batch=batch, resident=resident).graph)
    assert "int8_dequant_expert" not in text

    calls = _custom_op_calls(text, "moe_int8_qmv")
    # The others: the n-gram guard, the paged ops (three per layer) and the dense projections.
    others = 1 + 3 * dec.num_hidden_layers + _dense_qmv_calls(dec, decode=True)
    assert _op_counts(text)["mo.custom"] - others == len(calls) == 3 * n_moe
    rows = batch * dec.num_experts_per_tok
    lines = [line for line in text.splitlines() if 'symbol = "moe_int8_qmv"' in line]
    assert len(lines) == len(calls) and all(f"-> !mo.tensor<[{rows}, " in line for line in lines)

    defs = _parse_ssa_defs(text)
    memo: dict[str, bool] = {}
    foldable = [result for result, operands in calls if _is_weight_only(operands[1], defs, memo)]
    assert not foldable, f"{len(foldable)}/{len(calls)} moe_int8_qmv calls have a weight-only expert-ids operand"


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize(
    ("device", "resident"), [("cpu", False), ("gpu", False), ("gpu", True)], ids=["cpu", "gpu-plain", "gpu-resident"]
)
def test_bf16_decode_runs_moe_bf16_qmv_over_b_times_k_rows(device: str, resident: bool, batch: int) -> None:
    """bf16 decode at every ``B`` is :meth:`MoE._routed_qmv`, on every device.

    Three ``moe_bf16_qmv`` calls per MoE layer over ``B * k`` rows, and no
    other custom op but the one n-gram guard (KON-235) and MAX's paged ops
    (KON-237): none of MAX's native MoE kernels (``moe_create_indices``,
    ``grouped_matmul_ragged``), none of int8's. Every call's expert-ids operand
    traces back to a graph input, not only to declarations, so MAX cannot fold
    it at load (the SSA walk of the int8 test above), and the graph names the
    Mojo kernel package it needs.
    """
    config = _config(int8=False, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace
    text = str(_language_graph(config, DEVICES[device], decode=True, batch=batch, resident=resident).graph)
    assert "mo.moe.create.indices" not in text and "mo.grouped.matmul.ragged" not in text
    assert "moe_int8_qmv" not in text and "int8_dequant_expert" not in text
    assert "_kernel_library_paths = []" not in text and _MOJO_PACKAGE.search(text)

    calls = _custom_op_calls(text, "moe_bf16_qmv")
    assert len(_custom_op_calls(text, "ngram_block")) == 1  # KON-235: the n-gram guard on the logits
    paged = _symbol_count(text, _PAGED_STORE) + _symbol_count(text, _PAGED_ATTENTION)
    assert paged == 3 * dec.num_hidden_layers
    dense = _custom_op_calls(text, "dense_bf16_qmv")  # KON-238: every projection, lm_head included
    assert len(dense) == _dense_qmv_calls(dec, decode=True)
    assert _op_counts(text)["mo.custom"] - 1 - paged - len(dense) == len(calls) == 3 * n_moe
    rows = batch * dec.num_experts_per_tok
    lines = [line for line in text.splitlines() if 'symbol = "moe_bf16_qmv"' in line]
    assert len(lines) == len(calls) and all(f"-> !mo.tensor<[{rows}, " in line for line in lines)

    defs = _parse_ssa_defs(text)
    memo: dict[str, bool] = {}
    foldable = [result for result, operands in calls if _is_weight_only(operands[1], defs, memo)]
    assert not foldable, f"{len(foldable)}/{len(calls)} moe_bf16_qmv calls have a weight-only expert-ids operand"


# --------------------------------------------------------------------------
# KON-235: the n-gram guard, one op inside each decode graph
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("device", "int8"), [("gpu", False), ("gpu", True)], ids=["gpu", "int8-gpu"])
@pytest.mark.parametrize("batch", [1, 8])
def test_each_decode_graph_guards_its_logits_with_exactly_one_ngram_block(batch: int, device: str, int8: bool) -> None:
    """ONE ``ngram_block`` op for all ``B`` rows, reading the logits and the graph's last two inputs, feeding the logits output."""
    config = _config(int8=int8, num_hidden_layers=3)
    dec = config.decoder
    staged = _language_graph(config, DEVICES[device], decode=True, batch=batch, resident=device == "gpu")
    graph, text = staged.graph, str(staged.graph)

    calls = _custom_op_calls(text, "ngram_block")
    assert len(calls) == 1, f"{len(calls)} ngram_block ops"
    result, operands = calls[0]
    n_inputs = len(graph.inputs)
    assert operands[1:] == [f"%arg{n_inputs - 2}", f"%arg{n_inputs - 1}"]
    history, ngram = (value.type for value in graph.inputs[-2:])
    assert (history.dtype, [int(d) for d in history.shape]) == (DType.int32, [batch, dec.sliding_window_size])
    assert (ngram.dtype, [int(d) for d in ngram.shape]) == (DType.int32, [1])
    # Its logits operand is computed in the graph, and its result is the logits output.
    assert operands[0] in _parse_ssa_defs(text)
    (output,) = [line.split() for line in text.splitlines() if line.strip().startswith("mo.output")]
    assert output[1].rstrip(",") == result
    assert [int(d) for d in graph.output_types[0].shape] == [batch, dec.vocab_size]


@pytest.mark.parametrize(("device", "int8"), [("gpu", False), ("gpu", True)], ids=["gpu", "int8-gpu"])
def test_the_prefill_graph_stages_no_guard(device: str, int8: bool) -> None:
    """The prefill logits keep their own one-op graph (``NgramBlocker``); the prefill graph has no ``ngram_block``."""
    assert "ngram_block" not in str(_language_graph(_config(int8=int8, num_hidden_layers=3), DEVICES[device], decode=False).graph)


# --------------------------------------------------------------------------
# the pipeline: lazy per-batch graphs, and release_decode drops them
# --------------------------------------------------------------------------


def test_the_decode_graph_is_built_lazily_once_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built for each ``B`` alike (fresh decoder, its registry, the shared-weights flag), cached per ``batch``."""
    pipeline = _pipeline(DeviceRef.GPU(0))
    builds: list[dict[str, Any]] = []
    loads: list[tuple[Any, Any]] = []

    def build(config, decoder, **kwargs):
        assert config is pipeline.config
        builds.append(kwargs)
        return DecodeGraph(graph=f"graph-b{kwargs['batch']}", output_names=(), batch=kwargs["batch"])

    class _Decoder:
        def state_dict(self) -> dict[str, str]:
            return {"w": "registry-sentinel"}

    class _Session:
        def load(self, graph, *, weights_registry):
            loads.append((graph, weights_registry))
            return f"model-{graph}"

    monkeypatch.setattr(pipeline_module, "build_decode_graph", build)
    pipeline._decoder = _Decoder
    pipeline.session = _Session()

    staged, model = pipeline.decode_graph(2)
    assert (staged.graph, staged.batch, model) == ("graph-b2", 2, "model-graph-b2")
    assert pipeline.decode_graph(2)[0] is staged  # cached: no second build
    assert builds == [
        {
            "batch": 2,
            "max_seq_len": pipeline.max_total_len,
            "device": pipeline.device,
            "device_resident_weights": pipeline.shares_language_weights,
        }
    ]
    assert loads == [("graph-b2", {"w": "registry-sentinel"})]

    pipeline.decode_graph(3)
    pipeline.decode_graph()
    assert [kwargs["batch"] for kwargs in builds] == [2, 3, 1]
    pipeline.release_decode()
    pipeline.decode_graph(2)
    assert [kwargs["batch"] for kwargs in builds] == [2, 3, 1, 2]


def test_release_decode_drops_every_decode_graph_and_is_idempotent() -> None:
    pipeline = _pipeline(DeviceRef.GPU(0))
    pipeline.release_decode()  # nothing built: a no-op, not an error
    pipeline._decode.update({1: "b1-sentinel", 2: "b2-sentinel", 3: "b3-sentinel"})
    pipeline._language_device_weights = {"w": "registry-sentinel"}
    pipeline._kv_pool = "pool-sentinel"
    pipeline.release_decode()
    pipeline.release_decode()
    assert pipeline._decode == {}
    # Graphs only: the shared registry outlives every release (see release_decode), and so does
    # the page pool the live requests' rows are in.
    assert pipeline._language_device_weights == {"w": "registry-sentinel"}
    assert pipeline._kv_pool == "pool-sentinel"


# --------------------------------------------------------------------------
# decode_rows, model-free
# --------------------------------------------------------------------------

LAYERS, HEADS, HEAD_DIM, VOCAB = 2, 2, 3, 8


def _cpu_pipeline(window: int = 16) -> UnlimitedOcrPipeline:
    """A CPU pipeline at the stub dimensions and a ``window``-slot ring whose page pool holds four small requests; nothing is ever compiled."""
    decoder = SimpleNamespace(
        num_hidden_layers=LAYERS, num_key_value_heads=HEADS, head_dim=HEAD_DIM, vocab_size=VOCAB, sliding_window_size=window
    )
    return UnlimitedOcrPipeline(
        SimpleNamespace(decoder=decoder, dtype=DType.bfloat16),
        vision_state_dict={},
        language_state_dict={},
        seq_len=8,
        max_new_tokens=8,
        device=DeviceRef.CPU(),
        session=SimpleNamespace(),
        max_batch_size=4,
    )


def _host_cache(pipeline: UnlimitedOcrPipeline, prefill_len: int, *, seed: int) -> KvCache:
    """A cache in ``pipeline``'s page pool, seeded with ``prefill_len`` random rows per layer."""
    rng = np.random.default_rng(seed)
    pool = pipeline.kv_pool
    cache = pool.allocate(prefill_len=prefill_len, window=pipeline.window)
    shape = (prefill_len, pool.num_kv_heads, pool.head_dim)
    cache.seed(
        [rng.standard_normal(shape).astype(np.float32) for _ in range(pool.num_layers)],
        [rng.standard_normal(shape).astype(np.float32) for _ in range(pool.num_layers)],
    )
    return cache


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("must not be reached")


#: A decode row's history where the n-gram guard is off: never read (the graph gets ``n = 0``
#: and an all-zero history), so any sequence does.
_UNREAD: tuple[int, ...] = ()


class _Execute:
    """A stub ``_execute`` for a ``batch``-row decode graph: records each call, returns row-numbered logits.

    ``_step`` hands ``_execute`` buffers only. A staged input is a view of the
    pipeline's input stager, which the next step overwrites, so each call
    records every input but ``kv_blocks`` as a numpy copy of its values at
    the call (``kv_blocks`` itself, to be checked by identity).
    """

    def __init__(self, batch: int, kv_blocks: Buffer) -> None:
        self.batch = batch
        self.kv_blocks = kv_blocks
        self.calls: list[tuple[Any, tuple[str, ...], tuple[Any, ...]]] = []
        self.outputs: dict[str, np.ndarray] = {
            "logits": np.arange(batch * VOCAB, dtype=np.float32).reshape(batch, VOCAB)
        }

    def __call__(self, model: Any, names: Any, *arrays: Any) -> dict[str, np.ndarray]:
        assert all(isinstance(value, Buffer) for value in arrays)
        recorded = tuple(value if value is self.kv_blocks else np.array(value.to_numpy(), copy=True) for value in arrays)
        self.calls.append((model, tuple(names), recorded))
        return self.outputs


def _stub_graph(pipeline: UnlimitedOcrPipeline, batch: int) -> _Execute:
    """Stand ``batch``'s decode graph and ``_execute`` in on ``pipeline``; the stub execute is returned."""
    pipeline._decode[batch] = (DecodeGraph(graph=None, output_names=("logits",), batch=batch), f"model-b{batch}")
    execute = _Execute(batch, pipeline.kv_pool.kv_blocks)
    pipeline._execute = execute
    return execute


def _assert_pool_inputs(arrays: tuple[Any, ...], pool: Any, want: PagedStep) -> None:
    """An execute's page-pool inputs -- after tokens and positions, before the guard's two -- are ``want``'s
    :meth:`PagedStep.graph_inputs`: the pool's own ``kv_blocks``, then the step's arrays, the host-resident ones as host buffers."""
    got, expected = arrays[2:-2], want.graph_inputs(pool.kv_blocks)
    assert len(got) == len(expected)
    for value, reference in zip(got, expected, strict=True):
        if reference is pool.kv_blocks:
            assert value is reference
        elif isinstance(reference, Buffer):
            assert value.tolist() == reference.to_numpy().tolist()
        else:
            assert value.dtype == reference.dtype and np.array_equal(value, reference)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_decode_rows_feeds_one_execute_the_page_metadata_and_moves_only_host_state(batch: int) -> None:
    """One execute of the ``B``-row graph with the inputs in its order; the caches' rows are the graph's to write.

    The page-pool inputs are :meth:`PagedStep.graph_inputs` of
    :meth:`KvPagePool.step` for the caches as they were; afterwards each cache
    has only advanced its ring state, and ``decode_rows`` itself wrote no row.
    """
    pipeline = _cpu_pipeline()
    execute = _stub_graph(pipeline, batch)
    caches = [_host_cache(pipeline, 3 + 2 * b, seed=b) for b in range(batch)]  # a different length and position per row
    tokens = [11 + b for b in range(batch)]
    before = [(cache.write_index, cache.length, cache.position) for cache in caches]
    want = pipeline.kv_pool.step(caches)
    rows_before = [cache.host_rows() for cache in caches]

    logits = pipeline.decode_rows(caches, tokens, [_UNREAD] * batch)
    assert np.array_equal(logits, execute.outputs["logits"])

    assert len(execute.calls) == 1
    model, names, arrays = execute.calls[0]
    assert (model, names, len(arrays)) == (f"model-b{batch}", ("logits",), 11)
    assert arrays[0].dtype == np.int64 and arrays[0].tolist() == tokens
    assert arrays[1].dtype == np.int32 and arrays[1].tolist() == [position for _, _, position in before]
    _assert_pool_inputs(arrays, pipeline.kv_pool, want)
    assert want.attend_lengths.tolist() == [length for _, length, _ in before]
    assert want.write_index.tolist() == [row for row, _, _ in before]
    assert want.dispatch.tolist() == [batch, 1, 1, max(length for _, length, _ in before) + 1]
    # The guard is off on this pipeline: n = 0 and an all-zero history of the graph's shape.
    history, ngram = arrays[-2:]
    assert history.dtype == np.int32 and history.shape == (batch, pipeline.window) and not history.any()
    assert ngram.dtype == np.int32 and ngram.tolist() == [0]

    for cache, (row, length, position), rows in zip(caches, before, rows_before, strict=True):
        assert (cache.length, cache.position) == (length + 1, position + 1) and row == length
        for got, want_rows in zip([*cache.host_rows()[0], *cache.host_rows()[1]], [*rows[0], *rows[1]], strict=True):
            assert np.array_equal(got, want_rows)


def test_decode_step_is_the_batch1_graph_and_decode_rows_of_one_row_too() -> None:
    """Both run the batch-1 graph -- ``decode_step`` returns its ``[vocab]`` row, ``decode_rows`` the ``[1, vocab]`` matrix."""
    pipeline = _cpu_pipeline()
    execute = _stub_graph(pipeline, 1)
    cache = _host_cache(pipeline, 3, seed=0)
    one = pipeline.decode_step(cache, 5, [1, 2, 5])
    rows = pipeline.decode_rows([cache], [6], [[1, 2, 5, 6]])
    assert one.shape == (VOCAB,) and rows.shape == (1, VOCAB)
    assert [call[0] for call in execute.calls] == ["model-b1", "model-b1"]
    assert [call[2][0].tolist() for call in execute.calls] == [[5], [6]]
    assert cache.position == 5


def test_decode_rows_refuses_mismatched_rows() -> None:
    pipeline = _cpu_pipeline()
    pipeline._execute = _refuse
    caches = [_host_cache(pipeline, 3, seed=0), _host_cache(pipeline, 4, seed=1)]
    with pytest.raises(ValueError, match="2 caches for 1 tokens"):
        pipeline.decode_rows(caches, [1], [_UNREAD])
    with pytest.raises(ValueError, match="1 caches for 2 tokens"):
        pipeline.decode_rows(caches[:1], [1, 2], [_UNREAD] * 2)
    with pytest.raises(ValueError, match="1 histories for 2 tokens"):
        pipeline.decode_rows(caches, [1, 2], [_UNREAD])
    with pytest.raises(ValueError, match="at least one row"):
        pipeline.decode_rows([], [], [])
    assert [cache.position for cache in caches] == [3, 4]


def _guarded(pipeline: UnlimitedOcrPipeline, ngram_size: int = 3) -> UnlimitedOcrPipeline:
    """Turn the guard on after construction, which would otherwise compile ``NgramBlocker``'s graph."""
    pipeline.ngram_size = ngram_size
    return pipeline


def _sequence(length: int, token: int, *, seed: int) -> list[int]:
    """A request's sequence so far: ``length`` ids, ``token`` -- the one being fed -- last."""
    rng = np.random.default_rng(seed)
    return [*(int(i) for i in rng.integers(0, 1000, size=length - 1)), token]


@pytest.mark.parametrize("batch", [2, 3])
def test_decode_rows_hands_row_b_its_own_history_and_the_ngram_size(batch: int) -> None:
    """Guard on: row ``b``'s history is the last ``window`` ids of ITS sequence, int32, in row ``b``; ``n`` is the pipeline's."""
    pipeline = _guarded(_cpu_pipeline())
    window = pipeline.window
    execute = _stub_graph(pipeline, batch)
    tokens = [11 + b for b in range(batch)]
    sequences = [_sequence(window + 7 * b, tokens[b], seed=b) for b in range(batch)]  # the first is exactly one window

    pipeline.decode_rows([_host_cache(pipeline, 3 + b, seed=b) for b in range(batch)], tokens, sequences)

    history, ngram = execute.calls[0][2][-2:]
    assert history.dtype == np.int32 and history.shape == (batch, window)
    for b in range(batch):
        assert history[b].tolist() == sequences[b][-window:], b
        assert history[b, -1] == tokens[b]
    assert ngram.dtype == np.int32 and ngram.tolist() == [3]


def test_the_guard_refuses_a_history_that_cannot_be_the_requests_sequence() -> None:
    """Guard on: shorter than the window -- a decode step follows >= 273 prompt tokens."""
    pipeline = _guarded(_cpu_pipeline())
    _stub_graph(pipeline, 1)
    pipeline._execute = _refuse
    cache = _host_cache(pipeline, 3, seed=0)
    window = pipeline.window
    with pytest.raises(ValueError, match=f"has {window - 1} tokens, fewer than the {window}-token n-gram window"):
        pipeline.decode_step(cache, 7, _sequence(window - 1, 7, seed=0))
    assert cache.position == 3


class _RecordingBlocker:
    """The prefill guard's ``apply``: records the sequence it was handed and passes the logits through."""

    def __init__(self) -> None:
        self.seen: list[list[int]] = []

    def apply(self, logits: np.ndarray, sequence: Any) -> np.ndarray:
        self.seen.append(list(sequence))
        return logits


class _ReleaseRecorder:
    """A prefill's cache, stood in for: counts its releases."""

    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


def test_generate_guards_the_prefill_logits_once_and_hands_every_decode_step_its_sequence() -> None:
    """``generate``: ``blocker.apply`` on the prefill logits only; each decode step gets the sequence with its token
    appended; the request's pages go back to the pool at the end, once."""
    pipeline = _guarded(_pipeline(DeviceRef.CPU()))
    pipeline.max_new_tokens = 3
    blocker = _RecordingBlocker()
    pipeline._ngram = blocker
    prompt = list(range(100, 100 + pipeline.window + 4))
    steps: list[tuple[int, list[int]]] = []
    cache = _ReleaseRecorder()

    def decode_step(cache_: Any, token_id: int, history: Any) -> np.ndarray:
        assert cache_ is cache and cache.released == 0
        steps.append((token_id, list(history)))
        return np.eye(VOCAB, dtype=np.float32)[len(steps) + 2]  # argmax 3, 4, ...

    pipeline.run_vision = lambda pixels, local_pixels=None: {"image_embeds": None}
    pipeline.run_prefill = lambda token_ids, image_embeds: pipeline_module.PrefillResult(
        logits=np.eye(VOCAB, dtype=np.float32)[2], cache=cache
    )
    pipeline.decode_step = decode_step

    generated = pipeline.generate(pixels=np.zeros(1), token_ids=np.asarray(prompt))

    assert generated == [2, 3, 4]
    assert blocker.seen == [prompt]
    assert steps == [(2, [*prompt, 2]), (3, [*prompt, 2, 3]), (4, [*prompt, 2, 3, 4])]
    assert cache.released == 1


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
    """Host ``Buffer``s for every declared language weight: gammas near 1, everything else small bf16 noise.

    In an int8 config each routed expert stack is that noise quantized like
    the checkpoint (``_quantize``), next to its fp32 ``_scales``; a bf16
    config draws exactly what it always did.
    """
    rng = np.random.default_rng(seed)
    declared = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict()
    weights: dict[str, Buffer] = {}
    for name, weight in declared.items():
        if weight.dtype == DType.float32:
            continue  # an int8 stack's scales, made with the stack itself below
        shape = tuple(int(d) for d in weight.shape)
        if len(shape) == 1:
            values = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name == "embed_tokens.weight":
            values = rng.standard_normal(shape)
        else:
            values = 0.05 * rng.standard_normal(shape)
        tensor = torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16)
        if weight.dtype == DType.int8:
            q, scales, _ = _quantize(tensor)
            weights[name], weights[f"{name}_scales"] = Buffer.from_dlpack(q), Buffer.from_dlpack(scales)
        else:
            weights[name] = Buffer.from_dlpack(tensor)
    return weights


def _small_pipeline(accelerator_session: Any, *, int8: bool, seed: int, max_batch_size: int = 8) -> UnlimitedOcrPipeline:
    """The shipped accelerator configuration -- the shared device registry (bf16 projections, fp32-resident norms
    and router), the page pool on the device -- on ``_small_decoder_config(int8=...)`` with random weights."""
    driver, session = accelerator_session
    small = _small_decoder_config(int8=int8)
    # `UnlimitedOCRConfig` pins hidden 1280 through the projector; the language
    # path reads nothing of the config but these two.
    config = SimpleNamespace(decoder=small, dtype=DType.bfloat16)
    pipeline = UnlimitedOcrPipeline(
        config,
        vision_state_dict={},
        language_state_dict=_random_language_weights(config, seed=seed),
        seq_len=32,
        max_new_tokens=32,
        device=DeviceRef.GPU(0),
        driver_device=driver,
        session=session,
        max_batch_size=max_batch_size,
    )
    assert pipeline.shares_language_weights
    return pipeline


def _paged_cache(pipeline: UnlimitedOcrPipeline, rows: Any, *, prefill_len: int, length: int, ring_pos: int, position: int) -> KvCache:
    """A cache in ``pipeline``'s pool holding ``rows`` (``[capacity, heads, head_dim]`` per layer), its ring state set outright."""
    keys, values = rows
    cache = pipeline.kv_pool.allocate(prefill_len=prefill_len, window=pipeline.window)
    capacity = int(keys[0].shape[0])
    assert capacity == prefill_len + pipeline.window
    cache._write_prefix(keys, values, capacity)
    cache.prefill_len, cache.length, cache.ring_pos, cache.position = prefill_len, length, ring_pos, position
    return cache


def _random_rows(rng: np.random.Generator, pipeline: UnlimitedOcrPipeline, prefill_len: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dec = pipeline.config.decoder
    shape = (prefill_len + pipeline.window, dec.num_key_value_heads, dec.head_dim)
    return (
        [rng.standard_normal(shape).astype(np.float32) for _ in range(dec.num_hidden_layers)],
        [rng.standard_normal(shape).astype(np.float32) for _ in range(dec.num_hidden_layers)],
    )


def _whole(cache: KvCache) -> list[np.ndarray]:
    """Every row of every layer, keys then values, as host copies."""
    keys, values = cache.host_rows()
    return [*keys, *values]


@pytest.mark.slow
@gpu_only
def test_batched_decode_rows_against_two_batch1_steps(_accelerator_session) -> None:
    """B = 2 through ``decode_rows`` vs the two rows through ``decode_step``; bitwise-ness is printed, not asserted.

    Row 0 is warming up (an append at ``length``), row 1 has a full ring and
    overwrites slot 5, so the rows differ in cache length, position and write
    mode. Asserted besides shape and finiteness: the graph stored exactly one
    row per cache, at its write index, and moved nothing else.
    """
    pipeline = _small_pipeline(_accelerator_session, int8=False, seed=41)
    small = pipeline.config.decoder
    window = small.sliding_window_size
    rng = np.random.default_rng(43)
    states = [
        {"prefill_len": 5, "length": 7, "ring_pos": 0, "position": 7},  # warm-up: appends at row 7
        {"prefill_len": 9, "length": 9 + window, "ring_pos": 5, "position": 40},  # full ring: overwrites row 14
    ]
    host_rows = [_random_rows(rng, pipeline, state["prefill_len"]) for state in states]
    tokens = [3, 17]

    def caches() -> list[KvCache]:
        return [_paged_cache(pipeline, rows, **state) for rows, state in zip(host_rows, states, strict=True)]

    batched = caches()
    write_rows = [cache.write_index for cache in batched]
    assert write_rows == [7, 14]
    logits = pipeline.decode_rows(batched, tokens, [_UNREAD] * len(tokens))
    after_batched = [_whole(cache) for cache in batched]
    for cache in batched:
        cache.release()
    single = caches()
    reference = np.stack([pipeline.decode_step(cache, token, _UNREAD) for cache, token in zip(single, tokens, strict=True)])
    after_single = [_whole(cache) for cache in single]

    assert logits.shape == reference.shape == (2, small.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits)) and np.all(np.isfinite(reference))
    assert [(c.length, c.ring_pos, c.position) for c in batched] == [(8, 0, 8), (9 + window, 6, 41)]

    got: dict[str, np.ndarray] = {"logits": logits}
    want: dict[str, np.ndarray] = {"logits": reference}
    names = [f"{kind}_{i}" for kind in ("key", "value") for i in range(small.num_hidden_layers)]
    for j, name in enumerate(names):
        rows_got, rows_want = [], []
        for b in range(2):
            before = (host_rows[b][0] if j < small.num_hidden_layers else host_rows[b][1])[j % small.num_hidden_layers]
            after = after_batched[b][j][: before.shape[0]]
            row = write_rows[b]
            # The store: one row moved, the one at the write index, and nothing else.
            assert np.array_equal(np.delete(after, row, axis=0), np.delete(before, row, axis=0)), (name, b)
            assert not np.array_equal(after[row], before[row]), (name, b)
            rows_got.append(after[row])
            rows_want.append(after_single[b][j][row])
        got[name], want[name] = np.stack(rows_got), np.stack(rows_want)
        assert np.all(np.isfinite(got[name]))

    bitwise = {name: bool(np.array_equal(got[name], want[name])) for name in got}
    print(f"[uocr] batched decode B=2 vs two batch-1 steps: bitwise {all(bitwise.values())}")
    for name in got:
        delta = float(np.max(np.abs(got[name].astype(np.float64) - want[name].astype(np.float64))))
        print(f"[uocr]   {name}: bitwise {bitwise[name]}, max |delta| {delta:.3e}")
        # Not the numeric bar (the rows are not bitwise to batch 1 on Metal), but a row swapped,
        # a RoPE row misaligned or a cache mixed up moves these by O(1); fail on that.
        assert delta <= 1e-3, (name, delta)


# --------------------------------------------------------------------------
# on the accelerator: rows across B and against batch 1
# --------------------------------------------------------------------------


@pytest.mark.slow
@gpu_only
def test_int8_batched_rows_are_bitwise_across_b_and_close_to_batch1(_accelerator_session) -> None:
    """int8 B-row decode (``moe_int8_qmv``): each row's bits do not depend on ``B``; see :func:`_rows_across_b`."""
    _rows_across_b(_accelerator_session, int8=True, seed=53)


@pytest.mark.slow
@gpu_only
def test_bf16_batched_rows_are_bitwise_across_b_and_close_to_batch1(_accelerator_session) -> None:
    """bf16 B-row decode (``moe_bf16_qmv``, KON-234/239; paged attention, KON-237): each row's bits do not depend on ``B``; see :func:`_rows_across_b`."""
    _rows_across_b(_accelerator_session, int8=False, seed=61)


def _rows_across_b(accelerator_session: Any, *, int8: bool, seed: int) -> None:
    """B-row decode: each row's bits do not depend on ``B``; its distance to the batch-1 step is recorded.

    Eight requests differ in prefix length, position, write mode (even rows
    warm up and append, odd rows have a full ring and overwrite) and token.
    Rows ``0 .. B-1`` step once through the ``B``-row graph for ``B`` in 2,
    4, 8, each from identical starting caches in freshly allocated pages, and
    every row's logits and whole cache afterwards must be bitwise equal across
    the three graphs: the load-independence ``--max-batch-size N > 1`` serves
    on (every step runs a graph of at least two rows), the paged attention op
    included. With 8 experts and top-6 an expert serves about three
    quarters of the ``B`` rows, so the qmv kernels' shared expert reads
    (KON-239) -- at ``B = 8`` a group typically spans two tiles -- are
    covered too.

    Against the batch-1 graph the rows are a different graph's output, so the
    max |delta| is printed, not asserted bitwise; the 1e-3 bar only catches a
    swapped row, a misrouted expert or a mixed-up cache, which move it by O(1).
    """
    mode = "int8" if int8 else "bf16"
    pipeline = _small_pipeline(accelerator_session, int8=int8, seed=seed)
    small = pipeline.config.decoder
    window = small.sliding_window_size
    states = []
    for b in range(8):
        prefill_len = 3 + b
        if b % 2 == 0:  # warming up: appends at row `length`
            states.append({"prefill_len": prefill_len, "length": prefill_len + b, "ring_pos": 0, "position": prefill_len + b})
        else:  # full ring: overwrites row `prefill_len + ring_pos`
            states.append({"prefill_len": prefill_len, "length": prefill_len + window, "ring_pos": b, "position": 30 + b})
    tokens = [(7 * b + 3) % small.vocab_size for b in range(8)]
    rng = np.random.default_rng(59)
    host_rows = [_random_rows(rng, pipeline, state["prefill_len"]) for state in states]

    def caches(count: int) -> list[KvCache]:
        return [_paged_cache(pipeline, host_rows[b], **states[b]) for b in range(count)]

    stepped: dict[int, tuple[np.ndarray, list[list[np.ndarray]]]] = {}
    for batch in (2, 4, 8):
        rows = caches(batch)
        logits = pipeline.decode_rows(rows, tokens[:batch], [_UNREAD] * batch)
        assert logits.shape == (batch, small.vocab_size) and logits.dtype == np.float32
        assert np.all(np.isfinite(logits))
        stepped[batch] = (np.array(logits), [_whole(cache) for cache in rows])
        for cache in rows:
            cache.release()

    widest_logits, widest_kv = stepped[8]
    for batch in (2, 4):
        logits, kv = stepped[batch]
        for b in range(batch):
            assert np.array_equal(logits[b], widest_logits[b]), (batch, b)
            for got, want in zip(kv[b], widest_kv[b], strict=True):
                assert np.array_equal(got, want), (batch, b)
    print(f"[uocr] {mode} batched decode: every row bitwise equal across B = 2, 4, 8 (logits and caches)")

    for b in range(8):
        cache = _paged_cache(pipeline, host_rows[b], **states[b])
        reference = pipeline.decode_step(cache, tokens[b], _UNREAD)
        f64 = lambda a: a.astype(np.float64)  # noqa: E731
        delta_logits = float(np.max(np.abs(f64(widest_logits[b]) - f64(reference))))
        delta_kv = max(float(np.max(np.abs(f64(got) - f64(want)))) for got, want in zip(widest_kv[b], _whole(cache), strict=True))
        cache.release()
        print(
            f"[uocr]   row {b} (B = 2/4/8) vs the batch-1 {mode} step: bitwise {bool(np.array_equal(widest_logits[b], reference))}, "
            f"max |delta| logits {delta_logits:.3e}, kv {delta_kv:.3e}"
        )
        assert delta_logits <= 1e-3 and delta_kv <= 1e-3, (b, delta_logits, delta_kv)


# --------------------------------------------------------------------------
# KON-235, numerically: the in-graph guard is the reference guard on the guard-free logits
# --------------------------------------------------------------------------


def _guard_free(monkeypatch: pytest.MonkeyPatch, build: Any) -> Any:
    """``build()`` with :func:`~unlimited_ocr_max.ngram.apply_ngram_guard` staged as the identity: the graph without the op."""
    with monkeypatch.context() as patch:
        patch.setattr(graphs_module, "apply_ngram_guard", lambda logits, history, ngram_size: logits)
        return build()


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=gpu_only)])
def test_the_in_graph_guard_is_the_reference_guard_on_the_guard_free_logits(
    device: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Batch 1 and B = 2, the device's shipped configuration, ``_small_decoder_config`` with random weights.

    Each step runs the same graph twice from identical caches -- once at the
    pipeline's ``n = 3`` and once with the guard switched off (``n = 0`` and a
    zero history) -- next to the same graph built without the op. ``n = 0``
    must return the guard-free logits bit for bit, ``n = 3`` exactly those
    logits with the oracle's banned ids set to ``BLOCKED``: one loaded graph
    serves both, so switching the guard needs no recompile. Row 0's sequence
    plants a repeat (its last two ids recur eight ids back, inside the
    16-id window, so the id that followed them is banned); row 1's ids are all
    distinct, so nothing is banned there while row 0 bans in the same op.
    """
    from max.driver import CPU
    from max.engine import InferenceSession

    if device == "gpu":
        driver, session = request.getfixturevalue("_accelerator_session")
    else:
        driver = CPU()
        session = InferenceSession(devices=[driver])
    small = _small_decoder_config(int8=False)
    config = SimpleNamespace(decoder=small, dtype=DType.bfloat16)
    window, layers = small.sliding_window_size, small.num_hidden_layers
    pipeline = UnlimitedOcrPipeline(
        config,
        vision_state_dict={},
        language_state_dict=_random_language_weights(config, seed=61),
        seq_len=32,
        max_new_tokens=32,
        device=DEVICES[device],
        driver_device=driver,
        session=session,
        ngram_size=3,
        max_batch_size=2,
    )

    sequences = [[*range(10, 28), 20, 21], list(range(30, 50))]  # the fed token is each one's last id
    tokens = [sequence[-1] for sequence in sequences]
    prefill_len = len(sequences[0]) - 1
    banned = [_banned(sequence, ngram_size=3, window=window) for sequence in sequences]
    assert banned == [{22}, set()]
    rng = np.random.default_rng(67)
    shape = (prefill_len, small.num_key_value_heads, small.head_dim)
    prefixes = [
        ([rng.standard_normal(shape).astype(np.float32) for _ in range(layers)],
         [rng.standard_normal(shape).astype(np.float32) for _ in range(layers)])
        for _ in sequences
    ]

    def cache(b: int) -> KvCache:
        """Row ``b``'s cache, freshly seeded in fresh pages: every run starts from the same state."""
        fresh = pipeline.kv_pool.allocate(prefill_len=prefill_len, window=window)
        fresh.seed(*prefixes[b])
        return fresh

    def guarded(free: np.ndarray, b: int) -> np.ndarray:
        want = free.copy()
        want[sorted(banned[b])] = BLOCKED
        return want

    free_b1 = _guard_free(monkeypatch, lambda: pipeline.decode_graph(1))
    pipeline._decode.clear()
    with_op_b1 = pipeline.decode_graph(1)
    free_b2 = _guard_free(monkeypatch, lambda: pipeline.decode_graph(2))
    pipeline._decode.clear()
    with_op_b2 = pipeline.decode_graph(2)
    for graph in (free_b1, free_b2):
        assert "ngram_block" not in str(graph[0].graph)

    def step(graph_b1: Any, graph_b2: Any, ngram_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Row 0 alone through the batch-1 graph, then both rows through the B = 2 graph."""
        pipeline._decode[1], pipeline._decode[2] = graph_b1, graph_b2
        pipeline.ngram_size = ngram_size
        alone = cache(0)
        one = np.array(pipeline.decode_step(alone, tokens[0], sequences[0]))
        alone.release()
        pair = [cache(0), cache(1)]
        two = np.array(pipeline.decode_rows(pair, tokens, sequences))
        for each in pair:
            each.release()
        return one, two

    built: list[str] = []

    def counted(name: str, real: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            built.append(name)
            return real(*args, **kwargs)

        return wrapper

    # From here on no graph may be built or loaded: switching the guard is a new input, not a new graph.
    monkeypatch.setattr(pipeline_module, "build_decode_graph", counted("build", pipeline_module.build_decode_graph))
    monkeypatch.setattr(pipeline.session, "load", counted("load", pipeline.session.load))
    free_one, free_two = step(free_b1, free_b2, 3)
    on_one, on_two = step(with_op_b1, with_op_b2, 3)
    off_one, off_two = step(with_op_b1, with_op_b2, 0)

    assert np.all(np.isfinite(free_one)) and np.all(np.isfinite(free_two))
    assert np.array_equal(off_one, free_one) and np.array_equal(off_two, free_two)
    assert np.array_equal(on_one, guarded(free_one, 0))
    for b in range(2):
        assert np.array_equal(on_two[b], guarded(free_two[b], b)), b
    assert not np.array_equal(on_two[0], free_two[0]) and np.array_equal(on_two[1], free_two[1])
    # The same two loaded graphs served n = 3 and n = 0: nothing was built or loaded for either.
    assert built == []
