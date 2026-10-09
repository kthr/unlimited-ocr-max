"""The decoder's int8 expert mode, model-free.

Staging checks run weightless: every ``Weight`` gets its FQN the way
``load_state_dict`` would assign it, no tensor data is resident, and the graph
is only built (``str(graph)``), never compiled. That is enough to pin the
declared names/dtypes/shapes against the adapter's stacking, to see both
language graphs stage the Mojo ops and to count staged ops.

The numeric checks (marked ``slow``: they compile the Mojo kernels through an
``InferenceSession``) run a bare :class:`~unlimited_ocr_max.decoder.MoE` on
synthetic weights. They run on CPU by default (``UOCR_TEST_DEVICE=gpu`` moves
them to the accelerator): the kernels are device-agnostic and decode stages the
same qmv ops on every device in both dtypes (KON-234), so CPU exercises exactly
the decode graph ops the served GPU path stages, minus the device.
"""

from __future__ import annotations

import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType

from unlimited_ocr_max.decoder import MoE, UnlimitedOcrDecoder
from unlimited_ocr_max.graphs import DecodeGraph, LanguageGraph, build_decode_graph, build_language_graph
from unlimited_ocr_max.model_config import INT8_GROUP_SIZE, ConfigError, DecoderConfig, UnlimitedOCRConfig
from unlimited_ocr_max.ngram import MOJO_KERNELS
from unlimited_ocr_max.weight_adapters import check_against_declared, stack_expert_weights

from _harness import device_ref, driver, rel_err, session

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# --------------------------------------------------------------------------
# synthetic configs
# --------------------------------------------------------------------------

#: The real towers: ``UnlimitedOCRConfig`` refuses anything else, and the projector
#: pins ``hidden_size`` to 1280. Everything else is shrunk.
_HF_BASE: dict[str, Any] = {
    "torch_dtype": "bfloat16",
    "hidden_size": 1280,
    "num_hidden_layers": 2,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "intermediate_size": 256,
    "vocab_size": 64,
    "max_position_embeddings": 64,
    "sliding_window_size": 16,
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 128,
    "n_routed_experts": 8,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "topk_method": "greedy",
    "n_group": 1,
    "topk_group": 1,
    "use_mla": False,
    "kv_lora_rank": None,
    "q_lora_rank": None,
    "qk_nope_head_dim": 0,
    "qk_rope_head_dim": 0,
    "v_head_dim": 128,
    "lm_head": True,
    "rm_head": False,
    "eos_token_id": 1,
    "vision_config": {
        "image_size": 1024,
        "width": {
            "clip-l-14-224": {"heads": 16, "image_size": 224, "layers": 24, "patch_size": 14, "width": 1024},
            "sam_vit_b": {
                "downsample_channels": [512, 1024],
                "global_attn_indexes": [2, 5, 8, 11],
                "heads": 12,
                "layers": 12,
                "width": 768,
            },
        },
    },
    "projector_config": {"input_dim": 2048, "n_embed": 1280, "projector_type": "linear"},
}


def _config(*, int8: bool, **overrides: Any) -> UnlimitedOCRConfig:
    return UnlimitedOCRConfig.from_hf_dict({**_HF_BASE, **overrides}).with_int8_experts(int8)


def _small_decoder_config(*, int8: bool) -> DecoderConfig:
    """A bare ``DecoderConfig`` for the numeric checks: hidden 256 (two groups), ffn 128 (one group)."""
    keys = set(DecoderConfig.__dataclass_fields__) - {"int8_experts"}
    fields = {key: value for key, value in _HF_BASE.items() if key in keys}
    fields.update(hidden_size=256, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=3)
    fields.update(
        norm_topk_prob=False, scoring_func="softmax", routed_scaling_factor=1.0, moe_layer_freq=1,
        hidden_act="silu", rms_norm_eps=1e-6, rope_theta=10000.0, rope_scaling=None, attention_bias=False,
        tie_word_embeddings=False,
    )
    return DecoderConfig(**fields, int8_experts=int8)


def _named(decoder: UnlimitedOcrDecoder) -> UnlimitedOcrDecoder:
    """Assign every Weight its FQN without loading data (what ``load_state_dict`` does first)."""
    for item in decoder._iter_named_weights():
        item[1].name = item[0]
    return decoder


def _experts(config: UnlimitedOCRConfig, layer: int, *, device: DeviceRef) -> dict[str, Any]:
    prefix = f"layers.{layer}.mlp.experts."
    declared = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=device).raw_state_dict()
    return {name: weight for name, weight in declared.items() if name.startswith(prefix)}


# --------------------------------------------------------------------------
# op census
# --------------------------------------------------------------------------

#: One MLIR result line; the pinned build emits the generic quoted form ``%1 = "rmo.add"(...)``.
_OP_LINE = re.compile(r'^\s*(?:%\S+\s*=\s*)?"?((?:r?mo)\.[a-zA-Z_.0-9]+)"?')

#: Declarations, not kernels: a weight is one ``mo.constant.external``, a graph constant one ``mo.constant``.
_DECLARATIONS = ("mo.constant", "mo.constant.external")


def _op_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        match = _OP_LINE.match(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def _staged(counts: Counter[str]) -> int:
    return sum(n for kind, n in counts.items() if kind not in _DECLARATIONS)


def _language_graph(
    config: UnlimitedOCRConfig,
    device: DeviceRef,
    *,
    decode: bool,
    batch: int = 1,
    resident: bool = False,
    served: bool = False,
) -> DecodeGraph | LanguageGraph:
    """A weightless language graph, staged: the decode step at ``batch`` rows (``max_seq_len=64``) or the prefill
    (``seq_len=5``, two image rows).

    ``resident`` declares every weight device-side; ``served`` is the shipped
    accelerator declaration -- resident, with the norms and the router fp32.
    """
    decoder = _named(
        UnlimitedOcrDecoder(
            config.decoder, dtype=config.dtype, device=device, norm_router_dtype=DType.float32 if served else None
        )
    )
    resident = resident or served
    if decode:
        return build_decode_graph(
            config, decoder, batch=batch, max_seq_len=64, device=device, device_resident_weights=resident
        )
    return build_language_graph(
        config, decoder, seq_len=5, n_image_tokens=2, device=device, device_resident_weights=resident
    )


def _dense_qmv_calls(dec: DecoderConfig, *, decode: bool) -> int:
    """``dense_bf16_qmv`` calls a language graph stages (KON-238).

    Decode reads every projection through it -- per layer q/k/v/o and the
    three FFN projections, dense or shared -- and ``lm_head``; prefill only
    ``lm_head`` on its last row.
    """
    return 7 * dec.num_hidden_layers + 1 if decode else 1


# --------------------------------------------------------------------------
# fold-proofing: does an op's operand trace back only to constants/weights?
# --------------------------------------------------------------------------

#: A result-binding line: one or more comma-separated SSA names, then its defining expression.
_SSA_ASSIGN = re.compile(r"^\s*(%[\w]+(?:,\s*%[\w]+)*)\s*=\s*(.*)$")
#: The op mnemonic at the very start of a right-hand side, quoted or bare.
_SSA_OP = re.compile(r'^"?([A-Za-z_][\w.]*)"?')


def _top_level_boundary(rhs: str) -> int | None:
    """Index of the top-level ``:`` that opens the type signature, or ``None`` if there isn't one.

    Tracks nesting depth over ``(){}[]<>`` so a colon inside an attribute
    (``{axis = -1 : si64}``, ``{value = ... : tensor<1xsi32>}``) is never
    mistaken for the boundary -- only a ``:`` at depth 0 is. ``->`` is
    special-cased and skipped whole: it is two literal characters, not a
    closing ``>``, and a generic attribute can contain one well before the
    real boundary (``rmo.mo.arg_nonzero<() -> image_embedding_elements>(%result) : ...``).
    A line with no type signature at all (``mo.chain.create()``) has no
    top-level colon, so this returns ``None`` and the whole right-hand side
    is operand territory.
    """
    depth = 0
    i, n = 0, len(rhs)
    while i < n:
        ch = rhs[i]
        if ch == "-" and i + 1 < n and rhs[i + 1] == ">":
            i += 2
            continue
        if ch in "({[<":
            depth += 1
        elif ch in ")}]>":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
        i += 1
    return None


def _rhs_op_and_operands(rhs: str, line: str) -> tuple[str, list[str]]:
    """``(op mnemonic, operand names)`` for one assignment's right-hand side, parsed generally.

    Every ``%name`` token before the top-level type-signature boundary
    (:func:`_top_level_boundary`) is an operand: plain
    (``rmo.select(%187, %189, %190)``), bracketed
    (``rmo.mo.transfer[%44] %51``), or alongside an attribute dict that
    carries its own operand-free text on either side of it
    (``mo.custom {...}(%23, %20, %201)``, ``rmo.top_k(%183) {axis = ...}``).
    This never assumes a particular ``%N`` or a fixed operand position, so
    it survives SSA renumbering and a new mix of attribute/operand order.

    Raises if no op mnemonic can be found at all. A line this cannot
    classify must fail loudly, not silently omit a ``defs`` entry --
    :func:`_is_weight_only` would then treat its result as a genuine graph
    input (always safe to call "not weight-only"), which is only correct
    for an actual ``%argN`` block argument.
    """
    boundary = _top_level_boundary(rhs)
    before = rhs if boundary is None else rhs[:boundary]
    op_match = _SSA_OP.match(before.strip())
    if not op_match:
        raise AssertionError(f"cannot classify the defining op of an SSA assignment -- unrecognised MLIR line: {line!r}")
    return op_match.group(1), re.findall(r"%[\w]+", before)


def _parse_ssa_defs(text: str) -> dict[str, tuple[str, list[str]]]:
    """Every SSA result name in the staged MLIR -> ``(op mnemonic, operand names)``.

    Every assignment line is parsed (:func:`_rhs_op_and_operands` raises
    rather than skipping one it cannot classify), so a name with no entry is
    only ever an ``%argN`` block argument -- one of the graph's own inputs,
    never produced by an assignment line -- which :func:`_is_weight_only`
    must treat as genuinely runtime.
    """
    defs: dict[str, tuple[str, list[str]]] = {}
    for line in text.splitlines():
        assign = _SSA_ASSIGN.match(line)
        if not assign:
            continue
        op, operands = _rhs_op_and_operands(assign.group(2), line)
        for name in (result.strip() for result in assign.group(1).split(",")):
            defs[name] = (op, operands)
    return defs


def _is_weight_only(name: str, defs: dict[str, tuple[str, list[str]]], memo: dict[str, bool]) -> bool:
    """True if ``name``'s whole transitive dependency chain bottoms out in declarations only.

    Only ``mo.constant``/``mo.constant.external`` (a graph constant or a
    declared weight) are weight-only leaves; a block argument with no def
    line (a real graph input) never is. A zero-operand op that is *not* a
    declaration -- e.g. ``mo.chain.create()``, a synchronization token
    carrying no tensor data at all -- is deliberately *not* weight-only
    either: it has nothing a weight-only expression could be folded into,
    so counting it as a leaf would risk calling a chain-gated expression
    "weight-only" when MAX never would. Anything else is weight-only
    exactly when every one of its operands is -- MAX's own "weight-only
    expression" criterion for the load-time constant fold (module
    docstring, KON-142 / OQ-158-A), not an approximation of it: a call is
    folded at ``session.load`` iff every one of its transitive inputs is a
    declaration.
    """
    if name in memo:
        return memo[name]
    memo[name] = False  # cycle guard; the staged graph is a DAG, so this is never actually read back
    entry = defs.get(name)
    if entry is None:
        result = False  # a graph input (%argN): never declared by an assignment line
    else:
        op, operands = entry
        if op in _DECLARATIONS:
            result = True
        elif not operands:
            result = False  # a zero-operand non-declaration op (e.g. a chain token): not weight data either
        else:
            result = all(_is_weight_only(operand, defs, memo) for operand in operands)
    memo[name] = result
    return result


def _custom_op_calls(text: str, symbol: str) -> list[tuple[str, list[str]]]:
    """``(result name, operand names)`` for every ``mo.custom`` call naming ``symbol``."""
    calls: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if f'symbol = "{symbol}"' not in line:
            continue
        assign = _SSA_ASSIGN.match(line)
        if not assign:
            continue
        _, operands = _rhs_op_and_operands(assign.group(2), line)
        calls.append((assign.group(1).split(",")[0].strip(), operands))
    return calls


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------


def test_int8_declares_the_adapters_stacked_names() -> None:
    """Exactly the six names KON-145 stacks an int8 checkpoint into, with its dtypes and shapes."""
    config = _config(int8=True)
    dec = config.decoder
    experts = _experts(config, 1, device=DeviceRef.GPU(0))
    groups_in, groups_ffn = dec.hidden_size // INT8_GROUP_SIZE, dec.moe_intermediate_size // INT8_GROUP_SIZE
    e, ffn, hidden = dec.n_routed_experts, dec.moe_intermediate_size, dec.hidden_size
    want = {
        "gate_proj": (DType.int8, (e, ffn, hidden)),
        "up_proj": (DType.int8, (e, ffn, hidden)),
        "down_proj": (DType.int8, (e, hidden, ffn)),
        "gate_proj_scales": (DType.float32, (e, ffn, groups_in)),
        "up_proj_scales": (DType.float32, (e, ffn, groups_in)),
        "down_proj_scales": (DType.float32, (e, hidden, groups_ffn)),
    }
    got = {
        name.removeprefix("layers.1.mlp.experts."): (w.dtype, tuple(int(d) for d in w.shape.static_dims))
        for name, w in experts.items()
    }
    assert got == want


def test_bf16_declares_exactly_the_three_stacks() -> None:
    config = _config(int8=False)
    experts = _experts(config, 1, device=DeviceRef.CPU())
    assert set(experts) == {f"layers.1.mlp.experts.{proj}" for proj in PROJECTIONS}
    assert {w.dtype for w in experts.values()} == {DType.bfloat16}


def test_int8_declarations_pass_the_adapters_check() -> None:
    """An int8 checkpoint's per-expert tensors, stacked by the adapter, match the declared experts exactly."""
    config = _config(int8=True)
    dec = config.decoder
    e, ffn, hidden = dec.n_routed_experts, dec.moe_intermediate_size, dec.hidden_size
    renamed: dict[str, Any] = {}
    for expert in range(e):
        for proj in PROJECTIONS:
            n, k = (hidden, ffn) if proj == "down_proj" else (ffn, hidden)
            stem = f"layers.1.mlp.experts.{expert}.{proj}"
            renamed[f"{stem}.weight"] = Buffer.from_dlpack(torch.zeros((n, k), dtype=torch.int8))
            scales = torch.ones((n, k // INT8_GROUP_SIZE), dtype=torch.float32)
            renamed[f"{stem}.weight_scales"] = Buffer.from_dlpack(scales)
    stacked = stack_expert_weights(renamed, num_experts=e)
    check_against_declared(stacked, _experts(config, 1, device=DeviceRef.GPU(0)))
    # And the bf16 declaration refuses the int8 file: the dtype half of the check is what tells them apart.
    with pytest.raises(Exception, match="missing|unexpected|dtype"):
        check_against_declared(stacked, _experts(_config(int8=False), 1, device=DeviceRef.GPU(0)))


def test_int8_needs_group_aligned_dims() -> None:
    with pytest.raises(ConfigError, match="divisible by the group size"):
        _config(int8=True, moe_intermediate_size=100)


# --------------------------------------------------------------------------
# both language graphs stage in int8 mode
# --------------------------------------------------------------------------


def test_both_language_graphs_build_in_int8_mode() -> None:
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dref = DeviceRef.GPU(0)
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace

    decode = _op_counts(str(_language_graph(config, dref, decode=True).graph))
    # gate, up, down qmv per MoE layer, the n-gram guard, MAX's paged k/v stores and attention per layer,
    # and the dense projections
    assert decode["mo.custom"] == 3 * n_moe + 1 + 3 * dec.num_hidden_layers + _dense_qmv_calls(dec, decode=True)
    prefill = _op_counts(str(_language_graph(config, dref, decode=False).graph))
    # one dequant per expert per projection, and lm_head's last row
    assert prefill["mo.custom"] == 3 * dec.n_routed_experts * n_moe + _dense_qmv_calls(dec, decode=False)

    for text in (str(_language_graph(config, dref, decode=decode).graph) for decode in (True, False)):
        assert "moe_int8_qmv" in text or "int8_dequant_expert" in text
        assert "mo.grouped.matmul.ragged" not in text and "mo.moe.create.indices" not in text
        assert "layers.1.mlp.experts.gate_proj_scales" in text


def test_both_language_graphs_build_in_int8_mode_with_device_resident_weights() -> None:
    """KON-161: pre-adding the weights device-side intercepts the int8 stacks' implicit add path too.

    The stacks and their scales reach ``ops.custom`` whole through the same
    ``Graph.add_weight`` cache the matmul weights use, so the declaration flips
    their placement without changing what the graphs stage: the same kernel
    calls, the same scales names, no native-MoE ops.
    """
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dref = DeviceRef.GPU(0)
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace

    for decode, want_custom in (
        # + the n-gram guard (KON-235), the paged stores and attention, three per layer (KON-237),
        # and the dense projections (KON-238)
        (True, 3 * n_moe + 1 + 3 * dec.num_hidden_layers + _dense_qmv_calls(dec, decode=True)),
        (False, 3 * dec.n_routed_experts * n_moe + _dense_qmv_calls(dec, decode=False)),
    ):
        text = str(_language_graph(config, dref, decode=decode, resident=True).graph)
        counts = _op_counts(text)
        assert counts["mo.custom"] == want_custom
        assert "moe_int8_qmv" in text or "int8_dequant_expert" in text
        assert "mo.grouped.matmul.ragged" not in text and "mo.moe.create.indices" not in text
        assert "layers.1.mlp.experts.gate_proj_scales" in text


# --------------------------------------------------------------------------
# OQ-158-A / 82ab60f: the prefill dequant index must stay load-unfoldable
# --------------------------------------------------------------------------


def test_int8_prefill_dequant_expert_index_is_not_load_foldable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the property that ended OQ-158-A, at the staged-graph level, model-free.

    Before 82ab60f, ``StackedExperts.expert``'s int8 index was a bare
    ``ops.constant``: every transitive input of each ``int8_dequant_expert``
    call was a declaration (a graph constant or an external weight), so MAX
    treated the whole call as a weight-only expression, ran it at
    ``session.load``, and folded the fp32 upcast into device constants
    (9.02 GiB for the routed experts, EXPERIMENTS.md ID 27). Serving the
    shared device weight registry for that checkpoint then produced 0/12
    correct pages in both request orders while every in-process test --
    including the numeric ones in this file -- stayed bitwise clean: the
    fold is a load-time, registry-adjacent effect that a value comparison
    inside one process never exercises. 82ab60f added ``MoE._runtime_zero``:
    the index is now ``runtime_zero + constant``, where ``runtime_zero`` is
    derived from the router's top-k output, so the call keeps a genuine
    graph-input ancestor and MAX cannot fold it.

    Neither an op *count* nor the existing bitwise-equality check
    (``test_int8_prefill_with_a_runtime_expert_index_computes_the_same_bits``,
    below) pins this: a load-time fold leaves the op count and the computed
    values exactly as they were (same kernel, same stacks, same numbers) --
    whether MAX can evaluate the index once, outside the per-request graph,
    is a question about the staged MLIR's dependency shape, not about any
    value it produces. So this walks the SSA def-use chain of every
    ``int8_dequant_expert`` call's expert-index operand (parsing is
    SSA-renumbering-proof: every assignment line is classified generally, by
    finding its top-level type-signature boundary and regexing ``%name``
    tokens out of everything before it, never a hard-coded ``%N`` or a fixed
    operand position) and asserts it is not "weight-only" -- MAX's own
    criterion for the fold (module docstring, KON-142) -- built with
    ``device_resident_weights=True``, the registry declaration that made the
    corruption reachable in the first place.
    """
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dref = DeviceRef.GPU(0)
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace

    def _device_resident_prefill_text() -> str:
        return str(_language_graph(config, dref, decode=False, resident=True).graph)

    text = _device_resident_prefill_text()
    calls = _custom_op_calls(text, "int8_dequant_expert")
    assert len(calls) == 3 * dec.n_routed_experts * n_moe  # one dequant per expert per projection per MoE layer

    defs = _parse_ssa_defs(text)
    memo: dict[str, bool] = {}
    foldable = [result for result, operands in calls if _is_weight_only(operands[2], defs, memo)]
    assert not foldable, (
        f"{len(foldable)}/{len(calls)} int8_dequant_expert calls have a load-foldable expert-index "
        f"operand (every transitive input is mo.constant/mo.constant.external): {foldable[:5]} -- "
        "see 82ab60f / OQ-158-A"
    )

    # The detector has teeth: simulating the pre-82ab60f shape (a bare constant index, no runtime
    # zero) on the identical builder must flip every one of these calls back to foldable.
    monkeypatch.setattr(MoE, "_runtime_zero", lambda self, indices: None)
    pre_fix_text = _device_resident_prefill_text()
    pre_fix_calls = _custom_op_calls(pre_fix_text, "int8_dequant_expert")
    assert len(pre_fix_calls) == len(calls)
    pre_fix_defs = _parse_ssa_defs(pre_fix_text)
    pre_fix_memo: dict[str, bool] = {}
    assert all(_is_weight_only(operands[2], pre_fix_defs, pre_fix_memo) for _, operands in pre_fix_calls)


def test_int8_decode_graph_has_no_dequant_call_and_moe_int8_qmv_stays_runtime() -> None:
    """``build_decode_graph``'s int8 path (batch 1 here) never stages ``int8_dequant_expert``.

    The decode graph's MoE (:meth:`MoE.decode_rows`) takes ``_routed_qmv``, which calls
    ``moe_int8_qmv`` directly against the whole int8 stacks with the top-k
    expert ids from this step's own gate -- never a constant -- so there is
    no load-foldable index here for 82ab60f to have had to guard, and
    nothing for this test to pin that the fix commit changed. It pins both
    halves of that claim instead of assuming them: no ``int8_dequant_expert``
    call appears in the decode graph at all, and the ``moe_int8_qmv`` calls
    that do appear carry an expert-ids operand that is not weight-only
    either (the same SSA trace as the prefill test above), so a future
    change that routed decode through a constant expert id would be caught
    here too.
    """
    config = _config(int8=True, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dref = DeviceRef.GPU(0)
    text = str(_language_graph(config, dref, decode=True, resident=True).graph)
    assert "int8_dequant_expert" not in text

    calls = _custom_op_calls(text, "moe_int8_qmv")
    assert calls
    defs = _parse_ssa_defs(text)
    memo: dict[str, bool] = {}
    foldable = [result for result, operands in calls if _is_weight_only(operands[1], defs, memo)]
    assert not foldable, f"{len(foldable)}/{len(calls)} moe_int8_qmv calls have a weight-only expert-ids operand"


def test_model_detects_int8_from_the_expert_dtypes_only() -> None:
    """``UnlimitedOCRModel._weights_are_int8`` reads expert dtypes off ``Weights.data()`` and touches nothing else."""
    from unlimited_ocr_max.model import UnlimitedOCRModel

    class Source:
        def __init__(self, dtype: DType | None) -> None:
            self.dtype = dtype

        def data(self) -> Any:
            if self.dtype is None:
                raise AssertionError("a non-expert tensor was materialised")
            return SimpleNamespace(dtype=self.dtype)

    def checkpoint(weight: DType, scales: DType | None) -> dict[str, Any]:
        out: dict[str, Any] = {"model.norm.weight": Source(None), "lm_head.weight": Source(None)}
        for expert in range(2):
            for proj in PROJECTIONS:
                stem = f"model.layers.1.mlp.experts.{expert}.{proj}.weight"
                out[stem] = Source(weight)
                if scales is not None:
                    out[f"{stem}_scales"] = Source(scales)
        return out

    assert UnlimitedOCRModel._weights_are_int8(checkpoint(DType.int8, DType.float32)) is True
    assert UnlimitedOCRModel._weights_are_int8(checkpoint(DType.bfloat16, None)) is False


# --------------------------------------------------------------------------
# op census and routing: decode MoE layers in both dtypes, and bf16 prefill
# --------------------------------------------------------------------------


def test_decode_op_census_per_moe_layer_is_one_qmv_path_in_both_dtypes() -> None:
    """Per MoE layer, bf16 decode stages exactly int8's ops but for the scale declarations (KON-234).

    The per-layer figure is the difference between a 3-layer and a 2-layer
    decoder (layer 0 is dense in both), so everything outside the added layer
    cancels exactly; its attention is the same paged ops in both dtypes
    (KON-237). Both dtypes run :meth:`MoE._routed_qmv`: three kernel
    calls and the mixing matmul, no index op, no permute/restore gathers.
    The only kind whose count differs is ``mo.constant.external``: int8
    declares a scales stack next to each of its three weight stacks. Before
    KON-234 bf16 staged MAX's native routing here on an accelerator (83 staged
    ops per layer against int8's 79) and the hand-rolled top-6 chain on CPU
    (196).
    """
    dref = DeviceRef.GPU(0)
    per_layer: dict[str, dict[str, int]] = {}
    for mode, int8 in (("bf16", False), ("int8", True)):
        two, three = (
            _op_counts(str(_language_graph(_config(int8=int8, num_hidden_layers=n), dref, decode=True).graph)) for n in (2, 3)
        )
        per_layer[mode] = {kind: three[kind] - two[kind] for kind in three if three[kind] != two[kind]}
        print(f"[uocr] decode ops per MoE layer, {mode}: all {sum(per_layer[mode].values())}, staged {_staged(Counter(per_layer[mode]))}")
    bf16, int8 = per_layer["bf16"], per_layer["int8"]
    moved = {kind: (bf16.get(kind, 0), int8.get(kind, 0)) for kind in bf16.keys() | int8.keys() if bf16.get(kind, 0) != int8.get(kind, 0)}
    assert moved == {"mo.constant.external": (bf16["mo.constant.external"], bf16["mo.constant.external"] + 3)}
    # the three qmv calls, the layer's two paged stores and one paged attention, and its seven
    # dense_bf16_qmv projections (q/k/v/o and the shared experts' three, KON-238)
    assert bf16["mo.custom"] == 3 + 3 + 7
    assert _staged(Counter(bf16)) == _staged(Counter(int8)) > 0


@pytest.mark.parametrize(("device", "resident"), [("cpu", False), ("gpu", False), ("gpu", True)], ids=["cpu", "gpu-plain", "gpu-resident"])
def test_bf16_prefill_keeps_its_routing_and_reads_only_lm_head_through_a_kernel(device: str, resident: bool) -> None:
    """KON-234 moved bf16 decode only: the bf16 prefill graph's experts stage no Mojo op.

    On an accelerator its routed experts are still MAX's native kernels (one
    ``moe_create_indices`` and three ``grouped_matmul_ragged`` per MoE layer),
    on CPU the dense 64-term chain. Its one Mojo op is ``dense_bf16_qmv`` on
    ``lm_head``'s last row (KON-238), so it names the kernel package.
    """
    config = _config(int8=False, num_hidden_layers=3)  # layer 0 dense, layers 1-2 MoE
    dec = config.decoder
    n_moe = dec.num_hidden_layers - dec.first_k_dense_replace
    dref = DeviceRef.CPU() if device == "cpu" else DeviceRef.GPU(0)
    text = str(_language_graph(config, dref, decode=False, resident=resident).graph)
    assert "_kernel_library_paths = []" not in text
    assert "moe_bf16_qmv" not in text and "moe_int8_qmv" not in text and "int8_dequant_expert" not in text
    assert text.count('symbol = "dense_bf16_qmv"') == _dense_qmv_calls(dec, decode=False)
    if device == "cpu":
        assert _op_counts(text)["mo.custom"] == _dense_qmv_calls(dec, decode=False)
    else:
        assert text.count('symbol = "mo.moe.create.indices"') == n_moe
        assert text.count('symbol = "mo.grouped.matmul.ragged"') == 3 * n_moe
        assert _op_counts(text)["mo.custom"] == 4 * n_moe + _dense_qmv_calls(dec, decode=False)


# --------------------------------------------------------------------------
# numeric: a bare MoE layer, int8 decode vs bf16 (slow: compiles the kernels)
# --------------------------------------------------------------------------


def _bf16(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(torch.bfloat16)


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-group absmax int8 of a bf16 ``[E, N, K]`` stack: ``(q int8, scales fp32, dequantized bf16)``.

    The dequantized copy is ``bfloat16(float32(q) * scale)`` -- the value the
    kernel's ``int8_dequant_expert`` produces bit for bit.
    """
    w = weight.float()
    e, n, k = w.shape
    grouped = w.reshape(e, n, k // INT8_GROUP_SIZE, INT8_GROUP_SIZE)
    amax = grouped.abs().amax(dim=3)
    scales = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).contiguous()
    q = torch.round(grouped / scales[..., None]).clamp(-127, 127)
    dequantized = (q * scales[..., None]).reshape(e, n, k).to(torch.bfloat16)
    return q.to(torch.int8).reshape(e, n, k).contiguous(), scales, dequantized


def _moe_weights(config: DecoderConfig, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``(bf16 originals, int8 stacks + scales, bf16 dequantized)`` state dicts for a bare ``MoE``."""
    rng = np.random.default_rng(seed)
    e, hidden, ffn = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
    shared = config.shared_experts_dim
    common = {
        "gate.gate_score.weight": _bf16(rng.standard_normal((e, hidden)) * 0.1),
        "shared_experts.gate_proj.weight": _bf16(rng.standard_normal((shared, hidden)) * 0.05),
        "shared_experts.up_proj.weight": _bf16(rng.standard_normal((shared, hidden)) * 0.05),
        "shared_experts.down_proj.weight": _bf16(rng.standard_normal((hidden, shared)) * 0.05),
    }
    shapes = {"gate_proj": (e, ffn, hidden), "up_proj": (e, ffn, hidden), "down_proj": (e, hidden, ffn)}
    original, int8, dequantized = dict(common), dict(common), dict(common)
    for proj, shape in shapes.items():
        stack = _bf16(rng.standard_normal(shape) * 0.05)
        q, scales, deq = _quantize(stack)
        original[f"experts.{proj}"] = stack
        int8[f"experts.{proj}"] = q
        int8[f"experts.{proj}_scales"] = scales
        dequantized[f"experts.{proj}"] = deq
    return original, int8, dequantized


def _as_declared(weights: dict[str, Any]) -> dict[str, Any]:
    """A bare ``MoE``'s checkpoint-form state dict as the module declares it: the shared experts'
    projections K-blocked, ``[K / KBLOCK, N, KBLOCK]`` (the router and the stacks as they are)."""
    from unlimited_ocr_max.decoder import KBLOCK

    return {
        name: w.reshape(w.shape[0], -1, KBLOCK).permute(1, 0, 2).contiguous() if name.startswith("shared_experts.") else w
        for name, w in weights.items()
    }


def _moe_model(config: DecoderConfig, weights: dict[str, Any], *, seq: int, tag: str, decode_rows: bool = False):
    """A bare ``MoE`` over ``[seq, hidden]``: the decode step's (``decode_rows``) at one row or with ``decode_rows``, else its prefill call."""
    dref = device_ref()
    moe = MoE(config, dtype=DType.bfloat16, device=dref)
    moe.load_state_dict(_as_declared(weights))
    with Graph(
        f"test_moe_{tag}_{seq}",
        input_types=[TensorType(DType.float32, [seq, config.hidden_size], device=dref)],
        custom_extensions=[MOJO_KERNELS],  # decode runs a qmv kernel in both dtypes; int8 prefill dequantizes
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(moe.decode_rows(x) if decode_rows or seq == 1 else moe(x))
    return session().load(graph, weights_registry=moe.state_dict())


def _run(model, x: np.ndarray) -> np.ndarray:
    return model.execute(Buffer.from_numpy(np.ascontiguousarray(x)).to(driver()))[0].to(CPU()).to_numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _decode_reference_fp64(config: DecoderConfig, int8: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """The int8 MoE layer's exact math in float64: softmax gate, top-k, ``q * scale`` experts, shared experts.

    This is what the int8 graph computes up to fp32 rounding: the kernel uses
    the exact fp32 ``q * scale``, not a bf16-rounded copy of it, so a bf16
    graph fed the dequantized weights is *not* the tight reference (its weights
    carry ~2^-9 relative rounding noise each).
    """
    def dequantized(proj: str) -> np.ndarray:
        q, scales = _f64(int8[f"experts.{proj}"]), _f64(int8[f"experts.{proj}_scales"])
        return q * np.repeat(scales, INT8_GROUP_SIZE, axis=2)

    return _moe_reference_fp64(config, int8, x, [dequantized(proj) for proj in PROJECTIONS])


def _f64(tensor: torch.Tensor) -> np.ndarray:
    return tensor.float().numpy().astype(np.float64)


def _moe_reference_fp64(config: DecoderConfig, weights: dict[str, Any], x: np.ndarray, stacks: list[np.ndarray]) -> np.ndarray:
    """One token's MoE layer in float64: softmax gate, top-k, the given ``[E, N, K]`` gate/up/down stacks, shared experts."""
    xf = x.astype(np.float64)[0]
    silu = lambda a: a / (1.0 + np.exp(-a))  # noqa: E731
    logits = _f64(weights["gate.gate_score.weight"]) @ xf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(-probs, kind="stable")[: config.num_experts_per_tok]
    gate, up, down = stacks
    routed = sum(probs[e] * (down[e] @ (silu(gate[e] @ xf) * (up[e] @ xf))) for e in top)
    shared_gate, shared_up, shared_down = (_f64(weights[f"shared_experts.{proj}.weight"]) for proj in PROJECTIONS)
    shared = shared_down @ (silu(shared_gate @ xf) * (shared_up @ xf))
    return (routed + shared)[None, :]


def _bf16_decode_reference_fp64(config: DecoderConfig, original: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """The bf16 MoE layer's exact math in float64: the bf16 stacks' values, exactly, for every expert."""
    return _moe_reference_fp64(config, original, x, [_f64(original[f"experts.{proj}"]) for proj in PROJECTIONS])


@pytest.mark.slow
@pytest.mark.parametrize("layer", [1, 2])
def test_int8_decode_moe_matches_bf16_per_layer(layer: int) -> None:
    """seq == 1: ``moe_int8_qmv`` over the top-6 vs the bf16 decode path, one independent weight draw per MoE layer.

    Two bars. Against the bf16 layer on the *original* weights the gap is the
    int8 quantization itself: cosine >= 0.999 (the acceptance bar). Against the
    float64 evaluation of the exact int8 math the only difference left is fp32
    rounding, which pins the wiring -- the right experts, the right router
    weights, the right mixing: cosine >= 0.99999 and a 1e-4 relative max-abs gate.
    """
    bf16_cfg, int8_cfg = _small_decoder_config(int8=False), _small_decoder_config(int8=True)
    original, int8, _ = _moe_weights(bf16_cfg, seed=layer)
    rng = np.random.default_rng(100 + layer)
    x = rng.standard_normal((1, bf16_cfg.hidden_size)).astype(np.float32)

    got = _run(_moe_model(int8_cfg, int8, seq=1, tag=f"int8_l{layer}"), x)
    ref = _run(_moe_model(bf16_cfg, original, seq=1, tag=f"bf16_l{layer}"), x)
    exact = _decode_reference_fp64(int8_cfg, int8, x)

    cos, cos_exact = _cosine(got, ref), _cosine(got, exact)
    rel_exact = float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))
    print(
        f"[uocr] int8 decode MoE layer {layer}: cosine vs bf16 {cos:.6f}, "
        f"vs exact int8 math {cos_exact:.9f} (rel {rel_exact:.2e})"
    )
    assert np.all(np.isfinite(got))
    assert cos >= 0.999
    assert cos_exact >= 0.99999
    assert rel_exact <= 1e-4


@pytest.mark.slow
def test_int8_multi_row_qmv_matches_the_exact_math_and_the_dense_chain() -> None:
    """``MoE.decode_rows`` on 8 rows (``moe_int8_qmv`` over 48 (token, expert) rows) vs the dense chain on the same rows.

    The two are different arithmetic, not just a different order: the dense
    chain multiplies by the bf16-rounded ``int8_dequant_expert`` weights and
    sums every expert (zero-weighted ones included) in ascending id, while
    qmv uses the exact fp32 ``q * scale`` over the top-6 alone. So the tight
    bar is each row against the float64 evaluation of the exact int8 math
    (:func:`_decode_reference_fp64`, the batch-1 test's bars), and the dense
    chain only has to agree to the bf16 weight rounding. Each row must also
    match the one-row decode MoE (``_routed_qmv`` at ``seq == 1``) to fp32
    rounding: same kernel, but a different mixing and gate matmul shape.
    """
    int8_cfg = _small_decoder_config(int8=True)
    _, int8, _ = _moe_weights(_small_decoder_config(int8=False), seed=19)
    x = np.random.default_rng(23).standard_normal((8, int8_cfg.hidden_size)).astype(np.float32)

    rows = _run(_moe_model(int8_cfg, int8, seq=8, tag="int8_rows", decode_rows=True), x)
    dense = _run(_moe_model(int8_cfg, int8, seq=8, tag="int8_dense"), x)
    one_row = _moe_model(int8_cfg, int8, seq=1, tag="int8_one_row")
    single = np.concatenate([_run(one_row, x[i : i + 1]) for i in range(x.shape[0])])
    exact = np.concatenate([_decode_reference_fp64(int8_cfg, int8, x[i : i + 1]) for i in range(x.shape[0])])

    rel_exact, rel_dense, rel_single = rel_err(rows, exact), rel_err(rows, dense), rel_err(rows, single)
    cos_exact = min(_cosine(rows[i], exact[i]) for i in range(x.shape[0]))
    cos_dense = min(_cosine(rows[i], dense[i]) for i in range(x.shape[0]))
    print(
        f"[uocr] int8 multi-row qmv (8 rows): vs exact int8 math rel {rel_exact:.2e} (min cos {cos_exact:.9f}); "
        f"vs dense chain rel {rel_dense:.2e} (min cos {cos_dense:.9f}, dense vs exact rel {rel_err(dense, exact):.2e}); "
        f"vs one-row qmv rel {rel_single:.2e}, bitwise {bool(np.array_equal(rows, single))}"
    )
    assert np.all(np.isfinite(rows))
    assert cos_exact >= 0.99999 and rel_exact <= 1e-4
    assert cos_dense >= 0.9999 and rel_dense <= 1e-2
    assert rel_single <= 1e-5


@pytest.mark.slow
def test_int8_prefill_moe_matches_bf16_on_dequantized_weights() -> None:
    """seq > 1: ``int8_dequant_expert`` feeds the 64-term chain the same bf16 weights the bf16 path slices.

    The dequantized values are bit-exact (KON-144), and the chain after them is
    the same ops in the same order, so the two dense outputs agree to fp32
    rounding; against the original weights the gap is the quantization.
    """
    bf16_cfg, int8_cfg = _small_decoder_config(int8=False), _small_decoder_config(int8=True)
    original, int8, dequantized = _moe_weights(bf16_cfg, seed=7)
    x = np.random.default_rng(11).standard_normal((4, bf16_cfg.hidden_size)).astype(np.float32)

    got = _run(_moe_model(int8_cfg, int8, seq=4, tag="int8_prefill"), x)
    ref_deq = _run(_moe_model(bf16_cfg, dequantized, seq=4, tag="bf16deq_prefill"), x)
    ref = _run(_moe_model(bf16_cfg, original, seq=4, tag="bf16_prefill"), x)

    bitwise = bool(np.array_equal(got, ref_deq))
    cos = min(_cosine(got[i], ref[i]) for i in range(x.shape[0]))
    print(f"[uocr] int8 prefill MoE: bitwise vs dequantized bf16 {bitwise}, min row cosine vs bf16 {cos:.6f}")
    np.testing.assert_allclose(got, ref_deq, rtol=1e-5, atol=1e-6)
    assert cos >= 0.999


@pytest.mark.slow
def test_int8_prefill_with_a_runtime_expert_index_computes_the_same_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fold-proof prefill (the dequant index derived from the router) is bitwise the constant-index build.

    With only constants for inputs ``int8_dequant_expert`` is folded at load into
    an fp32 copy of every expert (ID 27); the runtime zero keeps it a
    prefill-time op. Same kernel, same int8 stacks, same index values, same
    64-term chain after it -- so the output must not move a bit.
    """
    from unlimited_ocr_max.decoder import MoE

    int8_cfg = _small_decoder_config(int8=True)
    _, int8, _ = _moe_weights(_small_decoder_config(int8=False), seed=13)
    x = np.random.default_rng(17).standard_normal((4, int8_cfg.hidden_size)).astype(np.float32)

    runtime = _run(_moe_model(int8_cfg, int8, seq=4, tag="int8_prefill_runtime_idx"), x)
    monkeypatch.setattr(MoE, "_runtime_zero", lambda self, indices: None)  # the pre-fix, constant-index build
    constant = _run(_moe_model(int8_cfg, int8, seq=4, tag="int8_prefill_const_idx"), x)
    assert np.all(np.isfinite(runtime))
    assert np.array_equal(runtime, constant), (
        f"max abs diff {float(np.max(np.abs(runtime.astype(np.float64) - constant.astype(np.float64)))):.3e}"
    )


# --------------------------------------------------------------------------
# numeric: a bare bf16 MoE layer, decode through moe_bf16_qmv (KON-234)
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("layer", [1, 2])
def test_bf16_decode_moe_matches_the_exact_math_and_the_prefill_path(layer: int) -> None:
    """seq == 1: ``moe_bf16_qmv`` over the top-6 vs the float64 evaluation of the bf16 layer, and vs the prefill path.

    Against float64 the only difference left is fp32 rounding, which pins the
    wiring -- the right experts, the right router weights, the right mixing:
    cosine >= 0.99999 and a 1e-4 relative max-abs gate (the int8 decode
    test's bars). The prefill path (the dense 64-term chain on CPU, MAX's
    native routing on an accelerator) on the same row, fed as a two-row
    sequence, is the other fp32 spelling of the same math; it has to agree to
    fp32 rounding as well.
    """
    bf16_cfg = _small_decoder_config(int8=False)
    original, _, _ = _moe_weights(bf16_cfg, seed=layer)
    rng = np.random.default_rng(100 + layer)
    x = rng.standard_normal((1, bf16_cfg.hidden_size)).astype(np.float32)

    got = _run(_moe_model(bf16_cfg, original, seq=1, tag=f"bf16_qmv_l{layer}"), x)
    prefill = _run(_moe_model(bf16_cfg, original, seq=2, tag=f"bf16_prefill_l{layer}"), np.concatenate([x, x]))
    exact = _bf16_decode_reference_fp64(bf16_cfg, original, x)

    cos_exact, rel_exact = _cosine(got, exact), rel_err(got, exact)
    rel_prefill = rel_err(got, prefill[:1])
    print(
        f"[uocr] bf16 decode MoE layer {layer}: vs exact bf16 math cosine {cos_exact:.9f} (rel {rel_exact:.2e}); "
        f"vs the prefill path rel {rel_prefill:.2e}"
    )
    assert np.all(np.isfinite(got))
    assert cos_exact >= 0.99999 and rel_exact <= 1e-4
    assert rel_prefill <= 1e-4


@pytest.mark.slow
def test_bf16_multi_row_qmv_matches_the_exact_math_and_the_prefill_path() -> None:
    """``MoE.decode_rows`` on 8 rows (``moe_bf16_qmv`` over 48 (token, expert) rows) vs float64, prefill and one-row decode.

    Each row against the float64 evaluation of the bf16 layer at the batch-1
    test's bars; against the prefill path on the same 8 rows (the dense chain
    on CPU, MAX's native routing on an accelerator) and against the one-row
    decode MoE to fp32 rounding (same kernel, but a different gate matmul
    and mixing shape).
    """
    bf16_cfg = _small_decoder_config(int8=False)
    original, _, _ = _moe_weights(bf16_cfg, seed=19)
    x = np.random.default_rng(23).standard_normal((8, bf16_cfg.hidden_size)).astype(np.float32)

    rows = _run(_moe_model(bf16_cfg, original, seq=8, tag="bf16_rows", decode_rows=True), x)
    prefill = _run(_moe_model(bf16_cfg, original, seq=8, tag="bf16_prefill_rows"), x)
    one_row = _moe_model(bf16_cfg, original, seq=1, tag="bf16_one_row")
    single = np.concatenate([_run(one_row, x[i : i + 1]) for i in range(x.shape[0])])
    exact = np.concatenate([_bf16_decode_reference_fp64(bf16_cfg, original, x[i : i + 1]) for i in range(x.shape[0])])

    rel_exact, rel_prefill, rel_single = rel_err(rows, exact), rel_err(rows, prefill), rel_err(rows, single)
    cos_exact = min(_cosine(rows[i], exact[i]) for i in range(x.shape[0]))
    print(
        f"[uocr] bf16 multi-row qmv (8 rows): vs exact bf16 math rel {rel_exact:.2e} (min cos {cos_exact:.9f}); "
        f"vs the prefill path rel {rel_prefill:.2e}; vs one-row qmv rel {rel_single:.2e}, "
        f"bitwise {bool(np.array_equal(rows, single))}"
    )
    assert np.all(np.isfinite(rows))
    assert cos_exact >= 0.99999 and rel_exact <= 1e-4
    assert rel_prefill <= 1e-4
    assert rel_single <= 1e-5
