"""The decoder's int8 expert mode, model-free.

Staging checks run weightless: every ``Weight`` gets its FQN the way
``load_state_dict`` would assign it, no tensor data is resident, and the graph
is only built (``str(graph)``), never compiled. That is enough to pin the
declared names/dtypes/shapes against the adapter's stacking, to see both
language graphs stage the Mojo ops, to count staged ops, and to hit the CPU
refusal.

The numeric checks (marked ``slow``: they compile the Mojo kernels through an
``InferenceSession``) run a bare :class:`~unlimited_ocr_max.decoder.MoE` on
synthetic weights. They run on CPU by default (``UOCR_TEST_DEVICE=gpu`` moves
them to the accelerator): the kernels are device-agnostic and the CPU bf16
decode path is bitwise the dense 64-term chain, so CPU exercises exactly the
graph ops the served GPU path stages, minus the device. The int8-on-CPU refusal
lives in the graph builders, which these checks bypass on purpose by opening
their own ``Graph``.
"""

from __future__ import annotations

import functools
import os
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from unlimited_ocr_max.decoder import MoE, UnlimitedOcrDecoder
from unlimited_ocr_max.graphs import build_decode_graph, build_language_graph, check_int8_device
from unlimited_ocr_max.model_config import INT8_GROUP_SIZE, ConfigError, DecoderConfig, UnlimitedOCRConfig
from unlimited_ocr_max.ngram import MOJO_KERNELS
from unlimited_ocr_max.weight_adapters import check_against_declared, stack_expert_weights

GPU = os.environ.get("UOCR_TEST_DEVICE", "cpu").strip().lower() == "gpu"
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

#: One MLIR result line; the pinned nightly emits the generic quoted form ``%1 = "rmo.add"(...)``.
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


def _decode_graph_text(config: UnlimitedOCRConfig, device: DeviceRef) -> str:
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=device))
    return str(build_decode_graph(config, decoder, max_seq_len=64, device=device).graph)


def _prefill_graph_text(config: UnlimitedOCRConfig, device: DeviceRef, *, seq_len: int = 5) -> str:
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=device))
    return str(build_language_graph(config, decoder, seq_len=seq_len, n_image_tokens=2, device=device).graph)


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
            renamed[f"{stem}.weight"] = torch.zeros((n, k), dtype=torch.int8)
            renamed[f"{stem}.weight_scales"] = torch.ones((n, k // INT8_GROUP_SIZE), dtype=torch.float32)
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

    decode = _op_counts(_decode_graph_text(config, dref))
    assert decode["mo.custom"] == 3 * n_moe  # gate, up, down qmv per MoE layer
    prefill = _op_counts(_prefill_graph_text(config, dref))
    assert prefill["mo.custom"] == 3 * dec.n_routed_experts * n_moe  # one dequant per expert per projection

    for text in (_decode_graph_text(config, dref), _prefill_graph_text(config, dref)):
        assert "moe_int8_qmv" in text or "int8_dequant_expert" in text
        assert "mo.grouped.matmul.ragged" not in text and "mo.moe.create.indices" not in text
        assert "layers.1.mlp.experts.gate_proj_scales" in text


def test_int8_on_cpu_is_refused_at_graph_construction() -> None:
    config = _config(int8=True)
    cpu = DeviceRef.CPU()
    # The decoder itself constructs on CPU: the adapter builds one to check the checkpoint's names.
    decoder = _named(UnlimitedOcrDecoder(config.decoder, dtype=config.dtype, device=cpu))
    with pytest.raises(ValueError, match="GPU-only"):
        build_decode_graph(config, decoder, max_seq_len=64, device=cpu)
    with pytest.raises(ValueError, match="GPU-only"):
        build_language_graph(config, decoder, seq_len=5, n_image_tokens=2, device=cpu)
    with pytest.raises(ValueError, match="GPU-only"):
        check_int8_device(config.decoder, cpu)
    check_int8_device(config.decoder, DeviceRef.GPU(0))
    check_int8_device(_config(int8=False).decoder, cpu)


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
# op census: staged ops per decode MoE layer, int8 vs bf16 (accelerator graphs)
# --------------------------------------------------------------------------


def test_decode_op_census_per_moe_layer_is_level_with_bf16() -> None:
    """Per MoE layer, int8 decode stages within 5% of bf16's ops (the served step is host-encode-bound).

    The per-layer figure is the difference between a 3-layer and a 2-layer
    decoder (layer 0 is dense in both), so everything outside the MoE layer
    cancels exactly.
    """
    dref = DeviceRef.GPU(0)
    per_layer: dict[str, dict[str, int]] = {}
    for mode, int8 in (("bf16", False), ("int8", True)):
        two, three = (_op_counts(_decode_graph_text(_config(int8=int8, num_hidden_layers=n), dref)) for n in (2, 3))
        per_layer[mode] = {"all": sum(three.values()) - sum(two.values()), "staged": _staged(three) - _staged(two)}
        print(f"[uocr] decode ops per MoE layer, {mode}: {per_layer[mode]}")
    for key in ("all", "staged"):
        bf16, int8 = per_layer["bf16"][key], per_layer["int8"][key]
        assert bf16 > 0
        assert abs(int8 - bf16) <= 0.05 * bf16, f"{key}: int8 {int8} vs bf16 {bf16}"
    # The int8 layer is three kernel calls and no permute/restore gathers.
    assert per_layer["int8"]["staged"] <= per_layer["bf16"]["staged"]


# --------------------------------------------------------------------------
# numeric: a bare MoE layer, int8 decode vs bf16 (slow: compiles the kernels)
# --------------------------------------------------------------------------


@functools.cache
def _driver():
    return Accelerator() if GPU else CPU()


@functools.cache
def _device_ref() -> DeviceRef:
    return DeviceRef.GPU() if GPU else DeviceRef.CPU()


@functools.cache
def _session() -> InferenceSession:
    return InferenceSession(devices=[_driver()])


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


def _moe_model(config: DecoderConfig, weights: dict[str, Any], *, seq: int, tag: str):
    dref = _device_ref()
    moe = MoE(config, dtype=DType.bfloat16, device=dref)
    moe.load_state_dict(weights)
    extensions = {"custom_extensions": [MOJO_KERNELS]} if config.int8_experts else {}
    with Graph(
        f"test_moe_{tag}_{seq}", input_types=[TensorType(DType.float32, [seq, config.hidden_size], device=dref)], **extensions
    ) as graph:
        graph.output(moe(graph.inputs[0].tensor))
    return _session().load(graph, weights_registry=moe.state_dict())


def _run(model, x: np.ndarray) -> np.ndarray:
    return model.execute(Buffer.from_numpy(np.ascontiguousarray(x)).to(_driver()))[0].to(CPU()).to_numpy()


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
    xf = x.astype(np.float64)[0]
    f64 = lambda t: t.float().numpy().astype(np.float64)  # noqa: E731
    silu = lambda a: a / (1.0 + np.exp(-a))  # noqa: E731
    logits = f64(int8["gate.gate_score.weight"]) @ xf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(-probs, kind="stable")[: config.num_experts_per_tok]

    def dequantized(proj: str) -> np.ndarray:
        q, scales = f64(int8[f"experts.{proj}"]), f64(int8[f"experts.{proj}_scales"])
        return q * np.repeat(scales, INT8_GROUP_SIZE, axis=2)

    gate, up, down = (dequantized(proj) for proj in PROJECTIONS)
    routed = sum(probs[e] * (down[e] @ (silu(gate[e] @ xf) * (up[e] @ xf))) for e in top)
    shared_gate, shared_up, shared_down = (f64(int8[f"shared_experts.{proj}.weight"]) for proj in PROJECTIONS)
    shared = shared_down @ (silu(shared_gate @ xf) * (shared_up @ xf))
    return (routed + shared)[None, :]


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
