"""Model-free checks on the weight adapters: expert stacking, int8 detection, declared-weight validation.

Synthetic checkpoints with tiny shapes and 4 experts stand in for the real
64-expert file; every expert tensor is filled with its own expert index so a
misordered stack cannot pass.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef, Weight

from unlimited_ocr_max.weight_adapters import (
    WeightMappingError,
    check_against_declared,
    is_int8_checkpoint,
    stack_expert_weights,
)

EXPERTS = 4
N, K = 6, 256  # K is two groups of 128
GROUPS = K // 128
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _expert_weight(index: int, dtype: torch.dtype) -> torch.Tensor:
    """An ``[N, K]`` tensor filled with ``index``, so slice ``j`` is identifiable."""
    return torch.full((N, K), index, dtype=dtype)


def _expert_scales(index: int) -> torch.Tensor:
    return torch.full((N, GROUPS), float(index), dtype=torch.float32)


def bf16_checkpoint(*, layers: tuple[int, ...] = (1,), experts: int = EXPERTS) -> dict[str, Any]:
    """The bf16 form: per-expert weights, no scales anywhere, plus one non-expert tensor."""
    out: dict[str, Any] = {"model.norm.weight": torch.ones(N, dtype=torch.bfloat16)}
    for layer in layers:
        for expert in range(experts):
            for proj in PROJECTIONS:
                stem = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                out[f"{stem}.weight"] = _expert_weight(expert, torch.bfloat16)
    return out


def int8_checkpoint(*, layers: tuple[int, ...] = (1,), experts: int = EXPERTS) -> dict[str, Any]:
    """The int8 form: int8 expert weights, each next to its fp32 group scales."""
    out: dict[str, Any] = {"model.norm.weight": torch.ones(N, dtype=torch.bfloat16)}
    for layer in layers:
        for expert in range(experts):
            for proj in PROJECTIONS:
                stem = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                out[f"{stem}.weight"] = _expert_weight(expert, torch.int8)
                out[f"{stem}.weight_scales"] = _expert_scales(expert)
    return out


def renamed(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """What ``language_state_dict`` hands to ``stack_expert_weights``: the ``model.`` prefix gone."""
    return {key.removeprefix("model."): value for key, value in checkpoint.items()}


# --- stacking -----------------------------------------------------------------


def test_bf16_stacking_puts_expert_j_at_slice_j() -> None:
    out = stack_expert_weights(renamed(bf16_checkpoint()), num_experts=EXPERTS)
    assert set(out) == {"norm.weight", *(f"layers.1.mlp.experts.{proj}" for proj in PROJECTIONS)}
    for proj in PROJECTIONS:
        stacked = out[f"layers.1.mlp.experts.{proj}"]
        assert stacked.shape == (EXPERTS, N, K)
        assert stacked.dtype == torch.bfloat16
        for expert in range(EXPERTS):
            assert torch.all(stacked[expert] == expert)


def test_int8_stacking_groups_the_scales_into_their_own_stack() -> None:
    out = stack_expert_weights(renamed(int8_checkpoint(layers=(1, 2))), num_experts=EXPERTS)
    expected = {"norm.weight"}
    for layer in (1, 2):
        for proj in PROJECTIONS:
            expected |= {f"layers.{layer}.mlp.experts.{proj}", f"layers.{layer}.mlp.experts.{proj}_scales"}
    assert set(out) == expected
    for layer in (1, 2):
        for proj in PROJECTIONS:
            weights = out[f"layers.{layer}.mlp.experts.{proj}"]
            scales = out[f"layers.{layer}.mlp.experts.{proj}_scales"]
            assert (weights.shape, weights.dtype) == ((EXPERTS, N, K), torch.int8)
            assert (scales.shape, scales.dtype) == ((EXPERTS, N, GROUPS), torch.float32)
            for expert in range(EXPERTS):
                assert torch.all(weights[expert] == expert)
                assert torch.all(scales[expert] == float(expert))


def test_stacking_rejects_a_gap_in_the_scales() -> None:
    state = renamed(int8_checkpoint())
    del state["layers.1.mlp.experts.2.up_proj.weight_scales"]
    with pytest.raises(WeightMappingError, match=r"up_proj_scales: expert indices must be 0\.\.n-1"):
        stack_expert_weights(state, num_experts=EXPERTS)


def test_stacking_rejects_a_short_scales_stack() -> None:
    state = renamed(int8_checkpoint())
    del state[f"layers.1.mlp.experts.{EXPERTS - 1}.down_proj.weight_scales"]
    with pytest.raises(WeightMappingError, match=rf"down_proj_scales: expected {EXPERTS} experts, got {EXPERTS - 1}"):
        stack_expert_weights(state, num_experts=EXPERTS)


def test_an_unknown_weight_suffix_is_never_half_matched() -> None:
    """``weight_scales`` is a suffix of its own, so a look-alike passes through untouched."""
    state = renamed(bf16_checkpoint())
    state["layers.1.mlp.experts.0.up_proj.weight_zeros"] = torch.zeros(N, GROUPS)
    state["layers.1.mlp.shared_experts.up_proj.weight"] = torch.zeros(N, K)
    out = stack_expert_weights(state, num_experts=EXPERTS)
    assert "layers.1.mlp.experts.0.up_proj.weight_zeros" in out
    assert "layers.1.mlp.shared_experts.up_proj.weight" in out
    assert "layers.1.mlp.experts.up_proj_zeros" not in out


# --- detection ----------------------------------------------------------------


def test_a_bf16_checkpoint_is_not_int8() -> None:
    assert is_int8_checkpoint(bf16_checkpoint()) is False
    assert is_int8_checkpoint(renamed(bf16_checkpoint())) is False


def test_an_int8_checkpoint_is_int8() -> None:
    assert is_int8_checkpoint(int8_checkpoint(layers=(1, 2))) is True
    assert is_int8_checkpoint(renamed(int8_checkpoint())) is True


def test_a_checkpoint_without_experts_at_all_is_not_int8() -> None:
    assert is_int8_checkpoint({"model.norm.weight": torch.ones(N, dtype=torch.bfloat16)}) is False


def test_int8_weights_without_scales_are_rejected() -> None:
    checkpoint = {key: value for key, value in int8_checkpoint().items() if not key.endswith("_scales")}
    with pytest.raises(WeightMappingError, match="int8 expert weights without weight_scales"):
        is_int8_checkpoint(checkpoint)


def test_scales_for_some_experts_but_not_all_are_rejected() -> None:
    checkpoint = int8_checkpoint()
    del checkpoint["model.layers.1.mlp.experts.3.gate_proj.weight_scales"]
    with pytest.raises(WeightMappingError, match="half-quantized checkpoint: no weight_scales for"):
        is_int8_checkpoint(checkpoint)


def test_scales_for_some_projections_but_not_all_are_rejected() -> None:
    checkpoint = int8_checkpoint()
    for expert in range(EXPERTS):
        del checkpoint[f"model.layers.1.mlp.experts.{expert}.down_proj.weight_scales"]
    with pytest.raises(WeightMappingError, match="half-quantized checkpoint: no weight_scales for"):
        is_int8_checkpoint(checkpoint)


def test_scales_next_to_bf16_weights_are_rejected() -> None:
    """Scales without int8 weights: the same keys, but the weights were never quantized."""
    checkpoint = int8_checkpoint()
    for key, value in list(checkpoint.items()):
        if key.endswith(".weight") and ".experts." in key:
            checkpoint[key] = value.to(torch.bfloat16)
    with pytest.raises(WeightMappingError, match=r"expert weights are \['bfloat16'\], not int8"):
        is_int8_checkpoint(checkpoint)


def test_scales_with_no_weight_to_scale_are_rejected() -> None:
    checkpoint = int8_checkpoint()
    del checkpoint["model.layers.1.mlp.experts.0.gate_proj.weight"]
    with pytest.raises(WeightMappingError, match="weight_scales without an expert weight to scale"):
        is_int8_checkpoint(checkpoint)


# --- declared-weight validation -----------------------------------------------


def _declared(dtype: DType) -> dict[str, Weight]:
    """A hand-built stand-in for ``UnlimitedOcrDecoder.raw_state_dict()``: one expert stack, bf16 or int8."""
    device = DeviceRef.CPU()
    return {"layers.1.mlp.experts.gate_proj": Weight("gate_proj", dtype, [EXPERTS, N, K], device=device)}


def test_declared_weights_accept_the_matching_dtype() -> None:
    check_against_declared(
        {"layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, K), dtype=torch.bfloat16)},
        _declared(DType.bfloat16),
    )
    check_against_declared(
        {"layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, K), dtype=torch.int8)},
        _declared(DType.int8),
    )


def test_int8_weights_against_a_bf16_declaration_fail_on_the_dtype() -> None:
    provided = {"layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, K), dtype=torch.int8)}
    with pytest.raises(WeightMappingError, match="checkpoint dtype int8 but the graph declares bfloat16"):
        check_against_declared(provided, _declared(DType.bfloat16))


def test_bf16_weights_against_an_int8_declaration_fail_on_the_dtype() -> None:
    provided = {"layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, K), dtype=torch.bfloat16)}
    with pytest.raises(WeightMappingError, match="checkpoint dtype bfloat16 but the graph declares int8"):
        check_against_declared(provided, _declared(DType.int8))


def test_declared_weights_still_check_names_and_shapes() -> None:
    with pytest.raises(WeightMappingError, match=r"missing \['layers.1.mlp.experts.gate_proj'\]"):
        check_against_declared({}, _declared(DType.bfloat16))
    extra = {
        "layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, K), dtype=torch.bfloat16),
        "layers.1.mlp.experts.gate_proj_scales": torch.zeros((EXPERTS, N, GROUPS), dtype=torch.float32),
    }
    with pytest.raises(WeightMappingError, match=r"unexpected \['layers.1.mlp.experts.gate_proj_scales'\]"):
        check_against_declared(extra, _declared(DType.bfloat16))
    wrong_shape = {"layers.1.mlp.experts.gate_proj": torch.zeros((EXPERTS, N, GROUPS), dtype=torch.bfloat16)}
    with pytest.raises(WeightMappingError, match="checkpoint shape"):
        check_against_declared(wrong_shape, _declared(DType.bfloat16))
