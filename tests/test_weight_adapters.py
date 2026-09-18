"""Model-free checks on the weight adapters: expert stacking, int8 detection, declared-weight validation.

Synthetic checkpoints with tiny shapes and 4 experts stand in for the real
64-expert file; every expert tensor is filled with its own expert index so a
misordered stack cannot pass.

The adapter's currency is :class:`~max.driver.Buffer`, so the fixtures build
``Buffer``s. torch stays on the *building* side only -- it is the independent
reference for bf16 bit patterns, which is exactly why the values are read back
through it rather than through the package's own conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from max.driver import Buffer
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


def _buffer(tensor: torch.Tensor) -> Buffer:
    """A checkpoint entry as the adapter meets it. The ``Buffer`` keeps ``tensor`` alive through DLPack."""
    return Buffer.from_dlpack(tensor)


def _expert_weight(index: int, dtype: torch.dtype) -> Buffer:
    """An ``[N, K]`` buffer filled with ``index``, so slice ``j`` is identifiable."""
    return _buffer(torch.full((N, K), index, dtype=dtype))


def _expert_scales(index: int) -> Buffer:
    return _buffer(torch.full((N, GROUPS), float(index), dtype=torch.float32))


def bf16_checkpoint(*, layers: tuple[int, ...] = (1,), experts: int = EXPERTS) -> dict[str, Any]:
    """The bf16 form: per-expert weights, no scales anywhere, plus one non-expert tensor."""
    out: dict[str, Any] = {"model.norm.weight": _buffer(torch.ones(N, dtype=torch.bfloat16))}
    for layer in layers:
        for expert in range(experts):
            for proj in PROJECTIONS:
                stem = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                out[f"{stem}.weight"] = _expert_weight(expert, torch.bfloat16)
    return out


def int8_checkpoint(*, layers: tuple[int, ...] = (1,), experts: int = EXPERTS) -> dict[str, Any]:
    """The int8 form: int8 expert weights, each next to its fp32 group scales."""
    out: dict[str, Any] = {"model.norm.weight": _buffer(torch.ones(N, dtype=torch.bfloat16))}
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
        assert isinstance(stacked, Buffer)
        assert (tuple(stacked.shape), stacked.dtype) == ((EXPERTS, N, K), DType.bfloat16)
        # Values, not just the count: a stack whose slices are in the wrong
        # order has the right shape and the wrong model.
        values = torch.from_dlpack(stacked)
        assert values.dtype == torch.bfloat16
        for expert in range(EXPERTS):
            assert torch.all(values[expert] == expert)


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
            assert (tuple(weights.shape), weights.dtype) == ((EXPERTS, N, K), DType.int8)
            assert (tuple(scales.shape), scales.dtype) == ((EXPERTS, N, GROUPS), DType.float32)
            weight_values, scale_values = torch.from_dlpack(weights), torch.from_dlpack(scales)
            for expert in range(EXPERTS):
                assert torch.all(weight_values[expert] == expert)
                assert torch.all(scale_values[expert] == float(expert))


def test_stacking_refuses_experts_that_disagree_on_dtype() -> None:
    """np.stack would promote them silently, and the re-view would then read the wrong bytes."""
    state = renamed(bf16_checkpoint())
    state["layers.1.mlp.experts.2.up_proj.weight"] = _expert_weight(2, torch.int8)
    with pytest.raises(WeightMappingError, match=r"up_proj: the experts disagree on dtype: \['bfloat16', 'int8'\]"):
        stack_expert_weights(state, num_experts=EXPERTS)


def test_stacking_refuses_experts_that_disagree_on_shape_and_names_the_stack() -> None:
    """np.stack rejects these on its own, but says only ``all input arrays must have the same shape``."""
    state = renamed(bf16_checkpoint())
    state["layers.1.mlp.experts.2.up_proj.weight"] = _buffer(torch.zeros((N, K // 2), dtype=torch.bfloat16))
    with pytest.raises(WeightMappingError, match=rf"up_proj: the experts disagree on shape: \[\({N}, {K // 2}\), \({N}, {K}\)\]"):
        stack_expert_weights(state, num_experts=EXPERTS)


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


@pytest.mark.parametrize(
    "name",
    [
        "layers.1.mlp.experts.0.up_proj.weight_zeros",
        "layers.1.mlp.experts.0.up_proj.weight_scale_inv",
        "layers.1.mlp.experts.0.up_proj.weight_scales2",
        "layers.1.mlp.experts.0.up_proj.weight_SCALES",
        "layers.1.self_attn.q_proj.weight_scales",
    ],
)
def test_an_unknown_weight_suffix_is_never_half_matched(name: str) -> None:
    """``weight_scales`` is a suffix of its own, so a look-alike passes through untouched."""
    state = renamed(bf16_checkpoint())
    state[name] = _buffer(torch.zeros(N, GROUPS))
    state["layers.1.mlp.shared_experts.up_proj.weight"] = _buffer(torch.zeros(N, K))
    out = stack_expert_weights(state, num_experts=EXPERTS)
    assert name in out
    assert "layers.1.mlp.shared_experts.up_proj.weight" in out
    assert not any(
        key.startswith("layers.1.mlp.experts.up_proj_") for key in out
    ), "a look-alike must never be stacked"


# --- detection ----------------------------------------------------------------


def test_a_bf16_checkpoint_is_not_int8() -> None:
    assert is_int8_checkpoint(bf16_checkpoint()) is False
    assert is_int8_checkpoint(renamed(bf16_checkpoint())) is False


def test_an_int8_checkpoint_is_int8() -> None:
    assert is_int8_checkpoint(int8_checkpoint(layers=(1, 2))) is True
    assert is_int8_checkpoint(renamed(int8_checkpoint())) is True


def test_a_checkpoint_without_experts_at_all_is_not_int8() -> None:
    assert is_int8_checkpoint({"model.norm.weight": _buffer(torch.ones(N, dtype=torch.bfloat16))}) is False


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
    checkpoint.update(bf16_checkpoint())  # the same weight keys unquantized; the scales keys stay
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


def _stack(dtype: torch.dtype, shape: tuple[int, ...] = (EXPERTS, N, K)) -> Buffer:
    """A stand-in stack as the adapter produces one: a ``Buffer`` that has to say its own dtype and shape."""
    return _buffer(torch.zeros(shape, dtype=dtype))


def test_declared_weights_accept_the_matching_dtype() -> None:
    check_against_declared({"layers.1.mlp.experts.gate_proj": _stack(torch.bfloat16)}, _declared(DType.bfloat16))
    check_against_declared({"layers.1.mlp.experts.gate_proj": _stack(torch.int8)}, _declared(DType.int8))


def test_int8_weights_against_a_bf16_declaration_fail_on_the_dtype() -> None:
    provided = {"layers.1.mlp.experts.gate_proj": _stack(torch.int8)}
    with pytest.raises(WeightMappingError, match="checkpoint dtype int8 but the graph declares bfloat16"):
        check_against_declared(provided, _declared(DType.bfloat16))


def test_bf16_weights_against_an_int8_declaration_fail_on_the_dtype() -> None:
    provided = {"layers.1.mlp.experts.gate_proj": _stack(torch.bfloat16)}
    with pytest.raises(WeightMappingError, match="checkpoint dtype bfloat16 but the graph declares int8"):
        check_against_declared(provided, _declared(DType.int8))


def test_declared_weights_still_check_names_and_shapes() -> None:
    with pytest.raises(WeightMappingError, match=r"missing \['layers.1.mlp.experts.gate_proj'\]"):
        check_against_declared({}, _declared(DType.bfloat16))
    extra = {
        "layers.1.mlp.experts.gate_proj": _stack(torch.bfloat16),
        "layers.1.mlp.experts.gate_proj_scales": _stack(torch.float32, (EXPERTS, N, GROUPS)),
    }
    with pytest.raises(WeightMappingError, match=r"unexpected \['layers.1.mlp.experts.gate_proj_scales'\]"):
        check_against_declared(extra, _declared(DType.bfloat16))
    wrong_shape = {"layers.1.mlp.experts.gate_proj": _stack(torch.bfloat16, (EXPERTS, N, GROUPS))}
    with pytest.raises(WeightMappingError, match="checkpoint shape"):
        check_against_declared(wrong_shape, _declared(DType.bfloat16))


def test_language_state_dict_rejects_a_half_quantized_file_at_the_detector() -> None:
    """The detector is wired into ``language_state_dict`` before anything needs the config,
    so a half-quantized file fails with the message that names the offending tensors
    rather than the downstream "language weights disagree" one."""
    from unlimited_ocr_max.weight_adapters import language_state_dict

    checkpoint = bf16_checkpoint()
    stem = "model.layers.1.mlp.experts.0.gate_proj"
    checkpoint[f"{stem}.weight"] = _expert_weight(0, torch.int8)  # int8 weight, no scales
    with pytest.raises(WeightMappingError, match="weight_scales"):
        language_state_dict(checkpoint, config=None)  # type: ignore[arg-type]  # never reached


# --- vision state dict -------------------------------------------------------


def _table(*shape: int) -> Buffer:
    """A SAM position table as the checkpoint stores it: bf16, which only a ``Buffer`` can name."""
    return _buffer(torch.ones(shape, dtype=torch.bfloat16))


def test_as_float32_widens_a_bf16_buffer_and_leaves_a_numpy_array_alone() -> None:
    """The two inputs the vision side ever sees, and the values torch agrees they hold."""
    from unlimited_ocr_max.layers.sam_vit import as_float32

    tensor = torch.tensor([[1.5, -2.25, 0.0], [256.0, 3.5, -0.125]], dtype=torch.bfloat16)
    widened = as_float32(_buffer(tensor))
    assert widened.dtype == np.float32
    assert np.array_equal(widened, tensor.float().numpy())

    # fp32 already: a Buffer passes through, and so does a plain numpy array.
    fp32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(as_float32(_buffer(torch.from_numpy(fp32))), fp32)
    assert np.array_equal(as_float32(fp32), fp32)
    # A numpy array of another dtype is cast, as it always was.
    assert as_float32(np.arange(6, dtype=np.int64)).dtype == np.float32


def test_sam_state_dict_rejects_mismatched_pos_embed_grid() -> None:
    """When pos_embed grid does not match the target image_size, raise with clear message."""
    from unlimited_ocr_max.layers.sam_vit import sam_state_dict

    # Create a checkpoint with pos_embed at 64x64 grid (1024px image, 16px patches)
    checkpoint = {"model.sam_model.pos_embed": _table(1, 64, 64, 768)}

    # Try to load at 512px (32x32 grid) — this should raise
    with pytest.raises(ValueError, match=r"pos_embed: found shape .+, need shape \(1, 32, 32, 768\)"):
        sam_state_dict(checkpoint, image_size=512)


def test_sam_state_dict_rejects_mismatched_rel_pos() -> None:
    """When rel_pos table does not match the target grid, raise with clear message."""
    from unlimited_ocr_max.layers.sam_vit import sam_state_dict

    # Create a checkpoint with rel_pos_h at global block with 64x64 grid (127=2*64-1)
    checkpoint = {
        "model.sam_model.blocks.2.attn.rel_pos_h": _table(127, 64),
        "model.sam_model.blocks.2.attn.rel_pos_w": _table(127, 64),
    }

    # Try to load at 512px (32x32 grid) — target for global block should be 63=2*32-1
    with pytest.raises(ValueError, match=r"blocks\.2\.attn\.rel_pos_h: found shape .+, need shape \(63, 64\)"):
        sam_state_dict(checkpoint, image_size=512)


def test_sam_state_dict_passes_through_matching_pos_embed_and_rel_pos() -> None:
    """When pos_embed and rel_pos tables already match the target grid, they pass through unchanged."""
    from unlimited_ocr_max.layers.sam_vit import sam_state_dict

    # Create a checkpoint with correctly sized tables for 1024px (64x64 grid)
    image_size = 1024
    grid = image_size // 16  # 64
    checkpoint = {
        "model.sam_model.pos_embed": _table(1, grid, grid, 768),
        # Global blocks (indices 2, 5, 8, 11) need (2*64-1, 64) = (127, 64)
        "model.sam_model.blocks.2.attn.rel_pos_h": _table(127, 64),
        "model.sam_model.blocks.2.attn.rel_pos_w": _table(127, 64),
        # Windowed blocks need (2*14-1, 64) = (27, 64)
        "model.sam_model.blocks.0.attn.rel_pos_h": _table(27, 64),
        "model.sam_model.blocks.0.attn.rel_pos_w": _table(27, 64),
    }

    result = sam_state_dict(checkpoint, image_size=image_size)

    # Verify shapes and values survive the bf16 -> fp32 widening (sam_state_dict returns numpy arrays)
    assert result["pos_embed"].shape == (1, grid, grid, 768)
    assert result["pos_embed"].dtype == np.float32
    assert np.all(result["pos_embed"] == 1.0)
    assert result["blocks.2.attn.rel_pos_h"].shape == (127, 64)
    assert result["blocks.2.attn.rel_pos_h"].dtype == np.float32
    assert result["blocks.0.attn.rel_pos_h"].shape == (27, 64)
    assert result["blocks.0.attn.rel_pos_h"].dtype == np.float32
