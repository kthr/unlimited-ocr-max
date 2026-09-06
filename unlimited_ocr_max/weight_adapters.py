"""Checkpoint tensors -> the two sub-models' MAX state dicts.

``language_model``: strip ``model.``, rename ``mlp.gate.weight`` to
``mlp.gate.gate_score.weight``, and stack each MoE layer's 64 routed experts into
three ``[64, N, K]`` tensors in ascending expert index (the layout
``grouped_matmul_ragged`` wants; slice ``j`` must be expert ``j``). Storage stays
bfloat16. The result is checked name-for-name and shape-for-shape against what
:class:`~unlimited_ocr_max.decoder.UnlimitedOcrDecoder` declares.

``vision``: strip ``model.``, upcast to fp32, resample SAM's position tables for
the resolution and transpose its conv filters to RSCF; the dead CLIP patch conv
is dropped by name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .decoder import UnlimitedOcrDecoder
from .layers.clip_l import UNUSED_CHECKPOINT_WEIGHTS
from .layers.sam_vit import CHECKPOINT_PREFIX as SAM_PREFIX
from .layers.sam_vit import as_float32, sam_state_dict
from .model_config import UnlimitedOCRConfig

__all__ = [
    "LANGUAGE_MODEL",
    "VISION",
    "WeightMappingError",
    "language_state_dict",
    "load_checkpoint",
    "stack_expert_weights",
    "vision_state_dict",
]


VISION = "vision"
LANGUAGE_MODEL = "language_model"

#: ``lm_head.weight`` is the one decoder tensor without the ``model.`` prefix.
LANGUAGE_PREFIXES = ("model.embed_tokens.", "model.layers.", "model.norm.")
VISION_PREFIXES = (
    "model.sam_model.",
    "model.vision_model.",
    "model.projector.",
    "model.image_newline",
    "model.view_seperator",
)

#: A post-rename per-expert weight; anchored so ``shared_experts`` never matches.
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<stem>layers\.(?P<layer>\d+)\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


class WeightMappingError(RuntimeError):
    """The checkpoint and the graph's declared weights do not line up exactly."""


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Every tensor of a safetensors shard as bfloat16 torch tensors (numpy has no bf16)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118


def language_weight_name(checkpoint_name: str) -> str:
    name = checkpoint_name.removeprefix("model.")
    if name.endswith(".mlp.gate.weight"):
        name = name[: -len(".weight")] + ".gate_score.weight"
    return name


def stack_expert_weights(state_dict: Mapping[str, Any], *, num_experts: int) -> dict[str, Any]:
    """Collapse ``layers.{i}.mlp.experts.{j}.<proj>.weight`` (torch tensors) into ``layers.{i}.mlp.experts.<proj>`` stacks."""
    import torch

    out: dict[str, Any] = {}
    grouped: dict[str, dict[int, Any]] = {}
    for name, value in state_dict.items():
        match = _EXPERT_WEIGHT_RE.match(name)
        if match is None:
            out[name] = value
            continue
        grouped.setdefault(f"{match['stem']}.{match['proj']}", {})[int(match["expert"])] = value
    for stacked, members in grouped.items():
        indices = sorted(members)
        if indices != list(range(len(indices))):
            raise WeightMappingError(f"{stacked}: expert indices must be 0..n-1 with no gaps, got {indices}")
        if len(indices) != num_experts:
            raise WeightMappingError(f"{stacked}: expected {num_experts} experts, got {len(indices)}")
        out[stacked] = torch.stack([members[j] for j in indices], dim=0)
    return out


def language_state_dict(checkpoint: Mapping[str, Any], config: UnlimitedOCRConfig) -> dict[str, Any]:
    """The decoder's tensors, renamed and expert-stacked, verified against the declared weights."""
    selected = {
        key: value
        for key, value in checkpoint.items()
        if key == "lm_head.weight" or key.startswith(LANGUAGE_PREFIXES)
    }
    if not selected:
        raise WeightMappingError("no language weights found in the checkpoint")
    renamed = {language_weight_name(key): value for key, value in selected.items()}
    out = stack_expert_weights(renamed, num_experts=config.decoder.n_routed_experts)

    declared = UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict()
    missing = sorted(set(declared) - set(out))
    extra = sorted(set(out) - set(declared))
    if missing or extra:
        raise WeightMappingError(
            f"language weights disagree with the checkpoint: missing {missing[:20]}, unexpected {extra[:20]}"
        )
    for name, weight in declared.items():
        want = tuple(int(d) for d in weight.shape.static_dims)
        got = tuple(int(d) for d in out[name].shape)
        if want != got:
            raise WeightMappingError(f"{name}: checkpoint shape {got} but the graph declares {want}")
    return out


def vision_state_dict(checkpoint: Mapping[str, Any], *, image_size: int) -> dict[str, np.ndarray]:
    """The vision tower, projector and layout tensors as fp32 numpy, at ``image_size``."""
    unused = {f"vision_model.{name}" for name in UNUSED_CHECKPOINT_WEIGHTS}
    out: dict[str, np.ndarray] = {
        f"sam_model.{key}": value for key, value in sam_state_dict(checkpoint, image_size, prefix=SAM_PREFIX).items()
    }
    for key, value in checkpoint.items():
        if not key.startswith(VISION_PREFIXES) or key.startswith(SAM_PREFIX):
            continue
        name = key.removeprefix("model.")
        if name in unused:
            continue
        out[name] = as_float32(value)
    if not out:
        raise WeightMappingError("no vision weights found in the checkpoint")
    return out
