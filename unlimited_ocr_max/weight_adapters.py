"""Checkpoint tensors -> the two sub-models' MAX state dicts.

``language_model``: strip ``model.``, rename ``mlp.gate.weight`` to
``mlp.gate.gate_score.weight``, and stack each MoE layer's 64 routed experts into
three ``[64, N, K]`` tensors in ascending expert index (the layout
``grouped_matmul_ragged`` wants; slice ``j`` must be expert ``j``). The result is
checked name-for-name, shape-for-shape and dtype-for-dtype against what
:class:`~unlimited_ocr_max.decoder.UnlimitedOcrDecoder` declares.

Two checkpoint forms are read, told apart by :func:`is_int8_checkpoint`:

* bfloat16 -- ``model.layers.{L}.mlp.experts.{E}.{proj}.weight`` bf16 ``[N, K]``
  and no scales keys anywhere. Storage stays bfloat16.
* int8 -- the same expert weights as int8 ``[N, K]``, each next to a
  ``…{proj}.weight_scales`` fp32 ``[N, K / 128]`` (128 is the group *size*, so
  the last dim counts groups). They stack into
  ``layers.{L}.mlp.experts.{proj}`` int8 ``[64, N, K]`` and
  ``layers.{L}.mlp.experts.{proj}_scales`` fp32 ``[64, N, K / 128]``, both in
  ascending expert index. Everything outside the routed experts stays
  bf16/fp32 as in the bf16 form.

A half-quantized file -- scales for some experts or projections but not all,
int8 expert weights with no scales, or scales next to non-int8 weights -- is a
:class:`WeightMappingError`, never a guess.

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
    "check_against_declared",
    "is_int8_checkpoint",
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

#: A post-rename per-expert weight, or the group scales of an int8 one. Anchored
#: at both ends so ``shared_experts`` never matches and ``_scales`` is a suffix
#: in its own right -- ``…weight_zeros`` is not ``weight`` plus junk, it is no
#: match at all.
_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<stem>layers\.(?P<layer>\d+)\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight(?P<scales>_scales)?$"
)

#: The dtype an int8 checkpoint's routed-expert weights are stored in.
_INT8 = "int8"

#: One routed expert's projection: ``(layer, expert, proj)``.
_ExpertKey = tuple[int, int, str]


class WeightMappingError(RuntimeError):
    """The checkpoint and the graph's declared weights do not line up exactly."""


def _dtype_name(tensor: Any) -> str:
    """A tensor's or a :class:`~max.graph.Weight`'s dtype as a bare name (``bfloat16``).

    ``DType`` and numpy dtypes carry ``.name``; a ``torch.dtype`` only stringifies
    (``torch.bfloat16``), so the three become directly comparable.
    """
    dtype = tensor.dtype
    return str(getattr(dtype, "name", None) or dtype).removeprefix("torch.")


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
    """Collapse the per-expert tensors (torch tensors) of every MoE layer into stacks.

    ``layers.{i}.mlp.experts.{j}.<proj>.weight`` becomes
    ``layers.{i}.mlp.experts.<proj>`` and, on an int8 checkpoint,
    ``…{j}.<proj>.weight_scales`` becomes ``layers.{i}.mlp.experts.<proj>_scales``.
    Each stack is built in ascending expert index and gets the same completeness
    check: indices 0..n-1 with no gaps, exactly ``num_experts`` of them.
    """
    import torch

    out: dict[str, Any] = {}
    grouped: dict[str, dict[int, Any]] = {}
    for name, value in state_dict.items():
        match = _EXPERT_WEIGHT_RE.match(name)
        if match is None:
            out[name] = value
            continue
        stacked = f"{match['stem']}.{match['proj']}{match['scales'] or ''}"
        grouped.setdefault(stacked, {})[int(match["expert"])] = value
    for stacked, members in grouped.items():
        indices = sorted(members)
        if indices != list(range(len(indices))):
            raise WeightMappingError(f"{stacked}: expert indices must be 0..n-1 with no gaps, got {indices}")
        if len(indices) != num_experts:
            raise WeightMappingError(f"{stacked}: expected {num_experts} experts, got {len(indices)}")
        out[stacked] = torch.stack([members[j] for j in indices], dim=0)
    return out


def _expert_tensors(checkpoint: Mapping[str, Any]) -> tuple[dict[_ExpertKey, Any], dict[_ExpertKey, Any]]:
    """``(weights, scales)`` keyed by ``(layer, expert, proj)``, from checkpoint or post-rename names."""
    weights: dict[_ExpertKey, Any] = {}
    scales: dict[_ExpertKey, Any] = {}
    for name, value in checkpoint.items():
        match = _EXPERT_WEIGHT_RE.match(name.removeprefix("model."))
        if match is None:
            continue
        key = (int(match["layer"]), int(match["expert"]), match["proj"])
        (scales if match["scales"] else weights)[key] = value
    return weights, scales


def _expert_names(keys: set[_ExpertKey], *, suffix: str = "") -> list[str]:
    """The first 20 of ``keys`` back as checkpoint names, for an error message."""
    return [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight{suffix}"
        for layer, expert, proj in sorted(keys)[:20]
    ]


def is_int8_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    """Whether the routed experts are int8 with per-group ``weight_scales``.

    ``False`` for the bfloat16 checkpoint, which carries no scales key at all.
    A half-quantized file raises :class:`WeightMappingError` rather than being
    read as one form or the other: scales for some experts or projections but
    not all, int8 expert weights with no scales, or scales sitting next to
    weights that are not int8.
    """
    weights, scales = _expert_tensors(checkpoint)
    quantized = {key for key, value in weights.items() if _dtype_name(value) == _INT8}
    if not scales:
        if quantized:
            raise WeightMappingError(
                f"int8 expert weights without weight_scales: {_expert_names(quantized)}"
            )
        return False
    orphans = set(scales) - set(weights)
    if orphans:
        raise WeightMappingError(
            f"weight_scales without an expert weight to scale: {_expert_names(orphans, suffix='_scales')}"
        )
    unquantized = set(weights) - quantized
    if unquantized:
        found = sorted({_dtype_name(weights[key]) for key in unquantized})
        raise WeightMappingError(
            f"weight_scales are present but these expert weights are {found}, not int8: "
            f"{_expert_names(unquantized)}"
        )
    bare = set(weights) - set(scales)
    if bare:
        raise WeightMappingError(f"half-quantized checkpoint: no weight_scales for {_expert_names(bare)}")
    return True


def check_against_declared(state_dict: Mapping[str, Any], declared: Mapping[str, Any]) -> None:
    """Check ``state_dict`` name-for-name, shape-for-shape and dtype-for-dtype against ``declared``.

    ``declared`` is the ``{name: Weight}`` mapping a ``max.nn.Module`` returns
    from ``raw_state_dict()``. It is an argument rather than something built here
    so the comparison -- the dtype half in particular, which is what makes an
    int8 checkpoint fed to a bf16-declaring graph (or the reverse) fail loudly --
    can be exercised against a hand-built mapping.
    """
    missing = sorted(set(declared) - set(state_dict))
    extra = sorted(set(state_dict) - set(declared))
    if missing or extra:
        raise WeightMappingError(
            f"language weights disagree with the checkpoint: missing {missing[:20]}, unexpected {extra[:20]}"
        )
    for name, weight in declared.items():
        want = tuple(int(d) for d in weight.shape.static_dims)
        got = tuple(int(d) for d in state_dict[name].shape)
        if want != got:
            raise WeightMappingError(f"{name}: checkpoint shape {got} but the graph declares {want}")
        want_dtype = _dtype_name(weight)
        got_dtype = _dtype_name(state_dict[name])
        if want_dtype != got_dtype:
            raise WeightMappingError(f"{name}: checkpoint dtype {got_dtype} but the graph declares {want_dtype}")


def language_state_dict(checkpoint: Mapping[str, Any], config: UnlimitedOCRConfig) -> dict[str, Any]:
    """The decoder's tensors, renamed and expert-stacked, verified against the declared weights."""
    selected = {
        key: value
        for key, value in checkpoint.items()
        if key == "lm_head.weight" or key.startswith(LANGUAGE_PREFIXES)
    }
    if not selected:
        raise WeightMappingError("no language weights found in the checkpoint")
    is_int8_checkpoint(selected)  # reject a half-quantized file here, where the names still say why
    renamed = {language_weight_name(key): value for key, value in selected.items()}
    out = stack_expert_weights(renamed, num_experts=config.decoder.n_routed_experts)
    check_against_declared(out, UnlimitedOcrDecoder(config.decoder, dtype=config.dtype).raw_state_dict())
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
