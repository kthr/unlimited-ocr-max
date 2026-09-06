"""Typed, validated view of the checkpoint's ``config.json``.

The language fields are read from the top level (what the reference reads) and
cross-checked against the duplicated ``language_config`` block. The vision and
projector blocks are parsed only to refuse a checkpoint the hardcoded towers in
``layers/`` do not match.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from max.dtype import DType

from .layers import clip_l, projector, sam_vit

__all__ = [
    "ROPE_THETA",
    "ClipVisionConfig",
    "ConfigError",
    "DecoderConfig",
    "ProjectorConfig",
    "SamVitConfig",
    "UnlimitedOCRConfig",
    "VisionConfig",
]


class ConfigError(ValueError):
    """``config.json`` does not describe the model this package implements."""


#: Absent from ``config.json``; the reference's ``DeepseekV2Config`` default.
ROPE_THETA = 10000.0

#: Reference defaults for keys ``config.json`` leaves unset.
_DECODER_DEFAULTS: dict[str, Any] = {
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "moe_layer_freq": 1,
    "norm_topk_prob": False,
    "scoring_func": "softmax",
    "routed_scaling_factor": 1.0,
    "rope_theta": ROPE_THETA,
    "rope_scaling": None,
}


@dataclass(frozen=True, kw_only=True)
class DecoderConfig:
    """The MoE decoder in plain multi-head-attention form (``use_mla: false``)."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window_size: int

    first_k_dense_replace: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    topk_method: str
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    scoring_func: str
    routed_scaling_factor: float
    moe_layer_freq: int

    use_mla: bool
    kv_lora_rank: int | None
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    attention_bias: bool
    tie_word_embeddings: bool
    lm_head: bool
    rm_head: bool
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.use_mla:
            raise ConfigError("use_mla is True; only the plain-MHA path is implemented")
        for name in ("kv_lora_rank", "q_lora_rank"):
            if getattr(self, name) is not None:
                raise ConfigError(f"{name} must be null when use_mla is False")
        for name in ("qk_nope_head_dim", "qk_rope_head_dim"):
            if getattr(self, name) != 0:
                raise ConfigError(f"{name} must be 0 when use_mla is False")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError("hidden_size is not divisible by num_attention_heads")
        if self.head_dim != self.v_head_dim:
            raise ConfigError(
                f"derived head_dim {self.head_dim} disagrees with v_head_dim {self.v_head_dim}"
            )
        if self.num_attention_heads != self.num_key_value_heads:
            raise ConfigError("grouped-query attention is not implemented")
        if self.head_dim % 2:
            raise ConfigError("head_dim must be even for RoPE")
        if self.attention_bias:
            raise ConfigError("attention_bias is not supported")
        if self.hidden_act != "silu":
            raise ConfigError(f"unsupported hidden_act {self.hidden_act!r}")
        if self.scoring_func != "softmax":
            raise ConfigError(f"unsupported scoring_func {self.scoring_func!r}")
        if self.topk_method != "greedy":
            raise ConfigError(f"unsupported topk_method {self.topk_method!r}")
        if self.n_group != 1 or self.topk_group != 1:
            raise ConfigError("grouped expert routing is not implemented")
        if self.moe_layer_freq != 1:
            raise ConfigError("moe_layer_freq != 1 is not supported")
        if self.n_shared_experts <= 0:
            raise ConfigError("this port expects shared experts")
        if self.rope_scaling is not None:
            raise ConfigError("rope_scaling is not supported")
        if self.norm_topk_prob or self.routed_scaling_factor != 1.0:
            raise ConfigError("the router is implemented without top-k renormalisation or scaling")
        if self.tie_word_embeddings:
            raise ConfigError("tie_word_embeddings is True, but the checkpoint stores both tables")
        if not self.lm_head:
            raise ConfigError("lm_head is False; no output projection to map")
        if self.rm_head:
            raise ConfigError("rm_head is True; reward heads are not ported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def shared_experts_dim(self) -> int:
        return self.n_shared_experts * self.moe_intermediate_size

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range")
        return self.n_routed_experts > 0 and layer_idx >= self.first_k_dense_replace


@dataclass(frozen=True, kw_only=True)
class SamVitConfig:
    """SAM ViT-B tower fields; must match the constants in ``layers/sam_vit.py``."""

    image_size: int
    width: int
    layers: int
    heads: int
    global_attn_indexes: tuple[int, ...]
    downsample_channels: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = {
            "width": sam_vit.EMBED_DIM,
            "layers": sam_vit.DEPTH,
            "heads": sam_vit.NUM_HEADS,
            "global_attn_indexes": sam_vit.GLOBAL_ATTN_INDEXES,
            "downsample_channels": sam_vit.DOWNSAMPLE_CHANNELS,
        }
        for name, want in expected.items():
            if getattr(self, name) != want:
                raise ConfigError(f"sam_vit_b.{name} is {getattr(self, name)!r}, the tower is built for {want!r}")
        if self.image_size % sam_vit.PATCH_SIZE != 0:
            raise ConfigError("sam image_size is not a multiple of the patch size")


@dataclass(frozen=True, kw_only=True)
class ClipVisionConfig:
    """CLIP-L/14-224 tower fields; must match ``layers/clip_l.py``'s ``ClipLConfig``."""

    layers: int
    width: int
    heads: int
    patch_size: int
    image_size: int

    def __post_init__(self) -> None:
        built = clip_l.ClipLConfig()
        expected = {
            "layers": built.num_layers,
            "width": built.hidden_size,
            "heads": built.num_attention_heads,
            "patch_size": built.patch_size,
            "image_size": built.image_size,
        }
        for name, want in expected.items():
            if getattr(self, name) != want:
                raise ConfigError(f"clip-l-14-224.{name} is {getattr(self, name)!r}, the tower is built for {want!r}")


@dataclass(frozen=True, kw_only=True)
class VisionConfig:
    image_size: int
    sam: SamVitConfig
    clip: ClipVisionConfig

    def __post_init__(self) -> None:
        if self.sam.image_size != self.image_size:
            raise ConfigError("sam.image_size disagrees with vision_config.image_size")


@dataclass(frozen=True, kw_only=True)
class ProjectorConfig:
    input_dim: int
    n_embed: int
    projector_type: str

    def __post_init__(self) -> None:
        if self.projector_type != "linear":
            raise ConfigError(f"only the 'linear' projector is ported, got {self.projector_type!r}")
        if (self.input_dim, self.n_embed) != (projector.PROJECTOR_INPUT_DIM, projector.PROJECTOR_OUTPUT_DIM):
            raise ConfigError(f"projector is {self.input_dim} -> {self.n_embed}; the port is built for 2048 -> 1280")


#: Keys duplicated between the top level and ``language_config``.
_DUPLICATED_LANGUAGE_KEYS = (
    "bos_token_id", "eos_token_id", "first_k_dense_replace", "hidden_size",
    "intermediate_size", "kv_lora_rank", "lm_head", "max_position_embeddings",
    "moe_intermediate_size", "n_group", "n_routed_experts", "n_shared_experts",
    "num_attention_heads", "num_experts_per_tok", "num_hidden_layers",
    "num_key_value_heads", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "rm_head", "sliding_window_size", "topk_group", "topk_method", "use_mla",
    "v_head_dim", "vocab_size",
)

_MISSING = object()


@dataclass(frozen=True, kw_only=True)
class UnlimitedOCRConfig:
    torch_dtype: str
    decoder: DecoderConfig
    vision: VisionConfig
    projector: ProjectorConfig

    def __post_init__(self) -> None:
        if self.projector.n_embed != self.decoder.hidden_size:
            raise ConfigError("projector.n_embed must equal decoder.hidden_size")
        expected = self.vision.sam.downsample_channels[-1] + self.vision.clip.width
        if self.projector.input_dim != expected:
            raise ConfigError(
                f"projector.input_dim {self.projector.input_dim} != sam channels + clip width ({expected})"
            )

    @property
    def dtype(self) -> DType:
        """The checkpoint storage dtype, ``bfloat16``."""
        if self.torch_dtype != "bfloat16":
            raise ConfigError(f"unexpected torch_dtype {self.torch_dtype!r}")
        return DType.bfloat16

    @classmethod
    def from_json_file(cls, path: str | Path) -> UnlimitedOCRConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_hf_dict(json.load(fh))

    @classmethod
    def from_hf_dict(cls, hf: Mapping[str, Any]) -> UnlimitedOCRConfig:
        language_config = hf.get("language_config", {})
        if not isinstance(language_config, Mapping):
            raise ConfigError("language_config is not an object")
        mismatched = {
            key: (hf.get(key, _MISSING), language_config[key])
            for key in _DUPLICATED_LANGUAGE_KEYS
            if key in language_config and hf.get(key, _MISSING) != language_config[key]
        }
        if mismatched:
            raise ConfigError(
                "top level and language_config disagree on: "
                + ", ".join(f"{k}: {t!r} vs {n!r}" for k, (t, n) in sorted(mismatched.items()))
            )
        if "sliding_window" in hf and hf["sliding_window"] != hf.get("sliding_window_size"):
            raise ConfigError("sliding_window disagrees with sliding_window_size")

        def req(key: str) -> Any:
            if key not in hf:
                raise ConfigError(f"config.json is missing required key {key!r}")
            return hf[key]

        def opt(key: str) -> Any:
            return hf.get(key, _DECODER_DEFAULTS[key])

        decoder = DecoderConfig(
            hidden_size=req("hidden_size"),
            num_hidden_layers=req("num_hidden_layers"),
            num_attention_heads=req("num_attention_heads"),
            num_key_value_heads=req("num_key_value_heads"),
            intermediate_size=req("intermediate_size"),
            vocab_size=req("vocab_size"),
            max_position_embeddings=req("max_position_embeddings"),
            sliding_window_size=req("sliding_window_size"),
            first_k_dense_replace=req("first_k_dense_replace"),
            moe_intermediate_size=req("moe_intermediate_size"),
            n_routed_experts=req("n_routed_experts"),
            n_shared_experts=req("n_shared_experts"),
            num_experts_per_tok=req("num_experts_per_tok"),
            topk_method=req("topk_method"),
            n_group=req("n_group"),
            topk_group=req("topk_group"),
            norm_topk_prob=opt("norm_topk_prob"),
            scoring_func=opt("scoring_func"),
            routed_scaling_factor=opt("routed_scaling_factor"),
            moe_layer_freq=opt("moe_layer_freq"),
            use_mla=req("use_mla"),
            kv_lora_rank=req("kv_lora_rank"),
            q_lora_rank=req("q_lora_rank"),
            qk_nope_head_dim=req("qk_nope_head_dim"),
            qk_rope_head_dim=req("qk_rope_head_dim"),
            v_head_dim=req("v_head_dim"),
            hidden_act=opt("hidden_act"),
            rms_norm_eps=opt("rms_norm_eps"),
            rope_theta=opt("rope_theta"),
            rope_scaling=opt("rope_scaling"),
            attention_bias=opt("attention_bias"),
            tie_word_embeddings=opt("tie_word_embeddings"),
            lm_head=req("lm_head"),
            rm_head=req("rm_head"),
            eos_token_id=req("eos_token_id"),
        )

        vision_hf = req("vision_config")
        sam_hf = vision_hf["width"]["sam_vit_b"]
        clip_hf = vision_hf["width"]["clip-l-14-224"]
        vision = VisionConfig(
            image_size=vision_hf["image_size"],
            sam=SamVitConfig(
                image_size=vision_hf["image_size"],
                width=sam_hf["width"],
                layers=sam_hf["layers"],
                heads=sam_hf["heads"],
                global_attn_indexes=tuple(sam_hf["global_attn_indexes"]),
                downsample_channels=tuple(sam_hf["downsample_channels"]),
            ),
            clip=ClipVisionConfig(
                layers=clip_hf["layers"],
                width=clip_hf["width"],
                heads=clip_hf["heads"],
                patch_size=clip_hf["patch_size"],
                image_size=clip_hf["image_size"],
            ),
        )
        projector_hf = req("projector_config")
        return cls(
            torch_dtype=req("torch_dtype"),
            decoder=decoder,
            vision=vision,
            projector=ProjectorConfig(
                input_dim=projector_hf["input_dim"],
                n_embed=projector_hf["n_embed"],
                projector_type=projector_hf["projector_type"],
            ),
        )
