"""The CLIP-L/14-224 half of the DeepEncoder, fed SAM's grid as its patch embeddings.

This CLIP never runs its own patch convolution: the reference calls
``vision_model(pixels, patch_embeds)`` with SAM's output grid, so the checkpoint's
``embeddings.patch_embedding.weight`` is dead and is not declared here. The
activation is ``quick_gelu`` (``x * sigmoid(1.702 x)``), not ``gelu``, and
``pre_layrnorm`` keeps its upstream misspelling because it is the checkpoint key.

Weight FQNs equal the checkpoint keys minus ``model.vision_model.``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Embedding, LayerList, LayerNorm, Linear, Module

__all__ = [
    "UNUSED_CHECKPOINT_WEIGHTS",
    "ClipL",
    "ClipLConfig",
    "fuse_clip_sam",
]

#: Checkpoint keys (relative to ``model.vision_model.``) nothing reads.
UNUSED_CHECKPOINT_WEIGHTS = ("embeddings.patch_embedding.weight",)


@dataclass(frozen=True)
class ClipLConfig:
    """``deepencoder.vit_model_cfg``, restricted to what the forward pass reads."""

    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    ffn_hidden_size: int = 4096
    layernorm_epsilon: float = 1e-5
    pre_layernorm_epsilon: float = 1e-5
    image_size: int = 224
    patch_size: int = 14

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_positions(self) -> int:
        """Rows in the position-embedding table: ``(224 / 14) ** 2 + 1`` for CLS."""
        return (self.image_size // self.patch_size) ** 2 + 1


QUICK_GELU_SCALE = 1.702


def quick_gelu(x: TensorValue) -> TensorValue:
    return x * ops.sigmoid(x * QUICK_GELU_SCALE)


#: torch's antialiased bicubic is the PIL-compatible Keys cubic with a = -0.5
#: (its non-antialiased path, and ``ops.resize_bicubic``, use a = -0.75).
AA_CUBIC_A = -0.5


def _cubic_filter(x: np.ndarray, a: float = AA_CUBIC_A) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=np.float64))
    w = np.zeros_like(x)
    near = x < 1.0
    far = (x >= 1.0) & (x < 2.0)
    w[near] = ((a + 2.0) * x[near] - (a + 3.0)) * x[near] * x[near] + 1.0
    w[far] = ((a * x[far] - 5.0 * a) * x[far] + 8.0 * a) * x[far] - 4.0 * a
    return w


def bicubic_antialias_matrix(src: int, dst: int) -> np.ndarray:
    """``[dst, src]`` matrix of torch's 1-D antialiased bicubic resample (``align_corners=False``)."""
    if src <= 0 or dst <= 0:
        raise ValueError(f"src and dst must be positive, got {src=} {dst=}")
    scale = src / dst
    support = 2.0 * scale if scale >= 1.0 else 2.0
    inv_scale = 1.0 / scale if scale >= 1.0 else 1.0
    matrix = np.zeros((dst, src), dtype=np.float64)
    for i in range(dst):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), src)
        taps = np.arange(lo, hi, dtype=np.float64)
        weights = _cubic_filter((taps - center + 0.5) * inv_scale)
        total = weights.sum()
        if total == 0.0:
            raise ValueError(f"degenerate resample row {i} for {src}->{dst}")
        matrix[i, lo:hi] = weights / total
    return matrix


def position_embedding_resize_operator(num_positions: int, num_tokens: int) -> np.ndarray | None:
    """``get_abs_pos`` as one ``[num_tokens - 1, num_positions - 1]`` matrix, or ``None`` when it is a no-op."""
    src = int(math.sqrt(num_positions - 1))
    dst = int(math.sqrt(num_tokens))
    if src * src != num_positions - 1:
        raise ValueError(f"num_positions - 1 = {num_positions - 1} is not a perfect square")
    if dst * dst + 1 != num_tokens:
        raise ValueError(f"num_tokens = {num_tokens} is not a perfect square plus one")
    if src == dst:
        return None
    row = bicubic_antialias_matrix(src, dst)
    return np.einsum("yj,xi->yxji", row, row).reshape(dst * dst, src * src)


class ClipVisionEmbeddings(Module):
    """CLS + SAM grid tokens + (possibly resized) absolute position embedding."""

    def __init__(self, config: ClipLConfig, *, grid_h: int, grid_w: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.config = config
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.device = device
        self.num_tokens = grid_h * grid_w + 1
        self.class_embedding = Weight("class_embedding", dtype, (config.hidden_size,), device=device)
        # Owns the table under the checkpoint FQN; its gather is never used.
        self.position_embedding = Embedding(config.num_positions, config.hidden_size, dtype, device)
        self._resize = position_embedding_resize_operator(config.num_positions, self.num_tokens)

    def _absolute_position_embedding(self) -> TensorValue:
        table = TensorValue(self.position_embedding.weight)
        hidden = self.config.hidden_size
        if self._resize is None:
            return table.reshape((1, self.config.num_positions, hidden))
        cls_row = ops.slice_tensor(table, [slice(0, 1), slice(None)])
        grid = ops.slice_tensor(table, [slice(1, self.config.num_positions), slice(None)])
        operator = ops.constant(self._resize.astype(np.float32), DType.float32, device=self.device)
        resized = ops.matmul(operator, grid.cast(DType.float32)).cast(table.dtype)
        return ops.concat([cls_row, resized], axis=0).reshape((1, self.num_tokens, hidden))

    def __call__(self, patch_embeds: TensorValue) -> TensorValue:
        """``[batch, hidden, grid_h, grid_w]`` -> ``[batch, grid_h * grid_w + 1, hidden]``."""
        hidden = self.config.hidden_size
        batch = patch_embeds.shape[0]
        expected = (hidden, self.grid_h, self.grid_w)
        actual = tuple(int(d) for d in patch_embeds.shape[1:])
        if actual != expected:
            raise ValueError(f"patch_embeds must be [batch, {hidden}, {self.grid_h}, {self.grid_w}], got {actual}")
        tokens = patch_embeds.reshape((batch, hidden, self.grid_h * self.grid_w)).transpose(1, 2)
        class_embeds = ops.broadcast_to(
            TensorValue(self.class_embedding).reshape((1, 1, hidden)), (batch, 1, hidden)
        )
        embeddings = ops.concat([class_embeds, tokens], axis=1)
        return embeddings + self._absolute_position_embedding()


class ClipAttention(Module):
    """Unmasked full self-attention over the checkpoint's fused ``qkv_proj``."""

    def __init__(self, config: ClipLConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_proj = Linear(config.hidden_size, 3 * config.hidden_size, dtype=dtype, device=device, has_bias=True)
        self.out_proj = Linear(config.hidden_size, config.hidden_size, dtype=dtype, device=device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        batch, seq = x.shape[0], x.shape[1]
        qkv = self.qkv_proj(x).reshape((batch, seq, 3, self.num_heads, self.head_dim))
        q, k, v = (ops.squeeze(part, 2).permute([0, 2, 1, 3]) for part in ops.split(qkv, [1, 1, 1], axis=2))
        scores = ops.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = ops.softmax(scores)
        attended = ops.matmul(weights, v).permute([0, 2, 1, 3])
        return self.out_proj(attended.reshape((batch, seq, self.hidden_size)))


class ClipMlp(Module):
    def __init__(self, config: ClipLConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.ffn_hidden_size, dtype=dtype, device=device, has_bias=True)
        self.fc2 = Linear(config.ffn_hidden_size, config.hidden_size, dtype=dtype, device=device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.fc2(quick_gelu(self.fc1(x)))


class ClipTransformerBlock(Module):
    def __init__(self, config: ClipLConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(config.hidden_size, devices=[device], dtype=dtype, eps=config.layernorm_epsilon)
        self.layer_norm2 = LayerNorm(config.hidden_size, devices=[device], dtype=dtype, eps=config.layernorm_epsilon)
        self.self_attn = ClipAttention(config, dtype=dtype, device=device)
        self.mlp = ClipMlp(config, dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class ClipTransformer(Module):
    def __init__(self, config: ClipLConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.layers = LayerList(
            [ClipTransformerBlock(config, dtype=dtype, device=device) for _ in range(config.num_layers)]
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ClipL(Module):
    """``VitModel`` as built by ``build_clip_l()``; one instance per view resolution."""

    def __init__(
        self,
        config: ClipLConfig | None = None,
        *,
        grid_h: int,
        grid_w: int,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        config = config or ClipLConfig()
        device = device or DeviceRef.CPU()
        self.embeddings = ClipVisionEmbeddings(config, grid_h=grid_h, grid_w=grid_w, dtype=dtype, device=device)
        self.pre_layrnorm = LayerNorm(
            config.hidden_size, devices=[device], dtype=dtype, eps=config.pre_layernorm_epsilon
        )
        self.transformer = ClipTransformer(config, dtype=dtype, device=device)

    def __call__(self, patch_embeds: TensorValue) -> TensorValue:
        """SAM's grid ``[batch, hidden, grid_h, grid_w]`` -> ``[batch, grid_h * grid_w + 1, hidden]``."""
        x = self.embeddings(patch_embeds)
        return self.transformer(self.pre_layrnorm(x))


def fuse_clip_sam(clip_out: TensorValue, sam_out: TensorValue) -> TensorValue:
    """``cat((clip_out[:, 1:], sam_out.flatten(2).permute(0, 2, 1)), dim=-1)``."""
    if clip_out.rank != 3:
        raise ValueError(f"clip_out must be rank 3, got {clip_out.rank}")
    if sam_out.rank != 4:
        raise ValueError(f"sam_out must be rank 4, got {sam_out.rank}")
    batch = sam_out.shape[0]
    channels, grid_h, grid_w = (int(d) for d in sam_out.shape[1:])
    tokens = int(clip_out.shape[1])
    if tokens != grid_h * grid_w + 1:
        raise ValueError(f"clip_out has {tokens} tokens but sam_out's grid implies {grid_h * grid_w + 1}")
    patches = ops.slice_tensor(clip_out, [slice(None), slice(1, tokens), slice(None)])
    sam_tokens = sam_out.reshape((batch, channels, grid_h * grid_w)).transpose(1, 2)
    return ops.concat([patches, sam_tokens], axis=-1)
