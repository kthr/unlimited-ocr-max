"""SAM ViT-B image encoder (the DeepEncoder's first tower) in ``max.nn``.

Everything inside the graph is channels-last; ``SamViT.__call__`` takes NCHW
pixels and returns NCHW. Convolutions use MAX's native RSCF ``(kh, kw, in, out)``
filters, so :func:`sam_state_dict` transposes the checkpoint's PyTorch
``(out, in, kh, kw)`` filters once on the host (the FCRS path has no fused
Metal kernel). ``LayerNorm2d`` normalises over channels, which channels-last
puts on the trailing axis ``ops.layer_norm`` reduces.

Weight FQNs are the checkpoint keys minus ``model.sam_model.``. Resolution is
static per instance: ``pos_embed`` and the global blocks' ``rel_pos_*`` tables
are resampled on the host for anything other than the 1024px pretraining grid
(``get_abs_pos_sam`` bicubic+antialias, ``get_rel_pos`` linear), so the graph
only ever sees correctly shaped tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Conv2d, LayerNorm, Linear
from max.nn.layer import LayerList, Module

__all__ = [
    "CHECKPOINT_PREFIX",
    "DEPTH",
    "DOWNSAMPLE_CHANNELS",
    "EMBED_DIM",
    "GLOBAL_ATTN_INDEXES",
    "IN_CHANS",
    "NUM_HEADS",
    "PATCH_SIZE",
    "WINDOW_SIZE",
    "SamViT",
    "sam_state_dict",
]

PATCH_SIZE = 16
IN_CHANS = 3
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0
OUT_CHANS = 256
WINDOW_SIZE = 14
GLOBAL_ATTN_INDEXES = (2, 5, 8, 11)
DOWNSAMPLE_CHANNELS = (512, 1024)
NORM_EPS = 1e-6
CHECKPOINT_PREFIX = "model.sam_model."


def window_partition(x: TensorValue, window_size: int) -> tuple[TensorValue, tuple[int, int]]:
    """NHWC -> ``(B * n_win, ws, ws, C)``, zero-padding H and W up to a multiple of ``ws``."""
    batch, height, width, channels = (int(d) for d in x.shape)
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = ops.pad(x, [0, 0, 0, pad_h, 0, pad_w, 0, 0])
    padded_h, padded_w = height + pad_h, width + pad_w
    x = ops.reshape(
        x,
        (batch, padded_h // window_size, window_size, padded_w // window_size, window_size, channels),
    )
    windows = ops.reshape(
        ops.permute(x, [0, 1, 3, 2, 4, 5]),
        (batch * (padded_h // window_size) * (padded_w // window_size), window_size, window_size, channels),
    )
    return windows, (padded_h, padded_w)


def window_unpartition(
    windows: TensorValue, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> TensorValue:
    """Exact inverse of :func:`window_partition`, cropping the padding back off."""
    padded_h, padded_w = pad_hw
    height, width = hw
    n_windows, _, _, channels = (int(d) for d in windows.shape)
    batch = n_windows // (padded_h * padded_w // window_size // window_size)
    x = ops.reshape(
        windows,
        (batch, padded_h // window_size, padded_w // window_size, window_size, window_size, channels),
    )
    x = ops.reshape(ops.permute(x, [0, 1, 3, 2, 4, 5]), (batch, padded_h, padded_w, channels))
    if padded_h > height or padded_w > width:
        x = ops.slice_tensor(x, [slice(None), slice(0, height), slice(0, width), slice(None)])
    return x


def rel_pos_index(q_size: int, k_size: int) -> np.ndarray:
    """The constant ``relative_coords`` gather table from ``get_rel_pos``."""
    q_coords = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return relative.astype(np.int64)


class SamAttention(Module):
    """MHA with a decomposed relative-position bias.

    The reference scales only ``q @ k^T`` (SDPA's ``1/sqrt(head_dim)``); the
    rel-pos bias is added unscaled, as a height term ``(..., k_h, 1)`` plus a
    width term ``(..., 1, k_w)`` broadcast into the ``(k_h, k_w)`` key view.
    """

    def __init__(
        self, dim: int, num_heads: int, input_size: tuple[int, int], dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Linear(dim, dim * 3, dtype, device, has_bias=True)
        self.proj = Linear(dim, dim, dtype, device, has_bias=True)
        self.rel_pos_h = Weight("rel_pos_h", dtype, (2 * input_size[0] - 1, self.head_dim), device=device)
        self.rel_pos_w = Weight("rel_pos_w", dtype, (2 * input_size[1] - 1, self.head_dim), device=device)
        self.device = device

    def _decomposed_rel_pos(
        self, q: TensorValue, height: int, width: int
    ) -> tuple[TensorValue, TensorValue]:
        batch, num_heads, _, head_dim = (int(d) for d in q.shape)
        r_h = ops.gather(
            self.rel_pos_h,
            ops.constant(rel_pos_index(height, height), DType.int64, device=self.device),
            axis=0,
        )
        r_w = ops.gather(
            self.rel_pos_w,
            ops.constant(rel_pos_index(width, width), DType.int64, device=self.device),
            axis=0,
        )
        r_q = ops.reshape(q, (batch, num_heads, height, width, head_dim))

        # einsum("bhwc,hkc->bhwk")
        lhs = ops.reshape(ops.permute(r_q, [2, 0, 1, 3, 4]), (height, batch * num_heads * width, head_dim))
        rel_h = ops.matmul(lhs, ops.transpose(r_h, -1, -2))
        rel_h = ops.permute(ops.reshape(rel_h, (height, batch, num_heads, width, height)), [1, 2, 0, 3, 4])

        # einsum("bhwc,wkc->bhwk")
        rhs = ops.reshape(ops.permute(r_q, [3, 0, 1, 2, 4]), (width, batch * num_heads * height, head_dim))
        rel_w = ops.matmul(rhs, ops.transpose(r_w, -1, -2))
        rel_w = ops.permute(ops.reshape(rel_w, (width, batch, num_heads, height, width)), [1, 2, 3, 0, 4])

        seq = height * width
        return (
            ops.reshape(rel_h, (batch, num_heads, seq, height, 1)),
            ops.reshape(rel_w, (batch, num_heads, seq, 1, width)),
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        batch, height, width, dim = (int(d) for d in x.shape)
        seq = height * width
        num_heads, head_dim = self.num_heads, self.head_dim

        qkv = ops.reshape(self.qkv(x), (batch, seq, 3, num_heads, head_dim))
        qkv = ops.permute(qkv, [2, 0, 3, 1, 4])
        q, k, v = (ops.squeeze(part, 0) for part in ops.split(qkv, [1, 1, 1], axis=0))

        logits = ops.matmul(q, ops.transpose(k, -1, -2)) * (head_dim**-0.5)
        rel_h, rel_w = self._decomposed_rel_pos(q, height, width)
        attn_bias = rel_h + rel_w
        logits = ops.reshape(logits, (batch, num_heads, seq, height, width))
        logits = ops.reshape(logits + attn_bias, (batch, num_heads, seq, seq))

        out = ops.matmul(ops.softmax(logits), v)
        out = ops.reshape(out, (batch, num_heads, height, width, head_dim))
        out = ops.reshape(ops.permute(out, [0, 2, 3, 1, 4]), (batch, height, width, dim))
        return self.proj(out)


class SamMlpBlock(Module):
    """``lin2(gelu(lin1(x)))`` with exact-erf GELU."""

    def __init__(self, embedding_dim: int, mlp_dim: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.lin1 = Linear(embedding_dim, mlp_dim, dtype, device, has_bias=True)
        self.lin2 = Linear(mlp_dim, embedding_dim, dtype, device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.lin2(ops.gelu(self.lin1(x)))


class SamBlock(Module):
    """Pre-norm attention (windowed unless ``window_size == 0``) + MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        grid_size: tuple[int, int],
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm1 = LayerNorm(dim, [device], dtype, eps=NORM_EPS)
        self.attn = SamAttention(
            dim, num_heads, grid_size if window_size == 0 else (window_size, window_size), dtype, device
        )
        self.norm2 = LayerNorm(dim, [device], dtype, eps=NORM_EPS)
        self.mlp = SamMlpBlock(dim, int(dim * MLP_RATIO), dtype, device)

    def __call__(self, x: TensorValue) -> TensorValue:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            height, width = int(x.shape[1]), int(x.shape[2])
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (height, width))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class LayerNorm2d(Module):
    """LayerNorm over the channel axis of an NHWC tensor."""

    def __init__(self, num_channels: int, dtype: DType, device: DeviceRef, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, (num_channels,), device=device)
        self.bias = Weight("bias", dtype, (num_channels,), device=device)
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.layer_norm(x, self.weight, self.bias, self.eps)


class SamPatchEmbed(Module):
    """NCHW pixels -> NHWC patch grid."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.proj = Conv2d(
            kernel_size=patch_size,
            in_channels=in_chans,
            out_channels=embed_dim,
            dtype=dtype,
            stride=patch_size,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.proj(ops.permute(x, [0, 2, 3, 1]))


class SamViT(Module):
    """``ImageEncoderViT`` as configured by ``build_sam_vit_b``.

    ``__call__`` takes NCHW pixels ``(B, 3, image_size, image_size)`` and returns
    the ``net_3`` output in NCHW: ``(B, 1024, image_size / 64, image_size / 64)``.
    """

    def __init__(self, image_size: int, dtype: DType = DType.float32, device: DeviceRef | None = None) -> None:
        super().__init__()
        if image_size % PATCH_SIZE != 0:
            raise ValueError(f"image_size {image_size} is not a multiple of the patch size {PATCH_SIZE}")
        device = device or DeviceRef.CPU()
        grid = image_size // PATCH_SIZE
        self.image_size = image_size
        self.device = device
        self.dtype = dtype

        self.patch_embed = SamPatchEmbed(PATCH_SIZE, IN_CHANS, EMBED_DIM, dtype, device)
        self.pos_embed = Weight("pos_embed", dtype, (1, grid, grid, EMBED_DIM), device=device)
        self.blocks = LayerList(
            [
                SamBlock(
                    EMBED_DIM,
                    NUM_HEADS,
                    0 if i in GLOBAL_ATTN_INDEXES else WINDOW_SIZE,
                    (grid, grid),
                    dtype,
                    device,
                )
                for i in range(DEPTH)
            ]
        )
        self.neck = LayerList(
            [
                Conv2d(kernel_size=1, in_channels=EMBED_DIM, out_channels=OUT_CHANS, dtype=dtype, device=device, has_bias=False),
                LayerNorm2d(OUT_CHANS, dtype, device),
                Conv2d(kernel_size=3, in_channels=OUT_CHANS, out_channels=OUT_CHANS, dtype=dtype, padding=1, device=device, has_bias=False),
                LayerNorm2d(OUT_CHANS, dtype, device),
            ]
        )
        self.net_2 = Conv2d(
            kernel_size=3, in_channels=OUT_CHANS, out_channels=DOWNSAMPLE_CHANNELS[0],
            dtype=dtype, stride=2, padding=1, device=device, has_bias=False,
        )
        self.net_3 = Conv2d(
            kernel_size=3, in_channels=DOWNSAMPLE_CHANNELS[0], out_channels=DOWNSAMPLE_CHANNELS[1],
            dtype=dtype, stride=2, padding=1, device=device, has_bias=False,
        )

    def __call__(self, pixels: TensorValue) -> TensorValue:
        x = self.patch_embed(pixels)
        x = x + TensorValue(self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.neck(x)
        x2 = self.net_2(x)
        return ops.permute(self.net_3(x2), [0, 3, 1, 2])


def _interpolate_abs_pos(pos_embed: np.ndarray, tgt_size: int) -> np.ndarray:
    """Host-side ``get_abs_pos_sam``: torch bicubic + antialias, ``align_corners=False``."""
    if pos_embed.shape[1] == tgt_size:
        return pos_embed
    import torch
    import torch.nn.functional as F

    old = torch.from_numpy(np.ascontiguousarray(pos_embed)).permute(0, 3, 1, 2)
    new = F.interpolate(
        old.to(torch.float32), size=(tgt_size, tgt_size), mode="bicubic", antialias=True, align_corners=False
    ).to(old.dtype)
    return new.permute(0, 2, 3, 1).contiguous().numpy()


def _interpolate_rel_pos(rel_pos: np.ndarray, max_rel_dist: int) -> np.ndarray:
    """Host-side interpolation branch of ``get_rel_pos`` (``mode="linear"``)."""
    if rel_pos.shape[0] == max_rel_dist:
        return rel_pos
    import torch
    import torch.nn.functional as F

    table = torch.from_numpy(np.ascontiguousarray(rel_pos)).to(torch.float32)
    resized = F.interpolate(table.reshape(1, table.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear")
    return resized.reshape(-1, max_rel_dist).permute(1, 0).contiguous().numpy()


def as_float32(value: Any) -> np.ndarray:
    """Contiguous float32 copy of a checkpoint tensor (torch or numpy); bf16 -> fp32 is lossless."""
    if hasattr(value, "detach") and hasattr(value, "numpy"):
        value = value.detach().to("cpu").float().numpy()
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def sam_state_dict(
    checkpoint: Mapping[str, Any], image_size: int, *, prefix: str = CHECKPOINT_PREFIX
) -> dict[str, np.ndarray]:
    """Checkpoint tensors -> :class:`SamViT` state dict at ``image_size``.

    Strips ``prefix``, upcasts to fp32, resamples ``pos_embed`` and the global
    blocks' ``rel_pos_*`` for the resolution, and transposes the five 4-D conv
    filters to RSCF.
    """
    grid = image_size // PATCH_SIZE
    out: dict[str, np.ndarray] = {}
    for key, value in checkpoint.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        array = as_float32(value)
        if name == "pos_embed":
            array = _interpolate_abs_pos(array, grid)
        elif name.endswith((".rel_pos_h", ".rel_pos_w")):
            block = int(name.split(".")[1])
            q_size = grid if block in GLOBAL_ATTN_INDEXES else WINDOW_SIZE
            array = _interpolate_rel_pos(array, 2 * q_size - 1)
        elif array.ndim == 4:
            array = array.transpose(2, 3, 1, 0)
        out[name] = np.ascontiguousarray(array, dtype=np.float32)
    if not out:
        raise ValueError(f"no checkpoint keys start with {prefix!r}")
    return out
