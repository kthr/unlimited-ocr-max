"""The Unlimited-OCR language decoder (``use_mla: false`` DeepSeek-V2) as ``max.nn`` modules.

12 pre-norm layers of hidden size 1280, 10-head MHA with ``rotate_half`` RoPE
(theta 10000), a dense FFN at layer 0 and a 64-expert / top-6 MoE with two
shared experts at layers 1-11, a final RMSNorm and ``lm_head``.

Invariants the emitted graph depends on:

* Weights stay in the checkpoint's bfloat16; every activation is fp32
  (:data:`COMPUTE_DTYPE`). MAX's matmul reads a bf16 weight against an fp32
  activation only through a weight-only upcast, and MAX compile-folds that
  upcast into an fp32 copy of the weight on the device at ``session.load``
  (OQ-113-A). So no bf16 weight meets a weight-only upcast here:
  :class:`Projection` reads the attention projections, the dense FFN, the
  shared experts and ``lm_head`` through the Mojo op ``dense_bf16_qmv``
  (``kernels/dense_bf16.mojo``) at the decode step and the prefill's last row,
  and through a run-time upcast at prefill (KON-238). Those weights are held
  K-blocked, ``[K / KBLOCK, N, KBLOCK]`` (:data:`KBLOCK`), the layout the op
  reads fastest; the weight adapter lays them out at load. The norms and the
  router, which MAX's own fp32 ops read whole, are stored as their exact fp32
  upcast under the shared registry (``norm_router_dtype=COMPUTE_DTYPE``); the
  routed expert stacks go to kernels that read bf16, and the table is
  gathered before its cast. Every language graph therefore needs
  ``custom_extensions=[MOJO_KERNELS]``.
* ``ops.rms_norm`` requires ``gamma.dtype == input.dtype``, so the bf16 norm
  weight is cast to fp32 inside :class:`RmsNorm`.
* The MoE gate is softmax over all 64 experts, then top-6, no renormalisation,
  no scaling (``norm_topk_prob`` false and ``routed_scaling_factor`` 1.0 are
  enforced by the config). The routed experts are three stacked ``[64, N, K]`` tensors with
  slice ``j`` == expert ``j``.
* The decode step (:meth:`MoE.decode_rows`, every row of the batched step,
  ``B = 1`` included) runs only the top-6 experts through a Mojo GEMV that
  reads them straight from the WHOLE stacks, on any device: ``moe_bf16_qmv``
  (``kernels/moe_bf16.mojo``) for bf16, ``moe_int8_qmv``
  (``kernels/moe_int8.mojo``) for int8, over all ``B * 6`` selected (token,
  expert) rows, reading each selected expert once for all the rows routed to
  it; the router weights are applied as one matmul, so it is not bitwise
  equal to the dense chain below. A graph that stages either kernel needs
  ``custom_extensions=[MOJO_KERNELS]``.
* Prefill (:meth:`MoE.__call__`) runs every expert, accumulated over
  ascending ``j`` as one unfused 64-term chain -- except bf16 on an
  accelerator, which runs the top-6 through MAX's native path
  (``moe_create_indices`` + ``grouped_matmul_ragged``, GPU-only on this MAX
  build): the dense chain consumes each expert as a weight-only
  ``x_fp32 @ w_bf16.T``, which MAX compile-folds into an fp32 copy of the
  whole expert stack on the device (9.02 GiB at prefill), while
  ``grouped_matmul_ragged`` reads the bf16 stack directly.
* With ``DecoderConfig.int8_experts`` the three stacks are int8 next to fp32
  per-group scales (``…experts.<proj>_scales``, group size
  :data:`~unlimited_ocr_max.model_config.INT8_GROUP_SIZE`) and reach the
  ``kernels/moe_int8.mojo`` ops WHOLE: ``moe_int8_qmv`` at decode (above), and
  at ``seq > 1`` ``int8_dequant_expert`` hands expert ``j`` to the same 64-term
  chain as a bf16 ``[N, K]`` weight. A graph-level slice of a weight stack is
  never staged in int8 mode: weight-only expressions are compile-folded and the
  folded slice is materialized on the device (KON-142), which the kernels exist
  to avoid. A custom op fed only constants is such an expression too, so the
  prefill dequantization takes a runtime-derived expert index
  (:meth:`MoE._runtime_zero`).
* Hidden states are rank-2 ``[seq_len, hidden]``; ``seq_len`` is a static graph
  dimension so the causal mask and the RoPE tables are graph constants. The
  decode step (:meth:`UnlimitedOcrDecoder.decode`) reuses the layout for ``B``
  independent one-token requests, ``B = 1`` included: there a row is a request.
* Decode attention is paged (:func:`paged_decode_attention`): per layer, one
  ``store_k_cache_ragged`` + ``store_v_cache_ragged`` writes every row's new
  k/v into the pipeline's page pool (:mod:`~unlimited_ocr_max.kv_cache`) and
  ONE ``flash_attention_ragged`` attends all ``B`` rows, each over its own
  cache length. It is not bitwise the old per-row chain (scores, scale,
  softmax, ``@ v`` per row); it is gated against a float64 reference of it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, Weight, ops
from max.nn import LayerList, Module
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    grouped_matmul_ragged,
    moe_create_indices,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues

from .model_config import INT8_GROUP_SIZE, DecoderConfig

__all__ = [
    "COMPUTE_DTYPE",
    "KBLOCK",
    "Attention",
    "DecoderLayer",
    "MoE",
    "PagedKv",
    "UnlimitedOcrDecoder",
    "causal_mask_bias",
    "dense_bf16_qmv",
    "paged_decode_attention",
    "rope_tables",
]

COMPUTE_DTYPE = DType.float32

#: bf16 elements per K block of a K-blocked projection weight: ``W [N, K]`` is held as
#: ``[K / KBLOCK, N, KBLOCK]``, block ``kb`` of every row side by side. One vector load of
#: ``dense_bf16_qmv`` (its ``KBLOCK``), so a SIMD group's loads are adjacent in memory.
KBLOCK = 16


def rope_tables(*, head_dim: int, seq_len: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """``(cos, sin)`` of shape ``[seq_len, head_dim]``, computed in float32 like ``LlamaRotaryEmbedding``."""
    exponent = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
    inv_freq = (1.0 / (np.float32(theta) ** (exponent / np.float32(head_dim)))).astype(np.float32)
    positions = np.arange(seq_len, dtype=np.int64).astype(np.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def causal_mask_bias(seq_len: int) -> np.ndarray:
    """Additive float32 causal mask: 0 where allowed, ``finfo(float32).min`` elsewhere."""
    allowed = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return np.where(allowed, np.float32(0.0), np.finfo(np.float32).min).astype(np.float32)


#: The fewest work items :func:`_rows_per_read` leaves a ``dense_bf16_qmv``
#: call, measured on the Apple M4: below it a narrow weight's call runs on too
#: few items to hide its loads. The 1280-wide attention projections at
#: ``B = 8``: 0.066-0.074 ms per call in two tiles of 4 rows (2560 items),
#: 0.14-0.15 ms in one tile of 8 (1280 items); at ``B = 5`` 0.060 ms in two
#: tiles of 3 against 0.092-0.096 ms in one of 5. The 6848-wide FFN and
#: ``lm_head`` hold 2048 items in one tile of every ``B``. That one tile is not
#: the fastest for ``lm_head`` at ``B = 4..6``: two tiles ran 2-4 % faster on
#: the final kernel (0.1-0.18 ms per step); the rule keeps one, so each weight
#: row is read once per call.
DENSE_MIN_ITEMS = 2048


def _rows_per_read(rows: int, out_dim: int) -> int:
    """``dense_bf16_qmv``'s tile: ``rows`` split as evenly as it goes into the fewest tiles that leave :data:`DENSE_MIN_ITEMS` items.

    The tile is ``ceil(rows / tiles)`` for the fewest ``tiles`` whose call has
    ``DENSE_MIN_ITEMS`` work items (``out_dim`` per tile); 1 when none has.
    ``rows`` is the decode step's ``B``, at most ``MAX_BATCH_CAP`` (8). A
    weight that wide on its own -- the 6848-wide FFN, ``lm_head`` -- takes one
    tile of all ``rows``, so each weight row is read once per call; the
    1280- and 1792-wide projections take two tiles of
    ``ceil(rows / 2)``. The kernel pads a last, short tile, so ``B = 7`` in
    tiles of 4 costs about what ``B = 8`` does (``kernels/dense_bf16.mojo``). A
    row's bits do not depend on the tile (the kernel's docstring), so it is a
    speed choice only.
    """
    tiles = 1
    while True:
        tile = math.ceil(rows / tiles)
        if tile == 1 or out_dim * math.ceil(rows / tile) >= DENSE_MIN_ITEMS:
            return tile
        tiles += 1


def dense_bf16_qmv(x: TensorValue, weight: TensorValue, *, tile: int | None = None) -> TensorValue:
    """``x @ W.T`` for fp32 ``x [M, K]`` and a bf16 ``W [N, K]`` held K-blocked: fp32 ``[M, N]`` through the Mojo op.

    ``weight`` is ``W`` as ``[K / KBLOCK, N, KBLOCK]`` (:data:`KBLOCK`). The op
    reads its bf16 bytes and accumulates in fp32 (``kernels/dense_bf16.mojo``);
    ``tile`` -- default :func:`_rows_per_read` -- is how many rows one work
    item serves from one read of a weight row. The graph needs
    ``custom_extensions=[MOJO_KERNELS]``.
    """
    if x.dtype != DType.float32 or weight.dtype != DType.bfloat16:
        raise TypeError(f"dense_bf16_qmv takes an fp32 activation and a bf16 weight, got {x.dtype} and {weight.dtype}")
    if weight.rank != 3 or int(weight.shape[2]) != KBLOCK:
        raise ValueError(f"dense_bf16_qmv takes a weight K-blocked by {KBLOCK}, got shape {weight.shape}")
    if tile is None:
        tile = _rows_per_read(int(x.shape[0]), int(weight.shape[1]))
    out_type = TensorType(DType.float32, [x.shape[0], weight.shape[1]], device=x.device)
    return ops.custom(
        "dense_bf16_qmv", device=x.device, values=[x, weight], out_types=[out_type], parameters={"tile": tile}
    )[0].tensor


def _runtime_one(x: TensorValue, dtype: DType) -> TensorValue:
    """A ``[1, 1]`` one in ``dtype`` that MAX cannot prove constant: ``1 + min(|x[0, 0]|, 0)`` of a rank-2 ``x``.

    ``|x[0, 0]| >= 0`` makes it exactly 1. For a NaN, ±Inf or -0.0 there it is
    also exactly 1: MAX's ``min(NaN, 0)`` returns the 0, not the NaN (measured
    on CPU and Metal). It is derived from ``x``, a runtime value, so
    ``weight * one`` -- and the upcast after it -- is not a weight-only
    expression and is not folded at load (the trick of
    :meth:`MoE._runtime_zero`). Multiplying a bf16 value by 1 is exact.
    """
    zero = ops.min(ops.abs(x[0:1, 0:1]), ops.constant(0.0, x.dtype, device=x.device))
    return ops.cast(zero + ops.constant(1.0, x.dtype, device=x.device), dtype)


class Projection(Module):
    """A bias-free ``W [out_dim, in_dim]`` applied as ``x @ W.T`` to an fp32 activation.

    A weight in the activation's dtype (the fp32-resident router) is MAX's
    matmul, a call. A bf16 weight never meets a weight-only upcast (the
    module docstring has why): :meth:`decode_rows` -- the decode step's rows
    and the prefill's last row -- reads it through :func:`dense_bf16_qmv`; a
    call -- the prefill's ``seq_len`` rows -- upcasts it at run time
    (:func:`_runtime_one`) and runs MAX's matmul on that transient fp32 copy,
    because the GEMV kernel is 3-5x slower than the matmul at ~300 rows.

    ``kblocked`` (a bf16 weight only) holds ``W`` as ``[in_dim / KBLOCK,
    out_dim, KBLOCK]``, the layout :func:`dense_bf16_qmv` reads; a call puts
    the fp32 copy back in ``[out_dim, in_dim]`` order (one permute), so the
    matmul sees the same values in the same order. The router, which only a
    call reads, is not K-blocked: under the shared registry it is fp32.
    """

    def __init__(self, in_dim: int, out_dim: int, *, dtype: DType, device: DeviceRef, kblocked: bool = True) -> None:
        super().__init__()
        self.kblocked = kblocked and dtype == DType.bfloat16
        if self.kblocked and in_dim % KBLOCK:
            raise ValueError(f"a K-blocked projection needs in_dim to be a multiple of {KBLOCK}, got {in_dim}")
        shape = [in_dim // KBLOCK, out_dim, KBLOCK] if self.kblocked else [out_dim, in_dim]
        self.weight = Weight("weight", dtype, shape, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.weight.dtype == x.dtype:
            return x @ self.weight.T
        upcast = ops.cast(self.weight * _runtime_one(x, self.weight.dtype), x.dtype)
        if self.kblocked:
            blocks, out_dim, width = (int(dim) for dim in upcast.shape)
            upcast = upcast.permute([1, 0, 2]).reshape((out_dim, blocks * width))
        return x @ upcast.T

    def decode_rows(self, x: TensorValue) -> TensorValue:
        """``x @ W.T`` for a few rows of the K-blocked bf16 weight: :func:`dense_bf16_qmv`."""
        return dense_bf16_qmv(x, self.weight)


class EmbeddingTable(Module):
    """A bare ``[vocab, hidden]`` lookup table declared as ``<attr>.weight``."""

    def __init__(self, num_embeddings: int, dim: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [num_embeddings, dim], device=device)

    def __call__(self, token_ids: TensorValue) -> TensorValue:
        return ops.gather(self.weight, token_ids, axis=0)


class RmsNorm(Module):
    """Llama-style ``ops.rms_norm`` with the bf16 gamma cast to the activation dtype (the op requires it)."""

    def __init__(self, dim: int, *, eps: float, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Weight("weight", dtype, [dim], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.rms_norm(x, ops.cast(self.weight, x.dtype), self.eps)


class GatedMlp(Module):
    """``down(silu(gate(x)) * up(x))`` -- the dense FFN and the shared experts."""

    def __init__(self, hidden_dim: int, ffn_dim: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.gate_proj = Projection(hidden_dim, ffn_dim, dtype=dtype, device=device)
        self.up_proj = Projection(hidden_dim, ffn_dim, dtype=dtype, device=device)
        self.down_proj = Projection(ffn_dim, hidden_dim, dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def decode_rows(self, x: TensorValue) -> TensorValue:
        """The same FFN for the decode step's rows: each projection through :meth:`Projection.decode_rows`."""
        return self.down_proj.decode_rows(ops.silu(self.gate_proj.decode_rows(x)) * self.up_proj.decode_rows(x))


class StackedExperts(Module):
    """The routed experts of one MoE layer as three ``[num_experts, N, K]`` stacks.

    A plain stack of the checkpoint's ``[out, in]`` per-expert tensors in
    ascending expert index -- the layout ``grouped_matmul_ragged`` requires.
    Declared without a ``.weight`` suffix: ``…mlp.experts.gate_proj``.

    With ``int8=True`` the stacks are int8 and each has an fp32
    ``[num_experts, N, K / INT8_GROUP_SIZE]`` sibling ``…experts.gate_proj_scales``
    -- the names the weight adapter stacks an int8 checkpoint into. The stacks
    reach the kernels whole -- the Mojo qmv at decode in either dtype, MAX's
    grouped matmul at bf16 accelerator prefill, ``int8_dequant_expert`` at
    int8 prefill (see the module docstring); only the CPU bf16 prefill chain
    slices them (:meth:`expert`).
    """

    def __init__(
        self, num_experts: int, hidden_dim: int, ffn_dim: int, *, dtype: DType, device: DeviceRef, int8: bool = False
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.int8 = int8
        self.device = device
        weight_dtype = DType.int8 if int8 else dtype
        self.gate_proj = Weight("gate_proj", weight_dtype, [num_experts, ffn_dim, hidden_dim], device=device)
        self.up_proj = Weight("up_proj", weight_dtype, [num_experts, ffn_dim, hidden_dim], device=device)
        self.down_proj = Weight("down_proj", weight_dtype, [num_experts, hidden_dim, ffn_dim], device=device)
        if int8:
            groups_in, groups_ffn = hidden_dim // INT8_GROUP_SIZE, ffn_dim // INT8_GROUP_SIZE
            fp32 = DType.float32
            self.gate_proj_scales = Weight("gate_proj_scales", fp32, [num_experts, ffn_dim, groups_in], device=device)
            self.up_proj_scales = Weight("up_proj_scales", fp32, [num_experts, ffn_dim, groups_in], device=device)
            self.down_proj_scales = Weight("down_proj_scales", fp32, [num_experts, hidden_dim, groups_ffn], device=device)

    def _stacks(self) -> tuple[tuple[Weight, Weight | None], ...]:
        """``(weight, scales)`` for gate, up, down; a bf16 stack has no scales (``None``)."""
        if not self.int8:
            return ((self.gate_proj, None), (self.up_proj, None), (self.down_proj, None))
        return (
            (self.gate_proj, self.gate_proj_scales),
            (self.up_proj, self.up_proj_scales),
            (self.down_proj, self.down_proj_scales),
        )

    def _dequant(self, weight: Weight, scales: Weight, expert_idx: TensorValue) -> TensorValue:
        """``int8_dequant_expert``: expert ``expert_idx`` (a ``[1]`` int32) of the whole stack as bf16 ``[N, K]``."""
        out_type = TensorType(DType.bfloat16, [weight.shape[1], weight.shape[2]], device=self.device)
        return ops.custom(
            "int8_dequant_expert", device=self.device, values=[weight, scales, expert_idx], out_types=[out_type]
        )[0].tensor

    def expert(
        self, expert_idx: int, runtime_zero: TensorValue | None = None
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Expert ``j``'s three rank-2 weights: static slices, or in int8 mode the kernel's bf16 dequantization.

        ``runtime_zero`` (int8 only) is a ``[1]`` int32 zero computed from a
        runtime value, added to the constant index. With only constants for
        inputs the dequantization is a weight-only expression: MAX runs it at
        ``session.load``, folds the fp32 upcast of the matmuls that consume it,
        and keeps the result on the device -- 9.02 GiB for every expert of the
        prefill graph (EXPERIMENTS.md, ID 27). A runtime-derived index keeps it
        a prefill-time op on the int8 stacks, computing the same values.
        """
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(f"expert {expert_idx} out of range for {self.num_experts}")
        if not self.int8:
            return self.gate_proj[expert_idx], self.up_proj[expert_idx], self.down_proj[expert_idx]
        index = ops.constant(np.asarray([expert_idx], dtype=np.int32), DType.int32, device=self.device)
        if runtime_zero is not None:
            index = runtime_zero + index
        gate, up, down = (self._dequant(weight, scales, index) for weight, scales in self._stacks())
        return gate, up, down

    def _qmv(self, x: TensorValue, expert_ids: TensorValue, weight: Weight, scales: Weight | None) -> TensorValue:
        """Row ``s`` of ``x`` against expert ``expert_ids[s]``, fp32 ``[k, N]``, by the stack's dtype.

        A bf16 stack runs ``moe_bf16_qmv(x, expert_ids, weight)``, an int8
        stack ``moe_int8_qmv(x, expert_ids, weight, scales)``.
        """
        out_type = TensorType(DType.float32, [x.shape[0], weight.shape[1]], device=self.device)
        if weight.dtype == DType.int8:
            symbol, values = "moe_int8_qmv", [x, expert_ids, weight, scales]
        else:
            symbol, values = "moe_bf16_qmv", [x, expert_ids, weight]
        return ops.custom(symbol, device=self.device, values=values, out_types=[out_type])[0].tensor

    def qmv(self, x: TensorValue, expert_ids: TensorValue) -> TensorValue:
        """``down(silu(gate(x)) * up(x))`` per row through the stack dtype's qmv kernel, ``[k, K] x [k] -> [k, hidden]``.

        Three kernel calls over the whole stacks; the kernel indexes the expert.
        The activation must already be fp32: both kernels pin their result, and
        so their ``x`` (and int8's ``scales``), to float32.
        """
        if x.dtype != DType.float32:
            raise TypeError(f"the qmv kernels take an fp32 activation, got {x.dtype}")
        (gate_w, gate_s), (up_w, up_s), (down_w, down_s) = self._stacks()
        gate_out = self._qmv(x, expert_ids, gate_w, gate_s)
        up_out = self._qmv(x, expert_ids, up_w, up_s)
        return self._qmv(ops.silu(gate_out) * up_out, expert_ids, down_w, down_s)

    def grouped(
        self,
        permuted: TensorValue,
        expert_start_indices: TensorValue,
        expert_ids: TensorValue,
        expert_usage_stats: TensorValue,
    ) -> TensorValue:
        """``down(silu(gate(x)) * up(x))`` through MAX's ragged grouped matmul (GPU only).

        Three kernel calls, not the fused gate|up of ``max.nn.moe``: this port
        does not store the fused copy. ``expert_ids`` is the full-width
        ``[num_experts]`` vector ``moe_create_indices`` returns.
        """
        args = (expert_start_indices, expert_ids, expert_usage_stats)
        gate_out = grouped_matmul_ragged(permuted, self.gate_proj, *args)
        up_out = grouped_matmul_ragged(permuted, self.up_proj, *args)
        return grouped_matmul_ragged(ops.silu(gate_out) * up_out, self.down_proj, *args)

    def apply(self, x: TensorValue, gate: TensorValue, up: TensorValue, down: TensorValue) -> TensorValue:
        return (ops.silu(x @ gate.T) * (x @ up.T)) @ down.T

    def __call__(self, expert_idx: int, x: TensorValue, runtime_zero: TensorValue | None = None) -> TensorValue:
        return self.apply(x, *self.expert(expert_idx, runtime_zero))


@dataclass(frozen=True)
class PagedKv:
    """A decode graph's view of the KV page pool: two collections over ONE ``kv_blocks``.

    ``store`` carries ``cache_lengths = write_index``, so the store writes row
    ``b``'s new k/v at its write index. ``attend`` carries ``cache_lengths =
    attend_len - 1``, so under the causal mask row ``b``'s one query attends
    rows ``[0, attend_len)``. An append (``write_index == attend_len - 1``) and
    an in-place ring overwrite (``write_index < attend_len - 1``) are both this
    one mapping. ``row_offsets`` is ``0 .. B``: one query per row.
    """

    params: MHAKVCacheParams
    store: PagedCacheValues
    attend: PagedCacheValues
    row_offsets: TensorValue


def paged_decode_attention(
    q: TensorValue, k_new: TensorValue, v_new: TensorValue, *, kv: PagedKv, layer_idx: int, scale: float
) -> TensorValue:
    """Store each row's new k/v at its write index, then ONE attention op for all ``B`` rows.

    ``q`` / ``k_new`` / ``v_new`` are post-RoPE ``[B, n_heads, head_dim]``,
    row ``b`` for request ``b``; returns ``[B, n_heads, head_dim]``, row ``b``
    attended over its own rows ``[0, attend_len_b)`` of layer ``layer_idx``.
    """
    layer = ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU())
    store_k_cache_ragged(kv.store, k_new, kv.row_offsets, layer)
    store_v_cache_ragged(kv.store, v_new, kv.row_offsets, layer)
    return flash_attention_ragged(kv.params, q, kv.row_offsets, kv.attend, layer, MHAMaskVariant.CAUSAL_MASK, scale)


class Attention(Module):
    """Plain multi-head attention with RoPE; ``o_proj`` naming as in the checkpoint."""

    def __init__(self, config: DecoderConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        q_dim = self.num_heads * self.head_dim
        self.q_proj = Projection(config.hidden_size, q_dim, dtype=dtype, device=device)
        self.k_proj = Projection(config.hidden_size, q_dim, dtype=dtype, device=device)
        self.v_proj = Projection(config.hidden_size, q_dim, dtype=dtype, device=device)
        self.o_proj = Projection(q_dim, config.hidden_size, dtype=dtype, device=device)

    def _heads(self, x: TensorValue) -> TensorValue:
        """``[seq, n_heads * head_dim]`` -> ``[n_heads, seq, head_dim]``."""
        return x.reshape((x.shape[0], self.num_heads, self.head_dim)).permute([1, 0, 2])

    def _rope(self, x: TensorValue, cos: TensorValue, sin: TensorValue) -> TensorValue:
        half = self.head_dim // 2
        rotated = ops.concat([-x[:, :, half:], x[:, :, :half]], axis=-1)
        return x * cos + rotated * sin

    def __call__(
        self, x: TensorValue, *, cos: TensorValue, sin: TensorValue, mask: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Full causal attention; also returns post-RoPE ``(k, v)``, head-major ``[n_heads, seq, head_dim]``."""
        seq = x.shape[0]
        q = self._rope(self._heads(self.q_proj(x)), cos, sin)
        k = self._rope(self._heads(self.k_proj(x)), cos, sin)
        v = self._heads(self.v_proj(x))
        # The reference scales after the matmul; pre-scaling q rounds differently.
        scores = (q @ k.transpose(-1, -2)) * self.scale + mask
        attended = ops.softmax(scores) @ v
        merged = attended.permute([1, 0, 2]).reshape((seq, self.hidden_size))
        return self.o_proj(merged), k, v

    def decode(self, x: TensorValue, *, cos: TensorValue, sin: TensorValue, kv: PagedKv, layer_idx: int) -> TensorValue:
        """One token for each of ``B`` independent requests, each over its own cached rows in the page pool.

        ``x`` is ``[B, hidden]``, one row per request; ``cos`` / ``sin`` are
        ``[B, 1, head_dim]``. The q/k/v projections (:meth:`Projection.decode_rows`),
        the ``rotate_half`` RoPE and ``o_proj`` run once over all ``B`` rows on
        ``[B, n_heads, head_dim]`` (no head permutes); attention is
        :func:`paged_decode_attention`, which also stores the new k/v rows --
        the graph has no KV outputs.
        """
        rows = x.shape[0]
        q = self._rope(self._rows(self.q_proj.decode_rows(x)), cos, sin)
        k_new = self._rope(self._rows(self.k_proj.decode_rows(x)), cos, sin)
        v_new = self._rows(self.v_proj.decode_rows(x))
        attended = paged_decode_attention(q, k_new, v_new, kv=kv, layer_idx=layer_idx, scale=self.scale)
        return self.o_proj.decode_rows(attended.reshape((rows, self.hidden_size)))

    def _rows(self, x: TensorValue) -> TensorValue:
        """``[B, n_heads * head_dim]`` -> ``[B, n_heads, head_dim]``."""
        return x.reshape((x.shape[0], self.num_heads, self.head_dim))


class MoEGate(Module):
    """Softmax over all experts in fp32, then top-k. Weight name ``gate.gate_score.weight``."""

    def __init__(self, hidden_dim: int, num_experts: int, num_experts_per_token: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.gate_score = Projection(hidden_dim, num_experts, dtype=dtype, device=device, kblocked=False)

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """``(topk_indices [seq, k] int64, topk_weights [seq, k] fp32)``, in score order."""
        logits = self.gate_score(ops.cast(x, DType.float32))
        scores = ops.softmax(logits)
        weights, indices = ops.top_k(scores, self.num_experts_per_token, -1)
        return indices, weights


class MoE(Module):
    """64 routed experts, top-6, plus the shared experts.

    The decode step -- ``B`` rows through :meth:`decode_rows`, ``B = 1``
    included -- runs only the six selected experts per row through the stack
    dtype's Mojo qmv kernel (:meth:`_routed_qmv`: ``moe_bf16_qmv`` or
    ``moe_int8_qmv``), on whichever device the module is built for: the
    kernels are device-agnostic.

    Prefill (a call) is per dtype and device. bf16 on an accelerator
    (``native_routing``) runs the six selected experts through MAX's grouped
    kernels (:meth:`_routed_native`); on CPU every expert runs, weighted by a
    dense ``[seq, 64]`` router matrix (:meth:`_routed_dense`). int8 runs that
    dense chain on each expert as dequantized by ``int8_dequant_expert``.
    """

    def __init__(
        self, config: DecoderConfig, *, dtype: DType, device: DeviceRef, norm_router_dtype: DType | None = None
    ) -> None:
        super().__init__()
        router_dtype = dtype if norm_router_dtype is None else norm_router_dtype
        self.num_experts = config.n_routed_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.native_routing = not device.is_cpu()
        self.int8 = config.int8_experts
        self.gate = MoEGate(
            config.hidden_size, config.n_routed_experts, config.num_experts_per_tok, dtype=router_dtype, device=device
        )
        self.experts = StackedExperts(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            dtype=dtype,
            device=device,
            int8=config.int8_experts,
        )
        self.shared_experts = GatedMlp(config.hidden_size, config.shared_experts_dim, dtype=dtype, device=device)

    def router_matrix(self, indices: TensorValue, weights: TensorValue) -> TensorValue:
        """Scatter ``[seq, k]`` top-k pairs into a dense ``[seq, num_experts]`` matrix."""
        seq = indices.shape[0]
        k = self.num_experts_per_token
        expert_ids = ops.constant(
            np.arange(self.num_experts, dtype=np.int64), DType.int64, device=indices.device
        ).reshape((1, 1, self.num_experts))
        selected = ops.equal(indices.reshape((seq, k, 1)), expert_ids)
        spread = ops.broadcast_to(weights.reshape((seq, k, 1)), (seq, k, self.num_experts))
        zero = ops.constant(0.0, weights.dtype, device=weights.device)
        return ops.sum(ops.where(selected, spread, zero), axis=1).reshape((seq, self.num_experts))

    def _accumulate(self, router: TensorValue, row_of: Callable[[int], TensorValue]) -> TensorValue:
        """``sum_j router[:, j] * row_of(j)`` over ascending ``j`` as an unfused 64-term chain.

        Both the order and the shape are load-bearing: MAX contracts each
        multiply into the add that consumes it, so any re-association (a stack
        and sum, a concat, a ``[1, k] @ [k, H]`` matmul, a 6-term sum) changes
        the fp32 result.
        """
        routed: TensorValue | None = None
        for expert_idx in range(self.num_experts):
            scaled = router[:, expert_idx : expert_idx + 1] * row_of(expert_idx)
            routed = scaled if routed is None else routed + scaled
        assert routed is not None
        return routed

    def _routed_dense(self, x: TensorValue, router: TensorValue, runtime_zero: TensorValue | None = None) -> TensorValue:
        return self._accumulate(router, lambda j: self.experts(j, x, runtime_zero))

    def _runtime_zero(self, indices: TensorValue) -> TensorValue:
        """A ``[1]`` int32 zero the compiler cannot prove is zero: ``min(indices[0, 0], 0)``.

        Top-k expert ids are never negative, so this is always 0; but it is
        derived from the router's runtime output, so anything it feeds is not a
        weight-only expression and cannot be folded at load (see
        :meth:`StackedExperts.expert`). ``x * 0`` or ``x - x`` could be
        canonicalized back to a constant; a ``min`` against 0 needs a range fact
        the compiler does not have.
        """
        first = ops.cast(ops.reshape(indices[0:1, 0:1], [1]), DType.int32)
        return ops.min(first, ops.constant(np.zeros(1, dtype=np.int32), DType.int32, device=indices.device))

    def _mix(self, x: TensorValue, weights: TensorValue, expert_out: TensorValue) -> TensorValue:
        """The k per-slot expert outputs ``[seq * k, hidden]`` weighted by the ``[seq, k]`` router as one matmul."""
        restored = expert_out.reshape((x.shape[0], self.num_experts_per_token, self.hidden_dim))
        # Both casts are no-ops at fp32 and guard the mixed-dtype matmul against a bf16 router.
        mixed = ops.unsqueeze(ops.cast(weights, restored.dtype), axis=1) @ restored
        return ops.cast(ops.squeeze(mixed, axis=1), x.dtype)

    def _routed_native(self, x: TensorValue, indices: TensorValue, weights: TensorValue) -> TensorValue:
        """Top-k experts through ``moe_create_indices`` + ``grouped_matmul_ragged``; router weights as one matmul."""
        k = self.num_experts_per_token
        token_expert_order, expert_start_indices, restore_token_order, expert_ids, expert_usage_stats = (
            moe_create_indices(ops.cast(ops.reshape(indices, [-1]), DType.int32), self.num_experts)
        )
        permuted = ops.gather(x, ops.cast(ops.floor_div(token_expert_order, k), DType.int32), axis=0)
        expert_out = self.experts.grouped(permuted, expert_start_indices, expert_ids, expert_usage_stats)
        return self._mix(x, weights, ops.gather(expert_out, restore_token_order, axis=0))

    def _routed_qmv(self, x: TensorValue, indices: TensorValue, weights: TensorValue) -> TensorValue:
        """Top-k experts through the stack dtype's qmv kernel on the whole stacks; router weights as one matmul.

        ``moe_bf16_qmv`` for bf16 stacks, ``moe_int8_qmv`` for int8
        (:meth:`StackedExperts.qmv`). For ``x [B, hidden]`` each token row is
        repeated ``k`` times, token-major (:meth:`_token_rows`), and
        ``indices [B, k]`` flattens in the same order, so kernel row
        ``b * k + s`` meets expert ``indices[b, s]`` -- no permutation, no
        restore gather -- and :meth:`_mix` folds the ``[B * k, hidden]``
        result back per token. The kernel groups the rows by expert itself
        and reads each selected expert once per call, however many tokens
        picked it. Three kernel calls per layer (gate, up, down) plus the
        same mixing matmul as :meth:`_routed_native`, without its index op
        and its two gathers.
        """
        expert_ids = ops.cast(ops.reshape(indices, [-1]), DType.int32)
        return self._mix(x, weights, self.experts.qmv(self._token_rows(x), expert_ids))

    def _token_rows(self, x: TensorValue) -> TensorValue:
        """``x [B, hidden]`` with every row repeated ``k`` times, token-major: ``[B * k, hidden]``."""
        k = self.num_experts_per_token
        rows = int(x.shape[0])
        spread = ops.broadcast_to(ops.unsqueeze(x, 1), (rows, k, self.hidden_dim))
        return spread.reshape((rows * k, self.hidden_dim))

    def decode_rows(self, x: TensorValue) -> TensorValue:
        """The MoE for the batched decode step, where row ``b`` of ``x [B, hidden]`` is request ``b``'s one token.

        Both dtypes route through their qmv kernel over the ``B * k`` selected
        (token, expert) rows (:meth:`_routed_qmv`), whatever ``B``: the
        prefill paths of :meth:`__call__` are the wrong tool for ``B`` single
        tokens (int8's dequantizes every expert to serve ``B * k`` of them).
        Rows that picked the same expert share its
        weight reads but not their sums: a row computes from its own token and
        experts only, so its bits do not depend on ``B``. The shared experts
        run :meth:`GatedMlp.decode_rows` (``dense_bf16_qmv``), which keeps that
        property.
        """
        indices, weights = self.gate(x)
        return self._routed_qmv(x, indices, weights) + self.shared_experts.decode_rows(x)

    def __call__(self, x: TensorValue) -> TensorValue:
        """The prefill's MoE over ``x [seq, hidden]``; the decode step is :meth:`decode_rows`."""
        indices, weights = self.gate(x)
        if self.int8:
            router = self.router_matrix(indices, weights)
            routed = self._routed_dense(x, router, runtime_zero=self._runtime_zero(indices))
        elif self.native_routing:
            routed = self._routed_native(x, indices, weights)
        else:
            routed = self._routed_dense(x, self.router_matrix(indices, weights))
        return routed + self.shared_experts(x)


class DecoderLayer(Module):
    """Pre-norm attention and FFN/MoE with two residual branches."""

    def __init__(
        self,
        config: DecoderConfig,
        layer_idx: int,
        *,
        dtype: DType,
        device: DeviceRef,
        norm_router_dtype: DType | None = None,
    ) -> None:
        super().__init__()
        norm_dtype = dtype if norm_router_dtype is None else norm_router_dtype
        self.self_attn = Attention(config, dtype=dtype, device=device)
        self.mlp: MoE | GatedMlp = (
            MoE(config, dtype=dtype, device=device, norm_router_dtype=norm_router_dtype)
            if config.is_moe_layer(layer_idx)
            else GatedMlp(config.hidden_size, config.intermediate_size, dtype=dtype, device=device)
        )
        norm_kwargs = {"eps": config.rms_norm_eps, "dtype": norm_dtype, "device": device}
        self.input_layernorm = RmsNorm(config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, **norm_kwargs)

    def __call__(
        self, x: TensorValue, *, cos: TensorValue, sin: TensorValue, mask: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Returns ``(hidden, key, value)`` with ``key``/``value`` head-major."""
        attended, key, value = self.self_attn(self.input_layernorm(x), cos=cos, sin=sin, mask=mask)
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x)), key, value

    def decode(self, x: TensorValue, *, cos: TensorValue, sin: TensorValue, kv: PagedKv, layer_idx: int) -> TensorValue:
        """One token for each of ``B`` independent rows; the norms and the FFN/MoE see all ``B`` at once.

        Attention is :meth:`Attention.decode` over the page pool; the FFN is
        :meth:`MoE.decode_rows` or :meth:`GatedMlp.decode_rows`.
        """
        x = x + self.self_attn.decode(self.input_layernorm(x), cos=cos, sin=sin, kv=kv, layer_idx=layer_idx)
        return x + self.mlp.decode_rows(self.post_attention_layernorm(x))


class UnlimitedOcrDecoder(Module):
    """``embed_tokens`` / ``layers`` / ``norm`` / ``lm_head``; FQNs are the checkpoint keys minus ``model.``.

    ``dtype`` is the storage dtype of every weight except, with
    ``config.int8_experts``, the routed expert stacks (int8 plus fp32 scales).
    ``norm_router_dtype`` (default: ``dtype``) overrides it for the RMSNorm
    gammas and the MoE router, the weights MAX's own fp32 ops read whole; see
    the module docstring for why the shared registry stores those in fp32 and
    every projection, ``lm_head`` included, in bf16.
    """

    def __init__(
        self,
        config: DecoderConfig,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef | None = None,
        norm_router_dtype: DType | None = None,
    ) -> None:
        super().__init__()
        device = device if device is not None else DeviceRef.CPU()
        norm_dtype = dtype if norm_router_dtype is None else norm_router_dtype
        self.config = config
        self.device = device
        self.embed_tokens = EmbeddingTable(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.layers = LayerList(
            [
                DecoderLayer(config, i, dtype=dtype, device=device, norm_router_dtype=norm_router_dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=norm_dtype, device=device)
        self.lm_head = Projection(config.hidden_size, config.vocab_size, dtype=dtype, device=device)

    def embed(self, token_ids: TensorValue) -> TensorValue:
        """The bf16 table lookup, upcast to fp32."""
        return ops.cast(self.embed_tokens(token_ids), COMPUTE_DTYPE)

    def __call__(self, hidden: TensorValue, *, seq_len: int) -> tuple[TensorValue, list[tuple[TensorValue, TensorValue]]]:
        """All layers and the final norm over a static ``seq_len``; also every layer's head-major ``(k, v)``."""
        cos_np, sin_np = rope_tables(head_dim=self.config.head_dim, seq_len=seq_len, theta=self.config.rope_theta)
        cos = ops.constant(cos_np.reshape(1, seq_len, self.config.head_dim), COMPUTE_DTYPE, device=self.device)
        sin = ops.constant(sin_np.reshape(1, seq_len, self.config.head_dim), COMPUTE_DTYPE, device=self.device)
        mask = ops.constant(causal_mask_bias(seq_len).reshape(1, seq_len, seq_len), COMPUTE_DTYPE, device=self.device)
        kv: list[tuple[TensorValue, TensorValue]] = []
        for layer in self.layers:
            hidden, key, value = layer(hidden, cos=cos, sin=sin, mask=mask)
            kv.append((key, value))
        return self.norm(hidden), kv

    def logits(self, normed: TensorValue) -> TensorValue:
        """``lm_head`` over the final row only, in fp32: one row, so :meth:`Projection.decode_rows`."""
        return ops.cast(self.lm_head.decode_rows(normed[-1:, :]), DType.float32)

    def rope_rows(self, positions: TensorValue, *, max_seq_len: int) -> tuple[TensorValue, TensorValue]:
        """``(cos, sin)`` at each of ``positions [B]`` as ``[B, 1, head_dim]``, gathered from a ``max_seq_len`` table."""
        head_dim = self.config.head_dim
        rows = int(positions.shape[0])
        cos_np, sin_np = rope_tables(head_dim=head_dim, seq_len=max_seq_len, theta=self.config.rope_theta)
        gathered = []
        for table in (cos_np, sin_np):
            constant = ops.constant(table, COMPUTE_DTYPE, device=self.device)
            gathered.append(ops.gather(constant, positions, axis=0).reshape((rows, 1, head_dim)))
        return gathered[0], gathered[1]

    def decode(self, hidden: TensorValue, *, positions: TensorValue, max_seq_len: int, kv: PagedKv) -> TensorValue:
        """One token for each of ``B`` independent requests (``B = 1`` included); ``normed [B, hidden]``.

        Row ``b`` of ``hidden`` is request ``b``, not a sequence position: it
        sits at ``positions[b]``, stores its new k/v rows and attends its own
        rows through ``kv`` (:class:`PagedKv`), every layer. Each MoE layer runs
        :meth:`MoE.decode_rows` on the ``[B, hidden]`` rows: the stack dtype's
        qmv kernel (``moe_bf16_qmv`` or ``moe_int8_qmv``) over the ``B * k``
        selected (token, expert) rows at any ``B`` and on any device, each
        selected expert read once per call, never a prefill path. Every
        projection runs :meth:`Projection.decode_rows`.
        """
        cos, sin = self.rope_rows(positions, max_seq_len=max_seq_len)
        for i, layer in enumerate(self.layers):
            hidden = layer.decode(hidden, cos=cos, sin=sin, kv=kv, layer_idx=i)
        return self.norm(hidden)

    def logits_rows(self, normed: TensorValue) -> TensorValue:
        """``lm_head`` over every row, in fp32: ``[B, vocab]`` for :meth:`decode`'s ``[B, hidden]``."""
        return ops.cast(self.lm_head.decode_rows(normed), DType.float32)
