"""The Unlimited-OCR language decoder (``use_mla: false`` DeepSeek-V2) as ``max.nn`` modules.

12 pre-norm layers of hidden size 1280, 10-head MHA with ``rotate_half`` RoPE
(theta 10000), a dense FFN at layer 0 and a 64-expert / top-6 MoE with two
shared experts at layers 1-11, a final RMSNorm and ``lm_head``.

Invariants the emitted graph depends on:

* Weights stay in the checkpoint's bfloat16; every activation is fp32
  (:data:`COMPUTE_DTYPE`). A bf16-weight x fp32-activation matmul is bitwise
  the fp32 matmul against the upcast weight on this MAX build.
* ``ops.rms_norm`` requires ``gamma.dtype == input.dtype``, so the bf16 norm
  weight is cast to fp32 inside :class:`RmsNorm`.
* The MoE gate is softmax over all 64 experts, then top-6, no renormalisation,
  no scaling (``norm_topk_prob`` false and ``routed_scaling_factor`` 1.0 are
  enforced by the config). The routed experts are three stacked ``[64, N, K]`` tensors with
  slice ``j`` == expert ``j``; the dense accumulation runs over ascending ``j``
  as one unfused 64-term chain, and the hand-rolled top-6 decode path reuses that
  exact chain so it is bitwise equal to the dense path. The native decode path
  (``moe_create_indices`` + ``grouped_matmul_ragged``) is GPU-only on this MAX
  build and is not bitwise equal to either.
* Hidden states are rank-2 ``[seq_len, hidden]``; ``seq_len`` is a static graph
  dimension so the causal mask and the RoPE tables are graph constants.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import LayerList, Module
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices

from .model_config import DecoderConfig

__all__ = [
    "COMPUTE_DTYPE",
    "Attention",
    "DecoderLayer",
    "MoE",
    "UnlimitedOcrDecoder",
    "causal_mask_bias",
    "rope_tables",
]

COMPUTE_DTYPE = DType.float32


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


class Projection(Module):
    """A bias-free ``[out_dim, in_dim]`` bf16 weight applied as ``x @ w.T`` to an fp32 activation."""

    def __init__(self, in_dim: int, out_dim: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [out_dim, in_dim], device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return x @ self.weight.T


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


class StackedExperts(Module):
    """The routed experts of one MoE layer as three ``[num_experts, N, K]`` stacks.

    A plain stack of the checkpoint's ``[out, in]`` per-expert tensors in
    ascending expert index -- the layout ``grouped_matmul_ragged`` requires.
    Declared without a ``.weight`` suffix: ``…mlp.experts.gate_proj``.
    """

    def __init__(self, num_experts: int, hidden_dim: int, ffn_dim: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate_proj = Weight("gate_proj", dtype, [num_experts, ffn_dim, hidden_dim], device=device)
        self.up_proj = Weight("up_proj", dtype, [num_experts, ffn_dim, hidden_dim], device=device)
        self.down_proj = Weight("down_proj", dtype, [num_experts, hidden_dim, ffn_dim], device=device)

    def expert(self, expert_idx: int) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Expert ``j``'s three rank-2 weights, as static slices."""
        if not 0 <= expert_idx < self.num_experts:
            raise IndexError(f"expert {expert_idx} out of range for {self.num_experts}")
        return self.gate_proj[expert_idx], self.up_proj[expert_idx], self.down_proj[expert_idx]

    def select(self, expert_ids: TensorValue) -> tuple[TensorValue, TensorValue, TensorValue]:
        """The three stacks gathered at a runtime-valued ``[k]`` set of expert ids."""
        return (
            ops.gather(self.gate_proj, expert_ids, axis=0),
            ops.gather(self.up_proj, expert_ids, axis=0),
            ops.gather(self.down_proj, expert_ids, axis=0),
        )

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

    def __call__(self, expert_idx: int, x: TensorValue) -> TensorValue:
        return self.apply(x, *self.expert(expert_idx))


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

    def decode(
        self,
        x: TensorValue,
        *,
        cos: TensorValue,
        sin: TensorValue,
        past_k: TensorValue,
        past_v: TensorValue,
        write_sel: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """One-token attention over a cached prefix, unmasked.

        ``past_k`` / ``past_v`` are sequence-major ``[past_len, n_kv_heads, head_dim]``
        cache rows; ``write_sel`` is a bool ``[1, past_len, 1]`` one-hot selector
        naming the row this token's KV replaces (R-SWA's in-place ring write; an
        append is the case where it points one row past the live end). The
        substitution happens head-major, after the permute, so the matmul sees a
        contiguous operand. Returns the new rows in the cache's sequence-major
        ``[1, n_kv_heads, head_dim]`` layout.
        """
        q = self._rope(self._heads(self.q_proj(x)), cos, sin)
        k_new = self._rope(self._heads(self.k_proj(x)), cos, sin)
        v_new = self._heads(self.v_proj(x))
        k = ops.where(write_sel, k_new, past_k.permute([1, 0, 2]))
        v = ops.where(write_sel, v_new, past_v.permute([1, 0, 2]))
        scores = (q @ k.transpose(-1, -2)) * self.scale
        attended = ops.softmax(scores) @ v
        merged = attended.permute([1, 0, 2]).reshape((1, self.hidden_size))
        return self.o_proj(merged), k_new.permute([1, 0, 2]), v_new.permute([1, 0, 2])


class MoEGate(Module):
    """Softmax over all experts in fp32, then top-k. Weight name ``gate.gate_score.weight``."""

    def __init__(self, hidden_dim: int, num_experts: int, num_experts_per_token: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.gate_score = Projection(hidden_dim, num_experts, dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """``(topk_indices [seq, k] int64, topk_weights [seq, k] fp32)``, in score order."""
        logits = self.gate_score(ops.cast(x, DType.float32))
        scores = ops.softmax(logits)
        weights, indices = ops.top_k(scores, self.num_experts_per_token, -1)
        return indices, weights


class MoE(Module):
    """64 routed experts, top-6, plus the shared experts.

    At ``seq > 1`` (prefill) every expert runs and a dense ``[seq, 64]`` router
    matrix weights them. At ``seq == 1`` (decode) only the six selected experts
    run: on CPU through the hand-rolled gather that reuses the dense 64-term
    accumulation (bitwise equal to it), on an accelerator through MAX's grouped
    kernels (``native_decode``).
    """

    def __init__(self, config: DecoderConfig, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.native_decode = not device.is_cpu()
        self.gate = MoEGate(
            config.hidden_size, config.n_routed_experts, config.num_experts_per_tok, dtype=dtype, device=device
        )
        self.experts = StackedExperts(
            config.n_routed_experts, config.hidden_size, config.moe_intermediate_size, dtype=dtype, device=device
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

    def _routed_dense(self, x: TensorValue, router: TensorValue) -> TensorValue:
        return self._accumulate(router, lambda j: self.experts(j, x))

    def _routed_sparse(self, x: TensorValue, indices: TensorValue, router: TensorValue) -> TensorValue:
        """Top-k experts only, then the same 64-term chain with zero-weight rows substituted.

        ``ops.top_k`` reports score order, so the k outputs are scattered into a
        ``[num_experts, hidden]`` ``picks`` tensor indexed by expert id
        (unselected experts point at slot 0 and are multiplied by exact 0.0).
        """
        k = self.num_experts_per_token
        selected = indices.reshape((k,))
        gate_w, up_w, down_w = self.experts.select(selected)
        outputs = ops.concat(
            [self.experts.apply(x, gate_w[slot], up_w[slot], down_w[slot]) for slot in range(k)], axis=0
        )
        expert_ids = ops.constant(
            np.arange(self.num_experts, dtype=np.int64), DType.int64, device=selected.device
        ).reshape((1, self.num_experts))
        slots = ops.broadcast_to(
            ops.constant(np.arange(k, dtype=np.int64), DType.int64, device=selected.device).reshape((k, 1)),
            (k, self.num_experts),
        )
        zero_slot = ops.constant(np.int64(0), DType.int64, device=selected.device)
        slot_of_expert = ops.sum(
            ops.where(ops.equal(selected.reshape((k, 1)), expert_ids), slots, zero_slot), axis=0
        ).reshape((self.num_experts,))
        picks = ops.gather(outputs, slot_of_expert, axis=0)
        return self._accumulate(router, lambda j: picks[j : j + 1, :])

    def _routed_native(self, x: TensorValue, indices: TensorValue, weights: TensorValue) -> TensorValue:
        """Top-k experts through ``moe_create_indices`` + ``grouped_matmul_ragged``; router weights as one matmul."""
        seq = x.shape[0]
        k = self.num_experts_per_token
        token_expert_order, expert_start_indices, restore_token_order, expert_ids, expert_usage_stats = (
            moe_create_indices(ops.cast(ops.reshape(indices, [-1]), DType.int32), self.num_experts)
        )
        permuted = ops.gather(x, ops.cast(ops.floor_div(token_expert_order, k), DType.int32), axis=0)
        expert_out = self.experts.grouped(permuted, expert_start_indices, expert_ids, expert_usage_stats)
        restored = ops.gather(expert_out, restore_token_order, axis=0).reshape((seq, k, self.hidden_dim))
        # Both casts are no-ops at fp32 and guard the mixed-dtype matmul against a bf16 router.
        mixed = ops.unsqueeze(ops.cast(weights, restored.dtype), axis=1) @ restored
        return ops.cast(ops.squeeze(mixed, axis=1), x.dtype)

    def __call__(self, x: TensorValue) -> TensorValue:
        indices, weights = self.gate(x)
        router = self.router_matrix(indices, weights)
        if x.shape[0] == 1:
            if self.native_decode:
                routed = self._routed_native(x, indices, weights)
            else:
                routed = self._routed_sparse(x, indices, router)
        else:
            routed = self._routed_dense(x, router)
        return routed + self.shared_experts(x)


class DecoderLayer(Module):
    """Pre-norm attention and FFN/MoE with two residual branches."""

    def __init__(self, config: DecoderConfig, layer_idx: int, *, dtype: DType, device: DeviceRef) -> None:
        super().__init__()
        self.self_attn = Attention(config, dtype=dtype, device=device)
        self.mlp: Module = (
            MoE(config, dtype=dtype, device=device)
            if config.is_moe_layer(layer_idx)
            else GatedMlp(config.hidden_size, config.intermediate_size, dtype=dtype, device=device)
        )
        norm_kwargs = {"eps": config.rms_norm_eps, "dtype": dtype, "device": device}
        self.input_layernorm = RmsNorm(config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, **norm_kwargs)

    def __call__(
        self, x: TensorValue, *, cos: TensorValue, sin: TensorValue, mask: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Returns ``(hidden, key, value)`` with ``key``/``value`` head-major."""
        attended, key, value = self.self_attn(self.input_layernorm(x), cos=cos, sin=sin, mask=mask)
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x)), key, value

    def decode(
        self,
        x: TensorValue,
        *,
        cos: TensorValue,
        sin: TensorValue,
        past_k: TensorValue,
        past_v: TensorValue,
        write_sel: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attended, key, value = self.self_attn.decode(
            self.input_layernorm(x), cos=cos, sin=sin, past_k=past_k, past_v=past_v, write_sel=write_sel
        )
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x)), key, value


class UnlimitedOcrDecoder(Module):
    """``embed_tokens`` / ``layers`` / ``norm`` / ``lm_head``; FQNs are the checkpoint keys minus ``model.``."""

    def __init__(self, config: DecoderConfig, *, dtype: DType = DType.bfloat16, device: DeviceRef | None = None) -> None:
        super().__init__()
        device = device if device is not None else DeviceRef.CPU()
        self.config = config
        self.device = device
        self.embed_tokens = EmbeddingTable(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.layers = LayerList(
            [DecoderLayer(config, i, dtype=dtype, device=device) for i in range(config.num_hidden_layers)]
        )
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)
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

    def rope_row(self, position: TensorValue, *, max_seq_len: int) -> tuple[TensorValue, TensorValue]:
        """``(cos, sin)`` at ``position`` as ``[1, 1, head_dim]``, gathered from a ``max_seq_len`` table."""
        head_dim = self.config.head_dim
        cos_np, sin_np = rope_tables(head_dim=head_dim, seq_len=max_seq_len, theta=self.config.rope_theta)
        rows = []
        for table in (cos_np, sin_np):
            constant = ops.constant(table, COMPUTE_DTYPE, device=self.device)
            rows.append(ops.gather(constant, position, axis=0).reshape((1, 1, head_dim)))
        return rows[0], rows[1]

    def decode(
        self,
        hidden: TensorValue,
        *,
        position: TensorValue,
        max_seq_len: int,
        past_kv: Sequence[tuple[TensorValue, TensorValue]],
        write_sel: TensorValue,
    ) -> tuple[TensorValue, list[tuple[TensorValue, TensorValue]]]:
        """One token against the cached prefix; returns ``(normed [1, hidden], new (k, v) rows per layer)``."""
        if len(past_kv) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} cache pairs, got {len(past_kv)}")
        cos, sin = self.rope_row(position, max_seq_len=max_seq_len)
        new_kv: list[tuple[TensorValue, TensorValue]] = []
        for layer, (past_k, past_v) in zip(self.layers, past_kv, strict=True):
            hidden, key, value = layer.decode(
                hidden, cos=cos, sin=sin, past_k=past_k, past_v=past_v, write_sel=write_sel
            )
            new_kv.append((key, value))
        return self.norm(hidden), new_kv

    def logits(self, normed: TensorValue) -> TensorValue:
        """``lm_head`` over the final row only, in fp32."""
        return ops.cast(self.lm_head(normed[-1:, :]), DType.float32)
