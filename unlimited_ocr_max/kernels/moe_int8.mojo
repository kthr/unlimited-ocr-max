"""Int8 grouped-quant MoE expert ops: decode GEMV and prefill dequantize.

Both ops take the FULL expert stacks -- ``w int8 [E, N, K]`` next to
``scales fp32 [E, N, K/G]`` -- and index the expert *inside* the kernel.
Slicing the weight at graph level is not an option on this build: weight-only
expressions are compile-folded and the folded result is materialized on the
device, so the stacks must reach an opaque op whole. ``G`` is the group SIZE,
derived as ``K // scales.dim(2)``; ``E`` comes from ``w.dim(0)``, never from a
constant.

``moe_int8_qmv``: ``x fp32 [k, K]``, ``expert_ids int32 [k]`` →
``fp32 [k, N]`` with ``e_s = expert_ids[s]`` and::

    out[s, n] = sum_g scales[e_s, n, g] * sum_{kk in group g} float(w[e_s, n, kk]) * x[s, kk]

fp32 accumulation, group sum first and the group scale after, matching the
formula's association.

``int8_dequant_expert``: ``expert_idx int32 [1]`` (a scalar carried as a
one-element tensor, the ``ngram`` idiom) → ``[N, K]`` in the output's dtype
(bf16 in the port) with::

    out[n, kk] = result.dtype(float(w[e, n, kk]) * scales[e, n, kk // G])

Expert ids are runtime device data, readable only inside the ``foreach``
closure, and a closure cannot raise -- so out-of-range ids are CLAMPED to
``[0, E)`` instead of aborting. Shape mismatches still raise host-side.
"""

from extensibility import InputTensor, OutputTensor, foreach, register
from layout import Coord
from max.gpu.host import DeviceContext
from std.utils import IndexList


@register("moe_int8_qmv")
struct MoeInt8Qmv:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor,
        x: InputTensor[dtype = result.dtype, rank=2, static_spec=_],
        expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        w: InputTensor[dtype = DType.int8, rank=3, static_spec=_],
        scales: InputTensor[dtype = result.dtype, rank=3, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime assert result.rank == 2, "moe_int8_qmv output is rank-2 [k, N]"
        comptime assert (
            result.dtype.is_floating_point()
        ), "moe_int8_qmv accumulates in floating point"

        if Int(x.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_int8_qmv: x and output disagree on k")
        if Int(expert_ids.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_int8_qmv: expert_ids and output disagree on k")
        if Int(w.dim_size(1)) != Int(result.dim_size(1)):
            raise Error("moe_int8_qmv: w and output disagree on N")
        if Int(w.dim_size(2)) != Int(x.dim_size(1)):
            raise Error("moe_int8_qmv: w and x disagree on K")
        if Int(scales.dim_size(0)) != Int(w.dim_size(0)):
            raise Error("moe_int8_qmv: scales and w disagree on E")
        if Int(scales.dim_size(1)) != Int(w.dim_size(1)):
            raise Error("moe_int8_qmv: scales and w disagree on N")
        var groups_host = Int(scales.dim_size(2))
        if groups_host < 1 or Int(w.dim_size(2)) % groups_host != 0:
            raise Error("moe_int8_qmv: K must be a multiple of the group count")

        @parameter
        def qmv[width: Int](idx: Coord[...]) -> SIMD[result.dtype, width]:
            # Every scalar is re-derived from the captured tensors: a runtime
            # scalar captured into a `foreach` closure is garbage on CPU worker
            # threads once the tensor is large enough to be split across them.
            var kdim = Int(w.dim_size(2))
            var groups = Int(scales.dim_size(2))
            var gsize = kdim // groups
            var num_experts = Int(w.dim_size(0))
            var s = Int(idx[0].value())
            var n0 = Int(idx[1].value())
            var e = Int(expert_ids.load[1](IndexList[1](s))[0])
            # Bound-check: clamp, because a closure cannot raise (see above).
            if e < 0:
                e = 0
            if e >= num_experts:
                e = num_experts - 1
            var out_v = SIMD[result.dtype, width](0)
            for lane in range(width):
                var n = n0 + lane
                var acc = SIMD[result.dtype, 1](0)
                for g in range(groups):
                    var base = g * gsize
                    var gsum = SIMD[result.dtype, 1](0)
                    for kk in range(gsize):
                        var wv = w.load[1](IndexList[3](e, n, base + kk))[0]
                        var xv = x.load[1](IndexList[2](s, base + kk))[0]
                        gsum += wv.cast[result.dtype]() * xv
                    acc += scales.load[1](IndexList[3](e, n, g))[0] * gsum
                out_v[lane] = acc[0]
            return out_v

        foreach[qmv, target=target, simd_width=1](result, ctx)


@register("int8_dequant_expert")
struct Int8DequantExpert:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor,
        w: InputTensor[dtype = DType.int8, rank=3, static_spec=_],
        scales: InputTensor[dtype = DType.float32, rank=3, static_spec=_],
        expert_idx: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            result.rank == 2
        ), "int8_dequant_expert output is rank-2 [N, K]"
        comptime assert (
            result.dtype.is_floating_point()
        ), "int8_dequant_expert dequantizes to floating point"

        if Int(w.dim_size(1)) != Int(result.dim_size(0)):
            raise Error("int8_dequant_expert: w and output disagree on N")
        if Int(w.dim_size(2)) != Int(result.dim_size(1)):
            raise Error("int8_dequant_expert: w and output disagree on K")
        if Int(scales.dim_size(0)) != Int(w.dim_size(0)):
            raise Error("int8_dequant_expert: scales and w disagree on E")
        if Int(scales.dim_size(1)) != Int(w.dim_size(1)):
            raise Error("int8_dequant_expert: scales and w disagree on N")
        if Int(expert_idx.dim_size(0)) != 1:
            raise Error("int8_dequant_expert: `expert_idx` must be a single int32")
        var groups_host = Int(scales.dim_size(2))
        if groups_host < 1 or Int(w.dim_size(2)) % groups_host != 0:
            raise Error(
                "int8_dequant_expert: K must be a multiple of the group count"
            )

        @parameter
        def dequant[width: Int](idx: Coord[...]) -> SIMD[result.dtype, width]:
            # Every scalar is re-derived from the captured tensors; see
            # `moe_int8_qmv` above for why.
            var kdim = Int(w.dim_size(2))
            var groups = Int(scales.dim_size(2))
            var gsize = kdim // groups
            var num_experts = Int(w.dim_size(0))
            var e = Int(expert_idx.load[1](IndexList[1](0))[0])
            # Bound-check: clamp, because a closure cannot raise (see above).
            if e < 0:
                e = 0
            if e >= num_experts:
                e = num_experts - 1
            var n = Int(idx[0].value())
            var k0 = Int(idx[1].value())
            var out_v = SIMD[result.dtype, width](0)
            for lane in range(width):
                var kk = k0 + lane
                var wv = w.load[1](IndexList[3](e, n, kk))[0]
                var sv = scales.load[1](IndexList[3](e, n, kk // gsize))[0]
                out_v[lane] = (wv.cast[DType.float32]() * sv).cast[
                    result.dtype
                ]()[0]
            return out_v

        foreach[dequant, target=target, simd_width=1](result, ctx)
