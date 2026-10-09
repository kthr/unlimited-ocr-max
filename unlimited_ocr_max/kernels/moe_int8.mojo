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
formula's association. Like ``moe_bf16_qmv``'s elementwise kernel
(``moe_bf16.mojo``, which explains the scheme) it reads each selected expert
ONCE per call: only the first row carrying an expert id works, serving every
row of a tile of up to ``TILE`` rows carrying it from one read of the weight
row and its scales (``moe_routing.mojo``). It runs on every device: a
hand-launched GPU kernel like ``moe_bf16_qmv``'s, whole SIMD groups or lane
groups per column, was no faster on the Apple M4 and up to 2.5x slower
(TODO ID 52).

A group is LOADED ``VEC`` int8 at a time (one 16-byte load, with ``VEC`` fp32
of ``x`` against it) but SUMMED one element at a time, lane by lane in ``kk``
order into one fp32 group sum, and each row keeps its own sums: the operation
order of the one-element kernel at 6e190af, so the int8 decode bits are
unchanged since then and do not depend on which rows share an expert
(``tests/test_kernels_int8.py`` pins them to a reference in that order). A
group size that is not a multiple of ``VEC`` takes the same loop one element
at a time (``SCALAR``): same sums, same bits. KON-240 first summed a group into ``VEC``
lane-wise partial sums reduced at the group's end -- another order, so other
int8 decode bits (a re-pin of the int8 transcripts) -- and with real weights
it measured +3.1 % on the B = 8 decode step and nothing above the ~2 % draw
spread at B = 1, so that was dropped. The vector LOADS stay because inside a
value-capture closure the one-element loads run 3x slower on the M4 (1.49 vs
0.49 ms per k = 48 call at the real shape; through raw pointers still 2x; only
the deprecated by-reference ``@parameter`` closure ran them at the base's
speed), while ``VEC``-wide loads with the same sums run at the base's speed.

``int8_dequant_expert``: ``expert_idx int32 [1]`` (a scalar carried as a
one-element tensor, the ``ngram`` idiom) → ``[N, K]`` in the output's dtype
(bf16 in the port) with::

    out[n, kk] = result.dtype(float(w[e, n, kk]) * scales[e, n, kk // G])

computed at the target's SIMD width: a lane-wise multiply and cast, so the
same bits as one element at a time.

Expert ids are runtime device data, readable only inside the kernels' closures,
and a closure cannot raise -- so out-of-range ids are CLAMPED to ``[0, E)``
instead of aborting. Shape mismatches still raise host-side.

Vector loads claim ``LOAD_ALIGN`` (``loads.mojo``) when ``execute``
finds both tensors packed and both bases aligned (``ALIGNED``), else one
element (``UNALIGNED``) -- without the claim a 16-wide load may be split and
the qmv runs at half speed on the M4; the sums, and so the bits, are the same
on all three paths -- and the shapes reach the closures as VALUE captures: see
``moe_bf16.mojo`` for both, and for why a by-reference capture of a host
scalar is garbage in the work items. ``int8_dequant_expert`` claims the
alignment under the same two conditions.
"""

from extensibility import InputTensor, OutputTensor, foreach, register
from layout import Coord
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.utils import IndexList, StaticTuple

from .loads import (
    ALIGNED,
    LOAD_ALIGN,
    UNALIGNED,
    load_alignment,
    load_path,
    packed,
)
from .moe_routing import (
    FOLLOWS,
    clamped_expert,
    group_tail,
    next_tile,
    routed_id,
)

#: int8 weights per vector load along a group: 16 bytes, with ``VEC`` fp32
#: lanes of ``x`` against them. The loads are vectors; the sums are not
#: (module docstring). 32 crashes Metal's pipeline compiler as it does for
#: ``moe_bf16_qmv``.
comptime VEC = 16

#: Rows served per read of a weight row; see ``moe_bf16.mojo``. The int8
#: accumulators are scalars, so registers do not cap the tile here, but 8 is
#: no faster than 4 at ``B = 8`` on the M4 and slower at ``B = 1``.
comptime TILE = 4


@always_inline
def _dot_rows_by[
    count: Int, width: Int, aligned: Bool
](
    result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
    x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
    w: InputTensor[dtype = DType.int8, rank=3, static_spec=_],
    scales: InputTensor[dtype = DType.float32, rank=3, static_spec=_],
    groups: Int,
    gsize: Int,
    expert: Int,
    n: Int,
    rows: StaticTuple[Int, TILE],
):
    """``result[rows[r], n]`` for ``r < count``, ``width`` elements per load, summed one at a time.

    ``gsize % width == 0``. The loads claim ``LOAD_ALIGN`` if ``aligned`` (the
    ``ALIGNED`` path's preconditions hold), else one element. The sums are the
    one-element kernel's: ``gsum = fma(w, x, gsum)`` lane by lane in ``kk``
    order, then ``acc = fma(scale, gsum, acc)`` per group. The fused
    multiply-adds are explicit because that is what the base's ``+=`` of a
    product compiled to on the CPU and the M4 alike, while a lane-wise ``gsum
    += wv[lane] * xv[lane]`` left to the compiler fuses on the M4 but not on
    the CPU (41-88 % of the outputs off in the low bits, measured). ``width``
    and ``aligned`` change the loads, not the bits.
    """
    var acc = StaticTuple[Float32, count](fill=0)
    for g in range(groups):
        var base = g * gsize
        var gsum = StaticTuple[Float32, count](fill=0)
        for kk in range(base, base + gsize, width):
            var wv = w.load[
                width,
                element_alignment = load_alignment[DType.int8, width, aligned](),
            ](IndexList[3](expert, n, kk)).cast[DType.float32]()
            comptime for r in range(count):
                var xv = x.load[
                    width,
                    element_alignment = load_alignment[
                        DType.float32, width, aligned
                    ](),
                ](IndexList[2](rows[r], kk))
                comptime for lane in range(width):
                    gsum[r] = wv[lane].fma(xv[lane], gsum[r])
        var scale = scales.load[1](IndexList[3](expert, n, g))[0]
        comptime for r in range(count):
            acc[r] = scale.fma(gsum[r], acc[r])
    comptime for r in range(count):
        result.store[1](IndexList[2](rows[r], n), SIMD[DType.float32, 1](acc[r]))


@always_inline
def _dot_rows[
    count: Int
](
    result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
    x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
    w: InputTensor[dtype = DType.int8, rank=3, static_spec=_],
    scales: InputTensor[dtype = DType.float32, rank=3, static_spec=_],
    groups: Int,
    gsize: Int,
    path: Int,
    expert: Int,
    n: Int,
    rows: StaticTuple[Int, TILE],
):
    """``result[rows[r], n]`` for ``r < count`` from one read of ``w[expert, n, :]`` and its scales: ``VEC``-wide loads on the ``ALIGNED`` and ``UNALIGNED`` paths, else one element at a time; the same sums on all three."""
    if path == ALIGNED:
        _dot_rows_by[count, VEC, True](
            result, x, w, scales, groups, gsize, expert, n, rows
        )
    elif path == UNALIGNED:
        _dot_rows_by[count, VEC, False](
            result, x, w, scales, groups, gsize, expert, n, rows
        )
    else:
        _dot_rows_by[count, 1, False](
            result, x, w, scales, groups, gsize, expert, n, rows
        )


@register("moe_int8_qmv")
struct MoeInt8Qmv:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
        x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
        expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        w: InputTensor[dtype = DType.int8, rank=3, static_spec=_],
        scales: InputTensor[dtype = DType.float32, rank=3, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        if Int(x.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_int8_qmv: x and output disagree on k")
        if Int(expert_ids.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_int8_qmv: expert_ids and output disagree on k")
        if Int(w.dim_size(1)) != Int(result.dim_size(1)):
            raise Error("moe_int8_qmv: w and output disagree on N")
        if Int(w.dim_size(2)) != Int(x.dim_size(1)):
            raise Error("moe_int8_qmv: w and x disagree on K")
        if Int(w.dim_size(0)) < 1:
            raise Error("moe_int8_qmv: the expert stack is empty")
        if Int(scales.dim_size(0)) != Int(w.dim_size(0)):
            raise Error("moe_int8_qmv: scales and w disagree on E")
        if Int(scales.dim_size(1)) != Int(w.dim_size(1)):
            raise Error("moe_int8_qmv: scales and w disagree on N")
        var groups = Int(scales.dim_size(2))
        if groups < 1 or Int(w.dim_size(2)) % groups != 0:
            raise Error("moe_int8_qmv: K must be a multiple of the group count")

        var rows = Int(result.dim_size(0))
        var nd = Int(result.dim_size(1))
        var gsize = Int(w.dim_size(2)) // groups
        var num_experts = Int(w.dim_size(0))
        var path = load_path(gsize, VEC, w, x)

        @always_inline
        def qmv[
            width: Int, alignment: Int = 1
        ](idx: Coord[...]) {
            var result,
            var x,
            var expert_ids,
            var w,
            var scales,
            var rows,
            var nd,
            var groups,
            var gsize,
            var num_experts,
            var path,
        }:
            var i = Int(idx[0].value())
            var s = i // nd
            var n = i - s * nd
            var later = group_tail(expert_ids, s, rows)
            if later == FOLLOWS:
                return
            var key = routed_id(expert_ids, s)
            var e = clamped_expert(key, num_experts)
            if later == 0:  # alone on its expert, as every row is at B = 1
                _dot_rows[1](
                    result, x, w, scales, groups, gsize, path, e, n,
                    StaticTuple[Int, TILE](fill=s),
                )
                return
            var cursor = s
            while cursor < rows:
                var tile_rows = StaticTuple[Int, TILE](fill=0)
                var count = next_tile(expert_ids, key, rows, cursor, tile_rows)
                comptime for c in range(1, TILE + 1):
                    if count == c:
                        _dot_rows[c](
                            result, x, w, scales, groups, gsize, path, e, n, tile_rows
                        )

        elementwise[simd_width=1, target=target](qmv, (rows * nd,), ctx)


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
        if Int(w.dim_size(0)) < 1:
            raise Error("int8_dequant_expert: the expert stack is empty")
        if Int(scales.dim_size(0)) != Int(w.dim_size(0)):
            raise Error("int8_dequant_expert: scales and w disagree on E")
        if Int(scales.dim_size(1)) != Int(w.dim_size(1)):
            raise Error("int8_dequant_expert: scales and w disagree on N")
        if Int(expert_idx.dim_size(0)) != 1:
            raise Error("int8_dequant_expert: `expert_idx` must be a single int32")
        var groups = Int(scales.dim_size(2))
        if groups < 1 or Int(w.dim_size(2)) % groups != 0:
            raise Error(
                "int8_dequant_expert: K must be a multiple of the group count"
            )

        var kdim = Int(w.dim_size(2))
        var gsize = kdim // groups
        var num_experts = Int(w.dim_size(0))
        var aligned = packed(w) and Int(w.unsafe_ptr()) % LOAD_ALIGN == 0

        def dequant[
            width: Int
        ](idx: Coord[...]) {
            var w,
            var scales,
            var expert_idx,
            var kdim,
            var gsize,
            var num_experts,
            var aligned,
        } -> SIMD[result.dtype, width]:
            var e = clamped_expert(
                expert_idx.load[1](IndexList[1](0))[0], num_experts
            )
            var n = Int(idx[0].value())
            var k0 = Int(idx[1].value())
            # `foreach` hands a width > 1 only to a pack inside one row that
            # starts a multiple of `width` elements into the flattened
            # tensor, so with K % width == 0 its offset in a packed `w` is
            # one too.
            var wv: SIMD[DType.int8, width]
            if aligned and kdim % width == 0:
                wv = w.load[
                    width,
                    element_alignment = load_alignment[DType.int8, width](),
                ](IndexList[3](e, n, k0))
            else:
                wv = w.load[width](IndexList[3](e, n, k0))
            var group = k0 // gsize
            var sv = SIMD[DType.float32, width](
                scales.load[1](IndexList[3](e, n, group))[0]
            )
            if (k0 + width - 1) // gsize != group:  # the pack spans groups
                comptime for lane in range(1, width):
                    sv[lane] = scales.load[1](
                        IndexList[3](e, n, (k0 + lane) // gsize)
                    )[0]
            return (wv.cast[DType.float32]() * sv).cast[result.dtype]()

        foreach[target=target](dequant, result, ctx)
