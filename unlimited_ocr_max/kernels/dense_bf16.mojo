"""The dense bf16 projection op: fp32 activation rows against a K-blocked bf16 weight.

``dense_bf16_qmv``: ``x fp32 [M, K]``, ``w bf16 [K / KBLOCK, N, KBLOCK]`` -> ``fp32 [M, N]`` with::

    out[m, n] = sum_kk float(W[n, kk]) * x[m, kk],   W[n, kk] = w[kk // KBLOCK, n, kk % KBLOCK]

fp32 accumulation. It is how the attention projections, the dense FFN, the
shared experts and ``lm_head`` read their bf16 weights at the decode step:
MAX's own matmul takes a bf16 weight against an fp32 activation only through
a weight-only upcast, and MAX folds that upcast into an fp32 copy of the
weight on the device at ``session.load`` (OQ-113-A, KON-238). This op reads
the bf16 bytes, so the shared registry holds those weights in bf16.

The weight is K-blocked: ``W [N, K]``, the checkpoint's ``[out, in]``, cut
into ``KBLOCK``-wide blocks along K, with block ``kb`` of every row side by
side (``weight_adapters.kblocked`` lays it out at load). An item reads its
row one block -- one vector load -- at a time, and the items of a SIMD group
serve adjacent rows, so their loads at each step are adjacent in memory and
coalesce. Against ``[N, K]`` rows (each item one 32-byte load in its own
row) the wide weights read faster at ``B = 1`` on the Apple M4 -- ``lm_head``
3.65-3.82 -> 3.30-3.47 ms, the dense FFN's gate/up 258-261 -> 243-245 us, the
shared experts' 84 -> 77-79 us; the 1280-wide ones and ``B = 8`` about as
before (TODO ID 45) -- with the same sums in the same order, so the same bits.

One ``elementwise`` work item per (column ``n``, tile of ``tile`` rows), the
same code on every device. A hand-launched GPU kernel on this layout (SIMD
groups of 4 to 32 lanes per column, ``moe_bf16.mojo``'s scheme) was measured
for TODO ID 52 and not taken: 0-12 % faster at ``B = 1`` (about 2 % of a
decode step) and up to 85 % slower at ``B = 8``. Item ``i`` is
column ``i // tiles``, rows from ``(i % tiles) * tile``, so the items of one
column are adjacent. An item reads its weight row ONCE for all its rows: with
``tile >= M`` -- the decode step's ``B`` rows on a wide weight -- every weight
row is read once per call. KON-239's lesson holds here too: ``lm_head`` at
``B = 5..7`` takes 3.9-4.6 ms in one tile of ``B`` rows, 6.4-10.0 ms in ``B``
tiles of one row (Apple M4). A narrow weight has too few columns to fill the
device, so the caller trades reads for items there: ``tile`` is a
compile-time parameter (``decoder._rows_per_read`` picks it).

Every item runs ``tile`` rows of arithmetic, the last tile of an ``M`` that
``tile`` does not divide too: its rows at or past ``M`` read row ``M - 1``
and are not stored. So all the items of a call run one code path. An item
that ran only its own row count would split a SIMD group between two
instantiations, run one after the other: ``lm_head`` at ``B = 3`` took
7.0-7.1 ms in tiles of 2 + 1 against 3.7 ms in one tile of 3, the 1280-wide
attention projections at ``B = 7`` 0.11-0.12 ms in tiles of 4 + 3 against
0.065-0.080 ms padded to 4 + 4 (Apple M4, KON-238).

Every row runs the same arithmetic whatever ``tile``, ``M`` or the other
rows: the weight row ``VEC`` bf16 at a time along K (one 32-byte load), times
``VEC`` fp32 values of the row, folded to ``LANES`` partial sums, accumulated,
and reduced once at the end. So a row's bits do not depend on which rows share
its call -- the batched decode's load independence. ``LANES`` (not ``VEC``)
accumulators per row keep a tile of 8 rows in registers.

The load path is the MoE kernels' (``loads.mojo``, :func:`load_path`): the
vector loads claim ``LOAD_ALIGN`` (16 bytes) when both tensors are packed and
both bases lie on it (``ALIGNED``, every served call), else one element
(``UNALIGNED``, the same loop, so the same bits). A non-unit stride along K,
which no vector load fits and no caller makes, and shape mismatches raise
host-side.
"""

from extensibility import InputTensor, OutputTensor, register
from layout import Coord
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.utils import IndexList, StaticTuple

from .loads import ALIGNED, SCALAR, load_alignment, load_path

#: bf16 elements per vector load along K: one 32-byte load. 16 is the width
#: ``moe_bf16.mojo`` measured fastest on the M4; 32 crashes Metal's pipeline
#: compiler on this build.
comptime VEC = 16

#: bf16 elements per K block of the weight: one vector load (``decoder.KBLOCK``).
comptime KBLOCK = VEC

#: Partial sums per row. With ``VEC`` of them a tile of 8 rows holds 128
#: accumulators and ``lm_head`` at ``B = 8`` ran at 11.8 ms against 4.4 ms.
comptime LANES = 4


@always_inline
def _dot_rows[
    tile: Int, aligned: Bool
](
    result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
    x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
    w: InputTensor[dtype = DType.bfloat16, rank=3, static_spec=_],
    n: Int,
    row0: Int,
    rows: Int,
):
    """``result[row0 + r, n] = W[n, :] . x[row0 + r, :]`` for ``r < tile`` and ``row0 + r < rows``, reading the weight row once.

    Always ``tile`` rows of arithmetic: a row at or past ``rows`` (the padding
    of a last, short tile) reads row ``rows - 1`` and is not stored. The
    loads claim ``LOAD_ALIGN`` if ``aligned``, else one element; ``aligned``
    changes the loads, not the sums.
    """
    var blocks = Int(w.dim_size(0))
    var acc = StaticTuple[SIMD[DType.float32, LANES], tile](fill=0)
    for kb in range(blocks):
        var wv = w.load[
            VEC, element_alignment = load_alignment[DType.bfloat16, VEC, aligned]()
        ](IndexList[3](kb, n, 0)).cast[DType.float32]()
        comptime for r in range(tile):
            var xv = x.load[
                VEC, element_alignment = load_alignment[DType.float32, VEC, aligned]()
            ](IndexList[2](min(row0 + r, rows - 1), kb * KBLOCK))
            acc[r] += (wv * xv).reduce_add[LANES]()
    comptime for r in range(tile):
        if row0 + r < rows:
            result.store[1](IndexList[2](row0 + r, n), acc[r].reduce_add())


@register("dense_bf16_qmv")
struct DenseBf16Qmv:
    @staticmethod
    def execute[
        tile: Int, target: StaticString
    ](
        result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
        x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
        w: InputTensor[dtype = DType.bfloat16, rank=3, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime assert tile >= 1, "dense_bf16_qmv: tile must be >= 1"
        var rows = Int(x.dim_size(0))
        var kdim = Int(x.dim_size(1))
        if Int(result.dim_size(0)) != rows:
            raise Error("dense_bf16_qmv: x and output disagree on M")
        if Int(w.dim_size(1)) != Int(result.dim_size(1)):
            raise Error("dense_bf16_qmv: w and output disagree on N")
        if Int(w.dim_size(2)) != KBLOCK:
            raise Error("dense_bf16_qmv: w is not K-blocked by KBLOCK")
        if Int(w.dim_size(0)) * KBLOCK != kdim:
            raise Error("dense_bf16_qmv: w and x disagree on K")

        var path = load_path(KBLOCK, VEC, w, x)
        if path == SCALAR:
            raise Error("dense_bf16_qmv: x or w has a non-unit stride along K")
        var tiles = ceildiv(rows, tile)

        # Every value the items need is captured by value: a runtime scalar a
        # closure reads by reference is garbage on CPU worker threads.
        def item[width: Int, alignment: Int = 1](idx: Coord[...]) {var result, var x, var w, var rows, var tiles, var path}:
            var i = Int(idx[0].value())
            var n = i // tiles
            var row0 = (i - n * tiles) * tile
            if path == ALIGNED:
                _dot_rows[tile, True](result, x, w, n, row0, rows)
            else:
                _dot_rows[tile, False](result, x, w, n, row0, rows)

        elementwise[simd_width=1, target=target](item, (tiles * Int(w.dim_size(1)),), ctx)
