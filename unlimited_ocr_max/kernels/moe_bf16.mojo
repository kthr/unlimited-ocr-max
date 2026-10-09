"""The bf16 MoE expert op: the decode GEMV over the whole expert stack.

``moe_bf16_qmv``: ``x fp32 [k, K]``, ``expert_ids int32 [k]``,
``w bf16 [E, N, K]`` -> ``fp32 [k, N]`` with ``e_s = expert_ids[s]`` and::

    out[s, n] = sum_kk float(w[e_s, n, kk]) * x[s, kk]

fp32 accumulation. The op takes the FULL stack and indexes the expert inside
the kernel, like ``moe_int8_qmv`` (``moe_int8.mojo``): weight-only expressions
are compile-folded and the folded result is materialized on the device, so a
graph-level slice of the stack is not an option. ``E`` comes from
``w.dim(0)``, never from a constant.

Expert ids are runtime device data, readable only inside the kernel, and a
kernel cannot raise -- so out-of-range ids are CLAMPED to ``[0, E)`` instead
of aborting. Shape mismatches still raise host-side.

Each selected expert is read ONCE per tile of rows, however many rows it
serves (KON-239): the batched decode step's ``B`` tokens pick overlapping
experts, and reading an expert once per (row, expert) pair made ``B = 8``
slower than MAX's grouped matmul. Rows group by their raw id
(``moe_routing.mojo``): a row works only when it starts a tile of the rows
carrying its id; the other rows return.

Two kernels compute the op, one per kind of device:

* **GPU** (TODO ID 52): a hand-launched kernel, one SIMD group per output
  column ``n`` and tile of up to ``GPU_TILE`` rows. Its lanes stride the
  weight row ``GPU_VEC`` bf16 at a time (adjacent lanes read adjacent memory),
  each into its own ``GPU_VEC``-wide fp32 partial per row; the partials meet
  in a SIMD-group sum. A row whose rank among the rows carrying its id is a
  multiple of ``GPU_TILE`` serves itself and the next rows of its group, so a
  large group runs as several SIMD groups side by side. The lanes scan the
  ids together, ``WARP_SIZE`` per load. Against the elementwise kernel below
  (one work item per output) it reads the stacks 12-21 % faster on the Apple
  M4 at ``B = 1`` and ``B = 8`` (EXPERIMENTS.md, ID 52), and it sums in
  another order, so its bits differ from the CPU's. It needs both tensors
  packed and on ``LOAD_ALIGN`` (``ALIGNED``), as every device buffer is; any
  other call runs the elementwise kernel on the GPU too.
* **CPU** (and the GPU fallback): ``elementwise``, one work item per output
  ``(s, n)``. Only the item of a row that leads its id's group does the work:
  it reads ``w[e, n, :]`` once per tile of up to ``TILE`` such rows and writes
  every row of the tile. A row alone on its expert -- every row at ``B = 1``
  -- goes straight to its dot product. A SIMD group's items share ``s``, so
  they all work or all return. The weight row is read ``VEC`` bf16 at a time
  along K (one 32-byte load) into ``VEC`` fp32 partial sums, reduced once at
  the end. Each tile size runs a body specialised for it, so no slot is tested
  inside the K loop. A K that is not a multiple of ``VEC`` takes the loop one
  element at a time (``SCALAR``: one fp32 sum per row, another summation
  order), so a vector load never straddles a row; the port's K (1280, 896)
  never does.

In both kernels every row keeps its own accumulators and its own arithmetic,
so a row's bits do not depend on which rows share its expert or its call --
the batched decode's load independence.

The elementwise vector loads claim ``LOAD_ALIGN`` (16 bytes, ``loads.mojo``)
as their ``element_alignment`` when ``execute`` finds, host-side, both tensors
packed row-major and both bases on 16 bytes (``load_path``, ``ALIGNED``):
then each load reads a multiple of ``VEC`` elements from its tensor's base.
Otherwise the same ``VEC``-wide loop runs with loads that claim one element
(``UNALIGNED``), so the bits do not depend on where the buffers lie. Without
a claim a load may be split: on the M4 that halves the int8 kernel's speed,
though not this one's.

The shapes reach the elementwise work items as VALUE captures: ``execute``
derives them once, host-side, and the closure -- passed to ``elementwise`` as
a runtime argument -- copies them (``{var ...}``). A ``@parameter`` closure
captures BY REFERENCE, and a host scalar read through such a capture is
garbage in the work items at every size tried, 1024 to 4 Mi items: zero on
the M4's GPU, through ``elementwise`` and ``foreach`` alike, and NaN or zero
on the CPU through ``foreach`` (``tests/test_kernel_capture.py`` checks the
value form). The kernels used to dodge it by re-deriving every scalar from the
captured tensors inside the closure, blaming large tensors split across CPU
worker threads; the cause is the capture by reference. The GPU kernel takes
its shapes as launch arguments.
"""

from extensibility import InputTensor, OutputTensor, register
from layout import Coord
from max.algorithm.functional import elementwise
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp
from std.math import ceildiv
from std.utils import IndexList, StaticTuple

from .loads import ALIGNED, LOAD_ALIGN, UNALIGNED, load_alignment, load_path
from .moe_routing import (
    FOLLOWS,
    clamped_expert,
    group_tail,
    next_tile,
    routed_id,
)

#: bf16 elements per vector load along K. On the M4, 16 reads the stacks at
#: 85-95 GB/s against 68-90 for 8; 32 crashes Metal's pipeline compiler
#: (XPC_ERROR_CONNECTION_INTERRUPTED) on this build, aligned loads or not.
comptime VEC = 16

#: Rows served per read of a weight row. A group is at most ``B`` rows (a
#: token's top-k experts are distinct), and 4 holds almost every group at
#: ``B = 8``; a larger group reads the row once per tile. On the M4 both 2
#: and 8 are slower at ``B = 8``: 2 re-reads more rows, and with 8 each slot
#: is a ``VEC``-lane accumulator whose registers cost occupancy.
comptime TILE = 4

#: GPU kernel: bf16 elements per lane per load (16 bytes; 16 is no faster on
#: the M4), SIMD groups per threadgroup (2 and 8 were no faster), and rows per
#: SIMD group (2 is as fast as 4 at ``B = 8`` with random routing, and 4-10 %
#: faster when eight rows share each expert).
comptime GPU_VEC = 8
comptime GPU_WARPS = 4
comptime GPU_TILE = 2


@always_inline
def _dot_rows_by[
    count: Int, width: Int, aligned: Bool
](
    result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
    x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
    w: InputTensor[dtype = DType.bfloat16, rank=3, static_spec=_],
    kdim: Int,
    expert: Int,
    n: Int,
    rows: StaticTuple[Int, TILE],
):
    """``result[rows[r], n] = w[expert, n, :] . x[rows[r], :]`` for ``r < count``, ``width`` elements per load.

    ``kdim % width == 0``. The loads claim ``LOAD_ALIGN`` if ``aligned`` (the
    ``ALIGNED`` path's preconditions hold), else one element; ``aligned``
    changes the loads, not the sums.
    """
    var acc = StaticTuple[SIMD[DType.float32, width], count](fill=0)
    for kk in range(0, kdim, width):
        var wv = w.load[
            width,
            element_alignment = load_alignment[DType.bfloat16, width, aligned](),
        ](IndexList[3](expert, n, kk)).cast[DType.float32]()
        comptime for r in range(count):
            acc[r] += wv * x.load[
                width,
                element_alignment = load_alignment[DType.float32, width, aligned](),
            ](IndexList[2](rows[r], kk))
    comptime for r in range(count):
        result.store[1](IndexList[2](rows[r], n), acc[r].reduce_add())


@always_inline
def _dot_rows[
    count: Int
](
    result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
    x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
    w: InputTensor[dtype = DType.bfloat16, rank=3, static_spec=_],
    kdim: Int,
    path: Int,
    expert: Int,
    n: Int,
    rows: StaticTuple[Int, TILE],
):
    """``result[rows[r], n]`` for ``r < count`` from one read of ``w[expert, n, :]``: ``VEC`` wide on the ``ALIGNED`` and ``UNALIGNED`` paths (the same sums), else one element at a time."""
    if path == ALIGNED:
        _dot_rows_by[count, VEC, True](result, x, w, kdim, expert, n, rows)
    elif path == UNALIGNED:
        _dot_rows_by[count, VEC, False](result, x, w, kdim, expert, n, rows)
    else:
        _dot_rows_by[count, 1, False](result, x, w, kdim, expert, n, rows)


@always_inline
def _gpu_dot_rows[
    count: Int
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    row: UnsafePointer[BFloat16, MutAnyOrigin],
    rows: StaticTuple[Int, GPU_TILE],
    n: Int,
    nd: Int,
    kdim: Int,
    lane: Int,
):
    """``out[rows[r], n] = row . x[rows[r], :]`` for ``r < count``: one SIMD group, lane ``l`` from element ``l * GPU_VEC`` in strides of ``WARP_SIZE * GPU_VEC``.

    ``kdim % GPU_VEC == 0`` and both tensors packed on ``LOAD_ALIGN``, so every
    load starts a multiple of 16 bytes from its tensor's base.
    """
    var acc = StaticTuple[SIMD[DType.float32, GPU_VEC], count](fill=0)
    var kk = lane * GPU_VEC
    while kk < kdim:
        var wv = row.load[width=GPU_VEC, alignment=LOAD_ALIGN](kk).cast[
            DType.float32
        ]()
        comptime for r in range(count):
            acc[r] += wv * (x_ptr + rows[r] * kdim).load[
                width=GPU_VEC, alignment=LOAD_ALIGN
            ](kk)
        kk += WARP_SIZE * GPU_VEC
    comptime for r in range(count):
        var total = warp.sum(acc[r].reduce_add())
        if lane == 0:
            out_ptr[rows[r] * nd + n] = total


def _gpu_qmv(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    w_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    row_count: Int32,
    expert_count: Int32,
    n_dim: Int32,
    k_dim: Int32,
):
    """The GPU kernel: SIMD group ``thread_idx.x // WARP_SIZE`` of threadgroup ``(bx, s)`` computes column ``bx * GPU_WARPS + that`` for the tile row ``s`` starts, if it starts one.

    The shapes arrive as ``Int32``: a launch argument must have a fixed width.
    """
    var rows = Int(row_count)
    var num_experts = Int(expert_count)
    var nd = Int(n_dim)
    var kdim = Int(k_dim)
    var s = Int(block_idx.y)
    var n = Int(block_idx.x) * GPU_WARPS + Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    if n >= nd:
        return
    # The rows carrying s's id before it (its rank) and after it, the lanes
    # comparing WARP_SIZE ids per step.
    var key = ids_ptr[s]
    var before = Int32(0)
    var after = Int32(0)
    for base in range(0, rows, WARP_SIZE):
        var j = base + lane
        if j < rows and ids_ptr[j] == key:
            if j < s:
                before += 1
            elif j > s:
                after += 1
    if Int(warp.sum(before)) % GPU_TILE != 0:
        return
    var count = min(Int(warp.sum(after)) + 1, GPU_TILE)
    # The tile's next rows: each the smallest row past the previous one that carries the id.
    var tile_rows = StaticTuple[Int, GPU_TILE](fill=s)
    var prev = s
    comptime for r in range(1, GPU_TILE):
        if r < count:
            var next = Int32(rows)
            for base in range(0, rows, WARP_SIZE):
                var j = base + lane
                if j < rows and j > prev and ids_ptr[j] == key:
                    next = min(next, Int32(j))
            prev = Int(warp.min(next))
            tile_rows[r] = prev
    var row = w_ptr + (clamped_expert(key, num_experts) * nd + n) * kdim
    comptime for c in range(1, GPU_TILE + 1):
        if count == c:
            _gpu_dot_rows[c](out_ptr, x_ptr, row, tile_rows, n, nd, kdim, lane)


@register("moe_bf16_qmv")
struct MoeBf16Qmv:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor[dtype = DType.float32, rank=2, static_spec=_],
        x: InputTensor[dtype = DType.float32, rank=2, static_spec=_],
        expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        w: InputTensor[dtype = DType.bfloat16, rank=3, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        if Int(x.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_bf16_qmv: x and output disagree on k")
        if Int(expert_ids.dim_size(0)) != Int(result.dim_size(0)):
            raise Error("moe_bf16_qmv: expert_ids and output disagree on k")
        if Int(w.dim_size(1)) != Int(result.dim_size(1)):
            raise Error("moe_bf16_qmv: w and output disagree on N")
        if Int(w.dim_size(2)) != Int(x.dim_size(1)):
            raise Error("moe_bf16_qmv: w and x disagree on K")
        if Int(w.dim_size(0)) < 1:
            raise Error("moe_bf16_qmv: the expert stack is empty")

        var rows = Int(result.dim_size(0))
        var nd = Int(result.dim_size(1))
        var kdim = Int(w.dim_size(2))
        var num_experts = Int(w.dim_size(0))
        var path = load_path(kdim, VEC, w, x)

        comptime if target != "cpu":
            # ALIGNED implies kdim % VEC == 0, so kdim % GPU_VEC == 0 too.
            if path == ALIGNED and rows > 0:
                ctx.enqueue_function[_gpu_qmv](
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](result.unsafe_ptr()),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](x.unsafe_ptr()),
                    rebind[UnsafePointer[Int32, MutAnyOrigin]](expert_ids.unsafe_ptr()),
                    rebind[UnsafePointer[BFloat16, MutAnyOrigin]](w.unsafe_ptr()),
                    Int32(rows),
                    Int32(num_experts),
                    Int32(nd),
                    Int32(kdim),
                    grid_dim=(ceildiv(nd, GPU_WARPS), rows),
                    block_dim=GPU_WARPS * WARP_SIZE,
                )
                return

        @always_inline
        def qmv[
            width: Int, alignment: Int = 1
        ](idx: Coord[...]) {
            var result,
            var x,
            var expert_ids,
            var w,
            var rows,
            var nd,
            var kdim,
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
                    result, x, w, kdim, path, e, n, StaticTuple[Int, TILE](fill=s)
                )
                return
            var cursor = s
            while cursor < rows:
                var tile_rows = StaticTuple[Int, TILE](fill=0)
                var count = next_tile(expert_ids, key, rows, cursor, tile_rows)
                comptime for c in range(1, TILE + 1):
                    if count == c:
                        _dot_rows[c](result, x, w, kdim, path, e, n, tile_rows)

        elementwise[simd_width=1, target=target](qmv, (rows * nd,), ctx)
