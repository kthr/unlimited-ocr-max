"""Row grouping shared by the qmv kernels: which rows of one call read the same expert.

A qmv op gets ``k`` (token, expert) rows and ``expert_ids int32 [k]``; at the
batched decode step several tokens pick the same expert. To read each selected
expert ONCE per call, a kernel's work item for row ``s`` runs only when ``s``
is the first row carrying its expert id -- the group's LEADER -- and then
serves the group's rows, ascending, in tiles of up to ``tile`` rows (one
weight read per tile). A row alone on its expert -- every row at ``B = 1``,
where a token's top-k are distinct -- skips the tile gathering.

Rows group by their RAW id. An out-of-range id is clamped to ``[0, E)`` only
for the read (a kernel cannot raise), so it and the in-range id it clamps to
read the same expert in two groups: still correct, one extra read, and the
router never emits such an id. Comparing raw ids keeps the per-row scan, which
every work item runs, to a load and a compare per id.

The row count ``total`` (``expert_ids.dim(0)``) is an argument: the callers
compute it host-side and capture it into their closures by value (see
``moe_bf16.mojo``).
"""

from extensibility import InputTensor
from std.utils import IndexList, StaticTuple

#: ``group_tail`` of a row that does not lead its group.
comptime FOLLOWS = -1


@always_inline
def routed_id(
    expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_], row: Int
) -> Int32:
    """``expert_ids[row]`` as given, the key rows group by."""
    return expert_ids.load[1](IndexList[1](row))[0]


@always_inline
def clamped_expert(key: Int32, num_experts: Int) -> Int:
    """The expert an id reads: ``key`` clamped to ``[0, num_experts)``."""
    var e = Int(key)
    if e < 0:
        e = 0
    if e >= num_experts:
        e = num_experts - 1
    return e


@always_inline
def group_tail(
    expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
    row: Int,
    total: Int,
) -> Int:
    """How many rows after ``row`` of the ``total`` carry its expert id, or ``FOLLOWS`` if a row before it does.

    One pass over the ids, one at a time.
    """
    var mine = routed_id(expert_ids, row)
    var later = 0
    for j in range(total):
        if routed_id(expert_ids, j) == mine:
            if j < row:
                return FOLLOWS
            if j > row:
                later += 1
    return later


@always_inline
def next_tile[
    tile: Int
](
    expert_ids: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
    key: Int32,
    total: Int,
    mut cursor: Int,
    mut rows: StaticTuple[Int, tile],
) -> Int:
    """Fill ``rows`` with the next rows at or after ``cursor`` (of the ``total``) carrying ``key``; return how many (0 when done).

    ``cursor`` advances past the last row taken. Slots past the count keep
    their previous value and must not be read.
    """
    var count = 0
    comptime for r in range(tile):
        while cursor < total and routed_id(expert_ids, cursor) != key:
            cursor += 1
        if cursor < total:
            rows[r] = cursor
            count = r + 1
            cursor += 1
    return count
