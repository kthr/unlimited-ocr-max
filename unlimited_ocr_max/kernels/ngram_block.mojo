"""Windowed no-repeat-n-gram logit blocking as a MAX custom op, one row or ``B``.

Port of the reference's ``SlidingWindowNoRepeatNgramProcessor``, per row::

    search_start = max(0, len(sequence) - window)
    search_end   = len(sequence) - ngram_size + 1
    if ngram_size > 1:
        current_prefix = tuple(sequence[-(ngram_size - 1):])
    else:
        current_prefix = tuple()
    for idx in range(search_start, search_end):
        ngram = sequence[idx : idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            scores[ngram[-1]] = -inf

(At ``ngram_size == 1`` the prefix is empty, so every id in the window is
banned; ``sequence[-0:]`` would be the whole sequence.)

One op, generic over rank:

- rank 1 -- ``logits (V,)``, ``history (H,) int32``: one row (the prefill
  guard, ``NgramBlocker``);
- rank 2 -- ``logits (B, V)``, ``history (B, H) int32``: ``B`` rows in one
  op (the decode graphs). Row ``b`` of the output reads row ``b`` of the
  logits and row ``b`` of the history, nothing else.

``history`` is the already windowed tail of the whole sequence, prompt
included, so every position the reference's loop touches is present;
``ngram (1,) int32`` is the n-gram size, shared by every row. ``ngram < 1``,
or a history shorter than ``ngram``, returns the logits unchanged. Banned ids
are set to ``BLOCKED``, a finite value that is argmax- and softmax-equivalent
to ``-inf`` and cannot produce a NaN.

The history length reaches the work items as a VALUE capture of the
closure ``foreach`` takes as a runtime argument; the n-gram size is device
data, read inside. A host scalar captured BY REFERENCE (a ``@parameter``
closure) is garbage in ``foreach``'s work items on the CPU and the GPU at
every size tried: the capture is the cause, not large tensors split across
CPU worker threads, as this kernel once assumed when it re-derived every
scalar inside the closure (see ``moe_bf16.mojo``).
"""

from extensibility import InputTensor, OutputTensor, foreach, register
from layout import Coord
from max.gpu.host import DeviceContext
from std.utils import IndexList

comptime BLOCKED = -3.0e38


@always_inline
def _at[rank: Int](row: Int, col: Int) -> IndexList[rank]:
    """``[col]`` in a rank-1 tensor (its one row), ``[row, col]`` in a rank-2 one."""
    var index = IndexList[rank](fill=0)
    comptime if rank == 2:
        index[0] = row
    index[rank - 1] = col
    return index


@register("ngram_block")
struct NgramBlock:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor,
        logits: InputTensor[
            dtype = result.dtype, rank = result.rank, static_spec=_
        ],
        history: InputTensor[
            dtype = DType.int32, rank = result.rank, static_spec=_
        ],
        ngram: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            result.rank == 1 or result.rank == 2
        ), "ngram_block takes one row (V,) or B rows (B, V)"
        comptime assert (
            result.dtype.is_floating_point()
        ), "ngram_block masks floating-point logits"
        comptime last = result.rank - 1

        for axis in range(result.rank):
            if Int(logits.dim_size(axis)) != Int(result.dim_size(axis)):
                raise Error("ngram_block: logits and output disagree on shape")
        comptime if result.rank == 2:
            if Int(history.dim_size(0)) != Int(result.dim_size(0)):
                raise Error(
                    "ngram_block: history and logits disagree on the row count"
                )
        if Int(ngram.dim_size(0)) != 1:
            raise Error("ngram_block: `ngram` must be a single int32")
        var hist_len = Int(history.dim_size(last))
        if hist_len < 1:
            raise Error("ngram_block: history is empty")

        def block[
            width: Int
        ](idx: Coord[...]) {
            var logits,
            var history,
            var ngram,
            var hist_len,
        } -> SIMD[result.dtype, width]:
            var ngram_size = Int(ngram.load[1](IndexList[1](0))[0])
            var row = 0
            comptime if result.rank == 2:
                row = Int(idx[0].value())
            var base = Int(idx[last].value())
            var scores = logits.load[width](_at[result.rank](row, base))

            if ngram_size < 1 or hist_len < ngram_size:
                return scores

            var prefix_len = ngram_size - 1
            var starts = hist_len - ngram_size + 1
            var prefix_at = hist_len - prefix_len

            for lane in range(width):
                var token = Int32(base + lane)
                var banned = False
                for start in range(starts):
                    if (
                        history.load[1](
                            _at[result.rank](row, start + prefix_len)
                        )[0]
                        != token
                    ):
                        continue
                    var matches = True
                    for j in range(prefix_len):
                        if (
                            history.load[1](_at[result.rank](row, start + j))[
                                0
                            ]
                            != history.load[1](
                                _at[result.rank](row, prefix_at + j)
                            )[0]
                        ):
                            matches = False
                            break
                    if matches:
                        banned = True
                        break
                if banned:
                    scores[lane] = Scalar[result.dtype](BLOCKED)
            return scores

        foreach[simd_width=1, target=target](block, result, ctx)
