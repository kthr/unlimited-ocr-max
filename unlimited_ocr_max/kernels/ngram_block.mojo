"""Windowed no-repeat-n-gram logit blocking as a MAX custom op.

Port of the reference's ``SlidingWindowNoRepeatNgramProcessor``::

    search_start = max(0, len(sequence) - window)
    search_end   = len(sequence) - ngram_size + 1
    current_prefix = tuple(sequence[-(ngram_size - 1):])
    for idx in range(search_start, search_end):
        ngram = sequence[idx : idx + ngram_size]
        if tuple(ngram[:-1]) == current_prefix:
            scores[ngram[-1]] = -inf

Inputs: ``logits (V,)``; ``history (H,) int32`` -- the already windowed tail of
the whole sequence, prompt included, so every position the reference's loop
touches is present; ``ngram (1,) int32``. Banned ids are set to ``BLOCKED``, a
finite value that is argmax- and softmax-equivalent to ``-inf`` and cannot
produce a NaN.
"""

from extensibility import InputTensor, OutputTensor, foreach, register
from layout import Coord
from max.gpu.host import DeviceContext
from std.utils import IndexList

comptime BLOCKED = -3.0e38


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
        history: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        ngram: InputTensor[dtype = DType.int32, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime assert result.rank == 1, "ngram_block is rank-1 only"
        comptime assert (
            result.dtype.is_floating_point()
        ), "ngram_block masks floating-point logits"

        if Int(logits.dim_size(0)) != Int(result.dim_size(0)):
            raise Error(
                "ngram_block: logits and output disagree on the vocabulary size"
            )
        if Int(ngram.dim_size(0)) != 1:
            raise Error("ngram_block: `ngram` must be a single int32")
        if Int(history.dim_size(0)) < 1:
            raise Error("ngram_block: history is empty")

        @parameter
        def block[width: Int](idx: Coord[...]) -> SIMD[result.dtype, width]:
            # Every scalar is re-derived from the captured tensors: a runtime
            # scalar captured into a `foreach` closure is garbage on CPU worker
            # threads once the tensor is large enough to be split across them.
            var hist_len = Int(history.dim_size(0))
            var ngram_size = Int(ngram.load[1](IndexList[1](0))[0])
            var base = Int(idx[0].value())
            var scores = logits.load[width](IndexList[1](base))

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
                        history.load[1](IndexList[1](start + prefix_len))[0]
                        != token
                    ):
                        continue
                    var matches = True
                    for j in range(prefix_len):
                        if (
                            history.load[1](IndexList[1](start + j))[0]
                            != history.load[1](IndexList[1](prefix_at + j))[0]
                        ):
                            matches = False
                            break
                    if matches:
                        banned = True
                        break
                if banned:
                    scores[lane] = Scalar[result.dtype](BLOCKED)
            return scores

        foreach[block, target=target, simd_width=1](result, ctx)
