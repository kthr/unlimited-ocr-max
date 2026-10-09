"""The capture probe behind ``tests/test_kernel_capture.py``: ``out = x + float(n)``.

``n`` is ``x``'s length, a host-side runtime scalar captured BY VALUE into
the closure, which ``foreach`` / ``elementwise`` take as a runtime argument
-- the form the kernels in ``unlimited_ocr_max/kernels`` use.
"""

from extensibility import InputTensor, OutputTensor, foreach, register
from layout import Coord
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.utils import IndexList


@register("capture_probe_foreach")
struct CaptureProbeForeach:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor[dtype = DType.float32, rank=1, static_spec=_],
        x: InputTensor[dtype = DType.float32, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var shift = Float32(Int(x.dim_size(0)))

        def body[
            width: Int
        ](idx: Coord[...]) {var x, var shift} -> SIMD[DType.float32, width]:
            return x.load[width](idx) + shift

        foreach[target=target](body, result, ctx)


@register("capture_probe_elementwise")
struct CaptureProbeElementwise:
    @staticmethod
    def execute[
        target: StaticString
    ](
        result: OutputTensor[dtype = DType.float32, rank=1, static_spec=_],
        x: InputTensor[dtype = DType.float32, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        var n = Int(x.dim_size(0))
        var shift = Float32(n)

        def body[
            width: Int, alignment: Int = 1
        ](idx: Coord[...]) {var result, var x, var shift}:
            var i = IndexList[1](Int(idx[0].value()))
            result.store[1](i, x.load[1](i) + shift)

        elementwise[simd_width=1, target=target](body, (n,), ctx)
