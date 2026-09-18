"""fp32 <-> bf16 conversion in numpy, bit-exact against torch.

numpy has no bfloat16 dtype. That is why the pixel round-trip in
:func:`~unlimited_ocr_max.batch_processor.normalise_view` and the widening of
bf16 checkpoint tensors both went through torch, which must not be a runtime
dependency of this package. Bit-exactness is the contract rather than an
approximation of it -- ``tests/test_bf16.py`` holds both directions against torch
over the whole bf16 range, an exhaustive tie sweep and the specials -- because a
pixel off by one mantissa bit moves the logits, and the port is gated on
byte-identical output.

torch's narrowing is round-to-nearest-even on the upper 16 bits *except* for NaN,
where it answers with one canonical quiet NaN, payload and sign dropped
(``c10::detail::round_to_nearest_even``). That is torch's behaviour rather than
IEEE's, so it is reproduced here deliberately.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bf16_to_fp32", "fp32_to_bf16_roundtrip"]


def bf16_to_fp32(values: np.ndarray) -> np.ndarray:
    """Widen ``values``, a uint16 array of bf16 bit patterns, to a contiguous fp32 array of the same shape.

    Lossless, and not a rounding decision at all: bf16 is the top half of an fp32
    word, so sign, exponent and mantissa all survive, NaN payloads included.
    """
    bits = np.asarray(values)
    if bits.dtype != np.uint16:
        raise ValueError(f"expected a uint16 array of bf16 bit patterns, got {bits.dtype}")
    return (bits.astype(np.uint32, order="C") << np.uint32(16)).view(np.float32)


def fp32_to_bf16_roundtrip(values: np.ndarray) -> np.ndarray:
    """``values`` (fp32) with every element snapped to its nearest bf16 value, returned as fp32 of the same shape.

    Bitwise what ``tensor.to(torch.bfloat16).to(torch.float32)`` produces. fp32
    input only: rounding a wider dtype through fp32 first would round twice, and
    the result would no longer be the one torch gives.
    """
    array = np.asarray(values, order="C")
    if array.dtype != np.float32:
        raise ValueError(f"expected a float32 array, got {array.dtype}")
    bits = array.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    # The NaN arm also covers the only inputs where `bits + bias` wraps the word.
    rounded = np.where(np.isnan(array), np.uint32(0x7FC00000), (bits + bias) & np.uint32(0xFFFF0000))
    return rounded.view(np.float32)
