"""The :class:`~max.driver.Buffer` <-> numpy crossing, and the fp32 view of a checkpoint tensor.

A checkpoint tensor is a ``Buffer`` throughout this package -- MAX's own mmap of
the safetensors file. ``Buffer`` is also the one representation in reach that can
*name* bfloat16: numpy has no such dtype, and torch is not a runtime dependency.
So :func:`buffer_to_numpy` and :func:`numpy_to_buffer` are the crossing in both
directions, and numpy only ever sees the bit patterns; the arithmetic on those
bit patterns lives one module over, in :mod:`unlimited_ocr_max.bf16`, which stays
numpy-only so a bare-numpy consumer can import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType

from .bf16 import bf16_to_fp32

__all__ = ["as_float32", "buffer_to_numpy", "numpy_to_buffer"]

#: What bf16 bytes are called while numpy holds them. Same width, so the
#: reinterpretation costs nothing and no bit pattern is touched -- not even a NaN's.
_BF16_STORAGE = DType.uint16


def buffer_to_numpy(buffer: Buffer) -> np.ndarray:
    """``buffer``'s elements as numpy, a bf16 one handed over as its uint16 bit patterns.

    numpy cannot consume a bf16 DLPack capsule at all -- it answers
    ``BufferError: Unsupported dtype in DLTensor`` -- so bf16 crosses under the
    name of its storage width and :func:`numpy_to_buffer` gives the dtype back on
    the far side. A host buffer aliases; nothing is copied.

    The returned array holds the memory alive itself, through the DLPack capsule
    at its ``.base``, so ``buffer`` may be dropped straight after. **Do not add
    a defensive copy here**: the callers' one-allocation-per-stack property is
    measured, and a copy would quietly double it.
    """
    dtype = _BF16_STORAGE if buffer.dtype == DType.bfloat16 else buffer.dtype
    return buffer.view(dtype, buffer.shape).to_numpy()


def numpy_to_buffer(array: np.ndarray, dtype: DType) -> Buffer:
    """``array``'s bytes as a ``Buffer`` of ``dtype``, the inverse of :func:`buffer_to_numpy`.

    ``dtype`` is what the bytes *mean*, which is not something a uint16 array
    carrying bf16 can still say -- so it is passed rather than inferred. A
    contiguous array aliases; nothing is copied.
    """
    return Buffer.from_numpy(array).view(dtype, array.shape)


def as_float32(value: Any) -> np.ndarray:
    """Contiguous float32 view of a checkpoint tensor -- a MAX ``Buffer`` or a plain numpy array.

    Widening bf16 is lossless and not a rounding decision at all: bf16 *is* the
    top half of the fp32 word. It goes through
    :func:`~unlimited_ocr_max.bf16.bf16_to_fp32` because numpy cannot hold the
    narrow form even long enough to cast it.
    """
    if isinstance(value, Buffer):
        bits = buffer_to_numpy(value)
        value = bf16_to_fp32(bits) if value.dtype == DType.bfloat16 else bits
    return np.ascontiguousarray(value, dtype=np.float32)
