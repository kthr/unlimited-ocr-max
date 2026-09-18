"""The ``Buffer`` <-> numpy crossing: bit patterns, the dtype round trip, lifetime, and no copy.

Two of these are contracts :func:`~unlimited_ocr_max.buffers.buffer_to_numpy`'s
docstring states outright, and both are load-bearing for callers rather than
cosmetic:

* the returned array holds the memory alive **itself**, through the DLPack capsule
  at its ``.base``, so ``stack_expert_weights`` may drop each member ``Buffer`` the
  moment it has the array;
* it does **not** copy -- the measured one-allocation-per-stack property (the
  ``np.stack`` and nothing else) would silently become two if a defensive copy
  ever appeared here.

torch is the reference for the bf16 bit patterns and the independent way to build
a bf16 ``Buffer`` -- it is not imported by the module under test. The expected
patterns are also spelled out as literals, so a torch that changed its mind could
not quietly take the tests with it.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
from max.driver import Buffer
from max.dtype import DType

from unlimited_ocr_max.buffers import buffer_to_numpy, numpy_to_buffer

#: Values chosen to be exactly representable in bf16, next to their bf16 bit patterns:
#: sign, 8 exponent bits, 7 mantissa bits -- the top half of the fp32 word.
VALUES = [[1.5, -2.0, 0.0], [3.25, -0.5, 7.0]]
BITS = np.array([[0x3FC0, 0xC000, 0x0000], [0x4050, 0xBF00, 0x40E0]], dtype=np.uint16)


def _bf16_buffer() -> tuple[Buffer, torch.Tensor]:
    """A bf16 ``Buffer`` and the torch tensor whose memory it aliases through DLPack."""
    tensor = torch.tensor(VALUES, dtype=torch.bfloat16)
    return Buffer.from_dlpack(tensor), tensor


def test_a_bf16_buffer_crosses_as_its_uint16_bit_patterns() -> None:
    """numpy has no bf16 dtype, so the storage width is what carries the bits over -- untouched."""
    buffer, tensor = _bf16_buffer()
    assert buffer.dtype == DType.bfloat16

    array = buffer_to_numpy(buffer)
    assert array.dtype == np.uint16
    assert array.shape == (2, 3)
    assert array.flags["C_CONTIGUOUS"]
    assert np.array_equal(array, BITS)
    assert np.array_equal(array, tensor.view(torch.int16).numpy().view(np.uint16))


def test_the_dtype_and_shape_survive_the_round_trip_back_to_a_buffer() -> None:
    """``dtype`` is passed rather than inferred, because a uint16 array cannot still say bfloat16."""
    buffer, _tensor = _bf16_buffer()

    back = numpy_to_buffer(buffer_to_numpy(buffer), DType.bfloat16)
    assert back.dtype == DType.bfloat16
    assert tuple(back.shape) == (2, 3)
    assert np.array_equal(buffer_to_numpy(back), BITS)


def test_the_array_keeps_its_memory_alive_after_the_buffer_is_dropped() -> None:
    """The DLPack capsule at ``.base`` is the owner, which is what lets callers drop the ``Buffer``.

    ``stack_expert_weights`` relies on exactly this: it holds 64 arrays and no
    ``Buffer``s while ``np.stack`` runs. If the capsule stopped owning the
    memory, the read would be a use-after-free rather than a failure.
    """
    buffer, tensor = _bf16_buffer()
    array = buffer_to_numpy(buffer)
    assert array.base is not None, "the array must own the memory, not borrow it from the Buffer"

    del buffer, tensor
    gc.collect()

    assert np.array_equal(array, BITS)


def test_the_crossing_does_not_copy() -> None:
    """A write through the array reaches the source, in both the bf16 and the plain-dtype arm.

    A defensive copy here would pass every value check above and silently double
    the one-allocation-per-stack property that was measured.
    """
    buffer, tensor = _bf16_buffer()
    buffer_to_numpy(buffer)[0, 0] = np.uint16(0xC000)  # 1.5 -> -2.0
    assert tensor[0, 0].item() == -2.0

    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    buffer_to_numpy(Buffer.from_numpy(source))[1, 1] = np.float32(99.0)
    assert source[1, 1] == np.float32(99.0)
