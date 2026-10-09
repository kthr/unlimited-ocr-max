"""What the numeric test modules share: the device they run on, and the bf16 / error / alignment helpers.

The device is CPU unless ``UOCR_TEST_DEVICE=gpu``. The accessors are lazy: CI
imports the slow modules to collect (and then deselect) their tests, and
collection must not open a device or start a session.
"""

from __future__ import annotations

import functools
import os

import numpy as np
from max.driver import CPU, Accelerator, Device
from max.engine import InferenceSession
from max.graph import DeviceRef

GPU = os.environ.get("UOCR_TEST_DEVICE", "cpu").strip().lower() == "gpu"


@functools.cache
def driver() -> Device:
    return Accelerator() if GPU else CPU()


@functools.cache
def device_ref() -> DeviceRef:
    return DeviceRef.GPU() if GPU else DeviceRef.CPU()


@functools.cache
def session() -> InferenceSession:
    return InferenceSession(devices=[driver()])


def bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even), as uint16 bits.

    In uint32 throughout -- the high half plus the rounding carry -- so a
    129280 x 1280 weight never takes a uint64 copy.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    carry = ((bits & np.uint32(0xFFFF)) + ((bits >> 16) & np.uint32(1)) + np.uint32(0x7FFF)) >> 16
    return ((bits >> 16) + carry).astype(np.uint16)


def bf16_values(bits: np.ndarray) -> np.ndarray:
    """The float64 value of each bfloat16 bit pattern (exact)."""
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """``max |got - ref|`` over ``max |ref|``, in float64; ``max |got|`` when ``ref`` is all zero."""
    got64, ref64 = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    denom = float(np.max(np.abs(ref64)))
    if denom == 0.0:
        return float(np.max(np.abs(got64)))
    return float(np.max(np.abs(got64 - ref64)) / denom)


def off_boundary(a: np.ndarray) -> np.ndarray:
    """A copy of ``a`` one element past a 16-byte boundary, the alignment the kernels' vector loads claim."""
    raw = np.zeros(a.nbytes + 32, dtype=np.uint8)
    start = (-raw.ctypes.data) % 16 + a.itemsize
    out = raw[start : start + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    assert out.ctypes.data % 16 == a.itemsize
    return out


def clamp(e: int, num_experts: int) -> int:
    """An expert id as the kernels read it: clamped to ``[0, num_experts)``."""
    return min(max(int(e), 0), num_experts - 1)
