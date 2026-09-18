"""The numpy bf16 conversions held bitwise against torch: pixel fuzz, exponent sweep, ties, specials, dense sweep.

Every comparison is on the ``uint32`` bit pattern rather than on float values, so
a NaN whose payload or sign drifted fails instead of comparing equal to itself,
and a signed zero that flipped fails instead of comparing equal to its opposite.

torch is the reference only; it is not imported by the module under test. Where a
property holds without a reference -- ties landing on an even mantissa, signed
zero surviving, the round trip being closed over the bf16 range -- it is asserted
directly too, so a torch that changed its mind could not quietly take the tests
with it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unlimited_ocr_max.bf16 import bf16_to_fp32, fp32_to_bf16_roundtrip

PATTERNS = 1 << 16  # every bf16 bit pattern
BF16_NAN_PATTERNS = 2 * 127  # exponent all ones, any of the 7 mantissa bits set, either sign


def torch_roundtrip(values: np.ndarray) -> np.ndarray:
    """The expression ``batch_processor.normalise_view`` applies to the pixels today."""
    return torch.from_numpy(values).to(torch.bfloat16).to(torch.float32).numpy()


def torch_widen(patterns: np.ndarray) -> np.ndarray:
    """torch's bf16 -> fp32, reached through ``int16`` because ``from_numpy`` has no bf16 entry point."""
    return torch.from_numpy(patterns.view(np.int16)).view(torch.bfloat16).to(torch.float32).numpy()


def as_float32(bits: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype=np.uint32).view(np.float32)


def assert_same_bits(got: np.ndarray, want: np.ndarray, case: str) -> None:
    assert got.dtype == np.float32, f"{case}: {got.dtype}"
    assert got.shape == want.shape, f"{case}: {got.shape} != {want.shape}"
    got_bits, want_bits = got.view(np.uint32).ravel(), want.view(np.uint32).ravel()
    if np.array_equal(got_bits, want_bits):
        return
    differ = np.flatnonzero(got_bits != want_bits)
    first = int(differ[0])
    raise AssertionError(
        f"{case}: {differ.size} of {got_bits.size} differ, first at flat index {first}: "
        f"got 0x{got_bits[first]:08X}, torch 0x{want_bits[first]:08X}"
    )


# --- the values the port actually rounds --------------------------------------


def test_a_large_uniform_fuzz_over_the_pixel_range_matches_torch() -> None:
    """Post-normalisation pixels live in [-1, 1]; that band gets the densest sample."""
    rng = np.random.default_rng(20260918)
    values = rng.uniform(-1.0, 1.0, size=1 << 20).astype(np.float32)
    assert values.min() < -0.999 and values.max() > 0.999
    assert_same_bits(fp32_to_bf16_roundtrip(values), torch_roundtrip(values), "pixel-range fuzz")


def test_every_reachable_pixel_value_matches_torch() -> None:
    """``(u8/255 - 0.5)/0.5`` has only 256 outcomes, so the reachable set can be checked whole."""
    pixels = (np.arange(256, dtype=np.uint8).astype(np.float32) / np.float32(255.0) - np.float32(0.5)) / np.float32(0.5)
    assert pixels.dtype == np.float32
    assert_same_bits(fp32_to_bf16_roundtrip(pixels), torch_roundtrip(pixels), "reachable pixels")


# --- exponents, ties, specials ------------------------------------------------


def test_a_sweep_across_every_exponent_matches_torch() -> None:
    """All 256 exponents against a spread of mantissas: subnormals, the overflow edge, Inf and NaN included.

    This is also where the specials are checked against torch: both zeros, both
    infinities, both quiet NaNs, a signalling NaN (mantissa ``0x1``), a NaN whose
    payload sits only below the kept half (``0x8000``), the all-ones NaN, both
    largest finites and the smallest subnormal are each ``sign | exponent |
    mantissa`` triples this product already contains. Shrinking the mantissa
    list would drop them silently -- the ones below that assert expected bits
    *without* torch cover fewer patterns.
    """
    signs = np.array([0x00000000, 0x80000000], dtype=np.uint32)
    exponents = np.arange(256, dtype=np.uint32) << np.uint32(23)
    mantissas = np.array([0x0, 0x1, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x123456, 0x400000, 0x7FFFFF], dtype=np.uint32)
    bits = (signs[:, None, None] | exponents[None, :, None] | mantissas[None, None, :]).ravel()
    values = as_float32(bits)
    assert_same_bits(fp32_to_bf16_roundtrip(values), torch_roundtrip(values), "exponent sweep")


def test_every_tie_rounds_to_even() -> None:
    """The exact halfway case -- round bit set, remainder zero -- over all 65536 kept halves, so both
    parities of the lowest kept bit (bit 16) are covered exhaustively."""
    kept = np.arange(PATTERNS, dtype=np.uint32) << np.uint32(16)
    bits = kept | np.uint32(0x8000)
    values = as_float32(bits)
    got = fp32_to_bf16_roundtrip(values)
    assert_same_bits(got, torch_roundtrip(values), "ties")

    # Round-to-even is visible without a reference: a tie always lands on an even
    # mantissa. NaN inputs are excluded -- torch answers those with one fixed NaN.
    finite = (bits & np.uint32(0x7FFFFFFF)) <= np.uint32(0x7F800000)
    assert np.all((got.view(np.uint32)[finite] >> np.uint32(16)) % 2 == 0)


def test_either_side_of_a_tie_rounds_the_ordinary_way() -> None:
    """One below the tie truncates, one above rounds up -- whatever the kept bit's parity."""
    kept = np.arange(0x0100, 0x7F00, dtype=np.uint32) << np.uint32(16)  # finite, no overflow at the top
    for low, expected in ((np.uint32(0x7FFF), kept), (np.uint32(0x8001), kept + np.uint32(0x10000))):
        values = as_float32(kept | low)
        got = fp32_to_bf16_roundtrip(values)
        assert_same_bits(got, torch_roundtrip(values), f"low 0x{int(low):04X}")
        assert np.array_equal(got.view(np.uint32), expected)


def test_signed_zero_and_infinity_come_back_untouched() -> None:
    bits = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000], dtype=np.uint32)
    got = fp32_to_bf16_roundtrip(as_float32(bits))
    assert np.array_equal(got.view(np.uint32), bits)


def test_every_nan_collapses_to_one_canonical_quiet_nan() -> None:
    """torch's conversion is not IS-a-NaN-preserving: payload and sign are dropped
    (``c10::detail::round_to_nearest_even``). Matching torch means dropping them too."""
    rng = np.random.default_rng(3)
    payloads = np.unique(np.concatenate([np.arange(1, 4096), rng.integers(1, 1 << 23, size=1 << 14)])).astype(np.uint32)
    bits = np.concatenate([np.uint32(0x7F800000) | payloads, np.uint32(0xFF800000) | payloads])
    values = as_float32(bits)
    got = fp32_to_bf16_roundtrip(values)
    assert_same_bits(got, torch_roundtrip(values), "nan payloads")
    assert np.array_equal(got.view(np.uint32), np.full(bits.size, 0x7FC00000, dtype=np.uint32))


def test_the_largest_finite_values_overflow_to_infinity_like_torch() -> None:
    """Above the largest bf16 the nearest value is Inf; the wrap this could cause in the
    32-bit add is confined to NaN inputs, which never reach the arithmetic."""
    bits = np.array([0x7F7FFFFF, 0x7F800000, 0xFF7FFFFF, 0xFF800000, 0x7F7F8000, 0xFF7F8000], dtype=np.uint32)
    values = as_float32(bits)
    got = fp32_to_bf16_roundtrip(values)
    assert_same_bits(got, torch_roundtrip(values), "overflow")
    infinities = np.array([0x7F800000] * 2 + [0xFF800000] * 2 + [0x7F800000, 0xFF800000], dtype=np.uint32)
    assert np.array_equal(got.view(np.uint32), infinities)


# --- widening -----------------------------------------------------------------


def test_widening_matches_torch_over_every_bf16_pattern() -> None:
    patterns = np.arange(PATTERNS, dtype=np.uint16)
    got = bf16_to_fp32(patterns)
    assert got.shape == patterns.shape
    assert got.flags["C_CONTIGUOUS"]
    assert_same_bits(got, torch_widen(patterns), "widening")


def test_every_non_nan_bf16_pattern_survives_the_round_trip() -> None:
    """Closed over the bf16 range: widen, round back, and the 16 bits are the ones we started with.
    NaNs are excluded because the narrowing direction answers all of them with ``0x7FC0``."""
    patterns = np.arange(PATTERNS, dtype=np.uint16)
    not_nan = (patterns & np.uint16(0x7FFF)) <= np.uint16(0x7F80)
    kept = np.ascontiguousarray(patterns[not_nan])
    assert kept.size == PATTERNS - BF16_NAN_PATTERNS
    back = (fp32_to_bf16_roundtrip(bf16_to_fp32(kept)).view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    assert np.array_equal(back, kept)


def test_widening_keeps_a_nan_payload_that_the_narrowing_direction_would_drop() -> None:
    """The asymmetry is real, and it is torch's: placing 16 bits in the high half cannot lose anything."""
    patterns = np.array([0x7F81, 0xFF81, 0x7FFF, 0xFFFF], dtype=np.uint16)
    got = bf16_to_fp32(patterns)
    assert np.array_equal(got.view(np.uint32), patterns.astype(np.uint32) << np.uint32(16))
    assert_same_bits(got, torch_widen(patterns), "nan widening")


# --- shape, layout, dtype -----------------------------------------------------


@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (1,), (2, 3, 4)])
def test_both_directions_preserve_shape(shape: tuple[int, ...]) -> None:
    narrowed = fp32_to_bf16_roundtrip(np.zeros(shape, dtype=np.float32))
    widened = bf16_to_fp32(np.zeros(shape, dtype=np.uint16))
    assert narrowed.shape == widened.shape == shape
    assert narrowed.dtype == widened.dtype == np.float32
    assert narrowed.flags["C_CONTIGUOUS"] and widened.flags["C_CONTIGUOUS"]


def test_a_non_contiguous_input_is_rounded_correctly() -> None:
    """Views reach this from ``permute``-shaped preprocessing, so the input layout must not matter."""
    rng = np.random.default_rng(11)
    dense = rng.uniform(-1.0, 1.0, size=(8, 12)).astype(np.float32)
    strided = np.asfortranarray(dense)[::-1, ::2]
    contiguous = np.ascontiguousarray(strided)
    got = fp32_to_bf16_roundtrip(strided)
    assert got.shape == strided.shape
    assert got.flags["C_CONTIGUOUS"]
    assert_same_bits(got, torch_roundtrip(contiguous), "strided input")


def test_the_input_is_not_modified_in_place() -> None:
    values = as_float32(np.array([0x3F800001, 0x40490FDB], dtype=np.uint32)).copy()
    before = values.copy()
    fp32_to_bf16_roundtrip(values)
    assert np.array_equal(values.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.uint32, np.int32])
def test_narrowing_refuses_anything_but_float32(dtype: type) -> None:
    """A float64 input would round twice and stop matching torch; a silent cast would hide that."""
    with pytest.raises(ValueError, match="expected a float32 array"):
        fp32_to_bf16_roundtrip(np.zeros(4, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.int16, np.uint32, np.float32])
def test_widening_refuses_anything_but_uint16(dtype: type) -> None:
    with pytest.raises(ValueError, match="expected a uint16 array"):
        bf16_to_fp32(np.zeros(4, dtype=dtype))
