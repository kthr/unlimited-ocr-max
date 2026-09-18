"""``normalise_view`` pinned bitwise against the torch expression it replaced.

KON-193 rewrote the pixel pipeline in numpy so ``batch_processor`` need not import
torch; ``bf16.fp32_to_bf16_roundtrip`` (KON-190) already holds the round-trip
against torch exhaustively, so this file only has to show that the surrounding
``ToTensor``/``Normalize`` arithmetic still lands on the exact bits torch produced.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unlimited_ocr_max import batch_processor
from unlimited_ocr_max.batch_processor import BASE_SIZE, PIXEL_MEAN, PIXEL_STD, normalise_view, preprocess_page


def _full_uint8_range_image() -> Image.Image:
    """256x256 RGB where each channel independently sweeps every value 0-255."""
    idx = np.arange(256, dtype=np.uint8)
    r = np.broadcast_to(idx, (256, 256))
    g = np.broadcast_to(idx[:, None], (256, 256))
    b = np.broadcast_to(idx[::-1], (256, 256))
    hwc = np.stack([r, g, b], axis=-1)
    for channel in range(3):
        assert set(np.unique(hwc[..., channel]).tolist()) == set(range(256))
    return Image.fromarray(hwc, mode="RGB")


def _torch_normalise_view(image: Image.Image) -> np.ndarray:
    """The exact expression ``normalise_view`` used before KON-193, kept here as the reference."""
    hwc = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(hwc).permute(2, 0, 1).contiguous().to(torch.float32).div(255.0)
    tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
    return np.ascontiguousarray(tensor.to(torch.bfloat16).to(torch.float32).numpy())


def test_normalise_view_matches_torch_over_the_full_uint8_range() -> None:
    image = _full_uint8_range_image()
    got = normalise_view(image)
    want = _torch_normalise_view(image)

    assert got.dtype == np.float32
    assert got.shape == want.shape == (3, 256, 256)
    assert got.flags["C_CONTIGUOUS"]

    got_bits, want_bits = got.view(np.uint32), want.view(np.uint32)
    if not np.array_equal(got_bits, want_bits):
        differ = np.flatnonzero(got_bits.ravel() != want_bits.ravel())
        first = int(differ[0])
        raise AssertionError(
            f"{differ.size} of {got_bits.size} pixels differ, first at flat index {first}: "
            f"got 0x{got_bits.ravel()[first]:08X}, torch 0x{want_bits.ravel()[first]:08X}"
        )


def test_preprocess_page_shape_and_dtype_are_unchanged() -> None:
    image = Image.new("RGB", (300, 500), color=(10, 20, 30))
    pixels = preprocess_page(image)
    assert pixels.shape == (1, 3, BASE_SIZE, BASE_SIZE)
    assert pixels.dtype == np.float32
    assert np.ascontiguousarray(pixels).flags["C_CONTIGUOUS"]


def test_normalise_view_rejects_a_non_rgb_shape() -> None:
    grey = Image.new("L", (8, 8))
    try:
        normalise_view(grey)
    except ValueError as exc:
        assert "expected an RGB HWC image" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a non-RGB image")


def test_batch_processor_source_has_no_torch_import() -> None:
    """The one parity-critical torch site (KON-193): the module must not import torch at all."""
    source = Path(batch_processor.__file__).read_text()
    assert re.search(r"^\s*(import torch\b|from torch\b)", source, re.MULTILINE) is None
