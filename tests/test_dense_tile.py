"""KON-238: the rows per weight-row read ``dense_bf16_qmv`` runs at (``decoder._rows_per_read``), model-free.

A wide weight -- ``lm_head`` (129280), the dense FFN's gate/up (6848) -- is
read once per call at every decode ``B``: one tile of all ``B`` rows, not the
power of two below ``B`` (round 1 served ``B = 3`` in tiles of 2 + 1 and
``B = 6`` in 4 + 2, so ``lm_head`` was read twice, at 7.0-7.4 ms against
3.7-4.1). The 1280- and 1792-wide projections take two tiles, as even as
``B`` allows, so their call keeps ``DENSE_MIN_ITEMS`` items. The kernel's
numerics at every tile are ``test_kernels_dense_bf16``'s (slow).
"""

from __future__ import annotations

import pytest

from unlimited_ocr_max.cli import MAX_BATCH_CAP
from unlimited_ocr_max.decoder import _rows_per_read

#: The decoder's projection widths: lm_head, the dense FFN's gate/up, the shared experts' gate/up, the rest.
WIDE = (129280, 6848)
NARROW = (1792, 1280)
DECODE_ROWS = range(1, MAX_BATCH_CAP + 1)


@pytest.mark.parametrize("out_dim", WIDE)
@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_a_wide_weight_is_read_once_for_all_the_decode_rows(rows: int, out_dim: int) -> None:
    assert _rows_per_read(rows, out_dim) == rows


@pytest.mark.parametrize("out_dim", NARROW)
def test_a_narrow_weight_takes_two_tiles_as_even_as_the_rows_allow(out_dim: int) -> None:
    got = {rows: _rows_per_read(rows, out_dim) for rows in DECODE_ROWS}
    assert got == {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
