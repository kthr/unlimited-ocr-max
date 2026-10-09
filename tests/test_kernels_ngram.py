"""Numeric checks for the ``ngram_block`` Mojo kernel against a Python oracle, one row and ``B`` rows.

Every test compiles the custom op through a MAX ``InferenceSession``, so the
whole module is marked ``slow`` and excluded from CI. The device is CPU unless
``UOCR_TEST_DEVICE=gpu``.

The oracle is a plain transcription of the reference's
``SlidingWindowNoRepeatNgramProcessor`` over the WHOLE sequence with an explicit
window; the kernel is handed only the sequence's last ``window`` ids, as the
callers do, so the windowing is checked too. Integer comparisons only: the
kernel either bans exactly the oracle's set (to the kernel's ``BLOCKED``) and
returns every other logit bit-for-bit, or it fails.

The one-row form is driven through :class:`~unlimited_ocr_max.ngram.NgramBlocker`
(the prefill guard, rank 1); the ``B``-row form through a one-op graph of the
rank-2 shape the decode graphs stage, ``logits [B, V]`` and ``history [B, H]``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType

from unlimited_ocr_max.ngram import MOJO_KERNELS, NgramBlocker, apply_ngram_guard

from _harness import device_ref, driver, session

pytestmark = pytest.mark.slow


#: The kernel's ``BLOCKED`` (``ngram_block.mojo``), as the float32 it lands in.
BLOCKED = np.float32(-3.0e38)
#: The R-SWA ring, which is the guard's window in this port.
WINDOW = 128
VOCAB = 4096
#: The checkpoint's vocabulary: the real-shape case splits across CPU worker threads.
REAL_VOCAB = 129280

_MODELS: dict[tuple[int, int, int], object] = {}


# --------------------------------------------------------------------------
# the oracle and the inputs
# --------------------------------------------------------------------------


def _banned(sequence: list[int], *, ngram_size: int, window: int) -> set[int]:
    """The reference's banned set for ``sequence`` (whitelist empty, as the reference builds it)."""
    ids = list(sequence)
    if ngram_size < 1 or len(ids) < ngram_size:
        return set()
    search_start = max(0, len(ids) - window)
    search_end = len(ids) - ngram_size + 1
    prefix = tuple(ids[-(ngram_size - 1) :]) if ngram_size > 1 else ()
    banned: set[int] = set()
    for idx in range(search_start, search_end):
        gram = ids[idx : idx + ngram_size]
        if ngram_size == 1 or tuple(gram[:-1]) == prefix:
            banned.add(gram[-1])
    return banned


def test_the_oracle_on_hand_worked_cases() -> None:
    """Spot checks, so a transcription error cannot pass by matching itself."""
    assert _banned([1, 2, 3], ngram_size=3, window=10) == set()  # the prefix's own occurrence never bans
    assert _banned([1, 2, 3, 1, 2], ngram_size=3, window=10) == {3}
    assert _banned([1, 2, 3, 1, 2], ngram_size=3, window=2) == set()  # out of the window
    assert _banned([1, 2], ngram_size=3, window=10) == set()  # too short for one n-gram
    assert _banned([4, 5, 6], ngram_size=1, window=2) == {5, 6}  # n == 1 bans every id in the window


def _distinct(rng: np.random.Generator, count: int, vocab: int) -> list[int]:
    return [int(i) for i in rng.permutation(vocab)[:count]]


def _rows(vocab: int, seed: int = 0) -> list[list[int]]:
    """Four whole sequences, each longer than :data:`WINDOW`, that the guard treats differently.

    0. ``filler(60) + motif(40) + filler(20) + motif(40)``: the last 34 ids
       recur inside the window, so n = 35 bans the id after the earlier motif
       (and n = 3 bans it too).
    1. distinct ids only: no prefix ever recurs, so nothing is banned at n >= 2.
    2. period 10: every prefix recurs ten ids back, so n = 3 and n = 35 each ban
       the id that followed it.
    3. ``motif(40) + filler(150) + motif(40)``: the same repeat as row 0, but the
       earlier motif sits outside the last 128 ids, so nothing is banned at n >= 2.
    """
    rng = np.random.default_rng(seed)
    ids = _distinct(rng, 310, vocab)
    head, motif, middle = ids[:60], ids[60:100], ids[100:120]
    planted = head + motif + middle + motif
    unique = _distinct(rng, 200, vocab)
    period = [int(i) for i in rng.integers(0, vocab, size=10)] * 20
    far_motif, filler = ids[120:160], ids[160:310]
    far = far_motif + filler + far_motif
    return [planted, unique, period, far]


def _history(sequence: list[int], window: int = WINDOW) -> np.ndarray:
    """What every caller hands the kernel: the sequence's last ``window`` ids."""
    return np.asarray(sequence[-window:], dtype=np.int32)


def _check_row(got: np.ndarray, logits: np.ndarray, sequence: list[int], ngram_size: int, label: str) -> set[int]:
    """``got`` bans exactly the oracle's set, to ``BLOCKED``, and returns every other logit bit-for-bit."""
    want = _banned(sequence, ngram_size=ngram_size, window=WINDOW)
    banned = {int(i) for i in np.flatnonzero(got == BLOCKED)}
    assert banned == want, f"{label}: n={ngram_size} banned {sorted(banned)[:8]}, oracle {sorted(want)[:8]}"
    keep = np.ones(got.shape[0], dtype=bool)
    keep[list(banned)] = False
    assert np.array_equal(got[keep], logits[keep]), f"{label}: n={ngram_size} moved an unbanned logit"
    return banned


# --------------------------------------------------------------------------
# the B-row graph
# --------------------------------------------------------------------------


def _rows_model(batch: int, vocab: int, window: int):
    """``out [B, V] = ngram_block(logits [B, V], history [B, H], ngram [1])``, compiled once per shape."""
    key = (batch, vocab, window)
    if key not in _MODELS:
        dref = device_ref()
        with Graph(
            f"test_ngram_block_rows_{batch}x{vocab}_h{window}",
            input_types=[
                TensorType(DType.float32, [batch, vocab], device=dref),
                TensorType(DType.int32, [batch, window], device=dref),
                TensorType(DType.int32, [1], device=dref),
            ],
            custom_extensions=[MOJO_KERNELS],
        ) as graph:
            graph.output(apply_ngram_guard(*(value.tensor for value in graph.inputs)))
        _MODELS[key] = session().load(graph)
    return _MODELS[key]


def _run_rows(logits: np.ndarray, history: np.ndarray, ngram_size: int) -> np.ndarray:
    model = _rows_model(logits.shape[0], logits.shape[1], history.shape[1])
    arrays = (logits, history, np.asarray([ngram_size], dtype=np.int32))
    buffers = [Buffer.from_numpy(np.ascontiguousarray(array)).to(driver()) for array in arrays]
    return model.execute(*buffers)[0].to(CPU()).to_numpy()


@pytest.mark.parametrize("ngram_size", [1, 3, 35])
def test_each_row_bans_exactly_what_the_reference_bans_for_its_own_sequence(ngram_size: int) -> None:
    """B = 4 rows, four different histories and logits, one op: row ``b`` against row ``b``'s sequence only."""
    sequences = _rows(VOCAB)
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((len(sequences), VOCAB)).astype(np.float32)
    history = np.stack([_history(sequence) for sequence in sequences])

    got = _run_rows(logits, history, ngram_size)
    assert got.shape == logits.shape and got.dtype == np.float32
    banned = [_check_row(got[b], logits[b], sequences[b], ngram_size, f"row {b}") for b in range(len(sequences))]
    if ngram_size == 1:
        # n == 1 bans every id in the window, on every row.
        assert all(banned[b] == set(sequences[b][-WINDOW:]) for b in range(len(sequences)))
    else:
        # Rows 0 and 2 trigger bans, rows 1 and 3 do not -- in the same op.
        assert banned[0] and banned[2] and not banned[1] and not banned[3], [len(b) for b in banned]
    if ngram_size == 35:
        assert banned[0] == {sequences[0][100]}  # the id after the earlier motif


def test_rows_do_not_mix() -> None:
    """Permuting the rows' histories permutes the outputs the same way; with identical logits the history is all that differs."""
    sequences = _rows(VOCAB)
    logits = np.tile(np.random.default_rng(6).standard_normal(VOCAB).astype(np.float32), (len(sequences), 1))
    history = np.stack([_history(sequence) for sequence in sequences])
    order = [2, 3, 0, 1]
    straight = _run_rows(logits, history, 3)
    swapped = _run_rows(logits, history[order], 3)
    assert np.array_equal(swapped, straight[order])
    assert not np.array_equal(straight[0], straight[2])  # the rows really differ


@pytest.mark.parametrize("ngram_size", [0, -1])
def test_n_below_one_returns_every_row_bit_for_bit(ngram_size: int) -> None:
    """The guard-off contract the decode graphs rely on: n < 1 is a pass-through, whatever the history holds."""
    sequences = _rows(VOCAB)
    logits = np.random.default_rng(7).standard_normal((len(sequences), VOCAB)).astype(np.float32)
    history = np.stack([_history(sequence) for sequence in sequences])
    assert np.array_equal(_run_rows(logits, history, ngram_size), logits)
    # A zero history -- what a guard-off caller passes -- too.
    assert np.array_equal(_run_rows(logits, np.zeros_like(history), ngram_size), logits)


def test_a_history_shorter_than_n_bans_nothing() -> None:
    sequences = _rows(VOCAB)
    logits = np.random.default_rng(8).standard_normal((len(sequences), VOCAB)).astype(np.float32)
    history = np.stack([_history(sequence) for sequence in sequences])
    assert np.array_equal(_run_rows(logits, history, WINDOW + 1), logits)


@pytest.mark.parametrize("ngram_size", [1, 35])
def test_the_real_shape_b8_over_the_whole_vocabulary(ngram_size: int) -> None:
    """B = 8 rows of the checkpoint's 129 280 logits: large enough to split across CPU worker threads.

    The kernel's closure captures the history length by value; read through a
    by-reference capture it was garbage here, and at every other size
    (``test_kernel_capture.py``).
    """
    sequences = _rows(REAL_VOCAB, seed=11) + _rows(REAL_VOCAB, seed=13)
    logits = np.random.default_rng(12).standard_normal((8, REAL_VOCAB)).astype(np.float32)
    history = np.stack([_history(sequence) for sequence in sequences])
    got = _run_rows(logits, history, ngram_size)
    for b in range(8):
        _check_row(got[b], logits[b], sequences[b], ngram_size, f"row {b}")


# --------------------------------------------------------------------------
# the one-row form: NgramBlocker, the prefill guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ngram_size", [1, 3, 35])
def test_the_one_row_form_matches_the_oracle_and_row_b_of_the_rows_form(ngram_size: int) -> None:
    """``NgramBlocker.apply`` (rank 1) bans the oracle's set, and row ``b`` of the B-row op computes the same bits."""
    sequences = _rows(VOCAB)
    rng = np.random.default_rng(9)
    logits = rng.standard_normal((len(sequences), VOCAB)).astype(np.float32)
    blocker = NgramBlocker(
        ngram_size=ngram_size,
        window=WINDOW,
        vocab_size=VOCAB,
        device=device_ref(),
        driver_device=driver(),
        session=session(),
    )
    rows = _run_rows(logits, np.stack([_history(sequence) for sequence in sequences]), ngram_size)
    for b, sequence in enumerate(sequences):
        got = blocker.apply(logits[b], sequence)
        assert got.shape == (VOCAB,)
        _check_row(got, logits[b], sequence, ngram_size, f"one row {b}")
        assert np.array_equal(got, rows[b]), f"row {b}: rank 1 and rank 2 disagree"
