"""Numeric checks for ``dense_bf16_qmv`` (``kernels/dense_bf16.mojo``) against a float64 reference.

Every test compiles the Mojo op through a MAX ``InferenceSession``, so the
module is marked ``slow`` and excluded from CI. The device is CPU unless
``UOCR_TEST_DEVICE=gpu``, which also enables the real-shape smoke.

The op is ``x fp32 [M, K]`` against a bf16 weight ``W [N, K]`` held K-blocked,
``[K / KBLOCK, N, KBLOCK]``, accumulated in fp32: compared with the float64
evaluation of the exact bf16 values at a 1e-5 relative gate (max abs
difference over the reference's amax). The weights travel as raw bfloat16 bit
patterns (numpy has no bfloat16), so the reference reads exactly the values
the kernel reads; each test builds ``W`` and ``_run`` lays it out with the
weight adapter's own :func:`~unlimited_ocr_max.weight_adapters.kblocked`.

A work item serves a tile of rows from one read of a weight row; the tile is a
compile-time parameter the decoder picks per call (``decoder._rows_per_read``,
any value from 1 to 8; ``test_dense_tile`` pins the choice). A last tile that
``M`` leaves short is padded: every item runs ``tile`` rows of arithmetic and
stores only its real rows. What must not move with the tile, the padding, or
how many rows share a call, is any row's bits: the batched decode serves on
that (KON-212). So every row is checked bitwise against its value computed
alone at every tile, with ``M`` below the tile, above it and not a multiple of it.

The vector loads claim 16-byte alignment (``kernels/loads.mojo``) when a
host-side check finds both bases on that boundary and both tensors packed; a
buffer off it -- the CPU runs on an input where it lies -- takes the same loop
with loads that claim one element, so the bits do not depend on the address.
A call with no rows returns an empty result.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops

from unlimited_ocr_max.buffers import buffer_to_numpy, numpy_to_buffer
from unlimited_ocr_max.decoder import KBLOCK, _rows_per_read, dense_bf16_qmv
from unlimited_ocr_max.ngram import MOJO_KERNELS
from unlimited_ocr_max.weight_adapters import kblocked

from _harness import GPU, bf16_bits, bf16_values, device_ref, driver, off_boundary, rel_err, session

pytestmark = pytest.mark.slow

TOL = 1e-5
TILES = tuple(range(1, 9))

_MODELS: dict[tuple, object] = {}


def _load(m: int, n: int, k: int, tile: int | None, *, x_cols: int | None = None, x_offset: int = 0):
    """``dense_bf16_qmv(x, w)`` with both as graph inputs; ``x_cols``/``x_offset`` make ``x`` a column slice of a wider input."""
    key = (m, n, k, tile, x_cols, x_offset)
    if key not in _MODELS:
        dref = device_ref()
        cols = k if x_cols is None else x_cols
        with Graph(
            f"test_dense_bf16_{m}x{n}x{k}_t{tile}_{cols}_{x_offset}",
            input_types=[
                TensorType(DType.float32, [m, cols], device=dref),
                TensorType(DType.bfloat16, [k // KBLOCK, n, KBLOCK], device=dref),
            ],
            custom_extensions=[MOJO_KERNELS],
        ) as graph:
            x = graph.inputs[0].tensor
            if x_cols is not None:
                x = x[:, x_offset : x_offset + k]
            graph.output(dense_bf16_qmv(x, graph.inputs[1].tensor, tile=tile))
        _MODELS[key] = session().load(graph)
    return _MODELS[key]


def _run(x: np.ndarray, w_bits: np.ndarray, tile: int | None = None, *, w_off: bool = False, **slicing) -> np.ndarray:
    """``W = w_bits [N, K]`` K-blocked and run; ``w_off`` places the blocked weight one element past a 16-byte boundary."""
    k = w_bits.shape[1]
    model = _load(x.shape[0], w_bits.shape[0], k, tile, **slicing)
    blocked = buffer_to_numpy(kblocked(numpy_to_buffer(np.ascontiguousarray(w_bits, dtype=np.uint16), DType.bfloat16)))
    w = numpy_to_buffer(off_boundary(blocked) if w_off else blocked, DType.bfloat16).to(driver())
    xb = Buffer.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(driver())
    return model.execute(xb, w)[0].to(CPU()).to_numpy()


def _ref(x: np.ndarray, w_bits: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ bf16_values(w_bits).T


def _rand(seed: int, m: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Weights over four decades with random signs, so a wrong K offset or row mixes terms of very different size."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, k)).astype(np.float32)
    magnitude = 10.0 ** rng.uniform(-3.0, 1.0, size=(n, k))
    return x, bf16_bits((magnitude * rng.choice([-1.0, 1.0], size=(n, k))).astype(np.float32))


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize(("m", "n", "k"), [(1, 24, 64), (5, 24, 64), (8, 40, 96), (13, 7, 32)])
def test_matches_the_float64_reference(m: int, n: int, k: int, tile: int) -> None:
    x, w = _rand(m * 1000 + n + k, m, n, k)
    rel = rel_err(_run(x, w, tile), _ref(x, w))
    print(f"[uocr] dense_bf16_qmv M={m} N={n} K={k} tile={tile}: rel={rel:.3e}")
    assert rel <= TOL


@pytest.mark.parametrize("m", [2, 3, 5, 8, 9, 17])
def test_every_row_is_bitwise_its_lone_value_at_every_tile(m: int) -> None:
    """A row's bits do not depend on the tile, on ``M`` or on the rows beside it (the decode's load independence)."""
    x, w = _rand(40 + m, m, 24, 64)
    alone = np.concatenate([_run(x[r : r + 1], w, 1) for r in range(m)])
    for tile in TILES:
        assert np.array_equal(_run(x, w, tile), alone), tile


def test_a_column_slice_activation_is_read_through_its_strides() -> None:
    """``x`` as a column window of a wider input at an odd offset: a base and row stride no vector load is aligned to.

    Whether MAX hands the op the strided view or a packed copy, the values
    must be the packed ones bit for bit: the unaligned and aligned vector
    paths are the same arithmetic.
    """
    x_wide = np.random.default_rng(7).standard_normal((6, 67)).astype(np.float32)
    _, w = _rand(8, 6, 24, 64)
    packed = np.ascontiguousarray(x_wide[:, 1:65])
    got = _run(x_wide, w, 4, x_cols=67, x_offset=1)
    assert np.array_equal(got, _run(packed, w, 4))
    assert rel_err(got, _ref(packed, w)) <= TOL


def test_one_hot_rows_pick_exactly_the_indexed_element() -> None:
    """``w[n, :]`` is one-hot at ``(3 n + 1) % K``: every output is exactly one ``x`` element."""
    m, n, k = 6, 24, 64
    x = (np.arange(m * k, dtype=np.float32).reshape(m, k) + 1.0) * np.float32(0.5)
    w = np.zeros((n, k), dtype=np.float32)
    for row in range(n):
        w[row, (3 * row + 1) % k] = 1.0
    want = np.array([[x[r, (3 * c + 1) % k] for c in range(n)] for r in range(m)], dtype=np.float32)
    for tile in TILES:
        assert np.array_equal(_run(x, bf16_bits(w), tile), want), tile


def test_all_zero_weights_are_exactly_zero() -> None:
    x, w = _rand(5, 4, 24, 64)
    assert not np.any(_run(x, np.zeros_like(w), 4))


def test_zero_rows() -> None:
    """M = 0: an empty ``[0, N]`` result, no error, at every tile."""
    _, w = _rand(6, 1, 24, 64)
    for tile in TILES:
        assert _run(np.zeros((0, 64), dtype=np.float32), w, tile).shape == (0, 24), tile


@pytest.mark.skipif(GPU, reason="a device copy is always aligned; only the CPU runs on an input where it lies")
@pytest.mark.parametrize("off", ["w", "x", "wx"])
def test_misaligned_inputs_give_the_aligned_bits(off: str) -> None:
    """Weights and/or activations one element past a 16-byte boundary, at every tile: the host check sends
    the call down the unclaimed loads, the same loop, so the bits are the aligned call's. The bits cannot
    show which loads ran -- the point of the path is that they do not -- so this checks the values a
    misaligned call returns, not the predicate."""
    x, w = _rand(9, 6, 24, 64)
    ref = _ref(x, w)
    for tile in TILES:
        got = _run(off_boundary(x) if "x" in off else x, w, tile, w_off="w" in off)
        assert np.array_equal(got, _run(x, w, tile)), tile
        assert rel_err(got, ref) <= TOL, tile


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "message"),
    [((2, 24), (2, 4, KBLOCK), "w and x disagree on K"), ((2, 32), (4, 4, 8), "w is not K-blocked by KBLOCK")],
    ids=["k_mismatch", "not_kblocked"],
)
def test_rejects_a_weight_that_does_not_fit(x_shape: tuple[int, int], w_shape: tuple[int, int, int], message: str) -> None:
    """The host-side shape checks are load-bearing: an ``x`` narrower than the weight, or a weight not
    blocked by ``KBLOCK``, must refuse, not read past it."""
    dref = device_ref()
    with Graph(
        f"test_dense_bf16_refuses_{w_shape[2]}_{x_shape[1]}",
        input_types=[TensorType(DType.float32, list(x_shape), device=dref), TensorType(DType.bfloat16, list(w_shape), device=dref)],
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        out = TensorType(DType.float32, [x_shape[0], w_shape[1]], device=dref)
        graph.output(ops.custom("dense_bf16_qmv", dref, list(graph.inputs), [out], parameters={"tile": 1})[0])
    with pytest.raises(Exception, match=rf"dense_bf16_qmv: {message}"):
        session().load(graph).execute(
            Buffer.from_numpy(np.zeros(x_shape, dtype=np.float32)).to(driver()),
            Buffer.from_numpy(np.zeros(w_shape, dtype=np.uint16)).view(DType.bfloat16).to(driver()),
        )


# --------------------------------------------------------------------------
# real-shape smoke (GPU only): the decoder's projections at the decode step's row counts
# --------------------------------------------------------------------------


@pytest.mark.skipif(not GPU, reason="real-shape smoke needs UOCR_TEST_DEVICE=gpu")
@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize(
    ("n", "k"),
    [(129280, 1280), (1280, 1280), (6848, 1280), (1280, 6848), (1792, 1280), (1280, 1792)],
    ids=["lm_head", "attention", "dense_gate_up", "dense_down", "shared_gate_up", "shared_down"],
)
def test_real_shape_smoke(n: int, k: int, m: int) -> None:
    """At the decoder's own tile choice (:func:`_rows_per_read`); a silent device shortfall would read as zeros."""
    rng = np.random.default_rng(n + k + m)
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k), dtype=np.float32)  # float32 throughout: lm_head in float64 is 1.3 GB a copy
    w *= np.float32(0.05)
    w = bf16_bits(w)
    got = _run(x, w)
    _MODELS.clear()
    ref = _ref(x, w)
    rel = rel_err(got, ref)
    print(f"[uocr] dense_bf16_qmv real shape N={n} K={k} M={m} tile={_rows_per_read(m, n)}: rel={rel:.3e}")
    assert np.all(np.isfinite(got)) and np.any(got)
    assert rel <= TOL
