"""The CUDA-only split of a projection wider than CUDA's grid-Y limit.

MAX's GEMV dispatcher checks ``ceildiv(n, 2)`` against ``MAX_GRID_DIM_Y`` and
then launches ``ceildiv(n, tile_n)`` blocks on that axis with ``tile_n`` as low
as 1, so an fp32 GEMV wider than 65535 passes the check and fails the launch
(``CUDA_ERROR_INVALID_VALUE``; A100, driver 595.84, 26.6.0 and nightly). This
package's ``lm_head`` is 129280 wide. These tests pin three things: the split
fires only on CUDA, it fires with the right geometry, and it does not move a
single bit.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from unlimited_ocr_max import decoder
from unlimited_ocr_max.decoder import CUDA_MAX_GRID_Y, Projection, _projection_split

IN_DIM = 8


def _logits(width: int, weight: np.ndarray, activation: np.ndarray) -> np.ndarray:
    """Build and run a one-``Projection`` graph on CPU, whatever split the gate chooses."""
    name = "lm_head.weight"
    projection = Projection(IN_DIM, width, dtype=DType.float32, device=DeviceRef.CPU())
    projection.weight.name = name
    with Graph(
        f"projection_{width}",
        input_types=[TensorType(DType.float32, [1, IN_DIM], device=DeviceRef.CPU())],
    ) as graph:
        graph.output(projection(graph.inputs[0].tensor))
    model = InferenceSession(devices=[CPU()]).load(
        graph, weights_registry={name: Buffer.from_numpy(np.ascontiguousarray(weight))}
    )
    return model.execute(Buffer.from_numpy(np.ascontiguousarray(activation)))[0].to_numpy()


def test_the_gate_is_closed_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metal and CPU have no grid-Y limit, and the graph they serve is the validated one."""
    for api in ("metal", "hip", "cpu"):
        monkeypatch.setattr(decoder, "accelerator_api", lambda api=api: api)
        assert _projection_split(129280) is None, api


def test_the_gate_is_closed_for_every_projection_but_lm_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decoder's other projections are 64 to 6848 wide; only the vocabulary crosses the cap."""
    monkeypatch.setattr(decoder, "accelerator_api", lambda: "cuda")
    for out_dim in (64, 1280, 6848, CUDA_MAX_GRID_Y):
        assert _projection_split(out_dim) is None, out_dim


def test_the_split_geometry_covers_the_vocabulary_within_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Balanced chunks, every one inside the cap, together covering every column exactly once."""
    monkeypatch.setattr(decoder, "accelerator_api", lambda: "cuda")
    for out_dim in (CUDA_MAX_GRID_Y + 1, 129280, 1_000_000):
        width = _projection_split(out_dim)
        assert width is not None and width <= CUDA_MAX_GRID_Y, out_dim
        starts = range(0, out_dim, width)
        assert sum(min(s + width, out_dim) - s for s in starts) == out_dim, out_dim


def test_the_split_does_not_move_a_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole justification: a column is an independent dot product, so chunking N is free.

    Run at a cap of 7 rather than 65535 so the split is exercised at a size a
    test can hold; the arithmetic that matters is identical at either scale.
    """
    rng = np.random.default_rng(20260922)
    width = 20
    weight = rng.standard_normal((width, IN_DIM), dtype=np.float32)
    activation = rng.standard_normal((1, IN_DIM), dtype=np.float32)

    monkeypatch.setattr(decoder, "accelerator_api", lambda: "metal")
    whole = _logits(width, weight, activation)

    monkeypatch.setattr(decoder, "accelerator_api", lambda: "cuda")
    monkeypatch.setattr(decoder, "CUDA_MAX_GRID_Y", 7)
    assert _projection_split(width) == 7
    split = _logits(width, weight, activation)

    assert whole.shape == split.shape == (1, width)
    assert np.array_equal(whole.view(np.uint32), split.view(np.uint32)), (
        f"{np.flatnonzero(whole.view(np.uint32) != split.view(np.uint32)).size} of {width} columns differ"
    )
