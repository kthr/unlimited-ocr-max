"""KON-238: the bf16-read projections against the fp32-weight path they replace, and a float64 reference.

Before KON-238 the shared registry held every projection's exact fp32 upcast
and MAX's matmul read it; now the registry keeps the bf16 weight and the
decoder reads it through ``dense_bf16_qmv`` (:meth:`Projection.decode_rows`:
the decode step's rows and the prefill's ``lm_head`` row) or a run-time upcast
(a call: the prefill's rows). bf16 -> fp32 is exact, so only the summation
order can move a value: each spelling is held to the float64 evaluation of the
same bf16 values at a 1e-5 relative gate (max abs difference over the
reference's amax), and against the fp32-weight path at the same gate. The
prefill spelling is the fp32 path's own matmul on the same fp32 values -- the
K-blocked bf16 weight upcast and put back in ``[N, K]`` order -- so it is also
checked bitwise. A decode row's bits must not depend on how many rows
share the call (the tile and its padding move with ``B``), so each row at
``B = 1..7`` is checked bitwise against the same row at ``B = 8``.

On CUDA, MAX's fp32 matmul is not fp32-exact: on an A100 (sm_80) it misses
the float64 reference by 4e-4 to 1.2e-3 at B = 2, 3, 5..8 and at prefill,
while ``dense_bf16_qmv`` stays at 4e-7 to 7e-7 (2026-10-08). That size is
TF32's (a 10-bit mantissa on the tensor cores), the hypothesis this file
works with. So every comparison that goes through MAX's matmul -- the
fp32-weight path, and the prefill spelling -- uses ``MATMUL_TOL`` there;
``dense_bf16_qmv`` against float64 keeps ``TOL`` on every device.

Synthetic weights at the decoder's real projection shapes (``lm_head`` cut to
4096 columns; ``test_kernels_dense_bf16`` runs the real width on a GPU).
Compiles graphs, so ``slow``; the device is CPU unless ``UOCR_TEST_DEVICE=gpu``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer, accelerator_api
from max.dtype import DType
from max.graph import Graph, TensorType
from max.nn import Module

from unlimited_ocr_max.decoder import KBLOCK, GatedMlp, Projection
from unlimited_ocr_max.ngram import MOJO_KERNELS

from _harness import GPU, device_ref, driver, rel_err, session

pytestmark = pytest.mark.slow

TOL = 1e-5
#: The gate for a result that went through MAX's fp32 matmul: ``TOL``, except on CUDA, where
#: the matmul measured up to 1.2e-3 against float64 (module docstring); about 3x that.
MATMUL_TOL = 4e-3 if GPU and accelerator_api() == "cuda" else TOL
HIDDEN = 1280


def _bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy((0.05 * rng.standard_normal(shape)).astype(np.float32)).to(torch.bfloat16)


def _module(kind: str, width: int, dtype: DType) -> Module:
    dref = device_ref()
    if kind == "projection":
        return Projection(HIDDEN, width, dtype=dtype, device=dref)
    return GatedMlp(HIDDEN, width, dtype=dtype, device=dref)


def _weights(kind: str, width: int, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if kind == "projection":
        return {"weight": _bf16(rng, (width, HIDDEN))}
    return {
        "gate_proj.weight": _bf16(rng, (width, HIDDEN)),
        "up_proj.weight": _bf16(rng, (width, HIDDEN)),
        "down_proj.weight": _bf16(rng, (HIDDEN, width)),
    }


def _kblocked(w: torch.Tensor) -> torch.Tensor:
    """``w [N, K]`` as ``[K / KBLOCK, N, KBLOCK]``, how the decoder holds a bf16 projection."""
    return w.reshape(w.shape[0], -1, KBLOCK).permute(1, 0, 2).contiguous()


def _run(kind: str, width: int, weights: dict[str, torch.Tensor], x: np.ndarray, *, fp32: bool, decode: bool) -> np.ndarray:
    """The module over ``x``: bf16 weights, K-blocked (the KON-238 path), or their exact fp32 upcast (the path it replaces)."""
    module = _module(kind, width, DType.float32 if fp32 else DType.bfloat16)
    module.load_state_dict({name: w.float() if fp32 else _kblocked(w) for name, w in weights.items()})
    dref = device_ref()
    with Graph(
        f"kon238_{kind}_{width}_{x.shape[0]}_{'fp32' if fp32 else 'bf16'}_{'decode' if decode else 'call'}",
        input_types=[TensorType(DType.float32, list(x.shape), device=dref)],
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        xv = graph.inputs[0].tensor
        # The fp32-weight path has no kernel: MAX's matmul at the decode step's rows too.
        graph.output(module.decode_rows(xv) if decode and not fp32 else module(xv))
    model = session().load(graph, weights_registry=module.state_dict())
    return model.execute(Buffer.from_numpy(np.ascontiguousarray(x)).to(driver()))[0].to(CPU()).to_numpy()


def _reference(kind: str, weights: dict[str, torch.Tensor], x: np.ndarray) -> np.ndarray:
    f64 = {name: w.float().numpy().astype(np.float64) for name, w in weights.items()}
    xf = x.astype(np.float64)
    if kind == "projection":
        return xf @ f64["weight"].T
    gate, up = xf @ f64["gate_proj.weight"].T, xf @ f64["up_proj.weight"].T
    return (gate / (1.0 + np.exp(-gate)) * up) @ f64["down_proj.weight"].T


#: (module, width): ``lm_head`` (cut), the attention projections, the shared experts, the dense FFN.
CASES = [("projection", 4096), ("projection", 1280), ("mlp", 1792), ("mlp", 6848)]
IDS = ["lm_head", "attention", "shared_experts", "dense_ffn"]


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 6, 7, 8])
@pytest.mark.parametrize(("kind", "width"), CASES, ids=IDS)
def test_decode_rows_match_the_fp32_weight_path(kind: str, width: int, rows: int) -> None:
    """The decode spelling (``dense_bf16_qmv``) at ``B`` rows: float64 at ``TOL``, the fp32-weight matmul at ``MATMUL_TOL``."""
    weights = _weights(kind, width, seed=width + rows)
    x = np.random.default_rng(rows).standard_normal((rows, HIDDEN)).astype(np.float32)
    got = _run(kind, width, weights, x, fp32=False, decode=True)
    fp32 = _run(kind, width, weights, x, fp32=True, decode=True)
    ref = _reference(kind, weights, x)
    rel_ref, rel_fp32, rel_old = rel_err(got, ref), rel_err(got, fp32), rel_err(fp32, ref)
    print(
        f"[uocr] {kind} {width} B={rows} decode: vs float64 {rel_ref:.2e} (fp32 path {rel_old:.2e}), "
        f"vs fp32 path {rel_fp32:.2e}"
    )
    assert np.all(np.isfinite(got)) and np.any(got)
    assert rel_ref <= TOL and rel_fp32 <= MATMUL_TOL


@pytest.mark.parametrize(("kind", "width"), CASES, ids=IDS)
def test_each_decode_row_is_bitwise_its_value_at_eight_rows(kind: str, width: int) -> None:
    """``B = 1..7`` against ``B = 8`` at the decoder's own tiles: padded tiles (``B = 3, 5, 7`` on the narrow widths) included."""
    weights = _weights(kind, width, seed=width)
    x = np.random.default_rng(8).standard_normal((8, HIDDEN)).astype(np.float32)
    widest = _run(kind, width, weights, x, fp32=False, decode=True)
    for rows in range(1, 8):
        got = _run(kind, width, weights, x[:rows], fp32=False, decode=True)
        assert np.array_equal(got, widest[:rows]), rows


@pytest.mark.parametrize("rows", [6, 277])
@pytest.mark.parametrize(("kind", "width"), CASES, ids=IDS)
def test_the_prefill_call_is_the_fp32_weight_path(kind: str, width: int, rows: int) -> None:
    """The prefill spelling (a run-time upcast, then MAX's matmul): float64 at ``MATMUL_TOL``, the fp32 path bitwise.

    277 rows is a served base-mode prompt (273 image tokens and the text).
    """
    weights = _weights(kind, width, seed=width)
    x = np.random.default_rng(rows).standard_normal((rows, HIDDEN)).astype(np.float32)
    got = _run(kind, width, weights, x, fp32=False, decode=False)
    fp32 = _run(kind, width, weights, x, fp32=True, decode=False)
    rel_ref = rel_err(got, _reference(kind, weights, x))
    print(
        f"[uocr] {kind} {width} prefill {rows} rows: vs float64 {rel_ref:.2e}, "
        f"bitwise the fp32 path {bool(np.array_equal(got, fp32))}"
    )
    assert np.all(np.isfinite(got)) and np.any(got)
    assert rel_ref <= MATMUL_TOL
    assert np.array_equal(got, fp32), f"max abs diff {float(np.max(np.abs(got - fp32))):.3e}"
