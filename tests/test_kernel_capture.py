"""The capture probe: a host scalar captured BY VALUE reaches a kernel's work items intact.

A ``@parameter`` closure captures by reference, and a host-side runtime
scalar read through such a capture is garbage in the work items at 1024,
129 280 and 4 Mi items alike (KON-240): zero on the M4's GPU through both
``foreach`` and ``elementwise``, NaN or zero on the CPU through ``foreach``.
The MoE and n-gram kernels therefore capture their host scalars by value, in
a closure passed as a runtime argument to ``foreach`` or ``elementwise``.
The probe ops (``tests/capture_probe``) do the same with
``out = x + float(n)``; ``n`` is a symbolic dimension, so each op compiles
once.

The probe compiles Mojo through a MAX ``InferenceSession``, so the module is
marked ``slow``. The device is CPU unless ``UOCR_TEST_DEVICE=gpu``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops

from _harness import device_ref, driver, session

pytestmark = pytest.mark.slow

PROBE = Path(__file__).resolve().parent / "capture_probe"


@functools.cache
def _model(op: str):
    dref = device_ref()
    with Graph(
        f"test_{op}", input_types=[TensorType(DType.float32, ["n"], device=dref)], custom_extensions=[PROBE]
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(ops.custom(op, device=dref, values=[x], out_types=[TensorType(DType.float32, x.shape, device=dref)])[0])
    return session().load(graph)


@pytest.mark.parametrize("n", [1024, 129_280, 1 << 22])
@pytest.mark.parametrize("op", ["capture_probe_foreach", "capture_probe_elementwise"])
def test_a_value_captured_host_scalar_arrives_intact(op: str, n: int) -> None:
    """Every item adds exactly ``n`` (all sums are exact in float32 below 2**24)."""
    x = np.arange(n, dtype=np.float32)
    got = _model(op).execute(Buffer.from_numpy(x).to(driver()))[0].to(CPU()).to_numpy()
    want = x + np.float32(n)
    assert np.array_equal(got, want), f"{np.count_nonzero(got != want)} of {n} items wrong"
