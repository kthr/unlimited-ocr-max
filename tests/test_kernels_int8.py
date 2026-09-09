"""Numeric checks for the int8 MoE Mojo kernels against numpy references.

Every test compiles a Mojo custom op through a MAX ``InferenceSession``, so the
whole module is marked ``slow`` and excluded from CI. The device is CPU unless
``UOCR_TEST_DEVICE=gpu``, which also enables the real-shape smoke.

``moe_int8_qmv`` is compared against a float64 reference at a 1e-6 relative
gate (max abs difference over the reference's amax -- the kernel accumulates in
fp32). ``int8_dequant_expert`` is compared bit-for-bit: the reference computes
``float32(w) * scale`` exactly as the kernel does and rounds to bfloat16 with
round-to-nearest-even. Out-of-range expert ids are CLAMPED to ``[0, E)`` by the
kernels (a ``foreach`` closure cannot raise); the references mirror that.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest

from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from unlimited_ocr_max.ngram import MOJO_KERNELS

pytestmark = pytest.mark.slow

GPU = os.environ.get("UOCR_TEST_DEVICE", "cpu").strip().lower() == "gpu"
TOL = 1e-6

# Small shapes shared by the adversarial cases: one compiled graph per shape,
# every weight is a runtime input, so all cases reuse the same models.
E, N, K, GROUPS = 5, 8, 32, 4
G = K // GROUPS

_MODELS: dict[tuple, object] = {}


# Lazy: CI imports this module to collect (and then deselect) the slow tests,
# and collection must not open a device or start a session.
@functools.cache
def _device():
    return Accelerator() if GPU else CPU()


@functools.cache
def _device_ref():
    return DeviceRef.GPU() if GPU else DeviceRef.CPU()


@functools.cache
def _session():
    return InferenceSession(devices=[_device()])


# --------------------------------------------------------------------------
# graph builders and runners
# --------------------------------------------------------------------------


def _load_qmv(k: int, e: int, n: int, kdim: int, groups: int):
    """Compile ``out fp32 [k, n] = moe_int8_qmv(x, expert_ids, w, scales)``."""
    dref = _device_ref()
    with Graph(
        f"test_moe_int8_qmv_{k}x{e}x{n}x{kdim}g{groups}",
        input_types=[
            TensorType(DType.float32, [k, kdim], device=dref),
            TensorType(DType.int32, [k], device=dref),
            TensorType(DType.int8, [e, n, kdim], device=dref),
            TensorType(DType.float32, [e, n, groups], device=dref),
        ],
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        graph.output(
            ops.custom(
                "moe_int8_qmv",
                device=dref,
                values=[inp.tensor for inp in graph.inputs],
                out_types=[TensorType(DType.float32, [k, n], device=dref)],
            )[0]
        )
    return _session().load(graph)


def _load_dequant(e: int, n: int, kdim: int, groups: int):
    """Compile ``out bf16 [n, kdim] = int8_dequant_expert(w, scales, expert_idx)``."""
    dref = _device_ref()
    with Graph(
        f"test_int8_dequant_expert_{e}x{n}x{kdim}g{groups}",
        input_types=[
            TensorType(DType.int8, [e, n, kdim], device=dref),
            TensorType(DType.float32, [e, n, groups], device=dref),
            TensorType(DType.int32, [1], device=dref),
        ],
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        graph.output(
            ops.custom(
                "int8_dequant_expert",
                device=dref,
                values=[inp.tensor for inp in graph.inputs],
                out_types=[TensorType(DType.bfloat16, [n, kdim], device=dref)],
            )[0]
        )
    return _session().load(graph)


def _execute(model, *arrays: np.ndarray):
    buffers = [
        Buffer.from_numpy(np.ascontiguousarray(a)).to(_device()) for a in arrays
    ]
    return model.execute(*buffers)[0]


def _qmv(x, ids, w, scales, *, model=None) -> np.ndarray:
    if model is None:
        key = ("qmv", x.shape[0], *w.shape, scales.shape[2])
        if key not in _MODELS:
            _MODELS[key] = _load_qmv(x.shape[0], *w.shape, scales.shape[2])
        model = _MODELS[key]
    return _execute(model, x, ids, w, scales).to(CPU()).to_numpy()


def _dequant_bits(w, scales, idx: int, *, model=None) -> np.ndarray:
    """The bf16 output as raw uint16 bit patterns (numpy has no bfloat16)."""
    if model is None:
        key = ("dequant", *w.shape, scales.shape[2])
        if key not in _MODELS:
            _MODELS[key] = _load_dequant(*w.shape, scales.shape[2])
        model = _MODELS[key]
    idx_arr = np.asarray([idx], dtype=np.int32)
    return _execute(model, w, scales, idx_arr).to(CPU()).view(DType.uint16).to_numpy()


# --------------------------------------------------------------------------
# numpy references
# --------------------------------------------------------------------------


def _clamp(e: int, num_experts: int) -> int:
    return min(max(int(e), 0), num_experts - 1)


def _qmv_ref(x, ids, w, scales) -> np.ndarray:
    """Float64 reference of the qmv contract, expert ids clamped."""
    xf, wf, sf = (a.astype(np.float64) for a in (x, w, scales))
    k = x.shape[0]
    num_experts, n, kdim = w.shape
    groups = scales.shape[2]
    gsize = kdim // groups
    out = np.empty((k, n), dtype=np.float64)
    for s in range(k):
        e = _clamp(ids[s], num_experts)
        gsums = (wf[e] * xf[s]).reshape(n, groups, gsize).sum(axis=2)
        out[s] = (sf[e] * gsums).sum(axis=1)
    return out


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even), as uint16 bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    wide = bits.astype(np.uint64)
    return ((wide + 0x7FFF + ((wide >> 16) & 1)) >> 16).astype(np.uint16)


def _dequant_ref_bits(w, scales, idx: int) -> np.ndarray:
    """Bit-level reference: float32 multiply exactly as the kernel does."""
    num_experts = w.shape[0]
    gsize = w.shape[2] // scales.shape[2]
    e = _clamp(idx, num_experts)
    prod = w[e].astype(np.float32) * np.repeat(
        scales[e].astype(np.float32), gsize, axis=1
    )
    return _bf16_bits(prod)


def _rel_err(got, ref) -> float:
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float(np.max(np.abs(got)))
    return float(np.max(np.abs(got.astype(np.float64) - ref)) / denom)


def _rand(seed: int, k: int = 6):
    """Random case; the scales are log-uniform over four decades, so a wrong
    group boundary mixes group sums at wildly different magnitudes."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((k, K)).astype(np.float32)
    ids = rng.integers(0, E, size=k).astype(np.int32)
    w = rng.integers(-127, 128, size=(E, N, K)).astype(np.int8)
    scales = (10.0 ** rng.uniform(-3.0, 1.0, size=(E, N, GROUPS))).astype(
        np.float32
    )
    return x, ids, w, scales


# --------------------------------------------------------------------------
# moe_int8_qmv
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qmv_random(seed: int) -> None:
    x, ids, w, scales = _rand(seed)
    rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv random seed={seed}: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_k1() -> None:
    x, ids, w, scales = _rand(3, k=1)
    rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv k=1: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_amax_edge() -> None:
    """Every weight at the quantization edge: alternating exactly +-127."""
    x, ids, _, scales = _rand(4)
    w = np.where(np.arange(K) % 2 == 0, 127, -127)[None, None, :]
    w = np.broadcast_to(w, (E, N, K)).astype(np.int8)
    rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv amax edge: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_all_zero_groups() -> None:
    """A zero weight group and a zero scale group contribute exactly nothing."""
    x, ids, w, scales = _rand(5)
    w = w.copy()
    scales = scales.copy()
    w[:, :, G : 2 * G] = 0  # group 1: zero weights
    scales[:, :, 2] = 0.0  # group 2: zero scale
    rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv zero groups: rel={rel:.3e}")
    assert rel <= TOL
    # Fully zero weights: exactly zero output, not merely close to it.
    assert not np.any(_qmv(x, ids, np.zeros_like(w), scales))


def test_qmv_expert_id_permutations_and_repeats() -> None:
    x, _, w, scales = _rand(6)
    reference_ids = np.arange(E, dtype=np.int32)  # k == 6 > E covers a repeat
    for ids in (
        np.array([4, 2, 0, 1, 3, 2], dtype=np.int32),  # permutation + repeat
        np.array([3, 3, 3, 3, 3, 3], dtype=np.int32),  # one expert for all
        np.concatenate([reference_ids, reference_ids[:1]]),  # 0..4, 0
    ):
        rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
        print(f"[uocr] qmv ids={ids.tolist()}: rel={rel:.3e}")
        assert rel <= TOL
    # Steering evidence: two id vectors must not produce the same output.
    a = _qmv(x, np.full(6, 0, dtype=np.int32), w, scales)
    b = _qmv(x, np.full(6, 4, dtype=np.int32), w, scales)
    assert np.max(np.abs(a - b)) > 0.0


def test_qmv_out_of_range_ids_clamp() -> None:
    """Out-of-range ids read the clamped expert -- bitwise the same output."""
    x, _, w, scales = _rand(7)
    raw = np.array([-1, E, E + 2, -100, 2, 4], dtype=np.int32)
    clamped = np.clip(raw, 0, E - 1).astype(np.int32)
    out_raw = _qmv(x, raw, w, scales)
    assert np.array_equal(out_raw, _qmv(x, clamped, w, scales))
    rel = _rel_err(out_raw, _qmv_ref(x, raw, w, scales))
    print(f"[uocr] qmv clamp: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_group_boundary_sentinels() -> None:
    """One-hot weights at both edges of every group, scales one decade apart:
    an off-by-one group boundary pairs a sentinel with the wrong decade."""
    x = np.arange(1, K + 1, dtype=np.float32)[None, :].repeat(2, axis=0)
    ids = np.array([0, E - 1], dtype=np.int32)
    w = np.zeros((E, N, K), dtype=np.int8)
    w[:, :, ::G] = 1  # first element of each group
    w[:, :, G - 1 :: G] = 1  # last element of each group
    scales = (10.0 ** np.arange(GROUPS, dtype=np.float32))[None, None, :]
    scales = np.ascontiguousarray(np.broadcast_to(scales, (E, N, GROUPS)))
    rel = _rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv boundary sentinels: rel={rel:.3e}")
    assert rel <= TOL


# --------------------------------------------------------------------------
# int8_dequant_expert
# --------------------------------------------------------------------------


def test_dequant_random_every_expert() -> None:
    _, _, w, scales = _rand(8)
    for e in range(E):
        got = _dequant_bits(w, scales, e)
        assert np.array_equal(got, _dequant_ref_bits(w, scales, e)), f"expert {e}"
    print(f"[uocr] dequant every expert: {E} stacks bit-exact")


def test_dequant_amax_edge() -> None:
    _, _, _, scales = _rand(9)
    w = np.where(np.arange(K) % 2 == 0, 127, -127)[None, None, :]
    w = np.broadcast_to(w, (E, N, K)).astype(np.int8)
    assert np.array_equal(_dequant_bits(w, scales, 1), _dequant_ref_bits(w, scales, 1))
    print("[uocr] dequant amax edge: bit-exact")


def test_dequant_zero_scale_group() -> None:
    """A zero scale against negative weights: the -0.0 bits must match too."""
    _, _, w, scales = _rand(10)
    scales = scales.copy()
    scales[:, :, 1] = 0.0
    got = _dequant_bits(w, scales, 2)
    assert np.array_equal(got, _dequant_ref_bits(w, scales, 2))
    assert np.any(got[:, G : 2 * G] == 0x8000)  # a -0.0 actually occurred
    print("[uocr] dequant zero-scale group: bit-exact incl. -0.0")


def test_dequant_out_of_range_idx_clamp() -> None:
    _, _, w, scales = _rand(11)
    for raw, clamped in ((E, E - 1), (E + 7, E - 1), (-3, 0)):
        got = _dequant_bits(w, scales, raw)
        assert np.array_equal(got, _dequant_bits(w, scales, clamped))
        assert np.array_equal(got, _dequant_ref_bits(w, scales, raw))
    print("[uocr] dequant clamp: bit-exact")


def test_dequant_nonuniform_scales_boundary() -> None:
    """All-ones weights, scales one decade apart: out[n, k] is exactly the
    group's scale, so a wrong ``k // G`` shows up as a decade jump."""
    w = np.ones((E, N, K), dtype=np.int8)
    scales = (10.0 ** np.arange(GROUPS, dtype=np.float32))[None, None, :]
    scales = np.ascontiguousarray(np.broadcast_to(scales, (E, N, GROUPS)))
    got = _dequant_bits(w, scales, 0)
    assert np.array_equal(got, _dequant_ref_bits(w, scales, 0))
    print("[uocr] dequant boundary decades: bit-exact")


# --------------------------------------------------------------------------
# real-shape smoke (GPU only): the production expert stacks, one at a time
# --------------------------------------------------------------------------


@pytest.mark.skipif(not GPU, reason="real-shape smoke needs UOCR_TEST_DEVICE=gpu")
@pytest.mark.parametrize(
    "n,kdim", [(896, 1280), (1280, 896)], ids=["gate_up", "down"]
)
def test_real_shape_smoke(n: int, kdim: int) -> None:
    """Production stacks ``[64, 896, 1280]`` (gate/up) and ``[64, 1280, 896]``
    (down) at G=128, one stack at a time; models are built uncached and dropped
    so the two shapes never coexist."""
    num_experts, gsize, k = 64, 128, 6
    groups = kdim // gsize
    rng = np.random.default_rng(20260909)
    x = rng.standard_normal((k, kdim)).astype(np.float32)
    ids = rng.integers(0, num_experts, size=k).astype(np.int32)
    w = rng.integers(-127, 128, size=(num_experts, n, kdim)).astype(np.int8)
    scales = rng.uniform(1e-3, 3e-2, size=(num_experts, n, groups)).astype(
        np.float32
    )

    qmv_model = _load_qmv(k, num_experts, n, kdim, groups)
    rel = _rel_err(
        _qmv(x, ids, w, scales, model=qmv_model), _qmv_ref(x, ids, w, scales)
    )
    print(f"[uocr] real-shape qmv [{num_experts},{n},{kdim}]: rel={rel:.3e}")
    del qmv_model
    assert rel <= TOL

    dequant_model = _load_dequant(num_experts, n, kdim, groups)
    e = int(ids[0])
    got = _dequant_bits(w, scales, e, model=dequant_model)
    del dequant_model
    assert np.array_equal(got, _dequant_ref_bits(w, scales, e))
    print(f"[uocr] real-shape dequant [{num_experts},{n},{kdim}]: bit-exact")


def test_qmv_rejects_a_group_miscount() -> None:
    """The host-side shape raises are load-bearing: K not divisible by the group
    count must refuse loudly instead of reading garbage group boundaries."""
    with pytest.raises(Exception, match=r"moe_int8_qmv: K must be a multiple of the group count"):
        model = _load_qmv(k=2, e=3, n=4, kdim=32, groups=5)
        _execute(
            model,
            np.zeros((2, 32), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            np.zeros((3, 4, 32), dtype=np.int8),
            np.zeros((3, 4, 5), dtype=np.float32),
        )


def test_dequant_rejects_a_group_miscount() -> None:
    with pytest.raises(Exception, match=r"int8_dequant_expert: K must be a multiple of the group count"):
        model = _load_dequant(e=3, n=4, kdim=32, groups=5)
        _execute(
            model,
            np.zeros((3, 4, 32), dtype=np.int8),
            np.zeros((3, 4, 5), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
        )
