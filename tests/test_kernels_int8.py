"""Numeric checks for the int8 MoE Mojo kernels against numpy references.

Every test compiles a Mojo custom op through a MAX ``InferenceSession``, so the
whole module is marked ``slow`` and excluded from CI. The device is CPU unless
``UOCR_TEST_DEVICE=gpu``, which also enables the real-shape smoke.

``moe_int8_qmv`` is compared against a float64 reference at a 1e-6 relative
gate (max abs difference over the reference's amax -- the kernel accumulates in
fp32). ``int8_dequant_expert`` is compared bit-for-bit: the reference computes
``float32(w) * scale`` exactly as the kernel does and rounds to bfloat16 with
round-to-nearest-even. Out-of-range expert ids are CLAMPED to ``[0, E)`` by the
kernels (a kernel closure cannot raise); the references mirror that.

``moe_int8_qmv`` reads each selected expert once for all the rows routed to it
(KON-239): the shared-id cases put several rows (k >= 24: more than one tile)
on one expert, and every row must stay bitwise what it is when computed alone.

Its bits are pinned to the base's (6e190af) operation order -- a group summed
one element at a time in ``kk`` order by fp32 fused multiply-adds, the scaled
group sums fused-added per group -- by a numpy reference in exactly that order
(KON-240 round 2 dropped a 16-lane group sum: other bits, measured +3.1 % at
B = 8 in the decode step). The kernel LOADS 16 int8 at a time when the group
size allows -- claiming 16-byte alignment when the bases are on that boundary
(checked host-side), else one element -- and one at a time otherwise, with
the same sums: the small shapes' group size, 32, takes the 16-wide loads,
weights or activations off a 16-byte boundary the same loads unclaimed,
``K_SHORT``'s group size of 8 the one-element loads, and all must give the
pinned bits.
``int8_dequant_expert``'s vector loads claim 16 bytes the same way. A call
with no rows (k = 0, as a device-graph warm-up launch can be) returns an
empty result.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest

from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops

from unlimited_ocr_max.ngram import MOJO_KERNELS

from _harness import GPU, bf16_bits, clamp, device_ref, driver, off_boundary, rel_err, session

pytestmark = pytest.mark.slow

TOL = 1e-6

# Small shapes shared by the adversarial cases: one compiled graph per shape,
# every weight is a runtime input, so all cases reuse the same models. The
# group size G is a multiple of the kernel's 16-wide load; K_SHORT's is not.
E, N, K, GROUPS = 5, 8, 128, 4
G = K // GROUPS
K_SHORT = 32
#: Rows the kernel serves per read of a weight row, read from ``moe_int8.mojo`` so it cannot drift.
TILE = int(re.search(r"^comptime TILE = (\d+)$", (MOJO_KERNELS / "moe_int8.mojo").read_text(), re.M).group(1))

_MODELS: dict[tuple, object] = {}


# --------------------------------------------------------------------------
# graph builders and runners
# --------------------------------------------------------------------------


def _load_qmv(k: int, e: int, n: int, kdim: int, groups: int):
    """Compile ``out fp32 [k, n] = moe_int8_qmv(x, expert_ids, w, scales)``."""
    dref = device_ref()
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
    return session().load(graph)


def _load_dequant(e: int, n: int, kdim: int, groups: int):
    """Compile ``out bf16 [n, kdim] = int8_dequant_expert(w, scales, expert_idx)``."""
    dref = device_ref()
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
    return session().load(graph)


def _execute(model, *arrays: np.ndarray):
    buffers = [
        Buffer.from_numpy(np.ascontiguousarray(a)).to(driver()) for a in arrays
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


def _qmv_ref(x, ids, w, scales) -> np.ndarray:
    """Float64 reference of the qmv contract, expert ids clamped."""
    xf, wf, sf = (a.astype(np.float64) for a in (x, w, scales))
    k = x.shape[0]
    num_experts, n, kdim = w.shape
    groups = scales.shape[2]
    gsize = kdim // groups
    out = np.empty((k, n), dtype=np.float64)
    for s in range(k):
        e = clamp(ids[s], num_experts)
        gsums = (wf[e] * xf[s]).reshape(n, groups, gsize).sum(axis=2)
        out[s] = (sf[e] * gsums).sum(axis=1)
    return out


def _fma32(a, b, c) -> np.ndarray:
    """fp32 ``a * b + c`` rounded once, as the kernel's fused multiply-adds are:
    the product of two fp32 is exact in float64, and the float64 sum rounded to
    fp32 is the fma's result (a double rounding could differ at ~2^-29 per
    operation; it never does in the fixed cases below)."""
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(np.float32)


def _qmv_base_order_fp32(x, ids, w, scales) -> np.ndarray:
    """fp32 reference in the kernel's exact operation order (``_dot_rows_by`` in
    ``moe_int8.mojo``, the order of 6e190af's one-element kernel, which the
    16-wide loads keep): per group, ``gsum = fma(w, x, gsum)`` per element in
    ``kk`` order into a scalar group sum, then ``acc = fma(scale, gsum, acc)``
    per group. Each ``+=`` of a product in the kernel compiles to one fused
    multiply-add on the CPU and on the M4 alike -- the bits are the same on
    both -- so a reference that rounds the product on its own differs in most
    outputs (60-88 %, measured)."""
    k = x.shape[0]
    num_experts, n, kdim = w.shape
    groups = scales.shape[2]
    gsize = kdim // groups
    out = np.empty((k, n), dtype=np.float32)
    for s in range(k):
        e = clamp(ids[s], num_experts)
        wf = w[e].astype(np.float32)
        acc = np.zeros(n, dtype=np.float32)
        for g in range(groups):
            gsum = np.zeros(n, dtype=np.float32)
            for kk in range(g * gsize, (g + 1) * gsize):
                gsum = _fma32(wf[:, kk], x[s, kk], gsum)
            acc = _fma32(scales[e, :, g], gsum, acc)
        out[s] = acc
    return out


def _dequant_ref_bits(w, scales, idx: int) -> np.ndarray:
    """Bit-level reference: float32 multiply exactly as the kernel does."""
    num_experts = w.shape[0]
    gsize = w.shape[2] // scales.shape[2]
    e = clamp(idx, num_experts)
    prod = w[e].astype(np.float32) * np.repeat(
        scales[e].astype(np.float32), gsize, axis=1
    )
    return bf16_bits(prod)


def _rand(seed: int, k: int = 6, kdim: int = K):
    """Random case; the scales are log-uniform over four decades, so a wrong
    group boundary mixes group sums at wildly different magnitudes."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((k, kdim)).astype(np.float32)
    ids = rng.integers(0, E, size=k).astype(np.int32)
    w = rng.integers(-127, 128, size=(E, N, kdim)).astype(np.int8)
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
    rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv random seed={seed}: rel={rel:.3e}")
    assert rel <= TOL


@pytest.mark.parametrize("kdim", [K, K_SHORT], ids=["loads16", "loads1"])
@pytest.mark.parametrize("k", [1, 6, 48])
def test_qmv_bits_are_the_base_summation_order(k: int, kdim: int) -> None:
    """Bitwise the fp32 reference in the base's (6e190af) operation order, so the
    int8 decode bits stay what the int8 transcripts are pinned to. Ids from
    ``[-2, E + 2)``: shared and clamped rows take the tiled path, and must sum
    in the same order as a row alone; G = 32 takes the 16-wide loads, G = 8
    the one-element loads, with the same sums. Any other order -- lane-wise
    partial sums, a multiply rounded apart from its add, the scale applied
    per element -- moves most outputs' low bits and fails here."""
    x, _, w, scales = _rand(12 + k, k=k, kdim=kdim)
    ids = np.random.default_rng(90 + k).integers(-2, E + 2, size=k).astype(np.int32)
    out = _qmv(x, ids, w, scales)
    want = _qmv_base_order_fp32(x, ids, w, scales)
    differ = int(np.count_nonzero(out.view(np.uint32) != want.view(np.uint32)))
    print(f"[uocr] qmv k={k} G={kdim // GROUPS} base-order fp32: {differ} of {out.size} outputs differ")
    assert differ == 0


def test_qmv_k1() -> None:
    x, ids, w, scales = _rand(3, k=1)
    rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv k=1: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_zero_rows() -> None:
    """k = 0: an empty ``[0, N]`` result, no error."""
    _, _, w, scales = _rand(13)
    out = _qmv(np.zeros((0, K), dtype=np.float32), np.zeros(0, dtype=np.int32), w, scales)
    assert out.shape == (0, N)


@pytest.mark.skipif(GPU, reason="a device copy is always aligned; only the CPU runs on an input where it lies")
@pytest.mark.parametrize("off", ["w", "x", "wx"])
def test_qmv_misaligned_inputs_give_the_same_bits(off: str) -> None:
    """Weights and/or activations one element past a 16-byte boundary: the host
    check sends the call down the unclaimed loads, which sum in the same order,
    so the bits are those of the aligned call (and of the base-order
    reference). Every path gives the same bits by design, so this cannot show
    which loads ran: it checks the values a misaligned call returns, not the
    predicate."""
    x, ids, w, scales = _rand(14)
    out = _qmv(off_boundary(x) if "x" in off else x, ids, off_boundary(w) if "w" in off else w, scales)
    assert np.array_equal(out, _qmv(x, ids, w, scales))
    assert np.array_equal(out, _qmv_base_order_fp32(x, ids, w, scales))
    print(f"[uocr] qmv misaligned {off}: bitwise the aligned call")


@pytest.mark.parametrize("k", [1, 6, 12, 24, 48])
def test_qmv_rows_sharing_experts_are_their_lone_values(k: int) -> None:
    """``k`` rows with ids drawn from ``[-2, E + 2)``: experts shared across rows,
    out-of-range ids reading experts 0 and E - 1 beside in-range rows of the same
    experts, and (k >= 24) groups larger than one tile. Within the gate, and every row bitwise its
    value computed alone (k = 1): a row's bits do not depend on which rows share
    its expert."""
    x, _, w, scales = _rand(40 + k, k=k)
    ids = np.random.default_rng(80 + k).integers(-2, E + 2, size=k).astype(np.int32)
    groups = Counter(ids.tolist())  # the kernel groups rows by raw id
    if k >= 24:
        assert max(groups.values()) > TILE, groups  # the case under test: a group spans tiles
    out = _qmv(x, ids, w, scales)
    rel = rel_err(out, _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv k={k} shared ids (largest group {max(groups.values())}): rel={rel:.3e}")
    assert rel <= TOL
    for s in range(k):
        assert np.array_equal(out[s : s + 1], _qmv(x[s : s + 1], ids[s : s + 1], w, scales)), s


def test_qmv_batched_decode_rows() -> None:
    """k = 48, the batched decode step's ``B * top-k`` at B = 8: within the gate,
    and each row bitwise what a k = 6 call computes for it (a row reads only its
    own ``x``, expert and scales, so its bits cannot depend on how many ride along)."""
    x, ids, w, scales = _rand(9, k=48)
    out = _qmv(x, ids, w, scales)
    rel = rel_err(out, _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv k=48: rel={rel:.3e}")
    assert rel <= TOL
    for start in range(0, 48, 6):
        rows = slice(start, start + 6)
        assert np.array_equal(out[rows], _qmv(x[rows], ids[rows], w, scales)), start


def test_qmv_amax_edge() -> None:
    """Every weight at the quantization edge: alternating exactly +-127."""
    x, ids, _, scales = _rand(4)
    w = np.where(np.arange(K) % 2 == 0, 127, -127)[None, None, :]
    w = np.broadcast_to(w, (E, N, K)).astype(np.int8)
    rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
    print(f"[uocr] qmv amax edge: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_all_zero_groups() -> None:
    """A zero weight group and a zero scale group contribute exactly nothing."""
    x, ids, w, scales = _rand(5)
    w = w.copy()
    scales = scales.copy()
    w[:, :, G : 2 * G] = 0  # group 1: zero weights
    scales[:, :, 2] = 0.0  # group 2: zero scale
    rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
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
        rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
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
    rel = rel_err(out_raw, _qmv_ref(x, raw, w, scales))
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
    rel = rel_err(_qmv(x, ids, w, scales), _qmv_ref(x, ids, w, scales))
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


def test_dequant_rows_not_a_multiple_of_the_simd_width() -> None:
    """K = 30 in groups of 6: SIMD packs that span a group boundary or end a row
    early, and unaligned loads -- still bit-exact."""
    rng = np.random.default_rng(15)
    w = rng.integers(-127, 128, size=(E, N, 30)).astype(np.int8)
    scales = (10.0 ** rng.uniform(-3.0, 1.0, size=(E, N, 5))).astype(np.float32)
    for e in (0, E - 1):
        assert np.array_equal(_dequant_bits(w, scales, e), _dequant_ref_bits(w, scales, e)), f"expert {e}"
    print("[uocr] dequant K=30 G=6: bit-exact")


@pytest.mark.skipif(GPU, reason="a device copy is always aligned; only the CPU runs on an input where it lies")
def test_dequant_misaligned_weights() -> None:
    """Weights one byte past a 16-byte boundary take the unclaimed load: still bit-exact.

    It checks the values, not the predicate: a claimed load on a misaligned
    address does not fault on AArch64 (this CPU), and the dequantization is the
    same lane-wise multiply either way, so a wrong host check would pass here too."""
    _, _, w, scales = _rand(16)
    assert np.array_equal(_dequant_bits(off_boundary(w), scales, 3), _dequant_ref_bits(w, scales, 3))
    print("[uocr] dequant misaligned weights: bit-exact")


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
@pytest.mark.parametrize("k", [6, 48], ids=["decode", "decode_b8"])
@pytest.mark.parametrize(
    "n,kdim", [(896, 1280), (1280, 896)], ids=["gate_up", "down"]
)
def test_real_shape_smoke(n: int, kdim: int, k: int) -> None:
    """Production stacks ``[64, 896, 1280]`` (gate/up) and ``[64, 1280, 896]``
    (down) at G=128, one stack at a time; models are built uncached and dropped
    so the two shapes never coexist.

    Ids are laid out like the decode step's: ``k / 6`` tokens, six distinct
    experts each, from a 28-expert pool, so about half of the B = 8 (k = 48)
    rows share an expert with another token; there every token's six rows must
    also be bitwise what a k = 6 call computes for that token alone. The
    dequantization is checked once, at k = 6."""
    num_experts, gsize, top_k = 64, 128, 6
    groups = kdim // gsize
    rng = np.random.default_rng(20260909 + k)
    x = rng.standard_normal((k, kdim)).astype(np.float32)
    pool = rng.permutation(num_experts)[:28]
    ids = np.concatenate([rng.choice(pool, top_k, replace=False) for _ in range(k // top_k)]).astype(np.int32)
    w = rng.integers(-127, 128, size=(num_experts, n, kdim)).astype(np.int8)
    scales = rng.uniform(1e-3, 3e-2, size=(num_experts, n, groups)).astype(
        np.float32
    )

    qmv_model = _load_qmv(k, num_experts, n, kdim, groups)
    out = _qmv(x, ids, w, scales, model=qmv_model)
    del qmv_model
    rel = rel_err(out, _qmv_ref(x, ids, w, scales))
    print(f"[uocr] real-shape qmv [{num_experts},{n},{kdim}] k={k} ({len(set(ids.tolist()))} experts): rel={rel:.3e}")
    assert rel <= TOL
    if k > top_k:
        qmv_model = _load_qmv(top_k, num_experts, n, kdim, groups)
        for start in range(0, k, top_k):
            rows = slice(start, start + top_k)
            assert np.array_equal(out[rows], _qmv(x[rows], ids[rows], w, scales, model=qmv_model)), start
        del qmv_model
        return

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
