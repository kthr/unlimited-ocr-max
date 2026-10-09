"""Numeric checks for the bf16 MoE Mojo kernel against a float64 reference.

Every test compiles a Mojo custom op through a MAX ``InferenceSession``, so the
whole module is marked ``slow`` and excluded from CI. The device is CPU unless
``UOCR_TEST_DEVICE=gpu``, which also enables the real-shape smoke.

``moe_bf16_qmv`` is compared against a float64 reference at a 1e-5 relative
gate (max abs difference over the reference's amax -- the kernel accumulates in
fp32). The weights travel as raw bfloat16 bit patterns (numpy has no bfloat16),
so the reference reads exactly the values the kernel reads. Out-of-range expert
ids are CLAMPED to ``[0, E)`` by the kernel (a kernel closure cannot raise);
the reference mirrors that.

The kernel reads each selected expert once for all the rows routed to it
(KON-239), so rows that share an expert are the case to break: with ``E = 5``
every k >= 6 call repeats an expert, the k = 24 and 48 calls put more rows on
one expert than a tile holds, and every row must stay bitwise what it is when
computed alone.

The kernel's vector loads claim 16-byte alignment (KON-240) when a host-side
check finds both bases on that boundary and both tensors packed; a buffer off
it -- the CPU runs on an input where it lies -- takes the same 16-wide loop
with loads that claim one element, so the bits do not depend on the address.
A call with no rows (k = 0, as a device-graph warm-up launch can be) returns
an empty result.

On the GPU the op runs its hand-launched kernel (TODO ID 52): one SIMD group
per output column and tile of ``GPU_TILE`` rows, in another summation order,
so every check here holds there with the GPU's own bits. Its tile is smaller
than ``TILE``, so the k >= 24 groups span its tiles too. The K = 30 call and
the k = 0 call run the elementwise kernel on the GPU as well.
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

from _harness import GPU, bf16_bits, bf16_values, clamp, device_ref, driver, off_boundary, rel_err, session

pytestmark = pytest.mark.slow

TOL = 1e-5

# Small shapes shared by the adversarial cases: one compiled graph per shape,
# every weight is a runtime input, so all cases reuse the same models. K is a
# multiple of the kernel's 16-wide vector load; K_ODD is not (the scalar path).
E, N, K = 5, 8, 32
K_ODD = 30
#: Rows the kernel serves per read of a weight row, read from ``moe_bf16.mojo`` so it cannot drift.
TILE = int(re.search(r"^comptime TILE = (\d+)$", (MOJO_KERNELS / "moe_bf16.mojo").read_text(), re.M).group(1))

_MODELS: dict[tuple, object] = {}


# --------------------------------------------------------------------------
# graph builder and runner
# --------------------------------------------------------------------------


def _load_qmv(k: int, e: int, n: int, kdim: int, *, x_kdim: int | None = None):
    """Compile ``out fp32 [k, n] = moe_bf16_qmv(x, expert_ids, w)``; ``x_kdim`` mis-sizes ``x`` on purpose."""
    dref = device_ref()
    with Graph(
        f"test_moe_bf16_qmv_{k}x{e}x{n}x{kdim}_{x_kdim}",
        input_types=[
            TensorType(DType.float32, [k, kdim if x_kdim is None else x_kdim], device=dref),
            TensorType(DType.int32, [k], device=dref),
            TensorType(DType.bfloat16, [e, n, kdim], device=dref),
        ],
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        graph.output(
            ops.custom(
                "moe_bf16_qmv",
                device=dref,
                values=[inp.tensor for inp in graph.inputs],
                out_types=[TensorType(DType.float32, [k, n], device=dref)],
            )[0]
        )
    return session().load(graph)


def _bf16_buffer(bits: np.ndarray) -> Buffer:
    """A bfloat16 buffer on the test device holding ``bits`` (uint16 bit patterns)."""
    host = Buffer.from_numpy(np.ascontiguousarray(bits, dtype=np.uint16)).view(DType.bfloat16)
    return host.to(driver())


def _execute(model, x: np.ndarray, ids: np.ndarray, w_bits: np.ndarray):
    buffers = [Buffer.from_numpy(np.ascontiguousarray(a)).to(driver()) for a in (x, ids)]
    return model.execute(*buffers, _bf16_buffer(w_bits))[0]


def _qmv(x, ids, w_bits, *, model=None) -> np.ndarray:
    if model is None:
        key = (x.shape[0], *w_bits.shape)
        if key not in _MODELS:
            _MODELS[key] = _load_qmv(x.shape[0], *w_bits.shape)
        model = _MODELS[key]
    return _execute(model, x, ids, w_bits).to(CPU()).to_numpy()


# --------------------------------------------------------------------------
# the float64 reference
# --------------------------------------------------------------------------


def _qmv_ref(x, ids, w_bits) -> np.ndarray:
    """Float64 reference of the qmv contract, expert ids clamped."""
    xf, wf = x.astype(np.float64), bf16_values(w_bits)
    num_experts = w_bits.shape[0]
    return np.stack([wf[clamp(ids[s], num_experts)] @ xf[s] for s in range(x.shape[0])])


def _rand(seed: int, k: int = 6, kdim: int = K, num_experts: int = E, n: int = N):
    """Random case: weights spread over four decades (with random signs), so a
    wrong K offset or a mis-read expert mixes terms of very different size."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((k, kdim)).astype(np.float32)
    ids = rng.integers(0, num_experts, size=k).astype(np.int32)
    magnitude = 10.0 ** rng.uniform(-3.0, 1.0, size=(num_experts, n, kdim))
    w = (magnitude * rng.choice([-1.0, 1.0], size=magnitude.shape)).astype(np.float32)
    return x, ids, bf16_bits(w)


# --------------------------------------------------------------------------
# moe_bf16_qmv
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qmv_random(seed: int) -> None:
    x, ids, w = _rand(seed)
    rel = rel_err(_qmv(x, ids, w), _qmv_ref(x, ids, w))
    print(f"[uocr] bf16 qmv random seed={seed}: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_k_not_a_multiple_of_the_vector_width() -> None:
    """K = 30 takes the scalar path: same contract, same gate (six ids over five
    experts, so at least one is shared)."""
    x, ids, w = _rand(3, kdim=K_ODD)
    rel = rel_err(_qmv(x, ids, w), _qmv_ref(x, ids, w))
    print(f"[uocr] bf16 qmv K={K_ODD}: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_k1() -> None:
    x, ids, w = _rand(4, k=1)
    rel = rel_err(_qmv(x, ids, w), _qmv_ref(x, ids, w))
    print(f"[uocr] bf16 qmv k=1: rel={rel:.3e}")
    assert rel <= TOL


@pytest.mark.parametrize("k", [1, 6, 12, 24, 48])
def test_qmv_rows_sharing_experts_are_their_lone_values(k: int) -> None:
    """``k`` rows with ids drawn from ``[-2, E + 2)``: experts shared across rows,
    out-of-range ids reading experts 0 and E - 1 beside in-range rows of the same
    experts, and (k >= 24) groups larger than one tile. Within the gate, and every row bitwise its
    value computed alone (k = 1): a row's bits do not depend on which rows share
    its expert, the property the batched decode's load independence rests on."""
    x, _, w = _rand(40 + k, k=k)
    ids = np.random.default_rng(80 + k).integers(-2, E + 2, size=k).astype(np.int32)
    groups = Counter(ids.tolist())  # the kernel groups rows by raw id
    if k >= 24:
        assert max(groups.values()) > TILE, groups  # the case under test: a group spans tiles
    out = _qmv(x, ids, w)
    rel = rel_err(out, _qmv_ref(x, ids, w))
    print(f"[uocr] bf16 qmv k={k} shared ids (largest group {max(groups.values())}): rel={rel:.3e}")
    assert rel <= TOL
    for s in range(k):
        assert np.array_equal(out[s : s + 1], _qmv(x[s : s + 1], ids[s : s + 1], w)), s


def test_qmv_batched_decode_rows() -> None:
    """k = 48, the batched decode step's ``B * top-k`` at B = 8: within the gate,
    and each row bitwise what a k = 6 call computes for it (a row reads only its
    own ``x`` and expert, so its bits cannot depend on how many ride along)."""
    x, ids, w = _rand(9, k=48)
    out = _qmv(x, ids, w)
    rel = rel_err(out, _qmv_ref(x, ids, w))
    print(f"[uocr] bf16 qmv k=48: rel={rel:.3e}")
    assert rel <= TOL
    for start in range(0, 48, 6):
        rows = slice(start, start + 6)
        assert np.array_equal(out[rows], _qmv(x[rows], ids[rows], w)), start


def test_qmv_one_hot_rows_pick_exactly_the_indexed_element() -> None:
    """``w[e, n, :]`` is one-hot at ``(n + 3 e) % K``: every output is exactly one
    ``x`` element, so a wrong expert, row, K offset or vector lane is an exact miss."""
    k = 6
    x = (np.arange(k * K, dtype=np.float32).reshape(k, K) + 1.0) * np.float32(0.5)
    ids = np.array([4, 0, 2, 3, 1, 4], dtype=np.int32)
    w = np.zeros((E, N, K), dtype=np.float32)
    for e in range(E):
        for n in range(N):
            w[e, n, (n + 3 * e) % K] = 1.0
    out = _qmv(x, ids, bf16_bits(w))
    want = np.array([[x[s, (n + 3 * ids[s]) % K] for n in range(N)] for s in range(k)], dtype=np.float32)
    assert np.array_equal(out, want)


def test_qmv_all_zero_weights_are_exactly_zero() -> None:
    x, ids, w = _rand(5)
    assert not np.any(_qmv(x, ids, np.zeros_like(w)))


def test_qmv_expert_id_permutations_and_repeats() -> None:
    x, _, w = _rand(6)
    reference_ids = np.arange(E, dtype=np.int32)  # k == 6 > E covers a repeat
    for ids in (
        np.array([4, 2, 0, 1, 3, 2], dtype=np.int32),  # permutation + repeat
        np.array([3, 3, 3, 3, 3, 3], dtype=np.int32),  # one expert for all
        np.concatenate([reference_ids, reference_ids[:1]]),  # 0..4, 0
    ):
        rel = rel_err(_qmv(x, ids, w), _qmv_ref(x, ids, w))
        print(f"[uocr] bf16 qmv ids={ids.tolist()}: rel={rel:.3e}")
        assert rel <= TOL
    # Steering evidence: two id vectors must not produce the same output.
    a = _qmv(x, np.full(6, 0, dtype=np.int32), w)
    b = _qmv(x, np.full(6, 4, dtype=np.int32), w)
    assert np.max(np.abs(a - b)) > 0.0


def test_qmv_out_of_range_ids_clamp() -> None:
    """Out-of-range ids read the clamped expert -- bitwise the same output."""
    x, _, w = _rand(7)
    raw = np.array([-1, E, E + 2, -100, 2, 4], dtype=np.int32)
    clamped = np.clip(raw, 0, E - 1).astype(np.int32)
    out_raw = _qmv(x, raw, w)
    assert np.array_equal(out_raw, _qmv(x, clamped, w))
    rel = rel_err(out_raw, _qmv_ref(x, raw, w))
    print(f"[uocr] bf16 qmv clamp: rel={rel:.3e}")
    assert rel <= TOL


def test_qmv_zero_rows() -> None:
    """k = 0: an empty ``[0, N]`` result, no error."""
    out = _qmv(np.zeros((0, K), dtype=np.float32), np.zeros(0, dtype=np.int32), _rand(10)[2])
    assert out.shape == (0, N)


@pytest.mark.skipif(GPU, reason="a device copy is always aligned; only the CPU runs on an input where it lies")
@pytest.mark.parametrize("off", ["w", "x", "wx"])
def test_qmv_misaligned_inputs_give_the_aligned_bits(off: str) -> None:
    """Weights and/or activations one element past a 16-byte boundary: the host
    check sends the call down the unclaimed loads, which sum in the vector loop's
    order, so the bits are the aligned call's. The bits cannot show which loads
    ran -- the point of the path is that they do not -- so this checks the values
    a misaligned call returns, not the predicate."""
    x, ids, w = _rand(11)
    out = _qmv(off_boundary(x) if "x" in off else x, ids, off_boundary(w) if "w" in off else w)
    assert np.array_equal(out, _qmv(x, ids, w))
    print(f"[uocr] bf16 qmv misaligned {off}: bitwise the aligned call")


def test_qmv_rejects_a_k_mismatch() -> None:
    """The host-side shape raises are load-bearing: an ``x`` narrower than the
    stack's K must refuse loudly instead of reading past the row."""
    with pytest.raises(Exception, match=r"moe_bf16_qmv: w and x disagree on K"):
        model = _load_qmv(2, 3, 4, 32, x_kdim=24)
        model.execute(
            Buffer.from_numpy(np.zeros((2, 24), dtype=np.float32)).to(driver()),
            Buffer.from_numpy(np.zeros(2, dtype=np.int32)).to(driver()),
            _bf16_buffer(np.zeros((3, 4, 32), dtype=np.uint16)),
        )


# --------------------------------------------------------------------------
# real-shape smoke (GPU only): the production expert stacks, one at a time
# --------------------------------------------------------------------------


@pytest.mark.skipif(not GPU, reason="real-shape smoke needs UOCR_TEST_DEVICE=gpu")
@pytest.mark.parametrize("k", [6, 48], ids=["decode", "decode_b8"])
@pytest.mark.parametrize("n,kdim", [(896, 1280), (1280, 896)], ids=["gate_up", "down"])
def test_real_shape_smoke(n: int, kdim: int, k: int) -> None:
    """Production stacks ``[64, 896, 1280]`` (gate/up) and ``[64, 1280, 896]``
    (down) at the batch-1 (k = 6) and B = 8 (k = 48) row counts; models are
    built uncached and dropped so the shapes never coexist.

    Ids are laid out like the decode step's: ``k / 6`` tokens, six distinct
    experts each, drawn from a 28-expert pool so that about half of the B = 8
    rows share an expert with another token. At k = 48 every token's six rows
    must also be bitwise what a k = 6 call computes for that token alone."""
    num_experts, top_k = 64, 6
    rng = np.random.default_rng(20261003 + k)
    x = rng.standard_normal((k, kdim)).astype(np.float32)
    pool = rng.permutation(num_experts)[:28]
    ids = np.concatenate([rng.choice(pool, top_k, replace=False) for _ in range(k // top_k)]).astype(np.int32)
    w = bf16_bits((0.05 * rng.standard_normal((num_experts, n, kdim))).astype(np.float32))

    model = _load_qmv(k, num_experts, n, kdim)
    out = _qmv(x, ids, w, model=model)
    del model
    rel = rel_err(out, _qmv_ref(x, ids, w))
    print(f"[uocr] real-shape bf16 qmv [{num_experts},{n},{kdim}] k={k} ({len(set(ids.tolist()))} experts): rel={rel:.3e}")
    assert rel <= TOL
    if k > top_k:
        model = _load_qmv(top_k, num_experts, n, kdim)
        for start in range(0, k, top_k):
            rows = slice(start, start + top_k)
            assert np.array_equal(out[rows], _qmv(x[rows], ids[rows], w, model=model)), start
        del model
