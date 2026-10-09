"""KON-237: the KV page pool, the paged ring cache and the one paged attention op per layer.

Every check here is model-free. The pool and the cache's ring state are host
logic; the graph checks stage the package's own pieces -- the decode graph's
page-pool inputs (:func:`~unlimited_ocr_max.graphs.paged_kv_input_types`, built
by :func:`~unlimited_ocr_max.graphs.paged_kv_from_inputs`), fed by
:meth:`~unlimited_ocr_max.kv_cache.KvPagePool.step`, and
:func:`~unlimited_ocr_max.decoder.paged_decode_attention` -- in a graph whose
q / k_new / v_new are inputs, so nothing but the paged store and attention is
under test. The reference everywhere is the per-row chain the paged op
replaced (``Attention.decode_rows`` until KON-237): the new row substituted at
the write index, scores times scale, softmax, ``@ v``, in float64.

The graph checks run on CPU, the device every CI leg has; the GPU variants are
``slow`` and need an accelerator.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache.utils import padded_lut_cols

from unlimited_ocr_max.decoder import paged_decode_attention
from unlimited_ocr_max.graphs import paged_kv_from_inputs, paged_kv_input_types
from unlimited_ocr_max.kv_cache import MAX_PROMPT_TOKENS, PAGE_SIZE, KvCache, KvPagePool, pages_for, to_host
from unlimited_ocr_max.pipeline import UnlimitedOcrPipeline

from test_weight_sharing import gpu_only


def _pool(*, pages: int, layers: int = 2, heads: int = 2, head_dim: int = 4, device: Any = None) -> KvPagePool:
    return KvPagePool(num_layers=layers, num_kv_heads=heads, head_dim=head_dim, pages=pages, device=device or CPU())


def _rows(rng: np.random.Generator, pool: KvPagePool, n: int) -> list[np.ndarray]:
    return [rng.standard_normal((n, pool.num_kv_heads, pool.head_dim)).astype(np.float32) for _ in range(pool.num_layers)]


# --------------------------------------------------------------------------
# the pool: sizing, the null page, allocation and refusal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_batch_size", "prefill_len", "window", "pages"),
    [
        (1, MAX_PROMPT_TOKENS, 128, 5 + 1),  # served: 640 rows are 5 pages, plus the null page
        (8, MAX_PROMPT_TOKENS, 128, 8 * 5 + 1),
        (1, 277, 128, 4 + 1),  # the default prompt alone: 405 rows are 4 pages
        (8, 277, 128, 8 * 4 + 1),
        (8, 282, 128, 8 * 4 + 1),  # 410 rows still fit 4 pages
        (2, 32, 16, 2 * 1 + 1),
        (1, 1, 127, 1 + 1),  # exactly one page
        (3, 1, 128, 3 * 2 + 1),  # one row over a page
    ],
)
def test_the_pool_is_max_batch_requests_of_pages_plus_the_null_page(
    max_batch_size: int, prefill_len: int, window: int, pages: int
) -> None:
    """``max_batch_size * ceil((prefill_len + window) / 128) + 1`` pages, fp32, MAX's block layout, null page last."""
    pool = KvPagePool.sized_for(
        max_batch_size=max_batch_size, prefill_len=prefill_len, window=window, num_layers=3, num_kv_heads=2,
        head_dim=4, device=CPU(),
    )
    assert PAGE_SIZE == 128
    assert pages_for(prefill_len + window) * max_batch_size + 1 == pool.pages == pages
    assert pool.kv_blocks.shape == (pages, 2, 3, PAGE_SIZE, 2, 4)
    assert pool.kv_blocks.dtype == DType.float32
    assert pool.null_page == pages - 1
    assert pool.free_pages == pages - 1
    # Exactly max_batch_size requests of that prompt fit, and no more.
    caches = [pool.allocate(prefill_len=prefill_len, window=window) for _ in range(max_batch_size)]
    assert all(pool.null_page not in cache.pages for cache in caches)
    assert sorted(page for cache in caches for page in cache.pages) == list(range(pages - 1))
    with pytest.raises(RuntimeError, match="the KV page pool cannot hold this request"):
        pool.allocate(prefill_len=prefill_len, window=window)


@pytest.mark.parametrize(
    ("max_batch_size", "seq_len", "pages"),
    [
        (1, 277, 1 * 5 + 1),  # served at batch 1: 6 pages, 90 MiB on the real decoder
        (8, 277, 8 * 5 + 1),  # served at batch 8: 41 pages, 615 MiB
        (2, 8, 2 * 5 + 1),  # a shorter prompt does not shrink the slots
        (2, 700, 2 * 7 + 1),  # a pipeline built around a longer prompt (gundam) sizes for it: 828 rows, 7 pages
    ],
)
def test_the_pipelines_pool_has_max_batch_slots_for_max_prompt_tokens(max_batch_size: int, seq_len: int, pages: int) -> None:
    """The pool is sized from :data:`MAX_PROMPT_TOKENS` and the ring, not from the prompt the pipeline is built around.

    Every slot then holds the longest prompt the tokenizer admits: exactly
    ``max_batch_size`` requests of ``MAX_PROMPT_TOKENS`` tokens fit, and no more.
    """
    decoder = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, head_dim=2, vocab_size=8, sliding_window_size=128)
    pipeline = UnlimitedOcrPipeline(
        SimpleNamespace(decoder=decoder, dtype=DType.bfloat16),
        vision_state_dict={},
        language_state_dict={},
        seq_len=seq_len,
        max_batch_size=max_batch_size,
        session=SimpleNamespace(),
    )
    pool = pipeline.kv_pool
    assert pool is pipeline.kv_pool  # built once
    assert pool.pages == pages and pool.null_page == pages - 1
    prompt = max(seq_len, MAX_PROMPT_TOKENS)
    caches = [pool.allocate(prefill_len=prompt, window=pipeline.window) for _ in range(max_batch_size)]
    assert pool.free_pages == 0 and all(len(cache.pages) == (pages - 1) // max_batch_size for cache in caches)
    with pytest.raises(RuntimeError, match="the KV page pool cannot hold this request"):
        pool.allocate(prefill_len=1, window=pipeline.window)


def test_run_prefill_takes_its_pages_before_it_compiles_and_gives_them_back_on_failure() -> None:
    """A request the pool cannot hold is refused before the prefill graph is compiled; a failed compile frees the pages."""
    decoder = SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, head_dim=2, hidden_size=2, vocab_size=8, sliding_window_size=128
    )
    pipeline = UnlimitedOcrPipeline(
        SimpleNamespace(decoder=decoder, dtype=DType.bfloat16),
        vision_state_dict={},
        language_state_dict={},
        seq_len=277,
        session=SimpleNamespace(),
    )
    compiled: list[int] = []

    def prefill_graph_for(seq_len: int) -> Any:
        compiled.append(seq_len)
        raise RuntimeError("compile failed")

    pipeline.prefill_graph_for = prefill_graph_for
    tokens, embeds = np.zeros(277, dtype=np.int64), np.zeros((pipeline.n_image_tokens, 2), dtype=np.float32)
    free = pipeline.kv_pool.free_pages

    held = pipeline.kv_pool.allocate(prefill_len=MAX_PROMPT_TOKENS, window=pipeline.window)  # the only slot
    with pytest.raises(RuntimeError, match="the KV page pool cannot hold this request"):
        pipeline.run_prefill(tokens, embeds)
    assert compiled == []
    held.release()

    with pytest.raises(RuntimeError, match="compile failed"):
        pipeline.run_prefill(tokens, embeds)
    assert compiled == [277] and pipeline.kv_pool.free_pages == free


def test_the_pool_refuses_a_request_that_needs_more_pages_than_are_free() -> None:
    """Refused with the numbers in the message; nothing is taken by the refusal, and released pages are reusable."""
    pool = KvPagePool.sized_for(
        max_batch_size=2, prefill_len=200, window=50, num_layers=1, num_kv_heads=1, head_dim=2, device=CPU()
    )
    assert pool.pages == 2 * 2 + 1
    first = pool.allocate(prefill_len=200, window=50)
    longer = r"a 400-token prompt and a 50-slot ring need 4 pages of 128 rows, but only 2 of the pool's 4 are free"
    with pytest.raises(RuntimeError, match=longer):
        pool.allocate(prefill_len=400, window=50)
    assert pool.free_pages == 2
    second = pool.allocate(prefill_len=200, window=50)
    assert set(first.pages).isdisjoint(second.pages) and pool.free_pages == 0
    first.release()
    first.release()  # idempotent
    assert first.pages == [] and pool.free_pages == 2
    third = pool.allocate(prefill_len=10, window=50)
    assert len(third.pages) == 1 and pool.free_pages == 1
    with pytest.raises(ValueError, match="not allocated pages"):
        pool.free([third.pages[0], pool.null_page])
    with pytest.raises(ValueError, match="not allocated pages"):
        pool.free([pool._free[0]])
    with pytest.raises(ValueError, match="at least one request page and the null page"):
        _pool(pages=1)


# --------------------------------------------------------------------------
# the cache: seed into the pages, host state only afterwards
# --------------------------------------------------------------------------


def test_seed_copies_the_prefix_into_the_pages_and_nothing_else() -> None:
    """A 130-row prefix spans two pages handed out out of order; ``host_rows`` reads it back in logical order."""
    pool = _pool(pages=6)
    blocker = pool.allocate(prefill_len=1, window=1)  # takes page 0, so the cache's pages are not 0, 1, ...
    cache = pool.allocate(prefill_len=130, window=60)
    blocker.release()
    assert cache.pages == [1, 2] and cache.max_seq_len == 2 * PAGE_SIZE
    rng = np.random.default_rng(3)
    keys, values = _rows(rng, pool, 130), _rows(rng, pool, 130)
    cache.seed(keys, values)

    assert (cache.prefill_len, cache.length, cache.position, cache.ring_pos) == (130, 130, 130, 0)
    host_keys, host_values = cache.host_rows()
    for layer in range(pool.num_layers):
        assert np.array_equal(host_keys[layer][:130], keys[layer])
        assert np.array_equal(host_values[layer][:130], values[layer])
        assert not host_keys[layer][130:].any() and not host_values[layer][130:].any()
    blocks = to_host(pool.kv_blocks).copy()
    blocks[cache.pages] = 0
    assert not blocks.any(), "seed wrote outside the cache's pages"

    with pytest.raises(ValueError, match="needs 251 rows, but its pages hold only 128"):
        pool.allocate(prefill_len=1, window=60).seed(_rows(rng, pool, 191), _rows(rng, pool, 191))


class _OldRing:
    """The ring as the pre-KON-237 ``KvCache`` kept it, verbatim: numpy rows, and ``append`` writes the row itself."""

    def __init__(self, keys: list[np.ndarray], values: list[np.ndarray], *, capacity: int, window: int) -> None:
        self.window = window
        self.prefill_len = self.length = self.position = int(keys[0].shape[0])
        self.ring_pos = 0
        self.keys = [np.zeros((capacity, *k.shape[1:]), np.float32) for k in keys]
        self.values = [np.zeros((capacity, *v.shape[1:]), np.float32) for v in values]
        for rows, src in ((self.keys, keys), (self.values, values)):
            for layer, array in enumerate(src):
                rows[layer][: self.length] = array

    @property
    def write_index(self) -> int:
        if self.length < self.prefill_len + self.window:
            return self.length
        return self.prefill_len + self.ring_pos

    @property
    def attend_len(self) -> int:
        return max(self.length, self.write_index + 1)

    def append(self, keys: list[np.ndarray], values: list[np.ndarray]) -> None:
        row = self.write_index
        appending = row == self.length
        for layer in range(len(keys)):
            self.keys[layer][row] = keys[layer]
            self.values[layer][row] = values[layer]
        if appending:
            self.length += 1
        else:
            self.ring_pos = (self.ring_pos + 1) % self.window
        self.position += 1


def test_append_moves_the_ring_state_step_by_step() -> None:
    """Warm-up appends, then two full ring wraps of overwrites: write index and attend length at every step, then the state."""
    pool = _pool(pages=3)
    rng = np.random.default_rng(5)
    cache = pool.allocate(prefill_len=7, window=4)
    cache.seed(_rows(rng, pool, 7), _rows(rng, pool, 7))
    seen = []
    for _ in range(4 + 2 * 4 + 1):
        seen.append((cache.write_index, cache.attend_len))
        cache.append()
    assert [row for row, _ in seen] == [7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7]
    assert [rows for _, rows in seen] == [8, 9, 10, 11] + [11] * 9  # one past the write while warming up, then the ring
    assert (cache.length, cache.ring_pos, cache.position) == (11, 1, 7 + 13)


def test_step_puts_each_row_on_its_pages_and_padding_on_the_null_page() -> None:
    """The step metadata: lookup table, both cache-length vectors and maxima, MAX's dispatch key; padding last, all null."""
    pool = _pool(pages=8)
    rng = np.random.default_rng(7)
    short = pool.allocate(prefill_len=5, window=4)
    short.seed(_rows(rng, pool, 5), _rows(rng, pool, 5))  # warming up: appends at row 5
    long = pool.allocate(prefill_len=200, window=100)  # 3 pages
    long.seed(_rows(rng, pool, 200), _rows(rng, pool, 200))
    for _ in range(100 + 3):  # a full ring, then three overwrites: writes row 203 of 300 attended
        long.append()
    assert (short.write_index, short.attend_len) == (5, 6)
    assert (long.write_index, long.attend_len) == (203, 300)

    step = pool.step([long, short], padding=2)

    null = pool.null_page
    assert step.lookup_table.dtype == np.uint32 and step.lookup_table.shape == (4, padded_lut_cols(3))
    assert step.lookup_table[0].tolist() == [*long.pages, *[null] * (padded_lut_cols(3) - 3)]
    assert step.lookup_table[1].tolist() == [*short.pages, *[null] * (padded_lut_cols(3) - 1)]
    assert (step.lookup_table[2:] == null).all()  # padding rows: only the null page
    assert step.attend_lengths.dtype == step.write_index.dtype == step.attend_max.dtype == np.uint32
    assert step.attend_lengths.tolist() == [299, 5, 0, 0]  # attend_len - 1; a padding row attends its one row
    assert step.write_index.tolist() == [203, 5, 0, 0]  # a padding row stores at the null page's slot 0
    assert step.attend_max.tolist() == [300] and step.write_max.tolist() == [204]
    assert step.dispatch.dtype == np.int64 and step.dispatch.tolist() == [4, 1, 1, 300]  # num_partitions pinned to 1

    alone = pool.step([short])
    assert alone.lookup_table.shape == (1, padded_lut_cols(1)) and alone.dispatch.tolist() == [1, 1, 1, 6]
    only_padding = pool.step([], padding=1)
    assert (only_padding.lookup_table == null).all() and only_padding.write_max.tolist() == [1]

    short.release()
    with pytest.raises(ValueError, match="released"):
        pool.step([short])


# --------------------------------------------------------------------------
# the paged attention op: one op for all B rows, against the old chain's math
# --------------------------------------------------------------------------


#: The op runs at layers 0 and 2 of a 3-layer pool at the real head dims, the same new rows at both.
_POOL_LAYERS, _LAYERS, _HEADS, _HEAD_DIM = 3, (0, 2), 10, 128
_SCALE = 1.0 / math.sqrt(_HEAD_DIM)


def _attention_graph(batch: int, device: DeviceRef) -> Graph:
    """``q, k_new, v_new [B, H, D]`` + the page-pool inputs -> :func:`paged_decode_attention` at each of :data:`_LAYERS`."""
    # The three fields the page-pool inputs read off a DecoderConfig.
    config = SimpleNamespace(num_hidden_layers=_POOL_LAYERS, num_key_value_heads=_HEADS, head_dim=_HEAD_DIM)
    qkv = [TensorType(DType.float32, [batch, _HEADS, _HEAD_DIM], device=device)] * 3
    tag = "cpu" if device.is_cpu() else "gpu"
    with Graph(f"kon237_paged_attention_b{batch}_{tag}", input_types=[*qkv, *paged_kv_input_types(config, batch, device)]) as graph:
        q, k_new, v_new = (value.tensor for value in graph.inputs[:3])
        kv = paged_kv_from_inputs(list(graph.inputs[3:]), config, batch, device)
        graph.output(*(paged_decode_attention(q, k_new, v_new, kv=kv, layer_idx=i, scale=_SCALE) for i in _LAYERS))
    return graph


#: ``device -> (driver, DeviceRef, session)`` and ``(device, batch) -> model``: one compile per graph per process.
_SESSIONS: dict[str, tuple[Any, DeviceRef, InferenceSession]] = {}
_MODELS: dict[tuple[str, int], Any] = {}


def _session(device: str) -> tuple[Any, DeviceRef, InferenceSession]:
    if device not in _SESSIONS:
        if device == "gpu":
            from max.driver import Accelerator

            driver, dref = Accelerator(), DeviceRef.GPU(0)
        else:
            driver, dref = CPU(), DeviceRef.CPU()
        _SESSIONS[device] = (driver, dref, InferenceSession(devices=[driver]))
    return _SESSIONS[device]


def _attention_model(device: str, batch: int) -> Any:
    if (device, batch) not in _MODELS:
        _, dref, session = _session(device)
        _MODELS[device, batch] = session.load(_attention_graph(batch, dref))
    return _MODELS[device, batch]


def _attention_pool(device: str, pages: int) -> KvPagePool:
    return _pool(pages=pages, layers=_POOL_LAYERS, heads=_HEADS, head_dim=_HEAD_DIM, device=_session(device)[0])


def _run_attention(device: str, pool: KvPagePool, caches: list[KvCache], padding: int, q, k_new, v_new) -> list[np.ndarray]:
    """One execute over ``caches`` (+ ``padding`` padding rows) through :meth:`KvPagePool.step`; each layer's output, copied."""
    model = _attention_model(device, len(caches) + padding)
    step = pool.step(caches, padding)

    def dev(array: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(array)).to(pool.device)

    paged = [value if isinstance(value, Buffer) else dev(value) for value in step.graph_inputs(pool.kv_blocks)]
    return [to_host(out).copy() for out in model.execute(dev(q), dev(k_new), dev(v_new), *paged)]


def _chain_reference(q: np.ndarray, k_new: np.ndarray, v_new: np.ndarray, past_k: np.ndarray, past_v: np.ndarray,
                     write_index: int) -> np.ndarray:
    """float64 of the old chain for one row: the new k/v substituted at ``write_index``, ``softmax(q k^T * scale) @ v``."""
    k = past_k.astype(np.float64).copy()
    v = past_v.astype(np.float64).copy()
    k[write_index], v[write_index] = k_new, v_new
    scores = np.einsum("hd,lhd->hl", q.astype(np.float64), k) * _SCALE
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hl,lhd->hd", weights, v)


def _relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got.astype(np.float64) - want)) / np.max(np.abs(want)))


#: Parity rows as ``(prefill, window, decode steps)``, then the ``(attend_len, write_index)`` those steps leave.
#: An append: 410 rows attended, writing row 409, on the row's fourth page.
_APPEND = ((282, 128, 409 - 282), (410, 409))
#: An overwrite: a full ring, 405 rows attended, writing ring slot 23 at row 300.
_OVERWRITE = ((277, 128, 128 + 23), (405, 300))
#: A short append: a 1-token prefix, 2 rows attended, writing row 1.
_SHORT = ((1, 128, 0), (2, 1))


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=[pytest.mark.slow, gpu_only])])
@pytest.mark.parametrize(
    ("rows", "padding"),
    [([_APPEND], 0), ([_OVERWRITE], 0), ([_APPEND, _OVERWRITE, _SHORT], 1)],
    ids=["b1-append", "b1-overwrite", "b4"],
)
def test_paged_attention_matches_the_old_chain_in_float64(device: str, rows: list[Any], padding: int) -> None:
    """Append and overwrite rows at the real head dims (10 x 128): ``<= 1e-5`` relative to the old chain; the store is bitwise.

    ``B = 1`` runs an append row (:data:`_APPEND`) and, separately, an
    overwrite row (:data:`_OVERWRITE`). ``B = 4`` runs both and a short
    append (:data:`_SHORT`) beside a padding row on the null page. The pool
    hands its pages out in a shuffled order, and the op runs at layers 0 and 2
    of 3. Afterwards exactly the written rows moved: each row's new k/v at its
    write index, the padding row's at the null page's slot 0.
    """
    batch = len(rows) + padding
    pool = _attention_pool(device, pages=16)
    rng = np.random.default_rng(11 + batch)
    # Shuffle the free list: every page out, then back in a random order.
    singles = [pool.allocate(prefill_len=1, window=1) for _ in range(pool.pages - 1)]
    for index in rng.permutation(len(singles)):
        singles[index].release()
    caches = []
    for (prefill, window, steps), _ in rows:
        cache = pool.allocate(prefill_len=prefill, window=window)
        written = prefill + min(steps, window)  # the prefix and every row the ring has written, as data
        cache._write_prefix(_rows(rng, pool, written), _rows(rng, pool, written), written)
        cache.prefill_len = cache.length = cache.position = prefill
        for _ in range(steps):
            cache.append()
        caches.append(cache)
    assert sorted(page for cache in caches for page in cache.pages) != [p for cache in caches for p in cache.pages]
    assert [(c.attend_len, c.write_index) for c in caches] == [state for _, state in rows]
    assert [c.write_index < c.length for c in caches] == [row is _OVERWRITE for row in rows]  # in place vs append

    q, k_new, v_new = (rng.standard_normal((batch, _HEADS, _HEAD_DIM)).astype(np.float32) for _ in range(3))
    before = [cache.host_rows() for cache in caches]
    blocks_before = to_host(pool.kv_blocks).copy()
    outputs = _run_attention(device, pool, caches, padding, q, k_new, v_new)

    for got, layer in zip(outputs, _LAYERS, strict=True):
        for b, cache in enumerate(caches):
            keys, values = before[b]
            n = cache.attend_len
            want = _chain_reference(q[b], k_new[b], v_new[b], keys[layer][:n], values[layer][:n], cache.write_index)
            assert _relative(got[b], want) <= 1e-5, (layer, b, _relative(got[b], want))
        if padding:
            # A padding row attends only its own new row: the output is that row's value.
            assert _relative(got[-1], v_new[-1].astype(np.float64)) <= 1e-6

    after = to_host(pool.kv_blocks).copy()
    written = [(cache.pages[cache.write_index // PAGE_SIZE], cache.write_index % PAGE_SIZE) for cache in caches]
    written += [(pool.null_page, 0)] * padding
    for layer in _LAYERS:
        for b, (page, slot) in enumerate(written):
            assert np.array_equal(after[page, 0, layer, slot], k_new[b]), (layer, b)
            assert np.array_equal(after[page, 1, layer, slot], v_new[b]), (layer, b)
            after[page, :, layer, slot] = blocks_before[page, :, layer, slot]
    assert np.array_equal(after, blocks_before), "the store wrote outside the rows' write slots"


def test_a_cache_walks_two_full_ring_wraps_like_the_old_ring() -> None:
    """Two requests at different phases and two padding rows, through warm-up and two full wraps of a 40-slot ring, on CPU.

    Every step runs the paged store + attention (layers 0 and 2 of 3) through
    the pool's own :meth:`~unlimited_ocr_max.kv_cache.KvPagePool.step`, then
    :meth:`~unlimited_ocr_max.kv_cache.KvCache.append`; beside it
    :class:`_OldRing` keeps the pre-KON-237 ring. After every step each row
    the step attended, in every layer, must be bitwise the old ring's, and
    the output must be the old chain's attention over the old ring's rows.
    The prefixes are 100 and 120 rows, so both rings cross the 128-row page
    edge.
    """
    pool = _attention_pool("cpu", pages=5)
    window = 40
    rng = np.random.default_rng(13)
    caches, olds = [], []
    for prefill in (100, 120):
        keys, values = _rows(rng, pool, prefill), _rows(rng, pool, prefill)
        cache = pool.allocate(prefill_len=prefill, window=window)
        cache.seed(keys, values)
        caches.append(cache)
        olds.append(_OldRing(keys, values, capacity=prefill + window, window=window))

    steps = window + 2 * window + 3
    overwrites = [0, 0]
    for _ in range(steps):
        q, k_new, v_new = (rng.standard_normal((4, _HEADS, _HEAD_DIM)).astype(np.float32) for _ in range(3))
        outputs = _run_attention("cpu", pool, caches, 2, q, k_new, v_new)
        for got, layer in zip(outputs, _LAYERS, strict=True):
            for b, old in enumerate(olds):
                n = old.attend_len
                want = _chain_reference(q[b], k_new[b], v_new[b], old.keys[layer][:n], old.values[layer][:n], old.write_index)
                assert _relative(got[b], want) <= 1e-5, (layer, b)
        for b, (cache, old) in enumerate(zip(caches, olds, strict=True)):
            assert (cache.write_index, cache.attend_len) == (old.write_index, old.attend_len)
            overwrites[b] += cache.write_index < cache.length
            attended = cache.attend_len
            cache.append()
            # The old ring writes the step's rows itself, at the layers the op stored them.
            new_keys = [old.keys[layer][old.write_index].copy() for layer in range(_POOL_LAYERS)]
            new_values = [old.values[layer][old.write_index].copy() for layer in range(_POOL_LAYERS)]
            for layer in _LAYERS:
                new_keys[layer], new_values[layer] = k_new[b], v_new[b]
            old.append(new_keys, new_values)
            # Every row this step attended, the one it stored included, is the old ring's row.
            keys, values = cache.host_rows()
            for layer in range(_POOL_LAYERS):
                assert np.array_equal(keys[layer][:attended], old.keys[layer][:attended]), (b, layer)
                assert np.array_equal(values[layer][:attended], old.values[layer][:attended]), (b, layer)
    assert overwrites == [2 * window + 3] * 2  # every step after the warm-up overwrote in place
    assert [cache.position for cache in caches] == [100 + steps, 120 + steps]
