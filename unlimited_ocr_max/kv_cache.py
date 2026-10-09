"""The per-request KV cache: R-SWA ring state over pages of one package-owned page pool.

Rows ``0 .. prefill_len - 1`` are the pinned reference region; the next
``window`` rows are a ring the generated tokens overwrite in place, so capacity
is ``prefill_len + window`` however long the generation runs. The reference
appends while ``length < prefill_len + window`` and overwrites from then on, and
RoPE uses the true :attr:`KvCache.position`, which keeps counting after the
cache stops growing.

**Storage.** Every request's rows live in ONE fp32 :class:`KvPagePool` per
pipeline, on the pipeline's device (CPU included):
``kv_blocks [pages, 2, num_layers, PAGE_SIZE, n_kv_heads, head_dim]``, the
layout MAX's paged ops read (``KVCacheParams.shape_per_block``). Its **last**
page is the null page: it is never handed to a request, every unused
lookup-table entry points at it, and a decode step's padding rows run on it.
A :class:`KvCache` is a page list in that pool plus the ring state; logical
row ``r`` of a request is slot ``r % PAGE_SIZE`` of its page
``pages[r // PAGE_SIZE]``. The pool is the package's own, not MAX's KV manager:
that one is not ring-aware and sizes itself from free memory. A served pool
holds ``--max-batch-size`` slots, each for a prompt of up to
:data:`MAX_PROMPT_TOKENS` tokens plus the ring; the tokenizer refuses a longer
prompt before it reaches the model worker.

**Who writes what.** The prefill's rows are copied into the request's pages
once (:meth:`KvCache.seed`). After that the decode graph writes every new row
itself -- ``store_k/v_cache_ragged`` at :attr:`KvCache.write_index` -- and
attends rows ``[0, attend_len)`` with one ``flash_attention_ragged`` for all
rows of the step, so :meth:`KvCache.append` only moves the host state.
:meth:`KvPagePool.step` builds that step's metadata: the lookup table and two
cache-length vectors over the same ``kv_blocks``, ``write_index`` for the store
and ``attend_len - 1`` for the attention (under the causal mask the one query
then sees exactly ``attend_len`` rows). Warm-up (an append) and a full ring (an
in-place overwrite) are both that one mapping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device, batch_inplace_copy
from max.dtype import DType
from max.nn.kv_cache.utils import MHAAttnKey, padded_lut_cols

__all__ = ["MAX_PROMPT_TOKENS", "PAGE_SIZE", "KvCache", "KvPagePool", "PagedStep", "pages_for", "to_host"]

#: Rows per page of the pool. The owner's choice (KON-236/OQ-3): at page 16 the
#: decode attention was no faster, and 128 keeps the lookup table short.
PAGE_SIZE = 128

#: The longest prompt a served request may carry, in tokens: BOS, the page's
#: image placeholders (273 in ``base`` mode) and the text. The owner's choice
#: (KON-237 review): each request's slot of the pool is
#: ``pages_for(MAX_PROMPT_TOKENS + window)`` pages -- 5 at the 128-slot ring --
#: and :meth:`UnlimitedOcrTokenizer.new_context
#: <unlimited_ocr_max.tokenizer.UnlimitedOcrTokenizer.new_context>` refuses a
#: longer prompt with an HTTP 400 in the API process, so no admitted request
#: needs more pages than its slot.
MAX_PROMPT_TOKENS = 512


def pages_for(rows: int) -> int:
    """Pages that hold ``rows`` rows."""
    return math.ceil(rows / PAGE_SIZE)


@dataclass(eq=False)
class KvCache:
    """One request's ring state over its pages of :attr:`pool`. Every slot decision and state transition lives here."""

    pool: KvPagePool
    #: Page ids in logical order: logical row ``r`` is slot ``r % PAGE_SIZE`` of ``pages[r // PAGE_SIZE]``.
    pages: list[int]
    window: int
    length: int = 0
    prefill_len: int = 0
    ring_pos: int = 0
    position: int = 0

    @property
    def max_seq_len(self) -> int:
        """Rows the pages hold."""
        return len(self.pages) * PAGE_SIZE

    @property
    def write_index(self) -> int:
        """``length`` while appending, the ring slot afterwards."""
        if self.length < self.prefill_len + self.window:
            return self.length
        return self.prefill_len + self.ring_pos

    @property
    def attend_len(self) -> int:
        """Rows the step attends: one past the write index while warming up, else the whole ring."""
        return max(self.length, self.write_index + 1)

    def seed(self, keys: Sequence[Any], values: Sequence[Any]) -> None:
        """Copy the prefill's ``[seq_len, n_kv_heads, head_dim]`` rows per layer into the pages and pin them."""
        length = int(keys[0].shape[0])
        if length + self.window > self.max_seq_len:
            raise ValueError(
                f"a {self.window}-slot ring over a {length}-token prefix needs "
                f"{length + self.window} rows, but its pages hold only {self.max_seq_len}"
            )
        self._write_prefix(keys, values, length)
        self.prefill_len = length
        self.length = length
        self.position = length
        self.ring_pos = 0

    def append(self) -> None:
        """Advance past the row the decode step just stored at :attr:`write_index`: an append during warm-up, else the next ring slot."""
        if self.write_index == self.length:
            self.length += 1
        else:
            self.ring_pos = (self.ring_pos + 1) % self.window
        self.position += 1

    def release(self) -> None:
        """Give the pages back to the pool. Idempotent; the cache holds no rows afterwards."""
        pages, self.pages = self.pages, []
        if pages:
            self.pool.free(pages)

    def _write_prefix(self, keys: Sequence[Any], values: Sequence[Any], length: int) -> None:
        """Rows ``[0, length)`` of every layer's keys and values, into the pages.

        ``batch_inplace_copy`` is asynchronous and ``Buffer.from_numpy`` does
        not keep its source alive: ``staged`` must outlive the copy, and the
        ``synchronize`` is what guarantees it has completed (KON-125).
        """
        if len(keys) != self.pool.num_layers or len(values) != self.pool.num_layers:
            raise ValueError(f"expected {self.pool.num_layers} key and value layers, got {len(keys)} and {len(values)}")
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        staged: list[np.ndarray] = []
        for kv, per_layer in enumerate((keys, values)):
            for layer, array in enumerate(per_layer):
                rows = np.asarray(array)
                for start in range(0, length, PAGE_SIZE):
                    stop = min(start + PAGE_SIZE, length)
                    chunk = np.ascontiguousarray(rows[start:stop], dtype=np.float32)
                    staged.append(chunk)
                    dsts.append(self.pool.slots(self.pages[start // PAGE_SIZE], kv, layer, stop - start))
                    srcs.append(Buffer.from_numpy(chunk))
        batch_inplace_copy(dsts, srcs)
        self.pool.device.synchronize()
        del staged

    def host_rows(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Every layer's ``(keys, values)`` as host ``[max_seq_len, n_kv_heads, head_dim]`` copies, in logical row order.

        A harness accessor (a device-to-host copy of every page), never for a
        step loop. Rows past what the ring has written hold whatever the page
        held before. ``np.concatenate`` makes the copy, so the result never
        aliases the pool, on CPU either.
        """
        out: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        for kv in (0, 1):
            for layer in range(self.pool.num_layers):
                pages = [to_host(self.pool.slots(page, kv, layer, PAGE_SIZE)) for page in self.pages]
                out[kv].append(np.concatenate(pages, axis=0))
        return out


@dataclass(frozen=True)
class PagedStep:
    """One decode step's paged-attention metadata for ``B`` rows, host arrays in the decode graph's input order.

    ``B`` is the real rows then the padding rows (:meth:`KvPagePool.step`).
    """

    #: ``[B, padded_lut_cols(max pages)]`` uint32: row ``b``'s pages, then the null page.
    lookup_table: np.ndarray
    #: ``[B]`` uint32, ``attend_len - 1``: the attention collection's cache lengths.
    attend_lengths: np.ndarray
    #: ``[1]`` uint32, the largest ``attend_len`` (host-resident in the graph).
    attend_max: np.ndarray
    #: ``[B]`` uint32, ``write_index``: the store collection's cache lengths.
    write_index: np.ndarray
    #: ``[1]`` uint32, the largest ``write_index + 1`` (host-resident in the graph).
    write_max: np.ndarray
    #: ``[4]`` int64, MAX's MHA decode dispatch key ``(B, 1, num_partitions = 1, max attend_len)`` (host-resident).
    dispatch: np.ndarray

    def graph_inputs(self, kv_blocks: Buffer) -> list[Buffer | np.ndarray]:
        """The decode graph's seven page-pool inputs, in its order: ``kv_blocks``, then this step's metadata.

        The order is the one :func:`~unlimited_ocr_max.graphs.paged_kv_input_types`
        types. The two maxima and the dispatch key are host-resident in the graph, so
        they come as host :class:`~max.driver.Buffer` views over this step's
        arrays, which must outlive the execute; the other arrays are numpy, for
        the caller to stage onto the device.
        """
        return [
            kv_blocks,
            self.lookup_table,
            self.attend_lengths,
            Buffer.from_numpy(self.attend_max),
            self.write_index,
            Buffer.from_numpy(self.write_max),
            Buffer.from_numpy(self.dispatch),
        ]


@dataclass(eq=False)
class KvPagePool:
    """The pipeline's KV page pool: one fp32 ``kv_blocks`` whose last page is the null page."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    pages: int
    device: Device = field(default_factory=CPU)
    kv_blocks: Buffer = field(init=False)
    _free: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.pages < 2:
            raise ValueError(f"a pool needs at least one request page and the null page, got {self.pages} pages")
        self.kv_blocks = Buffer.zeros(
            (self.pages, 2, self.num_layers, PAGE_SIZE, self.num_kv_heads, self.head_dim),
            DType.float32,
            device=self.device,
        )
        self._free = list(range(self.pages - 1))

    @classmethod
    def sized_for(
        cls,
        *,
        max_batch_size: int,
        prefill_len: int,
        window: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: Device,
    ) -> KvPagePool:
        """``max_batch_size`` slots of ``pages_for(prefill_len + window)`` pages each, plus the null page.

        ``prefill_len`` is the longest prompt a slot holds: :data:`MAX_PROMPT_TOKENS`
        for the served pipeline (``UnlimitedOcrPipeline.kv_pool``). At the real
        decoder a page is 15.0 MiB (fp32, 2 x 12 layers x 128 rows x 10 heads x
        128 dims x 4 B), so with the 128-slot ring the served pool is 6 pages
        (90 MiB) at batch 1 and 41 pages (615 MiB) at batch 8.
        """
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        pages = max_batch_size * pages_for(prefill_len + window) + 1
        return cls(num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, pages=pages, device=device)

    @property
    def null_page(self) -> int:
        return self.pages - 1

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def allocate(self, *, prefill_len: int, window: int) -> KvCache:
        """A fresh cache with pages for ``prefill_len + window`` rows; refused when fewer pages are free."""
        need = pages_for(prefill_len + window)
        if need > len(self._free):
            # The second line, not the first. Served, this cannot fire for an admitted request:
            # the tokenizer refuses a prompt above MAX_PROMPT_TOKENS before the request reaches
            # the model worker, so no request needs more than its slot's
            # pages_for(MAX_PROMPT_TOKENS + window) pages; the pool holds --max-batch-size slots;
            # MAX's scheduler admits at most --max-batch-size requests at once (prefill and decode
            # together); and every request it drops -- finished, cancelled or preempted -- goes
            # through UnlimitedOCRModel.release, which frees its pages, before that seat is used
            # again. A preempted request would be re-prefilled with its prompt AND its generated
            # tokens, which can need more pages than its slot, and this could raise. MAX preempts a
            # decoding request only when its own paged cache cannot grow it, and the placeholder
            # cache this model declares (model.placeholder_kv_params) never runs out: a page of it is
            # 1 KiB, so MAX sizes it at its cap, --max-batch-size requests of --max-length
            # tokens each, and no request grows past --max-length. An in-process caller can still
            # ask for more than is free, and gets this.
            raise RuntimeError(
                f"the KV page pool cannot hold this request: a {prefill_len}-token prompt and a {window}-slot ring "
                f"need {need} pages of {PAGE_SIZE} rows, but only {len(self._free)} of the pool's {self.pages - 1} "
                "are free"
            )
        pages, self._free = self._free[:need], self._free[need:]
        return KvCache(pool=self, pages=pages, window=window)

    def free(self, pages: Sequence[int]) -> None:
        """Return ``pages`` to the free list; a page that is already free, or the null page, is refused."""
        bad = [page for page in pages if page in self._free or not 0 <= page < self.null_page]
        if bad:
            raise ValueError(f"pages {bad} are not allocated pages of this pool")
        self._free.extend(pages)

    def slots(self, page: int, kv: int, layer: int, rows: int) -> Buffer:
        """Slots ``[0, rows)`` of ``page``'s key (``kv = 0``) or value (``kv = 1``) rows at ``layer``: a contiguous view."""
        flat = self.kv_blocks.view(
            DType.float32, (self.pages * 2 * self.num_layers * PAGE_SIZE, self.num_kv_heads, self.head_dim)
        )
        start = ((page * 2 + kv) * self.num_layers + layer) * PAGE_SIZE
        return flat[start : start + rows, :, :]

    def step(self, caches: Sequence[KvCache], padding: int = 0) -> PagedStep:
        """The metadata of one decode step over ``caches`` and then ``padding`` padding rows.

        A padding row sits on the null page: its lookup-table row is all null,
        it stores its row at slot 0 and attends that one row. Its output is
        discarded; nothing else reads the null page's rows.
        """
        if any(not cache.pages for cache in caches):
            raise ValueError("a released cache cannot decode")
        rows = len(caches) + padding
        widest = max((len(cache.pages) for cache in caches), default=1)
        lookup = np.full((rows, padded_lut_cols(widest)), self.null_page, dtype=np.uint32)
        for b, cache in enumerate(caches):
            lookup[b, : len(cache.pages)] = cache.pages
        attend = [cache.attend_len for cache in caches] + [1] * padding
        write = [cache.write_index for cache in caches] + [0] * padding
        dispatch = np.zeros(4, dtype=np.int64)
        MHAAttnKey(batch_size=rows, max_prompt_length=1, num_partitions=1).pack_into(dispatch, max(attend))
        return PagedStep(
            lookup_table=lookup,
            attend_lengths=np.asarray([n - 1 for n in attend], dtype=np.uint32),
            attend_max=np.asarray([max(attend)], dtype=np.uint32),
            write_index=np.asarray(write, dtype=np.uint32),
            write_max=np.asarray([max(write) + 1], dtype=np.uint32),
            dispatch=dispatch,
        )


def to_host(buffer: Any) -> np.ndarray:
    """A graph output as numpy, wherever it lives (on CPU this aliases the buffer)."""
    return buffer.to(CPU()).to_numpy()
