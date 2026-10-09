"""The in-process pipeline: vision graph(s) + prefill graph + decode graph(s), and greedy decoding.

Graphs are compiled lazily, except the decode graphs a caller loads up front
with :meth:`UnlimitedOcrPipeline.warm_decode_graphs`. On an accelerator the
pipeline binds every language graph to ONE shared device copy of the language
weights (:data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT` -- bf16 and int8 alike since
KON-225). With that registry the language graphs stay resident
(:attr:`UnlimitedOcrPipeline.releases_language_graphs` is False on an
accelerator). On CPU every graph stays resident.

Every request's KV rows live in the pipeline's one page pool
(:attr:`UnlimitedOcrPipeline.kv_pool`, :mod:`~unlimited_ocr_max.kv_cache`) on
the pipeline's device, CPU included: the prefill seeds a request's pages, and
each decode step stores its new rows and attends them inside the graph, so only
the logits leave the device. There is one decode graph per row count ``B``,
``B = 1`` included, all the same code path;
:meth:`UnlimitedOcrPipeline.release_decode` drops them all. Each applies the
no-repeat-n-gram guard to its own logits, each row against its own history; the
prefill logits go through the one-op
:class:`~unlimited_ocr_max.ngram.NgramBlocker` graph instead.
:attr:`UnlimitedOcrPipeline.min_decode_rows` pads a ``decode_rows`` call with
fewer real rows up to that many with padding rows on the pool's null page.
``base`` mode is one 1024px view; a tiled
:class:`~unlimited_ocr_max.batch_processor.CropLayout`
selects ``gundam`` (a 640px tile tower plus a layout graph), which is usable
here but not served.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceKind, DeviceRef
from max.nn.kv_cache.utils import padded_lut_cols
from max.pipelines.graph_input_stager import GraphInputStager, GraphInputStaging, InputDescriptor

from .batch_processor import BASE_SIZE, LOCAL_SIZE, CropLayout, ViewGeometry
from .buffers import as_float32, numpy_to_buffer
from .decoder import COMPUTE_DTYPE, UnlimitedOcrDecoder
from .graphs import (
    DecodeGraph,
    ImageTokenLayout,
    LanguageGraph,
    LayoutGraph,
    UnlimitedOcrVisionModel,
    VisionGraph,
    build_decode_graph,
    build_language_graph,
    build_layout_graph,
    build_vision_graph,
)
from .kv_cache import MAX_PROMPT_TOKENS, KvCache, KvPagePool, to_host
from .layers.projector import image_token_count
from .model_config import UnlimitedOCRConfig
from .ngram import NgramBlocker
from .weight_adapters import language_state_dict, vision_state_dict

__all__ = ["DECODE_STAGED_INPUTS", "SHARE_LANGUAGE_WEIGHTS_DEFAULT", "PrefillResult", "UnlimitedOcrPipeline"]

#: Whether an **accelerator** pipeline binds the language weights from ONE device
#: :class:`~max.driver.Buffer` registry and declares them device-side in both
#: language graphs (KON-113, shipped by KON-158), instead of letting each graph
#: materialise its own device copy of the same ~5.5 GiB. Served-validated in the
#: research port: 12-page transcripts byte-identical in both walk orders, decode
#: reload 8.4 -> 1.54 s. It shares the *weights*; the release policy is
#: :attr:`UnlimitedOcrPipeline.releases_language_graphs`, which keeps both
#: graphs resident wherever this registry is on: since 0.3.2 no language graph
#: folds a weight copy of its own (the norms and router are stored fp32, and
#: since KON-238 the projections are read as bf16, :attr:`norm_router_dtype
#: <UnlimitedOcrPipeline.norm_router_dtype>`). Accelerator-only via
#: :attr:`UnlimitedOcrPipeline.shares_language_weights`; CPU is untouched.
#:
#: **bf16 and int8 alike (KON-225).** From KON-162 until KON-225, int8 served
#: **unshared**: the served int8 identity gate at the commit that shipped this
#: default read 0/12 in both request orders against the pinned KON-149
#: transcripts, while every in-process value-level A/B of the mechanism stayed
#: bitwise green -- so the divergence was unlocalised above the pipeline and
#: int8 was excluded by :attr:`UnlimitedOcrPipeline.shares_language_weights`
#: (``and not int8_experts``). KON-224 localised it: a load-time fp32 fold of
#: the prefill dequant expert index, a registry-adjacent effect no in-process
#: value comparison exercises, removed by commit ``82ab60f`` and pinned by
#: ``test_int8_prefill_dequant_expert_index_is_not_load_foldable``. With that
#: gate lifted int8 serves 12/12 byte-identical on both MAX 26.6.0 and the
#: 26.7 nightly, so the variant exclusion is gone and int8 shares the registry
#: like bf16 -- and, via :attr:`UnlimitedOcrPipeline.releases_language_graphs`,
#: holds both language graphs instead of reloading them per request.
SHARE_LANGUAGE_WEIGHTS_DEFAULT = True

#: The token a padding row feeds (:attr:`UnlimitedOcrPipeline.min_decode_rows`): any valid id
#: does, its outputs are discarded; fixed so every padded step is the same computation.
PADDING_TOKEN_ID = 0

#: The decode graphs' inputs that a step copies to the device, in input order: every input except
#: ``kv_blocks`` (the pool, already there) and the three host-resident ones
#: (:func:`~unlimited_ocr_max.graphs.paged_kv_input_types`). :meth:`UnlimitedOcrPipeline._step`
#: sends all of them in one copy (:meth:`UnlimitedOcrPipeline._decode_input_stager`).
DECODE_STAGED_INPUTS = ("tokens", "positions", "lookup_table", "attend_lengths", "write_index", "history", "ngram_size")

#: A padding row's position. It sits on the pool's null page (:meth:`KvPagePool.step
#: <unlimited_ocr_max.kv_cache.KvPagePool.step>`), storing its row at slot 0 and attending
#: that one row, so every padded step computes the same thing for it.
PADDING_POSITION = 0


def _stage(staging: GraphInputStaging, name: str, array: np.ndarray) -> Buffer:
    """Write ``array`` into ``name``'s host staging and return its device buffer; the copy goes when ``staging`` closes."""
    host, (device,) = staging.get(name, array.shape)
    view = host.to_numpy()
    if view.dtype != array.dtype:
        raise TypeError(f"decode input {name!r} is {array.dtype}, staged as {view.dtype}")
    view[...] = array
    return device


@dataclass(frozen=True)
class PrefillResult:
    logits: np.ndarray
    cache: KvCache


class UnlimitedOcrPipeline:
    def __init__(
        self,
        config: UnlimitedOCRConfig,
        *,
        checkpoint: dict[str, Any] | None = None,
        vision_state_dict: dict[str, Any] | None = None,
        language_state_dict: dict[str, Any] | None = None,
        seq_len: int,
        max_new_tokens: int = 1024,
        image_size: int = BASE_SIZE,
        crop_layout: CropLayout | None = None,
        device: DeviceRef | None = None,
        driver_device: Device | None = None,
        session: InferenceSession | None = None,
        ngram_size: int = 0,
        max_batch_size: int = 1,
    ) -> None:
        """Pass either the raw ``checkpoint`` or both pre-split state dicts.

        ``seq_len`` is the prompt length the pipeline is built around (with
        ``max_new_tokens`` it sizes the decode RoPE table). ``max_batch_size``
        is how many request slots the KV page pool (:attr:`kv_pool`) holds,
        each for a prompt of up to ``max(seq_len, MAX_PROMPT_TOKENS)`` tokens
        plus the ring.
        """
        if checkpoint is None and (vision_state_dict is None or language_state_dict is None):
            raise ValueError("pass either checkpoint= or both vision_state_dict= and language_state_dict=")
        self.config = config
        self.seq_len = seq_len
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.crop_layout = crop_layout
        self.device = device or DeviceRef.CPU()
        self._driver_device = driver_device or CPU()
        self.session = session or InferenceSession(devices=[self._driver_device])
        self.window = config.decoder.sliding_window_size
        self.ngram_size = ngram_size
        self._ngram: NgramBlocker | None = None
        self._checkpoint = checkpoint
        self._vision_state_dict = vision_state_dict
        self._language_state_dict = language_state_dict
        #: Bind the language weights as one shared device registry and declare
        #: them device-side in every language graph (KON-113's spelling:
        #: ``add_weight(force_initial_weight_on_host=False)``, deliberately not
        #: ``Weight._has_alias``, which does not compile on Metal in tractable
        #: time). An attribute rather than a property so a probe can set it
        #: either way against the shipped default;
        #: :attr:`shares_language_weights` is the conjunction with
        #: :attr:`on_accelerator` (KON-225 dropped the ``not int8_experts``
        #: clause KON-162 added) and is the only thing the graph loaders read.
        self._share_language_weights = SHARE_LANGUAGE_WEIGHTS_DEFAULT
        #: The shared device registry, built once by
        #: :meth:`_resolved_language_weights` and never dropped: both language
        #: graphs bind it for the life of the process (the host arrays are gone
        #: once it exists, so it could not be rebuilt anyway).
        self._language_device_weights: dict[str, Buffer] | None = None
        self._vision: tuple[VisionGraph, Model] | None = None
        self._local_vision: tuple[VisionGraph, Model] | None = None
        self._layout: tuple[LayoutGraph, Model] | None = None
        self._prefill: dict[int, tuple[LanguageGraph, Model]] = {}
        #: The decode graphs, keyed by row count ``B >= 1``.
        self._decode: dict[int, tuple[DecodeGraph, Model]] = {}
        #: The fewest rows :meth:`decode_rows` executes: a call with fewer real
        #: rows is padded up to this many with padding rows on the pool's null
        #: page, whose outputs are discarded. 1 (the default) runs a lone
        #: request on the batch-1 graph; 2 runs it on the 2-row graph, which is
        #: not bitwise the batch-1 graph but whose row bits do not depend on
        #: what rides alongside -- so a request's output does not depend on
        #: load. A plain attribute: the served model sets it at startup.
        self.min_decode_rows = 1
        #: Built on first use by :attr:`kv_pool` and kept for the pipeline's life.
        self._kv_pool: KvPagePool | None = None
        #: Built on first use by :meth:`_decode_input_stager`, for this many rows.
        self._decode_stager: GraphInputStager | None = None
        self._decode_stager_rows = 0
        # Compile the prefill's Mojo guard now, so a missing Metal Toolchain fails at
        # load rather than on the first request (the decode graphs stage the same op).
        if self.ngram_blocker is not None:
            self.ngram_blocker.model

    @property
    def on_accelerator(self) -> bool:
        return self.device.device_type != DeviceKind.CPU

    @property
    def shares_language_weights(self) -> bool:
        """Whether every language graph binds ONE device registry of the language weights.

        :attr:`_share_language_weights` (:data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT`,
        **on**) gated by :attr:`on_accelerator` only -- since KON-225 there is no
        weight-variant exclusion: bf16 and int8 share alike. **False on CPU**,
        whatever the attribute says: there is no device budget to fit into, a
        host array *is* the graph's memory so there is no duplicate copy to
        remove, and the numerically gated path stays byte-identical by
        construction. int8 was excluded here from KON-162 until KON-225: the
        served int8 identity gate falsified the registry commit at 0/12 in both
        request orders, and the divergence stayed unlocalised above the
        pipeline until KON-224 traced it to a load-time fp32 fold of the
        prefill dequant expert index (commit ``82ab60f``, pinned by
        ``test_int8_prefill_dequant_expert_index_is_not_load_foldable``). The
        full record is on :data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT`.
        """
        return self._share_language_weights and self.on_accelerator

    @property
    def norm_router_dtype(self) -> DType | None:
        """Storage dtype of the RMSNorm gammas and the MoE router: fp32 under :attr:`shares_language_weights`, else the checkpoint's.

        MAX's own fp32 ops read those weights whole, and a bf16 weight read by
        an fp32 op is compile-folded into an fp32 copy on the device at every
        ``session.load`` (EXPERIMENTS.md, OQ-113-A), so the shared registry
        holds their exact fp32 upcast once (0.004 GiB at the real shapes).
        Every projection -- attention, the dense FFN, the shared experts,
        ``lm_head`` -- stays bf16 in the registry since KON-238: the decoder
        reads those without a weight-only upcast (``dense_bf16_qmv`` at the
        decode step, a run-time upcast at prefill), 0.645 GiB less registry
        than their fp32 upcast. Off the registry (CPU) ``None`` keeps every
        declaration in the checkpoint's dtype.
        """
        return COMPUTE_DTYPE if self.shares_language_weights else None

    @property
    def releases_language_graphs(self) -> bool:
        """Whether a caller must hold **one** language graph at a time.

        On an accelerator **without** the shared weight registry -- reachable
        today only by forcing :attr:`_share_language_weights` off for a probe,
        since bf16 and int8 alike share it by default since KON-225 -- each
        graph places its own weight copy, and holding both was measured over
        the Metal budget (KON-113: 18.629 of 17.760 GiB). **With** the registry
        (the default on an accelerator, either weight variant) the graphs hold
        no weights of their own since 0.3.2: the prefill graph costs 0.002 GiB
        and the decode graph 0.000 GiB beyond the registry, 7.67 of 17.760 GiB
        for everything on an M4 at 0.3.2 (the registry is smaller since
        KON-238), so releasing them would only re-pay a graph reload on every
        request -- for int8 since KON-225, that reload is now
        gone too. On CPU every graph stays resident and a reload would be
        pure loss. Named so the served path
        (``UnlimitedOCRModel._prefill``) carries the reason along instead of
        reading the device raw.

        A multi-row decode graph (:meth:`decode_graph` at ``B >= 2``) is one more
        language graph under the same rule: without the registry it places its
        own weight copy, so it counts against the one-at-a-time budget
        alongside the batch-1 decode graph -- reachable only by also forcing
        :attr:`_share_language_weights` off, the same probe path, since
        :func:`~unlimited_ocr_max.model.check_max_batch_size` no longer refuses
        int8 above batch 1 (KON-227). With the registry each batched graph
        costs 0.002 GiB beyond it (EXPERIMENTS.md, KON-213).
        """
        return self.on_accelerator and not self.shares_language_weights

    @property
    def max_total_len(self) -> int:
        """Longest position the decode RoPE table covers; the ring bounds the cache, not the positions."""
        return self.seq_len + self.max_new_tokens

    @property
    def is_gundam(self) -> bool:
        return self.crop_layout is not None and self.crop_layout.tiled

    @property
    def tile_grid(self) -> tuple[int, int] | None:
        return ViewGeometry(LOCAL_SIZE).token_grid if self.is_gundam else None

    @property
    def n_local_views(self) -> int:
        return self.crop_layout.n_tiles if self.is_gundam else 0

    @property
    def n_image_tokens(self) -> int:
        return image_token_count(
            global_grid=ViewGeometry(self.image_size).token_grid,
            local_grid=self.tile_grid,
            crop_grid=self.crop_layout.crop_grid if self.is_gundam else None,
        )

    # -- weights ------------------------------------------------------------ #
    def _resolved_vision_state_dict(self, image_size: int | None = None) -> dict[str, Any]:
        if image_size is not None and image_size != self.image_size:
            if self._checkpoint is None:
                raise ValueError("gundam needs the raw checkpoint to resample the vision weights for the tiles")
            return vision_state_dict(self._checkpoint, image_size=image_size)
        if self._vision_state_dict is None:
            assert self._checkpoint is not None
            self._vision_state_dict = vision_state_dict(self._checkpoint, image_size=self.image_size)
        return self._vision_state_dict

    def _resolved_language_state_dict(self) -> dict[str, Any]:
        if self._language_state_dict is None:
            assert self._checkpoint is not None
            self._language_state_dict = language_state_dict(self._checkpoint, self.config)
        return self._language_state_dict

    def _resolved_language_weights(self) -> dict[str, Any]:
        """The registry values the language graphs bind to.

        On CPU (sharing off): the host tensors, unchanged. Under
        :attr:`shares_language_weights`: **one device-resident**
        :class:`~max.driver.Buffer` per weight, built once here and handed to
        *every* language graph -- the graphs declare the weights device-side
        (:func:`~unlimited_ocr_max.graphs.declare_device_resident_weights`), so
        the registry values must already be on the device, and neither
        ``session.load`` materialises its own ~5.5 GiB copy.

        The host entries are already ``Buffer``s -- MAX's mmap of the checkpoint
        for the dense weights, one host allocation per expert stack -- so every
        dtype needs the same single step, ``.to(device)``: bf16 dense weights,
        int8 expert stacks and fp32 scales alike. The one exception is the set
        :attr:`norm_router_dtype` declares fp32 (:meth:`_fp32_resident_names`):
        those bf16 entries are widened to their exact fp32 upcast first, so the
        graphs bind fp32 and fold no copy of their own.

        **The conversion is incremental, and that is not cosmetic**: building
        the whole device dict beside the host one holds both ~5.5 GiB copies at
        once (measured 17.83 GiB host peak in the research port, two clamped
        arms), so each host entry is popped as its buffer is made. This method
        therefore **takes ownership** of the language state dict on an
        accelerator: after it runs, that mapping is empty and gone.

        *When* the host entries go is the whole point, not merely *that* they
        go, so the loop below must stay a loop that mutates ``host`` in place.
        A rewrite that built the full device dict first and cleared the host one
        afterwards -- a dict comprehension, say -- would leave the same empty
        mapping behind and hit the same 17.83 GiB peak it exists to avoid. The
        host peak this shape buys is one device copy plus the entries **not yet
        converted**: entry ``k`` is dropped while ``n-1-k`` remain, which is
        also why a ``Buffer``'s source array dies with it rather than being kept
        alive by the device copy. (An accelerator-only test used to record that
        liveness entry by entry; it was removed as untested-in-CI mechanics, so
        this paragraph is the record.)

        **What the ``synchronize`` actually covers -- stated precisely, because
        an overclaim here reads as a safety argument.** ``.to(device)`` is an
        async copy and a host-backed source must outlive the *copy*, not the
        call (KON-125). This drain runs **after** the loop, so the only entry
        whose source it strictly orders is the **last** one; every earlier
        source was already dropped inside the loop. It is kept as
        belt-and-braces -- one drain per process is not a term worth saving --
        and not as the thing that makes the loop safe. What says the bytes are
        right is value-level evidence: KON-161 round 2 byte-compared this
        shipped conversion at production expert-stack shapes and decoded real
        pages to EOS byte-identically with the registry on and off. Strict
        per-entry ordering would need a drain inside the loop; nothing has
        measured a need for one.
        """
        if not self.shares_language_weights:
            return self._resolved_language_state_dict()
        if self._language_device_weights is None:
            host = self._resolved_language_state_dict()
            widen = self._fp32_resident_names()
            built: dict[str, Buffer] = {}
            # `list(...)` because the loop mutates `host`.
            for name in list(host):
                value = host[name]
                if name in widen and value.dtype == DType.bfloat16:
                    # The exact upcast (lossless: bf16 is the top half of the
                    # fp32 word). Its numpy source is a temporary, so drain the
                    # async copy before the next iteration can free it (KON-125).
                    built[name] = numpy_to_buffer(as_float32(value), DType.float32).to(self._driver_device)
                    self._driver_device.synchronize()
                else:
                    built[name] = value.to(self._driver_device)
                # Drop the host bytes now, one weight at a time, so the two
                # copies never both exist in full.
                del host[name], value
            self._driver_device.synchronize()
            self._language_device_weights = built
            self._language_state_dict = None
            gc.collect()
        return self._language_device_weights

    def _fp32_resident_names(self) -> frozenset[str]:
        """The weights :attr:`norm_router_dtype` declares fp32, read off a weightless decoder (declarations only)."""
        if self.norm_router_dtype is None:
            return frozenset()
        declared = UnlimitedOcrDecoder(
            self.config.decoder, dtype=self.config.dtype, device=self.device, norm_router_dtype=self.norm_router_dtype
        ).raw_state_dict()
        return frozenset(name for name, weight in declared.items() if weight.dtype == DType.float32)

    def _decoder(self) -> UnlimitedOcrDecoder:
        """A fresh, weight-loaded decoder: a ``Weight`` binds to one graph, the arrays behind it are shared.

        It loads :meth:`_resolved_language_weights`, so ``decoder.state_dict()``
        -- what every language ``session.load`` passes as ``weights_registry``
        -- *is* that registry: host tensors on CPU, the shared device buffers
        under :attr:`shares_language_weights`.
        """
        decoder = UnlimitedOcrDecoder(
            self.config.decoder, dtype=self.config.dtype, device=self.device, norm_router_dtype=self.norm_router_dtype
        )
        decoder.load_state_dict(self._resolved_language_weights())
        return decoder

    def _load_vision(self, image_size: int, n_views: int) -> tuple[VisionGraph, Model]:
        model = UnlimitedOcrVisionModel(image_size=image_size, dtype=COMPUTE_DTYPE, device=self.device)
        model.load_state_dict(self._resolved_vision_state_dict(image_size))
        staged = build_vision_graph(model, n_views=n_views)
        return staged, self.session.load(staged.graph, weights_registry=model.state_dict())

    # -- graphs ------------------------------------------------------------- #
    @property
    def vision(self) -> tuple[VisionGraph, Model]:
        if self._vision is None:
            self._vision = self._load_vision(self.image_size, 1)
        return self._vision

    @property
    def local_vision(self) -> tuple[VisionGraph, Model]:
        """The tile tower, batched over this page's tiles (gundam only)."""
        if not self.is_gundam:
            raise ValueError("local_vision needs a tiled crop_layout")
        if self._local_vision is None:
            self._local_vision = self._load_vision(LOCAL_SIZE, self.n_local_views)
        return self._local_vision

    @property
    def layout(self) -> tuple[LayoutGraph, Model]:
        if not self.is_gundam:
            raise ValueError("the layout graph is only needed for gundam")
        if self._layout is None:
            assert self.crop_layout is not None
            model = ImageTokenLayout(
                global_grid=ViewGeometry(self.image_size).token_grid,
                local_grid=self.tile_grid,
                crop_grid=self.crop_layout.crop_grid,
                dtype=COMPUTE_DTYPE,
                device=self.device,
            )
            weights = self._resolved_vision_state_dict()
            model.load_state_dict({key: weights[key] for key in ("image_newline", "view_seperator")})
            staged = build_layout_graph(model)
            if staged.n_image_tokens != self.n_image_tokens:
                raise AssertionError(f"layout emits {staged.n_image_tokens} tokens, prompt geometry says {self.n_image_tokens}")
            self._layout = (staged, self.session.load(staged.graph, weights_registry=model.state_dict()))
        return self._layout

    def prefill_graph_for(self, seq_len: int) -> tuple[LanguageGraph, Model]:
        """The prefill graph compiled for exactly ``seq_len`` tokens (static shape), cached per length."""
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if seq_len > self.max_total_len:
            raise ValueError(f"prompt is {seq_len} tokens but the decode RoPE table covers {self.max_total_len}")
        cached = self._prefill.get(seq_len)
        if cached is None:
            decoder = self._decoder()
            staged = build_language_graph(
                self.config,
                decoder,
                seq_len=seq_len,
                n_image_tokens=self.n_image_tokens,
                device=self.device,
                device_resident_weights=self.shares_language_weights,
            )
            cached = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
            self._prefill[seq_len] = cached
        return cached

    def decode_graph(self, batch: int = 1) -> tuple[DecodeGraph, Model]:
        """The decode step for ``batch`` independent requests, compiled on first use (or up front by
        :meth:`warm_decode_graphs`) and cached per ``batch``.

        Every ``batch`` is built and loaded the same way -- a fresh decoder, the
        same ``max_seq_len``, the same weight declaration and registry -- and
        is its own graph.
        """
        cached = self._decode.get(batch)
        if cached is None:
            decoder = self._decoder()
            staged = build_decode_graph(
                self.config,
                decoder,
                batch=batch,
                max_seq_len=self.max_total_len,
                device=self.device,
                device_resident_weights=self.shares_language_weights,
            )
            cached = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
            self._decode[batch] = cached
        return cached

    @property
    def kv_pool(self) -> KvPagePool:
        """The one KV page pool every request's rows live in, on this pipeline's device; built on first use.

        :attr:`max_batch_size` slots, each for a prompt of up to
        :data:`~unlimited_ocr_max.kv_cache.MAX_PROMPT_TOKENS` tokens plus the
        :attr:`window`-slot ring, plus the null page
        (:meth:`KvPagePool.sized_for <unlimited_ocr_max.kv_cache.KvPagePool.sized_for>`,
        which has the sizes), whatever prompt a request carries. A pipeline
        built around a longer prompt (:attr:`seq_len`; ``gundam``, never
        served) sizes its slots for that one instead. The served model builds
        the pool at startup; a request that needs more pages than are free is
        refused at prefill, before the prefill graph is compiled or run.
        """
        if self._kv_pool is None:
            dec = self.config.decoder
            self._kv_pool = KvPagePool.sized_for(
                max_batch_size=self.max_batch_size,
                prefill_len=max(self.seq_len, MAX_PROMPT_TOKENS),
                window=self.window,
                num_layers=dec.num_hidden_layers,
                num_kv_heads=dec.num_key_value_heads,
                head_dim=dec.head_dim,
                device=self._driver_device,
            )
        return self._kv_pool

    def _decode_input_stager(self, rows: int) -> GraphInputStager:
        """MAX's per-step input staging for the decode graphs' :data:`DECODE_STAGED_INPUTS`.

        Each step writes those small arrays into one host staging buffer and
        sends them with ONE copy into device buffers allocated once and kept
        (``max.pipelines.graph_input_stager``, what MAX's own KV manager and
        batch processors use), instead of one allocation and one copy each:
        1.1 ms of a ~18 ms step on the M4, with the GPU idle, before this.
        Sized for ``max(rows, max_batch_size, min_decode_rows)`` rows on
        first use, and built again, larger, for a call with more rows.
        """
        if self._decode_stager is None or rows > self._decode_stager_rows:
            rows = max(rows, self.max_batch_size, self.min_decode_rows)
            shapes = {
                "tokens": (DType.int64, (rows,)),
                "positions": (DType.int32, (rows,)),
                "lookup_table": (DType.uint32, (rows, padded_lut_cols(self.kv_pool.pages))),
                "attend_lengths": (DType.uint32, (rows,)),
                "write_index": (DType.uint32, (rows,)),
                "history": (DType.int32, (rows, self.window)),
                "ngram_size": (DType.int32, (1,)),
            }
            assert tuple(shapes) == DECODE_STAGED_INPUTS
            self._decode_stager = GraphInputStager(
                InputDescriptor(name, dtype, shape, [self._driver_device]) for name, (dtype, shape) in shapes.items()
            )
            self._decode_stager_rows = rows
        return self._decode_stager

    def warm_decode_graphs(self, max_batch: int) -> None:
        """Load :meth:`decode_graph` for ``B = 2 .. max_batch``, in that order; a no-op below 2.

        The eager startup a served ``--max-batch-size N > 1`` pays once so no
        request waits on a compile (24-33 s cold per ``B`` on the real
        checkpoint before KON-237's paged attention, which compiles slower and
        is not re-measured there yet; ~0.8 s once MAX's cache holds it). Each graph binds the
        language weights, so on an accelerator the first one also builds the
        shared registry (:meth:`_resolved_language_weights`) if nothing has
        yet. The graphs stay loaded until :meth:`release_decode`.
        """
        for batch in range(2, max_batch + 1):
            self.decode_graph(batch)

    # -- lifetime ----------------------------------------------------------- #
    def drop_vision_weights(self) -> None:
        """Drop the Python-side fp32 vision arrays once the compiled tower holds them (serving path)."""
        if self._vision is None or self._checkpoint is not None:
            return
        self._vision_state_dict = None
        gc.collect()

    def release_vision(self) -> None:
        """Drop the vision graphs; they are rebuilt from the checkpoint if asked for again."""
        self._vision = None
        self._local_vision = None
        self._layout = None
        if self._checkpoint is not None:
            self._vision_state_dict = None
        gc.collect()

    def release_decode(self) -> None:
        """Drop every decode graph, at every ``B``. Idempotent, and safe before anything was built.

        The graphs (each a ``Model``) are what go; the KV page pool, which the
        live requests' rows are in, stays; and the shared language weight
        registry (:attr:`_language_device_weights`) deliberately does **not**.
        It is one device copy bound by *every* language graph across every
        release/reload cycle -- keeping it is the whole per-request saving
        (decode reload 8.4 -> ~1.5 s in the research port) -- and it could not
        be rebuilt anyway: :meth:`_resolved_language_weights` took the host
        arrays. Were a registry ever dropped here, the graphs would still have
        to go **first**: a ``Model`` reads those buffers, and dropping them
        under it would be a use-after-free of the weights the graph binds.
        """
        self._decode.clear()
        gc.collect()

    def release_prefill(self) -> None:
        """Drop the prefill graphs. Idempotent, and safe before anything was built.

        Same shape as :meth:`release_decode`: graphs only, never the shared
        weight registry.
        """
        self._prefill.clear()
        gc.collect()

    def retain_only_prefill(self, seq_len: int) -> None:
        """Drop every cached prefill graph except ``seq_len``'s.

        The bound a pipeline that holds its language graphs needs: prefill
        graphs are compiled per static prompt length and cached, and without a
        per-request release a server fed prompts of many lengths would keep one
        compiled graph for each.
        """
        stale = [length for length in self._prefill if length != seq_len]
        for length in stale:
            del self._prefill[length]
        if stale:
            gc.collect()

    # -- execution ---------------------------------------------------------- #
    def _execute(self, model: Model, names: Sequence[str], *arrays: np.ndarray | Buffer) -> dict[str, np.ndarray]:
        """Run one graph and bring every output back as numpy; a ``Buffer`` input is passed as is, arrays are staged onto the device."""
        buffers = [
            array if isinstance(array, Buffer) else Buffer.from_numpy(np.ascontiguousarray(array)).to(self._driver_device)
            for array in arrays
        ]
        outputs = model.execute(*buffers)
        return {name: to_host(out) for name, out in zip(names, outputs, strict=True)}

    def run_vision(self, pixels: np.ndarray, local_pixels: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Encode a page; the result carries ``image_embeds`` (tile stages come back suffixed ``_local``)."""
        staged, model = self.vision
        stages = self._execute(model, staged.output_names, pixels)
        if not self.is_gundam:
            if local_pixels is not None:
                raise ValueError("local_pixels was supplied but this pipeline has no tiled crop_layout")
            return stages
        if local_pixels is None or int(np.asarray(local_pixels).shape[0]) != self.n_local_views:
            raise ValueError(f"gundam needs exactly {self.n_local_views} local views")
        local_staged, local_model = self.local_vision
        for key, value in self._execute(local_model, local_staged.output_names, local_pixels).items():
            stages[f"{key}_local"] = value
        layout_staged, layout_model = self.layout
        stages.update(
            self._execute(layout_model, layout_staged.output_names, stages["projected"], stages["projected_local"])
        )
        return stages

    def run_prefill(self, token_ids: np.ndarray, image_embeds: np.ndarray) -> PrefillResult:
        """Prefill the prompt and seed a fresh KV cache from the same pass.

        The cache's pages come from :attr:`kv_pool` first, so a request the
        pool cannot hold is refused before this method compiles or runs the
        prefill graph (the vision tower, which the caller runs first, has
        already run by then). If anything after the allocation fails, the
        pages go back.
        """
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        prompt_len = int(token_ids.shape[0])
        hidden = self.config.decoder.hidden_size
        rows = int(np.asarray(image_embeds).size // hidden)
        if rows != self.n_image_tokens:
            raise ValueError(f"{rows} image embedding rows for {self.n_image_tokens} placeholders")
        cache = self.kv_pool.allocate(prefill_len=prompt_len, window=self.window)
        try:
            staged, model = self.prefill_graph_for(prompt_len)
            outputs = self._execute(model, staged.output_names, token_ids, image_embeds.reshape(-1, hidden))
            layers = self.config.decoder.num_hidden_layers
            cache.seed([outputs[f"key_{i}"] for i in range(layers)], [outputs[f"value_{i}"] for i in range(layers)])
        except BaseException:
            cache.release()
            raise
        return PrefillResult(logits=outputs["logits"].reshape(-1), cache=cache)

    def decode_step(self, cache: KvCache, token_id: int, history: Sequence[int]) -> np.ndarray:
        """Feed one token at ``cache.position`` through the batch-1 graph; its ``[vocab]`` logits, already n-gram-guarded.

        ``history`` is the request's token sequence so far -- prompt included,
        ``token_id`` last -- which is what :meth:`NgramBlocker.apply
        <unlimited_ocr_max.ngram.NgramBlocker.apply>` was handed for this
        step's logits before the guard moved into the graph. The graph reads
        its last :attr:`window` ids (:meth:`_guard_history`); with the guard
        off it reads nothing of it. Never padded: :meth:`decode_rows` is.
        """
        return self._step([cache], [token_id], [history], padding=0)[0]

    def decode_rows(
        self, caches: Sequence[KvCache], token_ids: Sequence[int], histories: Sequence[Sequence[int]]
    ) -> np.ndarray:
        """Feed ``token_ids[b]`` to request ``b`` at ``caches[b].position``; ``[B, vocab]`` fp32 logits, row ``b`` for it.

        ``histories[b]`` is request ``b``'s token sequence so far, prompt
        included and ``token_ids[b]`` last (:meth:`decode_step`'s
        ``history``); the graph guards row ``b`` against its last
        :attr:`window` ids and nothing else, so the logits come back already
        n-gram-guarded.

        Fewer than :attr:`min_decode_rows` real rows are padded up to it with
        padding rows on the pool's null page, after the real ones, each fed
        :data:`PADDING_TOKEN_ID` at :data:`PADDING_POSITION` and an all-zero
        history; only the real rows' logits come back. So with the default
        ``min_decode_rows == 1`` nothing is padded, and with 2 a lone request
        runs the 2-row graph, never the batch-1 one. No cache may appear twice:
        one execute would store two rows into it.
        """
        if len(caches) != len(token_ids):
            raise ValueError(f"{len(caches)} caches for {len(token_ids)} tokens")
        if len(histories) != len(token_ids):
            raise ValueError(f"{len(histories)} histories for {len(token_ids)} tokens")
        if not caches:
            raise ValueError("decode_rows needs at least one row")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("the same cache appears more than once in one decode_rows call")
        return self._step(caches, token_ids, histories, padding=max(self.min_decode_rows - len(caches), 0))

    def _step(
        self,
        caches: Sequence[KvCache],
        token_ids: Sequence[int],
        histories: Sequence[Sequence[int]],
        *,
        padding: int,
    ) -> np.ndarray:
        """One execute of :meth:`decode_graph` over the real rows and ``padding`` padding rows; the real rows' logits.

        The inputs are the graph's (:func:`~unlimited_ocr_max.graphs.build_decode_graph`):
        tokens, positions, the page pool's
        (:meth:`PagedStep.graph_inputs <unlimited_ocr_max.kv_cache.PagedStep.graph_inputs>`
        of this step's :meth:`~unlimited_ocr_max.kv_cache.KvPagePool.step`,
        which this frame holds until the logits are back), then the guard's
        two. Its numpy inputs, :data:`DECODE_STAGED_INPUTS` in that order, go
        to the device in one copy (:meth:`_decode_input_stager`). The graph
        stores every row's new k/v itself, so afterwards each cache only
        advances its ring state (:meth:`~unlimited_ocr_max.kv_cache.KvCache.append`).
        """
        real = len(caches)
        staged, model = self.decode_graph(real + padding)
        pool = self.kv_pool
        paged = pool.step(caches, padding)
        guard_rows = [self._guard_history(history) for history in histories]
        guard_rows += [np.zeros(self.window, dtype=np.int32)] * padding  # padding: never read back
        inputs = [
            np.asarray([*token_ids, *([PADDING_TOKEN_ID] * padding)], dtype=np.int64),
            np.asarray([*(cache.position for cache in caches), *([PADDING_POSITION] * padding)], dtype=np.int32),
            *paged.graph_inputs(pool.kv_blocks),
            np.stack(guard_rows),
            self._decode_ngram_arg(),
        ]
        # `staging` holds the host side of the copy; it stays alive in this frame
        # until `_execute` has the logits back, so past the copy.
        with self._decode_input_stager(real + padding).stage() as staging:
            names = iter(DECODE_STAGED_INPUTS)
            buffers = [value if isinstance(value, Buffer) else _stage(staging, next(names), value) for value in inputs]
            if next(names, None) is not None:
                raise AssertionError("a decode input named in DECODE_STAGED_INPUTS was not staged")
        outputs = self._execute(model, staged.output_names, *buffers)
        for cache in caches:
            cache.append()
        logits = outputs["logits"]
        return logits if padding == 0 else logits[:real]

    def generate(self, *, pixels: np.ndarray, token_ids: np.ndarray, local_pixels: np.ndarray | None = None) -> list[int]:
        """Greedy decode to EOS (included) or ``max_new_tokens``; the n-gram guard, if on, is applied to every step.

        The prefill logits go through :attr:`ngram_blocker`; every decode
        step's come back from :meth:`decode_step` already guarded, against the
        same ``sequence`` the blocker used to be handed for them. The request's
        pages go back to :attr:`kv_pool` when it ends, however it ends.
        """
        stages = self.run_vision(pixels, local_pixels)
        if self.releases_language_graphs:
            self.release_vision()
        prefill = self.run_prefill(token_ids, stages["image_embeds"])
        try:
            if self.releases_language_graphs:
                self.release_prefill()
            eos = self.config.decoder.eos_token_id
            blocker = self.ngram_blocker
            sequence = [int(i) for i in np.asarray(token_ids).reshape(-1)]
            generated: list[int] = []
            logits = prefill.logits if blocker is None else blocker.apply(prefill.logits, sequence)
            for _ in range(self.max_new_tokens):
                token = int(logits.argmax())
                generated.append(token)
                sequence.append(token)
                if token == eos:
                    break
                logits = self.decode_step(prefill.cache, token, sequence)
        finally:
            prefill.cache.release()
        return generated

    @property
    def ngram_guard_on(self) -> bool:
        """Whether the no-repeat-n-gram guard runs: ``ngram_size > 0``; its window is the R-SWA ring (:attr:`window`)."""
        return self.ngram_size > 0

    def _decode_ngram_arg(self) -> np.ndarray:
        """The decode graphs' n-gram-size input: :attr:`ngram_size`, or 0 -- the kernel's pass-through -- with the guard off."""
        return np.asarray([self.ngram_size if self.ngram_guard_on else 0], dtype=np.int32)

    def _guard_history(self, sequence: Sequence[int]) -> np.ndarray:
        """One decode row's guard history: the last :attr:`window` ids of ``sequence``, int32 ``[window]``.

        ``sequence`` is the row's token sequence so far, prompt included and
        the token this step feeds last, so the graph sees exactly the ids
        :meth:`NgramBlocker.apply <unlimited_ocr_max.ngram.NgramBlocker.apply>`
        windows the same sequence to (``ids[-window:]``). A decode step always
        has at least :attr:`window` ids behind it -- the prompt alone carries
        273 image tokens against a 128-id window -- so a shorter one is
        refused.

        With the guard off the graph is fed ``n = 0``, which returns the
        logits without reading the history, so this returns zeros of the
        right shape and does not look at ``sequence`` at all.
        """
        if not self.ngram_guard_on:
            return np.zeros(self.window, dtype=np.int32)
        if len(sequence) < self.window:
            raise ValueError(
                f"a decode row's history has {len(sequence)} tokens, fewer than the {self.window}-token n-gram "
                "window; a decode step follows a prompt at least that long, so this is not the request's sequence"
            )
        return np.asarray(sequence[-self.window :], dtype=np.int32)

    @property
    def ngram_blocker(self) -> NgramBlocker | None:
        """The prefill logits' guard, a one-op graph; ``None`` with the guard off. Decode guards in-graph."""
        if not self.ngram_guard_on:
            return None
        if self._ngram is None:
            self._ngram = NgramBlocker(
                ngram_size=self.ngram_size,
                window=self.window,
                vocab_size=self.config.decoder.vocab_size,
                device=self.device,
                driver_device=self._driver_device,
                session=self.session,
            )
        return self._ngram
