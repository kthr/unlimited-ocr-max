"""The in-process pipeline: vision graph(s) + prefill graph + decode graph(s), and greedy decoding.

Graphs are compiled lazily. On an accelerator the pipeline binds every language
graph to ONE shared device copy of the language weights
(:data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT` -- **bf16 only**; the int8 variant
serves unshared, KON-162) and keeps the KV cache device-resident. With that
registry the language graphs stay resident; without it (int8) the pipeline
holds one at a time -- prefill is released once its host outputs are back, the
decode graphs before the next prefill
(:attr:`UnlimitedOcrPipeline.releases_language_graphs`). On CPU every graph
stays resident and the cache is host numpy. The decode graphs are the batch-1
step and, for :meth:`UnlimitedOcrPipeline.decode_rows`, one batched step per
row count ``B >= 2``; :meth:`UnlimitedOcrPipeline.release_decode` drops them all.
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
    build_batched_decode_graph,
    build_decode_graph,
    build_language_graph,
    build_layout_graph,
    build_vision_graph,
)
from .kv_cache import KvCache, allocate_kv_cache, to_host
from .layers.projector import image_token_count
from .model_config import UnlimitedOCRConfig
from .ngram import NgramBlocker
from .weight_adapters import language_state_dict, vision_state_dict

__all__ = ["SHARE_LANGUAGE_WEIGHTS_DEFAULT", "PrefillResult", "UnlimitedOcrPipeline"]

#: Whether an **accelerator** pipeline binds the language weights from ONE device
#: :class:`~max.driver.Buffer` registry and declares them device-side in both
#: language graphs (KON-113, shipped by KON-158), instead of letting each graph
#: materialise its own device copy of the same ~5.5 GiB. Served-validated in the
#: research port: 12-page transcripts byte-identical in both walk orders, decode
#: reload 8.4 -> 1.54 s. It shares the *weights*; the release policy is
#: :attr:`UnlimitedOcrPipeline.releases_language_graphs`, which since 0.3.2's
#: fp32-resident non-expert weights keeps both graphs resident wherever this
#: registry is on (they no longer hold weights of their own). Accelerator-only via
#: :attr:`UnlimitedOcrPipeline.shares_language_weights`; CPU is untouched.
#:
#: **bf16 only (KON-162 / KON-161 round 2).** The served int8 identity gate at
#: the commit that shipped this default read **0/12 in both request orders**
#: against the pinned KON-149 transcripts -- deterministic byte-for-byte across
#: three server processes, two pages coordinate-drift-only, ten gross, one
#: deterministic empty response -- while every in-process value-level A/B of the
#: mechanism is bitwise green for BOTH variants: the bare MoE layer at the
#: production stack shapes, the real-weight decode graph at the served shape
#: (129280 logits, bit-equal), the served prefill -> release -> decode sequence,
#: and full real pages generated to EOS (toc_dotted 646/646 tokens,
#: byte-identical to the pinned transcript under both flag settings; the
#: registry buffers themselves round-trip sha-clean at ~3 GiB). The served
#: divergence is therefore not attributable to the graphs or the registry at
#: the value level and remains unlocalised above the pipeline, so the int8
#: variant serves **unshared** -- the pinned-transcript configuration -- until a
#: served gate clears it: :attr:`UnlimitedOcrPipeline.shares_language_weights`
#: is also ``and not int8_experts``. What int8 gives back is the per-request
#: decode-reload saving only (its reload population was already 3.1-6.5 s
#: pre-registry, against bf16's 8.4 s), and it sheds the registry's +4-6 %
#: steady decode-step cost with it.
SHARE_LANGUAGE_WEIGHTS_DEFAULT = True


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
        ngram_window: int = 0,
    ) -> None:
        """Pass either the raw ``checkpoint`` or both pre-split state dicts."""
        if checkpoint is None and (vision_state_dict is None or language_state_dict is None):
            raise ValueError("pass either checkpoint= or both vision_state_dict= and language_state_dict=")
        self.config = config
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.crop_layout = crop_layout
        self.device = device or DeviceRef.CPU()
        self._driver_device = driver_device or CPU()
        self.session = session or InferenceSession(devices=[self._driver_device])
        self.window = config.decoder.sliding_window_size
        self.ngram_size = ngram_size
        self.ngram_window = ngram_window
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
        #: :attr:`on_accelerator` and ``not int8_experts`` (KON-162) and is the
        #: only thing the graph loaders read.
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
        self._decode: tuple[DecodeGraph, Model] | None = None
        #: The batched decode graphs, keyed by row count ``B >= 2``.
        self._batched_decode: dict[int, tuple[DecodeGraph, Model]] = {}
        # Compile the Mojo guard now, so a missing Metal Toolchain fails at load
        # rather than on the first request. Its own graph; the model graphs are unaffected.
        if self.ngram_blocker is not None:
            self.ngram_blocker.model

    @property
    def on_accelerator(self) -> bool:
        return self.device.device_type != DeviceKind.CPU

    @property
    def shares_language_weights(self) -> bool:
        """Whether every language graph binds ONE device registry of the language weights.

        :attr:`_share_language_weights` (:data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT`,
        **on**) gated by :attr:`on_accelerator` and by the weight variant -- the
        one place that conjunction is spelled. **False on CPU**, whatever the
        attribute says: there is no device budget to fit into, a host array *is*
        the graph's memory so there is no duplicate copy to remove, and the
        numerically gated path stays byte-identical by construction. **False in
        int8 mode** (KON-162): the served int8 identity gate falsified the
        registry commit at 0/12 in both request orders while every in-process
        value-level A/B of the mechanism -- up to full real pages byte-identical
        to the pinned transcripts under both flag settings -- is green, so int8
        serves in the pinned-transcript configuration (per-graph placement)
        until a served gate clears the shared registry for it. The full record
        is on :data:`SHARE_LANGUAGE_WEIGHTS_DEFAULT`.
        """
        return self._share_language_weights and self.on_accelerator and not self.config.decoder.int8_experts

    @property
    def non_expert_dtype(self) -> DType | None:
        """Storage dtype of the non-expert language weights: fp32 under :attr:`shares_language_weights`, else the checkpoint's.

        A bf16 weight read by an fp32 matmul is compile-folded into an fp32
        copy on the device at every ``session.load`` -- ~1.29 GiB per language
        graph for the projections, norms, router and ``lm_head``
        (EXPERIMENTS.md, OQ-113-A). Holding the exact fp32 upcast ONCE in the
        shared registry removes both graphs' copies for +0.65 GiB of registry.
        Off the registry (CPU, int8) nothing changes: ``None`` keeps the
        decoder's declarations, and so every emitted graph, as they were.
        """
        return COMPUTE_DTYPE if self.shares_language_weights else None

    @property
    def releases_language_graphs(self) -> bool:
        """Whether a caller must hold **one** language graph at a time.

        On an accelerator **without** the shared weight registry -- int8 today --
        each graph places its own weight copy, and holding both was measured
        over the Metal budget (KON-113: 18.629 of 17.760 GiB); int8's own
        hold-both ledger is unmeasured. **With** the registry (bf16 on an
        accelerator) the graphs hold no weights of their own since 0.3.2: the
        prefill graph costs 0.002 GiB and the decode graph 0.000 GiB beyond the
        registry, 7.67 of 17.760 GiB for everything on an M4, so releasing
        them would only re-pay a graph reload on every request. On CPU every
        graph stays resident and a reload would be pure loss. Named so the
        served path (``UnlimitedOCRModel._prefill``) carries the reason along
        instead of reading the device raw.

        A batched decode graph (:meth:`batched_decode_graph`) is one more
        language graph under the same rule: without the registry it places its
        own weight copy, so it counts against the one-at-a-time budget
        alongside the batch-1 decode graph. Its footprint beyond the registry
        is not measured yet.
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
        :attr:`non_expert_dtype` declares fp32 (:meth:`_fp32_resident_names`):
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
        """The weights :attr:`non_expert_dtype` declares fp32, read off a weightless decoder (declarations only)."""
        if self.non_expert_dtype is None:
            return frozenset()
        declared = UnlimitedOcrDecoder(
            self.config.decoder, dtype=self.config.dtype, device=self.device, non_expert_dtype=self.non_expert_dtype
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
            self.config.decoder, dtype=self.config.dtype, device=self.device, non_expert_dtype=self.non_expert_dtype
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

    @property
    def decode_graph(self) -> tuple[DecodeGraph, Model]:
        if self._decode is None:
            decoder = self._decoder()
            staged = build_decode_graph(
                self.config,
                decoder,
                max_seq_len=self.max_total_len,
                device=self.device,
                device_resident_weights=self.shares_language_weights,
            )
            self._decode = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
        return self._decode

    def batched_decode_graph(self, batch: int) -> tuple[DecodeGraph, Model]:
        """The decode step for ``batch >= 2`` independent requests, compiled lazily and cached per ``batch``.

        Built and loaded exactly like :attr:`decode_graph` -- a fresh decoder,
        the same ``max_seq_len``, the same weight declaration and registry --
        but it is a separate graph: :attr:`decode_graph` is untouched by it.
        """
        cached = self._batched_decode.get(batch)
        if cached is None:
            decoder = self._decoder()
            staged = build_batched_decode_graph(
                self.config,
                decoder,
                batch=batch,
                max_seq_len=self.max_total_len,
                device=self.device,
                device_resident_weights=self.shares_language_weights,
            )
            cached = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
            self._batched_decode[batch] = cached
        return cached

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
        """Drop every decode graph, batch-1 and batched. Idempotent, and safe before anything was built.

        The graphs (each a ``Model``) are what go; the shared language weight
        registry (:attr:`_language_device_weights`) deliberately does **not**.
        It is one device copy bound by *every* language graph across every
        release/reload cycle -- keeping it is the whole per-request saving
        (decode reload 8.4 -> ~1.5 s in the research port) -- and it could not
        be rebuilt anyway: :meth:`_resolved_language_weights` took the host
        arrays. Were a registry ever dropped here, the graphs would still have
        to go **first**: a ``Model`` reads those buffers, and dropping them
        under it would be a use-after-free of the weights the graph binds.
        """
        self._decode = None
        self._batched_decode.clear()
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
    def _execute(
        self,
        model: Model,
        names: Sequence[str],
        *arrays: np.ndarray | Buffer,
        host_outputs: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run one graph; a ``Buffer`` input is passed as is, arrays are staged onto the device.

        ``host_outputs`` names the outputs brought back as numpy (default: all);
        the rest are returned as the device buffers the graph produced.
        """
        if host_outputs is not None:
            unknown = set(host_outputs) - set(names)
            if unknown:
                raise ValueError(f"host_outputs names {sorted(unknown)}, which this graph does not output")
        buffers = [
            array if isinstance(array, Buffer) else Buffer.from_numpy(np.ascontiguousarray(array)).to(self._driver_device)
            for array in arrays
        ]
        outputs = model.execute(*buffers)
        wanted = set(names) if host_outputs is None else set(host_outputs)
        return {name: (to_host(out) if name in wanted else out) for name, out in zip(names, outputs, strict=True)}

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
        """Prefill the prompt and seed a fresh KV cache from the same pass."""
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        prompt_len = int(token_ids.shape[0])
        staged, model = self.prefill_graph_for(prompt_len)
        hidden = self.config.decoder.hidden_size
        rows = int(np.asarray(image_embeds).size // hidden)
        if rows != self.n_image_tokens:
            raise ValueError(f"{rows} image embedding rows for {self.n_image_tokens} placeholders")
        outputs = self._execute(model, staged.output_names, token_ids, image_embeds.reshape(-1, hidden))
        dec = self.config.decoder
        cache = allocate_kv_cache(
            num_layers=dec.num_hidden_layers,
            prefill_len=prompt_len,
            window=self.window,
            num_kv_heads=dec.num_key_value_heads,
            head_dim=dec.head_dim,
            device=self._driver_device if self.on_accelerator else None,
        )
        cache.seed(
            [outputs[f"key_{i}"] for i in range(dec.num_hidden_layers)],
            [outputs[f"value_{i}"] for i in range(dec.num_hidden_layers)],
        )
        return PrefillResult(logits=outputs["logits"].reshape(-1), cache=cache)

    def decode_step(self, cache: KvCache, token_id: int) -> np.ndarray:
        """Feed one token at ``cache.position`` and return its logits."""
        staged, model = self.decode_graph
        outputs = self._execute(
            model,
            staged.output_names,
            np.asarray([token_id], dtype=np.int64),
            np.asarray([cache.position], dtype=np.int32),
            cache.selector(),
            *cache.views(),
            host_outputs=cache.HOST_OUTPUTS,
        )
        cache.append(
            [outputs[f"key_{i}"] for i in range(staged.num_layers)],
            [outputs[f"value_{i}"] for i in range(staged.num_layers)],
        )
        return outputs["logits"].reshape(-1)

    def decode_rows(self, caches: Sequence[KvCache], token_ids: Sequence[int]) -> np.ndarray:
        """Feed ``token_ids[b]`` to request ``b`` at ``caches[b].position``; ``[B, vocab]`` fp32 logits, row ``b`` for it.

        ``B == 1`` is :meth:`decode_step` -- the unchanged batch-1 graph --
        reshaped to ``[1, vocab]``. ``B >= 2`` is one execute of
        :meth:`batched_decode_graph`, then each cache appends its own row
        ``b`` of every ``key_i`` / ``value_i``. On a
        :class:`~unlimited_ocr_max.kv_cache.DeviceKvCache` those rows are
        ``[b : b + 1, :, :]`` views of the graph's own device outputs, ordered
        behind the submission that produced them, so the async copy needs no
        drain (the rule ``DeviceKvCache._write_row`` states).

        Every cache must be of one class: the single execute brings back one
        set of host outputs, ``caches[0].HOST_OUTPUTS``.
        """
        if len(caches) != len(token_ids):
            raise ValueError(f"{len(caches)} caches for {len(token_ids)} tokens")
        if not caches:
            raise ValueError("decode_rows needs at least one row")
        kinds = {type(cache) for cache in caches}
        if len(kinds) != 1:
            raise TypeError(f"every cache must be of one class, got {sorted(kind.__name__ for kind in kinds)}")
        if len(caches) == 1:
            return self.decode_step(caches[0], token_ids[0]).reshape(1, -1)
        staged, model = self.batched_decode_graph(len(caches))
        inputs: list[Any] = [
            np.asarray(token_ids, dtype=np.int64),
            np.asarray([cache.position for cache in caches], dtype=np.int32),
        ]
        for cache in caches:
            inputs += [cache.selector(), *cache.views()]
        outputs = self._execute(model, staged.output_names, *inputs, host_outputs=caches[0].HOST_OUTPUTS)
        for b, cache in enumerate(caches):
            cache.append(
                [outputs[f"key_{i}"][b : b + 1, :, :] for i in range(staged.num_layers)],
                [outputs[f"value_{i}"][b : b + 1, :, :] for i in range(staged.num_layers)],
            )
        return outputs["logits"]

    def generate(self, *, pixels: np.ndarray, token_ids: np.ndarray, local_pixels: np.ndarray | None = None) -> list[int]:
        """Greedy decode to EOS (included) or ``max_new_tokens``; the n-gram guard, if on, is applied to every step."""
        stages = self.run_vision(pixels, local_pixels)
        if self.releases_language_graphs:
            self.release_vision()
        prefill = self.run_prefill(token_ids, stages["image_embeds"])
        if self.releases_language_graphs:
            self.release_prefill()

        eos = self.config.decoder.eos_token_id
        blocker = self.ngram_blocker
        sequence = [int(i) for i in np.asarray(token_ids).reshape(-1)]
        generated: list[int] = []
        logits = prefill.logits
        for _ in range(self.max_new_tokens):
            if blocker is not None:
                logits = blocker.apply(logits, sequence)
            token = int(logits.argmax())
            generated.append(token)
            sequence.append(token)
            if token == eos:
                break
            logits = self.decode_step(prefill.cache, token)
        return generated

    @property
    def ngram_blocker(self) -> NgramBlocker | None:
        if self.ngram_size <= 0 or self.ngram_window <= 0:
            return None
        if self._ngram is None:
            self._ngram = NgramBlocker(
                ngram_size=self.ngram_size,
                window=self.ngram_window,
                vocab_size=self.config.decoder.vocab_size,
                device=self.device,
                driver_device=self._driver_device,
                session=self.session,
            )
        return self._ngram
