"""The in-process pipeline: vision graph(s) + prefill graph + decode graph, and greedy decoding.

Graphs are compiled lazily. On an accelerator the pipeline holds one language
graph at a time (prefill is released once its host outputs are back, the decode
graph before the next prefill) and keeps the KV cache device-resident; on CPU
every graph stays resident and the cache is host numpy. ``base`` mode is one
1024px view; a tiled :class:`~unlimited_ocr_max.batch_processor.CropLayout`
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
from max.engine import InferenceSession, Model
from max.graph import DeviceKind, DeviceRef

from .batch_processor import BASE_SIZE, LOCAL_SIZE, CropLayout, ViewGeometry
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
from .kv_cache import KvCache, allocate_kv_cache, to_host
from .layers.projector import image_token_count
from .model_config import UnlimitedOCRConfig
from .ngram import NgramBlocker
from .weight_adapters import language_state_dict, vision_state_dict

__all__ = ["PrefillResult", "UnlimitedOcrPipeline"]


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
        self._vision: tuple[VisionGraph, Model] | None = None
        self._local_vision: tuple[VisionGraph, Model] | None = None
        self._layout: tuple[LayoutGraph, Model] | None = None
        self._prefill: dict[int, tuple[LanguageGraph, Model]] = {}
        self._decode: tuple[DecodeGraph, Model] | None = None
        # Compile the Mojo guard now, so a missing Metal Toolchain fails at load
        # rather than on the first request. Its own graph; the model graphs are unaffected.
        if self.ngram_blocker is not None:
            self.ngram_blocker.model

    @property
    def on_accelerator(self) -> bool:
        return self.device.device_type != DeviceKind.CPU

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

    def _decoder(self) -> UnlimitedOcrDecoder:
        """A fresh, weight-loaded decoder: a ``Weight`` binds to one graph, the arrays behind it are shared."""
        decoder = UnlimitedOcrDecoder(self.config.decoder, dtype=self.config.dtype, device=self.device)
        decoder.load_state_dict(self._resolved_language_state_dict())
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
                self.config, decoder, seq_len=seq_len, n_image_tokens=self.n_image_tokens, device=self.device
            )
            cached = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
            self._prefill[seq_len] = cached
        return cached

    @property
    def decode_graph(self) -> tuple[DecodeGraph, Model]:
        if self._decode is None:
            decoder = self._decoder()
            staged = build_decode_graph(self.config, decoder, max_seq_len=self.max_total_len, device=self.device)
            self._decode = (staged, self.session.load(staged.graph, weights_registry=decoder.state_dict()))
        return self._decode

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
        self._decode = None
        gc.collect()

    def release_prefill(self) -> None:
        self._prefill.clear()
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

    def generate(self, *, pixels: np.ndarray, token_ids: np.ndarray, local_pixels: np.ndarray | None = None) -> list[int]:
        """Greedy decode to EOS (included) or ``max_new_tokens``; the n-gram guard, if on, is applied to every step."""
        stages = self.run_vision(pixels, local_pixels)
        if self.on_accelerator:
            self.release_vision()
        prefill = self.run_prefill(token_ids, stages["image_embeds"])
        if self.on_accelerator:
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
