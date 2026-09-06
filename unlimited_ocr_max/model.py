"""The registered ``max.pipelines`` model: three graphs, spliced embeddings, a per-request KV cache.

``TextGenerationPipeline`` requires a ``PipelineModelWithKVCache`` and MAX's
memory planner requires an ``ArchConfigWithKVCache``, so both declare a
deliberately empty paged cache (1 layer, 1 head, ``head_dim`` 1: 16 KiB at
``--max-length 2048``) that nothing reads. The real cache is
:class:`~unlimited_ocr_max.kv_cache.KvCache`, one per in-flight request, keyed
by request id. ``base`` mode and batch size 1 only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.pipeline_model import ModelInputs, ModelOutputs, PipelineModelWithKVCache

from .batch_processor import BASE_SIZE, UnlimitedOcrBatchProcessor, ViewGeometry, request_key
from .decoder import COMPUTE_DTYPE
from .kv_cache import KvCache
from .model_config import DecoderConfig, UnlimitedOCRConfig
from .ngram import DEFAULT_NGRAM_SIZE
from .pipeline import UnlimitedOcrPipeline

__all__ = [
    "NGRAM_SIZE_ENV",
    "UnlimitedOCRModel",
    "UnlimitedOcrArchConfig",
    "UnlimitedOcrInputs",
    "hf_config_as_dict",
    "placeholder_kv_params",
    "serve_ngram_size",
]

#: Internal transport, set by the ``unlimited-ocr-max serve`` CLI for the ``max serve`` child it
#: launches: ``no_repeat_ngram_size`` is neither an OpenAI request field nor a ``max serve`` flag.
#: Unset means the shipped default, so a direct ``max serve --custom-architectures`` user gets it too.
NGRAM_SIZE_ENV = "_UNLIMITED_OCR_MAX_NGRAM_SIZE"


def serve_ngram_size() -> int:
    raw = os.environ.get(NGRAM_SIZE_ENV, "").strip()
    if not raw:
        return DEFAULT_NGRAM_SIZE
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{NGRAM_SIZE_ENV}={raw!r} is not an integer") from exc


@dataclass(kw_only=True)
class UnlimitedOcrInputs(ModelInputs):
    tokens: Buffer
    pixel_values: list[Buffer] | None = None
    image_token_indices: list[Buffer] | None = None
    #: One request id per context, in batch order; the KV cache is addressed by it.
    request_ids: tuple[str, ...] = ()

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


PLACEHOLDER_PAGE_SIZE = 128


def placeholder_kv_params(dtype: DType, devices: list[DeviceRef]) -> Any:
    """The empty paged cache the scheduler insists on; it buys admission bookkeeping and nothing else."""
    from max.nn.kv_cache import MHAKVCacheParams

    return MHAKVCacheParams(
        dtype=dtype,
        head_dim=1,
        num_layers=1,
        n_kv_heads=1,
        devices=list(devices),
        page_size=PLACEHOLDER_PAGE_SIZE,
        enable_prefix_caching=False,
    )


def hf_config_as_dict(huggingface_config: Any) -> dict[str, Any]:
    """A ``config.json``-shaped dict; transformers 5 renames ``torch_dtype`` to ``dtype`` in ``to_dict()``."""
    raw = huggingface_config.to_dict() if hasattr(huggingface_config, "to_dict") else dict(huggingface_config)
    if "torch_dtype" not in raw and "dtype" in raw:
        raw["torch_dtype"] = raw["dtype"]
    return raw


@dataclass
class UnlimitedOcrArchConfig:
    """MAX's ``ArchConfigWithKVCache`` protocol over :class:`UnlimitedOCRConfig`."""

    #: Read by ``PipelineModel._resolved_encoding`` in the model worker; must equal ``arch.DEFAULT_ENCODING``.
    DEFAULT_ENCODING: ClassVar[str] = "bfloat16"

    model: UnlimitedOCRConfig
    max_seq_len: int
    devices: list[DeviceRef]
    dtype: DType = COMPUTE_DTYPE

    @classmethod
    def initialize(
        cls, pipeline_config: Any, model_config: Any | None = None, *, max_seq_len: int | None = None
    ) -> UnlimitedOcrArchConfig:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        parsed = UnlimitedOCRConfig.from_hf_dict(hf_config_as_dict(huggingface_config))
        if max_seq_len is None:
            max_seq_len = cls.calculate_max_seq_len(huggingface_config, model_config)
        return cls(
            model=parsed,
            max_seq_len=int(max_seq_len),
            devices=[DeviceRef.from_device(d) for d in getattr(pipeline_config, "devices", [])] or [DeviceRef.CPU()],
            dtype=COMPUTE_DTYPE,
        )

    @classmethod
    def calculate_max_seq_len(cls, huggingface_config: Any, model_config: Any) -> int:
        """``--max-length`` when set, else the checkpoint's ``max_position_embeddings``."""
        max_length = getattr(model_config, "max_length", None)
        if max_length:
            return int(max_length)
        parsed = UnlimitedOCRConfig.from_hf_dict(hf_config_as_dict(huggingface_config))
        return int(parsed.decoder.max_position_embeddings)

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_kv_params(self) -> Any:
        return placeholder_kv_params(self.dtype, self.devices)

    @property
    def decoder(self) -> DecoderConfig:
        return self.model.decoder


@dataclass
class ServedRequest:
    cache: KvCache
    #: Every token so far, prompt included -- what the n-gram guard sees.
    sequence: list[int]


class UnlimitedOCRModel(PipelineModelWithKVCache[TextAndVisionContext]):
    model_config_cls: ClassVar[type[Any]] = UnlimitedOcrArchConfig
    # Load-bearing: the model worker is a spawned process that re-imports this class
    # without the registration step that would otherwise set it from `arch.batching`.
    batch_processor_cls: ClassVar[type[Any]] = UnlimitedOcrBatchProcessor

    @classmethod
    def get_kv_params(
        cls, huggingface_config: Any, pipeline_config: Any, devices: list[DeviceRef], kv_cache_config: Any, cache_dtype: DType
    ) -> Any:
        del huggingface_config, pipeline_config, kv_cache_config
        return placeholder_kv_params(cache_dtype, devices)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session: InferenceSession = kwargs["session"]
        # `max_seq_len` is the memory plan's resolved length; the graphs must agree with it.
        self._arch_config = UnlimitedOcrArchConfig.initialize(self.pipeline_config, max_seq_len=self.max_seq_len)
        geometry = ViewGeometry(self._image_size())
        if self.adapter is None:
            raise ValueError("UnlimitedOCRModel needs the safetensors weight adapter registered in arch.py")
        renamed = self.adapter(dict(self.weights.items()), config=self._arch_config.model, image_size=geometry.image_size)
        prompt_len = self._prompt_len(geometry)
        self._pipeline = UnlimitedOcrPipeline(
            self._arch_config.model,
            vision_state_dict=renamed["vision"],
            language_state_dict=renamed["language_model"],
            seq_len=prompt_len,
            # Sizes the decode RoPE table to the whole context window.
            max_new_tokens=max(int(self.max_seq_len) - prompt_len, 1),
            image_size=geometry.image_size,
            device=self.device_refs[0],
            driver_device=self.devices[0],
            session=self._session,
            ngram_size=serve_ngram_size(),
            ngram_window=self._arch_config.decoder.sliding_window_size,
        )
        self._served: dict[str, ServedRequest] = {}

    def _image_size(self) -> int:
        resolutions = getattr(self.huggingface_config, "candidate_resolutions", None)
        return int(resolutions[0][0]) if resolutions else BASE_SIZE

    def _prompt_len(self, geometry: ViewGeometry) -> int:
        """The default prompt's length (277 in base mode), tokenised through the same delegate the tokenizer uses."""
        from .tokenizer import DEFAULT_PROMPT, build_prompt, load_delegate

        delegate = load_delegate(self.pipeline_config.model.model_path)
        return build_prompt(
            lambda text: delegate.encode(text, add_special_tokens=False), prompt=DEFAULT_PROMPT, geometry=geometry
        ).seq_len

    def release(self, request_id: Any) -> None:
        """Drop a finished request's KV cache (device bytes on an accelerator)."""
        self._served.pop(request_key(request_id), None)

    def _logits_buffer(self, logits: np.ndarray) -> Buffer:
        flat = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(1, -1))
        return Buffer.from_numpy(flat).to(self.devices[0])

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """One scheduler step: prefill when pixels are present, else one decode token. Guard applied to the logits."""
        assert isinstance(model_inputs, UnlimitedOcrInputs)
        if not model_inputs.request_ids:
            raise ValueError("UnlimitedOcrInputs carries no request ids; build inputs through the batch processor")
        request_id = request_key(model_inputs.request_ids[0])
        tokens = np.asarray(model_inputs.tokens.to(CPU()).to_numpy(), dtype=np.int64).reshape(-1)
        if model_inputs.has_vision_inputs:
            logits = self._prefill(request_id, model_inputs, tokens)
        else:
            logits = self._decode(request_id, tokens)
        blocker = self._pipeline.ngram_blocker
        if blocker is not None:
            logits = blocker.apply(logits, self._served[request_id].sequence)
        buffer = self._logits_buffer(logits)
        return ModelOutputs(next_token_logits=buffer, logits=buffer)

    def _prefill(self, request_id: str, model_inputs: UnlimitedOcrInputs, tokens: np.ndarray) -> np.ndarray:
        """Vision tower, then the static-length prefill graph; one language graph at a time on an accelerator."""
        assert model_inputs.pixel_values is not None
        transient = self._pipeline.on_accelerator
        if transient:
            self._pipeline.release_decode()
        pixels = model_inputs.pixel_values[0].to(CPU()).to_numpy()
        stages = self._pipeline.run_vision(np.ascontiguousarray(pixels))
        self._pipeline.drop_vision_weights()
        image_embeds = stages["image_embeds"]
        if model_inputs.image_token_indices is not None:
            expected = int(model_inputs.image_token_indices[0].shape[0])
            if expected != int(image_embeds.shape[0]):
                raise ValueError(f"vision produced {image_embeds.shape[0]} rows but the prompt has {expected} placeholders")
        result = self._pipeline.run_prefill(tokens, image_embeds)
        self._served[request_id] = ServedRequest(cache=result.cache, sequence=[int(i) for i in tokens])
        if transient:
            self._pipeline.release_prefill()
        return result.logits

    def _decode(self, request_id: str, tokens: np.ndarray) -> np.ndarray:
        state = self._served.get(request_id)
        if state is None:
            raise ValueError(f"request {request_id} asked for a decode step with no prefill on record")
        if tokens.shape[0] != 1:
            raise ValueError(f"this decoder steps one token at a time; the scheduler asked for {tokens.shape[0]}")
        token = int(tokens[0])
        state.sequence.append(token)
        return self._pipeline.decode_step(state.cache, token)
