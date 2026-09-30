"""The registered ``max.pipelines`` model: three graphs, spliced embeddings, a per-request KV cache.

``TextGenerationPipeline`` requires a ``PipelineModelWithKVCache`` and MAX's
memory planner requires an ``ArchConfigWithKVCache``, so both declare a
deliberately empty paged cache (1 layer, 1 head, ``head_dim`` 1: 16 KiB at
``--max-length 2048``) that nothing reads. The real cache is
:class:`~unlimited_ocr_max.kv_cache.KvCache`, one per in-flight request, keyed
by request id. ``base`` mode only; batch size 1 unless ``--max-batch-size`` raises it, bf16 on an
accelerator only, capped at ``MAX_BATCH_CAP``. A scheduler step is all-prefill or all-decode
(in-flight batching is off): a prefill batch runs as one batch-1 prefill per request, a decode
batch as one :meth:`~unlimited_ocr_max.pipeline.UnlimitedOcrPipeline.decode_rows` execute.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
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
from .graphs import check_int8_device
from .kv_cache import KvCache
from .model_config import DecoderConfig, UnlimitedOCRConfig
from .ngram import DEFAULT_NGRAM_SIZE
from .pipeline import UnlimitedOcrPipeline
from .weight_adapters import is_int8_checkpoint

__all__ = [
    "MAX_BATCH_CAP",
    "MAX_BATCH_SIZE_ENV",
    "NGRAM_SIZE_ENV",
    "UnlimitedOCRModel",
    "UnlimitedOcrArchConfig",
    "UnlimitedOcrInputs",
    "check_max_batch_size",
    "hf_config_as_dict",
    "placeholder_kv_params",
    "serve_max_batch_size",
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


#: Internal transport, set by the ``unlimited-ocr-max serve`` CLI for the ``max serve`` child it
#: launches: ``arch.py``'s ``required_arguments`` forces ``max_batch_size`` (MAX applies it over user
#: flags, so a ``--max-batch-size`` passed straight to ``max serve`` would just be overridden), so
#: this is how the CLI's flag actually reaches the architecture. Unset or blank means batch size 1,
#: so a direct ``max serve --custom-architectures`` user gets it too.
MAX_BATCH_SIZE_ENV = "_UNLIMITED_OCR_MAX_MAX_BATCH_SIZE"

#: Above this, ``--max-batch-size`` is refused: batched decoding (bf16 on an accelerator only) is
#: unvalidated past it.
MAX_BATCH_CAP = 8


def serve_max_batch_size() -> int:
    raw = os.environ.get(MAX_BATCH_SIZE_ENV, "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{MAX_BATCH_SIZE_ENV}={raw!r} is not an integer") from exc
    if value < 1:
        raise ValueError(f"{MAX_BATCH_SIZE_ENV}={raw!r} must be >= 1")
    return value


def check_max_batch_size(max_batch_size: int, *, device: DeviceRef, int8: bool) -> None:
    """Refuse a ``--max-batch-size`` neither this device nor this checkpoint can serve.

    Batched decoding is bf16 on an accelerator only. Called once both facts are known -- cheaply,
    and before any expensive pipeline construction."""
    if max_batch_size <= 1:
        return
    if device.is_cpu():
        raise ValueError(f"--max-batch-size {max_batch_size} needs an accelerator; the pipeline is on cpu")
    if int8:
        raise ValueError(f"--max-batch-size {max_batch_size} is bf16-only; this checkpoint is int8")
    if max_batch_size > MAX_BATCH_CAP:
        raise ValueError(f"--max-batch-size {max_batch_size} exceeds the cap of {MAX_BATCH_CAP}")


@dataclass(kw_only=True)
class UnlimitedOcrInputs(ModelInputs):
    #: Every context's active tokens, concatenated in batch order.
    tokens: Buffer
    #: How many of ``tokens`` each context owns, in batch order.
    token_counts: tuple[int, ...] = ()
    #: On a prefill step, one buffer per context, in batch order; ``None`` on a decode step.
    pixel_values: list[Buffer] | None = None
    #: On a prefill step, each context's own placeholder rows, one buffer per context (or ``None``).
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
        """Refuse what cannot be served, adapt the weights and build the pipeline.

        With ``--max-batch-size N > 1`` (bf16 on an accelerator; everything
        else was refused by :func:`check_max_batch_size`) every decode step
        runs a graph of at least two rows -- a lone request is padded to two
        (``min_decode_rows = 2``) -- so a request's output does not depend on
        how many others are in flight, and the ``B = 2 .. N`` decode graphs
        are loaded here, before the first request. Those graphs bind the
        shared language weight registry, so in this mode its one-time build
        moves from the first prefill to startup. ``N == 1`` changes nothing:
        the batch-1 decode graph, compiled lazily, and no padding.
        """
        super().__init__(*args, **kwargs)
        self._session: InferenceSession = kwargs["session"]
        # `max_seq_len` is the memory plan's resolved length; the graphs must agree with it.
        self._arch_config = UnlimitedOcrArchConfig.initialize(self.pipeline_config, max_seq_len=self.max_seq_len)
        geometry = ViewGeometry(self._image_size())
        if self.adapter is None:
            raise ValueError("UnlimitedOCRModel needs the safetensors weight adapter registered in arch.py")
        weights = dict(self.weights.items())
        is_int8 = self._weights_are_int8(weights)
        max_batch_size = serve_max_batch_size()
        check_max_batch_size(max_batch_size, device=self.device_refs[0], int8=is_int8)
        if int(self.max_batch_size) != max_batch_size:
            # `required_arguments` makes them equal; only `max serve --force` skips it, and then
            # the scheduler would batch rows this worker has neither warmed nor padded for.
            raise ValueError(
                f"the scheduler batches up to {self.max_batch_size} requests but {MAX_BATCH_SIZE_ENV} "
                f"is {max_batch_size}; set the batch size with `unlimited-ocr-max serve --max-batch-size`"
            )
        if is_int8:
            # The decoder the adapter checks the file against must declare the int8 stacks and their scales.
            self._arch_config.model = self._arch_config.model.with_int8_experts()
            check_int8_device(self._arch_config.decoder, self.device_refs[0])
        renamed = self.adapter(weights, config=self._arch_config.model, image_size=geometry.image_size)
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
        if max_batch_size > 1:
            self._pipeline.min_decode_rows = 2
            self._pipeline.warm_decode_graphs(max_batch_size)
        self._served: dict[str, ServedRequest] = {}

    @staticmethod
    def _weights_are_int8(weights: dict[str, Any]) -> bool:
        """Whether the checkpoint's routed experts are int8, read off the expert tensors' dtypes.

        Only the expert entries are materialised (mmap-backed ``WeightData``,
        cached inside each ``Weights`` so the adapter's own read reuses it).
        """
        experts = {name: source.data() for name, source in weights.items() if ".mlp.experts." in name}
        return is_int8_checkpoint(experts)

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

    def _logits_buffer(self, rows: Sequence[np.ndarray]) -> Buffer:
        """``[B, vocab]`` fp32 on the model's device, row ``b`` for context ``b``."""
        flat = np.ascontiguousarray(np.stack([np.asarray(row, dtype=np.float32).reshape(-1) for row in rows]))
        return Buffer.from_numpy(flat).to(self.devices[0])

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """One scheduler step over the batch's ``B`` contexts; ``[B, vocab]`` logits, row ``b`` for context ``b``.

        In-flight batching is off, so a step is all-prefill or all-decode. With
        pixels present it is a prefill batch: one batch-1 :meth:`_prefill` per
        context, in batch order, context ``b`` taking ``pixel_values[b]`` and
        ``image_token_indices[b]``. Otherwise every context decodes one token,
        all in one :meth:`_decode`. The n-gram guard is applied row by row,
        each row against its own request's sequence.
        """
        assert isinstance(model_inputs, UnlimitedOcrInputs)
        if not model_inputs.request_ids:
            raise ValueError("UnlimitedOcrInputs carries no request ids; build inputs through the batch processor")
        request_ids = [request_key(request_id) for request_id in model_inputs.request_ids]
        tokens = np.asarray(model_inputs.tokens.to(CPU()).to_numpy(), dtype=np.int64).reshape(-1)
        rows = self._split_tokens(tokens, model_inputs.token_counts, len(request_ids))
        logits: list[np.ndarray]
        if model_inputs.has_vision_inputs:
            pixels = model_inputs.pixel_values
            indices = model_inputs.image_token_indices
            assert pixels is not None
            if len(pixels) != len(request_ids) or (indices is not None and len(indices) != len(request_ids)):
                raise ValueError(
                    f"{len(request_ids)} requests need one pixel buffer (and one placeholder buffer) each; "
                    f"got {len(pixels)} and {'none' if indices is None else len(indices)}"
                )
            logits = [
                self._prefill(request_id, row, pixels[b], None if indices is None else indices[b])
                for b, (request_id, row) in enumerate(zip(request_ids, rows, strict=True))
            ]
        else:
            logits = list(self._decode(request_ids, rows))
        blocker = self._pipeline.ngram_blocker
        if blocker is not None:
            logits = [
                blocker.apply(row, self._served[request_id].sequence)
                for request_id, row in zip(request_ids, logits, strict=True)
            ]
        buffer = self._logits_buffer(logits)
        return ModelOutputs(next_token_logits=buffer, logits=buffer)

    @staticmethod
    def _split_tokens(tokens: np.ndarray, token_counts: Sequence[int], n_requests: int) -> list[np.ndarray]:
        """``tokens`` cut into each context's own, in batch order, by ``token_counts``."""
        if len(token_counts) != n_requests or sum(token_counts) != int(tokens.shape[0]):
            raise ValueError(
                f"token_counts {tuple(token_counts)} do not split {tokens.shape[0]} tokens over {n_requests} "
                "requests; build inputs through the batch processor"
            )
        return np.split(tokens, np.cumsum(token_counts)[:-1])

    def _prefill(
        self, request_id: str, tokens: np.ndarray, pixel_values: Buffer, image_token_indices: Buffer | None
    ) -> np.ndarray:
        """One request's vision tower, then the static-length prefill graph, over its own ``tokens``.

        The policy is the pipeline's named ``releases_language_graphs``
        predicate. Where it holds (an accelerator without the shared registry)
        the decode release comes **unconditionally first**: no failure path can
        reach the prefill load with the previous request's decode graph still
        resident. Where both graphs stay resident on an accelerator, the prefill
        cache is bounded to this prompt's length instead.
        """
        transient = self._pipeline.releases_language_graphs
        if transient:
            self._pipeline.release_decode()
        elif self._pipeline.on_accelerator:
            self._pipeline.retain_only_prefill(int(tokens.shape[0]))
        pixels = pixel_values.to(CPU()).to_numpy()
        stages = self._pipeline.run_vision(np.ascontiguousarray(pixels))
        self._pipeline.drop_vision_weights()
        image_embeds = stages["image_embeds"]
        if image_token_indices is not None:
            expected = int(image_token_indices.shape[0])
            if expected != int(image_embeds.shape[0]):
                raise ValueError(f"vision produced {image_embeds.shape[0]} rows but the prompt has {expected} placeholders")
        result = self._pipeline.run_prefill(tokens, image_embeds)
        self._served[request_id] = ServedRequest(cache=result.cache, sequence=[int(i) for i in tokens])
        if transient:
            self._pipeline.release_prefill()
        return result.logits

    def _decode(self, request_ids: Sequence[str], rows: Sequence[np.ndarray]) -> np.ndarray:
        """One token per request, all in ONE ``decode_rows`` call; ``[B, vocab]``, row ``b`` for request ``b``.

        Every request is checked before any sequence moves.
        """
        states: list[ServedRequest] = []
        for request_id, row in zip(request_ids, rows, strict=True):
            state = self._served.get(request_id)
            if state is None:
                raise ValueError(f"request {request_id} asked for a decode step with no prefill on record")
            if row.shape[0] != 1:
                raise ValueError(f"this decoder steps one token at a time; the scheduler asked for {row.shape[0]}")
            if any(state is seen for seen in states):
                raise ValueError(f"request {request_id} appears twice in one decode batch")
            states.append(state)
        tokens = [int(row[0]) for row in rows]
        for state, token in zip(states, tokens, strict=True):
            state.sequence.append(token)
        return self._pipeline.decode_rows([state.cache for state in states], tokens)
