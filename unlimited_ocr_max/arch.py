"""Registration of ``UnlimitedOCRForCausalLM`` as a MAX custom architecture.

``name`` must equal ``config.json``'s ``architectures[0]`` exactly. Prefix
caching and chunked prefill are forced off: the language graph splices the
image embeddings at 273 placeholder rows of the whole prompt, and batch size is
forced to 1 because the prefill graph's ``seq_len`` is static.
"""

from __future__ import annotations

from typing import Any

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.modeling.types import PipelineTask

from .batch_processor import BASE_SIZE, UnlimitedOcrBatchProcessor
from .model import UnlimitedOcrArchConfig, UnlimitedOCRModel
from .model_config import UnlimitedOCRConfig
from .tokenizer import UnlimitedOcrTokenizer
from .weight_adapters import LANGUAGE_MODEL, VISION, language_state_dict, vision_state_dict

#: What ``config.json`` says the weights are. ``SupportedEncoding`` is a ``Literal`` of strings.
DEFAULT_ENCODING: SupportedEncoding = "bfloat16"

#: What both devices are served under: MAX refuses ``bfloat16`` on CPU, and
#: ``float32`` is the compute dtype on either device (bf16 storage, fp32 arithmetic).
SERVED_ENCODING: SupportedEncoding = "float32"


def _as_tensor(source: Any) -> Any:
    """``Weights.data()`` is a ``WeightData`` exposing its buffer through DLPack; numpy has no bf16, so torch is the bridge."""
    import torch

    return torch.from_dlpack(source.data().data)


def convert_state_dict(
    state_dict: dict[str, Any], *, config: UnlimitedOCRConfig, image_size: int = BASE_SIZE
) -> dict[str, dict[str, Any]]:
    """``WeightsAdapter``: checkpoint names -> ``{"vision": ..., "language_model": ...}``."""
    checkpoint = {key: _as_tensor(source) for key, source in state_dict.items()}
    return {
        VISION: vision_state_dict(checkpoint, image_size=image_size),
        LANGUAGE_MODEL: language_state_dict(checkpoint, config),
    }


unlimited_ocr_arch = SupportedArchitecture(
    name="UnlimitedOCRForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["kthierbach/unlimited-ocr-max"],
    default_encoding=DEFAULT_ENCODING,
    supported_encodings={DEFAULT_ENCODING, SERVED_ENCODING},
    pipeline_model=UnlimitedOCRModel,
    tokenizer=UnlimitedOcrTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={WeightsFormat.safetensors: convert_state_dict},
    multi_gpu_supported=False,
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_batch_size": 1,
    },
    config=UnlimitedOcrArchConfig,
    batching=UnlimitedOcrBatchProcessor,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

__all__ = ["DEFAULT_ENCODING", "SERVED_ENCODING", "convert_state_dict", "unlimited_ocr_arch"]
