"""Image preprocessing, view geometry, gundam tiling and the ``max.pipelines`` batch processor.

Preprocessing reproduces the reference's ``infer()`` bitwise: pad to square with
the mean colour (``int(0.5 * 255) == 127``), ``ToTensor``, normalise with
mean/std 0.5, then a bf16 round-trip (``bf16.fp32_to_bf16_roundtrip``) so the
fp32 pixels hold only bf16-representable values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessor, BatchProcessorRuntime
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs

from .bf16 import fp32_to_bf16_roundtrip

if TYPE_CHECKING:
    from PIL import Image

    from .model import UnlimitedOcrInputs

__all__ = [
    "BASE_SIZE",
    "LOCAL_SIZE",
    "CropLayout",
    "GundamViews",
    "UnlimitedOcrBatchProcessor",
    "ViewGeometry",
    "preprocess_page",
    "preprocess_page_gundam",
    "request_key",
    "select_crop_layout",
]


def request_key(value: Any) -> str:
    """Fold a request id to ``str``: MAX hands ``release`` a ``RequestID`` dataclass, not the string it wraps."""
    return str(value)


PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
BASE_SIZE = 1024
LOCAL_SIZE = 640
MIN_TILES = 2
MAX_TILES = 32
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5
IMAGE_NDIMS = 4


@dataclass(frozen=True)
class ViewGeometry:
    """Token geometry of one square view: SAM's grid is ``image_size / 16 / 4`` per side."""

    image_size: int

    def __post_init__(self) -> None:
        if self.image_size % (PATCH_SIZE * DOWNSAMPLE_RATIO):
            raise ValueError(f"image_size {self.image_size} is not a multiple of {PATCH_SIZE * DOWNSAMPLE_RATIO}")

    @property
    def grid(self) -> int:
        return math.ceil((self.image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)

    @property
    def token_grid(self) -> tuple[int, int]:
        return (self.grid, self.grid)


def load_page(source: str | Path | Image.Image) -> Image.Image:
    from PIL import Image as PILImage

    image = PILImage.open(source) if isinstance(source, (str, Path)) else source
    return image if image.mode == "RGB" else image.convert("RGB")


def pad_to_square(image: Image.Image, size: int) -> Image.Image:
    from PIL import ImageOps

    fill = int(PIXEL_MEAN * 255)
    return ImageOps.pad(image, (size, size), color=(fill, fill, fill))


def normalise_view(image: Image.Image) -> np.ndarray:
    """``ToTensor`` + ``Normalize(0.5, 0.5)`` + the bf16 round-trip -> ``[3, H, W]`` fp32."""
    hwc = np.array(image, dtype=np.uint8, copy=True)
    if hwc.ndim != 3 or hwc.shape[2] != 3:
        raise ValueError(f"expected an RGB HWC image, got shape {hwc.shape}")
    chw = np.ascontiguousarray(hwc.transpose(2, 0, 1)).astype(np.float32) / np.float32(255.0)
    pixels = (chw - np.float32(PIXEL_MEAN)) / np.float32(PIXEL_STD)
    return fp32_to_bf16_roundtrip(pixels)


def preprocess_page(source: str | Path | Image.Image, *, base_size: int = BASE_SIZE) -> np.ndarray:
    """``base`` mode pixels: ``[1, 3, base_size, base_size]``."""
    return normalise_view(pad_to_square(load_page(source), base_size))[None, ...]


def candidate_crop_ratios() -> list[tuple[int, int]]:
    """``dynamic_preprocess``' ``target_ratios``: a set comprehension, then a stable sort by tile count.

    Reproduced verbatim because :func:`find_closest_aspect_ratio` resolves exact
    ties in favour of the first candidate reached.
    """
    target_ratios = set(
        (i, j)
        for n in range(MIN_TILES, MAX_TILES + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= MAX_TILES and i * j >= MIN_TILES
    )
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: Sequence[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    """The reference's grid choice; the strict ``>`` area tie-break is load-bearing."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


@dataclass(frozen=True)
class CropLayout:
    """The reference's ``crop_ratio`` in both orders: width-major as stored, ``(rows, cols)`` for the layout."""

    width_crop_num: int
    height_crop_num: int

    def __post_init__(self) -> None:
        if self.width_crop_num < 1 or self.height_crop_num < 1:
            raise ValueError(f"non-positive crop layout {self.crop_ratio}")

    @property
    def crop_ratio(self) -> tuple[int, int]:
        return (self.width_crop_num, self.height_crop_num)

    @property
    def crop_grid(self) -> tuple[int, int]:
        return (self.height_crop_num, self.width_crop_num)

    @property
    def n_tiles(self) -> int:
        return self.width_crop_num * self.height_crop_num

    @property
    def tiled(self) -> bool:
        return self.width_crop_num > 1 or self.height_crop_num > 1


def select_crop_layout(size: tuple[int, int], *, image_size: int = LOCAL_SIZE) -> CropLayout:
    """The tile grid ``infer(crop_mode=True)`` chooses; a page within one tile on both axes is not tiled."""
    width, height = size
    if width <= image_size and height <= image_size:
        return CropLayout(1, 1)
    ratio = find_closest_aspect_ratio(width / height, candidate_crop_ratios(), width, height, image_size)
    return CropLayout(int(ratio[0]), int(ratio[1]))


def dynamic_preprocess(image: Image.Image, image_size: int = LOCAL_SIZE) -> tuple[list[Image.Image], tuple[int, int]]:
    """Stretch the page to the chosen grid and crop it into row-major tiles."""
    orig_width, orig_height = image.size
    target_aspect_ratio = find_closest_aspect_ratio(
        orig_width / orig_height, candidate_crop_ratios(), orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    cols = target_width // image_size

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio


@dataclass(frozen=True)
class GundamViews:
    """One page for ``gundam``: the padded global view plus its stretched tiles."""

    pixels: np.ndarray
    local_pixels: np.ndarray | None
    layout: CropLayout


def preprocess_page_gundam(source: str | Path | Image.Image) -> GundamViews:
    image = load_page(source)
    layout = select_crop_layout(image.size)
    tiles: list[Image.Image] = []
    if layout.tiled:
        tiles, ratio = dynamic_preprocess(image)
        if ratio != layout.crop_ratio:
            raise AssertionError(f"select_crop_layout says {layout.crop_ratio} but dynamic_preprocess says {ratio}")
    global_pixels = normalise_view(pad_to_square(image, BASE_SIZE))[None, ...]
    local_pixels = np.ascontiguousarray(np.stack([normalise_view(tile) for tile in tiles])) if tiles else None
    return GundamViews(pixels=global_pixels, local_pixels=local_pixels, layout=layout)


class UnlimitedOcrBatchProcessor(BatchProcessor[TextAndVisionContext, "UnlimitedOcrInputs"]):
    """One page per request, up to ``--max-batch-size`` requests per scheduler step.

    The prefill graph's ``seq_len`` is static, so ``UnlimitedOCRModel.execute``
    runs a prefill batch as one batch-1 prefill per context: every input is
    kept per context -- ``token_counts`` splits the concatenated ``tokens``,
    and ``pixel_values`` / ``image_token_indices`` hold one buffer per context,
    in batch order.
    """

    def __init__(self, config: ArchConfig, runtime: BatchProcessorRuntime) -> None:
        super().__init__(config, runtime)
        self._devices = list(runtime.devices)

    def get_symbolic_inputs(
        self, *, kv_params: KVCacheParamInterface, device_refs: list[DeviceRef]
    ) -> list[TensorType | BufferType]:
        del kv_params
        from .decoder import COMPUTE_DTYPE

        device = device_refs[0]
        hidden = self.config.decoder.hidden_size
        return [
            TensorType(DType.int64, ["seq_len"], device=device),
            TensorType(COMPUTE_DTYPE, ["num_image_tokens", hidden], device=device),
        ]

    def _pixel_values(self, context_batch: Sequence[TextAndVisionContext]) -> list[Buffer] | None:
        """One ``[n_views, 3, H, W]`` buffer per context on a prefill step, in batch order; ``None`` on a decode step.

        A prefill step is one where the contexts still need vision encoding.
        In-flight batching is off, so a step is all-prefill or all-decode: a
        batch that mixes the two cannot be split into per-context prefills and
        is refused.
        """
        pending = [context.needs_vision_encoding for context in context_batch]
        if not any(pending):
            return None
        if not all(pending):
            raise ValueError(
                f"{pending.count(False)} of {len(pending)} requests in a prefill batch carry no page to encode; "
                "Unlimited-OCR takes exactly one page per request"
            )
        buffers: list[Buffer] = []
        for context in context_batch:
            images = context.next_images
            if len(images) != 1:
                raise ValueError(f"Unlimited-OCR takes exactly one page per request, got {len(images)}")
            pixels = np.asarray(images[0].pixel_values, dtype=np.float32)
            if pixels.ndim == IMAGE_NDIMS - 1:
                pixels = pixels[None, ...]
            elif pixels.ndim != IMAGE_NDIMS:
                raise ValueError(f"expected pixel_values of rank 3 or 4, got shape {pixels.shape}")
            buffers.append(Buffer.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)).to(self._devices[0]))
        return buffers

    def _image_token_indices(self, context_batch: Sequence[TextAndVisionContext]) -> list[Buffer] | None:
        """Each context's own placeholder rows, one buffer per context on a prefill step; else ``None``.

        Indices are into that context's own tokens (each context is prefilled
        on its own), so there is no offset across the batch. ``None`` too when
        no context carries them; a batch where only some do is refused.
        """
        pending = [context for context in context_batch if context.needs_vision_encoding]
        indices = [context.extra_model_args.get("image_token_indices") for context in pending]
        present = [value is not None for value in indices]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(
                f"{present.count(False)} of {len(present)} requests in a prefill batch carry no image_token_indices"
            )
        device = self._devices[0]
        return [Buffer.from_numpy(np.ascontiguousarray(value, dtype=np.int32)).to(device) for value in indices]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnlimitedOcrInputs:
        from .model import UnlimitedOcrInputs

        if len(replica_batches) != 1:
            raise ValueError("Unlimited-OCR does not support data parallelism")
        context_batch = replica_batches[0]
        if not context_batch:
            raise ValueError("an empty batch has nothing to run")
        active = [context.tokens.active for context in context_batch]
        tokens = np.concatenate(active).astype(np.int64, copy=False)
        del return_n_logits
        return UnlimitedOcrInputs(
            tokens=Buffer.from_numpy(tokens).to(self._devices[0]),
            token_counts=tuple(int(row.shape[0]) for row in active),
            pixel_values=self._pixel_values(context_batch),
            image_token_indices=self._image_token_indices(context_batch),
            request_ids=tuple(request_key(context.request_id) for context in context_batch),
            kv_cache_inputs=kv_cache_inputs,
        )

    def process_outputs(self, outputs: Sequence[Buffer | Any]) -> ModelOutputs:
        logits = outputs[0]
        assert isinstance(logits, Buffer)
        return ModelOutputs(next_token_logits=logits, logits=logits)
