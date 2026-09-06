"""Prompt construction and the ``max.pipelines`` tokenizer.

The prompt is not a chat template. ``infer()`` uses the ``plain`` format (the
user content, stripped), replaces each ``<image>`` marker with a run of
placeholder ids -- ``([id] * 16 + [id]) * 16 + [id]`` = 273 for a 1024px view --
and prepends a literal ``bos_id = 0``; no special tokens are added anywhere.

The tokenizer is built from ``tokenizer.json`` alone: ``AutoTokenizer`` resolves
this checkpoint to a slow ``LlamaTokenizer`` that mis-tokenises and drops every
space on decode. Nine ``"special": true`` tokens are content in this model's
output format (five grounding delimiters, four table-markup tags) and are kept
through MAX's ``skipped_special_token_ids`` hook while the other specials are
stripped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from max.pipelines.lib import TextAndVisionTokenizer

from .batch_processor import BASE_SIZE, LOCAL_SIZE, ViewGeometry, preprocess_page

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig
    from PIL import Image

__all__ = [
    "BOS_ID",
    "DEFAULT_PROMPT",
    "EOS_ID",
    "GROUNDING_TOKENS",
    "IMAGE_TOKEN",
    "IMAGE_TOKEN_ID",
    "TABLE_MARKUP_TOKENS",
    "PromptTokens",
    "UnlimitedOcrProcessor",
    "UnlimitedOcrTokenizer",
    "build_prompt",
    "image_placeholder_ids",
    "load_delegate",
    "skipped_special_token_ids",
]

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_ID = 0
EOS_ID = 1
GROUNDING_TOKENS = ("<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>")
TABLE_MARKUP_TOKENS = ("<td>", "</td>", "<tr>", "</tr>")
DEFAULT_PROMPT = "<image>document parsing."

def image_placeholder_ids(
    num_queries: int, *, local_queries: int | None = None, crop_ratio: tuple[int, int] | None = None
) -> list[int]:
    """The placeholder run for one image: the global grid, plus the stitched local page under ``gundam``."""
    if num_queries < 1:
        raise ValueError(f"num_queries must be >= 1, got {num_queries}")
    row = [IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]
    ids = row * num_queries + [IMAGE_TOKEN_ID]
    if crop_ratio is None:
        if local_queries is not None:
            raise ValueError("local_queries is meaningless without crop_ratio")
        return ids
    width_crop_num, height_crop_num = crop_ratio
    if width_crop_num < 1 or height_crop_num < 1:
        raise ValueError(f"non-positive crop_ratio {crop_ratio}")
    if width_crop_num > 1 or height_crop_num > 1:
        if local_queries is None or local_queries < 1:
            raise ValueError(f"crop_ratio {crop_ratio} is tiled, so local_queries is required")
        local_row = [IMAGE_TOKEN_ID] * (local_queries * width_crop_num) + [IMAGE_TOKEN_ID]
        ids += local_row * (local_queries * height_crop_num)
    return ids


@dataclass(frozen=True)
class PromptTokens:
    ids: np.ndarray
    image_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.ids.shape != self.image_mask.shape:
            raise ValueError("ids and image_mask must have the same shape")

    @property
    def seq_len(self) -> int:
        return int(self.ids.shape[0])

    @property
    def image_token_indices(self) -> np.ndarray:
        return np.flatnonzero(self.image_mask).astype(np.int32)


def build_prompt(
    encode: Callable[[str], Sequence[int]],
    *,
    prompt: str = DEFAULT_PROMPT,
    n_images: int = 1,
    geometry: ViewGeometry | None = None,
    crop_ratio: tuple[int, int] | None = None,
) -> PromptTokens:
    """``input_ids`` / ``images_seq_mask`` the way ``infer()`` builds them.

    ``encode`` must add no special tokens. ``crop_ratio`` is width-major, as the
    reference stores it; ``None`` or ``(1, 1)`` is ``base`` mode.
    """
    geometry = geometry or ViewGeometry(BASE_SIZE)
    splits = prompt.strip().split(IMAGE_TOKEN)
    if len(splits) != n_images + 1:
        raise ValueError(f"prompt has {len(splits) - 1} {IMAGE_TOKEN!r} markers but {n_images} image(s) were supplied")
    ids: list[int] = [BOS_ID]
    mask: list[bool] = [False]
    local_queries = None
    if crop_ratio is not None and (crop_ratio[0] > 1 or crop_ratio[1] > 1):
        local_queries = ViewGeometry(LOCAL_SIZE).grid
    placeholders = image_placeholder_ids(geometry.grid, local_queries=local_queries, crop_ratio=crop_ratio)
    for split in splits[:-1]:
        text = list(encode(split))
        ids += text
        mask += [False] * len(text)
        ids += placeholders
        mask += [True] * len(placeholders)
    tail = list(encode(splits[-1]))
    ids += tail
    mask += [False] * len(tail)
    return PromptTokens(ids=np.asarray(ids, dtype=np.int64), image_mask=np.asarray(mask, dtype=bool))


class UnlimitedOcrProcessor:
    """The ``AutoProcessor`` stand-in: chat request -> prompt string -> ids and preprocessed pixels (``base`` mode)."""

    def __init__(self, delegate: Any, *, base_size: int = BASE_SIZE) -> None:
        self.delegate = delegate
        self.base_size = base_size
        self.geometry = ViewGeometry(base_size)

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, **kwargs: Any) -> str:
        """Join the text parts and emit one ``<image>`` per image part -- images first.

        The model was trained image-first (``<image>...``), and the marker's
        position decides where the 273 image rows land, so the OpenAI
        text-then-image order is deliberately not honoured.
        """
        del tokenize, kwargs
        parts: list[str] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                texts: list[str] = []
                images: list[str] = []
                for item in content:
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif item.get("type") in ("image", "image_url"):
                        images.append(IMAGE_TOKEN)
                parts.extend(images)
                parts.extend(texts)
        joined = "".join(parts)
        return joined if joined else DEFAULT_PROMPT

    def __call__(
        self,
        text: str,
        images: list[Image.Image] | None = None,
        add_special_tokens: bool = False,
        return_tensors: str = "np",
        **kwargs: Any,
    ) -> dict[str, Any]:
        del return_tensors, kwargs, add_special_tokens

        def _encode(chunk: str) -> Sequence[int]:
            return self.delegate.encode(chunk, add_special_tokens=False)

        if not images:
            ids = np.asarray([BOS_ID, *_encode(text)], dtype=np.int64)
            return {"input_ids": [ids.tolist()]}
        if IMAGE_TOKEN not in text:
            text = IMAGE_TOKEN + text
        prompt = build_prompt(_encode, prompt=text, n_images=len(images), geometry=self.geometry)
        pixels = [preprocess_page(image, base_size=self.base_size) for image in images]
        return {
            "input_ids": [prompt.ids.tolist()],
            "pixel_values": [pixels],
            "image_token_indices": prompt.image_token_indices,
        }


def load_delegate(model_path: str | Path, *, max_length: int | None = None) -> Any:
    """``PreTrainedTokenizerFast`` over ``tokenizer.json``, with the special tokens from ``tokenizer_config.json``."""
    path = Path(model_path)
    tokenizer_file = path / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"{tokenizer_file} is required; there is no safe AutoTokenizer fallback")
    from transformers import PreTrainedTokenizerFast

    settings: dict[str, Any] = {}
    config_file = path / "tokenizer_config.json"
    if config_file.is_file():
        raw = json.loads(config_file.read_text())
        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = raw.get(key)
            if isinstance(value, str):
                settings[key] = value
    if max_length is not None:
        settings["model_max_length"] = max_length
    return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **settings)


def special_token_ids(delegate: Any) -> set[int]:
    """Every id flagged ``special`` in ``added_tokens_decoder`` -- what the Rust decoder's skip reads."""
    decoder = getattr(delegate, "added_tokens_decoder", None)
    if not decoder:
        raise ValueError("delegate exposes no added_tokens_decoder; refusing to guess the special set")
    return {int(token_id) for token_id, token in decoder.items() if getattr(token, "special", False)}


def kept_special_token_ids(delegate: Any) -> set[int]:
    """The nine specials this model emits as content, resolved against the tokenizer."""
    unk = getattr(delegate, "unk_token_id", None)
    ids: set[int] = set()
    for token in GROUNDING_TOKENS + TABLE_MARKUP_TOKENS:
        token_id = delegate.convert_tokens_to_ids(token)
        if token_id is None or (unk is not None and token_id == unk):
            raise ValueError(f"{token!r} does not resolve to a token id in this tokenizer")
        ids.add(int(token_id))
    return ids


def skipped_special_token_ids(delegate: Any) -> set[int]:
    """Specials to strip from a served response: all but the nine kept ones; EOS and ``<image>`` must be in it."""
    skipped = special_token_ids(delegate) - kept_special_token_ids(delegate)
    for name, token_id in (("EOS_ID", EOS_ID), ("IMAGE_TOKEN_ID", IMAGE_TOKEN_ID)):
        if token_id not in skipped:
            raise ValueError(f"{name} ({token_id}) is not special in this tokenizer; nothing would strip it")
    return skipped


class UnlimitedOcrTokenizer(TextAndVisionTokenizer):
    """``TextAndVisionTokenizer`` with this model's prompt construction; does not call ``super().__init__``."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        del revision, trust_remote_code
        self.model_path = model_path
        self.delegate = load_delegate(model_path, max_length=max_length)
        self.max_length = max_length or self.delegate.model_max_length

        config = pipeline_config.model.huggingface_config
        resolutions = getattr(config, "candidate_resolutions", None)
        base_size = int(resolutions[0][0]) if resolutions else BASE_SIZE
        self.processor = UnlimitedOcrProcessor(self.delegate, base_size=base_size)

        self.vision_token_ids = [IMAGE_TOKEN_ID]
        self._eos_token_ids = {EOS_ID}
        eos = self.delegate.eos_token_id
        if eos is not None:
            self._eos_token_ids.add(int(eos))
        # Read by getattr in max/serve's incremental detokenizer.
        self.skipped_special_token_ids = skipped_special_token_ids(self.delegate)
        self.enable_prefix_caching = pipeline_config.model.kv_cache.enable_prefix_caching

    async def decode(self, encoded: Any, **kwargs: Any) -> str:
        """The non-streaming path, filtered with the same exclusion set."""
        if isinstance(encoded, int):
            encoded = np.array(encoded)
        if not kwargs.get("skip_special_tokens", True):
            return await super().decode(encoded, **kwargs)
        kept = [t for t in np.asarray(encoded).reshape(-1).tolist() if t not in self.skipped_special_token_ids]
        return await super().decode(np.asarray(kept, dtype=np.int64), **{**kwargs, "skip_special_tokens": False})
