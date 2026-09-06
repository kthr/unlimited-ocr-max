"""The staged MAX graphs: vision tower(s), image-token layout, prefill and decode.

Every builder expects ``load_state_dict`` to have been called on its module
first (MAX assigns weight FQNs while walking the tree, and doing that with a
graph open raises), and a ``Weight`` binds to the first graph it is added to,
so each graph needs a fresh module instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Module

from .batch_processor import BASE_SIZE, ViewGeometry
from .decoder import COMPUTE_DTYPE, UnlimitedOcrDecoder
from .layers.clip_l import ClipL, ClipLConfig, fuse_clip_sam
from .layers.projector import (
    PROJECTOR_OUTPUT_DIM,
    ImageLayoutEmbeddings,
    MlpProjector,
    assemble_image_tokens,
    image_token_count,
)
from .layers.sam_vit import IN_CHANS, SamViT
from .model_config import UnlimitedOCRConfig

__all__ = [
    "IMAGE_TOKEN_ID",
    "DecodeGraph",
    "ImageTokenLayout",
    "LanguageGraph",
    "LayoutGraph",
    "UnlimitedOcrVisionModel",
    "VisionGraph",
    "build_decode_graph",
    "build_language_graph",
    "build_layout_graph",
    "build_vision_graph",
]

#: The placeholder token the image embeddings are spliced into.
IMAGE_TOKEN_ID = 128815

#: Vision-graph outputs, in order. A multi-view (tile) graph omits ``image_embeds``.
VISION_STAGES = ("sam_out", "clip_out", "fused", "projected", "image_embeds")
ENCODER_STAGES = VISION_STAGES[:-1]


class UnlimitedOcrVisionModel(Module):
    """SAM ViT-B -> CLIP-L (fed SAM's grid) -> fuse -> projector -> image-token rows.

    One instance serves one resolution: SAM's position tables and CLIP's are
    resampled per resolution in the state dict. FQNs are the checkpoint keys
    minus ``model.``.
    """

    def __init__(self, *, image_size: int = BASE_SIZE, dtype: DType = COMPUTE_DTYPE, device: DeviceRef | None = None) -> None:
        super().__init__()
        device = device or DeviceRef.CPU()
        self.geometry = ViewGeometry(image_size)
        self.image_size = image_size
        self.dtype = dtype
        self.device = device
        grid = self.geometry.grid
        self.sam_model = SamViT(image_size, dtype, device)
        self.vision_model = ClipL(ClipLConfig(), grid_h=grid, grid_w=grid, dtype=dtype, device=device)
        self.projector = MlpProjector(dtype=dtype, device=device)
        self.layout = ImageLayoutEmbeddings(dtype=dtype, device=device)

    @property
    def token_grid(self) -> tuple[int, int]:
        return self.geometry.token_grid

    def input_type(self, n_views: int = 1) -> TensorType:
        return TensorType(self.dtype, [n_views, IN_CHANS, self.image_size, self.image_size], device=self.device)

    def __call__(self, pixels: TensorValue) -> dict[str, TensorValue]:
        """``[views, 3, H, W]`` -> the stages in :data:`VISION_STAGES`; ``image_embeds`` only for one view."""
        n_views = int(pixels.shape[0])
        if n_views < 1:
            raise ValueError(f"need at least one view, got {n_views}")
        sam_out = self.sam_model(pixels)
        clip_out = self.vision_model(sam_out)
        fused = fuse_clip_sam(clip_out, sam_out)
        projected = self.projector(fused)
        stages = {"sam_out": sam_out, "clip_out": clip_out, "fused": fused, "projected": projected}
        if n_views == 1:
            image_newline, view_seperator = self.layout()
            stages["image_embeds"] = assemble_image_tokens(
                global_features=projected,
                global_grid=self.token_grid,
                image_newline=image_newline,
                view_seperator=view_seperator,
            )
        return stages


@dataclass(frozen=True)
class VisionGraph:
    graph: Graph
    output_names: tuple[str, ...]


def build_vision_graph(model: UnlimitedOcrVisionModel, *, n_views: int = 1) -> VisionGraph:
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    names = VISION_STAGES if n_views == 1 else ENCODER_STAGES
    with Graph(f"unlimited_ocr_vision_{model.image_size}_x{n_views}", input_types=[model.input_type(n_views)]) as graph:
        stages = model(graph.inputs[0].tensor)
        graph.output(*(stages[key] for key in names))
    return VisionGraph(graph=graph, output_names=names)


class ImageTokenLayout(Module):
    """The row layout on its own, for gundam: the two views come from two different towers."""

    def __init__(
        self,
        *,
        global_grid: tuple[int, int],
        local_grid: tuple[int, int] | None = None,
        crop_grid: tuple[int, int] | None = None,
        n_embed: int = PROJECTOR_OUTPUT_DIM,
        dtype: DType = COMPUTE_DTYPE,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        if (local_grid is None) != (crop_grid is None):
            raise ValueError("local_grid and crop_grid must be given together or not at all")
        device = device or DeviceRef.CPU()
        self.global_grid = global_grid
        self.local_grid = local_grid
        self.crop_grid = crop_grid
        self.n_embed = n_embed
        self.dtype = dtype
        self.device = device
        self.layout = ImageLayoutEmbeddings(n_embed, dtype=dtype, device=device)

    @property
    def n_tiles(self) -> int:
        return 0 if self.crop_grid is None else self.crop_grid[0] * self.crop_grid[1]

    @property
    def n_image_tokens(self) -> int:
        return image_token_count(global_grid=self.global_grid, local_grid=self.local_grid, crop_grid=self.crop_grid)

    def input_types(self) -> list[TensorType]:
        gh, gw = self.global_grid
        types = [TensorType(self.dtype, [1, gh * gw, self.n_embed], device=self.device)]
        if self.local_grid is not None:
            th, tw = self.local_grid
            types.append(TensorType(self.dtype, [self.n_tiles, th * tw, self.n_embed], device=self.device))
        return types

    def __call__(self, projected_global: TensorValue, projected_local: TensorValue | None = None) -> TensorValue:
        if (projected_local is None) != (self.local_grid is None):
            raise ValueError("local features must be given exactly when the layout is tiled")
        image_newline, view_seperator = self.layout()
        return assemble_image_tokens(
            global_features=projected_global,
            global_grid=self.global_grid,
            image_newline=image_newline,
            view_seperator=view_seperator,
            local_features=projected_local,
            local_grid=self.local_grid,
            crop_grid=self.crop_grid,
        )


@dataclass(frozen=True)
class LayoutGraph:
    graph: Graph
    output_names: tuple[str, ...]
    n_image_tokens: int


def build_layout_graph(model: ImageTokenLayout) -> LayoutGraph:
    with Graph(f"unlimited_ocr_layout_{model.n_image_tokens}", input_types=model.input_types()) as graph:
        graph.output(model(*(inp.tensor for inp in graph.inputs)))
    return LayoutGraph(graph=graph, output_names=("image_embeds",), n_image_tokens=model.n_image_tokens)


def splice_image_embeddings(text_embeds: TensorValue, token_ids: TensorValue, image_embeds: TensorValue) -> TensorValue:
    """The reference's ``masked_scatter_``: image rows consumed in order at the placeholder positions."""
    hidden = text_embeds.shape[-1]
    marker = ops.constant(IMAGE_TOKEN_ID, token_ids.dtype, device=token_ids.device)
    mask = ops.broadcast_to(
        ops.equal(token_ids, marker).reshape((token_ids.shape[0], 1)), (token_ids.shape[0], hidden)
    )
    return ops.masked_scatter(
        text_embeds, mask, ops.cast(image_embeds, text_embeds.dtype), out_dim="image_embedding_elements"
    )


@dataclass(frozen=True)
class LanguageGraph:
    graph: Graph
    output_names: tuple[str, ...]


def build_language_graph(
    config: UnlimitedOCRConfig,
    decoder: UnlimitedOcrDecoder,
    *,
    seq_len: int,
    n_image_tokens: int,
    device: DeviceRef,
) -> LanguageGraph:
    """The prefill graph: ``(token_ids [seq_len], image_embeds [n_image_tokens, hidden])`` at a static ``seq_len``.

    Outputs ``logits`` (last row), ``final_norm``, ``input_embeds``, then every
    layer's post-RoPE ``key_<i>`` / ``value_<i>`` in the cache's sequence-major
    layout, which is what seeds the KV cache.
    """
    hidden = config.decoder.hidden_size
    input_types = [
        TensorType(DType.int64, [seq_len], device=device),
        TensorType(COMPUTE_DTYPE, [n_image_tokens, hidden], device=device),
    ]
    with Graph(f"unlimited_ocr_language_tokens_{seq_len}", input_types=input_types) as graph:
        token_ids = graph.inputs[0].tensor
        image_embeds = graph.inputs[1].tensor
        input_embeds = splice_image_embeddings(decoder.embed(token_ids), token_ids, image_embeds)
        normed, kv = decoder(input_embeds, seq_len=seq_len)
        outputs: list[TensorValue] = [decoder.logits(normed), normed, input_embeds]
        names: list[str] = ["logits", "final_norm", "input_embeds"]
        for i, (key, value) in enumerate(kv):
            outputs += [key.permute([1, 0, 2]), value.permute([1, 0, 2])]
            names += [f"key_{i}", f"value_{i}"]
        graph.output(*outputs)
    return LanguageGraph(graph=graph, output_names=tuple(names))


@dataclass(frozen=True)
class DecodeGraph:
    graph: Graph
    output_names: tuple[str, ...]
    num_layers: int


def build_decode_graph(
    config: UnlimitedOCRConfig,
    decoder: UnlimitedOcrDecoder,
    *,
    max_seq_len: int,
    device: DeviceRef,
) -> DecodeGraph:
    """The one-token step: ``(token_id, position, write_sel, key_cache_0..n, value_cache_0..n) -> (logits, key_*, value_*)``.

    The cache dimension ``past_len`` is symbolic; ``max_seq_len`` only sizes the
    RoPE table. The MoE dispatch at ``seq == 1`` follows the device (see
    :class:`~unlimited_ocr_max.decoder.MoE`).
    """
    dec = config.decoder
    num_layers = dec.num_hidden_layers
    cache_type = TensorType(COMPUTE_DTYPE, ["past_len", dec.num_key_value_heads, dec.head_dim], device=device)
    input_types: list[TensorType] = [
        TensorType(DType.int64, [1], device=device),
        TensorType(DType.int32, [1], device=device),
        TensorType(DType.bool, [1, "past_len", 1], device=device),
        *([cache_type] * (2 * num_layers)),
    ]
    with Graph(f"unlimited_ocr_decode_{max_seq_len}_ring", input_types=input_types) as graph:
        token_id = graph.inputs[0].tensor
        position = graph.inputs[1].tensor
        write_sel = graph.inputs[2].tensor
        caches = [value.tensor for value in graph.inputs[3:]]
        past_kv = [(caches[i], caches[num_layers + i]) for i in range(num_layers)]
        normed, new_kv = decoder.decode(
            decoder.embed(token_id), position=position, max_seq_len=max_seq_len, past_kv=past_kv, write_sel=write_sel
        )
        outputs: list[TensorValue] = [decoder.logits(normed)]
        names: list[str] = ["logits"]
        for i, (key, _) in enumerate(new_kv):
            outputs.append(key)
            names.append(f"key_{i}")
        for i, (_, value) in enumerate(new_kv):
            outputs.append(value)
            names.append(f"value_{i}")
        graph.output(*outputs)
    return DecodeGraph(graph=graph, output_names=tuple(names), num_layers=num_layers)
