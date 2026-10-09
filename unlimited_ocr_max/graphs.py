"""The staged MAX graphs: vision tower(s), image-token layout, prefill and decode.

Every builder expects ``load_state_dict`` to have been called on its module
first (MAX assigns weight FQNs while walking the tree, and doing that with a
graph open raises), and a ``Weight`` binds to the first graph it is added to,
so each graph needs a fresh module instance.

The decode graph (one per row count ``B``, ``B = 1`` included) attends through
the pipeline's KV page pool -- MAX's ragged paged store and attention ops, one
attention op per layer for all ``B`` rows (:class:`~unlimited_ocr_max.decoder.PagedKv`)
-- stages the routed experts as ``ops.custom`` calls into the Mojo qmv kernels
in either weight dtype (``moe_bf16_qmv`` / ``moe_int8_qmv``) and applies the
no-repeat-n-gram guard to its own logits (``ngram_block``,
:func:`~unlimited_ocr_max.ngram.apply_ngram_guard`); an int8 prefill graph stages
``int8_dequant_expert``. Every language graph reads ``lm_head`` -- and the
decode graph every other projection -- through ``dense_bf16_qmv`` (KON-238),
so every one is opened with ``custom_extensions=[MOJO_KERNELS]``. The
builders take any ``DeviceRef``, in either weight dtype: serving refuses a CPU
device before any graph is built (:func:`~unlimited_ocr_max.model.check_serving_device`),
and the model-free tests stage graphs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, Value, ops
from max.nn import Module
from max.nn.kv_cache import PACKED_PAGE_STRIDE, KVCacheInputsPerDevice, MHAKVCacheParams

from .batch_processor import BASE_SIZE, ViewGeometry
from .decoder import COMPUTE_DTYPE, PagedKv, UnlimitedOcrDecoder
from .kv_cache import PAGE_SIZE
from .layers.clip_l import ClipL, ClipLConfig, fuse_clip_sam
from .layers.projector import (
    PROJECTOR_OUTPUT_DIM,
    ImageLayoutEmbeddings,
    MlpProjector,
    assemble_image_tokens,
    image_token_count,
)
from .layers.sam_vit import IN_CHANS, SamViT
from .model_config import DecoderConfig, UnlimitedOCRConfig
from .ngram import MOJO_KERNELS, apply_ngram_guard

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
    "declare_device_resident_weights",
    "paged_kv_from_inputs",
    "paged_kv_input_types",
]


def _guard_input_types(batch: int, window: int, device: DeviceRef) -> list[TensorType]:
    """A decode graph's last two inputs: each row's guard history ``[batch, window]`` and the n-gram size ``[1]``, int32."""
    return [
        TensorType(DType.int32, [batch, window], device=device),
        TensorType(DType.int32, [1], device=device),
    ]


def paged_kv_input_types(config: DecoderConfig, batch: int, device: DeviceRef) -> list[BufferType | TensorType]:
    """The decode graph's page-pool inputs, in the order :meth:`~unlimited_ocr_max.kv_cache.PagedStep.graph_inputs` returns them.

    ``kv_blocks [pages, 2, layers, PAGE_SIZE, n_kv_heads, head_dim]`` fp32 (the
    page count is symbolic: the pool's size is the pipeline's, not the
    graph's), the lookup table ``[B, cols]`` uint32 (``cols`` symbolic), the
    attention's cache lengths ``[B]`` and their max ``[1]``, the store's cache
    lengths ``[B]`` and their max ``[1]``, and MAX's MHA dispatch key ``[4]``
    int64. The two maxima and the dispatch key are host-resident, as MAX's
    kernels read them.
    """
    cpu = DeviceRef.CPU()
    blocks = [2, config.num_hidden_layers, PAGE_SIZE, config.num_key_value_heads, config.head_dim]
    return [
        BufferType(COMPUTE_DTYPE, ["total_num_pages", *blocks], device=device),
        TensorType(DType.uint32, [batch, "lut_cols"], device=device),
        TensorType(DType.uint32, [batch], device=device),
        TensorType(DType.uint32, [1], device=cpu),
        TensorType(DType.uint32, [batch], device=device),
        TensorType(DType.uint32, [1], device=cpu),
        TensorType(DType.int64, [4], device=cpu),
    ]


def paged_kv_from_inputs(values: list[Value], config: DecoderConfig, batch: int, device: DeviceRef) -> PagedKv:
    """:class:`~unlimited_ocr_max.decoder.PagedKv` over the graph values typed by :func:`paged_kv_input_types`.

    What is the same every step is a graph constant, as MAX itself spells the
    packed page stride (``packed_page_stride``): the page stride (``-1``,
    packed pages), the prompt width (1: one query per row) and the row offsets
    ``0 .. B``. The number of KV partitions is 1 on every device, written into
    the dispatch key by :meth:`~unlimited_ocr_max.kv_cache.KvPagePool.step`.
    """
    kv_blocks, lookup, attend_lengths, attend_max, write_index, write_max, dispatch = values
    cpu = DeviceRef.CPU()
    shared = {
        "kv_blocks": kv_blocks.buffer,
        "lookup_table": lookup.tensor,
        "max_prompt_length": ops.constant(np.asarray([1], dtype=np.uint32), DType.uint32, device=cpu),
        "page_stride": ops.constant(np.asarray([PACKED_PAGE_STRIDE], dtype=np.int64), DType.int64, device=cpu),
    }
    return PagedKv(
        params=MHAKVCacheParams(
            dtype=COMPUTE_DTYPE,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            devices=[device],
            n_kv_heads=config.num_key_value_heads,
            page_size=PAGE_SIZE,
        ),
        store=KVCacheInputsPerDevice(cache_lengths=write_index.tensor, max_cache_length=write_max.tensor, **shared),
        attend=KVCacheInputsPerDevice(
            cache_lengths=attend_lengths.tensor,
            max_cache_length=attend_max.tensor,
            attention_dispatch_metadata=dispatch.tensor,
            **shared,
        ),
        row_offsets=ops.constant(np.arange(batch + 1, dtype=np.uint32), DType.uint32, device=device),
    )


def declare_device_resident_weights(graph: Graph, decoder: UnlimitedOcrDecoder) -> int:
    """Declare every language weight as *already resident* on the graph's device.

    Called as the first statement inside a language graph's ``with Graph(...)``
    block, before any op reads a weight. Returns how many it declared.

    **Why it has to be explicit.** A :class:`~max.graph.Weight` used by an op
    adds itself lazily with ``force_initial_weight_on_host=not weight._has_alias``,
    so the only lever the implicit path offers is ``_has_alias`` -- and that
    attribute does not compile on Metal in tractable time (KON-100: >10-21 min
    against 0.9 s). Pre-adding is the same placement without the attribute:
    ``Graph.add_weight`` caches by name and by identity, so every later
    implicit use -- ``matmul`` operands and the int8 stacks consumed whole by
    ``ops.custom`` alike -- returns the value declared here.

    **What it changes in the emitted graph**, on an accelerator: each weight's
    ``mo.constant.external`` is declared on the *device* instead of on the host
    followed by a transfer, so the per-graph ``rmo.mo.transfer`` ops for the
    weights disappear. The declaration set itself is unchanged. The registry
    values ``session.load`` binds must then be device-resident
    (:meth:`~unlimited_ocr_max.pipeline.UnlimitedOcrPipeline._resolved_language_weights`
    builds them), which is what lets every language graph share **one** device
    copy of the weights instead of materialising one each (KON-113/KON-158).
    On CPU the weight's device *is* the host, so this is never called there.
    """
    weights = decoder.raw_state_dict()
    for weight in weights.values():
        graph.add_weight(weight, force_initial_weight_on_host=False)
    return len(weights)

#: The placeholder token the image embeddings are spliced into.
IMAGE_TOKEN_ID = 128815

#: Vision-graph outputs, in order. A multi-view (tile) graph omits ``image_embeds``.
VISION_STAGES = ("sam_out", "clip_out", "fused", "projected", "image_embeds")
ENCODER_STAGES = VISION_STAGES[:-1]


class UnlimitedOcrVisionModel(Module):
    """SAM ViT-B -> CLIP-L (fed SAM's grid) -> fuse -> projector -> image-token rows.

    One instance serves one resolution: CLIP's position tables are resampled
    per resolution in the state dict; SAM's position tables must already match
    the target resolution. FQNs are the checkpoint keys minus ``model.``.
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
    device_resident_weights: bool = False,
) -> LanguageGraph:
    """The prefill graph: ``(token_ids [seq_len], image_embeds [n_image_tokens, hidden])`` at a static ``seq_len``.

    Outputs ``logits`` (last row), ``final_norm``, ``input_embeds``, then every
    layer's post-RoPE ``key_<i>`` / ``value_<i>`` in the cache's sequence-major
    layout, which is what seeds the KV cache.

    ``device_resident_weights`` declares every decoder weight as already
    resident on ``device`` (:func:`declare_device_resident_weights`); the
    caller is :attr:`~unlimited_ocr_max.pipeline.UnlimitedOcrPipeline.shares_language_weights`,
    and the registry it loads with must then hold device buffers.
    Accelerator-only; the parameter itself defaults off.
    """
    hidden = config.decoder.hidden_size
    input_types = [
        TensorType(DType.int64, [seq_len], device=device),
        TensorType(COMPUTE_DTYPE, [n_image_tokens, hidden], device=device),
    ]
    with Graph(
        f"unlimited_ocr_language_tokens_{seq_len}",
        input_types=input_types,
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        if device_resident_weights:
            declare_device_resident_weights(graph, decoder)
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
    #: Requests stepped per execute.
    batch: int = 1


def build_decode_graph(
    config: UnlimitedOCRConfig,
    decoder: UnlimitedOcrDecoder,
    *,
    max_seq_len: int,
    device: DeviceRef,
    batch: int = 1,
    device_resident_weights: bool = False,
) -> DecodeGraph:
    """One decode step for ``batch >= 1`` independent requests in one execute; the only output is ``logits [B, vocab]``.

    Inputs, ``B = batch``:

    * ``tokens [B]`` int64 and ``positions [B]`` int32, row ``b`` for request ``b``;
    * the page pool, :func:`paged_kv_input_types` (``kv_blocks``, lookup table,
      the attention's and the store's cache lengths and maxima, the dispatch
      key) -- what :meth:`~unlimited_ocr_max.kv_cache.KvPagePool.step` builds;
    * last, the n-gram guard's two (:func:`~unlimited_ocr_max.ngram.apply_ngram_guard`):
      ``history [B, window]`` int32, row ``b``'s last ``window`` token ids
      with the fed token last (``window`` is the R-SWA ring,
      ``sliding_window_size``), and the n-gram size ``[1]`` int32, shared by
      every row, 0 with the guard off.

    Per layer the graph stores every row's new k/v into its pages and runs ONE
    attention op for all ``B`` rows (:func:`~unlimited_ocr_max.decoder.paged_decode_attention`):
    no per-row cache inputs, no ring write by selection, no KV outputs. The
    logits come out guarded, each row against its own history, by one
    ``ngram_block`` op. The MoE runs the stack dtype's Mojo qmv kernel on every
    device (:meth:`~unlimited_ocr_max.decoder.MoE.decode_rows`).
    ``max_seq_len`` only sizes the RoPE table. ``device_resident_weights`` is
    :func:`build_language_graph`'s parameter of the same name, same contract.

    The step is not batch-invariant as a whole (only the attention op is), so
    each ``B`` is its own graph and whether two are bitwise equal is measured,
    not assumed.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    dec = config.decoder
    input_types: list[BufferType | TensorType] = [
        TensorType(DType.int64, [batch], device=device),
        TensorType(DType.int32, [batch], device=device),
        *paged_kv_input_types(dec, batch, device),
        *_guard_input_types(batch, dec.sliding_window_size, device),
    ]
    with Graph(
        f"unlimited_ocr_decode_{max_seq_len}_paged_b{batch}",
        input_types=input_types,
        custom_extensions=[MOJO_KERNELS],
    ) as graph:
        if device_resident_weights:
            declare_device_resident_weights(graph, decoder)
        tokens = graph.inputs[0].tensor
        positions = graph.inputs[1].tensor
        kv = paged_kv_from_inputs(list(graph.inputs[2:-2]), dec, batch, device)
        history, ngram_size = (value.tensor for value in graph.inputs[-2:])
        normed = decoder.decode(decoder.embed(tokens), positions=positions, max_seq_len=max_seq_len, kv=kv)
        graph.output(apply_ngram_guard(decoder.logits_rows(normed), history, ngram_size))
    return DecodeGraph(graph=graph, output_names=("logits",), batch=batch)
