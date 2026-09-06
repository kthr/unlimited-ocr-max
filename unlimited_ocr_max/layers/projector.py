"""The vision projector and the image-token row layout.

The layout follows the reference's ``masked_scatter_`` order --
``cat([local_features, global_features, view_seperator])``, local views first --
not the order in which ``infer()`` emits placeholder tokens. Each token row is
closed by ``image_newline``; one ``view_seperator`` ends the sequence.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Linear, Module

__all__ = [
    "PROJECTOR_INPUT_DIM",
    "PROJECTOR_OUTPUT_DIM",
    "ImageLayoutEmbeddings",
    "MlpProjector",
    "assemble_image_tokens",
    "image_token_count",
]

PROJECTOR_INPUT_DIM = 2048
PROJECTOR_OUTPUT_DIM = 1280


class MlpProjector(Module):
    """``projector_type == "linear"``: one ``nn.Linear`` held as ``layers``."""

    def __init__(
        self,
        input_dim: int = PROJECTOR_INPUT_DIM,
        n_embed: int = PROJECTOR_OUTPUT_DIM,
        *,
        dtype: DType = DType.float32,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        device = device or DeviceRef.CPU()
        self.layers = Linear(input_dim, n_embed, dtype=dtype, device=device, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.layers(x)


class ImageLayoutEmbeddings(Module):
    """``image_newline`` / ``view_seperator``, declared at the parent's level.

    ``_omit_module_attr_name`` drops this module's attribute name from its
    children's FQNs, matching the checkpoint's top-level ``model.image_newline``
    and ``model.view_seperator``. It therefore needs an enclosing module.
    """

    @property
    def _omit_module_attr_name(self) -> bool:
        return True

    def __init__(
        self, n_embed: int = PROJECTOR_OUTPUT_DIM, *, dtype: DType = DType.float32, device: DeviceRef | None = None
    ) -> None:
        super().__init__()
        device = device or DeviceRef.CPU()
        self.image_newline = Weight("image_newline", dtype, (n_embed,), device=device)
        self.view_seperator = Weight("view_seperator", dtype, (n_embed,), device=device)

    def __call__(self) -> tuple[TensorValue, TensorValue]:
        return TensorValue(self.image_newline), TensorValue(self.view_seperator)


def append_row_newlines(
    features: TensorValue, *, rows: int, cols: int, image_newline: TensorValue
) -> TensorValue:
    """``[rows * cols, n]`` (or ``[1, rows * cols, n]``) -> ``[rows * (cols + 1), n]``."""
    grid = _as_token_matrix(features, rows * cols)
    n_embed = int(grid.shape[-1])
    grid = grid.reshape((rows, cols, n_embed))
    newline = ops.broadcast_to(image_newline.reshape((1, 1, n_embed)), (rows, 1, n_embed))
    return ops.concat([grid, newline], axis=1).reshape((rows * (cols + 1), n_embed))


def untile_local_features(
    features: TensorValue, *, crop_rows: int, crop_cols: int, tile_rows: int, tile_cols: int
) -> TensorValue:
    """Stitch ``[crop_rows * crop_cols, tile_rows * tile_cols, n]`` tiles into one page grid."""
    if features.rank != 3:
        raise ValueError(f"features must be rank 3, got {features.rank}")
    n_tiles, tokens, n_embed = (int(d) for d in features.shape)
    if n_tiles != crop_rows * crop_cols:
        raise ValueError(f"features has {n_tiles} tiles, expected {crop_rows * crop_cols}")
    if tokens != tile_rows * tile_cols:
        raise ValueError(f"features has {tokens} tokens per tile, expected {tile_rows * tile_cols}")
    return (
        features.reshape((crop_rows, crop_cols, tile_rows, tile_cols, n_embed))
        .permute([0, 2, 1, 3, 4])
        .reshape((crop_rows * tile_rows * crop_cols * tile_cols, n_embed))
    )


def assemble_image_tokens(
    *,
    global_features: TensorValue,
    global_grid: tuple[int, int],
    image_newline: TensorValue,
    view_seperator: TensorValue,
    local_features: TensorValue | None = None,
    local_grid: tuple[int, int] | None = None,
    crop_grid: tuple[int, int] | None = None,
) -> TensorValue:
    """Local tiles (if any), then the global view, each row closed by ``image_newline``, then the separator."""
    gh, gw = global_grid
    n_embed = int(global_features.shape[-1])
    parts: list[TensorValue] = []
    if local_features is not None:
        if local_grid is None or crop_grid is None:
            raise ValueError("local_features requires both local_grid and crop_grid")
        th, tw = local_grid
        crop_rows, crop_cols = crop_grid
        stitched = untile_local_features(
            local_features, crop_rows=crop_rows, crop_cols=crop_cols, tile_rows=th, tile_cols=tw
        )
        parts.append(
            append_row_newlines(stitched, rows=crop_rows * th, cols=crop_cols * tw, image_newline=image_newline)
        )
    elif local_grid is not None or crop_grid is not None:
        raise ValueError("local_grid / crop_grid are meaningless without local_features")
    parts.append(append_row_newlines(global_features, rows=gh, cols=gw, image_newline=image_newline))
    parts.append(view_seperator.reshape((1, n_embed)))
    return ops.concat(parts, axis=0)


def image_token_count(
    *,
    global_grid: tuple[int, int],
    local_grid: tuple[int, int] | None = None,
    crop_grid: tuple[int, int] | None = None,
) -> int:
    """Length of :func:`assemble_image_tokens`' output: 273 for a 1024px base page."""
    gh, gw = global_grid
    total = gh * (gw + 1) + 1
    if local_grid is not None:
        if crop_grid is None:
            raise ValueError("local_grid requires crop_grid")
        th, tw = local_grid
        crop_rows, crop_cols = crop_grid
        total += (crop_rows * th) * (crop_cols * tw + 1)
    elif crop_grid is not None:
        raise ValueError("crop_grid requires local_grid")
    return total


def _as_token_matrix(features: TensorValue, tokens: int) -> TensorValue:
    if features.rank == 3:
        batch = int(features.shape[0])
        if batch != 1:
            raise ValueError(f"a batched view must have batch 1, got {batch}")
        features = features.reshape((int(features.shape[1]), int(features.shape[2])))
    if features.rank != 2:
        raise ValueError(f"features must be rank 2 or 3, got {features.rank}")
    if int(features.shape[0]) != tokens:
        raise ValueError(f"features has {int(features.shape[0])} tokens, expected {tokens}")
    return features
