"""The per-request KV cache: R-SWA ring semantics, on the host or on the accelerator.

Rows ``0 .. prefill_len - 1`` are the pinned reference region; the next
``window`` rows are a ring the generated tokens overwrite in place, so capacity
is ``prefill_len + window`` however long the generation runs. The reference
appends while ``length < prefill_len + window`` and overwrites from then on, and
RoPE uses the true :attr:`position`, which keeps counting after the cache stops
growing.

Layout is sequence-major ``[capacity, n_kv_heads, head_dim]`` fp32, so a
leading-dimension prefix is contiguous and can be handed to the decode graph
without a copy. :class:`DeviceKvCache` keeps the rows in ``max.driver`` buffers
across steps and writes the graph's KV outputs back device-to-device, so only
the logits leave the device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from max.driver import CPU, Buffer, Device, batch_inplace_copy
from max.dtype import DType

__all__ = ["DeviceKvCache", "KvCache", "allocate_kv_cache"]


@dataclass
class KvCache:
    """Host numpy rows. Every slot decision and state transition lives here."""

    #: Which decode-graph outputs must come back to the host; ``None`` = all.
    HOST_OUTPUTS: ClassVar[tuple[str, ...] | None] = None

    keys: list[np.ndarray]
    values: list[np.ndarray]
    window: int
    length: int = 0
    prefill_len: int = 0
    ring_pos: int = 0
    position: int = 0

    @property
    def max_seq_len(self) -> int:
        return int(self.keys[0].shape[0])

    @property
    def write_index(self) -> int:
        """``length`` while appending, the ring slot afterwards."""
        if self.length < self.prefill_len + self.window:
            return self.length
        return self.prefill_len + self.ring_pos

    @property
    def attend_len(self) -> int:
        """Rows shown to the graph: one past the write index while warming up, else the whole ring."""
        return max(self.length, self.write_index + 1)

    def seed(self, keys: Sequence[Any], values: Sequence[Any]) -> None:
        """Install the prefill's ``[seq_len, n_kv_heads, head_dim]`` rows and pin them."""
        length = int(keys[0].shape[0])
        if length + self.window > self.max_seq_len:
            raise ValueError(
                f"a {self.window}-slot ring over a {length}-token prefix needs "
                f"{length + self.window} rows, but only {self.max_seq_len} are allocated"
            )
        self._write_prefix(keys, values, length)
        self.length = length
        self.prefill_len = length
        self.position = length
        self.ring_pos = 0

    def _write_prefix(self, keys: Sequence[Any], values: Sequence[Any], length: int) -> None:
        for i, (key, value) in enumerate(zip(keys, values, strict=True)):
            self.keys[i][:length] = key
            self.values[i][:length] = value

    def _write_row(self, keys: Sequence[Any], values: Sequence[Any], row: int) -> None:
        for i, (key, value) in enumerate(zip(keys, values, strict=True)):
            self.keys[i][row] = key.reshape(key.shape[-2:])
            self.values[i][row] = value.reshape(value.shape[-2:])

    def selector(self) -> np.ndarray:
        """The graph's ``write_sel``: bool ``[1, attend_len, 1]`` one-hot at :attr:`write_index`."""
        sel = np.zeros((1, self.attend_len, 1), dtype=bool)
        sel[0, self.write_index] = True
        return sel

    def append(self, keys: Sequence[Any], values: Sequence[Any]) -> None:
        """Write one decoded token's rows: append during warm-up, overwrite the ring slot afterwards."""
        row = self.write_index
        appending = row == self.length
        self._write_row(keys, values, row)
        if appending:
            self.length += 1
        else:
            self.ring_pos = (self.ring_pos + 1) % self.window
        self.position += 1

    def views(self) -> list[Any]:
        """Graph-input order: every key prefix, then every value prefix, sliced to :attr:`attend_len`."""
        n = self.attend_len
        return [key[:n] for key in self.keys] + [value[:n] for value in self.values]


@dataclass
class DeviceKvCache(KvCache):
    """Rows resident in accelerator buffers; ``keys``/``values`` stay empty."""

    HOST_OUTPUTS: ClassVar[tuple[str, ...] | None] = ("logits",)

    device_keys: list[Buffer] = field(default_factory=list)
    device_values: list[Buffer] = field(default_factory=list)
    capacity: int = 0

    @property
    def max_seq_len(self) -> int:
        return self.capacity

    def views(self) -> list[Buffer]:
        n = self.attend_len
        return [key[:n, :, :] for key in self.device_keys] + [value[:n, :, :] for value in self.device_values]

    def _write_prefix(self, keys: Sequence[Any], values: Sequence[Any], length: int) -> None:
        # `batch_inplace_copy` is asynchronous and `Buffer.from_numpy` does not
        # keep its source alive: `staged` must outlive the copy, and the
        # synchronize is what guarantees it has completed.
        dsts: list[Buffer] = []
        srcs: list[Buffer] = []
        staged: list[np.ndarray] = []
        for buf, array in zip((*self.device_keys, *self.device_values), (*keys, *values), strict=True):
            dsts.append(buf[:length, :, :])
            rows = np.ascontiguousarray(np.asarray(array), dtype=np.float32)
            staged.append(rows)
            srcs.append(Buffer.from_numpy(rows))
        batch_inplace_copy(dsts, srcs)
        if self.device_keys:
            self.device_keys[0].device.synchronize()

    def _write_row(self, keys: Sequence[Any], values: Sequence[Any], row: int) -> None:
        # Sources are the graph's own device outputs, ordered behind the
        # submission that produced them; no drain needed.
        batch_inplace_copy(
            [buf[row : row + 1, :, :] for buf in self.device_keys]
            + [buf[row : row + 1, :, :] for buf in self.device_values],
            [*keys, *values],
        )


def allocate_kv_cache(
    *,
    num_layers: int,
    prefill_len: int,
    window: int,
    num_kv_heads: int,
    head_dim: int,
    device: Device | None,
) -> KvCache:
    """Zeroed ``prefill_len + window`` rows per layer: numpy when ``device`` is ``None``, else device buffers."""
    shape = (prefill_len + window, num_kv_heads, head_dim)
    if device is None:
        return KvCache(
            keys=[np.zeros(shape, dtype=np.float32) for _ in range(num_layers)],
            values=[np.zeros(shape, dtype=np.float32) for _ in range(num_layers)],
            window=window,
        )
    return DeviceKvCache(
        keys=[],
        values=[],
        window=window,
        device_keys=[Buffer.zeros(shape, DType.float32, device=device) for _ in range(num_layers)],
        device_values=[Buffer.zeros(shape, DType.float32, device=device) for _ in range(num_layers)],
        capacity=shape[0],
    )


def to_host(buffer: Any) -> np.ndarray:
    """A graph output as numpy, wherever it lives."""
    return buffer.to(CPU()).to_numpy()
