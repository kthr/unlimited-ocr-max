"""The no-repeat-n-gram guard: a one-op graph around the ``ngram_block`` Mojo kernel.

Applied to the prefill logits and to every decode step over the whole sequence,
prompt included, as the reference's ``SlidingWindowNoRepeatNgramProcessor`` is.
Kept out of the decode graph so switching it off changes no decode numerics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

__all__ = ["DEFAULT_NGRAM_SIZE", "MOJO_KERNELS", "NgramBlocker"]

#: The Mojo custom-op package (``Graph(custom_extensions=...)`` wants a directory).
MOJO_KERNELS = Path(__file__).resolve().parent / "kernels"

#: The reference's pairing for this model; the window equals the R-SWA ring. `cli.py` repeats the 35.
DEFAULT_NGRAM_SIZE = 35


class NgramBlocker:
    def __init__(
        self,
        *,
        ngram_size: int,
        window: int,
        vocab_size: int,
        device: DeviceRef,
        driver_device: Device,
        session: InferenceSession,
    ) -> None:
        if ngram_size < 1 or window < 1:
            raise ValueError("ngram_size and window must be >= 1")
        self.ngram_size = ngram_size
        self.window = window
        self.vocab_size = vocab_size
        self.device = device
        self._driver_device = driver_device
        self.session = session
        self._ngram_arg = np.asarray([ngram_size], dtype=np.int32)
        self._model: Model | None = None

    @property
    def model(self) -> Model:
        """Compiled on first use: ``custom_extensions`` triggers a ``mojo precompile`` of the kernel package."""
        if self._model is None:
            with Graph(
                f"ngram_block_{self.ngram_size}",
                input_types=[
                    TensorType(DType.float32, [self.vocab_size], device=self.device),
                    TensorType(DType.int32, ["hist_len"], device=self.device),
                    TensorType(DType.int32, [1], device=self.device),
                ],
                custom_extensions=[MOJO_KERNELS],
            ) as graph:
                graph.output(
                    ops.custom(
                        "ngram_block",
                        device=self.device,
                        values=[graph.inputs[i].tensor for i in range(3)],
                        out_types=[TensorType(DType.float32, [self.vocab_size], device=self.device)],
                    )[0]
                )
            self._model = self.session.load(graph)
        return self._model

    def apply(self, logits: np.ndarray, sequence: Sequence[int]) -> np.ndarray:
        """``logits`` with every banned id set to the kernel's ``BLOCKED``; unchanged when the history is too short."""
        ids = np.asarray(sequence, dtype=np.int32).reshape(-1)
        history = np.ascontiguousarray(ids[-self.window :])
        if history.shape[0] < self.ngram_size:
            return logits
        flat = np.ascontiguousarray(np.asarray(logits, dtype=np.float32).reshape(-1))
        if flat.shape[0] != self.vocab_size:
            raise ValueError(f"logits has {flat.shape[0]} entries, expected {self.vocab_size}")
        buffers = [Buffer.from_numpy(array).to(self._driver_device) for array in (flat, history, self._ngram_arg)]
        result = self.model.execute(*buffers)[0]
        return result.to(CPU()).to_numpy().reshape(-1)
