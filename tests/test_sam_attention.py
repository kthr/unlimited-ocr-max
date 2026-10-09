"""SAM's windowed attention on the accelerator, against a float64 numpy reference of the same math.

``SamAttention`` computes its logits query-first, the faster GPU spelling. MAX
nightlies from ``26.7.0.dev2026092305`` (through at least ``…100205``) compute
that spelling wrongly on the CPU: a matmul output whose N axis is reshaped and
then added to a broadcast operand comes out 5-30 % off, with no error. That is
one reason CPU serving is unsupported, so this test builds and runs the module
on the GPU, at one SAM ViT-B window (14 x 14 tokens, 768 channels, 12 heads),
and fails loudly if the arithmetic drifts from the reference. It is marked
``slow`` (it compiles a graph) and skips when the machine has no accelerator.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from unlimited_ocr_max.layers.sam_vit import EMBED_DIM, NUM_HEADS, WINDOW_SIZE, SamAttention, rel_pos_index


def _reference(x: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """``Attention.forward`` of SAM's ``image_encoder.py`` with ``use_rel_pos``, in float64."""
    batch, height, width, dim = x.shape
    heads, head_dim, seq = NUM_HEADS, dim // NUM_HEADS, height * width
    qkv = (x.reshape(batch, seq, dim) @ w["qkv.weight"].T + w["qkv.bias"]).reshape(batch, seq, 3, heads, head_dim)
    q, k, v = qkv.transpose(2, 0, 3, 1, 4)
    logits = (q @ k.transpose(0, 1, 3, 2)) * head_dim**-0.5
    r_h = w["rel_pos_h"][rel_pos_index(height, height)]
    r_w = w["rel_pos_w"][rel_pos_index(width, width)]
    r_q = q.reshape(batch, heads, height, width, head_dim)
    rel_h = np.einsum("bnhwc,hkc->bnhwk", r_q, r_h)
    rel_w = np.einsum("bnhwc,wkc->bnhwk", r_q, r_w)
    logits = (
        logits.reshape(batch, heads, height, width, height, width) + rel_h[..., :, None] + rel_w[..., None, :]
    ).reshape(batch, heads, seq, seq)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    out = (p / p.sum(-1, keepdims=True)) @ v
    out = out.reshape(batch, heads, height, width, head_dim).transpose(0, 2, 3, 1, 4).reshape(batch, height, width, dim)
    return out @ w["proj.weight"].T + w["proj.bias"]


@pytest.mark.slow
def test_gpu_windowed_attention_matches_the_reference() -> None:
    # Checked inside the test: collection never queries or opens a device.
    if accelerator_count() == 0:
        pytest.skip("needs an accelerator: SamAttention's query-first spelling is wrong on the nightly CPU")
    rng = np.random.default_rng(0)
    size = (WINDOW_SIZE, WINDOW_SIZE)
    weights = {
        "qkv.weight": 0.05 * rng.standard_normal((3 * EMBED_DIM, EMBED_DIM)),
        "qkv.bias": 0.05 * rng.standard_normal(3 * EMBED_DIM),
        "proj.weight": 0.05 * rng.standard_normal((EMBED_DIM, EMBED_DIM)),
        "proj.bias": 0.05 * rng.standard_normal(EMBED_DIM),
        "rel_pos_h": rng.standard_normal((2 * WINDOW_SIZE - 1, EMBED_DIM // NUM_HEADS)),
        "rel_pos_w": rng.standard_normal((2 * WINDOW_SIZE - 1, EMBED_DIM // NUM_HEADS)),
    }
    weights = {name: value.astype(np.float32) for name, value in weights.items()}
    x = rng.standard_normal((2, *size, EMBED_DIM)).astype(np.float32)

    attention = SamAttention(EMBED_DIM, NUM_HEADS, size, DType.float32, DeviceRef.GPU())
    attention.load_state_dict(weights)
    with Graph("sam_attention", input_types=[TensorType(DType.float32, x.shape, device=DeviceRef.GPU())]) as graph:
        graph.output(attention(graph.inputs[0].tensor))
    device = Accelerator()
    model = InferenceSession(devices=[device]).load(graph, weights_registry=attention.state_dict())
    got = model.execute(Buffer.from_numpy(x).to(device))[0].to_numpy().astype(np.float64)

    want = _reference(x.astype(np.float64), {name: value.astype(np.float64) for name, value in weights.items()})
    rel_l2 = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel_l2 < 1e-5, f"GPU SamAttention is {rel_l2:.3e} (relative L2) off the float64 reference"
