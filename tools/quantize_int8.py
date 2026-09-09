#!/usr/bin/env python3
"""Requant the routed experts of the Unlimited-OCR checkpoint to int8, in place
of nothing else -- this is exactly how the published `model-int8.safetensors`
was produced from `model.safetensors` (see the model card's Files section). Every other tensor is copied **byte for byte**.

    # the tool's own proof, no checkpoint touched, ~5 s
    tools/quantize_int8.py --self-check

    # the real thing (2112 expert tensors of 2710), sha pinned; the source is
    # the hf-unlimited-ocr-max checkout's model.safetensors (the HF cache's
    # sharded model-00001-of-000001.safetensors is the same bytes)
    shasum -a 256 ../hf-unlimited-ocr-max/model.safetensors
    tools/quantize_int8.py ../hf-unlimited-ocr-max/model.safetensors model-int8.safetensors \
        --expect-sha 2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6

Runs anywhere numpy + a POSIX stdlib exist (the peak-RSS print uses
resource, so Windows is out of scope).

Why this exists
---------------
The routed experts are the checkpoint: 11 MoE layers x 64 experts x 3
projections at ``[896,1280]`` / ``[1280,896]`` bf16 is 2112 tensors and ~4.8 GB
of the 6.7 GB file, and they are the only weights a decode step reads through
``gather`` rather than as a resident constant. Halving them is the one weight
lever this port has that does not touch the graph's numerics anywhere else --
which is exactly why *everything else must not move*: a passthrough that
re-encodes bf16 (decode to fp32, re-round, write) would silently perturb
weights this task has no business perturbing, so the passthrough here is
``read bytes -> write bytes`` and never a numeric round trip.

The format contract, stated exactly
-----------------------------------
For ``L`` in ``1..11``, ``E`` in ``0..63``, ``proj`` in
``{gate_proj, up_proj, down_proj}``::

    model.layers.{L}.mlp.experts.{E}.{proj}.weight          bf16 [N,K]
      ->  model.layers.{L}.mlp.experts.{E}.{proj}.weight         int8 [N,K]
      +   model.layers.{L}.mlp.experts.{E}.{proj}.weight_scales  fp32 [N,K/128]

``gate_proj`` / ``up_proj`` are ``N=896, K=1280`` (scales ``[896,10]``);
``down_proj`` is ``N=1280, K=896`` (scales ``[1280,7]``). Selection is by
**name** (:data:`EXPERT_WEIGHT_RE`), never by shape -- a shape-matching rule
would also catch whatever else happens to be ``[896,1280]``. The shapes are
then only *asserted*: ``K % 128 != 0`` is a refusal, not a fallback, because a
partial trailing group would put a second convention in the format.

``__metadata__`` of the output carries ``source_sha256``, ``group_size``
(``"128"``), ``scheme`` (``"symmetric-int8-per-group-k"``) and ``tool``
(:data:`TOOL`). The source's own metadata keys are carried through first, so
the ``{"format": "pt"}`` this checkpoint ships keeps its meaning (a missing
``format`` is a hard error in ``transformers``' safetensors loader) and ours
win on collision.

The quant math, stated exactly
------------------------------
Groups run along ``K`` -- the **input** dim, the one a matmul reduces over, so
a group's scale can be folded into the accumulation -- ``G = 128`` of them at a
time, in fp32 throughout::

    amax  = max(|w_group|)                      # fp32, from the bf16 values
    scale = amax / 127                          # fp32; amax == 0 -> scale 1.0
    q     = clip(rint(w / scale), -127, 127)    # int8; -128 unused

``-128`` is left unused so ``-q`` is representable and the scheme stays
symmetric. ``rint`` is round-half-to-**even**, which is what makes the result a
function of the input bytes alone; an all-zero group takes ``scale = 1.0`` and
``q = 0`` so a dequantised zero stays exactly zero rather than depending on a
sentinel. Nothing here is data-dependent beyond the group's own amax: same
input bytes -> same output bytes, byte-identical sha256 across runs, which
:func:`self_check` proves by quantising the same synthetic file twice.

Why numpy + stdlib only
-----------------------
This ships publicly with the port, so it must run on a bare
``pip install numpy``: no torch, no MAX, and not even the ``safetensors``
library. Parsing safetensors is 8 bytes of little-endian header length, a JSON
header (``name -> {dtype, shape, data_offsets}``) and a byte buffer -- doing it
by hand is what makes both hard requirements cheap: byte-identical passthrough
(the bytes are never decoded) and streaming (one tensor resident at a time,
passthrough copied in 2 MiB chunks). Peak RSS is therefore a function of the
largest *quantised* tensor, not of the file: ~53 MiB measured, of which 25 MiB
is the bare interpreter plus numpy, against a 6.7 GB input. Phase 5 of
``--self-check`` measures it in a child process rather than asserting it, and
the largest tensor it quantises is already the real ``[1280,896]``, so the
number is the one the real run will show.

bf16 has no numpy dtype, so the read is: view the bytes as ``uint16``, widen to
``uint32``, ``<< 16``, view as ``float32``. Exact, and it is the only decode in
the tool.

``--self-check``
    Five phases, no checkpoint, no network, nothing outside a temp dir:

    1. a golden micro-tensor whose ``q``/``scales`` are hand-derived (including
       an amax edge at exactly ``+-127*scale``, both round-half-to-even ties,
       an all-zero group, and a group whose scale is *not* a power of two);
    2. a synthetic mini-checkpoint -- tiny shapes with ``K % 128 == 0``, a
       couple of non-expert bf16/fp32 tensors, and three near-miss names that
       must **not** be selected -- written, quantised, then verified against an
       independent scalar-loop reference quantiser, with the non-expert tensors
       required byte-identical and the metadata required present;
    3. determinism: the same mini-checkpoint quantised twice, sha256 identical;
    4. refusals: wrong ``--expect-sha``, ``K`` not divisible by 128, an expert
       weight that is not bf16, a stacked 3-D expert tensor, and a source that
       already carries scales;
    5. streaming: ~78 MiB of synthetic checkpoint through the **CLI**, with the
       child's peak RSS read off ``getrusage(RUSAGE_CHILDREN)`` and compared
       both to twice the input size and to a bare ``import numpy`` child.

    The reference quantiser in phase 2 is a deliberate twin: scalar Python
    loops over rows and groups, so a vectorisation bug in
    :func:`quantize_per_group` cannot hide in both.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import resource
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL = "quantize_int8.py 1.0"
SCHEME = "symmetric-int8-per-group-k"
GROUP_SIZE = 128
QMAX = 127
SCALES_SUFFIX = "_scales"

# The one selection rule. Anchored at both ends, so `...gate_proj.bias`,
# `...shared_experts.gate_proj.weight` and the stacked `...experts.gate_proj`
# (no expert index) all fall through to the byte-copy path.
EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

# 2 MiB: the passthrough path's whole memory cost is a couple of live copies of
# one chunk, and at 6.7 GB the read count is irrelevant next to the bandwidth.
COPY_CHUNK = 2 << 20
MAX_HEADER_BYTES = 100_000_000  # what the reference implementation accepts

# Only what a safetensors header can legally say. Unknown dtypes are a refusal
# rather than a guess, because the byte length check below is the only thing
# standing between a mis-sized copy and a silently corrupt output.
DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class QuantizeError(RuntimeError):
    """A refusal with a message, not a traceback: see :func:`main`."""


# --------------------------------------------------------------------------- #
# safetensors, by hand
# --------------------------------------------------------------------------- #
def read_header(fh) -> tuple[dict, dict, int]:
    """``(metadata, {name: entry}, data_start)`` from an open binary file.

    Validated on the way out: every entry has a known dtype, a shape whose
    product times the item size equals its byte span, and offsets that are
    ascending, gap-free and start at 0 -- the same invariants the reference
    implementation checks, and the reason a truncated output cannot pass for a
    complete one.
    """
    raw = fh.read(8)
    if len(raw) != 8:
        raise QuantizeError("not a safetensors file: shorter than 8 bytes")
    (header_len,) = struct.unpack("<Q", raw)
    if not 0 < header_len <= MAX_HEADER_BYTES:
        raise QuantizeError(f"implausible safetensors header length {header_len}")
    blob = fh.read(header_len)
    if len(blob) != header_len:
        raise QuantizeError(
            f"truncated safetensors header: wanted {header_len} bytes, got {len(blob)}"
        )
    try:
        header = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise QuantizeError(f"safetensors header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise QuantizeError("safetensors header is not a JSON object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise QuantizeError("__metadata__ is not a JSON object")

    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise QuantizeError(f"{name}: header entry is not an object")
        dtype = entry.get("dtype")
        if dtype not in DTYPE_SIZE:
            raise QuantizeError(f"{name}: unknown safetensors dtype {dtype!r}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            raise QuantizeError(f"{name}: bad shape {shape!r}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
            or offsets[0] > offsets[1]
        ):
            raise QuantizeError(f"{name}: bad data_offsets {offsets!r}")
        want = DTYPE_SIZE[dtype] * int(np.prod(shape, dtype=np.int64))
        if offsets[1] - offsets[0] != want:
            raise QuantizeError(
                f"{name}: {dtype}{shape} needs {want} bytes but the header spans "
                f"{offsets[1] - offsets[0]}"
            )

    end = 0
    for name, entry in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
        start, stop = entry["data_offsets"]
        if start != end:
            raise QuantizeError(
                f"{name}: data_offsets start at {start}, expected {end} "
                "(safetensors buffers are gap-free and ordered)"
            )
        end = stop
    return metadata, header, 8 + header_len


def header_bytes(
    entries: list[tuple[str, str, tuple[int, ...], int, int]], metadata: dict
) -> bytes:
    """The 8-byte-aligned JSON header for ``(name, dtype, shape, start, stop)``.

    ``sort_keys`` plus the compact separators plus ``ensure_ascii`` is the whole
    determinism story for the header: the same plan always serialises to the
    same bytes, whatever order it was built in.
    """
    header: dict = {"__metadata__": metadata}
    for name, dtype, shape, start, stop in entries:
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, stop],
        }
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # The reference writer pads with spaces so the data buffer is 8-byte
    # aligned; JSON does not care about the trailing whitespace and readers
    # that assume the alignment keep working.
    return blob + b" " * (-len(blob) % 8)


def bf16_bytes_to_fp32(buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """bf16 bytes -> fp32 array, exactly: widen ``uint16``, ``<< 16``, view.

    Explicit little-endian dtypes throughout, since a safetensors buffer is
    little-endian regardless of the host.
    """
    u16 = np.frombuffer(buf, dtype="<u2")
    u32 = u16.astype("<u4")
    np.left_shift(u32, np.uint32(16), out=u32)
    return u32.view("<f4").reshape(shape)


def sha256_file(path: Path, chunk: int = COPY_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the quant math
# --------------------------------------------------------------------------- #
def quantize_per_group(
    w: np.ndarray, group: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """``[N,K]`` fp32 -> ``(int8 [N,K], fp32 [N,K/group])``.

    Vectorised twin of :func:`reference_quantize`; the docstring of this module
    states the arithmetic and the self-check holds the two to it.
    """
    if w.ndim != 2:
        raise QuantizeError(f"expected a 2-D weight, got shape {w.shape}")
    n, k = w.shape
    if k % group:
        raise QuantizeError(f"K={k} is not divisible by the group size {group}")
    if not np.isfinite(w).all():
        raise QuantizeError("weight holds NaN or Inf; refusing to quantise it")

    grouped = w.reshape(n, k // group, group)
    # max(|w|) without materialising |w|: the two reductions cost [N,G] each,
    # where `np.abs(grouped)` would cost another copy of the whole tensor.
    amax = np.maximum(np.max(grouped, axis=2), -np.min(grouped, axis=2))
    scales = np.where(
        amax == np.float32(0.0),
        np.float32(1.0),
        amax / np.float32(QMAX),
    ).astype("<f4")
    # One fp32 buffer, reused: `np.rint(grouped / scale)` would hold the
    # quotient and the rounded copy at the same time.
    q = np.empty_like(grouped)
    np.divide(grouped, scales[:, :, None], out=q)
    np.rint(q, out=q)
    np.clip(q, -QMAX, QMAX, out=q)
    return q.astype(np.int8).reshape(n, k), scales


def reference_quantize(
    w: np.ndarray, group: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-loop reference for :func:`quantize_per_group`. Self-check only.

    Slow on purpose and structurally different -- an explicit max loop, one
    division per element -- so the two cannot share a vectorisation bug. Every
    operation is still fp32, which is what makes bit-identical agreement the
    expected result rather than an approximation.
    """
    n, k = w.shape
    q = np.zeros((n, k), dtype=np.int8)
    scales = np.ones((n, k // group), dtype="<f4")
    for row in range(n):
        for g in range(k // group):
            chunk = w[row, g * group : (g + 1) * group]
            amax = np.float32(0.0)
            for x in chunk:
                a = np.float32(abs(np.float32(x)))
                if a > amax:
                    amax = a
            s = np.float32(1.0) if amax == np.float32(0.0) else np.float32(
                amax / np.float32(QMAX)
            )
            scales[row, g] = s
            for i, x in enumerate(chunk):
                v = float(np.rint(np.float32(x) / s))
                v = max(-float(QMAX), min(float(QMAX), v))
                q[row, g * group + i] = np.int8(int(v))
    return q, scales


def tensor_error(
    q: np.ndarray, scales: np.ndarray, w: np.ndarray, group: int = GROUP_SIZE
) -> dict:
    """``q*scale`` against the bf16-as-fp32 original: max-abs-err and rel-RMS.

    The sums of squares accumulate in float64 -- via ``einsum``, so a
    1.1 M-element tensor never materialises a float64 copy of itself -- and the
    max-abs comes off ``min``/``max`` rather than ``abs``, for the same reason.
    Written this way because this function, not the file, is what sets the
    tool's peak RSS: the obvious ``np.mean(np.square(err.astype(np.float64)))``
    spelling cost ~45 MiB of temporaries per expert tensor.
    """
    n, k = w.shape
    grouped_q = q.reshape(n, k // group, group)
    err = grouped_q.astype(np.float32)  # the one fp32 buffer this needs
    np.multiply(err, scales[:, :, None], out=err)
    np.subtract(err, w.reshape(n, k // group, group), out=err)
    err_sq = float(np.einsum("ijk,ijk->", err, err, dtype=np.float64))
    ref_sq = float(np.einsum("ij,ij->", w, w, dtype=np.float64))
    rms = float(np.sqrt(err_sq / err.size))
    ref_rms = float(np.sqrt(ref_sq / w.size))
    # An all-zero group is the sentinel pair (scale 1.0, q all zero), not just
    # a scale of 1.0 -- a group whose amax happens to be exactly 127.0 also has
    # scale 1.0 and is not zero at all.
    zero_groups = int(
        np.count_nonzero(
            np.all(grouped_q == 0, axis=2) & (scales == np.float32(1.0))
        )
    )
    return {
        "max_abs_err": max(float(np.max(err)), -float(np.min(err))),
        "rel_rms": (rms / ref_rms) if ref_rms > 0.0 else 0.0,
        "zero_groups": zero_groups,
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
    }


# --------------------------------------------------------------------------- #
# the plan: what the output holds, before a byte of it is written
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Item:
    """One output tensor.

    ``kind`` is ``copy`` (bytes straight through), ``quant`` (the int8 weight,
    read and quantised here) or ``scales`` (the fp32 companion, produced by the
    ``quant`` item that precedes it).
    """

    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    src_start: int = 0
    src_stop: int = 0
    weight_name: str = ""


def build_plan(header: dict) -> list[Item]:
    """Sorted-by-name output plan. Nothing here reads tensor data."""
    items: list[Item] = []
    for name in sorted(header):
        entry = header[name]
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        start, stop = entry["data_offsets"]

        if name.endswith(SCALES_SUFFIX) and EXPERT_WEIGHT_RE.match(
            name[: -len(SCALES_SUFFIX)]
        ):
            raise QuantizeError(
                f"{name}: the source already carries per-group scales -- this "
                "looks like an already-quantised checkpoint"
            )

        if EXPERT_WEIGHT_RE.match(name) is None:
            items.append(
                Item(
                    name=name,
                    kind="copy",
                    dtype=dtype,
                    shape=shape,
                    nbytes=stop - start,
                    src_start=start,
                    src_stop=stop,
                )
            )
            continue

        if dtype != "BF16":
            raise QuantizeError(
                f"{name}: routed expert weights must be BF16, header says {dtype}"
            )
        if len(shape) != 2:
            raise QuantizeError(
                f"{name}: expected a per-expert 2-D [N,K] weight, header says "
                f"{list(shape)} (a stacked [E,N,K] tensor is not this format)"
            )
        n, k = shape
        if k % GROUP_SIZE:
            raise QuantizeError(
                f"{name}: K={k} is not divisible by the group size {GROUP_SIZE}"
            )
        items.append(
            Item(
                name=name,
                kind="quant",
                dtype="I8",
                shape=shape,
                nbytes=n * k,
                src_start=start,
                src_stop=stop,
            )
        )
        items.append(
            Item(
                name=name + SCALES_SUFFIX,
                kind="scales",
                dtype="F32",
                shape=(n, k // GROUP_SIZE),
                nbytes=n * (k // GROUP_SIZE) * 4,
                weight_name=name,
            )
        )

    names = [item.name for item in items]
    # The layout *is* the sorted order -- `x.weight` then `x.weight_scales`,
    # with nothing able to sort between them -- and that is what lets a `quant`
    # item hand its scales to the very next item instead of buffering them. A
    # real check rather than an `assert`, because `-O` must not be able to turn
    # the determinism contract off.
    if names != sorted(names):
        raise QuantizeError(
            "output layout is not in sorted-name order -- some tensor name "
            f"sorts between a weight and its {SCALES_SUFFIX} companion"
        )
    if len(set(names)) != len(names):
        raise QuantizeError("output plan has duplicate tensor names")
    return items


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def quantize_file(
    src: Path,
    dst: Path,
    *,
    expect_sha: str | None,
    progress_every: int = 512,
) -> dict:
    """Stream ``src`` to ``dst``, quantising the routed experts. Returns the report."""
    src_sha = sha256_file(src)
    if expect_sha is not None and src_sha != expect_sha.strip().lower():
        raise QuantizeError(
            f"source sha256 mismatch\n  expected {expect_sha.strip().lower()}\n"
            f"  actual   {src_sha}\nrefusing to quantise a checkpoint that is "
            "not the one this run was pinned to"
        )

    src_bytes = src.stat().st_size
    quantized: list[dict] = []

    with open(src, "rb") as fh:
        src_meta, header, data_start = read_header(fh)
        plan = build_plan(header)
        if not any(item.kind == "quant" for item in plan):
            raise QuantizeError(
                "no expert weights matched the pattern -- refusing to write a "
                "byte-copy that only looks quantised"
            )

        offsets: list[tuple[int, int]] = []
        cursor = 0
        for item in plan:
            offsets.append((cursor, cursor + item.nbytes))
            cursor += item.nbytes

        out_meta = {str(k): str(v) for k, v in sorted(src_meta.items())}
        out_meta.update(
            {
                "group_size": str(GROUP_SIZE),
                "scheme": SCHEME,
                "source_sha256": src_sha,
                "tool": TOOL,
            }
        )
        head = header_bytes(
            [
                (item.name, item.dtype, item.shape, start, stop)
                for item, (start, stop) in zip(plan, offsets)
            ],
            out_meta,
        )

        tmp = dst.with_name(dst.name + ".partial")
        digest = hashlib.sha256()
        pending: dict[str, bytes] = {}

        with open(tmp, "wb") as out:

            def emit(block: bytes) -> None:
                out.write(block)
                digest.update(block)

            emit(struct.pack("<Q", len(head)))
            emit(head)

            for index, item in enumerate(plan):
                if item.kind == "copy":
                    fh.seek(data_start + item.src_start)
                    left = item.nbytes
                    while left:
                        block = fh.read(min(COPY_CHUNK, left))
                        if len(block) != min(COPY_CHUNK, left):
                            raise QuantizeError(
                                f"{item.name}: source ended early ({left} bytes left)"
                            )
                        emit(block)
                        left -= len(block)
                elif item.kind == "quant":
                    fh.seek(data_start + item.src_start)
                    raw = fh.read(item.src_stop - item.src_start)
                    if len(raw) != item.src_stop - item.src_start:
                        raise QuantizeError(f"{item.name}: source ended early")
                    w = bf16_bytes_to_fp32(raw, item.shape)
                    q, scales = quantize_per_group(w)
                    emit(q.tobytes())
                    pending[item.name + SCALES_SUFFIX] = scales.tobytes()
                    stats = {"name": item.name, "shape": list(item.shape)}
                    stats.update(tensor_error(q, scales, w))
                    quantized.append(stats)
                    del raw, w, q
                else:
                    emit(pending.pop(item.name))
                # One `quant` item is ever in flight, by the sorted-order
                # argument in `build_plan`: this is the streaming guarantee,
                # so it is checked rather than asserted.
                if len(pending) > 1:
                    raise QuantizeError(
                        f"{len(pending)} tensors' scales in flight at once -- "
                        "the streaming invariant is broken"
                    )

                if progress_every and index and index % progress_every == 0:
                    print(
                        f"  ... {index}/{len(plan)} tensors, "
                        f"{out.tell() / 2**20:.0f} MiB written",
                        file=sys.stderr,
                        flush=True,
                    )

            written = out.tell()

        if pending:
            raise QuantizeError(f"scales never written: {sorted(pending)}")
        expected = 8 + len(head) + cursor
        if written != expected:
            raise QuantizeError(f"wrote {written} bytes, planned {expected}")

    os.replace(tmp, dst)

    metrics = {
        key: [t[key] for t in quantized] for key in ("max_abs_err", "rel_rms")
    }
    worst_rel = max(quantized, key=lambda t: t["rel_rms"]) if quantized else None
    worst_abs = max(quantized, key=lambda t: t["max_abs_err"]) if quantized else None
    return {
        "tool": TOOL,
        "scheme": SCHEME,
        "group_size": GROUP_SIZE,
        "source": str(src),
        "source_sha256": src_sha,
        "source_bytes": src_bytes,
        "output": str(dst),
        "output_sha256": digest.hexdigest(),
        "output_bytes": written,
        "tensors_total": len(plan),
        "tensors_quantized": len(quantized),
        "tensors_passthrough": sum(1 for i in plan if i.kind == "copy"),
        "metadata": out_meta,
        "worst_rel_rms": worst_rel,
        "worst_max_abs_err": worst_abs,
        "summary": {
            key: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for key, values in metrics.items()
            if values
        },
        "quantized": quantized,
    }


def print_summary(report: dict, report_path: Path) -> None:
    def gb(n: int) -> str:
        return f"{n / 10**9:.2f} GB"

    print(f"source  {report['source']}")
    print(f"        {gb(report['source_bytes'])}  sha {report['source_sha256']}")
    print(f"output  {report['output']}")
    print(f"        {gb(report['output_bytes'])}  sha {report['output_sha256']}")
    saved = report["source_bytes"] - report["output_bytes"]
    print(
        f"        {gb(saved)} smaller "
        f"({100 * saved / max(1, report['source_bytes']):.1f}%)"
    )
    print()
    print(
        f"quantised {report['tensors_quantized']} routed-expert weights "
        f"(+ {report['tensors_quantized']} fp32 scale tensors); "
        f"{report['tensors_passthrough']} tensors copied byte for byte"
    )
    print(f"scheme    {report['scheme']}, group {report['group_size']} along K")
    if report["quantized"]:
        s = report["summary"]
        print()
        print(f"{'':<10}{'max-abs-err':>14}{'rel-RMS':>14}")
        for stat in ("min", "median", "max"):
            print(
                f"{stat:<10}{s['max_abs_err'][stat]:>14.3e}"
                f"{s['rel_rms'][stat]:>14.3e}"
            )
        worst = report["worst_rel_rms"]
        print(
            f"\nworst rel-RMS  {worst['rel_rms']:.3e}  {worst['name']} "
            f"{worst['shape']}"
        )
        worst = report["worst_max_abs_err"]
        print(
            f"worst max-abs  {worst['max_abs_err']:.3e}  {worst['name']} "
            f"{worst['shape']}"
        )
    print()
    print(f"report -> {report_path}")


def write_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# self-check scaffolding: a synthetic checkpoint writer that shares no code
# with the production header writer above, on purpose
# --------------------------------------------------------------------------- #
def _to_bf16_bytes(a: np.ndarray) -> bytes:
    """fp32 -> bf16 bytes, round-half-to-even. Test scaffolding only.

    The tool never writes bf16; this exists so the synthetic checkpoints hold
    real bf16. Values are small and finite by construction, so the ``+ 0x7FFF``
    rounding bias cannot overflow the exponent field.
    """
    u = np.ascontiguousarray(a, dtype="<f4").view("<u4")
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype("<u2").tobytes()


@dataclasses.dataclass
class _Synth:
    """One tensor of a synthetic checkpoint; ``chunks`` may stream."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunks: object  # callable() -> iterable[bytes]


def _synth_from_array(name: str, dtype: str, a: np.ndarray) -> _Synth:
    if dtype == "BF16":
        blob = _to_bf16_bytes(a)
    elif dtype == "F32":
        blob = np.ascontiguousarray(a, dtype="<f4").tobytes()
    else:
        raise AssertionError(dtype)
    return _Synth(name, dtype, tuple(a.shape), len(blob), lambda blob=blob: [blob])


def _write_synthetic(path: Path, tensors: list[_Synth], metadata: dict) -> None:
    """Write a safetensors file from ``tensors``, independently of the tool."""
    header: dict = {"__metadata__": metadata}
    cursor = 0
    for t in sorted(tensors, key=lambda t: t.name):
        header[t.name] = {
            "dtype": t.dtype,
            "shape": list(t.shape),
            "data_offsets": [cursor, cursor + t.nbytes],
        }
        cursor += t.nbytes
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for t in sorted(tensors, key=lambda t: t.name):
            written = 0
            for block in t.chunks():
                fh.write(block)
                written += len(block)
            assert written == t.nbytes, (t.name, written, t.nbytes)


def _read_tensor(path: Path, name: str) -> tuple[dict, bytes]:
    with open(path, "rb") as fh:
        _, header, data_start = read_header(fh)
        entry = header[name]
        start, stop = entry["data_offsets"]
        fh.seek(data_start + start)
        return entry, fh.read(stop - start)


def _rss_bytes(usage: resource.struct_rusage) -> int:
    """``ru_maxrss`` is bytes on macOS and kilobytes on Linux."""
    return usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024


# --------------------------------------------------------------------------- #
# the golden micro-tensor
# --------------------------------------------------------------------------- #
def _golden() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w, q, scales)`` -- a ``[2,256]`` weight whose quantisation is derived
    by hand, group by group.

    Every value below is exactly representable in bf16 (its fp32 pattern has a
    zero low half), so the arithmetic in the comments is the arithmetic the
    tool performs -- no rounding hides between them.

    row 0, group 0 -- amax ``127/128``, so ``scale = 2**-7`` is **exact** and
    the edges are unambiguous::

        +0.9921875  = +127*scale -> +127   (amax edge, positive)
        -0.9921875  = -127*scale -> -127   (amax edge, negative)
        +0.0078125  =    1*scale ->   +1
        +0.00390625 =  0.5*scale ->    0    (rint tie -> even)
        +0.01171875 =  1.5*scale ->   +2    (rint tie -> even)
        -0.00390625 = -0.5*scale ->    0
        -0.01171875 = -1.5*scale ->   -2
        +0.5        =   64*scale ->  +64
        -0.25       =  -32*scale ->  -32

    row 0, group 1 -- all zero: ``scale = 1.0``, ``q = 0``. The scale is the
    documented sentinel, not ``amax/127 = 0``.

    row 1, group 0 -- ``amax = 2.0``, so ``scale = fp32(2/127) =
    0x1.020408p-6`` is **not** a power of two::

        +2.0    -> 2.0/scale     = 127.0   -> +127
        -1.5    -> -1.5/scale    = -95.25  ->  -95
        -0.5    -> -0.5/scale    = -31.75  ->  -32
        +0.25   -> 0.25/scale    =  15.875 ->  +16
        +0.0625 -> 0.0625/scale  =   3.96875 -> +4

    row 1, group 1 -- ``amax = 127*2**-14``, ``scale = 2**-14`` exactly, and a
    value far below the scale::

        +0.00775146484375 = +127*scale -> +127
        +6.103515625e-05  =    1*scale ->   +1
        +9.5367431640625e-07 = 0.015625*scale -> 0   (underflows to zero)
    """
    scale0 = 2.0**-7
    scale1 = float.fromhex("0x1.020408p-6")  # fp32(2/127)
    scale3 = 2.0**-14

    w = np.zeros((2, 256), dtype=np.float32)
    q = np.zeros((2, 256), dtype=np.int8)

    row0 = [
        (0, +0.9921875, +127),
        (1, -0.9921875, -127),
        (2, +0.0078125, +1),
        (3, +0.00390625, 0),
        (4, +0.01171875, +2),
        (5, -0.00390625, 0),
        (6, -0.01171875, -2),
        (7, +0.5, +64),
        (8, -0.25, -32),
    ]
    for i, value, expect in row0:
        w[0, i] = value
        q[0, i] = expect

    row1_g0 = [
        (0, +2.0, +127),
        (1, -1.5, -95),
        (2, -0.5, -32),
        (3, +0.25, +16),
        (4, +0.0625, +4),
    ]
    for i, value, expect in row1_g0:
        w[1, i] = value
        q[1, i] = expect

    row1_g1 = [
        (0, 127 * 2.0**-14, +127),
        (1, 2.0**-14, +1),
        (2, 2.0**-20, 0),
    ]
    for i, value, expect in row1_g1:
        w[1, 128 + i] = value
        q[1, 128 + i] = expect

    scales = np.array([[scale0, 1.0], [scale1, scale3]], dtype="<f4")
    return w, q, scales


# --------------------------------------------------------------------------- #
# --self-check
# --------------------------------------------------------------------------- #
def _mini_checkpoint(seed: int = 20260909) -> list[_Synth]:
    """Tiny shapes, ``K % 128 == 0``, plus the near-misses and the edge cases.

    The three ``must not be selected`` names are the ones a looser rule would
    catch: a shared expert, a bias on a real expert projection, and the stacked
    ``experts.gate_proj`` (no expert index) this port's MAX side actually uses.
    """
    rng = np.random.default_rng(seed)

    gate = rng.standard_normal((4, 256), dtype=np.float32) * 0.05
    # an all-zero group inside a real expert tensor, and an amax edge value in
    # situ: group (0, 0) holds one 127*2**-7 and nothing else, so its scale is
    # exactly 2**-7 and its q is +-127 at index 0 and 0 everywhere else
    gate[2, 128:] = 0.0
    gate[0, 0] = 0.9921875
    gate[0, 1:128] = 0.0

    down = rng.standard_normal((3, 256), dtype=np.float32) * 0.02
    down[1, :128] = 0.0  # the second all-zero group, in a second tensor
    up = rng.standard_normal((2, 384), dtype=np.float32) * 0.1

    return [
        _synth_from_array(
            "model.layers.1.mlp.experts.0.gate_proj.weight", "BF16", gate
        ),
        _synth_from_array(
            "model.layers.1.mlp.experts.0.down_proj.weight", "BF16", down
        ),
        _synth_from_array("model.layers.2.mlp.experts.63.up_proj.weight", "BF16", up),
        # non-expert bf16 and fp32 passengers; [5,7] is 70 bytes, so it also
        # proves the writer does not assume per-tensor alignment
        _synth_from_array(
            "model.embed_tokens.weight",
            "BF16",
            rng.standard_normal((5, 7), dtype=np.float32),
        ),
        _synth_from_array(
            "model.layers.0.mlp.gate_proj.weight",
            "BF16",
            rng.standard_normal((3, 5), dtype=np.float32),
        ),
        _synth_from_array(
            "model.norm.weight", "F32", rng.standard_normal((4,), dtype=np.float32)
        ),
        # near-misses: same neighbourhood, must be copied not quantised
        _synth_from_array(
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "BF16",
            rng.standard_normal((2, 256), dtype=np.float32),
        ),
        _synth_from_array(
            "model.layers.1.mlp.experts.0.gate_proj.bias",
            "BF16",
            rng.standard_normal((4,), dtype=np.float32),
        ),
        _synth_from_array(
            "model.layers.1.mlp.experts.gate_proj",
            "BF16",
            rng.standard_normal((2, 4, 256), dtype=np.float32),
        ),
    ]


def _phase_golden() -> int:
    w, want_q, want_scales = _golden()

    # the hand-derived values must survive bf16, or the derivation is about
    # numbers the tool will never see
    roundtrip = bf16_bytes_to_fp32(_to_bf16_bytes(w), w.shape)
    bad = 0
    if not np.array_equal(roundtrip, w):
        print("  BROKEN golden values are not bf16-exact")
        bad += 1

    q, scales = quantize_per_group(w)
    for label, got, want in (
        ("q", q, want_q),
        ("scales", scales, want_scales),
    ):
        ok = np.array_equal(got, want)
        print(f"  golden {label:<7} {'ok' if ok else 'BROKEN'}")
        if not ok:
            bad += 1
            diff = np.argwhere(got != want)[:8]
            for idx in diff:
                idx = tuple(int(i) for i in idx)
                print(f"    {idx}: got {got[idx]!r} want {want[idx]!r}")

    ref_q, ref_scales = reference_quantize(w)
    ok = np.array_equal(ref_q, want_q) and np.array_equal(ref_scales, want_scales)
    print(f"  golden via reference quantiser {'ok' if ok else 'BROKEN'}")
    bad += 0 if ok else 1

    # the sentinel, stated as its own assertion
    ok = scales[0, 1] == np.float32(1.0) and not q[0, 128:].any()
    print(f"  all-zero group -> scale 1.0, q all zero {'ok' if ok else 'BROKEN'}")
    bad += 0 if ok else 1

    edge = q[0, 0] == 127 and q[0, 1] == -127
    print(f"  amax edge -> +-127 exactly {'ok' if edge else 'BROKEN'}")
    bad += 0 if edge else 1
    return bad


def _phase_roundtrip(tmp: Path) -> int:
    bad = 0
    src = tmp / "mini.safetensors"
    dst = tmp / "mini-int8.safetensors"
    tensors = _mini_checkpoint()
    _write_synthetic(src, tensors, {"format": "pt"})
    src_sha = sha256_file(src)

    report = quantize_file(src, dst, expect_sha=src_sha, progress_every=0)
    write_report(report, Path(str(dst) + ".report.json"))

    with open(src, "rb") as fh:
        src_meta, src_header, _ = read_header(fh)
    with open(dst, "rb") as fh:
        out_meta, out_header, _ = read_header(fh)

    experts = sorted(n for n in src_header if EXPERT_WEIGHT_RE.match(n))
    print(f"  mini-checkpoint: {len(src_header)} tensors, {len(experts)} routed expert")
    if len(experts) != 3:
        print("  BROKEN expected 3 expert tensors in the fixture")
        bad += 1

    for name in experts:
        n, k = src_header[name]["shape"]
        entry, blob = _read_tensor(dst, name)
        s_entry, s_blob = _read_tensor(dst, name + SCALES_SUFFIX)
        _, raw = _read_tensor(src, name)
        w = bf16_bytes_to_fp32(raw, (n, k))
        ref_q, ref_scales = reference_quantize(w)

        checks = {
            "dtype I8": entry["dtype"] == "I8",
            "shape [N,K]": entry["shape"] == [n, k],
            "scales dtype F32": s_entry["dtype"] == "F32",
            f"scales shape [{n},{k // GROUP_SIZE}]": s_entry["shape"]
            == [n, k // GROUP_SIZE],
            "q == reference": blob == ref_q.tobytes(),
            "scales == reference": s_blob == ref_scales.tobytes(),
        }
        failed = [label for label, ok in checks.items() if not ok]
        print(
            f"  {name.split('.mlp.')[-1]:<28} [{n},{k}] -> int8 + [{n},"
            f"{k // GROUP_SIZE}] fp32  " + ("ok" if not failed else f"BROKEN {failed}")
        )
        bad += len(failed)

    passthrough = sorted(n for n in src_header if not EXPERT_WEIGHT_RE.match(n))
    identical = 0
    for name in passthrough:
        _, want = _read_tensor(src, name)
        entry, got = _read_tensor(dst, name)
        if (
            got == want
            and entry["dtype"] == src_header[name]["dtype"]
            and entry["shape"] == src_header[name]["shape"]
        ):
            identical += 1
        else:
            print(f"  BROKEN {name} is not byte-identical")
            bad += 1
    print(
        f"  {identical}/{len(passthrough)} non-expert tensors byte-identical "
        f"(incl. {sum(1 for n in passthrough if 'experts' in n)} near-miss names "
        "left alone)"
    )

    want_meta = {
        "source_sha256": src_sha,
        "group_size": "128",
        "scheme": SCHEME,
        "tool": TOOL,
    }
    missing = {k: v for k, v in want_meta.items() if out_meta.get(k) != v}
    print(
        f"  __metadata__ {sorted(out_meta)} "
        + ("ok" if not missing else f"BROKEN {missing}")
    )
    bad += 1 if missing else 0
    if out_meta.get("format") != src_meta.get("format"):
        print("  BROKEN source metadata not carried through")
        bad += 1

    leftover = [
        n
        for n, e in out_header.items()
        if EXPERT_WEIGHT_RE.match(n) and e["dtype"] != "I8"
    ]
    scale_count = sum(
        1
        for n in out_header
        if n.endswith(SCALES_SUFFIX)
        and EXPERT_WEIGHT_RE.match(n[: -len(SCALES_SUFFIX)])
    )
    ok = not leftover and scale_count == len(experts)
    print(
        f"  {scale_count} scale tensors, 0 bf16 expert weights left "
        + ("ok" if ok else f"BROKEN {leftover}")
    )
    bad += 0 if ok else 1

    # the fixture's two all-zero groups have to show up as the sentinel pair
    zero_groups = sum(t["zero_groups"] for t in report["quantized"])
    print(
        f"  report: {len(report['quantized'])} tensors, "
        f"{zero_groups} all-zero groups"
    )
    if zero_groups != 2:  # gate row 2 group 1, down row 1 group 0
        print(f"  BROKEN expected exactly 2 all-zero groups, report says {zero_groups}")
        bad += 1

    # and the in-situ amax edge: gate group (0, 0) is one 127*2**-7 among zeros
    _, gate_q = _read_tensor(dst, "model.layers.1.mlp.experts.0.gate_proj.weight")
    _, gate_s = _read_tensor(
        dst, "model.layers.1.mlp.experts.0.gate_proj.weight" + SCALES_SUFFIX
    )
    q0 = np.frombuffer(gate_q, dtype=np.int8).reshape(4, 256)[0, :128]
    s0 = np.frombuffer(gate_s, dtype="<f4").reshape(4, 2)[0, 0]
    ok = q0[0] == 127 and not q0[1:].any() and s0 == np.float32(2.0**-7)
    print(f"  in-situ amax edge -> q 127, scale 2**-7  {'ok' if ok else 'BROKEN'}")
    bad += 0 if ok else 1

    if not Path(str(dst) + ".report.json").exists():
        print("  BROKEN sidecar report missing")
        bad += 1
    return bad


def _phase_determinism(tmp: Path) -> int:
    src = tmp / "det.safetensors"
    _write_synthetic(src, _mini_checkpoint(), {"format": "pt"})
    sha = sha256_file(src)
    shas = []
    for run in ("a", "b"):
        dst = tmp / f"det-{run}.safetensors"
        report = quantize_file(src, dst, expect_sha=sha, progress_every=0)
        on_disk = sha256_file(dst)
        assert on_disk == report["output_sha256"], "streamed digest != file digest"
        shas.append(on_disk)
    ok = shas[0] == shas[1]
    print(
        f"  two runs -> {shas[0][:16]}... / {shas[1][:16]}...  "
        f"{'ok' if ok else 'BROKEN'}"
    )
    return 0 if ok else 1


def _phase_refusals(tmp: Path) -> int:
    bad = 0
    src = tmp / "refuse.safetensors"
    _write_synthetic(src, _mini_checkpoint(), {"format": "pt"})
    sha = sha256_file(src)

    cases: list[tuple[str, list[_Synth] | None, str, str]] = [
        ("wrong --expect-sha", None, "0" * 64, "sha256 mismatch"),
        (
            "K not divisible by 128",
            [
                _synth_from_array(
                    "model.layers.3.mlp.experts.1.up_proj.weight",
                    "BF16",
                    np.zeros((2, 200), dtype=np.float32),
                )
            ],
            sha,
            "not divisible",
        ),
        (
            "expert weight not bf16",
            [
                _synth_from_array(
                    "model.layers.3.mlp.experts.1.up_proj.weight",
                    "F32",
                    np.zeros((2, 256), dtype=np.float32),
                )
            ],
            sha,
            "must be BF16",
        ),
        (
            "stacked 3-D expert weight",
            [
                _synth_from_array(
                    "model.layers.3.mlp.experts.1.up_proj.weight",
                    "BF16",
                    np.zeros((4, 2, 256), dtype=np.float32),
                )
            ],
            sha,
            "2-D",
        ),
        (
            "source already has scales",
            [
                _synth_from_array(
                    "model.layers.3.mlp.experts.1.up_proj.weight",
                    "BF16",
                    np.zeros((2, 256), dtype=np.float32),
                ),
                _synth_from_array(
                    "model.layers.3.mlp.experts.1.up_proj.weight_scales",
                    "F32",
                    np.zeros((2, 2), dtype=np.float32),
                ),
            ],
            sha,
            "already carries",
        ),
    ]

    for index, (label, tensors, expect_sha, needle) in enumerate(cases):
        if tensors is None:
            case_src = src
        else:
            case_src = tmp / f"refuse-{index}.safetensors"
            _write_synthetic(case_src, tensors, {"format": "pt"})
            expect_sha = sha256_file(case_src)
        dst = tmp / f"refuse-out-{index}.safetensors"
        try:
            quantize_file(case_src, dst, expect_sha=expect_sha, progress_every=0)
        except QuantizeError as exc:
            hit = needle in str(exc)
            print(f"  {label:<26} refused {'ok' if hit else 'BROKEN (wrong message)'}")
            bad += 0 if hit else 1
        else:
            print(f"  {label:<26} BROKEN: accepted")
            bad += 1
        if dst.exists():
            print(f"  {label:<26} BROKEN: wrote an output anyway")
            bad += 1
    return bad


def _phase_streaming(tmp: Path) -> int:
    """A ~78 MiB checkpoint through the CLI, with the child's peak RSS measured.

    Peak RSS is read off ``RUSAGE_CHILDREN`` rather than this process: a
    high-water mark in the parent (which has already written the fixture) would
    mask exactly what is being claimed. The baseline child is a bare
    ``import numpy``, so the interpreter's own footprint is visible next to the
    number rather than folded into it.
    """
    bad = 0
    src = tmp / "big.safetensors"

    # 64 MiB of passthrough, written in chunks so the *fixture* does not need
    # the memory either, plus two experts at the real shapes.
    rows, cols = 4096, 8192

    def big_chunks(rows=rows, cols=cols):
        for start in range(0, rows, 256):
            block = (
                np.arange(start * cols, min(start + 256, rows) * cols, dtype=np.uint32)
                % np.uint32(30000)
            ).astype("<u2")
            # keep the exponent field small so the bf16 values stay finite and
            # tiny; the bytes are what matter here, not the numerics
            yield (block | np.uint16(0x3C00)).astype("<u2").tobytes()

    rng = np.random.default_rng(7)
    tensors = [
        _Synth(
            "model.embed_tokens.weight",
            "BF16",
            (rows, cols),
            rows * cols * 2,
            big_chunks,
        )
    ]
    for expert in (0, 1):
        for proj, shape in (
            ("gate_proj", (896, 1280)),
            ("up_proj", (896, 1280)),
            ("down_proj", (1280, 896)),
        ):
            tensors.append(
                _synth_from_array(
                    f"model.layers.1.mlp.experts.{expert}.{proj}.weight",
                    "BF16",
                    (rng.standard_normal(shape, dtype=np.float32) * 0.05),
                )
            )
    _write_synthetic(src, tensors, {"format": "pt"})
    src_sha = sha256_file(src)
    src_bytes = src.stat().st_size

    subprocess.run([sys.executable, "-c", "import numpy"], check=True)
    baseline = _rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))

    dst = tmp / "big-int8.safetensors"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(src),
            str(dst),
            "--expect-sha",
            src_sha,
        ],
        capture_output=True,
        text=True,
    )
    peak = _rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))
    if proc.returncode != 0:
        print(f"  BROKEN CLI exited {proc.returncode}\n{proc.stdout}{proc.stderr}")
        return bad + 1

    mib = 2**20
    print(
        f"  input {src_bytes / mib:.1f} MiB -> output "
        f"{dst.stat().st_size / mib:.1f} MiB via the CLI"
    )
    print(
        f"  peak RSS: quantiser <= {peak / mib:.1f} MiB, bare `import numpy` "
        f"{baseline / mib:.1f} MiB, 2x input {2 * src_bytes / mib:.1f} MiB"
    )
    if peak >= 2 * src_bytes:
        print("  BROKEN peak RSS is not far below 2x the input size")
        bad += 1
    if peak > baseline + 48 * mib:
        print("  BROKEN peak RSS grew more than 48 MiB over a bare interpreter")
        bad += 1

    report_path = Path(str(dst) + ".report.json")
    if not report_path.exists():
        print("  BROKEN CLI wrote no sidecar report")
        bad += 1
    else:
        report = json.loads(report_path.read_text())
        ok = report["tensors_quantized"] == 6 and report[
            "output_sha256"
        ] == sha256_file(dst)
        print(
            f"  sidecar: {report['tensors_quantized']} quantised, "
            f"worst rel-RMS {report['worst_rel_rms']['rel_rms']:.3e}  "
            + ("ok" if ok else "BROKEN")
        )
        bad += 0 if ok else 1

    # and the CLI must refuse the same file under the wrong sha
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(src),
            str(tmp / "never.safetensors"),
            "--expect-sha",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
    )
    refused = proc.returncode != 0 and not (tmp / "never.safetensors").exists()
    print(f"  CLI refuses a wrong --expect-sha  {'ok' if refused else 'BROKEN'}")
    bad += 0 if refused else 1
    return bad


def self_check() -> int:
    """Five phases, no checkpoint. See this module's docstring."""
    bad = 0
    with tempfile.TemporaryDirectory(prefix="quantize_int8-selfcheck-") as tmpdir:
        tmp = Path(tmpdir)
        for label, phase in (
            ("1 golden micro-tensor", lambda: _phase_golden()),
            ("2 mini-checkpoint round trip", lambda: _phase_roundtrip(tmp)),
            ("3 determinism", lambda: _phase_determinism(tmp)),
            ("4 refusals", lambda: _phase_refusals(tmp)),
            ("5 streaming / peak RSS", lambda: _phase_streaming(tmp)),
        ):
            print(f"[{label}]")
            bad += phase()
            print()

    print(
        "SELF-CHECK: "
        + (
            "ok -- goldens, format contract, byte-identical passthrough, "
            "determinism, refusals and streaming all hold"
            if bad == 0
            else f"BROKEN on {bad} checks"
        )
    )
    return 0 if bad == 0 else 1


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "src",
        nargs="?",
        type=Path,
        help="the bf16 model.safetensors to read (never modified)",
    )
    ap.add_argument(
        "dst",
        nargs="?",
        type=Path,
        help="the int8 model-int8.safetensors to write, plus <dst>.report.json",
    )
    ap.add_argument(
        "--expect-sha",
        default=None,
        help=(
            "sha256 of src, required for a real run: the scheme is pinned to a "
            "specific checkpoint and quantising a different one silently would "
            "be worse than refusing. `shasum -a 256 <src>`"
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite dst if it already exists",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help=(
            "prove the tool against hand-derived goldens and a synthetic "
            "mini-checkpoint. No checkpoint, no network, nothing outside a "
            "temp dir"
        ),
    )
    args = ap.parse_args(argv)

    if args.self_check:
        if args.src is not None:
            ap.error("--self-check takes no src/dst")
        return self_check()

    if args.src is None or args.dst is None:
        ap.error("src and dst are required (or pass --self-check)")
    if args.expect_sha is None:
        ap.error(
            "--expect-sha is required: run `shasum -a 256 "
            f"{args.src}` and pass the digest"
        )

    try:
        if not args.src.is_file():
            raise QuantizeError(f"{args.src}: not a file")
        if args.src.resolve() == args.dst.resolve():
            raise QuantizeError("src and dst are the same file")
        if args.dst.exists() and not args.force:
            raise QuantizeError(f"{args.dst} exists; pass --force to overwrite")
        report = quantize_file(args.src, args.dst, expect_sha=args.expect_sha)
    except (QuantizeError, OSError) as exc:
        # A mid-stream failure (disk full, truncated source, missing dst dir)
        # must not orphan a multi-GB partial next to a believable dst name.
        args.dst.with_name(args.dst.name + ".partial").unlink(missing_ok=True)
        print(f"quantize_int8: {exc}", file=sys.stderr)
        return 2
    except BaseException:
        args.dst.with_name(args.dst.name + ".partial").unlink(missing_ok=True)
        raise

    report_path = Path(str(args.dst) + ".report.json")
    write_report(report, report_path)
    print_summary(report, report_path)
    print(
        f"peak RSS {_rss_bytes(resource.getrusage(resource.RUSAGE_SELF)) / 2**20:.1f} "
        f"MiB against a {report['source_bytes'] / 2**20:.0f} MiB input"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
