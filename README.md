# unlimited-ocr-max

[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) served through
[MAX](https://docs.modular.com/max/) on Apple Silicon. The vision tower, the MoE
decoder and its sliding-window attention are rebuilt as a MAX custom architecture
with a [Mojo](https://docs.modular.com/mojo/) custom op (`ngram_block`, the
no-repeat-n-gram guard), and exposed as an OpenAI-compatible endpoint on the
Metal GPU or the CPU. The weights are baidu's, unchanged, served from
[`kthierbach/unlimited-ocr-max`](https://huggingface.co/kthierbach/unlimited-ocr-max).

## Install

```bash
uv tool install unlimited-ocr-max
# or, into a venv
pip install unlimited-ocr-max
```

**From v0.3.0 on, PyPI alone is enough:** the package pins one exact MAX release
(`max[all]==26.6.0`), and that release — with the `mojo` it depends on — is
published on PyPI. Up to and including v0.2.1 the pin was a `26.6.0.dev*`
nightly, because the 26.6 fixes this port needs had not reached a stable release
yet — so installing **those** versions needs
`--extra-index-url https://whl.modular.com/nightly/simple/` (and `--pre` with
`pip`, for the pre-release `mojo`). From v0.3.0, and from this source tree,
neither is needed.

## Serve

```bash
unlimited-ocr-max serve --devices gpu   # Metal
unlimited-ocr-max serve --devices cpu   # supported, slow
```

This downloads the model repository once (6.2 GiB) and runs `max serve` with
this port's flags, on `http://127.0.0.1:8010` under the model id
`unlimited-ocr-max`.

* `--revision` defaults to `DEFAULT_REVISION` in `unlimited_ocr_max/cli.py`
  (tied by a package test to the package's own version) — the model-repo tag
  this package version was validated against, so a fixed
  package version serves fixed weights; the tag must exist or the download fails
  before MAX starts. Ignored for a local directory. `unlimited-ocr-max serve
  --help` prints the value the installed build carries.
* `--model <dir>` serves a local copy with the repository's layout
  (`config.json`, the tokenizer files, `model.safetensors`).
* `--weights bf16` (default) selects the unquantised `model.safetensors`; it
  serves on `--devices cpu` or `gpu`. `--weights int8` selects
  `model-int8.safetensors`, a symmetric per-group int8 quantisation of the 64
  routed experts (group size 128, along the input dimension; every other tensor
  stays bf16), and is **GPU only** — the dequantise-and-matmul runs in a Mojo
  custom op (`moe_int8`), since MAX's own int8 matmul is gated to a later Apple
  GPU. The chosen file is passed as `max serve --weight-path`; neither name
  carries an encoding token, because MAX reads hints such as `bf16` out of weight
  filenames and would refuse the CPU path or mislabel the GPU one (a package test
  pins this). int8 reproduces bf16's transcribed **text byte-for-byte** on the
  port's twelve-page set; only grounding bounding-box coordinates differ, by one
  or two pixels. It cuts the served peak memory by ~2.5 GiB (16.7 → 14.2 GiB on
  the M4 24 GB), which is what brings the model within reach of a 16 GB machine. It also
  decodes about 1.8× faster (~37 vs ~21 tokens/second on that M4, greedy), because
  the routed-expert decode is weight-bandwidth-bound and int8 halves the bytes read.
* `--ngram-size` sets the no-repeat-n-gram guard, default 35; `0` switches it
  off, which reproduces the PyTorch reference byte for byte.

On the GPU the server holds one language graph at a time — the vision tower,
the prefill graph and the decode graph together do not fit the Metal budget —
so every request reloads a graph. With `--weights bf16` the language weights
are bound as **one shared device registry** that both graphs declare
device-side, which takes that per-request decode reload from ~10.5 s to **~1.5 s**
and the time to first token from ~8.5 s to ~5.3 s. It is not free: holding the
registry through decode costs about **+9 %** on the steady decode step, and the
net is still **−7 to −15 s per page** on every page measured. `--weights int8`
deliberately does **not** share: the same registry commit was falsified by the
served identity gate on int8 (0 of 12 pages byte-identical in both request
orders, including one page that returned an empty response), while a flag-off
control on the same machine and harness served 12 of 12 — so int8 serves in the
per-graph configuration its published transcripts were taken in, and reloads in
the 3–6 s class (the measured population is the research port's, 3.1–6.5 s;
this package's own int8 reload population has not been measured directly).
Both variants reproduce their pinned transcripts byte for byte
as shipped.

One page per request, `base` mode, image first:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "unlimited-ocr-max", "temperature": 0, "max_tokens": 1766,
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,$(base64 < page.png)"}}]}]}
EOF
```

## Tested on

Apple M4, 24 GB unified memory, macOS 26.5.2, Python 3.12,
`max==26.6.0`. The GPU path needs full Xcode plus the Metal
Toolchain (`xcodebuild -downloadComponent MetalToolchain`); the Command Line
Tools do not ship the Metal compiler MAX shells out to. Greedy output with the
guard off is byte-identical to the fp32 PyTorch reference on both devices. The
server is large for a 24 GB machine; run one at a time and leave it the memory.

## Not supported

`gundam` (tiled) mode — the tiling code is in the package and usable in-process,
but the served batch holds one resolution; batch sizes above 1 (the prefill
graph's sequence length is static); multi-GPU.

## References

The model is described in *Unlimited OCR Works* (Yin et al., 2026),
[arXiv:2606.23050](https://arxiv.org/abs/2606.23050); this package changes the
serving runtime only.

## License and authorship

MIT for this code; the weights, tokenizer and `config.json` are baidu's,
redistributed unchanged under baidu's MIT (both notices in `LICENSE`).
