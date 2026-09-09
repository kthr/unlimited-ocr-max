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
uv tool install --extra-index-url https://whl.modular.com/nightly/simple/ unlimited-ocr-max
# or, into a venv
pip install --extra-index-url https://whl.modular.com/nightly/simple/ unlimited-ocr-max
```

The extra index is required: the package pins one exact MAX nightly build
(`max[all]==26.6.0.dev2026082707`) because the port depends on fixes no stable
MAX release carries yet, and that build is published only on Modular's nightly
index. The pin and the flag go away with the next stable Modular release.

## Serve

```bash
unlimited-ocr-max serve --devices gpu   # Metal
unlimited-ocr-max serve --devices cpu   # supported, slow
```

This downloads the model repository once (6.2 GiB) and runs `max serve` with
this port's flags, on `http://127.0.0.1:8010` under the model id
`unlimited-ocr-max`.

* `--revision` defaults to `v0.2.0`, the model-repo tag this package version was
  validated against, so a fixed package version serves fixed weights; the tag
  must exist or the download fails before MAX starts. Ignored for a local
  directory.
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
`max==26.6.0.dev2026082707`. The GPU path needs full Xcode plus the Metal
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
