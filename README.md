# unlimited-ocr-max

[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) served through
[MAX](https://docs.modular.com/max/) as an OpenAI-compatible endpoint on Metal,
CUDA or ROCm.
Code: [`github.com/kthr/unlimited-ocr-max`](https://github.com/kthr/unlimited-ocr-max) ·
Weights: [`kthierbach/unlimited-ocr-max`](https://huggingface.co/kthierbach/unlimited-ocr-max).

* **bf16** — baidu's `model.safetensors`, unchanged.
* **int8** — `model-int8.safetensors`, this port's weight-only quantisation of
  the routed experts (symmetric, per-group G=128).
* `config.json` is baidu's minus `auto_map` and `model_type`, so MAX loads it
  without remote code.

## Prerequisites

Python 3.12 or 3.13, plus one of:

### Apple silicon — M1–M5, macOS 15+, 24 GB recommended

```bash
# requires full Xcode (App Store); the Command Line Tools alone are not enough
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
# MAX compiles Metal kernels with `xcrun metallib`, shipped in the Metal Toolchain
xcodebuild -downloadComponent MetalToolchain
xcrun -f metallib          # must print a path
```

### NVIDIA — Linux (glibc 2.34+, e.g. Ubuntu 22.04+), Ampere or newer, driver 580+

Follow NVIDIA's [package-manager installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation);
for example, on Ubuntu 24.04 x86_64:

```bash
# MAX loads these at runtime but does not ship them; without them the
# first request fails with: symbol not found: cublasCreate_v2
# 1. add NVIDIA's CUDA repository (ubuntu2404/x86_64 shown; use your distro/arch)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# 2. libcublas + libcublasLt (one package), CUDA 13
sudo apt-get install -y libcublas-13-0
# 3. check: MAX loads these exact files
ls -l /usr/local/cuda-13.0/lib64/libcublas.so.13 /usr/local/cuda-13.0/lib64/libcublasLt.so.13
```

### AMD — Linux, driver 6.3.3+ (MI355X: ROCm 7+)

MAX also loads ROCm's rocBLAS, hipBLASLt and MIOpen from `/opt/rocm/lib`;
untested beyond compilation.

## Serve

```bash
uv tool install --extra-index-url https://whl.modular.com/nightly/simple/ unlimited-ocr-max
# or: pip install --pre --extra-index-url https://whl.modular.com/nightly/simple/ unlimited-ocr-max
unlimited-ocr-max serve --devices gpu
```

This installs `max[all]==26.7.0.dev2026100105` from Modular's nightly index. The extra index is required: 26.7 has no stable release yet.
Endpoint: `http://127.0.0.1:8010/v1/chat/completions`, model `unlimited-ocr-max`.

| flag | values | default | what it does |
|---|---|---|---|
| `--devices` | `gpu` | **required** | Metal, CUDA or ROCm; `cpu` is refused |
| `--weights` | `bf16` \| `int8` | `bf16` | `int8`: quantised routed experts, faster decode |
| `--model` | Hub repo or local dir | `kthierbach/unlimited-ocr-max` | a local dir needs this repository's layout |
| `--revision` | tag | `v0.4.0` | the model-repo tag this package version was validated against |
| `--port` | integer | `8010` | |
| `--ngram-size` | integer | `35` | no-repeat n-gram guard; `0` disables it |
| `--max-batch-size` | `1`–`8` | `1` | concurrent requests decoded together each step; a lone request runs slower; output is independent of what else is in flight |

A prompt may be at most 512 tokens, the page's image tokens included: 274 in `base` mode, which
leaves about 238 tokens of text. A longer prompt gets an HTTP 400 and never reaches the model.
The server allocates its KV page pool at startup and holds it: 90 MiB at `--max-batch-size 1`,
615 MiB at 8.

### First run

The first run compiles every graph for this build, which can take several minutes. On Apple
silicon the int8 first compile briefly raises the server's physical footprint far above its
steady level -- a transient during compilation; steady state is much lower. Apple M4, with the
graphs not yet cached (one draw each):
- the first bf16 request took 100 s (the server was ready after 29 s);
- the first int8 request took 331 s, almost all of it compiling the int8 language graphs. During
  that compile macOS reported a footprint of about 42 GiB, which counts memory it compresses; the
  resident memory peaked at about 16.6 GiB and swap grew by less than 1 GiB on the 24 GB M4.
  Afterwards the footprint is 9–13 GiB;
- with `--max-batch-size 8`, startup compiles the batched graphs: 212 s (bf16) and 181 s (int8)
  uncached, against 20–31 s cached.

Later runs read the compiled graphs from the cache.

### Measure it on your machine

```bash
unlimited-ocr-max profile --devices gpu            # add --weights int8
unlimited-ocr-max profile --devices gpu --max-batch-size 8   # concurrency defaults to 8
```

Starts its own server, sends the 12 bundled pages, prints one row of the table
below and writes `profile.json`. It refuses to run unless no other
process is using the GPU. `--out DIR` sets the output directory (default
`./unlimited-ocr-max-profile-<UTC time>`). `--concurrency N` (profile-only)
keeps N requests in flight over the corpus instead of one; it defaults to
`--max-batch-size` and must not exceed it.

One page to Markdown:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "unlimited-ocr-max", "temperature": 0, "max_tokens": 1766,
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,$(base64 < page.png | tr -d '\n')"}}]}]}
EOF
```

Offline:

```bash
uvx --from huggingface_hub hf download kthierbach/unlimited-ocr-max --revision v0.4.0 --local-dir ocr-model
unlimited-ocr-max serve --devices gpu --model ocr-model
```

## Where it has run

`temperature 0`, default guard. Text is compared against the fp32 PyTorch
reference (transformers 4.46.3, CPU); CER = edited characters / reference
characters over all pages.

| hardware | weights | status | decode | prefill | memory, peak / steady | text vs reference |
|---|---|---|---|---|---|---|
| Apple M4 24 GB | bf16 | 12 pages | **66.8 tok/s** | 2.12 s | 16.6 / 10.8 GiB | **12/12 byte-identical** |
| Apple M4 24 GB | int8 | 12 pages | **69.3 tok/s** | 3.05 s | 13.0 / 9.9 GiB | 6/12; CER 0.0011, all edits bbox digits |
| NVIDIA A100 80 GB | bf16 | 12 pages | **141.8 tok/s** | 1.94 s | device 10.1 / 10.1 GiB | 10/12; CER 0.022 |
| NVIDIA A100 80 GB | int8 | 12 pages | **123.3 tok/s** | 1.68 s | device 10.4 / 10.4 GiB | 6/12; CER 0.025 |
| AMD gfx90a / gfx942 / gfx950 / gfx1100 | both | compiles, never served | — | — | — | — |

### Continuous batching (`--max-batch-size 8`, Apple M4)

`unlimited-ocr-max profile --devices gpu [--weights int8] --max-batch-size 8 --concurrency N`,
12 pages, one draw each. In every row the text is byte-identical to the same weights at
`--max-batch-size 1`, so bf16 stays 12/12 against the reference and int8 6/12 (CER 0.0011).

| weights | concurrency | aggregate tok/s | per-page latency (median) | decode step (full batch) |
|---|---|---|---|---|
| bf16 | 1 | 35.8 | 14.2 s | 23.9 ms (a lone request, padded to 2 rows) |
| bf16 | 2 | 58.3 | 17.1 s | 23.3 ms |
| bf16 | 4 | 74.8 | 25.0 s | 31.0 ms |
| bf16 | 8 | **85.7** | 40.9 s | 47.6 ms |
| int8 | 1 | 38.4 | 13.3 s | 21.0 ms (a lone request, padded to 2 rows) |
| int8 | 2 | 57.6 | 17.8 s | 21.1 ms |
| int8 | 4 | 72.5 | 25.0 s | 24.9 ms |
| int8 | 8 | **82.2** | 45.7 s | 37.9 ms |

## Not supported

`gundam` (tiled) mode; multi-GPU. A new request's prefill pauses every
running decode -- there is no in-flight batching.

CPU serving was removed: `serve`, `profile` and a plain `max serve` refuse `--devices cpu`.
The MAX 26.7 nightlies miscompile a CPU `reshape(matmul, split N)` plus broadcast, so served
CPU pages came out wrong with no error.

On Apple silicon, MAX does not report a GPU allocation it cannot satisfy. The
computation returns wrong values instead of an error.

## References

*Unlimited OCR Works* (Yin et al., 2026), [arXiv:2606.23050](https://arxiv.org/abs/2606.23050).
This port changes how the model is served, not the model (int8 excepted, as
quantified above); model-level behaviour, limitations and biases are those of
`baidu/Unlimited-OCR`.

```bibtex
@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works},
      author={Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
      year={2026},
      eprint={2606.23050},
      archivePrefix={arXiv}
}
```

## Disclaimer

This port is a research project by Konstantin Thierbach and Claude (Anthropic's
AI model, working in Claude Code). It asks how far one gets migrating a current
model to [MAX](https://docs.modular.com/max/). `baidu/Unlimited-OCR` is the test
case: a SAM and CLIP vision tower feeding a 64-expert mixture-of-experts decoder.
The port rebuilds it as MAX graphs, with custom [Mojo](https://docs.modular.com/mojo/)
kernels where MAX has none, and serves it locally on Apple silicon as well as on
NVIDIA and AMD GPUs.

The code, the measurements and this documentation came out of that collaboration.
Every change is gated against pinned reference transcripts, and every figure comes
from a recorded run. It remains a research artefact rather than a supported
product, though. It tracks MAX's own development closely, at times pinning a
nightly build, and can change or break with it. It is not affiliated with or
endorsed by Baidu or Modular.

## License

MIT — see [`LICENSE`](LICENSE): this port's MIT notice and baidu's verbatim. The
bf16 weights, tokenizer files and `config.json` are baidu's, from
`baidu/Unlimited-OCR` (MIT, Copyright (c) 2026 Baidu), redistributed under that
license; the bf16 weights are unchanged, `config.json` has two keys removed, and
`model-int8.safetensors` is derived from those weights by this port's quantiser.
The `profile` corpus bundles renders of four pages of that paper
(arXiv:2606.23050); see `unlimited_ocr_max/profile_data/NOTICE.md`.
