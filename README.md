# unlimited-ocr-max

[`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) served through
[MAX](https://docs.modular.com/max/) as an OpenAI-compatible endpoint on Metal,
CUDA, ROCm or CPU. Weights: [`kthierbach/unlimited-ocr-max`](https://huggingface.co/kthierbach/unlimited-ocr-max).

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

Use these system packages, not the `nvidia-*-cu13` pip wheels — the loader does
not find those. NVIDIA needs **v0.3.1 or later**: v0.3.0 aborts on the first
request with `CUDA_ERROR_INVALID_VALUE`.

### AMD — Linux, driver 6.3.3+ (MI355X: ROCm 7+)

MAX also loads ROCm's rocBLAS, hipBLASLt and MIOpen from `/opt/rocm/lib`;
untested beyond compilation.

### CPU

Nothing beyond Python. Supported, slow.

## Serve

```bash
uv tool install unlimited-ocr-max        # or: pip install unlimited-ocr-max
unlimited-ocr-max serve --devices gpu
```

This installs `max[all]==26.6.0`, `numpy` and `pillow` from PyPI (no extra
index). The first run downloads the weights (6.2 GiB bf16, 4.0 GiB int8) and
compiles the kernels. Endpoint: `http://127.0.0.1:8010/v1/chat/completions`, model
`unlimited-ocr-max`.

| flag | values | default | what it does |
|---|---|---|---|
| `--devices` | `gpu` \| `cpu` | **required** | `gpu` is Metal, CUDA or ROCm; `cpu` is slow |
| `--weights` | `bf16` \| `int8` | `bf16` | `int8`: quantised routed experts, faster decode, **gpu only** |
| `--model` | Hub repo or local dir | `kthierbach/unlimited-ocr-max` | a local dir needs this repository's layout |
| `--revision` | tag | `v0.3.3` | the model-repo tag this package version was validated against |
| `--port` | integer | `8010` | |
| `--ngram-size` | integer | `35` | no-repeat n-gram guard; `0` disables it |

### Measure it on your machine

```bash
unlimited-ocr-max profile --devices gpu            # add --weights int8, or --devices cpu
```

Starts its own server, sends the 12 bundled pages, prints one row of the table
below and writes `profile.json`. On a GPU it refuses to run unless no other
process is using the GPU. `--out DIR` sets the output directory (default
`./unlimited-ocr-max-profile-<UTC time>`).

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
uvx --from huggingface_hub hf download kthierbach/unlimited-ocr-max --revision v0.3.3 --local-dir ocr-model
unlimited-ocr-max serve --devices gpu --model ocr-model
```

## Where it has run

`temperature 0`, default guard. Text is compared against the fp32 PyTorch
reference (transformers 4.46.3, CPU); CER = edited characters / reference
characters over all pages.

| hardware | weights | status | decode | prefill | memory, peak / steady | text vs reference |
|---|---|---|---|---|---|---|
| Apple M4 24 GB | bf16 | 12 pages | **21.2 tok/s** | 2.45 s | 15.6 / 11.6–13.6 GiB | **12/12 byte-identical** |
| Apple M4 24 GB | int8 | 12 pages | **36.4 tok/s** | 6.79 s | 13.5 / 11.1 GiB | 6/12; CER 0.0011, all edits bbox digits |
| NVIDIA A100 80 GB | bf16 | re-run pending | — | — | — | — |
| NVIDIA A100 80 GB | int8 | re-run pending | — | — | — | — |
| NVIDIA T4 (Turing, sm_75) | any | **does not run** ¹ | — | — | — | — |
| AMD gfx90a / gfx942 / gfx950 / gfx1100 | both | compiles, never served | — | — | — | — |
| CPU (Apple M4) | bf16 | 12 pages ² | 5.1 tok/s | 47.9 s | 21.1 / 21.0 GiB | **12/12 byte-identical** |

Every measured row is one draw with `unlimited-ocr-max profile` on v0.3.3
(Apple M4: macOS 26.5.2; A100: Linux; `max` 26.6.0 on both), except the CPU
row, which is v0.3.2's: the CPU path is unchanged in v0.3.3. Memory is the server's **physical
footprint** on Apple silicon -- what `footprint` and `vmmap` report, which on
unified memory includes the Metal allocations and excludes clean page cache --
and its device memory on NVIDIA/AMD. Peak is reached in the first request
(weight upload and graph compile); steady is the server idle between pages.
¹ Upstream: MAX's `ldmatrix` PTX needs sm_80, and Turing has no bf16 tensor
cores ([modular/modular#6653](https://github.com/modular/modular/issues/6653),
[#6659](https://github.com/modular/modular/issues/6659)). ² System swap grew
during these runs, so their timings are indicative; the text results are
exact. ³ On CUDA two of the twelve pages differ from the fp32 reference. The
bf16 output is byte-identical to v0.3.1's on the same A100, so this is CUDA's
arithmetic rather than a change in this release.

## Not supported

`gundam` (tiled) mode; batch size > 1; multi-GPU.

## References

*Unlimited OCR Works* (Yin et al., 2026), [arXiv:2606.23050](https://arxiv.org/abs/2606.23050).
This port changes how the model is served, not the model (int8 excepted, as
quantified above); model-level behaviour, limitations and biases are those of
`baidu/Unlimited-OCR`.

## Authorship

This port, its serve configuration and this documentation were written with
substantial AI assistance (Claude, via Claude Code); Konstantin Thierbach
reviewed them and is accountable for their contents. Assisted-by: AI.

## License

MIT — see [`LICENSE`](LICENSE): this port's MIT notice and baidu's verbatim. The
bf16 weights, tokenizer files and `config.json` are baidu's, from
`baidu/Unlimited-OCR` (MIT, Copyright (c) 2026 Baidu), redistributed under that
license; the bf16 weights are unchanged, `config.json` has two keys removed, and
`model-int8.safetensors` is derived from those weights by this port's quantiser.
The `profile` corpus bundles renders of four pages of that paper
(arXiv:2606.23050); see `unlimited_ocr_max/profile_data/NOTICE.md`.
