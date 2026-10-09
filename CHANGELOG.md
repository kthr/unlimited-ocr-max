# Changelog

## [0.4.0] — 2026-10-09

All figures: `unlimited-ocr-max profile`, the 12 bundled pages, one draw each, on an Apple M4
(24 GB) and an NVIDIA A100 80GB PCIe, MAX nightly `26.7.0.dev2026100105`.

### Highlights
- **Decode is 1.5× (A100) to 3× (Apple M4) faster, and prefill is faster too:**

  | | 0.3.2 | 0.4.0 |
  |---|---|---|
  | Apple M4, bf16 decode | 21.3 tok/s | **66.8 tok/s** |
  | Apple M4, int8 decode | 36.6 tok/s | **69.3 tok/s** |
  | NVIDIA A100, bf16 decode | 93.6 tok/s | **141.8 tok/s** |
  | NVIDIA A100, int8 decode | 82.7 tok/s | **123.3 tok/s** |
  | Apple M4, prefill bf16 / int8 | 3.20 / 6.86 s | **2.12 / 3.05 s** |
  | NVIDIA A100, prefill bf16 / int8 | 3.56 / 6.23 s | **1.94 / 1.68 s** |

  The text is unchanged: bf16 is 12/12 byte-identical to the reference on the M4 (10/12 on the
  A100, as before), and int8 matches its pinned transcripts.
- **Serve several requests at once: `--max-batch-size N`** (`serve` and `profile`, up to 8, bf16
  and int8). At 8 concurrent requests the aggregate is 85.7 tok/s on the M4 and 150.7 tok/s on
  the A100 (bf16). Each request's output does not depend on what else is in flight: on the
  bundled pages it is byte-identical to `--max-batch-size 1`. A new request's prefill briefly
  pauses the running decodes. `profile --concurrency N` measures it.

### Changed
- **Install from Modular's nightly index again.** MAX is pinned to `26.7.0.dev2026100105`, so
  installs need `--extra-index-url https://whl.modular.com/nightly/simple/` (see the README). The
  pin moves to the 26.7 stable release once it ships.
- **Prompts are limited to 512 tokens**, the page's image tokens included (274 in `base` mode),
  which leaves about 238 tokens of text. A longer prompt gets an HTTP 400.
- **The KV cache is allocated at startup**: 90 MiB at `--max-batch-size 1`, 615 MiB at 8. A GPU
  that cannot hold it fails at startup instead of at the first request.
- **Startup and first requests:** with `--max-batch-size` above 1 the server compiles the batched
  graphs at startup (20–31 s cached; 3–3.5 minutes the first time on the M4). On a fresh
  install the first bf16 request takes about 100 s on the M4 and the first int8 request about
  5.5 minutes. That int8 compile puts heavy memory pressure on the machine while it runs (on the
  24 GB M4: resident memory peaked at about 16.6 GiB, swap grew by less than 1 GiB). It happens
  once: MAX caches the compiled graphs.
- **Less memory for int8:** server peak footprint on the M4 20.7 → 13.0 GiB; device memory on the
  A100 17.4 → 10.4 GiB.
- **Leave GPU memory free on Apple silicon:** bf16 needs 2.0 GiB free and int8 2.5 GiB. Below that
  MAX returns wrong values without an error.
- **The numerics changed** (new kernels sum in another order), but the served text on the bundled
  pages is as stated above.
- `serve` now passes `--sample-on-host`: MAX's 26.7 nightlies sample much more slowly on Metal.

### Removed
- **CPU serving.** `serve` and `profile` refuse `--devices cpu`. The MAX 26.7 nightlies compute
  part of the model wrongly on CPU without an error, so served CPU pages were wrong. `--devices`
  stays required, with `gpu` (Metal, CUDA or ROCm) as the only value. `profile.json` loses its
  `gpu_guard` key.

### Fixed
- The package imports again with MAX 26.7, which removed `KVCacheInputsInterface`.

## [0.3.2] — 2026-09-28

### Performance
- **bf16 on a GPU: MAX no longer builds an fp32 copy of the weights inside each
  language graph.** A bf16 weight read by an fp32 matmul is compile-folded by
  MAX into an fp32 copy on the device at every `session.load`. The prefill
  graph folded every expert stack this way (its dense 64-expert path), and
  both graphs folded the projections, norms, router and `lm_head`. Two changes
  remove it:
  - prefill's routed experts now run through MAX's `grouped_matmul_ragged`, as
    decode already did, which reads the bf16 stacks directly;
  - the shared weight registry holds the non-expert weights as their exact fp32
    upcast, once, so neither graph folds its own copy.

  Device memory per graph at a 282-token prompt: prefill ~10.3 GiB → 0.002 GiB,
  decode 1.29 → 0.000 GiB, registry 5.47 → 6.11 GiB. Holding both language
  graphs now needs ~7.7 GiB of the M4's 17.8 GiB Metal budget, against ~18.6.

  Served on an Apple M4 (`profile`, 12 pages):

  | | v0.3.1 | v0.3.2 |
  |---|---|---|
  | prefill median | 4.94 s | **3.20 s** |
  | TTFT median | 4.99 s | **3.25 s** |
  | decode step median | 50.7–51.3 ms | **46.8 ms** (three draws) |
  | total time, 12 pages | 391 s | **344 s** |
  | model worker's peak physical footprint (`vmmap`) | 21.2 GB | **15.1 GB** |

  Between requests the worker's footprint is unchanged, at 10.7 GB.

### Behavior
- **bf16 output is byte-identical to v0.3.1 on all 12 bundled pages**, on the
  Apple M4 and on an NVIDIA A100.
  - The fp32-resident weights are bitwise neutral: the prefill logits and eight
    decode-step logits are sha256-identical, and a slow test pins it.
  - The native prefill kernels sum in a different order, so the prefill logits
    are not bitwise v0.3.1's; the served text is identical.
- **CPU serving and `--weights int8` are unchanged.** Their emitted graphs are
  identical to v0.3.1. On the M4, int8 serves 12/12 byte-identical to its
  pinned transcripts.
- **NVIDIA A100** (`profile`, 12 pages, the same host for both versions):

  | | v0.3.1 | v0.3.2 |
  |---|---|---|
  | bf16 prefill median | 5.49 s | **3.56 s** |
  | bf16 decode | 94.0 tok/s | 93.6 tok/s |
  | bf16 device memory, peak / steady | 19.1 / 19.1 GiB | **10.6 / 10.6 GiB** |
  | bf16 text vs reference | 10/12, CER 0.022 | 10/12, CER 0.022 |

  - The two pages that differ from the fp32 reference are CUDA's arithmetic,
    not this release: v0.3.2's output is byte-identical to v0.3.1's on that
    host.
  - int8 on the A100 is 6/12, CER 0.025. It decodes slower than bf16 there
    (82.7 against 93.6 tok/s); its Mojo kernels have not been tuned for CUDA.
  - AMD is not re-validated.

### Changed
- **`profile` reports the server's memory as its physical footprint on macOS.**
  It used to report process-tree RSS. On Apple silicon RSS leaves out every
  Metal allocation (on unified memory, the weights and graphs) and counts the
  checkpoint's clean page cache, so it followed the host's free RAM rather than
  the server: two builds with the same 10.7 GB footprint read 1.2 and 6.3 GiB.
  It now sums `proc_pid_rusage().ri_phys_footprint` over the server's
  processes, the figure `footprint` and `vmmap` print.
  - Device memory still fills the cell on NVIDIA and AMD.
  - `profile.json` keeps RSS as `host_memory`, next to the new
    `host_footprint`.
  - The README's M4 rows are re-measured this way, so their memory figures are
    not comparable with v0.3.1's.

## [0.3.1] — 2026-09-27

### Added
- **`unlimited-ocr-max profile`**: starts its own server, sends a bundled
  12-page corpus, and turns the scheduler log, host/device samples and the
  returned text into one row of the README's "Where it has run" table plus a
  complete `profile.json`. On `--devices gpu` it refuses to run unless the GPU
  is otherwise idle, so a measurement is never diluted by another process's
  load. The corpus (four pages of *Unlimited OCR Works* plus eight synthetic
  pages) ships under `unlimited_ocr_max/profile_data/`, with attribution in
  `profile_data/NOTICE.md`. Measured on this release's Apple M4: bf16 12/12
  pages byte-identical to the reference; int8's text fidelity is unchanged
  from the last measured run; CPU is now measured rather than "not measured".

### Fixed
- **NVIDIA: the `lm_head` projection is split so the GEMV launch fits CUDA's
  grid limit.** v0.3.0 aborts on the first request on every CUDA device with
  `CUDA_ERROR_INVALID_VALUE` — MAX's GEMV dispatcher guards on
  `ceildiv(n, 2) <= MAX_GRID_DIM_Y` but then launches `ceildiv(n, tile_n)`
  blocks in grid-Y with a `tile_n` as low as 1, so this checkpoint's 129280-wide
  vocabulary projection asks for 129280 blocks against the hardware's 65535.
  `unlimited_ocr_max.decoder.Projection` now splits any projection wider than
  65535 into equal-width matmuls concatenated on the last axis, and only when
  `max.driver.accelerator_api() == "cuda"`. `lm_head` is the only projection in
  this model that is wide enough; the others are 64, 1280 and 6848. The split is
  **bitwise identical** to the unsplit matmul — each output column is the same
  dot product of the same two rows, and no accumulation crosses a split boundary
  — which `tests/test_projection_split.py` asserts on the `uint32` bit patterns
  with the cap monkeypatched low enough to force a split. Metal, CPU and AMD
  take the unsplit path unchanged. Upstream defect, not this port's; the guard
  and the launch are in MAX's own dispatcher.

### Behavior
- **torch is no longer a runtime dependency.** It moves from `[project].dependencies`
  to the `test` extra (same `>=2.13.0,<2.15` pin), where it now serves only as the
  bit-exactness oracle `tests/test_bf16.py` checks numpy's fp32↔bf16 round-trip
  against, and as the fixture builder for the weight-adapter tests. `pip install
  unlimited-ocr-max` no longer resolves torch — on Linux that also means no
  `nvidia-*` CUDA wheels, which torch pulls in as its own dependencies regardless
  of whether a GPU is present: several gigabytes an installing user was paying for
  a library this port's runtime never touched. The dependency declaration is
  catching up with the code: `unlimited_ocr_max/bf16.py` (new in this release) does the
  fp32↔bf16 conversion in numpy, bit-exact against torch per its own test, and the
  checkpoint-tensor path already moved onto `max.driver.Buffer`; between them
  nothing left in `unlimited_ocr_max/` imports torch. Guarded by
  `tests/test_no_runtime_torch.py` — the `pyproject.toml` declaration and a
  package-wide sweep for `import torch`/`from torch` — and, in CI, by a step that
  installs the wheel with **no extras** and there exercises the package import,
  the CLI parser, `preprocess_page` and a tokenizer helper: the property is proved
  where torch is genuinely absent rather than merely unimported.
- **A SAM position table that does not match the target resolution is now an
  error instead of being resampled.** `sam_state_dict` — and so
  `vision_state_dict`, which wraps it — raises `ValueError`, naming both the shape
  found and the shape needed, when `pos_embed` or a global block's
  `rel_pos_h`/`rel_pos_w` arrives at the wrong grid. It previously interpolated
  the table on the host instead (torch bicubic with antialias for `pos_embed`,
  torch linear for the relative tables). That resampling was a gundam-mode
  requirement this package does not serve, and it was unreachable for everything
  it does: at the 1024px grid this package serves, the checkpoint already stores
  `pos_embed` as `[1, 64, 64, 768]` and the relative tables at 127 (global) and 27
  (windowed) rows, which is exactly what the graph declares. So no conversion that
  ever ran changes — a silent branch that never fired became a loud one — and two
  more torch call sites left the weight path with it. `ValueError` rather than
  `WeightMappingError` because `sam_vit` is imported *by* `weight_adapters`.
- **`weight_adapters.load_checkpoint` is gone** — an exported (`__all__`) function
  that read a safetensors shard into torch tensors and had no callers anywhere in
  the shipped package.
### Documentation
- **The CUDA math libraries are a prerequisite, and they are not installed by
  `pip`.** MAX binds `libcublas` and `libcublasLt` at runtime but neither ships
  nor declares them, so on a machine without them a serve aborts at the first
  request with `symbol not found: cublasCreate_v2`. Only `libcublas-13-0` is
  needed: MAX loads the versioned sonames (`libcublas.so.13`,
  `libcublasLt.so.13`) via `/usr/local/cuda-13.0/lib64`, which that package
  ships; `nvrtc` is not referenced anywhere in the wheels, so `cuda-nvrtc-13-0`
  is no longer in the install line. README and the model card say to install
  it as a system library (NVIDIA's CUDA apt repository) rather than as a pip
  wheel, which lands in `site-packages/nvidia/` where the dynamic loader does
  not look.
- **AMD's ROCm libraries are documented too**: MAX loads rocBLAS, hipBLASLt
  and MIOpen from `/opt/rocm/lib`, the same way it loads cuBLAS on NVIDIA.
- **The README now records what has been run on which hardware**, with the
  quantisation and the measured decode rate, prefill and text-fidelity figures
  per row, rather than describing the Apple machine alone.

## [0.3.0] — 2026-09-18

### Performance
- **bf16: the language weights are one device copy, shared by both language
  graphs** (device-weight registry, on by default on GPU; CPU serving and
  `--weights int8` are unaffected). Per-request decode reload ~10.5 s →
  **~1.5 s** — two populations, stated as such: the before is this package's
  own measured pre-registry median (10.56 s); the after is the research port's
  directly measured post-registry class for the same mechanism, this package's
  own post-registry evidence being that the reload no longer reaches the
  scheduler's ~3 s logging tick (i.e. < ~3 s). TTFT ~8.5 → ~5.3 s, net −7 to
  −15 s per page on the 12-page corpus (per-page wall-clock deltas, measured
  directly — not derived from the reload pair); steady decode step ~+9 % (a
  priced tradeoff, net-positive on every
  measured page). The registry is built once per process, on the first request
  that builds a language graph, and survives every graph release — it is not
  rebuilt per request. Output byte-identical to v0.2.1 on all 12 pages, both
  request orders.

### Behavior
- **MAX pin is now the `26.6.0` stable release, and the nightly index is gone.**
  Over this cycle the pin went `26.6.0.dev2026082707` (what v0.2.1 shipped) →
  `26.6.0.dev2026091105` → the `26.6.0` release itself. That release, and the
  `mojo` 1.1.0 toolchain `max[all]` pulls with it, are on PyPI, so the
  `[[tool.uv.index]]` block, CI's extra index and the `--extra-index-url` /
  `--pre` in the install instructions are all gone: `pip install
  unlimited-ocr-max` now resolves from PyPI alone. Resolving the outgoing and
  the incoming pin on one day, inputs otherwise identical, moves the eight
  Modular distributions and nothing else — except seven transitive dependencies
  (`pydantic`, `pydantic-core`, `tokenizers`, `safetensors`, `networkx`,
  `cyclopts`, `hf-xet`) that land on stable versions instead of the
  pre-release-driven versions the dev pin's `--prerelease allow` admitted
  (six were themselves pre-releases; `pydantic-core` moved because the
  pre-release `pydantic` pins it). Validated at the
  stable in a venv rebuilt from scratch: the 61 model-free tests pass — the pin
  guard and MAX's weight-filename parser among them — and so do all 25 `slow`
  tests, which compile the Mojo kernels (int8 included) through a MAX
  `InferenceSession` on Metal; run again with the kernels on CPU, 23 of them
  pass and the two GPU-only real-shape smokes skip. The research port's full
  gate suite and its byte-identical page transcripts are green on
  `max==26.6.0` as well.
- **int8 deliberately does not share weights** — serving int8 is unchanged from
  v0.2.1. A served gate showed the shared registry corrupts int8 output
  deterministically while every in-process check is bit-clean; a flag-off
  control confirmed the gate. Evidence: the research repo's EXPERIMENTS.md,
  KON-158 blocks (b)/(c).
- The serve path's graph-release policy is now a named predicate
  (`releases_language_graphs`), which the served prefill reads instead of the
  raw device. The policy itself — one language graph at a time on an
  accelerator — is unchanged from v0.2.1 and is deliberately *not* gated by the
  weight sharing above. It carries the package's first release-policy and
  weight-sharing tests (value-level, not count-level).

### Documentation
- README: `--revision`'s stated default was `v0.2.0` while `cli.DEFAULT_REVISION`
  has been `v0.2.1` since that release; the claim now names `DEFAULT_REVISION`
  as its source.

## [0.2.1] — 2026-09-10
- Fix: serving from a Hub id downloads the tokenizer at the pinned revision
  (v0.1.0/v0.2.0 required a local checkout). README: `pip install --pre`.

## [0.2.0] — 2026-09-09 (PyPI release deleted; superseded by 0.2.1)
- int8 weight-only variant (`--weights int8`): symmetric per-group G=128,
  routed experts only, dequant inside Mojo custom ops. ~2.5 GiB lower peak RSS,
  ~1.76× decode vs bf16, text byte-identical (bbox ±1–2 px).

## [0.1.0] — 2026-09-06
- Initial release: Unlimited-OCR as a MAX custom architecture on Apple-silicon
  GPUs, OpenAI-compatible serving via `unlimited-ocr-max serve`.
