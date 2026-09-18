# Changelog

## [Unreleased] — collecting for v0.3.0

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
