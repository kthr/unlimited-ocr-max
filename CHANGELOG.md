# Changelog

## [Unreleased] — collecting for v0.3.0

### Performance
- **bf16: the language weights are one device copy, shared by both language
  graphs** (device-weight registry, on by default on GPU). Per-request decode
  reload ~10.5 s → **~1.5 s**, TTFT ~8.5 → ~5.3 s, net −7 to −15 s per page on
  the 12-page corpus; steady decode step ~+9 % (a priced tradeoff, net-positive
  on every measured page). Output byte-identical to v0.2.1 on all 12 pages,
  both request orders.

### Behavior
- **MAX pin bumped to `26.6.0.dev2026091105`** (from `26.6.0.dev2026082707`).
  Next-day successor of the dev2026091005 re-probe; validated by the research
  repo's bump smokes: package suite green including the pin guard, the Mojo
  kernels (int8 included) compile and run on Metal, and MAX's filename parser
  still resolves no encoding from our weight filenames.
- **int8 deliberately does not share weights** — serving int8 is unchanged from
  v0.2.1. A served gate showed the shared registry corrupts int8 output
  deterministically while every in-process check is bit-clean; a flag-off
  control confirmed the gate. Evidence: the research repo's EXPERIMENTS.md,
  KON-158 blocks (b)/(c).
- The serve path's graph-release policy is now a named predicate
  (`releases_language_graphs`) and carries the package's first release-policy
  and weight-sharing tests (value-level, not count-level).

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
