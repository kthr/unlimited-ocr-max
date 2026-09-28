"""Pure, model-free metrics for the ``unlimited-ocr-max profile`` command.

Turns a served ``max serve`` scheduler log, a set of response texts and a set
of memory samples into the figures the README's benchmark table quotes. Every
function here is stdlib-only and takes no model, no device and no I/O -- it is
the arithmetic layer underneath a later CLI command, not the command itself.

Two rules this module exists to enforce, both learned the hard way in the
research repo (``unlimited-ocr-experiments/CLAUDE.md``, "Reading a benchmark
log here without fooling yourself"):

* **The scheduler logs ``TG batch`` periodically, not per step.** A sampled
  mean is not the distribution's -- quote the floor and the median, never a
  mean, and :func:`decode_stats` does not expose one.
* **Parse ``us|ms|s``.** A parser anchored on ``ms`` silently drops every step
  logged in whole seconds, i.e. exactly the per-request decode-graph reloads on
  an accelerator. :func:`decode_stats` counts them instead of dropping them.
"""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "decode_stats",
    "decode_stats_by_batch",
    "levenshtein",
    "memory_stats",
    "parse_scheduler_log",
    "parse_tg_batch_sizes",
    "prefill_stats",
    "text_stats",
]

#: One scheduler batch line's execution time. ``Executed (CE|TG) batch`` marks
#: the line; the non-greedy ``.*?`` then runs up to the first literal
#: ``Execution:`` on that same line, so an earlier ``Batch creation: 1.02ms``
#: is skipped over rather than matched -- there is nothing in the pattern that
#: names it, so it is never a candidate.
_LINE = re.compile(
    r"Executed (?P<stage>CE|TG) batch\b.*?Execution:\s*(?P<value>[0-9.]+)(?P<unit>us|ms|s)\b"
)

#: Like ``_LINE``, but only ``TG`` lines, and it also captures the batch size MAX logs right on
#: the marker (``Executed TG batch with <N> reqs``) -- a real scheduler log always logs it, but a
#: line missing the field (or the ``Execution:`` field) contributes nothing, same as ``_LINE``.
_TG_BATCH_LINE = re.compile(
    r"Executed TG batch with (?P<batch_size>\d+) reqs\b.*?Execution:\s*(?P<value>[0-9.]+)(?P<unit>us|ms|s)\b"
)

_TO_MS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


def parse_scheduler_log(text: str) -> dict[str, list[tuple[str, float]]]:
    """Every ``Executed CE/TG batch ... Execution:`` line as ``(unit, ms)``, in log order.

    ``unit`` is the unit the scheduler logged (``"us"``, ``"ms"`` or ``"s"``);
    the float is always converted to milliseconds. A line with no ``Execution:``
    field (or no ``Executed CE/TG batch`` marker at all) contributes nothing.
    """
    out: dict[str, list[tuple[str, float]]] = {"CE": [], "TG": []}
    for line in text.splitlines():
        match = _LINE.search(line)
        if match is None:
            continue
        unit = match.group("unit")
        ms = float(match.group("value")) * _TO_MS[unit]
        out[match.group("stage")].append((unit, ms))
    return out


def decode_stats(tg: Sequence[tuple[str, float]]) -> dict:
    """Decode-step floor, median and rate over the *steady* population.

    The steady population is every entry logged in ``us`` or ``ms``; entries
    logged in whole seconds are per-request decode-graph reloads or stalls
    (``unlimited-ocr-experiments/CLAUDE.md``, "The decode KV stays on the
    device") and are excluded from the statistics but counted in
    ``n_whole_second_excluded`` rather than silently dropped.

    Returns ``median_ms``, ``floor_ms`` (the min), ``n`` (steady population
    size), ``tok_s`` (``1000 / median_ms``) and ``n_whole_second_excluded``.
    Never a mean -- the log is a periodic sample of the step distribution, not
    a per-step trace, so a sampled mean is not the distribution's mean.

    Raises ``ValueError`` if the steady population is empty.
    """
    steady = [ms for unit, ms in tg if unit != "s"]
    n_excluded = sum(1 for unit, _ in tg if unit == "s")
    if not steady:
        raise ValueError("no steady (us/ms) TG entries to compute decode stats from")
    median_ms = statistics.median(steady)
    return {
        "median_ms": median_ms,
        "floor_ms": min(steady),
        "n": len(steady),
        "tok_s": 1000.0 / median_ms,
        "n_whole_second_excluded": n_excluded,
    }


def parse_tg_batch_sizes(text: str) -> dict[int, list[tuple[str, float]]]:
    """Every ``Executed TG batch with N reqs ... Execution:`` line, grouped by ``N`` (the batch
    size the scheduler served that step with), each entry as ``(unit, ms)`` in log order -- the
    batch-size-aware sibling of :func:`parse_scheduler_log`, whose own return shape is unchanged
    for its existing callers. A ``TG`` line missing the ``N reqs`` field or the ``Execution:``
    field contributes nothing, the same as :func:`parse_scheduler_log` does for a line missing
    ``Execution:``.
    """
    out: dict[int, list[tuple[str, float]]] = {}
    for line in text.splitlines():
        match = _TG_BATCH_LINE.search(line)
        if match is None:
            continue
        unit = match.group("unit")
        ms = float(match.group("value")) * _TO_MS[unit]
        out.setdefault(int(match.group("batch_size")), []).append((unit, ms))
    return out


def decode_stats_by_batch(text: str) -> dict[int, dict]:
    """:func:`decode_stats` computed separately for every ``TG`` batch size found in ``text`` (a
    scheduler log), keyed by that size -- so a batching server's decode step time can be compared
    batch size by batch size, not just averaged across all of them.

    Whole-second entries are excluded from each batch size's statistics and counted, exactly as
    :func:`decode_stats` does for the un-grouped population. A batch size whose every entry is a
    whole-second reload/stall contributes no key -- its steady population is empty, same as
    :func:`decode_stats` raising ``ValueError`` for that case, just swallowed here per key instead
    of raised for the whole result.
    """
    return {
        batch_size: decode_stats(entries)
        for batch_size, entries in parse_tg_batch_sizes(text).items()
        if any(unit != "s" for unit, _ in entries)
    }


def prefill_stats(ce: Sequence[tuple[str, float]], skip_first: int = 1) -> dict:
    """Prefill-step floor and median, in seconds, over ``ce[skip_first:]``.

    The first ``CE`` entry is the warmup request and is dropped by default
    (``skip_first=1``). Unlike :func:`decode_stats`, every remaining entry
    counts regardless of its logged unit -- prefill has no whole-second
    reload population to exclude.

    Raises ``ValueError`` if nothing remains after the skip.
    """
    population = [ms for _, ms in ce[skip_first:]]
    if not population:
        raise ValueError("no CE entries left after skip_first")
    return {
        "median_s": statistics.median(population) / 1000.0,
        "floor_s": min(population) / 1000.0,
        "n": len(population),
    }


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance, O(len(a)*len(b)) time, O(min(len)) memory.

    A common prefix and suffix are stripped first -- free to detect and, for
    the identical-page case ``text_stats`` already special-cases, it means a
    near-identical page still costs only the length of its actual diff. The DP
    then runs over one row of length ``min(len(a), len(b)) + 1``.
    """
    start = 0
    len_a, len_b = len(a), len(b)
    while start < len_a and start < len_b and a[start] == b[start]:
        start += 1
    end_a, end_b = len_a, len_b
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a = a[start:end_a]
    b = b[start:end_b]
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Iterate the longer string against a row sized by the shorter one.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                current[j] = previous[j - 1]
            else:
                current[j] = 1 + min(previous[j - 1], previous[j], current[j - 1])
        previous = current
    return previous[-1]


def text_stats(refs: dict[str, str], cands: dict[str, str]) -> dict:
    """Identity and character-error-rate over a corpus of pages.

    ``cer`` is ``edits / ref_chars`` -- one denominator over the whole corpus,
    never a mean of each page's own ratio, so a long page's errors are not
    diluted by a short page's near-perfect one. ``pages`` is a per-page detail
    list, in ``refs`` order (Python dicts preserve insertion order); a page
    whose candidate is exactly equal to its reference skips the Levenshtein
    call entirely, since its edit count is known to be 0 either way.

    Raises ``ValueError`` if ``refs`` and ``cands`` do not share the same set
    of keys, or if that shared set is empty (``cer`` has no denominator then).
    """
    if refs.keys() != cands.keys():
        only_ref = sorted(refs.keys() - cands.keys())
        only_cand = sorted(cands.keys() - refs.keys())
        raise ValueError(
            f"refs and cands key sets differ: only in refs={only_ref}, only in cands={only_cand}"
        )
    if not refs:
        raise ValueError("refs and cands are both empty: no pages to compute text stats from")
    identical = 0
    edits = 0
    ref_chars = 0
    pages = []
    for page, ref in refs.items():
        cand = cands[page]
        is_identical = cand == ref
        page_edits = 0 if is_identical else levenshtein(ref, cand)
        identical += is_identical
        edits += page_edits
        ref_chars += len(ref)
        pages.append(
            {
                "page": page,
                "identical": is_identical,
                "edits": page_edits,
                "ref_chars": len(ref),
                "cand_chars": len(cand),
            }
        )
    return {
        "identical": identical,
        "n": len(refs),
        "edits": edits,
        "ref_chars": ref_chars,
        "cer": edits / ref_chars,
        "pages": pages,
    }


def memory_stats(
    samples: Sequence[tuple[float, int]],
    busy: Sequence[tuple[float, float]],
    steady_after: float,
) -> dict:
    """Peak and steady-state memory bounds over a series of ``(t, value)`` samples.

    ``peak`` is the max value over *all* samples (``None`` if ``samples`` is
    empty). A sample is in the steady population iff its timestamp is at or
    after ``steady_after`` and does not fall inside any ``busy`` window
    (``[start, end]``, inclusive on both ends) -- so a memory reading taken
    while a request is in flight is excluded the same way a warmup reading is.
    ``steady_min``/``steady_max`` are ``None`` and ``n_steady`` is 0 when no
    sample qualifies.

    Generic over the sample value's type (host RSS bytes, device bytes, or
    anything else orderable).
    """
    if not samples:
        return {"peak": None, "steady_min": None, "steady_max": None, "n_steady": 0}
    peak = max(value for _, value in samples)
    steady = [
        value
        for t, value in samples
        if t >= steady_after and not any(start <= t <= end for start, end in busy)
    ]
    if not steady:
        return {"peak": peak, "steady_min": None, "steady_max": None, "n_steady": 0}
    return {
        "peak": peak,
        "steady_min": min(steady),
        "steady_max": max(steady),
        "n_steady": len(steady),
    }
