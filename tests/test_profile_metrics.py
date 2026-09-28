"""Model-free gates for ``unlimited_ocr_max.profile_metrics``.

Every real log line below is copied verbatim (as a string literal, not read
from a file) from ``grep -m3 "Executed TG batch"`` and the first
``Executed CE batch`` line of
``unlimited-ocr-experiments/.scratch/serve/kon180-cpu/serve.log`` -- a CPU
serve, so its ``TG`` lines carry no per-request decode-graph reload and its
first ``TG`` step (20.52s) is instead the cold-compile stall the reference log
happens to start with. ``CE`` is 11.31s; the first ``TG`` after it is 20.52s,
a whole-second (excluded) entry.
"""

from __future__ import annotations

import pytest

from unlimited_ocr_max.profile_metrics import (
    decode_stats,
    decode_stats_by_batch,
    levenshtein,
    memory_stats,
    parse_scheduler_log,
    parse_tg_batch_sizes,
    prefill_stats,
    text_stats,
)

# Real, verbatim lines from kon180-cpu/serve.log.
_CE_LINE = (
    "11:06:35.896 INFO: Executed CE batch with 1 reqs | Terminated: 0 reqs, "
    "Pending: 0 reqs | Input Tokens: 282/8192 toks | Prompt Tput: 24.9 tok/s, "
    "Generation Tput: 0.1 tok/s | Batch creation: 1.02ms, Execution: 11.31s | "
    "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
)
_TG_LINE_S = (
    "11:06:56.422 INFO: Executed TG batch with 1 reqs | Terminated: 0 reqs, "
    "Pending: 0 reqs | Input Tokens: 1/8192 toks | Prompt Tput: 0.0 tok/s, "
    "Generation Tput: 0.0 tok/s | Batch creation: 391.71us, Execution: 20.52s | "
    "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
)
_TG_LINE_S2 = (
    "11:07:12.171 INFO: Executed TG batch with 1 reqs | Terminated: 0 reqs, "
    "Pending: 0 reqs | Input Tokens: 1/8192 toks | Prompt Tput: 0.4 tok/s, "
    "Generation Tput: 0.4 tok/s | Batch creation: 129.08us, Execution: 2.32s | "
    "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
)
_TG_LINE_MS = (
    "11:07:15.555 INFO: Executed TG batch with 1 reqs | Terminated: 0 reqs, "
    "Pending: 0 reqs | Input Tokens: 1/8192 toks | Prompt Tput: 2.5 tok/s, "
    "Generation Tput: 2.5 tok/s | Batch creation: 109.25us, Execution: 398.94ms | "
    "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
)
# kon180-cpu/serve.log has no TG line logged in `us` at all (415 TG lines,
# all ms or s -- a CPU serve has no per-request reload to log as a fast us
# outlier either). This line is `_TG_LINE_MS` with its Execution field only
# edited from "398.94ms" to "850.00us", to exercise the third unit.
_TG_LINE_US = (
    "11:07:15.555 INFO: Executed TG batch with 1 reqs | Terminated: 0 reqs, "
    "Pending: 0 reqs | Input Tokens: 1/8192 toks | Prompt Tput: 2.5 tok/s, "
    "Generation Tput: 2.5 tok/s | Batch creation: 109.25us, Execution: 850.00us | "
    "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
)


def test_unit_conversion_for_all_three_units() -> None:
    """us/ms/s all convert to milliseconds, and stage separation is exact."""
    text = "\n".join([_CE_LINE, _TG_LINE_S, _TG_LINE_MS, _TG_LINE_US])
    parsed = parse_scheduler_log(text)
    assert parsed["CE"] == [("s", 11310.0)]
    assert parsed["TG"] == [
        ("s", 20520.0),
        ("ms", 398.94),
        ("us", 0.85),
    ]


def test_lines_are_matched_in_log_order() -> None:
    """The three real TG lines come back in the order they were logged."""
    text = "\n".join([_TG_LINE_S, _TG_LINE_S2, _TG_LINE_MS])
    parsed = parse_scheduler_log(text)
    assert [unit for unit, _ in parsed["TG"]] == ["s", "s", "ms"]
    assert [ms for _, ms in parsed["TG"]] == [20520.0, 2320.0, 398.94]


def test_batch_creation_is_never_captured() -> None:
    """`Batch creation:` precedes `Execution:` on every real line and must lose.

    Every real line above pairs a `Batch creation:` value in us or ms with a
    *different* `Execution:` value -- if the regex ever matched the wrong
    field, the captured number and unit would betray it immediately.
    """
    parsed = parse_scheduler_log(_TG_LINE_S)
    # Batch creation was "391.71us" on this line; Execution was "20.52s".
    assert parsed["TG"] == [("s", 20520.0)]

    parsed = parse_scheduler_log(_CE_LINE)
    # Batch creation was "1.02ms" on this line; Execution was "11.31s".
    assert parsed["CE"] == [("s", 11310.0)]


def test_a_line_with_no_execution_field_is_ignored() -> None:
    """A batch marker with no `Execution:` on the line contributes nothing."""
    line = "12:00:00.000 INFO: Executed TG batch with 1 reqs | no execution field here"
    assert parse_scheduler_log(line) == {"CE": [], "TG": []}


def test_decode_stats_excludes_and_counts_whole_second_entries() -> None:
    """Whole-second TG lines (reloads/stalls) are excluded from the stats and counted separately."""
    # Values are already in milliseconds (as `parse_scheduler_log` returns
    # them); only the unit tag decides steady-vs-excluded.
    tg = [("s", 20520.0), ("ms", 50.0), ("ms", 40.0), ("us", 45.0), ("s", 2320.0)]
    stats = decode_stats(tg)
    assert stats["n"] == 3
    assert stats["n_whole_second_excluded"] == 2
    assert stats["floor_ms"] == 40.0
    # steady population: [50.0, 40.0, 45.0] -> median 45.0
    assert stats["median_ms"] == 45.0
    assert stats["tok_s"] == pytest.approx(1000.0 / 45.0)


def test_decode_stats_never_reports_a_mean() -> None:
    """The returned dict has no mean-shaped key at all."""
    stats = decode_stats([("ms", 10.0), ("ms", 1000.0)])
    assert set(stats) == {"median_ms", "floor_ms", "n", "tok_s", "n_whole_second_excluded"}


def test_decode_stats_raises_on_empty_steady_population() -> None:
    """All-whole-second input (every step a reload/stall) has no steady population to summarize."""
    with pytest.raises(ValueError):
        decode_stats([("s", 20.0), ("s", 21.0)])


def test_decode_stats_raises_on_totally_empty_input() -> None:
    """No TG lines at all is also an empty steady population."""
    with pytest.raises(ValueError):
        decode_stats([])


def test_oracle_kon180_tg_values_are_all_steady() -> None:
    """The three real TG lines are s/s/ms -- no `us` line exists in that log (see the module comment)."""
    text = "\n".join([_TG_LINE_S, _TG_LINE_S2, _TG_LINE_MS])
    tg = parse_scheduler_log(text)["TG"]
    stats = decode_stats(tg)
    assert stats["n"] == 1
    assert stats["n_whole_second_excluded"] == 2
    assert stats["floor_ms"] == stats["median_ms"] == 398.94


# --------------------------------------------------------------------------- #
# decode stats grouped by TG batch size (KON-216)
# --------------------------------------------------------------------------- #
def _tg_line(batch_size: int, execution: str, creation: str = "109.25us") -> str:
    """A synthetic (not verbatim) ``Executed TG batch`` line, only ``batch_size`` and
    ``Execution:`` varied -- same shape as the real ``_TG_LINE_*`` fixtures above."""
    return (
        f"11:07:15.555 INFO: Executed TG batch with {batch_size} reqs | Terminated: 0 reqs, "
        f"Pending: 0 reqs | Input Tokens: {batch_size}/8192 toks | Prompt Tput: 2.5 tok/s, "
        f"Generation Tput: 2.5 tok/s | Batch creation: {creation}, Execution: {execution} | "
        "KVCache usage: 18.8% of 16 blocks | All Preemptions: 0 reqs"
    )


def test_parse_tg_batch_sizes_groups_by_batch_size_across_all_three_units() -> None:
    text = "\n".join([
        _tg_line(1, "50.00ms"),
        _tg_line(4, "80.00ms"),
        _tg_line(1, "45.00ms"),
        _tg_line(4, "0.85ms"),  # exercises ms alongside batch 4's other entry
        _tg_line(4, "900.00us"),
        _tg_line(1, "20.00s"),  # a per-request reload, still grouped under its own batch size
    ])
    parsed = parse_tg_batch_sizes(text)
    assert parsed == {
        1: [("ms", 50.0), ("ms", 45.0), ("s", 20000.0)],
        4: [("ms", 80.0), ("ms", 0.85), ("us", 0.9)],
    }


def test_parse_tg_batch_sizes_ignores_ce_lines_and_lines_missing_the_field() -> None:
    """A CE line never carries ``Executed TG batch``; a line with no ``N reqs`` after the marker,
    or no ``Execution:`` field, contributes nothing -- same as ``parse_scheduler_log``."""
    no_batch_field = "11:07:15.555 INFO: Executed TG batch | Execution: 50.00ms"
    no_execution_field = "11:07:15.555 INFO: Executed TG batch with 1 reqs | no execution field here"
    text = "\n".join([_CE_LINE, no_batch_field, no_execution_field, _tg_line(2, "10.00ms")])
    assert parse_tg_batch_sizes(text) == {2: [("ms", 10.0)]}


def test_decode_stats_by_batch_excludes_whole_second_entries_per_batch_size() -> None:
    """Each batch size gets its own steady population and its own excluded count, exactly as
    ``decode_stats`` computes for the un-grouped population."""
    text = "\n".join([
        _tg_line(1, "50.00ms"), _tg_line(1, "40.00ms"), _tg_line(1, "20.00s"),
        _tg_line(4, "80.00ms"), _tg_line(4, "90.00ms"),
    ])
    stats = decode_stats_by_batch(text)
    assert stats[1] == {"median_ms": 45.0, "floor_ms": 40.0, "n": 2, "tok_s": pytest.approx(1000.0 / 45.0),
                        "n_whole_second_excluded": 1}
    assert stats[4] == {"median_ms": 85.0, "floor_ms": 80.0, "n": 2, "tok_s": pytest.approx(1000.0 / 85.0),
                        "n_whole_second_excluded": 0}


def test_decode_stats_by_batch_drops_a_batch_size_with_no_steady_entries() -> None:
    """A batch size logged only as whole-second reload/stall entries has no steady population to
    summarize -- ``decode_stats`` would raise for it alone; grouped, it is just left out."""
    text = "\n".join([_tg_line(1, "50.00ms"), _tg_line(8, "20.00s"), _tg_line(8, "21.00s")])
    stats = decode_stats_by_batch(text)
    assert set(stats) == {1}
    assert stats[1]["n"] == 1


def test_prefill_stats_skips_the_warmup_request_by_default() -> None:
    """`ce[0]` (the warmup) is excluded; every later CE entry counts regardless of unit.

    Values are already in milliseconds (as `parse_scheduler_log` returns them);
    the unit tag itself is not consulted by `prefill_stats`.
    """
    ce = [("s", 98900.0), ("s", 8190.0), ("ms", 8260.0), ("us", 8380.0)]
    stats = prefill_stats(ce)
    assert stats["n"] == 3
    assert stats["floor_s"] == pytest.approx(8.19)
    assert stats["median_s"] == pytest.approx(8.26)


def test_prefill_stats_skip_first_zero_keeps_everything() -> None:
    """`skip_first=0` counts the warmup too -- an explicit opt-out, not a default."""
    ce = [("s", 98900.0), ("s", 8190.0)]
    stats = prefill_stats(ce, skip_first=0)
    assert stats["n"] == 2
    assert stats["floor_s"] == pytest.approx(8.19)


def test_prefill_stats_raises_when_nothing_survives_the_skip() -> None:
    """A single CE entry (only the warmup) leaves nothing to summarize."""
    with pytest.raises(ValueError):
        prefill_stats([("s", 11310.0)])


def test_levenshtein_empty_strings() -> None:
    """Two empty strings are zero edits apart."""
    assert levenshtein("", "") == 0


def test_levenshtein_one_empty_string() -> None:
    """An empty string against `n` characters costs exactly `n` insertions."""
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_kitten_sitting() -> None:
    """The textbook case: substitute k->s, e->i, insert g."""
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_non_ascii() -> None:
    """A non-ASCII, non-BMP-adjacent case: CJK OCR output differing by one character."""
    assert levenshtein("北京市", "北京") == 1  # "北京市" vs "北京"
    assert levenshtein("café", "cafe") == 1  # "café" vs "cafe"


def test_levenshtein_identical_strings_is_zero() -> None:
    """Common-prefix/suffix stripping must not break the trivial identical case."""
    assert levenshtein("same text", "same text") == 0


def test_text_stats_corpus_cer_is_not_the_mean_of_page_ratios() -> None:
    """One denominator over the corpus, not a mean of per-page CERs.

    Page 1: 10 ref chars, 1 edit -> per-page ratio 0.1.
    Page 2: 1000 ref chars, 500 edits -> per-page ratio 0.5.
    Mean of the two ratios is 0.3. The corpus CER -- edits summed over
    ref_chars summed -- is 501/1010, nowhere near 0.3: the long page's much
    higher error rate must not be diluted by the short page's low one.
    """
    refs = {"short": "a" * 10, "long": "b" * 1000}
    cands = {"short": "a" * 9 + "z", "long": "b" * 500 + "z" * 500}
    stats = text_stats(refs, cands)

    per_page_mean = (0.1 + 0.5) / 2
    assert stats["cer"] != pytest.approx(per_page_mean)
    assert stats["cer"] == pytest.approx(501 / 1010)
    assert stats["edits"] == 501
    assert stats["ref_chars"] == 1010
    assert stats["identical"] == 0
    assert stats["n"] == 2


def test_text_stats_identical_pages_skip_levenshtein_and_count_as_identical() -> None:
    """An exact match costs 0 edits and is flagged identical, in `refs` order."""
    refs = {"a": "hello", "b": "world"}
    cands = {"a": "hello", "b": "world!"}
    stats = text_stats(refs, cands)

    assert stats["identical"] == 1
    assert stats["cer"] == pytest.approx(1 / 10)  # one insertion, 10 ref chars total
    assert [p["page"] for p in stats["pages"]] == ["a", "b"]
    assert stats["pages"][0] == {
        "page": "a",
        "identical": True,
        "edits": 0,
        "ref_chars": 5,
        "cand_chars": 5,
    }
    assert stats["pages"][1]["edits"] == 1


def test_text_stats_raises_on_key_mismatch() -> None:
    """refs and cands must name exactly the same pages."""
    with pytest.raises(ValueError):
        text_stats({"a": "x"}, {"b": "x"})
    with pytest.raises(ValueError):
        text_stats({"a": "x", "b": "y"}, {"a": "x"})


def test_text_stats_raises_on_empty_corpus() -> None:
    """No pages means no cer denominator; a clear ValueError, not a ZeroDivisionError."""
    with pytest.raises(ValueError, match="no pages"):
        text_stats({}, {})


def test_memory_stats_excludes_pre_steady_after_samples() -> None:
    """A sample before `steady_after` is out of the steady population, even if it is the peak."""
    samples = [(0.0, 5000), (10.0, 1000), (20.0, 1100)]
    stats = memory_stats(samples, busy=[], steady_after=10.0)
    assert stats["peak"] == 5000  # peak is over ALL samples
    assert stats["steady_min"] == 1000
    assert stats["steady_max"] == 1100
    assert stats["n_steady"] == 2


def test_memory_stats_excludes_in_busy_window_samples() -> None:
    """A sample inside a busy window (inclusive both ends) is excluded even though it is >= steady_after."""
    samples = [(10.0, 1000), (15.0, 9000), (20.0, 1050), (25.0, 1000)]
    busy = [(14.0, 21.0)]  # covers t=15.0 and t=20.0 inclusive
    stats = memory_stats(samples, busy=busy, steady_after=0.0)
    assert stats["n_steady"] == 2
    assert stats["steady_min"] == 1000
    assert stats["steady_max"] == 1000
    assert stats["peak"] == 9000


def test_memory_stats_busy_window_is_inclusive_at_both_ends() -> None:
    """A sample exactly at a busy window's start or end boundary is excluded."""
    samples = [(5.0, 100), (10.0, 200)]
    busy = [(5.0, 10.0)]
    stats = memory_stats(samples, busy=busy, steady_after=0.0)
    assert stats["n_steady"] == 0
    assert stats["steady_min"] is None
    assert stats["steady_max"] is None


def test_memory_stats_no_steady_samples() -> None:
    """Every sample excluded (too early or all busy) reports steady_min/max as None, n_steady 0, but a real peak."""
    samples = [(0.0, 500), (1.0, 600)]
    stats = memory_stats(samples, busy=[], steady_after=100.0)
    assert stats == {"peak": 600, "steady_min": None, "steady_max": None, "n_steady": 0}


def test_memory_stats_no_samples_at_all() -> None:
    """An empty sample list has no peak either."""
    stats = memory_stats([], busy=[], steady_after=0.0)
    assert stats == {"peak": None, "steady_min": None, "steady_max": None, "n_steady": 0}
