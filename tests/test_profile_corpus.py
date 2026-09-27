"""The bundled profiling corpus (KON-202): manifest shape, hash verification, and the
no-trailing-newline guarantee on reference transcripts.

The 12 page names and their order are pinned by the research repo's
``tests/data/page_hashes.json`` (not available from this public checkout, so the
order is hardcoded here rather than re-read from it); what this suite asserts is
that the *shipped* manifest agrees with that pin and that every file it names is
present and hashes correctly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from unlimited_ocr_max import profile_corpus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "unlimited_ocr_max" / "profile_data"

PINNED_PAGE_ORDER = [
    "plain_text",
    "dense_body",
    "toc_dotted",
    "figure_wide",
    "syn_plain_note",
    "syn_dense_prose",
    "syn_dense_numbered",
    "syn_toc_leaders",
    "syn_figure_caption",
    "syn_table_grid",
    "syn_table_ruled",
    "syn_two_column",
]


def test_manifest_lists_the_twelve_pinned_pages_in_order() -> None:
    """Page names and order must match the research corpus's pin, not just be *a* set of 12."""
    assert profile_corpus.page_names() == PINNED_PAGE_ORDER


def test_manifest_names_all_36_files() -> None:
    """12 pages + 12 bf16 references + 12 int8 references, no more, no fewer."""
    manifest = profile_corpus.load_manifest()
    assert len(manifest["pages"]) == 12
    assert len(manifest["files"]) == 36
    for name in PINNED_PAGE_ORDER:
        assert f"pages/{name}.png" in manifest["files"]
        assert f"references/bf16/{name}.md" in manifest["files"]
        assert f"references/int8/{name}.md" in manifest["files"]


def test_every_manifest_file_is_present_on_disk() -> None:
    """Every one of the 36 paths the manifest names actually exists under profile_data/."""
    manifest = profile_corpus.load_manifest()
    for rel_path in manifest["files"]:
        assert (DATA_DIR / rel_path).is_file(), rel_path


def test_page_png_and_reference_are_loadable_for_every_page() -> None:
    """The two loaders resolve every pinned page in both reference variants."""
    for name in profile_corpus.page_names():
        assert len(profile_corpus.page_png(name)) > 0
        assert len(profile_corpus.reference(name, "bf16")) > 0
        assert len(profile_corpus.reference(name, "int8")) > 0


def test_reference_rejects_an_unknown_variant() -> None:
    with pytest.raises(ValueError):
        profile_corpus.reference("plain_text", "fp32")


def test_reference_adds_no_trailing_newline() -> None:
    """The pinned transcripts have no trailing newline; decoding must not add one."""
    for name in profile_corpus.page_names():
        for variant in profile_corpus.VARIANTS:
            raw = (DATA_DIR / "references" / variant / f"{name}.md").read_bytes()
            assert profile_corpus.reference(name, variant) == raw.decode("utf-8")
            assert not raw.endswith(b"\n"), f"{variant}/{name}.md"


def test_verify_passes_on_the_shipped_data() -> None:
    profile_corpus.verify()


def test_verify_raises_naming_the_file_on_a_tampered_copy(tmp_path: Path) -> None:
    """A one-byte corruption in a tmp copy of profile_data must fail verify() by that file's name."""
    tmp_data = tmp_path / "profile_data"
    shutil.copytree(DATA_DIR, tmp_data)

    target = tmp_data / "pages" / "plain_text.png"
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(data)

    with pytest.raises(ValueError, match="pages/plain_text.png"):
        profile_corpus.verify(root=tmp_data)


def test_verify_raises_naming_the_file_on_a_missing_file(tmp_path: Path) -> None:
    """A missing file is reported by name, distinct from a hash mismatch."""
    tmp_data = tmp_path / "profile_data"
    shutil.copytree(DATA_DIR, tmp_data)
    (tmp_data / "references" / "int8" / "syn_two_column.md").unlink()

    with pytest.raises(ValueError, match="references/int8/syn_two_column.md"):
        profile_corpus.verify(root=tmp_data)


_COLD_CACHE_NO_NETWORK_MARKERS = (
    "failed to fetch",
    "failed to download",
    "failed to resolve",
    "no such host",
    "network is disabled",
    "could not find a version",
    "requires network",
)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_the_wheel_bundles_the_corpus(tmp_path: Path) -> None:
    """A wheel built with `uv build` contains the manifest, NOTICE.md, and all 36 pinned data
    files, so the corpus (and its attribution) reaches an installed environment and not just
    this checkout.

    Uses a private `UV_CACHE_DIR` under `tmp_path`: this worktree shares uv's default cache with
    every other concurrent agent session on the machine, and a build-environment cache entry
    (e.g. hatchling) can be mid-write from another session's `uv build` at the same moment,
    which otherwise makes this test flaky for a reason that has nothing to do with this package.
    """
    env = dict(os.environ, UV_CACHE_DIR=str(tmp_path / "uv-cache"))
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
        cwd=ROOT, capture_output=True, text=True, timeout=300, env=env,
    )
    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if any(marker in stderr_lower for marker in _COLD_CACHE_NO_NETWORK_MARKERS):
            pytest.skip(
                "uv build needs to fetch its build backend (e.g. hatchling) and could not "
                f"reach the network from a cold UV_CACHE_DIR: stderr={result.stderr!r}"
            )
        assert False, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    wheels = list((tmp_path / "dist").glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
    manifest = profile_corpus.load_manifest()
    assert "unlimited_ocr_max/profile_data/manifest.json" in names
    assert "unlimited_ocr_max/profile_data/NOTICE.md" in names
    for rel_path in manifest["files"]:
        assert f"unlimited_ocr_max/profile_data/{rel_path}" in names, rel_path


def test_notice_names_the_paper_and_the_four_page_files() -> None:
    """NOTICE.md must exist and carry the attribution: the arXiv id and each of the four
    paper-derived page files, so a reader of the shipped data (not just the manifest string)
    can see which pages are the paper's and which are this project's own synthetic pages."""
    notice = (DATA_DIR / "NOTICE.md").read_text()
    assert "2606.23050" in notice
    for name in ("plain_text", "dense_body", "toc_dotted", "figure_wide"):
        assert f"{name}.png" in notice, name


def test_manifest_matches_the_committed_data_bytes() -> None:
    """The manifest's own sha256 values, recomputed here with a second hashlib pass, so a hand-edit
    of manifest.json that drifted from the actual bytes is caught even without a tamper copy."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    for rel_path, expected_sha256 in manifest["files"].items():
        actual = hashlib.sha256((DATA_DIR / rel_path).read_bytes()).hexdigest()
        assert actual == expected_sha256, rel_path
