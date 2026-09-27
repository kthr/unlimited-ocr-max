"""The bundled profiling corpus: 12 pinned pages and their reference transcripts.

``unlimited_ocr_max/profile_data/`` ships inside the wheel so ``profile-model`` runs
reproducibly without a checkout of the private research repo. The 12 pages and their
order are pinned by the research repo's ``tests/data/page_hashes.json``; the two
reference variants (``bf16``, ``int8``) are this package's own served output for the
same pages. ``manifest.json`` records the sha256 of every one of those 36 files plus
a short provenance note, and :func:`verify` is what a test (or a caller) uses to
detect a corrupted or substituted copy.

Stdlib only (``importlib.resources`` + ``hashlib`` + ``json``), so importing this
module never pulls in MAX -- same constraint as :mod:`.cli`.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

_PACKAGE = "unlimited_ocr_max"
_DATA_DIR = "profile_data"
VARIANTS = ("bf16", "int8")


def _root(root: str | Path | None) -> Any:
    """The traversable directory to read `profile_data` from.

    Defaults to the packaged resource (works from a wheel, an sdist, or a checkout);
    an explicit `root` points at a plain directory instead, so a test can run the same
    checks against a tampered tmp copy without touching the installed package.
    """
    if root is not None:
        return Path(root)
    return resources.files(_PACKAGE).joinpath(_DATA_DIR)


def load_manifest(root: str | Path | None = None) -> dict[str, Any]:
    """Parse and return `profile_data/manifest.json`."""
    return json.loads(_root(root).joinpath("manifest.json").read_bytes())


def page_names(root: str | Path | None = None) -> list[str]:
    """The 12 page names, in the pinned corpus order."""
    return list(load_manifest(root)["pages"])


def page_png(name: str, root: str | Path | None = None) -> bytes:
    """The raw PNG bytes for one bundled page."""
    return _root(root).joinpath("pages", f"{name}.png").read_bytes()


def reference(name: str, variant: str = "bf16", root: str | Path | None = None) -> str:
    """The pinned reference transcript for one page, decoded as UTF-8 with no bytes added
    or removed -- in particular, no trailing newline is appended if the file has none."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown reference variant {variant!r}; expected one of {VARIANTS}")
    data = _root(root).joinpath("references", variant, f"{name}.md").read_bytes()
    return data.decode("utf-8")


def verify(root: str | Path | None = None) -> None:
    """Check every file `manifest.json` lists against its pinned sha256.

    Raises `ValueError` naming the first file that is missing or does not match.
    """
    base = _root(root)
    manifest = load_manifest(root)
    for rel_path, expected_sha256 in manifest["files"].items():
        target = base.joinpath(*rel_path.split("/"))
        try:
            data = target.read_bytes()
        except FileNotFoundError:
            raise ValueError(f"missing profile_data file: {rel_path}") from None
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"sha256 mismatch for profile_data file: {rel_path}")
