"""torch is a dev/test-only dependency (KON-194): declaration, source, and a live runtime check.

Three independent guards, each catching a different way torch could creep back
into the runtime dependency set: the ``pyproject.toml`` declaration itself, an
``import torch`` statement anywhere under the package, and — the one that
actually matters for an installing user — the package working when torch is
not there to import at all.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_IMPORT_RE = re.compile(r"^\s*(import torch\b|from torch\b)", re.MULTILINE)


def test_pyproject_declares_no_runtime_torch() -> None:
    """``[project].dependencies`` must not name torch; it belongs to the ``test`` extra only."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = data["project"]["dependencies"]
    assert not any(re.match(r"^torch\b", dep) for dep in dependencies), dependencies

    test_extra = data["project"]["optional-dependencies"]["test"]
    assert any(re.match(r"^torch\b", dep) for dep in test_extra), test_extra


def test_no_torch_imports_under_the_package() -> None:
    """Package-wide: no ``.py`` file under ``unlimited_ocr_max/`` imports torch at module scope.

    ``tests/test_batch_processor.py::test_batch_processor_source_has_no_torch_import`` already
    pins this for the one module where a torch import would be most tempting (the pixel
    round-trip); this is the same check swept over every source file in the package.
    """
    offenders = {
        str(path.relative_to(ROOT)): match.group(0).strip()
        for path in sorted((ROOT / "unlimited_ocr_max").rglob("*.py"))
        for match in [TORCH_IMPORT_RE.search(path.read_text())]
        if match is not None
    }
    assert offenders == {}


# Exercises every weight-free path with torch made unavailable, inside a throwaway subprocess.
_GUARD_SCRIPT = """
import sys

# A plain ``"torch" not in sys.modules`` check does not prove torch is unneeded: MAX's own
# dependencies (transformers.utils.is_torch_available, called at import time from deep inside
# max.pipelines) probe for torch opportunistically and would put it in sys.modules through no
# fault of this package, without this package ever needing it.
#
# The property this guard proves instead is: does the code still work with torch genuinely
# gone? That rules out a meta_path finder whose find_spec() *raises* for "torch" -- verified
# directly against a real torch-free venv, is_torch_available() calls
# importlib.util.find_spec("torch"), which there returns None with no exception, and the whole
# exercise below succeeds cleanly. A finder that raises inside find_spec() breaks that probe
# (find_spec() is contractually a "None on absence" API, not a "raise on absence" one) and
# reports a false failure that does not occur for real -- confirmed by running the identical
# exercise in a fresh venv built with `uv pip install -e .` (no torch resolved at all).
#
# sys.modules["torch"] = None is CPython's own documented "this name resolves to nothing"
# sentinel (see importlib._bootstrap._find_and_load): find_spec("torch") returns None (so
# probing code sees a clean absence, matching genuine absence), while an actual
# `import torch` / `from torch import x` raises ModuleNotFoundError (matching genuine absence's
# exception type). Every `torch.<sub>` import fails the same way, because Python always
# resolves the parent package first.
sys.modules["torch"] = None

# (1) the package import
import unlimited_ocr_max  # noqa: F401

# (2) the CLI argument parser
from unlimited_ocr_max.cli import build_parser

args = build_parser().parse_args(["serve", "--devices", "cpu"])
assert args.command == "serve"

# (3) batch_processor.preprocess_page on a small synthetic image
from PIL import Image

from unlimited_ocr_max.batch_processor import preprocess_page

image = Image.new("RGB", (48, 32), color=(10, 20, 30))
pixels = preprocess_page(image, base_size=64)
assert pixels.shape == (1, 3, 64, 64), pixels.shape

# (4) a tokenizer helper that needs no network
from unlimited_ocr_max.tokenizer import build_prompt

prompt = build_prompt(lambda text: [7] * len(text.split()), prompt="<image>hello world")
assert prompt.seq_len > 0

assert sys.modules["torch"] is None  # never really got imported
print("GUARD_OK")
"""


def test_package_works_with_torch_unavailable() -> None:
    """Subprocess guard: make torch unimportable, then exercise every weight-free path.

    Runs in a subprocess so the blocking sentinel never leaks into the rest of the test session
    (other tests, ``tests/test_bf16.py`` and the weight-adapter fixtures among them, need torch
    for real). This runs correctly in CI, where torch is installed as a test dependency: what it
    proves is that nothing exercised here reaches for torch even though it is available.
    """
    result = subprocess.run(
        [sys.executable, "-c", _GUARD_SCRIPT],
        capture_output=True, text=True, timeout=120, env=dict(os.environ),
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "GUARD_OK" in result.stdout
