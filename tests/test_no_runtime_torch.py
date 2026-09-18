"""torch is a dev/test-only dependency (KON-194): the declaration and the source sweep.

Two guards, each catching a different way torch could creep back into the runtime
dependency set: the ``pyproject.toml`` declaration itself, and an ``import torch``
statement anywhere under the package.

The third property -- the package *working* with torch not there to import at all
-- cannot be proved from here, because torch is installed in this session (it is
the bit-exactness oracle the conversion tests hold their numpy against). Simulating
its absence only ever proves something about the simulation. It is pinned instead
where the absence is real: the ``The wheel installs and imports without torch`` step
in ``.github/workflows/ci.yml`` installs the wheel with **no extras** into its own
venv and there exercises the weight-free paths -- the package import, the CLI
parser, ``preprocess_page`` and a tokenizer helper.
"""

from __future__ import annotations

import re
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
    """Package-wide: no ``.py`` file under ``unlimited_ocr_max/`` imports torch at module scope."""
    offenders = {
        str(path.relative_to(ROOT)): match.group(0).strip()
        for path in sorted((ROOT / "unlimited_ocr_max").rglob("*.py"))
        for match in [TORCH_IMPORT_RE.search(path.read_text())]
        if match is not None
    }
    assert offenders == {}
