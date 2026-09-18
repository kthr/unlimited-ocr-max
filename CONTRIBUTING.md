# Contributing

```bash
uv venv && uv pip install -e '.[test]'   # everything, MAX included, comes from PyPI
.venv/bin/pytest tests -q -m "not slow"  # 61 model-free tests, no weights needed
```

CI runs the same tests against the *built wheel* in a throwaway venv on Linux
and macOS, and asserts both `kernels/*.mojo` are inside it.

## Release

1. Bump `project.version` in `pyproject.toml` **and** `DEFAULT_REVISION` in
   `unlimited_ocr_max/cli.py` to the same `vX.Y.Z` (a test ties them together),
   and make sure that tag exists on the model repo.
2. Commit.
3. `git tag -a vX.Y.Z -m "<the release notes>"` — the tag message becomes the
   GitHub Release body.
4. `git push origin main --follow-tags`.
5. `.github/workflows/publish.yml` then checks the tag against
   `project.version`, builds the wheel and sdist, publishes them to PyPI
   through Trusted Publishing, and attaches them to a GitHub Release (created
   with `gh release create` and the workflow's own token -- no third-party
   release action). Nothing is ever published from a developer machine.

## One-time PyPI setup

`unlimited-ocr-max` does not exist on PyPI yet, so register it as a *pending*
trusted publisher: pypi.org → *Your account* → *Publishing* → *Add a new
pending publisher* → GitHub, with

| field | value |
| --- | --- |
| PyPI Project Name | `unlimited-ocr-max` |
| Owner | `kthr` |
| Repository name | `unlimited-ocr-max` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

All five values must match the workflow exactly. The first tag push creates the
project and turns the pending publisher into a real one; the `pypi` environment
is created by GitHub on first use. No API token, no repository secret.

Once that environment exists, add yourself under GitHub → *Settings* →
*Environments* → `pypi` → *Required reviewers*, so every upload to PyPI waits
for a human approval instead of going out on any tag push.

The install instructions need no extra index: the pinned `max[all]==26.6.0` is
on PyPI, so `pip install unlimited-ocr-max` resolves the whole dependency set
from PyPI alone. (Up to and including v0.2.1 the pin was a `26.6.0.dev*`
nightly, which required `--extra-index-url
https://whl.modular.com/nightly/simple/`.)
