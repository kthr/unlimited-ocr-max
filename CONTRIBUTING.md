# Contributing

```bash
uv venv && uv pip install -e '.[test]'   # nightly index comes from [[tool.uv.index]]
.venv/bin/pytest tests -q                # 15 model-free tests, no weights needed
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
   through Trusted Publishing, and attaches them to a GitHub Release. Nothing
   is ever published from a developer machine.

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

Being on PyPI does not remove `--extra-index-url
https://whl.modular.com/nightly/simple/` from the install instructions: the
pinned `max[all]==26.6.0.dev2026082707` is published on that index only, until
this port's fixes reach a stable MAX release.
