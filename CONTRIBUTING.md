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

## One-time PyPI setup — already done

`unlimited-ocr-max` exists on PyPI. The *pending* trusted publisher was
registered before the first release (pypi.org → *Your account* → *Publishing* →
*Add a new pending publisher* → GitHub); the first tag push created the project
(**v0.1.0**) and turned the pending publisher into a real one, and GitHub
created the `pypi` environment on that first use. Both exist now — nothing here
has to be redone. These are the registered values, kept for the case where the
publisher has to be re-created:

| field | value |
| --- | --- |
| PyPI Project Name | `unlimited-ocr-max` |
| Owner | `kthr` |
| Repository name | `unlimited-ocr-max` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

All five values must match the workflow exactly. No API token, no repository
secret.

**Still to do, if it has not been:** add yourself under GitHub → *Settings* →
*Environments* → `pypi` → *Required reviewers*, so every upload to PyPI waits
for a human approval instead of going out on any tag push.

The install instructions need no extra index **from the next release on**: the
pinned `max[all]==26.6.0` is on PyPI, so `pip install unlimited-ocr-max`
resolves the whole dependency set from PyPI alone. That is not yet true of what
is *published* — the latest release on PyPI is **0.2.1**, which still pins
`max[all]==26.6.0.dev2026082707`, a nightly that is not on PyPI, so a plain
`pip install unlimited-ocr-max` today installs 0.2.1 and fails to resolve `max`
without `--extra-index-url https://whl.modular.com/nightly/simple/`.
