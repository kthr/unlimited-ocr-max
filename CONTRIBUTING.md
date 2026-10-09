# Contributing

```bash
uv venv && uv pip install -e '.[test]'   # the nightly index comes from [[tool.uv.index]]
.venv/bin/pytest tests -q -m "not slow"  # 454 model-free tests, no weights needed
```

CI runs the same tests against the *built wheel* in a throwaway venv on Linux
and macOS, and asserts every `kernels/*.mojo` is inside it.

## Release

Every release goes through a pull request, so CI has run on exactly what is merged and tagged.

1. On a branch `release/X.Y.Z`: bump `project.version` in `pyproject.toml` **and**
   `DEFAULT_REVISION` in `unlimited_ocr_max/cli.py` to the same `vX.Y.Z` (a test ties them
   together), date the version's section in `CHANGELOG.md`, and make sure the tag `vX.Y.Z`
   exists on the model repo.
2. Run CI's lint step and the model-free tests locally:
   ```bash
   uvx ruff@0.16.6 check --select F unlimited_ocr_max tests   # the lint step of ci.yml
   .venv/bin/pytest tests -q -m "not slow"
   ```
3. Push the branch and open a pull request into `main`. `.github/workflows/ci.yml` runs the
   lint and the wheel jobs (Linux and macOS) on it.
4. When CI is green, **squash-merge** the pull request; the squash commit message summarises
   the release for users.
5. Tag the squash commit and push the tag -- the tag message becomes the GitHub Release body:
   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "<the release notes>"
   git push origin vX.Y.Z
   ```
6. `.github/workflows/publish.yml` then checks the tag against
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

From **0.4.0** on, the install instructions need `--extra-index-url
https://whl.modular.com/nightly/simple/` again: the pinned `max[all]==26.7.0.dev2026100105`
is a nightly that is not on PyPI. With `pip` they need `--pre` too, for the
pre-release `mojo` it depends on. The pin moves to the 26.7 stable release
once that ships. Releases 0.3.0 to 0.3.2 pin `max[all]==26.6.0`, from PyPI
alone. Releases up to and including **0.2.1** pin
`max[all]==26.6.0.dev2026082707`, which also needs the extra index.
