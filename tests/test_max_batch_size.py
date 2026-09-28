"""``--max-batch-size`` (KON-214): env parsing, the refusal matrix (CLI and model-side), the
architecture's ``required_arguments`` following the env, and that the flag never reaches
``max serve``'s own argv. Model-free: nothing here constructs an ``UnlimitedOCRModel`` or loads a
checkpoint; ``DeviceRef`` is metadata only (see ``test_decoder_int8.py``'s use of it the same way)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from unlimited_ocr_max import cli

# --------------------------------------------------------------------------- #
# env parsing: model.serve_max_batch_size()
# --------------------------------------------------------------------------- #
def test_serve_max_batch_size_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from unlimited_ocr_max.model import MAX_BATCH_CAP, MAX_BATCH_SIZE_ENV, serve_max_batch_size

    assert MAX_BATCH_SIZE_ENV == cli.MAX_BATCH_SIZE_ENV
    assert MAX_BATCH_CAP == cli.MAX_BATCH_CAP

    monkeypatch.delenv(MAX_BATCH_SIZE_ENV, raising=False)
    assert serve_max_batch_size() == 1  # unset

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "")
    assert serve_max_batch_size() == 1  # blank

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "  ")
    assert serve_max_batch_size() == 1  # blank (whitespace only)

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "not-a-number")
    with pytest.raises(ValueError, match=MAX_BATCH_SIZE_ENV):
        serve_max_batch_size()  # garbage

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "0")
    with pytest.raises(ValueError, match=MAX_BATCH_SIZE_ENV):
        serve_max_batch_size()  # < 1

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "-1")
    with pytest.raises(ValueError, match=MAX_BATCH_SIZE_ENV):
        serve_max_batch_size()  # < 1

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "1")
    assert serve_max_batch_size() == 1

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "4")
    assert serve_max_batch_size() == 4  # N


# --------------------------------------------------------------------------- #
# the refusal matrix: CLI side (cli.check_devices_support_variant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("weights", "devices", "max_batch_size", "match"), [
    ("bf16", "cpu", 2, "needs an accelerator"),
    ("int8", "gpu", 2, "bf16-only"),
    ("bf16", "gpu", cli.MAX_BATCH_CAP + 1, "exceeds the cap"),
    ("bf16", "gpu", 0, "must be >= 1"),
    ("bf16", "cpu", -1, "must be >= 1"),
])
def test_cli_refuses_a_max_batch_size_the_configuration_cannot_serve(
    weights: str, devices: str, max_batch_size: int, match: str
) -> None:
    with pytest.raises(SystemExit, match=match):
        cli.check_devices_support_variant(weights, devices, max_batch_size)


@pytest.mark.parametrize(("weights", "devices", "max_batch_size"), [
    ("bf16", "gpu", 1),
    ("int8", "gpu", 1),
    ("bf16", "cpu", 1),
    ("bf16", "gpu", 2),
    ("bf16", "gpu", cli.MAX_BATCH_CAP),
])
def test_cli_allows_a_max_batch_size_the_configuration_can_serve(
    weights: str, devices: str, max_batch_size: int
) -> None:
    cli.check_devices_support_variant(weights, devices, max_batch_size)  # must not raise


# --------------------------------------------------------------------------- #
# the refusal matrix: model side (model.check_max_batch_size)
# --------------------------------------------------------------------------- #
def test_model_refuses_a_max_batch_size_the_configuration_cannot_serve() -> None:
    from max.graph import DeviceRef

    from unlimited_ocr_max.model import MAX_BATCH_CAP, check_max_batch_size

    with pytest.raises(ValueError, match="needs an accelerator"):
        check_max_batch_size(2, device=DeviceRef.CPU(), int8=False)
    with pytest.raises(ValueError, match="bf16-only"):
        check_max_batch_size(2, device=DeviceRef.GPU(0), int8=True)
    with pytest.raises(ValueError, match="exceeds the cap"):
        check_max_batch_size(MAX_BATCH_CAP + 1, device=DeviceRef.GPU(0), int8=False)


def test_model_allows_a_max_batch_size_the_configuration_can_serve() -> None:
    from max.graph import DeviceRef

    from unlimited_ocr_max.model import MAX_BATCH_CAP, check_max_batch_size

    check_max_batch_size(1, device=DeviceRef.CPU(), int8=False)  # batch 1 is always fine
    check_max_batch_size(1, device=DeviceRef.GPU(0), int8=True)
    check_max_batch_size(2, device=DeviceRef.GPU(0), int8=False)
    check_max_batch_size(MAX_BATCH_CAP, device=DeviceRef.GPU(0), int8=False)  # at the cap, not over it


# --------------------------------------------------------------------------- #
# arch.py: required_arguments follows the env
# --------------------------------------------------------------------------- #
def test_arch_required_arguments_follows_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from unlimited_ocr_max import arch
    from unlimited_ocr_max.model import MAX_BATCH_SIZE_ENV

    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, "4")
    try:
        importlib.reload(arch)
        assert arch.unlimited_ocr_arch.required_arguments["max_batch_size"] == 4
    finally:
        monkeypatch.delenv(MAX_BATCH_SIZE_ENV, raising=False)
        importlib.reload(arch)  # restore: the module must not leak the monkeypatched env
    assert arch.unlimited_ocr_arch.required_arguments["max_batch_size"] == 1


# --------------------------------------------------------------------------- #
# the flag never reaches `max serve`'s own argv; serve_env carries it
# --------------------------------------------------------------------------- #
def test_serve_env_carries_max_batch_size() -> None:
    env = cli.serve_env(35, 4)
    assert env[cli.NGRAM_ENV] == "35"
    assert env[cli.MAX_BATCH_SIZE_ENV] == "4"


def test_serve_command_never_carries_max_batch_size(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    model, weight_path, revision = cli.resolve_model(str(tmp_path), "bf16", None)
    baseline = cli.serve_command(
        max_exe="/venv/bin/max", model=model, weight_path=weight_path, devices="gpu", port=8010, revision=revision
    )
    assert "--max-batch-size" not in baseline  # serve_command takes no such argument at all


def test_max_batch_size_reaches_the_env_but_never_the_serve_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``cmd_serve``'s full path: for every N, the built argv is unaffected and never carries the
    flag, while the child's environment does."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(cli, "max_executable", lambda: "/venv/bin/max")
    captured: dict[str, object] = {}

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        captured["argv"] = args
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execvpe", fake_execvpe)
    baseline_argv: list[str] | None = None
    for n in (1, 2, cli.MAX_BATCH_CAP):
        with pytest.raises(SystemExit):
            cli.main(["serve", "--devices", "gpu", "--model", str(tmp_path), "--max-batch-size", str(n)])
        assert "--max-batch-size" not in captured["argv"]
        assert captured["env"][cli.MAX_BATCH_SIZE_ENV] == str(n)
        if n == 1:
            baseline_argv = captured["argv"]  # type: ignore[assignment]

    # N = 1 is also what a caller gets without passing the flag at all: argv unchanged.
    with pytest.raises(SystemExit):
        cli.main(["serve", "--devices", "gpu", "--model", str(tmp_path)])
    assert captured["argv"] == baseline_argv
    assert captured["env"][cli.MAX_BATCH_SIZE_ENV] == "1"


def test_the_architecture_forces_in_flight_batching_off() -> None:
    """``execute`` serves all-prefill or all-decode steps only, so the scheduler must never mix them."""
    from unlimited_ocr_max import arch

    assert arch.unlimited_ocr_arch.required_arguments["enable_in_flight_batching"] is False
