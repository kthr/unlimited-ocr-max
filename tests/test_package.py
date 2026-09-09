"""Model-free checks: the pin, the lazy architecture export, the serve command, the weight variants,
MAX's weight-filename parser, prompt construction."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unlimited_ocr_max import cli

ROOT = Path(__file__).resolve().parent.parent
PINNED_MAX_VERSION = "26.6.0.dev2026082707"


def test_installed_max_is_the_pinned_nightly() -> None:
    assert importlib.metadata.version("max") == PINNED_MAX_VERSION
    assert f'"max[all]=={PINNED_MAX_VERSION}"' in (ROOT / "pyproject.toml").read_text()


def _modules_after(code: str) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-c", code + "\nimport sys, json; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True, text=True, check=True, timeout=120, env=dict(os.environ),
    )
    return set(json.loads(out.stdout.splitlines()[-1]))


def test_cli_and_package_import_do_not_import_max() -> None:
    loaded = _modules_after("import unlimited_ocr_max, unlimited_ocr_max.cli")
    assert not any(name == "max" or name.startswith("max.") for name in loaded)
    assert "unlimited_ocr_max.arch" not in loaded


def test_architectures_is_resolved_lazily() -> None:
    import unlimited_ocr_max

    archs = unlimited_ocr_max.ARCHITECTURES
    assert [arch.name for arch in archs] == ["UnlimitedOCRForCausalLM"]
    assert "unlimited_ocr_max.arch" in sys.modules


def test_mojo_kernel_ships_with_the_package() -> None:
    kernels = cli.PACKAGE_DIR / "kernels"
    assert (kernels / "__init__.mojo").is_file()
    assert (kernels / "ngram_block.mojo").is_file()
    assert (kernels / "moe_int8.mojo").is_file()


TAIL = [
    "--custom-architectures", str(cli.PACKAGE_DIR),
    "--devices", "gpu",
    "--quantization-encoding", "float32",
    "--max-length", "2048",
    "--served-model-name", "unlimited-ocr-max",
    "--port", "8010",
]


def test_serve_command_for_a_hub_model() -> None:
    model, weight_path, revision = cli.resolve_model(cli.DEFAULT_MODEL, "bf16", cli.DEFAULT_REVISION)
    cmd = cli.serve_command(max_exe="/venv/bin/max", model=model, weight_path=weight_path, devices="gpu", port=8010, revision=revision)
    assert cmd == [
        "/venv/bin/max", "serve", "--model", "kthierbach/unlimited-ocr-max",
        "--huggingface-model-revision", cli.DEFAULT_REVISION, "--huggingface-weight-revision", cli.DEFAULT_REVISION,
        "--weight-path", "kthierbach/unlimited-ocr-max/model.safetensors",
        *TAIL,
    ]


def test_serve_command_for_a_local_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    model, weight_path, revision = cli.resolve_model(str(tmp_path), "bf16", cli.DEFAULT_REVISION)
    assert (model, weight_path, revision) == (str(tmp_path), str(tmp_path / "model.safetensors"), None)
    cmd = cli.serve_command(max_exe="/venv/bin/max", model=model, weight_path=weight_path, devices="gpu", port=8010, revision=revision)
    assert "--huggingface-model-revision" not in cmd
    assert cmd[2:6] == ["--model", str(tmp_path), "--weight-path", str(tmp_path / "model.safetensors")]


def test_default_revision_is_this_versions_tag() -> None:
    assert cli.DEFAULT_REVISION == f"v{importlib.metadata.version('unlimited-ocr-max')}"


def test_refuses_a_directory_without_config(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(SystemExit):
        cli.resolve_model(str(tmp_path), "bf16", None)


def test_refuses_a_directory_without_the_weight_variant(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        cli.resolve_model(str(tmp_path), "bf16", None)


def test_refuses_a_path_looking_model_that_is_not_a_directory() -> None:
    for model in ("./missing", "~/missing", "/nonexistent/dir", "no-slash", "a/b/c"):
        with pytest.raises(SystemExit):
            cli.resolve_model(model, "bf16", None)


def test_weight_file_maps_every_variant_to_its_filename() -> None:
    assert cli.WEIGHT_VARIANTS == ("bf16", "int8")
    assert cli.weight_file("bf16") == "model.safetensors"  # unquantised: no encoding hint in the name
    assert cli.weight_file("int8") == "model-int8.safetensors"
    with pytest.raises(SystemExit):
        cli.weight_file("int4")


def test_max_filename_parser_ignores_our_weight_filenames() -> None:
    """MAX infers a quantisation encoding from the weight filename and then refuses a bf16 hint on
    CPU -- the failure mode that broke CPU serving while the shard was named
    ``model-bf16.safetensors``. This guard fails loudly on a pin bump whose token list starts
    matching our filenames. The positive control keeps a green guard meaningful: without it, an
    import or parser that stopped resolving encodings would pass the guard for the wrong reason."""
    from max.pipelines.modeling.config_enums import parse_supported_encoding_from_file_name

    for variant in cli.WEIGHT_VARIANTS:  # model.safetensors, model-int8.safetensors
        assert parse_supported_encoding_from_file_name(cli.weight_file(variant)) is None, variant
    assert parse_supported_encoding_from_file_name("model-bf16.safetensors") is not None


def test_int8_on_cpu_is_refused_before_any_path_work() -> None:
    missing = "/nonexistent/unlimited-ocr-max"  # resolve_model would fail on this, with another message
    with pytest.raises(SystemExit) as refused:
        cli.main(["serve", "--devices", "cpu", "--weights", "int8", "--model", missing])
    assert str(refused.value) == "--weights int8 is GPU-only; serve on cpu with --weights bf16"

    for weights, devices in (("bf16", "cpu"), ("int8", "gpu")):  # every other combination gets that far
        with pytest.raises(SystemExit) as reached_resolve_model:
            cli.main(["serve", "--devices", devices, "--weights", weights, "--model", missing])
        assert "neither an existing directory" in str(reached_resolve_model.value)


def test_ngram_size_travels_to_the_server_through_the_private_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    from unlimited_ocr_max.model import NGRAM_SIZE_ENV, serve_ngram_size

    assert NGRAM_SIZE_ENV == cli.NGRAM_ENV
    monkeypatch.delenv(NGRAM_SIZE_ENV, raising=False)
    assert serve_ngram_size() == 35
    monkeypatch.setenv(NGRAM_SIZE_ENV, "0")
    assert serve_ngram_size() == 0


def test_prompt_expansion() -> None:
    from unlimited_ocr_max.tokenizer import DEFAULT_PROMPT, IMAGE_TOKEN, UnlimitedOcrProcessor, build_prompt

    prompt = build_prompt(lambda text: [7] * len(text.split()), prompt=DEFAULT_PROMPT)
    assert int(prompt.image_mask.sum()) == 273
    assert prompt.seq_len == 1 + 273 + 2
    assert prompt.ids[0] == 0 and not prompt.image_mask[0]
    processor = UnlimitedOcrProcessor(delegate=None)
    chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Convert."}, {"type": "image_url", "image_url": {"url": "x"}}]}]
    )
    assert chat == IMAGE_TOKEN + "Convert."
    assert processor.apply_chat_template([]) == DEFAULT_PROMPT


@pytest.mark.parametrize("argv", [["--help"], ["serve", "--help"]])
def test_cli_help(argv: list[str]) -> None:
    script = Path(sys.executable).with_name("unlimited-ocr-max")
    cmd = [str(script), *argv] if script.is_file() else [sys.executable, "-m", "unlimited_ocr_max.cli", *argv]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=dict(os.environ))
    assert result.returncode == 0, result.stderr
    assert "unlimited-ocr-max" in result.stdout
