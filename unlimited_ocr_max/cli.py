"""``unlimited-ocr-max serve``: build the ``max serve`` command for this port and hand off to it.

Imports only the standard library, so ``--help`` never loads MAX.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_MODEL = "kthierbach/unlimited-ocr-max"
#: The model-repo tag this package version was validated against; ignored for a local ``--model``.
DEFAULT_REVISION = "v0.1.0"
WEIGHT_VARIANTS = ("bf16",)
PACKAGE_DIR = Path(__file__).resolve().parent
SERVED_MODEL_NAME = "unlimited-ocr-max"
MAX_LENGTH = 2048
#: Internal transport to the architecture inside the ``max serve`` child; see ``model.NGRAM_SIZE_ENV``.
NGRAM_ENV = "_UNLIMITED_OCR_MAX_NGRAM_SIZE"


def weight_file(variant: str) -> str:
    """The unquantised weights are the plain ``model.safetensors``: MAX reads encoding hints (``bf16``,
    ``fp16``, ``q4_k_m``, ...) out of weight filenames, and a ``bf16`` hint is refused on CPU."""
    if variant not in WEIGHT_VARIANTS:
        raise SystemExit(f"unknown --weights {variant!r}; choose from {', '.join(WEIGHT_VARIANTS)}")
    return "model.safetensors" if variant == "bf16" else f"model-{variant}.safetensors"


def max_executable() -> str:
    beside = Path(sys.executable).with_name("max")
    if beside.is_file():
        return str(beside)
    found = shutil.which("max")
    if found:
        return found
    raise SystemExit("`max` not found next to the interpreter or on PATH; install max[all]")


def resolve_model(model: str, weights: str, revision: str | None) -> tuple[str, str, str | None]:
    """``(model, weight_path, revision)``: a local directory is checked and loses its revision."""
    filename = weight_file(weights)
    model_dir = Path(model).expanduser()
    if model_dir.is_dir():
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"{model} has no config.json -- not a servable model directory")
        weight_path = model_dir / filename
        if not weight_path.is_file():
            raise SystemExit(f"{model} has no {filename} (the --weights {weights} file); expected the layout of {DEFAULT_MODEL}")
        return str(model_dir), str(weight_path), None
    if model.startswith((".", "~", "/")) or model.count("/") != 1:
        raise SystemExit(f"{model!r} is neither an existing directory nor a Hub id of the form <user>/<repo>")
    return model, f"{model}/{filename}", revision


def serve_command(*, max_exe: str, model: str, weight_path: str, devices: str, port: int, revision: str | None) -> list[str]:
    cmd = [max_exe, "serve", "--model", model]
    if revision is not None:
        cmd += ["--huggingface-model-revision", revision, "--huggingface-weight-revision", revision]
    return cmd + [
        "--weight-path", weight_path,
        "--custom-architectures", str(PACKAGE_DIR),
        "--devices", devices,
        "--quantization-encoding", "float32",
        "--max-length", str(MAX_LENGTH),
        "--served-model-name", SERVED_MODEL_NAME,
        "--port", str(port),
    ]


def cmd_serve(args: argparse.Namespace) -> int:
    model, weight_path, revision = resolve_model(args.model, args.weights, args.revision)
    cmd = serve_command(
        max_exe=max_executable(), model=model, weight_path=weight_path, devices=args.devices, port=args.port, revision=revision
    )
    env = dict(os.environ)
    env[NGRAM_ENV] = str(args.ngram_size)
    print("[unlimited-ocr-max] " + " ".join(cmd), file=sys.stderr, flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 1  # unreachable: execvpe only returns on failure, which raises


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="unlimited-ocr-max", description="Serve baidu/Unlimited-OCR through MAX on Apple Silicon (Metal GPU or CPU)."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    srv = sub.add_parser("serve", help="run `max serve` with this port's flags")
    srv.add_argument("--devices", choices=("cpu", "gpu"), required=True,
                     help="gpu (Metal; needs Xcode + Metal Toolchain) or cpu (supported, slow)")
    srv.add_argument("--model", default=DEFAULT_MODEL,
                     help="Hub repository or a local directory with its layout (default: %(default)s)")
    srv.add_argument("--revision", default=DEFAULT_REVISION,
                     help="Hub revision for config, tokenizer and weights; ignored for a local directory (default: %(default)s)")
    srv.add_argument("--weights", default="bf16", choices=WEIGHT_VARIANTS,
                     help="weight variant: bf16 is the unquantised model.safetensors, others model-<variant>.safetensors (default: %(default)s)")
    srv.add_argument("--port", type=int, default=8010, help="(default: %(default)s)")
    # 35 is `ngram.DEFAULT_NGRAM_SIZE`, repeated here so this module imports no MAX.
    srv.add_argument("--ngram-size", type=int, default=35,
                     help="no-repeat n-gram guard; 0 disables, which reproduces the PyTorch reference (default: %(default)s)")
    srv.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
