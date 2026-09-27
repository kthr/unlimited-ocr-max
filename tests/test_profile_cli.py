"""``unlimited-ocr-max profile`` end to end against a stub ``max`` (KON-206).

Every run here is a real process tree -- ``tests/profile_stub_max.py`` behind a wrapper named
``max``, on a free ephemeral port -- so the teardown is exercised on real processes: after each
run, the stub and the child it spawned must be gone. The real model never runs here. GPU
behaviour is driven through a fake device probe and a monkeypatched foreign-process list; this
development machine has no NVIDIA/AMD tooling to exercise it for real.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from unlimited_ocr_max import cli, profile, profile_corpus, profile_sampling

STUB = Path(__file__).resolve().with_name("profile_stub_max.py")
PAGES = profile_corpus.page_names()
#: Mirrors of tests/profile_stub_max.py's constants.
TG_PER_REQUEST = 3
CHUNK_CHARS = 7
WRONG_PAGE = "dense_body"
GIB = 2**30


@dataclass
class Stub:
    exe: str
    dir: Path

    def pids(self) -> dict[str, Any]:
        return json.loads((self.dir / "pids.json").read_text())

    def requests(self) -> list[dict[str, Any]]:
        path = self.dir / "requests.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stub:
    """A wrapper named ``max`` that runs the stub under this interpreter; faster polling and dwell."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    exe = tmp_path / "bin" / "max"
    exe.parent.mkdir()
    exe.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(STUB))} "$@"\n')
    exe.chmod(0o755)
    monkeypatch.setenv("PROFILE_STUB_DIR", str(stub_dir))
    monkeypatch.delenv("PROFILE_STUB_MODE", raising=False)
    monkeypatch.delenv("PROFILE_STUB_PREFILL_S", raising=False)
    monkeypatch.setattr(profile, "READY_POLL_S", 0.2)
    monkeypatch.setattr(profile, "EXIT_POLL_S", 0.1)
    monkeypatch.setattr(profile, "STEADY_DWELL_S", 1.0)
    return Stub(exe=str(exe), dir=stub_dir)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _args(out: Path, port: int, *extra: str, devices: str = "cpu") -> argparse.Namespace:
    return cli.build_parser().parse_args(
        ["profile", "--devices", devices, "--port", str(port), "--out", str(out), *extra]
    )


def _gone(pid: int, *, allow_zombie: bool) -> bool:
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return stat == "" or (allow_zombie and stat.startswith("Z"))


def _tree_gone(stub: Stub, timeout: float = 10.0) -> bool:
    """The stub is gone and reaped (profile reaps it); its child is gone (init may not have reaped it yet)."""
    pids = stub.pids()
    deadline = time.monotonic() + timeout
    while True:
        done = _gone(pids["pid"], allow_zombie=False) and _gone(pids["child"], allow_zombie=True)
        if done or time.monotonic() > deadline:
            return done
        time.sleep(0.05)


class _AtexitRecorder:
    """Stands in for ``atexit`` inside :mod:`unlimited_ocr_max.profile`, delegating to the real one."""

    def __init__(self) -> None:
        self.registered: list[Any] = []
        self.pending: list[Any] = []

    def register(self, fn: Any) -> Any:
        self.registered.append(fn)
        self.pending.append(fn)
        return atexit.register(fn)

    def unregister(self, fn: Any) -> None:
        self.pending.remove(fn)
        atexit.unregister(fn)


@pytest.fixture
def atexit_recorder(monkeypatch: pytest.MonkeyPatch) -> _AtexitRecorder:
    recorder = _AtexitRecorder()
    monkeypatch.setattr(profile, "atexit", recorder)
    return recorder


def _cells(row: str) -> list[str]:
    assert row.startswith("| ") and row.endswith(" |"), row
    return [cell.strip() for cell in row[1:-1].split("|")]


class _FakeProbe:
    """A device probe with one NVIDIA GPU: 3 GiB for the server's tree, 40 % utilisation."""

    device_available = True

    def __init__(self) -> None:
        self.closed = False

    def stats(self) -> dict[str, dict[str, int]]:
        return {"nv0": {"used_bytes": GIB, "total_bytes": 80 * GIB, "gpu_usage_percent": 40}}

    def process_bytes(self, pids: Any) -> int | None:
        return 3 * GIB

    def close(self) -> None:
        self.closed = True


class _NoDeviceProbe(_FakeProbe):
    """What ``open_device_probe`` returns on Metal, or where MAX's GPU diagnostics failed."""

    device_available = False

    def stats(self) -> dict[str, dict[str, int]]:
        return {}

    def process_bytes(self, pids: Any) -> int | None:
        return None


def _no_spawn(self: Any) -> None:
    pytest.fail("profile started a server although it had to refuse first")


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def _modules_and_stdout(code: str) -> tuple[set[str], str]:
    wrapped = (
        "import contextlib, io, json, sys\n"
        "buffer = io.StringIO()\n"
        "with contextlib.redirect_stdout(buffer):\n"
        "    try:\n"
        + "".join(f"        {line}\n" for line in code.splitlines())
        + "    except SystemExit:\n"
        "        pass\n"
        "print(json.dumps({'stdout': buffer.getvalue(), 'modules': sorted(sys.modules)}))\n"
    )
    result = subprocess.run([sys.executable, "-c", wrapped], capture_output=True, text=True, check=True,
                            timeout=120, env=dict(os.environ))
    payload = json.loads(result.stdout.splitlines()[-1])
    return set(payload["modules"]), payload["stdout"]


def _imports_max(modules: set[str]) -> bool:
    return any(name == "max" or name.startswith("max.") for name in modules)


def test_profile_help_does_not_import_max() -> None:
    modules, help_text = _modules_and_stdout("from unlimited_ocr_max import cli\ncli.main(['profile', '--help'])")
    assert not _imports_max(modules)
    assert "unlimited_ocr_max.profile" not in modules
    for flag in ("--devices", "--model", "--revision", "--weights", "--port", "--ngram-size", "--out", "--ready-timeout-s"):
        assert flag in help_text


def test_importing_the_profile_module_does_not_import_max() -> None:
    modules, _ = _modules_and_stdout("import unlimited_ocr_max.profile")
    assert not _imports_max(modules)


def test_profile_help_through_the_console_script() -> None:
    script = Path(sys.executable).with_name("unlimited-ocr-max")
    cmd = [str(script)] if script.is_file() else [sys.executable, "-m", "unlimited_ocr_max.cli"]
    result = subprocess.run([*cmd, "profile", "--help"], capture_output=True, text=True, timeout=60, env=dict(os.environ))
    assert result.returncode == 0, result.stderr
    assert "usage: unlimited-ocr-max profile" in result.stdout


def test_serve_and_profile_share_one_definition_of_the_server_flags() -> None:
    parser = cli.build_parser()
    (subparsers,) = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]

    def spec(sub: argparse.ArgumentParser) -> dict[str, tuple]:
        return {
            action.dest: (tuple(action.option_strings), action.default, action.choices, action.required, action.help, action.type)
            for action in sub._actions
            if action.dest != "help"
        }

    serve, prof = spec(subparsers.choices["serve"]), spec(subparsers.choices["profile"])
    assert list(serve) == ["devices", "model", "revision", "weights", "port", "ngram_size"]
    assert {dest: prof[dest] for dest in serve} == serve
    assert list(prof)[len(serve):] == ["out", "ready_timeout_s"]
    assert prof["ready_timeout_s"][1] == 1800
    assert prof["out"][1] is None  # resolved at run time, to the run's own UTC timestamp


def test_int8_on_cpu_is_refused_as_serve_refuses_it() -> None:
    missing = "/nonexistent/unlimited-ocr-max"
    with pytest.raises(SystemExit) as refused:
        cli.main(["profile", "--devices", "cpu", "--weights", "int8", "--model", missing])
    assert str(refused.value) == "--weights int8 is GPU-only; serve on cpu with --weights bf16"
    for weights, devices in (("bf16", "cpu"), ("int8", "gpu")):
        with pytest.raises(SystemExit) as reached_resolve_model:
            cli.main(["profile", "--devices", devices, "--weights", weights, "--model", missing])
        assert "neither an existing directory" in str(reached_resolve_model.value)


def test_default_out_is_a_fresh_utc_stamped_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = profile._out_dir(None)
    assert out.parent == tmp_path.absolute()
    assert re.fullmatch(r"unlimited-ocr-max-profile-\d{8}T\d{6}Z", out.name)


def test_a_non_empty_out_directory_is_refused(stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    out.mkdir()
    (out / "profile.json").write_text("{}")
    with pytest.raises(SystemExit, match="not an empty directory"):
        profile.run(_args(out, _free_port()), max_exe=stub.exe)
    assert (out / "profile.json").read_text() == "{}"


def test_chat_body_is_byte_for_byte_the_research_harness_body() -> None:
    png = b"\x89PNG\r\n\x1a\n not really"
    expected = json.dumps({
        "model": "unlimited-ocr-max",
        "temperature": 0,
        "max_tokens": 8192,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ],
        }],
    }).encode("utf-8")
    assert profile.chat_body(png, 8192) == expected


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_profile_end_to_end_against_the_stub(stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             atexit_recorder: _AtexitRecorder) -> None:
    out = tmp_path / "run"
    port = _free_port()
    code = profile.run(_args(out, port), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    assert len(atexit_recorder.registered) == 1 and atexit_recorder.pending == []

    # The server ran with exactly `serve`'s command and environment.
    model, weight_path, revision = cli.resolve_model(cli.DEFAULT_MODEL, "bf16", cli.DEFAULT_REVISION)
    served = cli.serve_command(max_exe=stub.exe, model=model, weight_path=weight_path, devices="cpu", port=port,
                               revision=revision)
    pids = stub.pids()
    assert pids["argv"] == served[1:]
    assert pids["ngram"] == "35"
    # Warmup on the first page with 8 tokens, then every page in manifest order with 8192.
    assert stub.requests() == [{"page": PAGES[0], "max_tokens": 8}] + [{"page": p, "max_tokens": 8192} for p in PAGES]
    for page in PAGES:
        assert (out / "pages" / f"{page}.md").read_bytes() == profile_corpus.reference(page, "bf16").encode("utf-8")

    doc = json.loads((out / "profile.json").read_text())
    assert doc["schema"] == 1
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["started_utc"])
    assert doc["started_utc"] <= doc["ended_utc"]
    host = doc["host"]
    for key in ("platform", "machine", "cpu_brand", "ram_bytes", "accelerator", "gpus", "hardware"):
        assert key in host
    assert {"api", "architecture"} <= set(host["accelerator"])
    assert host["hardware"].endswith(", CPU")
    assert set(doc["versions"]) == {"unlimited-ocr-max", "max", "mojo"}
    assert doc["flags"] == {"devices": "cpu", "weights": "bf16", "model": cli.DEFAULT_MODEL, "revision": cli.DEFAULT_REVISION,
                            "port": port, "ngram_size": 35, "ready_timeout_s": 1800, "out": str(out)}
    assert doc["served_command"] == served
    assert isinstance(doc["device_baseline"], dict)
    assert doc["void"] == [] and doc["teardown_warnings"] == []

    figures = doc["figures"]
    # 13 requests x 3 TG lines of 50 ms; the one 2.50 s line is excluded, and counted.
    assert figures["decode"] == {"median_ms": 50.0, "floor_ms": 50.0, "n": 13 * TG_PER_REQUEST, "tok_s": 20.0,
                                 "n_whole_second_excluded": 1}
    # The warmup's 30 s CE line is skipped; the 12 page prefills are 5.00 s each.
    assert figures["prefill"] == {"median_s": 5.0, "floor_s": 5.0, "n": 12}
    assert figures["text"] == {"identical": 12, "n": 12, "edits": 0,
                               "ref_chars": sum(len(profile_corpus.reference(p)) for p in PAGES), "cer": 0.0}
    memory = figures["host_memory"]
    assert memory["peak_bytes"] > 0 and memory["n"] > 0
    assert memory["n_steady"] >= 1 and memory["steady_min_bytes"] <= memory["steady_max_bytes"] <= memory["peak_bytes"]
    assert figures["device_memory"] is None and "device memory" in doc["unavailable"]["device_memory"]
    assert figures["gpu_utilisation"] is None and doc["unavailable"]["gpu_utilisation"]
    assert doc["warmup"]["completion_tokens"] == 8 and doc["warmup"]["finish_reason"] == "length"

    assert [row["page"] for row in doc["pages"]] == PAGES
    for row in doc["pages"]:
        assert row["identical"] is True and row["edits"] == 0
        assert row["completion_tokens"] == math.ceil(len(profile_corpus.reference(row["page"])) / CHUNK_CHARS)
        assert row["finish_reason"] == "stop"
        # The empty role delta arrives before the stub's 0.05 s "prefill"; TTFT is the first *non-empty* one.
        assert 0.05 <= row["ttft_s"] <= row["wall_s"]

    lines = captured.out.splitlines()
    assert lines[0] == doc["row"]
    hardware, weights, status, decode, prefill, memory_cell, text = _cells(lines[0])
    assert hardware == host["hardware"]
    assert (weights, status) == ("bf16", "profiled, 12 pages")
    assert decode == "**20.0 tok/s** (50.0 ms/step, n=39)"
    assert prefill == "5.00 s"
    assert re.fullmatch(r"\d+\.\d / \d+\.\d(–\d+\.\d)? GiB", memory_cell)
    assert text == "12/12 byte-identical, CER 0"
    assert lines[1:] == [f"profile.json: {out / 'profile.json'}"]


def test_a_wrong_page_is_counted_not_hidden(stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", "wrong-page")
    out = tmp_path / "run"
    out.mkdir()  # an existing *empty* directory is accepted
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err  # a text mismatch is a result, not a void measurement
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    text = doc["figures"]["text"]
    assert (text["identical"], text["edits"]) == (11, 1) and text["cer"] > 0
    assert [row["page"] for row in doc["pages"] if not row["identical"]] == [WRONG_PAGE]
    assert (out / "pages" / f"{WRONG_PAGE}.md").read_bytes() != profile_corpus.reference(WRONG_PAGE).encode("utf-8")
    assert _cells(captured.out.splitlines()[0])[6] == f"11/12 byte-identical, CER {text['cer']:.2g}"


def test_int8_weights_are_also_compared_against_the_pinned_int8_references(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(profile_sampling, "open_device_probe", _FakeProbe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", lambda own: [])
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), "--weights", "int8", devices="gpu"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    assert "model-int8.safetensors" in " ".join(stub.pids()["argv"])
    same = sum(profile_corpus.reference(p, "bf16") == profile_corpus.reference(p, "int8") for p in PAGES)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["figures"]["text"]["identical"] == 12  # the stub streams the bf16 references
    assert doc["figures"]["text_vs_int8"]["identical"] == same
    assert sum(row["identical_vs_int8"] for row in doc["pages"]) == same
    assert f"vs pinned int8: {same}/12 byte-identical, CER " in captured.out


def test_a_server_that_exits_early_is_reported_and_its_tree_torn_down(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", "exit-early")
    out = tmp_path / "run"
    start = time.monotonic()
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    elapsed = time.monotonic() - start
    captured = capsys.readouterr()
    assert code == 4
    assert elapsed < 15, elapsed
    assert "the server exited before it was ready" in captured.err
    assert "failing before the server is up" in captured.err  # the serve.log tail
    assert _tree_gone(stub)  # including the child the stub left behind in its process group
    assert not (out / "profile.json").exists()


def test_a_server_that_never_gets_ready_times_out(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", "never-ready")
    code = profile.run(_args(tmp_path / "run", _free_port(), "--ready-timeout-s", "1"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 4
    assert "not ready after 1 s" in captured.err
    assert _tree_gone(stub)


def test_a_port_already_listening_is_refused_before_spawning(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        code = profile.run(_args(out, listener.getsockname()[1]), max_exe=stub.exe)
    assert code == 2
    assert "already listens" in capsys.readouterr().err
    assert not out.exists()


def test_gpu_guard_refuses_a_foreign_process_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    probe = _FakeProbe()
    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: probe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes",
                        lambda own: [{"pid": 4242, "name": "python", "used_bytes": 1536 * 2**20}])
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe="/nonexistent/max")
    err = capsys.readouterr().err
    assert code == 3
    assert "otherwise idle GPU" in err and "pid 4242 python 1536 MiB" in err
    assert probe.closed
    assert not out.exists()


def test_gpu_guard_refuses_when_the_process_list_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(own: set[int]) -> list[dict[str, Any]]:
        raise profile_sampling.GpuProcessListUnavailable("an NVIDIA GPU is reported but nvidia-smi is not on PATH")

    monkeypatch.setattr(profile_sampling, "open_device_probe", _FakeProbe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", unavailable)
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    code = profile.run(_args(tmp_path / "run", _free_port(), devices="gpu"), max_exe="/nonexistent/max")
    err = capsys.readouterr().err
    assert code == 3
    assert "otherwise idle GPU" in err and "nvidia-smi is not on PATH" in err


def test_gpu_guard_runs_without_device_statistics_and_refuses_an_unlistable_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CUDA/ROCm host whose MAX GPU diagnostics failed: no device statistics, yet the guard
    runs -- and a process list it cannot get refuses the run instead of measuring unguarded."""

    def unavailable(own: set[int]) -> list[dict[str, Any]]:
        raise profile_sampling.GpuProcessListUnavailable(
            "cannot determine which GPUs are present: max.profiler.gpu's GPUDiagContext failed to report device stats"
        )

    probe = _NoDeviceProbe()
    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: probe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", unavailable)
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe="/nonexistent/max")
    err = capsys.readouterr().err
    assert code == 3
    assert "otherwise idle GPU" in err and "GPUDiagContext failed to report device stats" in err
    assert probe.closed
    assert not out.exists()


def test_gpu_guard_runs_without_device_statistics_and_refuses_a_foreign_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[set[int]] = []

    def foreign(own: set[int]) -> list[dict[str, Any]]:
        calls.append(set(own))
        return [{"pid": 4242, "name": "python", "used_bytes": 1536 * 2**20}]

    monkeypatch.setattr(profile_sampling, "open_device_probe", _NoDeviceProbe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", foreign)
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe="/nonexistent/max")
    err = capsys.readouterr().err
    assert code == 3
    assert calls == [set()]
    assert "otherwise idle GPU" in err and "pid 4242 python 1536 MiB" in err
    assert not out.exists()


def test_the_post_run_gpu_recheck_also_runs_without_device_statistics(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[set[int]] = []

    def foreign(own: set[int]) -> list[dict[str, Any]]:
        calls.append(set(own))
        return [] if len(calls) == 1 else [{"pid": 4242, "name": "intruder", "used_bytes": None}]

    monkeypatch.setattr(profile_sampling, "open_device_probe", _NoDeviceProbe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", foreign)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 5, captured.err
    assert _tree_gone(stub)
    assert len(calls) == 2 and {stub.pids()["pid"], stub.pids()["child"]} <= calls[1]
    doc = json.loads((out / "profile.json").read_text())
    assert doc["gpu_guard"] is True
    assert len(doc["void"]) == 1 and "pid 4242 intruder ? MiB" in doc["void"][0]
    assert doc["figures"]["device_memory"] is None


def test_a_cpu_run_never_consults_the_gpu_process_list(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_run(own: set[int]) -> list[dict[str, Any]]:
        raise AssertionError("foreign_gpu_processes was called for a --devices cpu run")

    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", must_not_run)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["gpu_guard"] is False and doc["void"] == []


def test_a_foreign_gpu_process_found_after_the_run_voids_it(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_PREFILL_S", "0.3")  # page windows long enough for several device samples
    probe = _FakeProbe()
    calls: list[set[int]] = []

    def foreign(own: set[int]) -> list[dict[str, Any]]:
        calls.append(set(own))
        return [] if len(calls) == 1 else [{"pid": 4242, "name": "intruder", "used_bytes": None}]

    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: probe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", foreign)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 5, captured.err
    assert _tree_gone(stub)
    pids = stub.pids()
    assert calls[0] == set() and {pids["pid"], pids["child"]} <= calls[1]
    doc = json.loads((out / "profile.json").read_text())
    assert len(doc["void"]) == 1 and "pid 4242 intruder ? MiB" in doc["void"][0]
    assert doc["gpu_guard"] is True
    assert doc["device_baseline"] == probe.stats()
    assert doc["figures"]["device_memory"]["peak_bytes"] == 3 * GIB
    assert doc["figures"]["gpu_utilisation"]["median_percent"] == 40
    lines = captured.out.splitlines()
    assert _cells(lines[0])[2] == "void, 12 pages"
    assert lines[1] == "device memory peak / steady 3.0 / 3.0 GiB, median GPU utilisation 40 %"
    assert lines[2] == f"void: {doc['void'][0]}"
    assert probe.closed


def test_an_exception_mid_run_still_tears_the_server_down(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atexit_recorder: _AtexitRecorder
) -> None:
    original = profile._stream_chat
    calls = 0

    def fails_after_two_pages(*args: Any, **kwargs: Any) -> profile.Exchange:
        nonlocal calls
        calls += 1
        if calls > 3:  # the warmup and two pages went through
            raise RuntimeError("boom after two pages")
        return original(*args, **kwargs)

    monkeypatch.setattr(profile, "_stream_chat", fails_after_two_pages)
    with pytest.raises(RuntimeError, match="boom after two pages"):
        profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    assert _tree_gone(stub)
    assert len(atexit_recorder.registered) == 1 and atexit_recorder.pending == []
    assert len(stub.requests()) == 3


def test_a_failed_request_voids_the_run_but_still_writes_the_figures(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = profile._stream_chat
    calls = 0

    def server_dies_after_two_pages(*args: Any, **kwargs: Any) -> profile.Exchange:
        nonlocal calls
        calls += 1
        if calls > 3:
            raise profile.ExchangeFailed("ConnectionRefusedError: [Errno 61] Connection refused")
        return original(*args, **kwargs)

    monkeypatch.setattr(profile, "_stream_chat", server_dies_after_two_pages)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 5
    assert _tree_gone(stub)
    assert calls == 4  # nothing is sent after the first failure
    doc = json.loads((out / "profile.json").read_text())
    assert doc["void"] == [f"request {PAGES[2]} failed: ConnectionRefusedError: [Errno 61] Connection refused"]
    assert doc["figures"]["text"] is None and doc["unavailable"]["text"] == "only 2 of 12 pages completed"
    assert [row["page"] for row in doc["pages"]] == PAGES[:2]
    assert doc["figures"]["decode"]["n"] == 3 * TG_PER_REQUEST  # what the log holds is still reported
    assert _cells(captured.out.splitlines()[0])[2] == "void, 2 pages"
    assert "last 40 lines of" in captured.err


def test_ctrl_c_during_the_teardown_is_held_until_it_has_finished(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        pytest.skip("SIGINT does not raise KeyboardInterrupt in this test process")
    original = profile._Server._record_identities

    def interrupted_mid_teardown(self: Any, *args: Any) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)  # the handler runs here: it must hold the signal, not raise
        original(self, *args)

    monkeypatch.setattr(profile._Server, "_record_identities", interrupted_mid_teardown)
    code = profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 130  # re-delivered once the teardown finished
    assert "SIGINT received; still stopping the server" in captured.err
    assert _tree_gone(stub)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_sigterm_mid_run_tears_the_server_down_and_exits_143(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        pytest.skip("SIGTERM is not at its default disposition in this test process")
    original = profile._stream_chat
    calls = 0

    def terminated_after_two_pages(*args: Any, **kwargs: Any) -> profile.Exchange:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(profile, "_stream_chat", terminated_after_two_pages)
    with pytest.raises(SystemExit) as exited:
        profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    assert exited.value.code == 128 + signal.SIGTERM
    assert _tree_gone(stub)
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_a_child_that_left_the_process_group_is_still_stopped(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", "detach")
    code = profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"SIGTERM to pid(s) [{stub.pids()['child']}] of the server's tree, still alive" in captured.err
    assert _tree_gone(stub)


def test_a_server_that_ignores_sigterm_is_sigkilled_after_the_grace(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", "ignore-term")
    monkeypatch.setattr(profile, "STOP_GRACE_S", 1.0)
    code = profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    assert code == 0, capsys.readouterr().err
    assert _tree_gone(stub)
