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
import shutil
import signal
import socket
import subprocess
import sys
import threading
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
    """A wrapper named ``max`` that runs the stub under this interpreter; faster polling and dwell;
    a constant system swap, so this machine paging during a test cannot flag its timings."""
    monkeypatch.setattr(profile_sampling, "swap_used_bytes", lambda: GIB)
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


class _TwoGpuProbe(_FakeProbe):
    """Two NVIDIA GPUs: nv0 stays idle at its baseline; nv1 serves -- 3 GiB above its baseline and
    90 % busy from the first sample after the pre-start baseline on."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def stats(self) -> dict[str, dict[str, int]]:
        self.calls += 1
        serving = self.calls > 1  # the first call is the pre-start baseline
        return {
            "nv0": {"used_bytes": GIB, "total_bytes": 80 * GIB, "gpu_usage_percent": 0},
            "nv1": {"used_bytes": (4 if serving else 1) * GIB, "total_bytes": 80 * GIB,
                    "gpu_usage_percent": 90 if serving else 0},
        }


class _ProbeFailingMidRun(_FakeProbe):
    """Device statistics raise once the stub has received the warmup and seven pages."""

    def __init__(self, stub: Stub) -> None:
        super().__init__()
        self.stub = stub

    def stats(self) -> dict[str, dict[str, int]]:
        if len(self.stub.requests()) >= 8:
            raise RuntimeError("the device went away")
        return super().stats()


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
    # ... and kept in prefill_all_ce, the population the figures published before `profile` used.
    assert figures["prefill_all_ce"] == {"median_s": 5.0, "floor_s": 5.0, "n": 13}
    assert figures["text"] == {"identical": 12, "n": 12, "edits": 0,
                               "ref_chars": sum(len(profile_corpus.reference(p)) for p in PAGES), "cer": 0.0}
    memory = figures["host_memory"]
    assert memory["peak_bytes"] > 0 and memory["n"] > 0
    assert memory["n_steady"] >= 1 and memory["steady_min_bytes"] <= memory["steady_max_bytes"] <= memory["peak_bytes"]
    assert figures["device_memory"] is None and doc["unavailable"]["device_memory"] == "--devices cpu"
    assert figures["gpu_utilisation"] is None and doc["unavailable"]["gpu_utilisation"] == "--devices cpu"
    assert doc["swap"] == {"start_bytes": GIB, "max_bytes": GIB, "growth_bytes": 0, "reason": None}
    assert doc["sampling"]["swap_samples"] > 0
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
    assert captured.err.count("stopping the server") == 1  # said once, by the teardown itself


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


def test_a_cpu_run_never_consults_the_gpu_process_list_and_reports_no_device_figures(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a host with a GPU (the fake probe samples one), a --devices cpu run still has no device
    figures: whatever the GPU did meanwhile is not the server's work."""

    def must_not_run(own: set[int]) -> list[dict[str, Any]]:
        raise AssertionError("foreign_gpu_processes was called for a --devices cpu run")

    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", must_not_run)
    monkeypatch.setattr(profile_sampling, "open_device_probe", _FakeProbe)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["gpu_guard"] is False and doc["void"] == []
    assert doc["sampling"]["device_samples"] > 0  # sampled, and deliberately not reported
    for name in ("device_memory", "gpu_utilisation"):
        assert doc["figures"][name] is None and doc["unavailable"][name] == "--devices cpu"
    lines = captured.out.splitlines()
    assert _cells(lines[0])[5] == profile._memory_cell(doc["figures"]["host_memory"])  # host RSS, not "device ..."
    assert lines[1:] == [f"profile.json: {out / 'profile.json'}"]


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
    assert doc["figures"]["device_memory"]["partial"] is False
    # nv0's used_bytes never rises above its baseline, so the first (only) id is the serving GPU.
    assert doc["figures"]["gpu_utilisation"]["median_percent"] == 40
    assert doc["figures"]["gpu_utilisation"]["gpus"] == ["nv0"]
    lines = captured.out.splitlines()
    cells = _cells(lines[0])
    assert cells[2] == "void, 12 pages"
    assert cells[5] == "device 3.0 / 3.0 GiB"
    assert lines[1] == (f"host RSS peak / steady {profile._memory_cell(doc['figures']['host_memory'])}, "
                        "median GPU utilisation 40 %")
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
    assert "interrupted; the server is stopped" in captured.err
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


# --------------------------------------------------------------------------- #
# hardening before the real measurements (KON-210)
# --------------------------------------------------------------------------- #
def _ps_field(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def _poll(condition: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_host_without_ps_is_refused_before_any_probe_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_which = shutil.which
    monkeypatch.setattr(profile.shutil, "which",
                        lambda name, *a, **k: None if name == "ps" else real_which(name, *a, **k))
    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: pytest.fail("the device probe was opened"))
    monkeypatch.setattr(profile._Server, "start", _no_spawn)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe="/nonexistent/max")
    assert code == 2
    assert "`ps` is required (install procps)" in capsys.readouterr().err
    assert not out.exists()


def test_the_leader_exit_is_seen_without_reaping_it_so_its_group_stays_pinned(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """exit-early: the stub exits and leaves its child in its process group. ``waitid(WNOWAIT)``
    sees the exit, yet the leader stays an unreaped zombie -- so its pid is still the group id,
    the group signal reaches the child, and the straggler pass finds nothing left."""
    monkeypatch.setenv("PROFILE_STUB_MODE", "exit-early")
    server = profile._Server([stub.exe, "serve", "--port", str(_free_port())], dict(os.environ),
                             tmp_path / "serve.log", _NoDeviceProbe())
    server.start()
    try:
        assert _poll(server.leader_exited)
        pids = stub.pids()
        assert server.proc.returncode is None  # seen, not reaped
        assert _ps_field(pids["pid"], "stat").startswith("Z")
        assert _ps_field(pids["child"], "pgid") == str(pids["pid"])
        os.killpg(pids["pid"], 0)  # the group can still be signalled
    finally:
        server.stop()
    assert server.proc.returncode == 1  # reaped, at the very end of the teardown
    assert _tree_gone(stub)
    assert "still alive after its process group was killed" not in capsys.readouterr().err


def test_a_stopped_leader_is_not_taken_for_an_exited_one(tmp_path: Path) -> None:
    """macOS answers ``waitid(WEXITED)`` for a stopped child too (``CLD_STOPPED``); only an exit counts."""
    server = profile._Server(["sleep", "30"], dict(os.environ), tmp_path / "serve.log", _NoDeviceProbe())
    server.start()
    try:
        os.kill(server.proc.pid, signal.SIGSTOP)
        assert _poll(lambda: _ps_field(server.proc.pid, "stat").startswith("T"))
        assert not server.leader_exited()
        os.kill(server.proc.pid, signal.SIGCONT)
        assert not server.leader_exited()
    finally:
        server.stop()
    assert server.proc.returncode == -signal.SIGTERM
    assert server.leader_exited()


def test_a_server_that_exits_during_the_run_voids_it(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The port race: our server fails to bind and exits while whatever holds the port answers.
    Here every page is answered, and only the exit shows that the answers are not our server's."""
    monkeypatch.setenv("PROFILE_STUB_MODE", "exit-after-pages")
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 5, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["void"] == ["the server exited during the run; another process may have answered on the port"]
    assert doc["figures"]["text"]["identical"] == 12
    lines = captured.out.splitlines()
    assert _cells(lines[0])[2] == "void, 12 pages"
    assert lines[1] == f"void: {doc['void'][0]}"


def _gpu(used_gib: int, percent: int) -> dict[str, int]:
    return {"used_bytes": used_gib * GIB, "total_bytes": 80 * GIB, "gpu_usage_percent": percent}


def test_only_the_serving_gpus_count_towards_utilisation() -> None:
    baseline = {"nv0": _gpu(1, 0), "nv1": _gpu(1, 0)}
    stats = [
        (1.0, {"nv0": _gpu(1, 0), "nv1": _gpu(4, 90)}),
        (2.0, {"nv0": _gpu(1, 2), "nv1": _gpu(4, 80)}),
        (9.0, {"nv0": _gpu(1, 0), "nv1": _gpu(4, 10)}),  # outside every page request
    ]
    gpus = profile._serving_gpus(baseline, stats)
    assert gpus == ["nv1"]
    assert profile._gpu_utilisation(stats, [(0.5, 2.5)], gpus) == {"median_percent": 85, "n": 2, "gpus": ["nv1"]}
    # Every GPU that rose counts; with none risen, the first id does.
    both = [(1.0, {"nv0": _gpu(2, 50), "nv1": _gpu(4, 90)})]
    assert profile._serving_gpus(baseline, both) == ["nv0", "nv1"]
    assert profile._gpu_utilisation(both, [(0.5, 2.5)], ["nv0", "nv1"])["median_percent"] == 70
    assert profile._serving_gpus(baseline, [(1.0, baseline)]) == ["nv0"]
    assert profile._serving_gpus({}, []) == []


def test_a_gpu_run_counts_only_the_serving_gpu_and_puts_device_memory_in_the_row(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_PREFILL_S", "0.3")  # page windows long enough for several device samples
    monkeypatch.setattr(profile_sampling, "open_device_probe", _TwoGpuProbe)
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", lambda own: [])
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    utilisation = doc["figures"]["gpu_utilisation"]
    assert utilisation["gpus"] == ["nv1"] and utilisation["median_percent"] == 90  # not 45: idle nv0 is left out
    assert utilisation["partial"] is False
    assert doc["figures"]["host_memory"] is not None  # JSON keeps both memories
    lines = captured.out.splitlines()
    assert _cells(lines[0])[5] == "device 3.0 / 3.0 GiB"
    assert lines[1] == (f"host RSS peak / steady {profile._memory_cell(doc['figures']['host_memory'])}, "
                        "median GPU utilisation 90 %")


def test_device_figures_are_marked_partial_when_device_sampling_stops_mid_run(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_STUB_PREFILL_S", "0.3")
    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: _ProbeFailingMidRun(stub))
    monkeypatch.setattr(profile_sampling, "foreign_gpu_processes", lambda own: [])
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port(), devices="gpu"), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err  # partial is marked, not void
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["sampling"]["device_error"] == "the device went away"
    for name in ("device_memory", "gpu_utilisation"):
        figure = doc["figures"][name]
        assert figure is not None, doc["unavailable"]
        assert figure["partial"] is True and figure["device_error"] == "the device went away"
    assert captured.out.splitlines()[1].endswith(
        ", median GPU utilisation 40 % (partial: device sampling stopped: the device went away)"
    )


def test_the_row_memory_cell_is_device_memory_where_it_was_measured_else_host_rss() -> None:
    host = {"peak_bytes": 2 * GIB, "steady_min_bytes": GIB, "steady_max_bytes": GIB}
    device = {"peak_bytes": 3 * GIB, "steady_min_bytes": GIB * 5 // 2, "steady_max_bytes": GIB * 5 // 2}
    discrete = profile.row("hw", "bf16", "profiled, 12 pages", {"host_memory": host, "device_memory": device})
    metal_or_cpu = profile.row("hw", "bf16", "profiled, 12 pages", {"host_memory": host, "device_memory": None})
    assert _cells(discrete)[5] == "device 3.0 / 2.5 GiB"
    assert _cells(metal_or_cpu)[5] == "2.0 / 1.0 GiB"


def test_swap_growth_flags_the_timings_without_voiding_the_run(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    readings = iter([GIB])  # read once before the server starts; 200 MiB more on every sample after
    monkeypatch.setattr(profile_sampling, "swap_used_bytes", lambda: next(readings, GIB + 200 * 2**20))
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["void"] == []
    assert doc["swap"] == {"start_bytes": GIB, "max_bytes": GIB + 200 * 2**20, "growth_bytes": 200 * 2**20,
                           "reason": None}
    lines = captured.out.splitlines()
    assert _cells(lines[0])[2] == "profiled, 12 pages, swap moved"
    assert doc["row"] == lines[0]
    assert "note: timings are unreliable: system swap grew 0.20 GiB during the run" in lines


def test_unreadable_swap_is_null_with_its_reason(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unreadable() -> int:
        raise OSError("no swap accounting here")

    monkeypatch.setattr(profile_sampling, "swap_used_bytes", unreadable)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert doc["swap"] == {"start_bytes": None, "max_bytes": None, "growth_bytes": None,
                           "reason": "system swap cannot be read: no swap accounting here"}
    assert doc["sampling"]["swap_samples"] == 0
    lines = captured.out.splitlines()
    assert _cells(lines[0])[2] == "profiled, 12 pages"
    assert not any(line.startswith("note:") for line in lines)


def _ce_line(execution: str) -> str:
    return (f"12:00:00.000 INFO: Executed CE batch with 1 reqs | Batch creation: 1.00ms, Execution: {execution} | "
            "KVCache usage: 18.8% of 16 blocks")


def test_prefill_all_ce_is_the_median_over_every_ce_line_the_warmups_included() -> None:
    figures = profile._Figures()
    profile._scheduler_figures(figures, "\n".join(_ce_line(v) for v in ("30.00s", "1.00s", "2.00s", "3.00s")))
    assert figures.values["prefill"] == {"median_s": 2.0, "floor_s": 1.0, "n": 3}  # the warmup's line skipped
    assert figures.values["prefill_all_ce"] == {"median_s": 2.5, "floor_s": 1.0, "n": 4}  # median(30, 1, 2, 3)
    assert _cells(profile.row("hw", "bf16", "s", figures.values))[4] == "2.00 s"  # the row keeps the primary


def test_the_stop_message_is_said_while_signals_are_held(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, Any]] = []
    monkeypatch.setattr(profile, "_say", lambda message: events.append((message, signal.getsignal(signal.SIGINT))))
    monkeypatch.setattr(profile._Server, "_teardown",
                        lambda self: events.append(("teardown", signal.getsignal(signal.SIGINT))))
    server = profile._Server(["unused"], {}, Path("unused.log"), _NoDeviceProbe())
    server.proc = object()  # type: ignore[assignment]  # "started"
    server.stop()
    assert [event for event, _ in events] == ["stopping the server", "teardown"]
    held = events[0][1]
    assert isinstance(getattr(held, "__self__", None), profile._HeldSignals)
    assert events[1][1] == held


class _FakeSampler:
    def __init__(self, alive: bool) -> None:
        self.alive = alive


@pytest.mark.parametrize(("stopped", "sampler_alive", "interrupted", "said", "probe_closed"), [
    (True, False, True, "interrupted; the server is stopped", True),
    (False, True, True, "interrupted; the server's teardown did not finish, and continues at exit", False),
    (True, True, False, None, False),  # the sampler thread outlived its stop bound
    (True, False, False, None, True),
])
def test_an_interrupt_is_reported_and_the_probe_closed_only_as_far_as_the_teardown_got(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    stopped: bool, sampler_alive: bool, interrupted: bool, said: str | None, probe_closed: bool,
) -> None:
    probe = _FakeProbe()
    monkeypatch.setattr(profile_sampling, "open_device_probe", lambda: probe)

    def fake_profile(args: argparse.Namespace, server: profile._Server, out: Path, **kwargs: Any) -> int:
        server.proc = object()  # type: ignore[assignment]  # started
        server._stopped = stopped
        server.sampler = _FakeSampler(sampler_alive)  # type: ignore[assignment]
        if interrupted:
            raise KeyboardInterrupt
        return profile.EXIT_OK

    monkeypatch.setattr(profile, "_profile", fake_profile)
    code = profile.run(_args(tmp_path / "run", _free_port()), max_exe="/nonexistent/max")
    err = capsys.readouterr().err
    assert code == (130 if interrupted else 0)
    assert [line for line in err.splitlines() if "interrupted" in line] == (
        [f"[unlimited-ocr-max profile] {said}"] if said else []
    )
    assert probe.closed is probe_closed


@pytest.mark.parametrize(("mode", "why"), [
    ("choices-dict", "choices is not a list"),
    ("choices-empty", "an event with neither choices nor usage"),
    ("content-not-text", "delta content is not text"),
])
def test_a_malformed_stream_fails_the_request_instead_of_crashing(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    mode: str, why: str,
) -> None:
    monkeypatch.setenv("PROFILE_STUB_MODE", mode)
    out = tmp_path / "run"
    code = profile.run(_args(out, _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 5, captured.err
    assert _tree_gone(stub)
    doc = json.loads((out / "profile.json").read_text())
    assert len(doc["void"]) == 1 and doc["void"][0].startswith(f"request warmup failed: {why}: "), doc["void"]


def test_each_teardown_check_reads_one_ps_snapshot(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tree membership is walked in the same ``ps`` snapshot the alive/pgid checks read. ``detach``
    runs every check: the group kill, then the straggler pass for the child that left the group."""
    monkeypatch.setenv("PROFILE_STUB_MODE", "detach")
    snapshots: list[int] = []
    real_table = profile._ps_table

    def counted_table() -> dict[int, Any]:
        snapshots.append(1)
        return real_table()

    main_thread_walks: list[int] = []
    real_tree = profile_sampling.process_tree

    def watched_tree(root: int) -> dict[int, int]:
        if threading.current_thread() is threading.main_thread():  # the sampler thread keeps its own
            main_thread_walks.append(root)
        return real_tree(root)

    monkeypatch.setattr(profile, "_ps_table", counted_table)
    monkeypatch.setattr(profile_sampling, "process_tree", watched_tree)
    per_check: dict[str, list[int]] = {}
    for name in ("_record_identities", "_group_alive", "_verified_alive", "leader_exited"):
        def counting(self: Any, *args: Any, _original: Any = getattr(profile._Server, name), _name: str = name) -> Any:
            before = len(snapshots)
            try:
                return _original(self, *args)
            finally:
                per_check.setdefault(_name, []).append(len(snapshots) - before)

        monkeypatch.setattr(profile._Server, name, counting)

    code = profile.run(_args(tmp_path / "run", _free_port()), max_exe=stub.exe)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"SIGTERM to pid(s) [{stub.pids()['child']}]" in captured.err  # the straggler pass ran
    assert _tree_gone(stub)
    assert per_check["_record_identities"] == [1]
    assert per_check["_group_alive"] and set(per_check["_group_alive"]) == {1}
    assert len(per_check["_verified_alive"]) >= 2 and set(per_check["_verified_alive"]) == {1}
    assert set(per_check["leader_exited"]) == {0}  # waitid, no ps at all
    assert len(snapshots) == sum(map(sum, per_check.values()))  # no other snapshot was taken
    assert main_thread_walks == []
