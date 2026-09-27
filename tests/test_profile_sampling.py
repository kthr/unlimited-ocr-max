"""Process-tree RSS, per-process device memory, and the idle-GPU guard (KON-204).

This development machine is an Apple M4 (Metal): ``max.profiler.gpu``'s own
``GPUDiagContext().get_stats()`` returns ``{}`` here and there is no
``nvidia-smi``/``amd-smi``/``rocm-smi`` on ``PATH``. Every device-facing path
is therefore exercised through a fake :class:`~unlimited_ocr_max.profile_sampling.DeviceProbe`
or monkeypatched ``shutil.which``/``subprocess.run`` rather than real hardware;
only :func:`process_tree` and :class:`Sampler`'s RSS path get a real process
tree, because those do not need a GPU at all.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from unlimited_ocr_max import profile_sampling
from unlimited_ocr_max.profile_sampling import (
    DeviceProbe,
    GpuProcessListUnavailable,
    Sampler,
    foreign_gpu_processes,
    open_device_probe,
    process_tree,
)

ROOT = Path(__file__).resolve().parent.parent

_POLL_TIMEOUT_S = 10.0


def _poll_until(condition, timeout: float = _POLL_TIMEOUT_S) -> bool:
    """Poll ``condition`` (a zero-arg callable) every 10ms until true or ``timeout`` elapses.

    Returns whatever the last call to ``condition`` returned (truthy on
    success, falsy on timeout) so callers can ``assert _poll_until(...)`` with
    a useful failure. Used in place of a fixed ``time.sleep`` window
    everywhere a test waits on the Sampler background thread to have taken a
    certain number of ticks -- a fixed sleep is either flaky under load (too
    short) or slow (padded long), where polling is both robust and fast.
    """
    deadline = time.monotonic() + timeout
    result = condition()
    while not result and time.monotonic() < deadline:
        time.sleep(0.01)
        result = condition()
    return result


class _FakeProbe:
    """A minimal stand-in for DeviceProbe: same three members Sampler uses."""

    def __init__(self, per_pid: dict[int, int], fail_after: int | None = None) -> None:
        self.device_available = True
        self._per_pid = per_pid
        self._fail_after = fail_after
        self._calls = 0

    def process_bytes(self, pids) -> int | None:
        self._calls += 1
        if self._fail_after is not None and self._calls > self._fail_after:
            raise RuntimeError("device probe went away mid-run")
        return sum(self._per_pid.get(pid, 0) for pid in pids)

    def stats(self) -> dict[str, dict[str, int]]:
        return {"nv0": {"used_bytes": 123, "total_bytes": 456, "gpu_usage_percent": 7}}


class _FakeProbeStatsFails:
    """process_bytes always succeeds; stats() starts raising after ``fail_after`` calls.

    Used to test that a ``stats()`` failure right after a successful
    ``process_bytes()`` on the SAME tick leaves ``device_process`` and
    ``device_stats`` equal length -- neither call's result is appended unless
    both succeed.
    """

    def __init__(self, fail_after: int) -> None:
        self.device_available = True
        self._fail_after = fail_after
        self._calls = 0

    def process_bytes(self, pids) -> int | None:
        return 0

    def stats(self) -> dict[str, dict[str, int]]:
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("stats went away mid-run")
        return {"nv0": {"used_bytes": 1, "total_bytes": 2, "gpu_usage_percent": 3}}


def test_process_tree_sees_a_real_grandchild_and_root_gone_is_empty() -> None:
    """A shell spawns a python grandchild in its OWN process group; the parent/child walk still sees it.

    The grandchild backgrounds itself into a fresh process group
    (``os.setpgid(0, 0)``) to reproduce the exact shape that broke a
    process-group-based walk (a ``uv run`` child does the same). It then
    allocates and touches ~200 MB after a short delay, so a ``before``
    snapshot and a later ``after`` snapshot bracket the rise.

    Both snapshots are polled for, not waited-for on a fixed clock: a fixed
    2.2s window was measured to catch the allocation only ~45 MiB into its
    200 MiB (reproduced on a back-to-back full-suite run under load) --
    exactly the kind of flakiness a tick-count/condition poll avoids.
    """
    child_code = textwrap.dedent(
        """
        import os, time
        os.setpgid(0, 0)
        time.sleep(0.5)
        buf = bytearray(200 * 2**20)
        for i in range(0, len(buf), 4096):
            buf[i] = 1
        time.sleep(10)
        """
    )
    shell_cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(child_code)} & wait"
    proc = subprocess.Popen(["/bin/sh", "-c", shell_cmd])
    try:
        assert _poll_until(lambda: proc.pid in process_tree(proc.pid)), "root pid never appeared in ps"
        before = process_tree(proc.pid)
        assert proc.pid in before

        snapshots: dict[str, dict[int, int]] = {}

        def risen_enough() -> bool:
            after = process_tree(proc.pid)
            snapshots["after"] = after
            grandchild_pids = set(after) - {proc.pid}
            risen = sum(after.values()) - sum(before.values())
            return bool(grandchild_pids) and risen >= 150 * 2**20

        assert _poll_until(risen_enough, timeout=20.0), (
            f"tree RSS never rose by 150 MiB; last snapshot: {snapshots.get('after')}"
        )
        after = snapshots["after"]

        assert len(after) >= 2, "expected the shell plus its python grandchild"
        grandchild_pids = set(after) - {proc.pid}
        assert grandchild_pids, "no grandchild pid seen in the tree"

        risen = sum(after.values()) - sum(before.values())
        assert risen >= 150 * 2**20, f"tree RSS rose by only {risen} bytes"
    finally:
        for pid in process_tree(proc.pid):
            if pid != proc.pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.kill()
        proc.wait(timeout=5)

    assert process_tree(proc.pid) == {}


def test_process_tree_unknown_root_is_empty() -> None:
    """A pid that (almost certainly) never existed yields an empty tree, not an error."""
    assert process_tree(2**30 - 1) == {}


def test_open_device_probe_on_this_metal_host_is_unavailable() -> None:
    """No monkeypatching: this machine really has no NVML/ROCm-SMI GPU."""
    probe = open_device_probe()
    try:
        assert probe.device_available is False
        assert probe.stats() == {}
        assert probe.process_bytes([os.getpid()]) is None
    finally:
        probe.close()
        probe.close()  # idempotent


def test_sampler_on_a_real_child_records_rss_with_no_device_available() -> None:
    """A Sampler over a real child process, with the real (unavailable) device probe."""
    child = subprocess.Popen(["sleep", "5"])
    probe = None
    try:
        probe = open_device_probe()
        assert probe.device_available is False
        with Sampler(child.pid, probe, interval_s=0.1) as sampler:
            assert _poll_until(lambda: len(sampler.rss) >= 2)
        assert len(sampler.rss) >= 2
        assert child.pid in sampler.pids_seen
        assert sampler.device_process == []
        assert sampler.device_stats == []
        assert sampler.device_error is None
        assert sampler.stop_timed_out is False
    finally:
        child.kill()
        child.wait(timeout=5)
        if probe is not None:
            probe.close()


def test_sampler_device_process_is_the_sum_over_tree_pids() -> None:
    """.device_process values equal Σ over the sampled tree's pids, from the fake probe's own map."""
    child = subprocess.Popen(["sleep", "5"])
    try:
        probe = _FakeProbe(per_pid={child.pid: 999_999})
        with Sampler(child.pid, probe, interval_s=0.05) as sampler:
            assert _poll_until(lambda: len(sampler.device_process) >= 2)
        assert len(sampler.device_process) >= 2
        assert all(value == 999_999 for _, value in sampler.device_process)
        assert len(sampler.device_stats) == len(sampler.device_process)
        assert sampler.device_error is None
    finally:
        child.kill()
        child.wait(timeout=5)


def test_sampler_probe_failure_mid_run_sets_device_error_once_and_rss_continues() -> None:
    """A probe that starts failing after a few ticks stops device sampling but not RSS sampling."""
    child = subprocess.Popen(["sleep", "5"])
    try:
        probe = _FakeProbe(per_pid={child.pid: 42}, fail_after=2)
        with Sampler(child.pid, probe, interval_s=0.05) as sampler:
            assert _poll_until(lambda: sampler.device_error is not None)
            device_samples_at_failure = len(sampler.device_process)
            # Let RSS sampling take a few more ticks past where device sampling stopped.
            target_rss_samples = device_samples_at_failure + 3
            assert _poll_until(lambda: len(sampler.rss) >= target_rss_samples)
        assert sampler.device_error == "device probe went away mid-run"
        assert device_samples_at_failure > 0
        # RSS kept accumulating well past where device sampling stopped.
        assert len(sampler.rss) > device_samples_at_failure
    finally:
        child.kill()
        child.wait(timeout=5)


def test_sampler_stats_failure_after_process_bytes_keeps_lists_equal_length() -> None:
    """(e) stats() raising right after a successful process_bytes() on the same tick: lists stay equal length.

    Before the fix, ``device_process`` was appended before ``stats()`` ran, so
    a ``stats()`` failure left it one entry longer than ``device_stats``.
    """
    child = subprocess.Popen(["sleep", "5"])
    try:
        probe = _FakeProbeStatsFails(fail_after=2)
        with Sampler(child.pid, probe, interval_s=0.05) as sampler:
            assert _poll_until(lambda: sampler.device_error is not None)
        assert sampler.device_error == "stats went away mid-run"
        assert len(sampler.device_process) == len(sampler.device_stats) == 2
    finally:
        child.kill()
        child.wait(timeout=5)


def test_sampler_ps_failure_on_one_tick_is_skipped_and_rss_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(f) A single ``ps`` (``process_tree``) failure mid-run is skipped; RSS sampling keeps going after it."""
    child = subprocess.Popen(["sleep", "5"])
    real_process_tree = profile_sampling.process_tree
    calls = {"n": 0}

    def flaky_process_tree(root_pid: int) -> dict[int, int]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("ps went away for one tick")
        return real_process_tree(root_pid)

    monkeypatch.setattr(profile_sampling, "process_tree", flaky_process_tree)
    probe = open_device_probe()
    try:
        with Sampler(child.pid, probe, interval_s=0.05) as sampler:
            assert _poll_until(lambda: calls["n"] >= 3 and len(sampler.rss) >= 2)
        assert calls["n"] >= 3, "expected the flaky process_tree to be called past its one failure"
        # The failed tick (call #2) contributed no RSS sample, but ticks before and after it did.
        assert len(sampler.rss) >= 2
        assert child.pid in sampler.pids_seen
    finally:
        child.kill()
        child.wait(timeout=5)
        probe.close()


def test_private_max_profiler_gpu_process_api_pin() -> None:
    """Pin check for max[all]==26.6.0's private per-process GPU memory API.

    ``NVMLContext``/``RSMIContext`` and their ``get_process_memory_bytes``
    methods are private MAX modules that :func:`DeviceProbe.process_bytes`
    depends on directly. If this test fails after a MAX version bump, that
    API moved or disappeared upstream and ``profile_sampling.py`` needs a
    matching update -- not a silently-empty result. Needs no GPU: it is an
    import plus a ``hasattr`` check. Skipped only where MAX is not installed.
    """
    pytest.importorskip("max.profiler.gpu")
    from max.profiler.gpu._nvml import NVMLContext
    from max.profiler.gpu._rsmi import RSMIContext

    assert hasattr(NVMLContext, "get_process_memory_bytes")
    assert hasattr(RSMIContext, "get_process_memory_bytes")


def test_importing_profile_sampling_does_not_import_max() -> None:
    """A fresh interpreter, in a subprocess: importing the module alone must not pull in max."""
    code = "import sys, unlimited_ocr_max.profile_sampling; print('max' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.stdout.strip() == "False"


# --- foreign_gpu_processes -------------------------------------------------


def test_foreign_gpu_processes_parses_nvidia_csv_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Own pid excluded; foreign pid's bytes are MiB * 2**20; ``[N/A]`` is tolerated, not fatal."""
    own_pid = 111
    foreign_pid = 222
    na_pid = 333
    csv_out = (
        f"{own_pid}, python3, 512\n"
        f"{foreign_pid}, python3, 1024\n"
        f"{na_pid}, other, [N/A]\n"
    )

    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=csv_out, stderr=""),
    )

    result = foreign_gpu_processes({own_pid})

    assert all(p["pid"] != own_pid for p in result)
    assert {"pid": foreign_pid, "name": "python3", "used_bytes": 1024 * 2**20} in result
    na_entry = next(p for p in result if p["pid"] == na_pid)
    assert na_entry["used_bytes"] is None


def test_foreign_gpu_processes_takes_a_comma_in_the_nvidia_process_name_as_part_of_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvidia-smi does not quote the name: the pid is the first field, used_memory the last."""
    csv_out = "1234, /opt/a,b/python, 512\n1235, a, b, c, [N/A]\n"
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv"})
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=csv_out, stderr=""),
    )
    assert foreign_gpu_processes(set()) == [
        {"pid": 1234, "name": "/opt/a,b/python", "used_bytes": 512 * 2**20},
        {"pid": 1235, "name": "a, b, c", "used_bytes": None},
    ]


@pytest.mark.parametrize("bad_line", [
    "444, python3",                 # a field missing
    "445",                          # only a pid
    "[N/A], python3, 100",          # no pid
    "/opt/a,b/python, 446, 100",    # no pid first
])
def test_foreign_gpu_processes_refuses_a_malformed_nvidia_csv_line(
    monkeypatch: pytest.MonkeyPatch, bad_line: str
) -> None:
    """A line it cannot parse could be a foreign process: it raises (quoting the line), never skips it."""
    csv_out = f"222, python3, 1024\n{bad_line}\n"
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv"})
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=csv_out, stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable, match=re.escape(repr(bad_line))):
        foreign_gpu_processes(set())


def test_foreign_gpu_processes_empty_on_no_gpu_host_with_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: set())
    assert foreign_gpu_processes(set()) == []


def test_foreign_gpu_processes_raises_when_gpu_present_but_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv"})
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_foreign_gpu_processes_raises_when_vendors_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_reported_gpu_vendors`` returning ``None`` (vendor presence unknown) must raise immediately.

    Unknown presence is never satisfied by whatever tool happens to be on
    ``PATH`` -- not even one that runs cleanly and returns no processes -- so
    this must raise both with no tools present at all, and with
    ``nvidia-smi`` present and returning a clean empty list.
    """
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: None)

    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())

    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_reported_gpu_vendors_get_stats_failure_after_successful_enter_is_unknown_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(round-2 reproducer) ``GPUDiagContext`` constructs/enters fine but ``get_stats()`` raises.

    Exercises the REAL ``_reported_gpu_vendors`` body (not a monkeypatch of
    the function itself): a fake ``GPUDiagContext`` is installed at the
    import site the function actually uses, so ``get_stats()`` raising must
    make ``_reported_gpu_vendors`` return ``None`` -- and ``foreign_gpu_processes``
    must then raise, whether or not a vendor tool happens to be on ``PATH``.
    """
    gpu_module = pytest.importorskip("max.profiler.gpu")

    class _FakeGPUDiagContext:
        def __enter__(self) -> "_FakeGPUDiagContext":
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def get_stats(self) -> dict[str, Any]:
            raise RuntimeError("nvml went away")

    monkeypatch.setattr(gpu_module, "GPUDiagContext", _FakeGPUDiagContext)

    assert profile_sampling._reported_gpu_vendors() is None

    # No tools on PATH at all.
    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())

    # nvidia-smi present and returns a clean empty list -- still must raise:
    # unknown vendor presence is never satisfied by a tool being on PATH.
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_reported_gpu_vendors_import_failure_is_empty_not_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max.profiler.gpu`` genuinely failing to import means "none", not "unknown" -- ``set()``, not ``None``.

    Setting a module to ``None`` in ``sys.modules`` is what makes Python's
    import system raise ``ImportError`` (a ``ModuleNotFoundError``, its
    subclass) for it, without needing to uninstall anything real.
    """
    monkeypatch.setitem(sys.modules, "max.profiler.gpu", None)

    assert profile_sampling._reported_gpu_vendors() == set()

    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    assert foreign_gpu_processes(set()) == []


def test_foreign_gpu_processes_raises_nvidia_reported_only_rocm_smi_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) An NVIDIA GPU is reported, but only ``rocm-smi`` is on PATH -- must raise, not return ``[]``.

    A single combined "found some tool" flag would let ``rocm-smi``'s clean
    ``[]`` mask the missing ``nvidia-smi``; the guard is per vendor, so this
    must raise regardless of rocm-smi succeeding.
    """
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv"})
    monkeypatch.setattr(
        profile_sampling.shutil,
        "which",
        lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None,
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="{}", stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_foreign_gpu_processes_raises_amd_reported_only_nvidia_smi_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) An AMD GPU is reported, but only ``nvidia-smi`` is on PATH -- must raise."""
    own_pid = 111
    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"amd"})
    monkeypatch.setattr(
        profile_sampling.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes({own_pid})


def test_foreign_gpu_processes_both_vendors_reported_both_tools_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) Both vendors reported, both tools on PATH -- processes from both come back."""
    nvidia_pid, amd_pid = 201, 202
    nvidia_csv = f"{nvidia_pid}, python3, 1024\n"
    amd_payload = json.dumps(
        [
            {
                "gpu": 0,
                "process_list": [
                    {
                        "process_info": {
                            "pid": amd_pid,
                            "name": "other",
                            "memory_usage": {"vram_mem": {"value": 256, "unit": "MB"}},
                        }
                    }
                ],
            }
        ]
    )

    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv", "amd"})
    monkeypatch.setattr(
        profile_sampling.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("nvidia-smi", "amd-smi") else None,
    )

    def fake_run(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, 0, stdout=nvidia_csv, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=amd_payload, stderr="")

    monkeypatch.setattr(profile_sampling.subprocess, "run", fake_run)

    result = foreign_gpu_processes(set())
    pids = {p["pid"] for p in result}
    assert pids == {nvidia_pid, amd_pid}


def test_foreign_gpu_processes_nvidia_reported_rocm_smi_present_and_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) NVIDIA reported + nvidia-smi present; rocm-smi also present but returns ``[]`` -- NVIDIA result counts."""
    foreign_pid = 301
    nvidia_csv = f"{foreign_pid}, python3, 2048\n"

    monkeypatch.setattr(profile_sampling, "_reported_gpu_vendors", lambda: {"nv"})
    monkeypatch.setattr(
        profile_sampling.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("nvidia-smi", "rocm-smi") else None,
    )

    def fake_run(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, 0, stdout=nvidia_csv, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    monkeypatch.setattr(profile_sampling.subprocess, "run", fake_run)

    result = foreign_gpu_processes(set())
    assert result == [{"pid": foreign_pid, "name": "python3", "used_bytes": 2048 * 2**20}]


def test_foreign_gpu_processes_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="", stderr="no permission"),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_foreign_gpu_processes_raises_when_ps_binary_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``subprocess.run`` raising OSError (tool vanished between ``which`` and ``run``) is also fatal, not silent."""
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    )

    def fake_run(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(profile_sampling.subprocess, "run", fake_run)
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_foreign_gpu_processes_parses_amd_smi_json_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    own_pid = 10
    foreign_pid = 20
    payload = json.dumps(
        [
            {
                "gpu": 0,
                "process_list": [
                    {
                        "process_info": {
                            "pid": own_pid,
                            "name": "self",
                            "memory_usage": {"vram_mem": {"value": 100, "unit": "MB"}},
                        }
                    },
                    {
                        "process_info": {
                            "pid": foreign_pid,
                            "name": "other",
                            "memory_usage": {"vram_mem": {"value": 256, "unit": "MB"}},
                        }
                    },
                ],
            }
        ]
    )
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=payload, stderr=""),
    )

    result = foreign_gpu_processes({own_pid})
    assert result == [{"pid": foreign_pid, "name": "other", "used_bytes": 256 * 2**20}]


def test_foreign_gpu_processes_raises_on_unrecognised_amd_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profile_sampling.shutil, "which", lambda name: "/usr/bin/amd-smi" if name == "amd-smi" else None
    )
    monkeypatch.setattr(
        profile_sampling.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout='{"unexpected": true}', stderr=""),
    )
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


def test_device_probe_process_bytes_sums_across_both_vendor_contexts() -> None:
    """Whichever of NVML/ROCm-SMI initialised, process_bytes sums their per-pid answers; None results are skipped."""

    class _Ctx:
        def __init__(self, answers: dict[int, int | None]) -> None:
            self._answers = answers

        def get_process_memory_bytes(self, pid: int) -> int | None:
            return self._answers.get(pid)

    probe = DeviceProbe(diag=None, nvml=_Ctx({1: 100, 2: None}), rsmi=_Ctx({1: 5, 3: 7}))
    assert probe.process_bytes([1, 2, 3]) == 100 + 5 + 7

    probe_none_found = DeviceProbe(diag=None, nvml=_Ctx({}), rsmi=None)
    assert probe_none_found.process_bytes([1, 2]) is None

    probe_no_context = DeviceProbe(diag=None, nvml=None, rsmi=None)
    assert probe_no_context.process_bytes([1]) is None


# --- _try_open --------------------------------------------------------------


def test_try_open_enter_failure_is_cleaned_up_and_returns_none() -> None:
    """Construct succeeds but ``__enter__`` raises: ``_try_open`` returns ``None`` and still calls ``__exit__``.

    Without the cleanup call, a context that acquired a resource in
    ``__init__`` (or partially, before ``__enter__`` blew up) would leak it --
    ``_try_open`` must not let a failed ``__enter__`` skip ``__exit__``.
    """
    exit_calls: list[tuple[object, object, object]] = []

    class _EntersBadly:
        def __enter__(self) -> "_EntersBadly":
            raise RuntimeError("enter blew up")

        def __exit__(self, *exc_info: object) -> None:
            exit_calls.append(exc_info)

    result = profile_sampling._try_open(_EntersBadly)

    assert result is None
    assert len(exit_calls) == 1


# --- Sampler.stop() bound ----------------------------------------------------


def test_sampler_stop_is_bounded_and_flags_timeout_when_a_tick_blocks() -> None:
    """A probe whose ``stats()`` blocks forever: ``stop()`` still returns within its bound.

    ``entered`` (not a fixed sleep) is the synchronization: the test waits
    for confirmation that the sampler thread is actually stuck inside
    ``stats()`` before timing ``stop()``, so the test cannot be flaky about
    whether the block was reached in time.
    """
    entered = threading.Event()
    release = threading.Event()

    class _BlockingProbe:
        device_available = True

        def process_bytes(self, pids: object) -> int:
            return 0

        def stats(self) -> dict[str, dict[str, int]]:
            entered.set()
            release.wait()
            return {}

    child = subprocess.Popen(["sleep", "5"])
    sampler = Sampler(child.pid, _BlockingProbe(), interval_s=0.01)
    try:
        sampler.__enter__()
        assert entered.wait(timeout=_POLL_TIMEOUT_S), "tick never reached the blocking stats() call"

        bound = max(5.0, 10 * sampler.interval_s)
        started = time.monotonic()
        sampler.stop()
        elapsed = time.monotonic() - started

        assert sampler.stop_timed_out is True
        assert elapsed < bound + 2.0, f"stop() took {elapsed}s, expected to return near the {bound}s bound"
    finally:
        release.set()  # let the blocked (daemon) thread finish so it doesn't linger.
        sampler._thread.join(timeout=5)
        child.kill()
        child.wait(timeout=5)
    assert sampler.alive is False


# --- system swap ---------------------------------------------------------------


def test_swap_used_from_macos_swapusage() -> None:
    parse = profile_sampling._swap_used_from_swapusage
    assert parse("total = 3072.00M  used = 1776.31M  free = 1295.69M  (encrypted)") == round(1776.31 * 2**20)
    # sysctl follows the locale's decimal separator (de_DE, fr_FR, ...) unless LC_ALL=C.
    assert parse("total = 3072,00M  used = 1776,31M  free = 1295,69M  (encrypted)") == round(1776.31 * 2**20)
    assert parse("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)\n") == 0
    assert parse("total = 4.00G  used = 1.50G  free = 2.50G") == 3 * 2**29
    assert parse("total = 4,00G  used = 1,50G  free = 2,50G") == 3 * 2**29
    with pytest.raises(ValueError, match="unrecognised vm.swapusage"):
        parse("vm.swapusage: unknown oid")


def test_swap_used_bytes_asks_sysctl_in_the_c_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        stdout = "total = 3072,00M  used = 1776,31M  free = 1295,69M  (encrypted)"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    monkeypatch.setattr(profile_sampling.sys, "platform", "darwin")
    monkeypatch.setattr(profile_sampling.subprocess, "run", run)
    assert profile_sampling.swap_used_bytes() == round(1776.31 * 2**20)
    (call,) = calls
    assert call["argv"] == ["sysctl", "-n", "vm.swapusage"]
    assert call["env"]["LC_ALL"] == "C"
    assert {k: v for k, v in call["env"].items() if k != "LC_ALL"} == {
        k: v for k, v in os.environ.items() if k != "LC_ALL"
    }


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sysctl")
def test_swap_used_bytes_reads_this_mac_under_a_comma_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    used = profile_sampling.swap_used_bytes()
    assert isinstance(used, int) and used >= 0


def test_swap_used_from_linux_meminfo() -> None:
    parse = profile_sampling._swap_used_from_meminfo
    meminfo = (
        "MemTotal:       16384000 kB\nSwapCached:        1024 kB\n"
        "SwapTotal:       2097148 kB\nSwapFree:        1048572 kB\n"
    )
    assert parse(meminfo) == (2097148 - 1048572) * 1024
    with pytest.raises(ValueError, match="SwapTotal/SwapFree"):
        parse("MemTotal:       16384000 kB\n")


@pytest.mark.skipif(not (sys.platform == "darwin" or sys.platform.startswith("linux")), reason="macOS/Linux only")
def test_swap_used_bytes_reads_this_host() -> None:
    used = profile_sampling.swap_used_bytes()
    assert isinstance(used, int) and used >= 0


def test_sampler_records_swap_and_a_failing_swap_read_stops_only_swap_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([5 * 2**20, 6 * 2**20])

    def swap() -> int:
        value = next(readings, None)
        if value is None:
            raise OSError("swap went away")
        return value

    monkeypatch.setattr(profile_sampling, "swap_used_bytes", swap)
    child = subprocess.Popen(["sleep", "5"])
    probe = open_device_probe()
    try:
        with Sampler(child.pid, probe, interval_s=0.05) as sampler:
            assert _poll_until(lambda: sampler.swap_error is not None and len(sampler.rss) >= 5)
        assert [value for _, value in sampler.swap] == [5 * 2**20, 6 * 2**20]
        assert sampler.swap_error == "swap went away"
        assert len(sampler.rss) > len(sampler.swap)  # host RSS sampling went on
    finally:
        child.kill()
        child.wait(timeout=5)
        probe.close()
