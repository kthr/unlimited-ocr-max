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
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

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


def test_process_tree_sees_a_real_grandchild_and_root_gone_is_empty() -> None:
    """A shell spawns a python grandchild in its OWN process group; the parent/child walk still sees it.

    The grandchild backgrounds itself into a fresh process group
    (``os.setpgid(0, 0)``) to reproduce the exact shape that broke a
    process-group-based walk (a ``uv run`` child does the same). It then
    allocates and touches ~200 MB after a short delay, so a sample taken
    before that delay and one taken after it bracket the rise.
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
        time.sleep(0.2)
        before = process_tree(proc.pid)
        assert proc.pid in before

        time.sleep(2.0)
        after = process_tree(proc.pid)

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
            time.sleep(0.5)
        assert len(sampler.rss) >= 2
        assert child.pid in sampler.pids_seen
        assert sampler.device_process == []
        assert sampler.device_stats == []
        assert sampler.device_error is None
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
            time.sleep(0.3)
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
            time.sleep(0.6)
        assert sampler.device_error == "device probe went away mid-run"
        device_samples_at_failure = len(sampler.device_process)
        assert device_samples_at_failure > 0
        # RSS kept accumulating well past where device sampling stopped.
        assert len(sampler.rss) > device_samples_at_failure
    finally:
        child.kill()
        child.wait(timeout=5)


def test_private_max_profiler_gpu_process_api_pin() -> None:
    """Pin check for max[all]==26.6.0's private per-process GPU memory API.

    ``NVMLContext``/``RSMIContext`` and their ``get_process_memory_bytes``
    methods are private MAX modules that :func:`DeviceProbe.process_bytes`
    depends on directly. If this test fails after a MAX version bump, that
    API moved or disappeared upstream and ``profile_sampling.py`` needs a
    matching update -- not a silently-empty result. Needs no GPU: it is an
    import plus a ``hasattr`` check.
    """
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


def test_foreign_gpu_processes_empty_on_no_gpu_host_with_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    monkeypatch.setattr(profile_sampling, "_gpu_reported", lambda: False)
    assert foreign_gpu_processes(set()) == []


def test_foreign_gpu_processes_raises_when_gpu_present_but_no_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_sampling.shutil, "which", lambda name: None)
    monkeypatch.setattr(profile_sampling, "_gpu_reported", lambda: True)
    with pytest.raises(GpuProcessListUnavailable):
        foreign_gpu_processes(set())


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
