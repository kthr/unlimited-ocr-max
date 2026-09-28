"""Timestamped host and device samplers for ``unlimited-ocr-max profile`` (KON-204).

Two things a profiling run needs that ``max.profiler.gpu.BackgroundRecorder``
does not give: a process-tree RSS sample that survives a runtime whose worker
is a *grandchild* rather than a direct child (a ``uv run`` wrapper reparents
into its own process group -- a process-*group* walk has been measured to
report 0 bytes for exactly that shape), and samples that carry a timestamp at
all, so they can be aligned against a request window afterwards.
``BackgroundRecorder`` gives neither: it walks by process group, and its
samples are unstamped. Hence :func:`process_tree` (parent/child links, one
``ps`` call) and :class:`Sampler` (a plain background thread) instead. Its
tick also records the system's swap in use (:func:`swap_used_bytes`), so a
run whose timings were taken while the host was paging can say so.

On macOS the tick also sums the tree's **physical footprint**
(:func:`phys_footprint_bytes`). That, not RSS, is the memory figure on Apple
silicon: ``ps`` RSS leaves out every Metal allocation -- on unified memory the
model's weights and graphs -- and counts clean file-backed pages such as the
mmap'd checkpoint, so it moves with the host's free RAM rather than with the
server. The footprint is the kernel's own per-process accounting, the number
``footprint`` and ``vmmap`` print as "Physical footprint".

The owner's rule for device sampling: *"the measurement is only doable if
there are no other processes running on the GPU."* :func:`foreign_gpu_processes`
is the guard that rule needs -- it lists every compute process on any GPU
that is not one of the caller's own pids, so a profiling command can refuse
to report device numbers instead of silently attributing someone else's
workload to its own.

Every import of ``max.profiler.gpu`` (and its private ``_nvml``/``_rsmi``
submodules) is lazy -- inside a function -- so importing this module never
imports ``max``. That matters here specifically: this development machine is
an Apple M4 (Metal), where ``GPUDiagContext().get_stats()`` returns ``{}`` and
there is no ``nvidia-smi``/``amd-smi``/``rocm-smi`` on ``PATH``, so every
device-facing code path in this module is exercised in tests through fakes
and monkeypatching rather than real hardware.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from typing import Any

__all__ = [
    "DeviceProbe",
    "GpuProcessListUnavailable",
    "Sampler",
    "descendants",
    "FOOTPRINT_SUPPORTED",
    "foreign_gpu_processes",
    "open_device_probe",
    "phys_footprint_bytes",
    "process_tree",
    "swap_used_bytes",
]

#: Whether :func:`phys_footprint_bytes` can answer on this platform.
FOOTPRINT_SUPPORTED = sys.platform == "darwin"


class _RusageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` from ``<sys/resource.h>``: a 16-byte uuid, then 18 ``uint64_t``."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
            "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_RUSAGE_INFO_V2 = 2


@functools.cache
def _proc_pid_rusage() -> Any:
    """libproc's ``proc_pid_rusage``, bound on first use (libSystem is always loaded on macOS)."""
    fn = ctypes.CDLL(None, use_errno=True).proc_pid_rusage
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV2)]
    fn.restype = ctypes.c_int
    return fn


def phys_footprint_bytes(pid: int) -> int | None:
    """``pid``'s physical footprint in bytes (macOS only), or ``None`` if the process is gone.

    ``proc_pid_rusage(pid, RUSAGE_INFO_V2).ri_phys_footprint`` -- what ``footprint``
    and ``vmmap`` report as "Physical footprint": the process's dirty and
    compressed memory, Metal allocations included, and none of its clean
    file-backed pages. Readable for the caller's own processes without
    privileges. Raises :class:`OSError` for any failure other than ESRCH, and on
    a platform without it (:data:`FOOTPRINT_SUPPORTED`).
    """
    if not FOOTPRINT_SUPPORTED:
        raise OSError(f"physical footprint is not readable on {sys.platform}")
    info = _RusageInfoV2()
    if _proc_pid_rusage()(pid, _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        err = ctypes.get_errno()
        if err == errno.ESRCH:
            return None
        raise OSError(err, f"proc_pid_rusage({pid}): {os.strerror(err)}")
    return int(info.ri_phys_footprint)


def descendants(children: dict[int, list[int]], root: int) -> set[int]:
    """``root`` and everything reachable from it by walking ``children`` (``ppid -> [pid, ...]``)
    down, with an explicit stack so a cycle in ``children`` cannot loop forever (each pid is added
    to the result at most once, and only pushed again if not already in it).

    ``root`` is always included, whether or not it has an entry in ``children`` -- shared by both
    of this module's process-tree walks (:func:`process_tree` here; ``profile._tree`` builds
    ``children`` from its own ``ps`` columns and calls this too). Whether ``root`` itself should
    count at all (e.g. because it is not in the process snapshot ``children`` was built from) is
    for each caller to decide before calling this, from its own snapshot -- this function has no
    such snapshot to check against.
    """
    tree: set[int] = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid not in tree:
            tree.add(pid)
            stack.extend(children.get(pid, ()))
    return tree


def process_tree(root_pid: int) -> dict[int, int]:
    """Return ``{pid: rss_bytes}`` for ``root_pid`` and all of its descendants.

    Built from a single ``ps -A -o pid=,ppid=,rss=`` call (``rss`` is reported
    in KiB on both macOS and Linux, hence the ``* 1024``) by walking
    parent/child (``ppid``) links from ``root_pid`` down -- deliberately not
    the process *group*: a ``uv run`` child reparents into its own group, and
    a group-based walk of that shape has been measured to report 0 bytes.

    Raises whatever ``subprocess.run`` raises if ``ps`` itself cannot be run
    or exits non-zero; callers that sample repeatedly (:class:`Sampler`)
    decide how to treat that. If ``root_pid`` is not in the snapshot at all
    (already exited), the *call* still succeeds and this returns ``{}``.
    """
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,rss="],
        capture_output=True,
        text=True,
        check=True,
    )

    rss_kib_by_pid: dict[int, int] = {}
    children_by_ppid: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        rss_kib_by_pid[pid] = rss_kib
        children_by_ppid.setdefault(ppid, []).append(pid)

    if root_pid not in rss_kib_by_pid:
        return {}

    return {pid: rss_kib_by_pid[pid] * 1024 for pid in descendants(children_by_ppid, root_pid)}


class DeviceProbe:
    """Per-GPU stats and per-process device memory, backed by MAX's own GPU diagnostics.

    Construct through :func:`open_device_probe`, never directly -- the
    constructor takes already-entered (or ``None``) vendor contexts, which is
    exactly the shape ``open_device_probe`` builds after trying to import and
    initialise each of them.
    """

    def __init__(self, diag: Any, nvml: Any, rsmi: Any) -> None:
        self._diag = diag
        self._nvml = nvml
        self._rsmi = rsmi
        self.device_available = diag is not None

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-GPU id (``nv0``, ``amd0``, ...): used/total bytes and utilisation percent.

        ``{}`` when no device was available at :func:`open_device_probe` time.
        """
        if self._diag is None:
            return {}
        raw = self._diag.get_stats()
        return {
            gpu_id: {
                "used_bytes": gpu_stats.memory.used_bytes,
                "total_bytes": gpu_stats.memory.total_bytes,
                "gpu_usage_percent": gpu_stats.utilization.gpu_usage_percent,
            }
            for gpu_id, gpu_stats in raw.items()
        }

    def process_bytes(self, pids: Iterable[int]) -> int | None:
        """Sum of device memory attributed to any of ``pids``, across whichever vendor context initialised.

        ``None`` if neither the NVML nor the ROCm-SMI context is available, or
        if none of ``pids`` was found on any GPU by either.
        """
        pid_list = list(pids)
        total = 0
        found = False
        for ctx in (self._nvml, self._rsmi):
            if ctx is None:
                continue
            for pid in pid_list:
                used = ctx.get_process_memory_bytes(pid)
                if used is not None:
                    total += used
                    found = True
        return total if found else None

    def close(self) -> None:
        """Exit every vendor context this probe holds open. Idempotent."""
        for attr in ("_diag", "_nvml", "_rsmi"):
            ctx = getattr(self, attr)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                setattr(self, attr, None)

    def __enter__(self) -> DeviceProbe:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _try_open(ctx_factory: Any) -> Any:
    """Construct and ``__enter__`` a vendor context; ``None`` if either step fails.

    If construction succeeds but ``__enter__`` raises, the partially-acquired
    context still gets its ``__exit__`` called (best-effort) before the
    reference is dropped, so a failure after partial acquisition cannot leak
    whatever the context picked up.
    """
    try:
        ctx = ctx_factory()
    except Exception:
        return None
    try:
        ctx.__enter__()
    except Exception:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
        return None
    return ctx


def open_device_probe() -> DeviceProbe:
    """Build a :class:`DeviceProbe`, trying (and tolerating the absence of) every vendor backend.

    ``device_available`` ends up ``False`` when ``max.profiler.gpu`` cannot be
    imported or initialised at all, or when a successfully-entered
    ``GPUDiagContext`` reports no GPUs (``get_stats() == {}``, as on this
    Metal development machine and on any CPU-only host).
    """
    try:
        from max.profiler.gpu import GPUDiagContext
        from max.profiler.gpu._nvml import NVMLContext
        from max.profiler.gpu._rsmi import RSMIContext
    except Exception:
        return DeviceProbe(diag=None, nvml=None, rsmi=None)

    diag = _try_open(GPUDiagContext)

    if diag is not None:
        try:
            has_stats = bool(diag.get_stats())
        except Exception:
            has_stats = False
        if not has_stats:
            try:
                diag.__exit__(None, None, None)
            except Exception:
                pass
            diag = None

    nvml = _try_open(NVMLContext)
    rsmi = _try_open(RSMIContext)

    return DeviceProbe(diag=diag, nvml=nvml, rsmi=rsmi)


class GpuProcessListUnavailable(RuntimeError):
    """A GPU is present but its compute-process list could not be obtained.

    Raised instead of guessing whenever ``foreign_gpu_processes`` cannot get a
    trustworthy answer: no vendor tool on ``PATH`` despite a GPU being
    reported, a vendor tool that exited non-zero, or output in a shape its
    parser does not recognise.
    """


def _reported_gpu_vendors() -> set[str] | None:
    """Which vendors (a subset of ``{"nv", "amd"}``) MAX's own GPU diagnostics report a GPU for.

    ``GPUDiagContext().get_stats()`` keys are vendor-prefixed GPU ids
    (``"nv0"``, ``"amd0"``, ...; wheel ``max/profiler/gpu/multi.py``), so the
    vendor is just the non-digit prefix of each key. Used to decide, per
    vendor, whether that vendor's own tool is *required* to be on ``PATH`` --
    a single combined bool would let one vendor's present-but-unrelated tool
    (or its own GPU-less ``[]``) mask the other vendor's missing tool.

    Three distinct outcomes, not two -- "no GPU diagnostics installed" and
    "installed but broken" must not collapse into the same answer, because
    the caller (:func:`foreign_gpu_processes`) treats them differently:

    * ``max.profiler.gpu`` cannot be **imported** at all (``ImportError``) --
      no MAX GPU diagnostics on this host -- returns ``set()``: vendor
      presence is legitimately "none".
    * ``GPUDiagContext`` imports fine but constructing it, entering it, or
      calling ``get_stats()`` raises **anything** -- returns ``None``:
      vendor presence is *unknown*, not "none". The caller must never treat
      this the same as a genuinely GPU-less host.
    * Otherwise, the vendor-prefix set of whatever ``get_stats()`` returned
      (``{}`` -> ``set()``).
    """
    try:
        from max.profiler.gpu import GPUDiagContext
    except ImportError:
        return set()
    try:
        with GPUDiagContext() as ctx:
            gpu_ids = list(ctx.get_stats())
    except Exception:
        return None
    return {gpu_id.rstrip("0123456789") for gpu_id in gpu_ids}


def _run_smi(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True)
    except OSError as e:
        raise GpuProcessListUnavailable(f"{argv[0]} failed to run: {e}") from e
    if result.returncode != 0:
        raise GpuProcessListUnavailable(
            f"{argv[0]} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def _used_bytes_from_mib_field(field: str) -> int | None:
    """``used_memory`` from ``nvidia-smi --query-compute-apps``: MiB, or ``[N/A]``."""
    try:
        return int(field) * 2**20
    except ValueError:
        return None


def _nvidia_foreign_processes(own_pids: set[int]) -> list[dict[str, Any]]:
    raw = _run_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # A line in any other shape is refused, never skipped: a skipped line could be a
        # foreign process, and the guard must not under-report (the AMD parsers refuse too).
        # nvidia-smi does not quote the name, which may itself hold commas: the pid is the
        # first field, used_memory the last, and the name everything in between.
        fields = line.split(",")
        if len(fields) < 3:
            raise GpuProcessListUnavailable(
                f"unrecognised nvidia-smi line (expected pid, process_name, used_memory): {line!r}"
            )
        pid_field, name, mem_field = fields[0].strip(), ",".join(fields[1:-1]).strip(), fields[-1].strip()
        try:
            pid = int(pid_field)
        except ValueError:
            raise GpuProcessListUnavailable(f"unrecognised nvidia-smi pid in line: {line!r}") from None
        if pid in own_pids:
            continue
        processes.append(
            {
                "pid": pid,
                "name": name,
                "used_bytes": _used_bytes_from_mib_field(mem_field),
            }
        )
    return processes


_AMD_MEM_UNIT_SCALE = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}


def _amd_mem_bytes(memory_usage: Any) -> int | None:
    if not isinstance(memory_usage, dict):
        return None
    vram = memory_usage.get("vram_mem")
    if not isinstance(vram, dict) or "value" not in vram:
        return None
    scale = _AMD_MEM_UNIT_SCALE.get(str(vram.get("unit", "B")).upper())
    if scale is None:
        return None
    return int(vram["value"]) * scale


def _amd_smi_processes(raw: str) -> list[tuple[int, str, int | None]]:
    """Extract ``(pid, name, used_bytes)`` triples from ``amd-smi process --json``.

    Expected shape: a JSON list, one entry per GPU, each with a
    ``"process_list"`` of ``{"process_info": {"pid", "name", "memory_usage": {...}}}``.
    Anything else raises (``ValueError``/``KeyError``/``TypeError``), which the
    caller turns into :class:`GpuProcessListUnavailable`.
    """
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of per-GPU entries")
    triples: list[tuple[int, str, int | None]] = []
    for gpu_entry in data:
        for proc in gpu_entry["process_list"]:
            info = proc["process_info"]
            pid = int(info["pid"])
            name = str(info.get("name", ""))
            triples.append((pid, name, _amd_mem_bytes(info.get("memory_usage"))))
    return triples


def _rocm_smi_processes(raw: str) -> list[tuple[int, str, int | None]]:
    """Extract ``(pid, name, used_bytes)`` triples from ``rocm-smi --showpids --json``.

    Expected shape: a JSON object of sections, each a dict keyed by pid
    (digit strings) mapping to a process-info dict with a ``"Process name"``
    and a ``"VRAM Usage (B)"`` (or similarly-named) field. Anything else
    raises, turned into :class:`GpuProcessListUnavailable` by the caller.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    triples: list[tuple[int, str, int | None]] = []
    for section in data.values():
        if not isinstance(section, dict):
            continue
        for pid_field, info in section.items():
            if not isinstance(info, dict) or not str(pid_field).isdigit():
                continue
            name = str(info.get("Process name", info.get("process_name", "")))
            used_bytes = None
            for key in ("VRAM Usage (B)", "GPU Memory Usage (B)"):
                if key in info:
                    try:
                        used_bytes = int(info[key])
                    except (TypeError, ValueError):
                        used_bytes = None
                    break
            triples.append((int(pid_field), name, used_bytes))
    return triples


def _amd_foreign_processes(
    tool: str, argv: list[str], own_pids: set[int]
) -> list[dict[str, Any]]:
    raw = _run_smi(argv)
    try:
        triples = _amd_smi_processes(raw) if tool == "amd-smi" else _rocm_smi_processes(raw)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise GpuProcessListUnavailable(
            f"unrecognised {tool} JSON shape: {raw[:200]!r}"
        ) from e
    return [
        {"pid": pid, "name": name, "used_bytes": used_bytes}
        for pid, name, used_bytes in triples
        if pid not in own_pids
    ]


def foreign_gpu_processes(own_pids: set[int]) -> list[dict[str, Any]]:
    """Every compute process on any GPU whose pid is not in ``own_pids``.

    Each entry is ``{"pid": int, "name": str, "used_bytes": int | None}``.
    Tries ``nvidia-smi`` and an AMD tool (``amd-smi``, else ``rocm-smi``)
    independently and concatenates whichever are on ``PATH``, so a
    heterogeneous host is covered by both.

    The guard is per vendor, not a single combined flag: :func:`_reported_gpu_vendors`
    says which of ``{"nv", "amd"}`` MAX's own GPU diagnostics see a GPU for,
    and EACH reported vendor's own tool must be present *and* succeed, or this
    raises :class:`GpuProcessListUnavailable` -- regardless of what the other
    vendor's tool reported. So an NVIDIA GPU plus only ``rocm-smi`` on
    ``PATH`` raises, even though ``rocm-smi`` runs fine and returns ``[]``; a
    vendor tool that happens to be present with no GPU of that vendor
    reported is still consulted (its results are concatenated in), but its
    ``[]`` never satisfies the other vendor's requirement.

    If :func:`_reported_gpu_vendors` cannot even determine which vendors are
    present (``None`` -- MAX's own GPU diagnostics are installed but raised),
    this raises immediately, before looking at ``PATH`` at all: an unknown
    vendor set must never be satisfied by whatever tool happens to be lying
    around, however cleanly that tool runs.

    Returns ``[]`` only when no vendor tool is on ``PATH`` *and* no vendor is
    reported at all (a Metal or CPU-only host). A present tool that exits
    non-zero, or returns output in an unrecognised shape, also raises.
    """
    reported_vendors = _reported_gpu_vendors()
    if reported_vendors is None:
        raise GpuProcessListUnavailable(
            "cannot determine which GPUs are present: "
            "max.profiler.gpu's GPUDiagContext failed to report device stats"
        )
    processes: list[dict[str, Any]] = []

    if shutil.which("nvidia-smi"):
        processes.extend(_nvidia_foreign_processes(own_pids))
    elif "nv" in reported_vendors:
        raise GpuProcessListUnavailable(
            "an NVIDIA GPU is reported but nvidia-smi is not on PATH"
        )

    if shutil.which("amd-smi"):
        processes.extend(_amd_foreign_processes("amd-smi", ["amd-smi", "process", "--json"], own_pids))
    elif shutil.which("rocm-smi"):
        processes.extend(
            _amd_foreign_processes("rocm-smi", ["rocm-smi", "--showpids", "--json"], own_pids)
        )
    elif "amd" in reported_vendors:
        raise GpuProcessListUnavailable(
            "an AMD GPU is reported but neither amd-smi nor rocm-smi is on PATH"
        )

    return processes


_SWAPUSAGE_USED = re.compile(r"\bused = ([0-9]+(?:[.,][0-9]+)?)([KMGT])\b")
_SWAPUSAGE_UNIT = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def _swap_used_from_swapusage(text: str) -> int:
    """Bytes of swap in use from macOS ``sysctl -n vm.swapusage``:
    ``total = 3072.00M  used = 1776.31M  free = 1295.69M  (encrypted)``. The decimal separator
    follows the locale (``1776,31M`` under ``de_DE``), so either is accepted."""
    match = _SWAPUSAGE_USED.search(text)
    if match is None:
        raise ValueError(f"unrecognised vm.swapusage: {text.strip()!r}")
    return round(float(match.group(1).replace(",", ".")) * _SWAPUSAGE_UNIT[match.group(2)])


def _swap_used_from_meminfo(text: str) -> int:
    """Bytes of swap in use from Linux ``/proc/meminfo``: ``SwapTotal`` - ``SwapFree`` (both in kB)."""
    kib: dict[str, int] = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name in ("SwapTotal", "SwapFree"):
            kib[name] = int(value.split()[0])
    if kib.keys() != {"SwapTotal", "SwapFree"}:
        raise ValueError("/proc/meminfo has no SwapTotal/SwapFree")
    return (kib["SwapTotal"] - kib["SwapFree"]) * 1024


def swap_used_bytes() -> int:
    """System-wide swap in use, in bytes (macOS and Linux). Raises when it cannot be read."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True, timeout=10,
            env={**os.environ, "LC_ALL": "C"},  # its numbers follow the locale otherwise
        )
        return _swap_used_from_swapusage(result.stdout)
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo", encoding="ascii", errors="replace") as meminfo:
            return _swap_used_from_meminfo(meminfo.read())
    raise OSError(f"system swap is not readable on {sys.platform}")


class Sampler:
    """Background thread sampling host RSS and (if available) device memory/utilisation.

    A context manager: enter to start the daemon sampling thread, exit (or
    call :meth:`stop`) to stop and join it. Each tick appends one timestamped
    sample -- ``(time.monotonic(), value)`` -- to :attr:`rss` (summed process-tree
    RSS bytes) and every pid seen to :attr:`pids_seen`. When ``probe.device_available``,
    it also appends to :attr:`device_process` (``probe.process_bytes`` over the
    tree's pids) and :attr:`device_stats` (``probe.stats()``). Every tick also
    appends the system's swap in use (:func:`swap_used_bytes`) to :attr:`swap`,
    and, on macOS, the tree's summed physical footprint to :attr:`footprint`
    (a pid that exits between the ``ps`` call and its read counts 0).

    A ``ps`` failure on a single tick is skipped (RSS sampling just continues
    on the next tick). A device-probe exception is recorded once, as a string,
    in :attr:`device_error`, and device sampling stops for the rest of the run
    -- RSS sampling is unaffected. A swap read that fails is recorded the same
    way, in :attr:`swap_error`, and stops swap sampling only; so does a failed
    footprint read, in :attr:`footprint_error`. The background thread never
    raises into the caller.

    :meth:`stop` is bounded: it never waits longer than
    ``max(5.0, 10 * interval_s)`` for the thread to finish, even if a tick is
    stuck inside a blocking probe call. If the thread is still alive once
    that bound elapses, :attr:`stop_timed_out` is set to ``True`` and
    :meth:`stop` returns anyway -- a slow or hung probe must not hang whatever
    called ``stop()`` (or exited the ``with`` block).
    """

    def __init__(self, root_pid: int, probe: DeviceProbe, interval_s: float = 0.5) -> None:
        self.root_pid = root_pid
        self.probe = probe
        self.interval_s = interval_s

        self.rss: list[tuple[float, int]] = []
        self.pids_seen: set[int] = set()
        self.device_process: list[tuple[float, int | None]] = []
        self.device_stats: list[tuple[float, dict[str, dict[str, int]]]] = []
        self.device_error: str | None = None
        self.swap: list[tuple[float, int]] = []
        self.swap_error: str | None = None
        self.footprint: list[tuple[float, int]] = []
        self.footprint_error: str | None = None
        self.stop_timed_out = False

        self._device_ok = True
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop and join the sampling thread, bounded. Safe to call more than once.

        Signals the stop event, then joins with a timeout of
        ``max(5.0, 10 * interval_s)`` so a tick stuck inside a blocking probe
        call cannot hang the caller forever. If the thread is still alive
        once that bound elapses, sets :attr:`stop_timed_out` and returns --
        the (daemon) thread is left to finish on its own whenever the probe
        call it is stuck in eventually returns.
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(5.0, 10 * self.interval_s))
            if self._thread.is_alive():
                self.stop_timed_out = True

    @property
    def alive(self) -> bool:
        """Whether the sampling thread is running -- and so may be inside a probe call right now."""
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(self.interval_s)

    def _tick(self) -> None:
        self._sample_swap()  # first, so its subprocess does not sit between `now` and the ps sample
        now = time.monotonic()
        try:
            tree = process_tree(self.root_pid)
        except Exception:
            # A ps failure on this tick is skipped, not fatal.
            return

        pids = list(tree.keys())
        self.pids_seen.update(pids)
        self.rss.append((now, sum(tree.values())))
        self._sample_footprint(now, pids)

        if self._device_ok and self.probe.device_available:
            try:
                process_bytes = self.probe.process_bytes(pids)
                stats = self.probe.stats()
            except Exception as e:
                self.device_error = str(e) or type(e).__name__  # never "": callers test it for None
                self._device_ok = False
            else:
                # Only append once BOTH calls succeeded, so device_process and
                # device_stats can never end up different lengths (a stats()
                # failure right after a successful process_bytes() must not
                # leave a process_bytes sample with no matching stats sample).
                self.device_process.append((now, process_bytes))
                self.device_stats.append((now, stats))

    def _sample_footprint(self, now: float, pids: list[int]) -> None:
        if not FOOTPRINT_SUPPORTED or self.footprint_error is not None:
            return
        try:
            total = sum(phys_footprint_bytes(pid) or 0 for pid in pids)
        except Exception as e:
            self.footprint_error = str(e) or type(e).__name__
            return
        self.footprint.append((now, total))

    def _sample_swap(self) -> None:
        if self.swap_error is not None:
            return
        try:
            used = swap_used_bytes()
        except Exception as e:
            self.swap_error = str(e) or type(e).__name__
            return
        self.swap.append((time.monotonic(), used))
