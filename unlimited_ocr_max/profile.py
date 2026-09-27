"""``unlimited-ocr-max profile``: measure this port on the host it runs on (KON-206).

Starts its own ``max serve`` -- the command and environment ``serve`` would run -- sends the
bundled 12-page corpus (:mod:`.profile_corpus`) one page at a time, stops the server, and turns
the scheduler log, the host/device samples (:mod:`.profile_sampling`) and the returned text
into one row of the README's "Where it has run" table (:mod:`.profile_metrics`) plus a complete
``profile.json``.

Three rules shape this module:

* **It never measures someone else's server or someone else's GPU load.** A port that already
  answers is refused (exit 2). Every ``--devices gpu`` run is guarded: another process on the
  GPU -- or a GPU whose processes cannot be listed -- refuses the run before the server starts
  (exit 3) or voids it afterwards (exit 5).
* **It never leaves the server behind.** ``max serve`` runs in its own session, so a Ctrl-C in
  the terminal reaches only this process; one teardown -- from ``finally``, from ``atexit`` as a
  fallback, and on SIGTERM/SIGHUP as well as Ctrl-C -- kills the server's process group, then
  every process of its tree that left the group. Signals that arrive during the teardown are
  held until it has finished.
* **It never signals a pid it cannot prove is the server's.** The server process is reaped only
  at the very end of the teardown -- its exit is seen with ``waitid(..., WNOWAIT)``, which leaves
  it a zombie -- so until then its pid, which is also its process group id, cannot be reused,
  and ``pgid == server pid`` proves membership. A process outside the group is signalled only
  while it is a descendant of the server, or while it has the pid *and* start time it had when
  it verifiably belonged to the server. Each such check reads one ``ps`` snapshot.

Stdlib only at import, like :mod:`.cli`: MAX is touched only through the server subprocess,
:func:`.profile_sampling.open_device_probe`, and a short-lived subprocess that asks
``max.driver`` which accelerator this host has -- in a subprocess because constructing an
``Accelerator`` opens a device context, which must not live in this process next to the server.
"""

from __future__ import annotations

import atexit
import base64
import datetime as dt
import http.client
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import cli, profile_corpus, profile_metrics, profile_sampling

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

#: The prompt the pinned references were produced with (the research harness's ``OCR_PROMPT``).
PROMPT = "<|grounding|>Convert the document to markdown."
WARMUP_MAX_TOKENS = 8
PAGE_MAX_TOKENS = 8192
#: Per socket operation, not per request: a CPU prefill can keep the first byte away for minutes.
REQUEST_TIMEOUT_S = 3600.0
READY_POLL_S = 2.0
EXIT_POLL_S = 0.5
#: How long the server gets to exit on SIGTERM before its process group is SIGKILLed.
STOP_GRACE_S = 30.0
STRAGGLER_GRACE_S = 5.0
#: Idle time after the last page, so the steady-memory population is not empty.
STEADY_DWELL_S = 5.0
LOG_TAIL_LINES = 40
#: System swap growth over the run above which the timings are flagged (never voided).
SWAP_MOVED_BYTES = 0.05 * 2**30

EXIT_OK = 0
EXIT_PORT_IN_USE = 2
EXIT_NO_PS = 2
EXIT_GPU_BUSY = 3
EXIT_NOT_READY = 4
EXIT_VOID = 5
EXIT_INTERRUPTED = 130

SCHEMA = 1
_GIB = 2**30
_MIB = 2**20
#: ``waitid`` ``si_code`` values that mean the child has terminated.
_EXITED = (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)
#: No proxies: the server is on 127.0.0.1, and a system proxy must never sit in between.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_ACCELERATOR_PROBE = (
    "import json\n"
    "from max.driver import accelerator_api, accelerator_architecture_name\n"
    "print(json.dumps({'api': accelerator_api(), 'architecture': accelerator_architecture_name()}))\n"
)
_ROCM_SERIES = re.compile(r"^GPU\[(\d+)\]\s*:\s*Card [Ss]eries:\s*(.+?)\s*$", re.MULTILINE)


def _say(message: str) -> None:
    """Progress and diagnostics, on stderr. Never raises: it runs inside the teardown and inside
    signal handlers, where a closed terminal (after SIGHUP) or a reentrant write must not stop it."""
    try:
        print(f"[unlimited-ocr-max profile] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# processes and signals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Ps:
    ppid: int
    pgid: int
    stat: str
    lstart: str

    @property
    def alive(self) -> bool:
        return not self.stat.startswith("Z")


def _ps_table() -> dict[int, _Ps]:
    """Every process: ``{pid: (ppid, pgid, stat, start time)}``, zombies included (stat ``Z``).
    One ``ps`` call: every membership check of the teardown reads one such consistent snapshot."""
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pgid=,stat=,lstart="],
        capture_output=True, text=True, errors="replace", check=True, env={**os.environ, "LC_ALL": "C"},
    )
    table: dict[int, _Ps] = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        try:
            pid, ppid, pgid = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            continue
        table[pid] = _Ps(ppid, pgid, fields[3], fields[4].strip())
    return table


def _tree(table: dict[int, _Ps], root: int) -> set[int]:
    """``root`` and all of its descendants in ``table``, by walking ``ppid`` links; empty when
    ``root`` is not in it."""
    if root not in table:
        return set()
    children: dict[int, list[int]] = {}
    for pid, row in table.items():
        children.setdefault(row.ppid, []).append(pid)
    tree: set[int] = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid not in tree:
            tree.add(pid)
            stack.extend(children.get(pid, ()))
    return tree


def _snapshot(pids: set[int]) -> set[int]:
    """A copy of a set another thread may still be adding to."""
    while True:
        try:
            return set(pids)
        except RuntimeError:  # "Set changed size during iteration"
            continue


def _wait_until(condition: Callable[[], bool], timeout_s: float, poll_s: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
    return True


def _restore(sig: int, handler: Any) -> None:
    signal.signal(sig, handler if handler is not None else signal.SIG_DFL)


class _HeldSignals:
    """Defers SIGINT/SIGTERM/SIGHUP while a section that must not be cut short runs, then
    re-delivers the first one under whatever handler was active before. A no-op off the main
    thread, where Python cannot install handlers."""

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self, doing: str) -> None:
        self.doing = doing

    def __enter__(self) -> _HeldSignals:
        self.pending: int | None = None
        self._previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self._previous[sig] = signal.signal(sig, self._hold)
        return self

    def _hold(self, signum: int, frame: object) -> None:
        if self.pending is None:
            self.pending = signum
            _say(f"{signal.Signals(signum).name} received; still {self.doing}, please wait")

    def __exit__(self, *exc_info: object) -> None:
        for sig, handler in self._previous.items():
            _restore(sig, handler)
        if self.pending is not None:
            os.kill(os.getpid(), self.pending)


class _TerminateOnSignal:
    """SIGTERM and SIGHUP raise ``SystemExit(128 + n)`` instead of ending this process outright,
    so the teardown in ``finally`` runs: the server is in its own session and never sees them.
    Only a signal still at its default disposition is taken over (``nohup`` stays in force)."""

    SIGNALS = (signal.SIGTERM, signal.SIGHUP)

    def __enter__(self) -> _TerminateOnSignal:
        self._previous: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                if signal.getsignal(sig) == signal.SIG_DFL:
                    self._previous[sig] = signal.signal(sig, self._exit)
        return self

    @staticmethod
    def _exit(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    def __exit__(self, *exc_info: object) -> None:
        for sig, handler in self._previous.items():
            _restore(sig, handler)


class _Server:
    """One ``max serve`` in its own session, its sampler, and the teardown that guarantees both are gone.

    ``stop`` is idempotent and safe to call from ``finally`` and ``atexit`` alike.
    """

    def __init__(self, cmd: list[str], env: dict[str, str], log_path: Path, probe: profile_sampling.DeviceProbe) -> None:
        self.cmd = cmd
        self.env = env
        self.log_path = log_path
        self.probe = probe
        self.proc: subprocess.Popen[bytes] | None = None
        self.sampler: profile_sampling.Sampler | None = None
        #: Reasons the measurement is void, found while stopping.
        self.void: list[str] = []
        #: Anything the teardown could not guarantee, e.g. a process it could not stop.
        self.warnings: list[str] = []
        self._identity: dict[int, str] = {}
        self._stopped = False

    def start(self) -> None:
        with open(self.log_path, "wb") as log:
            self.proc = subprocess.Popen(
                self.cmd, env=self.env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT
            )
        self.sampler = profile_sampling.Sampler(self.proc.pid, self.probe)
        self.sampler.__enter__()

    def leader_exited(self) -> bool:
        """Whether the server process has exited -- without reaping it (see the module docstring):
        ``WNOWAIT`` leaves it a zombie, so its pid and process group id stay pinned. ``False``
        while that cannot be determined."""
        assert self.proc is not None
        if self.proc.returncode is not None:
            return True
        try:
            info = os.waitid(os.P_PID, self.proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:  # no longer our child to wait for: it exited and was reaped
            return True
        except OSError:
            return False
        # macOS also answers WEXITED for a *stopped* child (si_code CLD_STOPPED); only an exit counts.
        return info is not None and info.si_code in _EXITED

    @property
    def stopped(self) -> bool:
        return self._stopped or self.proc is None

    def stop(self) -> None:
        if self.stopped:
            return
        with _HeldSignals("stopping the server"):
            _say("stopping the server")
            self._teardown()

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        _say(f"WARNING: {message}")

    def _teardown(self) -> None:
        proc = self.proc
        assert proc is not None
        seen: set[int] = set()
        if self.sampler is not None:
            self.sampler.stop()
            if self.sampler.stop_timed_out:
                self.void.append("the sampler thread did not stop within its bound; its samples may be incomplete")
            seen = _snapshot(self.sampler.pids_seen)
        # Unreaped, the server's pid is still its process group id and cannot belong to anyone else.
        pinned = proc.returncode is None
        self._record_identities(seen, pinned)
        if pinned:
            self._killpg(signal.SIGTERM)
            _wait_until(self.leader_exited, STOP_GRACE_S)
            self._killpg(signal.SIGKILL)
            _wait_until(lambda: not self._group_alive(), STRAGGLER_GRACE_S)  # SIGKILL is not instantaneous
        self._stop_stragglers()
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._warn(f"the server (pid {proc.pid}) is still alive after SIGKILL")
        self._stopped = True

    def _record_identities(self, seen: set[int], pinned: bool) -> None:
        """Pin ``pid -> start time`` for every process that is verifiably the server's right now:
        a descendant, or (while the leader is unreaped) a member of its process group."""
        assert self.proc is not None
        try:
            table = _ps_table()
        except (OSError, subprocess.SubprocessError) as e:
            self._warn(f"cannot list processes ({e}); only the server's process group is signalled")
            return
        tree = _tree(table, self.proc.pid) if pinned else set()
        for pid, row in table.items():
            if pid in tree or (pinned and row.pgid == self.proc.pid):
                self._identity.setdefault(pid, row.lstart)
        for pid in sorted(seen - self._identity.keys()):
            row = table.get(pid)
            if row is not None and row.alive:
                self._warn(
                    f"pid {pid} was in the server's process tree during the run but is no longer verifiably "
                    "the server's (a reused pid, or a process that left both its parent and its group); not signalled"
                )

    def _group_alive(self) -> bool:
        """Whether a live process is still in the server's (pinned) process group."""
        assert self.proc is not None
        try:
            return any(row.alive and row.pgid == self.proc.pid for row in _ps_table().values())
        except (OSError, subprocess.SubprocessError):
            return False

    def _verified_alive(self) -> set[int]:
        """The server's processes still alive, re-verified from a fresh process listing."""
        assert self.proc is not None
        pinned = self.proc.returncode is None
        try:
            table = _ps_table()
        except (OSError, subprocess.SubprocessError):
            return set()
        tree = _tree(table, self.proc.pid) if pinned else set()
        me = os.getpid()
        return {
            pid
            for pid, row in table.items()
            if row.alive and pid != me
            and (pid in tree or (pinned and row.pgid == self.proc.pid) or self._identity.get(pid) == row.lstart)
        }

    def _killpg(self, sig: int) -> None:
        """Signal the server's process group. ``EPERM`` is expected, not an error: macOS returns it
        for a group whose only member left is the (unreaped, zombie) leader. Whatever of the group
        really survives is found, signalled one by one and reported by :meth:`_stop_stragglers`."""
        assert self.proc is not None
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _stop_stragglers(self) -> None:
        """SIGTERM, then SIGKILL, whatever of the server survived its group being killed -- in
        practice processes that left the group (``setsid``/``setpgid``)."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            targets = self._verified_alive()
            if not targets:
                return
            _say(f"{signal.Signals(sig).name} to pid(s) {sorted(targets)} of the server's tree, still alive "
                 "after its process group was killed")
            for pid in targets:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError as e:
                    self._warn(f"{signal.Signals(sig).name} to pid {pid}: {e}")
            _wait_until(lambda: not self._verified_alive() & targets, STRAGGLER_GRACE_S)
        left = self._verified_alive()
        if left:
            self._warn(f"could not stop pid(s) {sorted(left)}: still alive after SIGKILL")


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #
def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _served_models(port: int) -> list[str]:
    try:
        with _OPENER.open(f"http://127.0.0.1:{port}/v1/models", timeout=5.0) as response:
            payload = json.loads(response.read())
        return [str(entry.get("id")) for entry in payload.get("data", [])]
    except (OSError, ValueError, http.client.HTTPException, AttributeError, TypeError):
        return []


def chat_body(png: bytes, max_tokens: int) -> bytes:
    """The request body, exactly as the research harness sent it when the references were made:
    the text part first, then the image."""
    b64 = base64.b64encode(png).decode("ascii")
    return json.dumps({
        "model": cli.SERVED_MODEL_NAME,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }).encode("utf-8")


@dataclass
class Exchange:
    """One streamed request; every ``t_*`` is ``time.monotonic()``."""

    max_tokens: int
    t_start: float
    t_first: float | None = None
    t_end: float = 0.0
    text: str = ""
    completion_tokens: int | None = None
    finish_reason: str | None = None

    @property
    def ttft_s(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_start

    def summary(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens, "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason, "ttft_s": self.ttft_s, "wall_s": self.wall_s, "chars": len(self.text),
        }


class ExchangeFailed(Exception):
    """A request that did not complete: transport error, HTTP error, error event, or no ``[DONE]``."""


def _stream_chat(port: int, png: bytes, max_tokens: int) -> Exchange:
    """Send one page and read its SSE stream to ``[DONE]``.

    The text is every ``choices[0].delta.content`` concatenated (``None`` skipped); TTFT is the
    first non-empty delta; ``completion_tokens`` comes from the usage chunk. Every event but the
    usage chunk must carry a non-empty ``choices`` list; any other shape fails the exchange.
    """
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=chat_body(png, max_tokens), method="POST", headers={"Content-Type": "application/json"},
    )
    exchange = Exchange(max_tokens=max_tokens, t_start=time.monotonic())
    pieces: list[str] = []
    done = False
    try:
        with _OPENER.open(request, timeout=REQUEST_TIMEOUT_S) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ExchangeFailed(f"an event that is not a JSON object: {data[:200]}")
                if event.get("error"):
                    raise ExchangeFailed(f"error event: {event['error']}")
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    exchange.completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices", [])
                if not isinstance(choices, list):
                    raise ExchangeFailed(f"choices is not a list: {data[:200]}")
                if not choices:
                    if usage is None:  # only the closing usage chunk carries no choice
                        raise ExchangeFailed(f"an event with neither choices nor usage: {data[:200]}")
                    continue
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict | None):
                    raise ExchangeFailed(f"choices[0] is not a choice object with a delta object: {data[:200]}")
                content = (choice.get("delta") or {}).get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise ExchangeFailed(f"delta content is not text: {data[:200]}")
                    if content and exchange.t_first is None:
                        exchange.t_first = time.monotonic()
                    pieces.append(content)
                if choice.get("finish_reason"):
                    exchange.finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as e:
        try:
            detail = e.read()[:500].decode("utf-8", "replace")
        except Exception:
            detail = str(e.reason)
        raise ExchangeFailed(f"HTTP {e.code}: {detail}") from e
    except (OSError, ValueError, AttributeError, TypeError, http.client.HTTPException) as e:
        raise ExchangeFailed(f"{type(e).__name__}: {e}") from e
    exchange.t_end = time.monotonic()
    if not done:
        raise ExchangeFailed("the stream ended before data: [DONE]")
    exchange.text = "".join(pieces)
    return exchange


def _wait_ready(server: _Server, port: int, timeout_s: float) -> str | None:
    """``None`` once ``/v1/models`` lists the served model, else why it never did."""
    deadline = time.monotonic() + timeout_s
    next_poll = time.monotonic()
    while True:
        if server.leader_exited():
            return "the server exited before it was ready"
        now = time.monotonic()
        if now >= next_poll:
            if cli.SERVED_MODEL_NAME in _served_models(port):
                return None
            next_poll = now + READY_POLL_S
        if now >= deadline:
            return f"the server was not ready after {timeout_s:g} s (--ready-timeout-s)"
        time.sleep(EXIT_POLL_S)


def _log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        text = path.read_bytes().decode("utf-8", "replace")
    except OSError as e:
        return f"(cannot read {path}: {e})"
    return "\n".join(text.splitlines()[-lines:])


def _print_log_tail(path: Path) -> None:
    _say(f"last {LOG_TAIL_LINES} lines of {path}:")
    try:
        print(_log_tail(path), file=sys.stderr, flush=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# the GPU guard
# --------------------------------------------------------------------------- #
def _describe(processes: Iterable[dict[str, Any]]) -> list[str]:
    return [
        f"pid {p['pid']} {p['name'] or '?'} "
        + (f"{p['used_bytes'] / _MIB:.0f} MiB" if p.get("used_bytes") is not None else "? MiB")
        for p in processes
    ]


def _gpu_not_idle(own_pids: set[int], *, after_run: bool) -> list[str]:
    """Why the GPU is not otherwise idle, one line each; empty when it is."""
    when = "at the end of the run" if after_run else "now"
    try:
        foreign = profile_sampling.foreign_gpu_processes(own_pids)
    except profile_sampling.GpuProcessListUnavailable as e:
        return [f"the processes on the GPU cannot be listed {when}: {e}"]
    return [f"another process is on the GPU {when}: {line}" for line in _describe(foreign)]


# --------------------------------------------------------------------------- #
# host facts
# --------------------------------------------------------------------------- #
def _command_text(argv: list[str], timeout: float = 30.0) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _proc_field(path: str, key: str) -> str | None:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip() == key:
            return value.strip()
    return None


def _accelerator() -> dict[str, Any]:
    """``max.driver``'s accelerator API and architecture, asked in a short-lived subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _ACCELERATOR_PROBE], capture_output=True, text=True, errors="replace", timeout=300
        )
        if result.returncode == 0:
            facts = json.loads(result.stdout.strip().splitlines()[-1])
            return {"api": facts.get("api"), "architecture": facts.get("architecture")}
        error = result.stderr.strip()[-300:] or f"exit {result.returncode}"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        error = f"{type(e).__name__}: {e}"
    return {"api": None, "architecture": None, "error": error}


def _gpus(baseline: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        text = _command_text(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) or ""
        for line in text.splitlines():
            name, _, memory = line.rpartition(",")
            try:
                total = int(memory.split()[0]) * _MIB  # "81920 MiB"
            except (IndexError, ValueError):
                total = None
            if name.strip():
                gpus.append({"name": name.strip(), "memory_total_bytes": total, "source": "nvidia-smi"})
    if shutil.which("rocm-smi"):
        text = _command_text(["rocm-smi", "--showproductname"]) or ""
        for match in _ROCM_SERIES.finditer(text):
            total = baseline.get(f"amd{match.group(1)}", {}).get("total_bytes")
            gpus.append({"name": match.group(2), "memory_total_bytes": total, "source": "rocm-smi"})
    return gpus


def _host(baseline: dict[str, dict[str, int]]) -> dict[str, Any]:
    system = platform.system()
    cpu_brand: str | None = None
    ram_bytes: int | None = None
    if system == "Darwin":
        cpu_brand = _command_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        memsize = _command_text(["sysctl", "-n", "hw.memsize"])
        ram_bytes = int(memsize) if memsize and memsize.isdigit() else None
    elif system == "Linux":
        cpu_brand = _proc_field("/proc/cpuinfo", "model name")
        mem_total = _proc_field("/proc/meminfo", "MemTotal")  # "16384000 kB"
        if mem_total and mem_total.split()[0].isdigit():
            ram_bytes = int(mem_total.split()[0]) * 1024
    return {
        "platform": platform.platform(),
        "system": system,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_brand": cpu_brand or platform.processor() or None,
        "ram_bytes": ram_bytes,
        "accelerator": _accelerator(),
        "gpus": _gpus(baseline),
    }


def _hardware(host: dict[str, Any], devices: str) -> str:
    """The README's hardware cell: CPU brand and RAM, then what served (``Metal``, the GPU, ``CPU``)."""
    cpu = host.get("cpu_brand") or host.get("machine") or "unknown CPU"
    if host.get("ram_bytes"):
        cpu += f" {round(host['ram_bytes'] / _GIB)} GB"
    if devices == "cpu":
        return f"{cpu}, CPU"
    named = [gpu for gpu in host["gpus"] if gpu.get("name")]
    if named:
        gpu = named[0]
        memory = f" {round(gpu['memory_total_bytes'] / _GIB)} GB" if gpu.get("memory_total_bytes") else ""
        return f"{cpu}, {gpu['name']}{memory}"
    accelerator = host["accelerator"]
    if accelerator.get("api") == "metal":
        return f"{cpu}, Metal"
    return f"{cpu}, {accelerator.get('architecture') or accelerator.get('api') or 'GPU'}"


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist in ("unlimited-ocr-max", "max", "mojo"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


# --------------------------------------------------------------------------- #
# figures and the row
# --------------------------------------------------------------------------- #
class _Figures:
    """Every figure by name; one that cannot be computed is ``None``, with its reason in ``unavailable``."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.unavailable: dict[str, str] = {}

    def compute(self, name: str, fn: Callable[[], Any]) -> None:
        try:
            self.values[name] = fn()
        except Exception as e:  # a figure that cannot be computed is recorded, never a crash
            self.values[name] = None
            self.unavailable[name] = str(e) or type(e).__name__


def _memory(samples: list[tuple[float, int]], busy: list[tuple[float, float]], steady_after: float | None,
            *, empty: str, figures: _Figures, name: str) -> dict[str, Any]:
    """Peak and steady memory; ``steady_*`` is ``None`` (reason under ``<name>.steady``) when no
    sample fell after the warmup and outside every request."""
    if not samples:
        raise ValueError(empty)
    stats = profile_metrics.memory_stats(samples, busy, steady_after if steady_after is not None else float("inf"))
    if stats["n_steady"] == 0:
        figures.unavailable[f"{name}.steady"] = (
            "the warmup request did not complete" if steady_after is None
            else "no sample fell after the warmup and outside every request"
        )
    return {
        "peak_bytes": stats["peak"], "steady_min_bytes": stats["steady_min"], "steady_max_bytes": stats["steady_max"],
        "n": len(samples), "n_steady": stats["n_steady"],
    }


def _serving_gpus(baseline: dict[str, dict[str, int]],
                  device_stats: list[tuple[float, dict[str, dict[str, int]]]]) -> list[str]:
    """The GPUs the server ran on: every id whose ``used_bytes`` rose above its pre-start baseline
    in some sample; if none rose, the first id. An idle GPU next to it must not dilute its figures."""
    ids = list(dict.fromkeys([*baseline, *(gpu_id for _, per_gpu in device_stats for gpu_id in per_gpu)]))
    rose = [
        gpu_id for gpu_id in ids
        if gpu_id in baseline and any(
            gpu_id in per_gpu and per_gpu[gpu_id]["used_bytes"] > baseline[gpu_id]["used_bytes"]
            for _, per_gpu in device_stats
        )
    ]
    return rose or ids[:1]


def _gpu_utilisation(device_stats: list[tuple[float, dict[str, dict[str, int]]]],
                     windows: list[tuple[float, float]], gpus: list[str]) -> dict[str, Any]:
    """Median ``gpu_usage_percent`` of the serving ``gpus`` over the samples taken during a page request."""
    if not device_stats:
        raise ValueError("no device statistics sampled on this host")
    values = [
        per_gpu[gpu_id]["gpu_usage_percent"]
        for t, per_gpu in device_stats
        if any(start <= t <= end for start, end in windows)
        for gpu_id in gpus
        if gpu_id in per_gpu
    ]
    if not values:
        raise ValueError("no device utilisation sample fell inside a page request")
    return {"median_percent": statistics.median(values), "n": len(values), "gpus": gpus}


def _text(refs: dict[str, str], responses: dict[str, str]) -> dict[str, Any]:
    if responses.keys() != refs.keys():
        raise ValueError(f"only {len(responses)} of {len(refs)} pages completed")
    stats = profile_metrics.text_stats(refs, responses)
    return {key: value for key, value in stats.items() if key != "pages"}


def _gib(n: int) -> str:
    return f"{n / _GIB:.1f}"


def _memory_cell(memory: dict[str, Any] | None) -> str:
    if memory is None:
        return "—"
    low, high = memory["steady_min_bytes"], memory["steady_max_bytes"]
    if low is None:
        steady = "—"
    else:
        steady = _gib(low) if _gib(low) == _gib(high) else f"{_gib(low)}–{_gib(high)}"
    return f"{_gib(memory['peak_bytes'])} / {steady} GiB"


def _cer(cer: float) -> str:
    return "0" if cer == 0 else f"{cer:.2g}"


def _text_cell(text: dict[str, Any] | None) -> str:
    if text is None:
        return "—"
    return f"{text['identical']}/{text['n']} byte-identical, CER {_cer(text['cer'])}"


def row(hardware: str, weights: str, status: str, figures: dict[str, Any]) -> str:
    """One row in the README's column order:
    ``| hardware | weights | status | decode | prefill | memory, peak / steady | text vs reference |``.

    The memory cell is the server's device memory where it was measured (a discrete GPU), else
    its host RSS (Metal, where device memory is host memory, and CPU)."""
    decode, prefill, device_memory = figures.get("decode"), figures.get("prefill"), figures.get("device_memory")
    cells = [
        hardware,
        weights,
        status,
        f"**{decode['tok_s']:.1f} tok/s** ({decode['median_ms']:.1f} ms/step, n={decode['n']})" if decode else "—",
        f"{prefill['median_s']:.2f} s" if prefill else "—",
        f"device {_memory_cell(device_memory)}" if device_memory else _memory_cell(figures.get("host_memory")),
        _text_cell(figures.get("text")),
    ]
    return "| " + " | ".join(cells) + " |"


def _page_rows(pages: list[str], exchanges: dict[str, Exchange], refs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    done = [page for page in pages if page in exchanges]
    if not done:
        return []
    detail: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, variant_refs in refs.items():
        stats = profile_metrics.text_stats(
            {page: variant_refs[page] for page in done}, {page: exchanges[page].text for page in done}
        )
        detail[variant] = {entry["page"]: entry for entry in stats["pages"]}
    rows = []
    for page in done:
        exchange = exchanges[page]
        entry: dict[str, Any] = {
            "page": page,
            "identical": detail["bf16"][page]["identical"],
            "edits": detail["bf16"][page]["edits"],
            "completion_tokens": exchange.completion_tokens,
            "ttft_s": exchange.ttft_s,
            "wall_s": exchange.wall_s,
            "finish_reason": exchange.finish_reason,
            "chars": len(exchange.text),
        }
        if "int8" in detail:
            entry["identical_vs_int8"] = detail["int8"][page]["identical"]
            entry["edits_vs_int8"] = detail["int8"][page]["edits"]
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #
def _out_dir(out: Path | None) -> Path:
    if out is None:
        out = Path(f"unlimited-ocr-max-profile-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}")
    out = out.expanduser().absolute()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"--out {out} exists and is not an empty directory; profile never overwrites a run")
    return out


def run(args: argparse.Namespace, *, max_exe: str | None = None) -> int:
    """``unlimited-ocr-max profile``; returns the exit code (0 ok, 2 port in use or no ``ps``,
    3 GPU not idle, 4 server never ready, 5 void -- figures still written, 130 interrupted)."""
    started_utc = _utc_now()
    # The teardown proves which processes are the server's from `ps`, and the sampler reads it.
    if shutil.which("ps") is None:
        _say("refusing: `ps` is required (install procps)")
        return EXIT_NO_PS
    cli.check_devices_support_variant(args.weights, args.devices)
    model, weight_path, revision = cli.resolve_model(args.model, args.weights, args.revision)
    out = _out_dir(args.out)
    cmd = cli.serve_command(
        max_exe=max_exe if max_exe is not None else cli.max_executable(),
        model=model, weight_path=weight_path, devices=args.devices, port=args.port, revision=revision,
    )
    try:
        profile_corpus.verify()
    except ValueError as e:
        raise SystemExit(f"the bundled profiling corpus is damaged: {e}") from None
    if _port_in_use(args.port):
        _say(f"refusing: something already listens on 127.0.0.1:{args.port}; profile measures only a server it started "
             "(stop that one, or pick another --port)")
        return EXIT_PORT_IN_USE

    probe = profile_sampling.open_device_probe()
    server: _Server | None = None
    try:
        # Every GPU run is guarded, whatever the probe found: on Metal the process list is
        # legitimately empty, and a CUDA/ROCm host whose device diagnostics failed must be
        # refused (GpuProcessListUnavailable), never measured unguarded.
        guarded = args.devices == "gpu"
        if guarded and not probe.device_available:
            _say("no NVIDIA/AMD device statistics on this host (Metal, or none found): device figures do not apply")
        if guarded:
            reasons = _gpu_not_idle(set(), after_run=False)
            if reasons:
                _say("refusing: the measurement needs an otherwise idle GPU")
                for reason in reasons:
                    _say(f"  {reason}")
                return EXIT_GPU_BUSY
        baseline = probe.stats()
        server = _Server(cmd, cli.serve_env(args.ngram_size), out / "serve.log", probe)
        return _profile(args, server, out, baseline=baseline, guarded=guarded, started_utc=started_utc)
    except KeyboardInterrupt:
        if server is None or server.proc is None:
            _say("interrupted")
        elif server.stopped:
            _say("interrupted; the server is stopped")
        else:
            _say("interrupted; the server's teardown did not finish, and continues at exit")
        return EXIT_INTERRUPTED
    finally:
        # Only once the sampler thread has stopped: one that outlived its stop bound, or that no
        # teardown stopped yet, may be inside a probe call right now.
        if server is None or server.sampler is None or not server.sampler.alive:
            probe.close()


def _profile(args: argparse.Namespace, server: _Server, out: Path, *, baseline: dict[str, dict[str, int]],
             guarded: bool, started_utc: str) -> int:
    pages = profile_corpus.page_names()
    (out / "pages").mkdir(parents=True)
    exchanges: dict[str, Exchange] = {}
    warmup: Exchange | None = None
    busy: list[tuple[float, float]] = []
    void: list[str] = []
    ready_s: float | None = None
    swap_start, swap_start_error = _swap_now()

    atexit.register(server.stop)
    try:
        with _TerminateOnSignal():
            try:
                _say("starting: " + " ".join(server.cmd))
                with _HeldSignals("starting the server"):
                    server.start()
                t_spawn = time.monotonic()
                _say(f"waiting for the server (the first run compiles for minutes; up to {args.ready_timeout_s:g} s)")
                not_ready = _wait_ready(server, args.port, args.ready_timeout_s)
                if not_ready is not None:
                    _say(not_ready)
                    _print_log_tail(server.log_path)
                    return EXIT_NOT_READY
                ready_s = time.monotonic() - t_spawn
                _say(f"ready after {ready_s:.1f} s")

                for label, page, max_tokens in [("warmup", pages[0], WARMUP_MAX_TOKENS)] + [
                    (page, page, PAGE_MAX_TOKENS) for page in pages
                ]:
                    t_start = time.monotonic()
                    try:
                        exchange = _stream_chat(args.port, profile_corpus.page_png(page), max_tokens)
                    except ExchangeFailed as e:
                        busy.append((t_start, time.monotonic()))
                        void.append(f"request {label} failed: {e}")
                        _say(void[-1])
                        _print_log_tail(server.log_path)
                        break
                    busy.append((exchange.t_start, exchange.t_end))
                    if label == "warmup":
                        warmup = exchange
                    else:
                        exchanges[page] = exchange
                        (out / "pages" / f"{page}.md").write_bytes(exchange.text.encode("utf-8"))
                    _say(f"{label}: {exchange.completion_tokens} tokens in {exchange.wall_s:.1f} s")

                if not void:
                    time.sleep(STEADY_DWELL_S)
                if guarded:
                    own = _snapshot(server.sampler.pids_seen) | {server.proc.pid}
                    try:
                        own |= set(profile_sampling.process_tree(server.proc.pid))
                    except (OSError, subprocess.SubprocessError):
                        pass
                    void += _gpu_not_idle(own, after_run=True)
                # Last, so it covers the whole run. A server that could not bind the port (taken
                # after the check in `run`) exits -- while whatever took it answers the requests.
                if server.leader_exited():
                    void.append("the server exited during the run; another process may have answered on the port")
            finally:
                server.stop()
    finally:
        if server.stopped:  # otherwise the teardown failed part-way; leave atexit to retry it
            atexit.unregister(server.stop)
    void += server.void
    return _report(args, server, out, pages, exchanges, warmup, busy, void, baseline=baseline, guarded=guarded,
                   ready_s=ready_s, started_utc=started_utc, swap_start=swap_start, swap_start_error=swap_start_error)


def _swap_now() -> tuple[int | None, str | None]:
    """The system's swap in use now, or why it cannot be read."""
    try:
        return profile_sampling.swap_used_bytes(), None
    except Exception as e:
        return None, str(e) or type(e).__name__


def _swap(start: int | None, start_error: str | None, sampler: profile_sampling.Sampler) -> dict[str, Any]:
    """System swap before the server started and its maximum over the run; ``None`` fields, with
    the reason, where it cannot be read."""
    if start is None:
        return {"start_bytes": None, "max_bytes": None, "growth_bytes": None,
                "reason": f"system swap cannot be read: {start_error}"}
    peak = max([start, *(used for _, used in list(sampler.swap))])
    return {"start_bytes": start, "max_bytes": peak, "growth_bytes": peak - start,
            "reason": None if sampler.swap_error is None else f"swap sampling stopped: {sampler.swap_error}"}


def _scheduler_figures(figures: _Figures, log: str) -> None:
    """Decode and prefill from the server's scheduler log."""
    sched = profile_metrics.parse_scheduler_log(log)
    figures.compute("decode", lambda: profile_metrics.decode_stats(sched["TG"]))
    figures.compute("prefill", lambda: profile_metrics.prefill_stats(sched["CE"], skip_first=1))
    # Every CE line, the warmup's included: how the figures published before `profile` existed
    # were computed, so a new row can be compared with them like for like.
    figures.compute("prefill_all_ce", lambda: profile_metrics.prefill_stats(sched["CE"], skip_first=0))


def _device_figures(figures: _Figures, args: argparse.Namespace, sampler: profile_sampling.Sampler, *,
                    baseline: dict[str, dict[str, int]], busy: list[tuple[float, float]],
                    steady_after: float | None, page_windows: list[tuple[float, float]]) -> None:
    """Device memory and GPU utilisation, counted on the serving GPU(s) only; ``partial`` once
    device sampling stopped early."""
    if args.devices == "cpu":  # the server ran on no GPU: whatever a GPU did meanwhile is not its work
        for name in ("device_memory", "gpu_utilisation"):
            figures.values[name] = None
            figures.unavailable[name] = "--devices cpu"
        return
    device_process = [(t, v) for t, v in list(sampler.device_process) if v is not None]
    device_stats = list(sampler.device_stats)
    no_device = "no per-process device memory sampled on this host" + (
        f" (device sampling stopped: {sampler.device_error})" if sampler.device_error is not None else ""
    )
    # Per process: the server tree's own allocations, which are only on the GPU(s) it serves from.
    figures.compute("device_memory", lambda: _memory(
        device_process, busy, steady_after, empty=no_device, figures=figures, name="device_memory"))
    gpus = _serving_gpus(baseline, device_stats)
    figures.compute("gpu_utilisation", lambda: _gpu_utilisation(device_stats, page_windows, gpus))
    for name in ("device_memory", "gpu_utilisation"):
        figure = figures.values[name]
        if figure is not None:
            figure["partial"] = sampler.device_error is not None
            if sampler.device_error is not None:
                figure["device_error"] = sampler.device_error


def _report(args: argparse.Namespace, server: _Server, out: Path, pages: list[str], exchanges: dict[str, Exchange],
            warmup: Exchange | None, busy: list[tuple[float, float]], void: list[str], *,
            baseline: dict[str, dict[str, int]], guarded: bool, ready_s: float | None, started_utc: str,
            swap_start: int | None, swap_start_error: str | None) -> int:
    sampler = server.sampler
    assert sampler is not None
    rss = list(sampler.rss)
    steady_after = warmup.t_end if warmup is not None else None
    page_windows = [(e.t_start, e.t_end) for e in exchanges.values()]

    refs = {"bf16": {page: profile_corpus.reference(page, "bf16") for page in pages}}
    if args.weights == "int8":
        refs["int8"] = {page: profile_corpus.reference(page, "int8") for page in pages}
    responses = {page: exchange.text for page, exchange in exchanges.items()}

    figures = _Figures()
    _scheduler_figures(figures, server.log_path.read_bytes().decode("utf-8", "replace"))
    figures.compute("host_memory", lambda: _memory(
        rss, busy, steady_after, empty="no host RSS samples", figures=figures, name="host_memory"))
    _device_figures(figures, args, sampler, baseline=baseline, busy=busy, steady_after=steady_after,
                    page_windows=page_windows)
    for variant, name in (("bf16", "text"), ("int8", "text_vs_int8")):
        if variant in refs:
            figures.compute(name, lambda variant=variant: _text(refs[variant], responses))
    page_rows = _page_rows(pages, exchanges, refs)
    swap = _swap(swap_start, swap_start_error, sampler)
    swap_moved = swap["growth_bytes"] is not None and swap["growth_bytes"] > SWAP_MOVED_BYTES

    host = _host(baseline)
    hardware = _hardware(host, args.devices)
    status = f"{'void' if void else 'profiled'}, {len(exchanges)} pages" + (", swap moved" if swap_moved else "")
    table_row = row(hardware, args.weights, status, figures.values)
    profile_path = out / "profile.json"
    document = {
        "schema": SCHEMA,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "host": {**host, "hardware": hardware},
        "versions": _versions(),
        "flags": {
            "devices": args.devices, "weights": args.weights, "model": args.model, "revision": args.revision,
            "port": args.port, "ngram_size": args.ngram_size, "ready_timeout_s": args.ready_timeout_s, "out": str(out),
        },
        "served_command": server.cmd,
        "server_ready_s": ready_s,
        "device_baseline": baseline,
        "gpu_guard": guarded,
        "figures": figures.values,
        "unavailable": figures.unavailable,
        "sampling": {
            "interval_s": sampler.interval_s, "host_samples": len(rss), "device_samples": len(sampler.device_stats),
            "device_error": sampler.device_error, "swap_samples": len(sampler.swap),
        },
        "swap": swap,
        "warmup": warmup.summary() if warmup is not None else None,
        "pages": page_rows,
        "void": void,
        "teardown_warnings": server.warnings,
        "row": table_row,
    }
    profile_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(table_row)
    if figures.values.get("device_memory") is not None:  # the row's memory cell is the device's
        utilisation = figures.values.get("gpu_utilisation")
        device_line = (f"host RSS peak / steady {_memory_cell(figures.values.get('host_memory'))}, "
                       "median GPU utilisation " + (f"{utilisation['median_percent']:.0f} %" if utilisation else "—"))
        if sampler.device_error is not None:
            device_line += f" (partial: device sampling stopped: {sampler.device_error})"
        print(device_line)
    if "int8" in refs:
        print(f"vs pinned int8: {_text_cell(figures.values.get('text_vs_int8'))}")
    if swap_moved:
        print(f"note: timings are unreliable: system swap grew {swap['growth_bytes'] / _GIB:.2f} GiB during the run")
    for reason in void:
        print(f"void: {reason}")
    print(f"profile.json: {profile_path}", flush=True)
    return EXIT_VOID if void else EXIT_OK
