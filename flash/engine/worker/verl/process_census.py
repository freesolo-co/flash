"""Privacy-safe GRPO descendant process and thread census."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

_UNAVAILABLE = -1


def _numeric_children(pid: int) -> set[int]:
    children: set[int] = set()
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return children
    for tid in tids:
        if not tid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/task/{tid}/children", encoding="ascii") as handle:
                values = handle.read().split()
        except OSError:
            continue
        children.update(int(value) for value in values if value.isdigit())
    return children


def _descendants(root_pid: int) -> set[int]:
    found: set[int] = set()
    pending = list(_numeric_children(root_pid))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(_numeric_children(pid) - found)
    return found


def _thread_count(pid: int) -> int | None:
    try:
        return sum(name.isdigit() for name in os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return None


def _read_scalar(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _cgroup_pids() -> tuple[int, int, int]:
    current_raw = _read_scalar("/sys/fs/cgroup/pids.current")
    maximum_raw = _read_scalar("/sys/fs/cgroup/pids.max")
    if current_raw is None or maximum_raw is None:
        return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE
    try:
        current = int(current_raw)
        maximum = _UNAVAILABLE if maximum_raw == "max" else int(maximum_raw)
    except ValueError:
        return _UNAVAILABLE, _UNAVAILABLE, _UNAVAILABLE
    headroom = maximum - current if maximum >= 0 else _UNAVAILABLE
    return current, maximum, headroom


def _cpu_quota_millicores() -> int:
    raw = _read_scalar("/sys/fs/cgroup/cpu.max")
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota * 1000 // period
            except ValueError:
                pass
    quota_raw = _read_scalar("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_raw = _read_scalar("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota, period = int(quota_raw or ""), int(period_raw or "")
    except ValueError:
        return _UNAVAILABLE
    return quota * 1000 // period if quota > 0 and period > 0 else _UNAVAILABLE


@dataclass(frozen=True)
class ProcessCensusSnapshot:
    available: int
    descendant_processes: int
    descendant_threads: int
    max_descendant_threads: int
    pids_current: int
    pids_max: int
    pids_headroom: int
    cpu_quota_millicores: int
    affinity_cpus: int


class GrpoProcessCensus:
    """Sample numeric process pressure without inspecting process metadata."""

    def __init__(self, root_pid: int, *, interval_s: float = 0.1) -> None:
        self._root_pid = int(root_pid)
        self._interval_s = max(0.02, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: ProcessCensusSnapshot | None = None
        self._per_step: ProcessCensusSnapshot | None = None
        self._terminal: ProcessCensusSnapshot | None = None
        self._peak_processes = 0
        self._peak_threads = 0
        self._peak_single_process_threads = 0
        self._minimum_headroom = _UNAVAILABLE

    def _snapshot(self) -> ProcessCensusSnapshot:
        pids = _descendants(self._root_pid)
        thread_counts = [count for pid in pids if (count := _thread_count(pid)) is not None]
        current, maximum, headroom = _cgroup_pids()
        try:
            affinity = len(os.sched_getaffinity(self._root_pid))
        except (AttributeError, OSError, ProcessLookupError):
            affinity = _UNAVAILABLE
        return ProcessCensusSnapshot(
            available=int(os.path.isdir(f"/proc/{self._root_pid}/task")),
            descendant_processes=len(pids),
            descendant_threads=sum(thread_counts),
            max_descendant_threads=max(thread_counts, default=0),
            pids_current=current,
            pids_max=maximum,
            pids_headroom=headroom,
            cpu_quota_millicores=_cpu_quota_millicores(),
            affinity_cpus=affinity,
        )

    def _observe(self, snapshot: ProcessCensusSnapshot) -> None:
        with self._lock:
            self._peak_processes = max(self._peak_processes, snapshot.descendant_processes)
            self._peak_threads = max(self._peak_threads, snapshot.descendant_threads)
            self._peak_single_process_threads = max(
                self._peak_single_process_threads,
                snapshot.max_descendant_threads,
            )
            if snapshot.pids_headroom >= 0:
                self._minimum_headroom = (
                    snapshot.pids_headroom
                    if self._minimum_headroom < 0
                    else min(self._minimum_headroom, snapshot.pids_headroom)
                )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._observe(self._snapshot())

    def start(self) -> GrpoProcessCensus:
        self._baseline = self._snapshot()
        self._observe(self._baseline)
        self._thread = threading.Thread(
            target=self._run,
            name="flash-grpo-process-census",
            daemon=True,
        )
        self._thread.start()
        return self

    def sample_step(self) -> None:
        snapshot = self._snapshot()
        self._observe(snapshot)
        with self._lock:
            self._per_step = snapshot

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_s * 4))
        terminal = self._snapshot()
        self._observe(terminal)
        with self._lock:
            self._terminal = terminal
        return self.summary()

    def summary(self) -> dict[str, int]:
        with self._lock:
            baseline = self._baseline
            per_step = self._per_step
            terminal = self._terminal
            return {
                "available": baseline.available if baseline is not None else 0,
                "baseline_processes": baseline.descendant_processes if baseline else _UNAVAILABLE,
                "baseline_threads": baseline.descendant_threads if baseline else _UNAVAILABLE,
                "per_step_processes": per_step.descendant_processes if per_step else _UNAVAILABLE,
                "per_step_threads": per_step.descendant_threads if per_step else _UNAVAILABLE,
                "peak_processes": self._peak_processes,
                "peak_threads": self._peak_threads,
                "peak_single_process_threads": self._peak_single_process_threads,
                "terminal_processes": terminal.descendant_processes if terminal else _UNAVAILABLE,
                "terminal_threads": terminal.descendant_threads if terminal else _UNAVAILABLE,
                "pids_current": terminal.pids_current
                if terminal
                else (baseline.pids_current if baseline else _UNAVAILABLE),
                "pids_max": terminal.pids_max
                if terminal
                else (baseline.pids_max if baseline else _UNAVAILABLE),
                "minimum_pids_headroom": self._minimum_headroom,
                "cpu_quota_millicores": baseline.cpu_quota_millicores if baseline else _UNAVAILABLE,
                "affinity_cpus": baseline.affinity_cpus if baseline else _UNAVAILABLE,
            }
