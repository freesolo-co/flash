"""Privacy-safe GRPO descendant process and thread census."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

_UNAVAILABLE = -1


@dataclass(frozen=True, order=True)
class _ProcessIdentity:
    pid: int
    start_time: int


def _read_start_time(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            raw = handle.read()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 1 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def _list_task_ids(pid: int) -> tuple[int, ...] | None:
    try:
        entries = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return None
    return tuple(int(entry) for entry in entries if entry.isdigit())


def _read_task_children(pid: int, task_id: int) -> set[int] | None:
    try:
        with open(f"/proc/{pid}/task/{task_id}/children", encoding="ascii") as handle:
            values = handle.read().split()
    except OSError:
        return None
    return {int(value) for value in values if value.isdigit()}


def _stable_children(identity: _ProcessIdentity) -> set[int] | None:
    if _read_start_time(identity.pid) != identity.start_time:
        return None
    task_ids = _list_task_ids(identity.pid)
    if task_ids is None:
        return None
    children: set[int] = set()
    for task_id in task_ids:
        task_children = _read_task_children(identity.pid, task_id)
        if task_children is None:
            return None
        children.update(task_children)
    if _read_start_time(identity.pid) != identity.start_time:
        return None
    return children


def _stable_thread_count(identity: _ProcessIdentity) -> int | None:
    if _read_start_time(identity.pid) != identity.start_time:
        return None
    task_ids = _list_task_ids(identity.pid)
    if task_ids is None:
        return None
    if _read_start_time(identity.pid) != identity.start_time:
        return None
    return len(task_ids)


def _complete_descendants(root: _ProcessIdentity) -> set[_ProcessIdentity] | None:
    known = {root.pid: root.start_time}
    descendants: set[_ProcessIdentity] = set()
    pending = [root]
    while pending:
        identity = pending.pop()
        children = _stable_children(identity)
        if children is None:
            return None
        for pid in children:
            start_time = _read_start_time(pid)
            if start_time is None:
                return None
            previous = known.get(pid)
            if previous is not None:
                if previous != start_time:
                    return None
                continue
            child = _ProcessIdentity(pid, start_time)
            known[pid] = start_time
            descendants.add(child)
            pending.append(child)
    return descendants


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
        self._root_identity: _ProcessIdentity | None = None
        self._baseline: ProcessCensusSnapshot | None = None
        self._per_step: ProcessCensusSnapshot | None = None
        self._terminal: ProcessCensusSnapshot | None = None
        self._peak_processes = 0
        self._peak_threads = 0
        self._peak_single_process_threads = 0
        self._minimum_headroom = _UNAVAILABLE
        self._complete_samples = 0
        self._all_samples_complete = True

    def _snapshot(self) -> ProcessCensusSnapshot:
        current, maximum, headroom = _cgroup_pids()
        try:
            affinity = len(os.sched_getaffinity(self._root_pid))
        except (AttributeError, OSError, ProcessLookupError):
            affinity = _UNAVAILABLE
        root = self._root_identity
        descendants = _complete_descendants(root) if root is not None else None
        thread_counts: list[int] | None = [] if descendants is not None else None
        if descendants is not None:
            for identity in descendants:
                count = _stable_thread_count(identity)
                if count is None:
                    thread_counts = None
                    break
                thread_counts.append(count)
        if root is None or _read_start_time(root.pid) != root.start_time:
            thread_counts = None
        complete = descendants is not None and thread_counts is not None
        return ProcessCensusSnapshot(
            available=int(complete),
            descendant_processes=len(descendants) if complete else _UNAVAILABLE,
            descendant_threads=sum(thread_counts) if complete else _UNAVAILABLE,
            max_descendant_threads=max(thread_counts, default=0) if complete else _UNAVAILABLE,
            pids_current=current,
            pids_max=maximum,
            pids_headroom=headroom,
            cpu_quota_millicores=_cpu_quota_millicores(),
            affinity_cpus=affinity,
        )

    def _observe(self, snapshot: ProcessCensusSnapshot) -> None:
        with self._lock:
            if snapshot.available != 1:
                self._all_samples_complete = False
                return
            self._complete_samples += 1
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
        start_time = _read_start_time(self._root_pid)
        self._root_identity = (
            _ProcessIdentity(self._root_pid, start_time) if start_time is not None else None
        )
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
            have_peaks = self._complete_samples > 0
            return {
                "available": int(have_peaks and self._all_samples_complete),
                "baseline_processes": baseline.descendant_processes if baseline else _UNAVAILABLE,
                "baseline_threads": baseline.descendant_threads if baseline else _UNAVAILABLE,
                "per_step_processes": per_step.descendant_processes if per_step else _UNAVAILABLE,
                "per_step_threads": per_step.descendant_threads if per_step else _UNAVAILABLE,
                "peak_processes": self._peak_processes if have_peaks else _UNAVAILABLE,
                "peak_threads": self._peak_threads if have_peaks else _UNAVAILABLE,
                "peak_single_process_threads": (
                    self._peak_single_process_threads if have_peaks else _UNAVAILABLE
                ),
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
