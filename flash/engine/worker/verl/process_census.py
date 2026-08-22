"""Privacy-safe GRPO descendant process and thread census."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass

_UNAVAILABLE = -1
_SNAPSHOT_ATTEMPTS = 3

# a walked process is in exactly one of these states. only reuse is an integrity failure:
# a rollout tree is always creating and reaping workers, so exit is the normal case.
_ALIVE = "alive"
_GONE = "gone"
_REUSED = "reused"


class _VanishedProcess:
    """Marker for a process that exited during the walk, which is benign for a descendant."""

    __slots__ = ()


_VANISHED = _VanishedProcess()


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


def _task_present(pid: int, task_id: int) -> bool:
    """Whether one thread of a process still exists."""
    return os.path.exists(f"/proc/{pid}/task/{task_id}")


def _liveness(identity: _ProcessIdentity) -> str:
    """Classify a pid as still alive, exited, or replaced by an unrelated process."""
    start_time = _read_start_time(identity.pid)
    if start_time is None:
        return _GONE
    return _ALIVE if start_time == identity.start_time else _REUSED


def _stable_children(identity: _ProcessIdentity) -> set[int] | _VanishedProcess | None:
    """Children of a live process, ``_VANISHED`` if it exited, ``None`` on an unusable read."""
    state = _liveness(identity)
    if state != _ALIVE:
        return _VANISHED if state == _GONE else None
    task_ids = _list_task_ids(identity.pid)
    if task_ids is None:
        return _VANISHED if _liveness(identity) == _GONE else None
    children: set[int] = set()
    for task_id in task_ids:
        task_children = _read_task_children(identity.pid, task_id)
        if task_children is None:
            # a rollout tree churns threads, so one task listed by `_list_task_ids` can exit
            # before its `children` file is read while the process itself stays alive. the
            # departed thread's children do NOT leave with it -- linux reparents them within the
            # same thread group, so they resurface under a sibling task's `children` and this walk
            # still counts them. skipping the unreadable task can undercount only if every
            # remaining sibling is also read before the reparent lands, which costs one low
            # census sample of a metric, never a wrong decision: `host_census/*` is published as
            # step telemetry (rl_train_runner) and gates nothing.
            # only a thread that is STILL PRESENT and unreadable makes the walk unusable.
            if not _task_present(identity.pid, task_id):
                continue
            return _VANISHED if _liveness(identity) == _GONE else None
        children.update(task_children)
    state = _liveness(identity)
    if state != _ALIVE:
        return _VANISHED if state == _GONE else None
    return children


def _stable_thread_count(identity: _ProcessIdentity) -> int | _VanishedProcess | None:
    """Thread count of a live process, ``_VANISHED`` if it exited, ``None`` on an unusable read."""
    state = _liveness(identity)
    if state != _ALIVE:
        return _VANISHED if state == _GONE else None
    task_ids = _list_task_ids(identity.pid)
    if task_ids is None:
        return _VANISHED if _liveness(identity) == _GONE else None
    state = _liveness(identity)
    if state != _ALIVE:
        return _VANISHED if state == _GONE else None
    return len(task_ids)


def _scan_process_tree(root: _ProcessIdentity) -> frozenset[_ProcessIdentity] | None:
    """Walk the live descendant tree.

    A descendant exiting mid-walk is expected on a rollout tree and is skipped along with the
    subtree that left with it. Only an unusable read of a live process, pid reuse, or the root
    itself disappearing invalidates the walk.
    """
    known = {root.pid: root.start_time}
    identities = {root}
    pending = [root]
    while pending:
        parent = pending.pop()
        children = _stable_children(parent)
        if children is None:
            return None
        if isinstance(children, _VanishedProcess):
            if parent == root:
                return None
            continue
        for pid in children:
            start_time = _read_start_time(pid)
            if start_time is None:
                continue
            previous = known.get(pid)
            if previous is not None:
                if previous != start_time:
                    return None
                continue
            known[pid] = start_time
            child = _ProcessIdentity(pid, start_time)
            identities.add(child)
            pending.append(child)
    return frozenset(identities)


def _reuse_contradiction(
    first: frozenset[_ProcessIdentity], second: frozenset[_ProcessIdentity]
) -> bool:
    """True when a pid is present in both scans under two different start times."""
    starts = {identity.pid: identity.start_time for identity in first}
    return any(
        starts.get(identity.pid, identity.start_time) != identity.start_time for identity in second
    )


def _descendant_counts(root: _ProcessIdentity | None) -> tuple[int, int, int] | None:
    """Count live descendants and their threads.

    The paired scan no longer demands an identical tree, which never holds while workers are
    being created and reaped. It asserts the weaker invariant that actually signals corruption:
    no pid may be seen under two different start times. Processes are counted only when their
    thread count was also read, so the reported pair stays self-consistent.
    """
    if root is None:
        return None
    first = _scan_process_tree(root)
    if first is None:
        return None
    second = _scan_process_tree(root)
    if second is None or _reuse_contradiction(first, second):
        return None
    thread_counts: list[int] = []
    for identity in second - {root}:
        count = _stable_thread_count(identity)
        if count is None:
            return None
        if isinstance(count, _VanishedProcess):
            continue
        thread_counts.append(count)
    if _liveness(root) != _ALIVE:
        return None
    return len(thread_counts), sum(thread_counts), max(thread_counts, default=0)


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
    incomplete_attempts: int = 0


def _step_evidence(step: int, snapshot: ProcessCensusSnapshot) -> dict[str, int]:
    return {
        "optimizer_step": step,
        "processes": snapshot.descendant_processes,
        "threads": snapshot.descendant_threads,
        "max_process_threads": snapshot.max_descendant_threads,
        "pids_current": snapshot.pids_current,
        "pids_max": snapshot.pids_max,
        "pids_headroom": snapshot.pids_headroom,
    }


def _required_snapshot_complete(snapshot: ProcessCensusSnapshot | None) -> bool:
    return bool(
        snapshot is not None
        and snapshot.available == 1
        and snapshot.pids_current >= 0
        and snapshot.pids_max >= 0
        and snapshot.pids_headroom > 0
    )


class GrpoProcessCensus:
    """Sample numeric process pressure without inspecting process metadata."""

    def __init__(
        self,
        root_pid: int,
        *,
        expected_steps: Iterable[int],
        interval_s: float = 0.1,
    ) -> None:
        parsed_steps = tuple(int(step) for step in expected_steps)
        if any(step <= 0 for step in parsed_steps) or len(set(parsed_steps)) != len(parsed_steps):
            raise ValueError("expected GRPO census steps must be unique positive integers")
        self._root_pid = int(root_pid)
        self._expected_steps = tuple(sorted(parsed_steps))
        self._interval_s = max(0.02, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_identity: _ProcessIdentity | None = None
        self._baseline: ProcessCensusSnapshot | None = None
        self._step_snapshots: dict[int, ProcessCensusSnapshot] = {}
        self._step_attempts: dict[int, ProcessCensusSnapshot] = {}
        self._terminal: ProcessCensusSnapshot | None = None
        self._peak_processes = 0
        self._peak_threads = 0
        self._peak_single_process_threads = 0
        self._peak_pids_current = _UNAVAILABLE
        self._minimum_headroom = _UNAVAILABLE
        self._complete_samples = 0
        self._incomplete_sample_count = 0
        self._duplicate_step_count = 0
        self._conflicting_step_count = 0
        self._unexpected_step_count = 0

    def _snapshot(self) -> ProcessCensusSnapshot:
        current, maximum, headroom = _cgroup_pids()
        try:
            affinity = len(os.sched_getaffinity(self._root_pid))
        except (AttributeError, OSError, ProcessLookupError):
            affinity = _UNAVAILABLE
        counts = None
        incomplete_attempts = 0
        for _attempt in range(_SNAPSHOT_ATTEMPTS):
            counts = _descendant_counts(self._root_identity)
            if counts is not None:
                break
            incomplete_attempts += 1
        complete = counts is not None
        processes, threads, maximum_threads = counts or (
            _UNAVAILABLE,
            _UNAVAILABLE,
            _UNAVAILABLE,
        )
        return ProcessCensusSnapshot(
            available=int(complete),
            descendant_processes=processes,
            descendant_threads=threads,
            max_descendant_threads=maximum_threads,
            pids_current=current,
            pids_max=maximum,
            pids_headroom=headroom,
            cpu_quota_millicores=_cpu_quota_millicores(),
            affinity_cpus=affinity,
            incomplete_attempts=incomplete_attempts,
        )

    def _observe(self, snapshot: ProcessCensusSnapshot) -> None:
        with self._lock:
            self._incomplete_sample_count += snapshot.incomplete_attempts
            if snapshot.pids_current >= 0:
                self._peak_pids_current = max(self._peak_pids_current, snapshot.pids_current)
            if snapshot.pids_headroom >= 0:
                self._minimum_headroom = (
                    snapshot.pids_headroom
                    if self._minimum_headroom < 0
                    else min(self._minimum_headroom, snapshot.pids_headroom)
                )
            if snapshot.available != 1:
                return
            self._complete_samples += 1
            self._peak_processes = max(self._peak_processes, snapshot.descendant_processes)
            self._peak_threads = max(self._peak_threads, snapshot.descendant_threads)
            self._peak_single_process_threads = max(
                self._peak_single_process_threads,
                snapshot.max_descendant_threads,
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

    def sample_step(self, optimizer_step: int) -> None:
        step = int(optimizer_step)
        snapshot = self._snapshot()
        self._observe(snapshot)
        with self._lock:
            if step not in self._expected_steps:
                self._unexpected_step_count += 1
                return
            existing = self._step_attempts.get(step)
            if existing is not None:
                if existing == snapshot:
                    self._duplicate_step_count += 1
                else:
                    self._conflicting_step_count += 1
                return
            self._step_attempts[step] = snapshot
            if snapshot.available == 1:
                self._step_snapshots[step] = snapshot

    def stop(self) -> dict[str, int | list[dict[str, int]]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_s * 4))
        terminal = self._snapshot()
        self._observe(terminal)
        with self._lock:
            self._terminal = terminal
        return self.summary()

    def summary(self) -> dict[str, int | list[dict[str, int]]]:
        with self._lock:
            baseline = self._baseline
            terminal = self._terminal
            steps = [
                _step_evidence(step, self._step_snapshots[step])
                for step in sorted(self._step_snapshots)
            ]
            latest = (
                self._step_snapshots[max(self._step_snapshots)] if self._step_snapshots else None
            )
            exact_steps = tuple(sorted(self._step_snapshots)) == self._expected_steps
            required = [baseline, terminal, *self._step_snapshots.values()]
            required_complete = all(_required_snapshot_complete(snapshot) for snapshot in required)
            step_errors = (
                self._duplicate_step_count
                + self._conflicting_step_count
                + self._unexpected_step_count
            )
            missing_step_count = len(set(self._expected_steps) - set(self._step_snapshots))
            have_peaks = self._complete_samples > 0
            return {
                "available": int(
                    have_peaks
                    and exact_steps
                    and required_complete
                    and step_errors == 0
                    and missing_step_count == 0
                ),
                "baseline_processes": baseline.descendant_processes if baseline else _UNAVAILABLE,
                "baseline_threads": baseline.descendant_threads if baseline else _UNAVAILABLE,
                "per_step_processes": latest.descendant_processes if latest else _UNAVAILABLE,
                "per_step_threads": latest.descendant_threads if latest else _UNAVAILABLE,
                "steps": steps,
                "peak_processes": self._peak_processes if have_peaks else _UNAVAILABLE,
                "peak_threads": self._peak_threads if have_peaks else _UNAVAILABLE,
                "peak_single_process_threads": (
                    self._peak_single_process_threads if have_peaks else _UNAVAILABLE
                ),
                "peak_pids_current": self._peak_pids_current,
                "terminal_processes": terminal.descendant_processes if terminal else _UNAVAILABLE,
                "terminal_threads": terminal.descendant_threads if terminal else _UNAVAILABLE,
                "pids_current": terminal.pids_current
                if terminal
                else (baseline.pids_current if baseline else _UNAVAILABLE),
                "pids_max": terminal.pids_max
                if terminal
                else (baseline.pids_max if baseline else _UNAVAILABLE),
                "minimum_pids_headroom": self._minimum_headroom,
                "incomplete_sample_count": self._incomplete_sample_count,
                "missing_step_count": missing_step_count,
                "duplicate_step_count": self._duplicate_step_count,
                "conflicting_step_count": self._conflicting_step_count,
                "unexpected_step_count": self._unexpected_step_count,
                "cpu_quota_millicores": baseline.cpu_quota_millicores if baseline else _UNAVAILABLE,
                "affinity_cpus": baseline.affinity_cpus if baseline else _UNAVAILABLE,
            }
