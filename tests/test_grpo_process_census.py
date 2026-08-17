from __future__ import annotations

import subprocess
import sys
import time

import pytest

import flash.engine.worker.verl.process_census as census_module
from flash.engine.worker.verl.process_census import (
    GrpoProcessCensus,
    ProcessCensusSnapshot,
    _ProcessIdentity,
)


def _census(root_start: int | None = 100) -> GrpoProcessCensus:
    census = GrpoProcessCensus(1, interval_s=60)
    census._root_identity = _ProcessIdentity(1, root_start) if root_start is not None else None
    return census


def _stable_auxiliary_metrics(monkeypatch):
    monkeypatch.setattr(census_module, "_cgroup_pids", lambda: (10, 100, 90))
    monkeypatch.setattr(census_module, "_cpu_quota_millicores", lambda: 2000)
    monkeypatch.setattr(census_module.os, "sched_getaffinity", lambda _pid: {0, 1})


def _complete_proc_view(monkeypatch, *, with_child: bool):
    starts = {1: 100, 2: 200}
    tasks = {1: (1,), 2: (2, 3)}
    children = {(1, 1): {2} if with_child else set(), (2, 2): set(), (2, 3): set()}
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: starts.get(pid))
    monkeypatch.setattr(census_module, "_list_task_ids", lambda pid: tasks.get(pid))
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: children.get((pid, task_id)),
    )
    _stable_auxiliary_metrics(monkeypatch)


def test_snapshot_root_read_failure_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=False)
    monkeypatch.setattr(census_module, "_read_start_time", lambda _pid: None)
    snapshot = _census()._snapshot()
    assert snapshot.available == 0
    assert (
        snapshot.descendant_processes,
        snapshot.descendant_threads,
        snapshot.max_descendant_threads,
    ) == (-1, -1, -1)


def test_snapshot_partial_child_task_read_failure_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    original = census_module._read_task_children
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: None if pid == 2 else original(pid, task_id),
    )
    assert _census()._snapshot().available == 0


def test_snapshot_thread_count_failure_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    calls = {2: 0}
    original = census_module._list_task_ids

    def list_tasks(pid):
        if pid == 2:
            calls[2] += 1
            if calls[2] == 2:
                return None
        return original(pid)

    monkeypatch.setattr(census_module, "_list_task_ids", list_tasks)
    assert _census()._snapshot().available == 0


def test_snapshot_child_exit_race_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    calls = {2: 0}

    def start_time(pid):
        if pid == 2:
            calls[2] += 1
            return 200 if calls[2] == 1 else None
        return 100

    monkeypatch.setattr(census_module, "_read_start_time", start_time)
    assert _census()._snapshot().available == 0


def test_snapshot_descendant_pid_reuse_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    calls = {2: 0}

    def start_time(pid):
        if pid == 2:
            calls[2] += 1
            return 200 if calls[2] == 1 else 201
        return 100

    monkeypatch.setattr(census_module, "_read_start_time", start_time)
    assert _census()._snapshot().available == 0


def test_snapshot_root_pid_reuse_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=False)
    calls = {1: 0}

    def start_time(_pid):
        calls[1] += 1
        return 100 if calls[1] <= 2 else 101

    monkeypatch.setattr(census_module, "_read_start_time", start_time)
    assert _census()._snapshot().available == 0


def test_snapshot_complete_empty_tree_reports_real_zero(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=False)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert (
        snapshot.descendant_processes,
        snapshot.descendant_threads,
        snapshot.max_descendant_threads,
    ) == (0, 0, 0)


def test_snapshot_complete_nonempty_tree_reports_process_and_threads(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert (
        snapshot.descendant_processes,
        snapshot.descendant_threads,
        snapshot.max_descendant_threads,
    ) == (1, 2, 2)


def _snapshot(available: int, processes: int, threads: int, maximum: int) -> ProcessCensusSnapshot:
    return ProcessCensusSnapshot(
        available=available,
        descendant_processes=processes,
        descendant_threads=threads,
        max_descendant_threads=maximum,
        pids_current=10,
        pids_max=100,
        pids_headroom=90,
        cpu_quota_millicores=2000,
        affinity_cpus=2,
    )


def test_incomplete_sample_latches_unavailable_without_changing_complete_peaks():
    census = _census()
    baseline = _snapshot(1, 2, 7, 4)
    incomplete = _snapshot(0, -1, -1, -1)
    later = _snapshot(1, 1, 3, 2)
    census._baseline = baseline
    census._per_step = incomplete
    census._terminal = later
    census._observe(baseline)
    census._observe(incomplete)
    census._observe(later)

    summary = census.summary()
    assert summary["available"] == 0
    assert summary["per_step_processes"] == -1
    assert summary["per_step_threads"] == -1
    assert summary["peak_processes"] == 2
    assert summary["peak_threads"] == 7
    assert summary["peak_single_process_threads"] == 4


def test_process_census_reports_numeric_bounded_summary():
    census = GrpoProcessCensus(__import__("os").getpid(), interval_s=0.02).start()
    script = """
import subprocess
import sys
import threading
import time

threads = [threading.Thread(target=time.sleep, args=(0.35,)) for _ in range(3)]
for thread in threads:
    thread.start()
children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)']) for _ in range(2)]
for child in children:
    child.wait()
for thread in threads:
    thread.join()
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    deadline = time.monotonic() + 2.0
    while proc.poll() is None and time.monotonic() < deadline:
        census.sample_step()
        time.sleep(0.02)
    assert proc.wait(timeout=1) == 0
    summary = census.stop()

    assert all(isinstance(value, int) for value in summary.values())
    assert summary["peak_processes"] >= 3
    assert summary["peak_threads"] >= 5
    assert summary["peak_single_process_threads"] >= 4
    assert len(summary) <= 16


def test_process_census_source_avoids_sensitive_proc_metadata():
    import inspect

    source = inspect.getsource(census_module)
    for forbidden in ("cmdline", "environ", "/fd", "/maps", "/exe", "username"):
        assert forbidden not in source


@pytest.mark.parametrize(
    "field", ["descendant_processes", "descendant_threads", "max_descendant_threads"]
)
def test_unavailable_snapshot_never_reports_healthy_zero(monkeypatch, field):
    _complete_proc_view(monkeypatch, with_child=False)
    monkeypatch.setattr(census_module, "_read_start_time", lambda _pid: None)
    assert getattr(_census()._snapshot(), field) == -1
