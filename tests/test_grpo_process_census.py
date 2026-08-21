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


def _census(
    root_start: int | None = 100,
    *,
    expected_steps: tuple[int, ...] = (1,),
) -> GrpoProcessCensus:
    census = GrpoProcessCensus(1, expected_steps=expected_steps, interval_s=60)
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
    assert snapshot.incomplete_attempts == 3
    assert (
        snapshot.descendant_processes,
        snapshot.descendant_threads,
        snapshot.max_descendant_threads,
    ) == (-1, -1, -1)


def test_snapshot_partial_child_task_read_failure_on_a_live_process_is_unavailable(monkeypatch):
    """A thread of a process that is still alive must be readable, or the sample is unusable."""
    _complete_proc_view(monkeypatch, with_child=True)
    original = census_module._read_task_children
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: None if pid == 2 else original(pid, task_id),
    )
    assert _census()._snapshot().available == 0


def test_snapshot_thread_that_exits_mid_walk_is_skipped_not_invalidating(monkeypatch):
    """A rollout tree churns threads; one exiting mid-walk must not discard the whole sample.

    `_list_task_ids` snapshots the task list, so a thread can exit before its `children` file is
    read while the PROCESS stays alive. that thread's descendants left with it, so skipping it is
    exact. treating it as unusable would make one benign race drop the step's census evidence.
    """
    _complete_proc_view(monkeypatch, with_child=True)
    original = census_module._read_task_children
    # thread 3 of pid 2 exits mid-walk: its `children` read fails AND it is no longer present.
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: None if (pid, task_id) == (2, 3) else original(pid, task_id),
    )
    monkeypatch.setattr(
        census_module,
        "_task_present",
        lambda pid, task_id: (pid, task_id) != (2, 3),
    )
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.descendant_processes == 1


def test_snapshot_present_but_unreadable_thread_is_still_unavailable(monkeypatch):
    """The skip above must require the thread to be GONE, not merely unreadable.

    a thread that is still present but whose `children` file cannot be read is a real unusable
    read: its descendants are unknown rather than known-departed.
    """
    _complete_proc_view(monkeypatch, with_child=True)
    original = census_module._read_task_children
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: None if (pid, task_id) == (2, 3) else original(pid, task_id),
    )
    monkeypatch.setattr(census_module, "_task_present", lambda _pid, _task_id: True)
    assert _census()._snapshot().available == 0


def test_snapshot_child_task_read_failure_after_exit_is_tolerated(monkeypatch):
    """The same failed read is benign once the child has actually exited."""
    _stable_auxiliary_metrics(monkeypatch)
    starts: dict[int, int | None] = {1: 100, 2: 200}
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: starts.get(pid))
    monkeypatch.setattr(census_module, "_list_task_ids", lambda pid: {1: (1,), 2: (2,)}.get(pid))

    def read_children(pid, task_id):
        if pid == 2:
            starts[2] = None  # the child exits mid-read
            return None
        return {2} if (pid, task_id) == (1, 1) else set()

    monkeypatch.setattr(census_module, "_read_task_children", read_children)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.descendant_processes == 0


def test_snapshot_thread_count_failure_on_a_live_process_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    original = census_module._list_task_ids
    monkeypatch.setattr(
        census_module,
        "_list_task_ids",
        lambda pid: None if pid == 2 else original(pid),
    )
    assert _census()._snapshot().available == 0


def test_snapshot_thread_count_failure_after_exit_is_tolerated(monkeypatch):
    """A descendant that exits before its threads are counted is dropped, not fatal."""
    _stable_auxiliary_metrics(monkeypatch)
    root = _ProcessIdentity(1, 100)
    child = _ProcessIdentity(2, 200)
    monkeypatch.setattr(census_module, "_scan_process_tree", lambda _root: _tree(root, child))
    monkeypatch.setattr(
        census_module,
        "_stable_thread_count",
        lambda identity: census_module._VANISHED if identity.pid == 2 else 3,
    )
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: 100 if pid == 1 else None)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert (snapshot.descendant_processes, snapshot.descendant_threads) == (0, 0)


def test_snapshot_retries_a_transient_proc_exit_race(monkeypatch):
    _stable_auxiliary_metrics(monkeypatch)
    calls = 0

    def descendant_counts(_root):
        nonlocal calls
        calls += 1
        return None if calls == 1 else (1, 2, 2)

    monkeypatch.setattr(census_module, "_descendant_counts", descendant_counts)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.incomplete_attempts == 1
    assert snapshot.descendant_processes == 1


def _tree(*identities):
    return frozenset(identities)


def test_snapshot_rejects_pid_reuse_between_complete_scans(monkeypatch):
    """A pid seen under two start times means the tree is corrupt, not merely churning."""
    _stable_auxiliary_metrics(monkeypatch)
    root = _ProcessIdentity(1, 100)
    child = _ProcessIdentity(2, 200)
    replacement = _ProcessIdentity(2, 201)
    scans = iter([_tree(root, child), _tree(root, replacement)] * 3)
    monkeypatch.setattr(census_module, "_scan_process_tree", lambda _root: next(scans))

    snapshot = _census()._snapshot()
    assert snapshot.available == 0
    assert snapshot.incomplete_attempts == 3


@pytest.mark.parametrize("drift", ["child_exit", "child_creation", "full_turnover"])
def test_snapshot_tolerates_benign_churn_between_scans(monkeypatch, drift):
    """A rollout tree constantly creates and reaps workers; that must still be measurable."""
    _stable_auxiliary_metrics(monkeypatch)
    root = _ProcessIdentity(1, 100)
    child = _ProcessIdentity(2, 200)
    created = _ProcessIdentity(3, 300)
    first = _tree(root, child)
    second = {
        "child_exit": _tree(root),
        "child_creation": _tree(root, child, created),
        "full_turnover": _tree(root, created),
    }[drift]
    scans = iter((first, second))
    monkeypatch.setattr(census_module, "_scan_process_tree", lambda _root: next(scans))
    monkeypatch.setattr(census_module, "_stable_thread_count", lambda _identity: 2)
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: 100 if pid == 1 else 200)

    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.incomplete_attempts == 0
    assert snapshot.descendant_processes == len(second - {root})


def test_snapshot_discards_pid_reuse_attempt_before_accepting_a_stable_retry(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=True)
    calls = {2: 0}

    def start_time(pid):
        if pid == 2:
            calls[2] += 1
            return 200 if calls[2] == 1 else 201
        return 100

    monkeypatch.setattr(census_module, "_read_start_time", start_time)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.incomplete_attempts == 1


def test_snapshot_root_pid_reuse_is_unavailable(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=False)
    calls = {1: 0}

    def start_time(_pid):
        calls[1] += 1
        return 100 if calls[1] <= 2 else 101

    monkeypatch.setattr(census_module, "_read_start_time", start_time)
    assert _census()._snapshot().available == 0


def test_snapshot_root_exit_is_unavailable_not_an_empty_tree(monkeypatch):
    """Tolerating descendant exit must never let the root's own exit report a healthy zero."""
    _stable_auxiliary_metrics(monkeypatch)
    monkeypatch.setattr(census_module, "_read_start_time", lambda _pid: None)
    monkeypatch.setattr(census_module, "_list_task_ids", lambda _pid: (1,))
    monkeypatch.setattr(census_module, "_read_task_children", lambda _pid, _task: set())
    snapshot = _census()._snapshot()
    assert snapshot.available == 0
    assert snapshot.descendant_processes == -1


def test_snapshot_root_exit_during_the_walk_is_unavailable(monkeypatch):
    """The root disappearing mid-walk invalidates the sample even though children looked fine."""
    _stable_auxiliary_metrics(monkeypatch)
    starts: dict[int, int | None] = {1: 100}
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: starts.get(pid))
    monkeypatch.setattr(census_module, "_list_task_ids", lambda _pid: (1,))

    def read_children(_pid, _task_id):
        starts[1] = None  # the root exits while its own children are being read
        return

    monkeypatch.setattr(census_module, "_read_task_children", read_children)
    assert _census()._snapshot().available == 0


def test_descendant_pid_reuse_within_one_scan_is_unavailable(monkeypatch):
    """Two parents reporting the same pid under different start times is corruption."""
    _stable_auxiliary_metrics(monkeypatch)
    starts = {1: 100, 2: 200, 3: 300}
    tasks = {1: (1,), 2: (2,), 3: (3,)}
    children = {(1, 1): {2, 3}, (2, 2): {4}, (3, 3): {4}}
    shared_reads = {"count": 0}

    def read_start_time(pid):
        if pid != 4:
            return starts.get(pid)
        # the two parents always disagree, so no retry can settle the contradiction
        shared_reads["count"] += 1
        return 400 if shared_reads["count"] % 2 == 1 else 401

    monkeypatch.setattr(census_module, "_read_start_time", read_start_time)
    monkeypatch.setattr(census_module, "_list_task_ids", lambda pid: tasks.get(pid, (pid,)))
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: children.get((pid, task_id), set()),
    )
    assert _census()._snapshot().available == 0


def test_snapshot_complete_empty_tree_reports_real_zero(monkeypatch):
    _complete_proc_view(monkeypatch, with_child=False)
    snapshot = _census()._snapshot()
    assert snapshot.available == 1
    assert snapshot.incomplete_attempts == 0
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


def _snapshot(
    available: int,
    processes: int,
    threads: int,
    maximum: int,
    *,
    pids_current: int = 10,
    pids_max: int = 100,
    incomplete_attempts: int = 0,
) -> ProcessCensusSnapshot:
    return ProcessCensusSnapshot(
        available=available,
        descendant_processes=processes,
        descendant_threads=threads,
        max_descendant_threads=maximum,
        pids_current=pids_current,
        pids_max=pids_max,
        pids_headroom=pids_max - pids_current,
        cpu_quota_millicores=2000,
        affinity_cpus=2,
        incomplete_attempts=incomplete_attempts,
    )


def _set_baseline(census: GrpoProcessCensus, snapshot: ProcessCensusSnapshot) -> None:
    census._baseline = snapshot
    census._observe(snapshot)


def _set_terminal(census: GrpoProcessCensus, snapshot: ProcessCensusSnapshot) -> None:
    census._terminal = snapshot
    census._observe(snapshot)


def _sample(census: GrpoProcessCensus, monkeypatch, step: int, snapshot) -> None:
    monkeypatch.setattr(census, "_snapshot", lambda: snapshot)
    census.sample_step(step)


def test_transient_background_partial_does_not_erase_complete_required_checkpoints(monkeypatch):
    census = _census(expected_steps=(1, 2))
    _set_baseline(census, _snapshot(1, 2, 7, 4, pids_current=12))
    census._observe(_snapshot(0, -1, -1, -1, pids_current=95, incomplete_attempts=1))
    _sample(census, monkeypatch, 1, _snapshot(1, 3, 8, 5, pids_current=20))
    _sample(census, monkeypatch, 2, _snapshot(1, 4, 9, 6, pids_current=30))
    _set_terminal(census, _snapshot(1, 1, 3, 2, pids_current=15))

    summary = census.summary()
    assert summary["available"] == 1
    assert summary["incomplete_sample_count"] == 1
    assert summary["peak_pids_current"] == 95
    assert summary["minimum_pids_headroom"] == 5
    assert summary["peak_processes"] == 4
    assert summary["peak_threads"] == 9


def test_zero_step_resume_is_complete_with_baseline_terminal_and_cgroup_data():
    census = _census(expected_steps=())
    _set_baseline(census, _snapshot(1, 2, 7, 4, pids_current=12))
    _set_terminal(census, _snapshot(1, 1, 3, 2, pids_current=15))

    summary = census.summary()
    assert summary["available"] == 1
    assert summary["steps"] == []
    assert summary["missing_step_count"] == 0
    assert summary["per_step_processes"] == -1
    assert summary["per_step_threads"] == -1


@pytest.mark.parametrize("checkpoint", ["baseline", "step", "terminal"])
def test_partial_required_checkpoint_is_unavailable(monkeypatch, checkpoint):
    census = _census(expected_steps=(1,))
    complete = _snapshot(1, 2, 7, 4)
    partial = _snapshot(0, -1, -1, -1, incomplete_attempts=3)
    _set_baseline(census, partial if checkpoint == "baseline" else complete)
    _sample(census, monkeypatch, 1, partial if checkpoint == "step" else complete)
    _set_terminal(census, partial if checkpoint == "terminal" else complete)

    summary = census.summary()
    assert summary["available"] == 0
    assert summary["incomplete_sample_count"] == 3


def test_missing_duplicate_and_conflicting_steps_fail_closed(monkeypatch):
    complete = _snapshot(1, 2, 7, 4)

    missing = _census(expected_steps=(1, 2))
    _set_baseline(missing, complete)
    _sample(missing, monkeypatch, 1, complete)
    _set_terminal(missing, complete)
    assert missing.summary()["missing_step_count"] == 1
    assert missing.summary()["available"] == 0

    duplicate = _census(expected_steps=(1,))
    _set_baseline(duplicate, complete)
    _sample(duplicate, monkeypatch, 1, complete)
    _sample(duplicate, monkeypatch, 1, complete)
    _set_terminal(duplicate, complete)
    assert duplicate.summary()["duplicate_step_count"] == 1
    assert duplicate.summary()["available"] == 0

    conflicting = _census(expected_steps=(1,))
    _set_baseline(conflicting, complete)
    _sample(conflicting, monkeypatch, 1, complete)
    _sample(conflicting, monkeypatch, 1, _snapshot(1, 3, 8, 5))
    _set_terminal(conflicting, complete)
    assert conflicting.summary()["conflicting_step_count"] == 1
    assert conflicting.summary()["available"] == 0


def test_step_evidence_is_deterministic_and_separates_growth_from_terminal_state(monkeypatch):
    census = _census(expected_steps=(1, 2))
    _set_baseline(census, _snapshot(1, 1, 2, 2, pids_current=10))
    _sample(census, monkeypatch, 2, _snapshot(1, 7, 12, 8, pids_current=30))
    _sample(census, monkeypatch, 1, _snapshot(1, 3, 6, 4, pids_current=20))
    _set_terminal(census, _snapshot(1, 4, 7, 5, pids_current=18))

    summary = census.summary()
    assert summary["available"] == 1
    assert summary["steps"] == [
        {
            "optimizer_step": 1,
            "processes": 3,
            "threads": 6,
            "max_process_threads": 4,
            "pids_current": 20,
            "pids_max": 100,
            "pids_headroom": 80,
        },
        {
            "optimizer_step": 2,
            "processes": 7,
            "threads": 12,
            "max_process_threads": 8,
            "pids_current": 30,
            "pids_max": 100,
            "pids_headroom": 70,
        },
    ]
    assert summary["terminal_processes"] == 4
    assert summary["terminal_processes"] > summary["baseline_processes"]
    assert "cleanup" not in summary
    assert summary == census.summary()


def test_pids_peak_and_headroom_update_on_partial_samples():
    census = _census()
    census._observe(_snapshot(0, -1, -1, -1, pids_current=87, incomplete_attempts=1))
    summary = census.summary()
    assert summary["peak_pids_current"] == 87
    assert summary["minimum_pids_headroom"] == 13
    assert summary["incomplete_sample_count"] == 1


def test_process_census_reports_numeric_bounded_summary():
    census = GrpoProcessCensus(
        __import__("os").getpid(), expected_steps=(1,), interval_s=0.02
    ).start()
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
    time.sleep(0.1)
    census.sample_step(1)
    assert proc.wait(timeout=2) == 0
    summary = census.stop()

    cgroup_available = summary["pids_max"] >= 0 and summary["pids_current"] >= 0
    assert summary["available"] == int(cgroup_available)
    assert summary["peak_processes"] >= 3
    assert summary["peak_threads"] >= 5
    assert summary["peak_single_process_threads"] >= 4
    if cgroup_available:
        assert summary["peak_pids_current"] > 0
        assert summary["minimum_pids_headroom"] > 0
    assert summary["steps"][0]["optimizer_step"] == 1


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


def test_scan_keeps_the_tree_when_a_descendant_exits_mid_walk(monkeypatch):
    """The real walk, not a stubbed one: a rollout worker exiting must not void the sample.

    The churn test above replaces ``_scan_process_tree`` wholesale, so it cannot see a regression
    inside the walk itself. On a live GRPO tree something is always exiting, so aborting the walk on
    any descendant exit made a fully healthy run unmeasurable.
    """
    _stable_auxiliary_metrics(monkeypatch)
    starts = {1: 100, 2: 200, 3: 300}
    tasks = {1: (1,), 2: (2,), 3: (3,)}
    # pid 2 is reaped the moment the walk reaches it; pid 3 stays alive.
    children = {(1, 1): {2, 3}, (2, 2): None, (3, 3): set()}
    monkeypatch.setattr(census_module, "_read_start_time", lambda pid: starts.get(pid))
    monkeypatch.setattr(census_module, "_list_task_ids", lambda pid: tasks.get(pid))
    monkeypatch.setattr(
        census_module,
        "_read_task_children",
        lambda pid, task_id: children.get((pid, task_id)),
    )
    exited = {2}
    original_alive = census_module._liveness
    monkeypatch.setattr(
        census_module,
        "_liveness",
        lambda identity: (
            census_module._GONE if identity.pid in exited else original_alive(identity)
        ),
    )

    scanned = census_module._scan_process_tree(_ProcessIdentity(1, 100))

    assert scanned is not None, "a descendant exiting mid-walk must not void the whole scan"
    assert _ProcessIdentity(1, 100) in scanned
    assert _ProcessIdentity(3, 300) in scanned
