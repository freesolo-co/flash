"""Hermetic cleanup coverage for per-run process and thread locks."""

from __future__ import annotations

import multiprocessing
import os

import pytest

import flash.server.platform.locks as locks


def _try_submission_lock(runs_dir: str, lock_name: str, result) -> None:
    import flash.runner.lifecycle.state as process_state
    import flash.server.platform.locks as process_locks

    process_state.RUNS_DIR = runs_dir
    lock = process_locks.submission_lock(lock_name)
    acquired = lock.acquire(blocking=False)
    result.put(acquired)
    if acquired:
        lock.release()


def test_nonblocking_file_lock_failure_closes_descriptor_and_releases_thread_lock(
    monkeypatch,
) -> None:
    """A contended process lock must close its descriptor and leave the local mutex reusable."""
    closed = []
    lock = locks._RunLock("flash-1")
    monkeypatch.setattr(locks.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(locks.os, "open", lambda *args, **kwargs: 55)
    monkeypatch.setattr(locks.os, "close", closed.append)
    monkeypatch.setattr(
        locks.fcntl,
        "flock",
        lambda *args: (_ for _ in ()).throw(BlockingIOError("busy")),
    )

    assert lock.acquire(blocking=False) is False
    assert closed == [55]
    assert lock._lock.acquire(blocking=False) is True
    lock._lock.release()


def test_unexpected_file_lock_failure_cleans_up_and_reraises(monkeypatch) -> None:
    """Unexpected flock errors must release both resources before propagating the original exception."""
    closed = []
    lock = locks._RunLock("flash-1")
    monkeypatch.setattr(locks.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(locks.os, "open", lambda *args, **kwargs: 77)
    monkeypatch.setattr(locks.os, "close", closed.append)
    monkeypatch.setattr(
        locks.fcntl,
        "flock",
        lambda *args: (_ for _ in ()).throw(OSError("filesystem failure")),
    )

    with pytest.raises(OSError, match="filesystem failure"):
        lock.acquire()

    assert closed == [77]
    assert lock._lock.acquire(blocking=False) is True
    lock._lock.release()


def test_releasing_a_nonheld_lock_is_rejected() -> None:
    """Release without ownership must fail clearly instead of unlocking an unrelated mutex."""
    with pytest.raises(RuntimeError, match="deploy lock is not held"):
        locks._RunLock("flash-1").release()


def test_submission_lock_contends_across_processes(tmp_path, monkeypatch) -> None:
    import flash.runner.lifecycle.state as runner_state

    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(runner_state, "RUNS_DIR", runs_dir)
    lock_name = "a" * 64
    held = locks.submission_lock(lock_name)
    assert held.acquire(blocking=False) is True
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_try_submission_lock, args=(runs_dir, lock_name, result))
    try:
        process.start()
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert result.get(timeout=5) is False
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        held.release()
    assert os.path.exists(os.path.join(runs_dir, f"{lock_name}.submission.lock"))
