"""Hermetic cleanup coverage for per-run process and thread locks."""

from __future__ import annotations

import pytest

import flash.server.platform.locks as locks


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
