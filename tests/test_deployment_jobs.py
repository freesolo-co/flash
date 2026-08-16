"""The background job runner: thread handling, shutdown draining, and start refusal.

The runner is instance-scoped, so these also prove two planes in one process cannot close each
other's intake.
"""

from __future__ import annotations

import threading

import pytest

from flash.server.platform.deployment_jobs import (
    DeploymentJobStartError,
    ThreadDeploymentJobRunner,
)


def test_a_job_runs_on_a_daemon_background_thread(monkeypatch):
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()
    done = threading.Event()
    seen: dict = {}

    def record(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        seen["daemon"] = threading.current_thread().daemon
        seen["is_background"] = threading.current_thread() is not threading.main_thread()
        done.set()

    assert runner.start(record, "done", final=True) is False
    assert done.wait(5) is True
    assert seen["args"] == ("done",)
    assert seen["kwargs"] == {"final": True}
    assert seen["daemon"] is True
    assert seen["is_background"] is True


def test_a_finished_job_is_dropped_from_the_live_set(monkeypatch):
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()
    release = threading.Event()

    runner.start(release.wait, 5)
    assert len(runner.live_jobs()) == 1

    release.set()
    assert runner.drain(5) is True
    assert runner.live_jobs() == ()


def test_sync_mode_runs_the_job_inline_and_reports_it(monkeypatch):
    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
    runner = ThreadDeploymentJobRunner()
    ran = []

    assert runner.start(lambda: ran.append(threading.current_thread())) is True
    assert ran == [threading.current_thread()]
    assert runner.live_jobs() == ()


def test_draining_refuses_new_jobs(monkeypatch):
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()

    assert runner.drain(1) is True
    with pytest.raises(DeploymentJobStartError, match="shutting down"):
        runner.start(lambda: None)


def test_reopening_accepts_jobs_again(monkeypatch):
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()
    runner.drain(1)

    runner.open()

    done = threading.Event()
    assert runner.start(done.set) is False
    assert done.wait(5) is True


def test_a_drain_that_outlasts_its_timeout_reports_failure(monkeypatch):
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()
    release = threading.Event()
    runner.start(release.wait, 30)

    try:
        assert runner.drain(0.05) is False
    finally:
        release.set()
        runner.drain(5)


def test_a_thread_that_cannot_start_is_not_left_in_the_live_set(monkeypatch):
    """A job that never ran must not make shutdown wait for it forever."""
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()

    def refuse_to_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse_to_start)

    with pytest.raises(DeploymentJobStartError, match="can't start new thread"):
        runner.start(lambda: None)

    assert runner.live_jobs() == ()


def test_two_runners_do_not_close_each_others_intake(monkeypatch):
    """Instance-scoped: one plane draining must not stop another plane accepting."""
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    draining = ThreadDeploymentJobRunner()
    other = ThreadDeploymentJobRunner()

    draining.drain(1)

    done = threading.Event()
    assert other.start(done.set) is False
    assert done.wait(5) is True


def test_a_job_is_started_before_a_waiter_can_observe_the_live_set(monkeypatch):
    """The thread must start while the intake lock is held.

    Registering the job and starting it have to be one atomic step. If the lock were released
    between them, a concurrent `drain` could see an empty live set, conclude every job had
    finished, and let shutdown proceed while this job was still about to run.
    """
    monkeypatch.delenv("FLASH_DEPLOY_SYNC", raising=False)
    runner = ThreadDeploymentJobRunner()
    started_while_locked = []

    class ThreadStub:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            started_while_locked.append(runner._lock.locked())

    monkeypatch.setattr(threading, "Thread", ThreadStub)

    assert runner.start(lambda: None) is False
    assert started_while_locked == [True]


def test_the_built_service_locks_on_the_same_mutex_cancellation_and_export_take():
    """The wired lock provider must BE `platform.locks._deploy_lock`, not an equivalent of it.

    Deploy hands its held lock to the background job, and undeploy, export, cancellation teardown,
    and startup recovery all contend on the same per-run mutex. A provider that returned a private
    lock would pass every deployment test in isolation and still let a cancel tear a run down while
    its deploy was mid-activation, because the two would be serialising on different objects.
    """
    from flash.server.platform.deployment_adapters import build_deployment_service
    from flash.server.platform.locks import _deploy_lock

    service = build_deployment_service()

    assert service._lock_provider is _deploy_lock
    # and it is per-run identity, not a fresh lock per call: the handoff depends on the job
    # releasing the very object the request acquired.
    assert service._lock_provider("run-a") is _deploy_lock("run-a")
    assert service._lock_provider("run-a") is not _deploy_lock("run-b")
