"""A failed persistence read must not stop cleanup for every other run."""

from threading import Event

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.supervise import transitions
from flash.serve.deployment import deploy
from flash.server.routes import serving_completion
from tests._helpers.deployments import seed_ready_checkpoint
from tests.test_server_api import _make_run, _new_user_token
from tests.test_server_api import api as api


@pytest.mark.parametrize("failure", [OSError("storage unavailable"), ValueError("corrupt status")])
@pytest.mark.parametrize("failure_point", ["status", "lock"])
def test_unreadable_run_does_not_starve_later_cleanup(
    api, monkeypatch, caplog, failure, failure_point
):
    run_id = _make_run(api, _new_user_token(), "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_cleanup_pending(run_id, "interrupted", target)
    original_get = status.get_status

    def read_status(name):
        if name == "unreadable-run" and failure_point == "status":
            raise failure
        return original_get(name)

    monkeypatch.setattr(status, "get_status", read_status)
    original_lock = serving_completion.deployment_locks._deploy_lock

    class UnavailableLock:
        def acquire(self, blocking):
            raise failure

    def lock_for_run(name):
        if name == "unreadable-run" and failure_point == "lock":
            return UnavailableLock()
        return original_lock(name)

    monkeypatch.setattr(serving_completion.deployment_locks, "_deploy_lock", lock_for_run)
    monkeypatch.setattr(
        serving_completion.db,
        "all_runs",
        lambda: [{"run_id": "unreadable-run"}, {"run_id": run_id}],
    )
    deleted = []
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **kwargs: deleted.append(target))
    assert serving_completion.cleanup_pending_deployments(Event()) is True
    assert deleted == [target]
    assert status.get_status(run_id).deployments[target]["state"] == "undeployed"
    assert "unreadable-run" in caplog.text
    lock = original_lock("unreadable-run")
    assert lock.acquire(blocking=False)
    lock.release()


def test_cleanup_worker_survives_transient_database_failure(monkeypatch, caplog):
    stop = Event()
    scans, waits = [], []

    def all_runs():
        scans.append(None)
        if len(scans) == 1:
            raise OSError("database temporarily unavailable")
        return []

    def wait(delay):
        waits.append(delay)
        return len(waits) == 2

    monkeypatch.setattr(serving_completion.db, "all_runs", all_runs)
    monkeypatch.setattr(stop, "wait", wait)
    serving_completion.run_deployment_cleanup_worker(stop)
    assert waits == [1.0, 30.0]
    assert len(scans) == 2
    assert "database temporarily unavailable" in caplog.text


def test_failed_launch_lock_acquisition_closes_descriptor(tmp_path, monkeypatch):
    import os

    from flash.runner.lifecycle import claim_lock

    monkeypatch.setattr(state, "RUNS_DIR", tmp_path)
    descriptors = []

    def fail_flock(fd, operation):
        descriptors.append(fd)
        raise OSError("lock filesystem unavailable")

    monkeypatch.setattr(claim_lock.fcntl, "flock", fail_flock)
    with pytest.raises(OSError, match="lock filesystem unavailable"):
        claim_lock.try_acquire("audit-run")
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(descriptors[0])
