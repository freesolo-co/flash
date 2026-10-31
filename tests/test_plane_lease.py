"""One control plane per state dir runs recovery; a plane started beside it waits its turn."""

import fcntl
import os
import time

import pytest
from fastapi.testclient import TestClient

import flash.runner.lifecycle.submit as runner_submit
import flash.server.platform.deployment_jobs as deployment_jobs
from flash.server.platform import plane_lease


def _hold_elsewhere() -> int:
    """Lock the lease through another open file description, as a second process would."""
    fd = os.open(plane_lease.lease_path(), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _release_elsewhere(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def test_lease_is_exclusive_until_the_holder_releases():
    other = _hold_elsewhere()
    assert not plane_lease.try_acquire()
    assert not plane_lease.held()

    _release_elsewhere(other)
    assert plane_lease.try_acquire()
    assert plane_lease.held()
    # a second plane is now the one shut out.
    with pytest.raises(BlockingIOError):
        _hold_elsewhere()

    plane_lease.release()
    assert not plane_lease.held()
    _release_elsewhere(_hold_elsewhere())


def _stub_lifespan(monkeypatch, recovered: list) -> None:
    import flash.runner.lifecycle.reporting as runner_reporting
    import flash.server.asgi.app as app_mod
    import flash.server.billing.retry as billing_retry
    import flash.server.domain.ops.repo_cleanup as repo_cleanup
    import flash.server.platform.runtime as server_runtime
    from flash.providers.core import preflight
    from flash.server.routes import serving_completion

    monkeypatch.setattr(preflight, "check_run_preflight", lambda: None)
    monkeypatch.setattr(deployment_jobs, "_open_deployment_jobs", lambda: None)
    monkeypatch.setattr(deployment_jobs, "start_deployment_job", lambda *a, **k: False)
    monkeypatch.setattr(deployment_jobs, "_wait_for_deployment_jobs", lambda _timeout: True)
    monkeypatch.setattr(runner_reporting, "_open_status_reporter", lambda: None)
    monkeypatch.setattr(runner_reporting, "_shutdown_status_reporter", lambda *a, **k: True)
    monkeypatch.setattr(server_runtime, "recover_runs", lambda: recovered.append("runs"))
    monkeypatch.setattr(
        serving_completion, "recover_deployments", lambda: recovered.append("deployments") or 0
    )
    monkeypatch.setattr(serving_completion, "replay_status_reports", lambda _stop: 0)
    monkeypatch.setattr(billing_retry, "charge_retry_enabled", lambda: False)
    monkeypatch.setattr(repo_cleanup, "repo_cleanup_enabled", lambda: False)
    monkeypatch.setattr(app_mod, "_instance_providers_configured", lambda: False)


def test_lone_plane_recovers_before_serving(monkeypatch):
    import flash.server.asgi.app as app_mod

    recovered = []
    _stub_lifespan(monkeypatch, recovered)

    with TestClient(app_mod.create_app()) as client:
        assert recovered == ["runs", "deployments"]
        assert not getattr(client.app.state, "plane_standby", False)
        assert plane_lease.held()

    # shutdown hands the lease to whichever plane starts next.
    assert not plane_lease.held()


def test_plane_beside_a_draining_one_serves_and_takes_over_when_it_exits(monkeypatch):
    import flash.server.asgi.app as app_mod

    recovered = []
    _stub_lifespan(monkeypatch, recovered)
    old_plane = _hold_elsewhere()

    with TestClient(app_mod.create_app()) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.app.state.plane_standby is True
        assert recovered == []

        _release_elsewhere(old_plane)
        deadline = time.monotonic() + 10
        while recovered != ["runs", "deployments"] and time.monotonic() < deadline:
            time.sleep(0.05)

        assert recovered == ["runs", "deployments"]
        assert client.app.state.plane_standby is False
        assert plane_lease.held()


def test_queue_only_submission_persists_without_dispatch(monkeypatch):
    calls = []

    class Prepared:
        public_spec = type("Spec", (), {"run_id": "run-1"})()

    class Owner:
        def reserve(self, *_args, **_kwargs):
            raise AssertionError("a standby plane must not reserve local supervision")

    monkeypatch.setattr(
        runner_submit,
        "_submit_prepared_job",
        lambda prepared, *, dispatch=None, **kwargs: calls.append(dispatch) or "queued-status",
    )

    status = runner_submit.submit_job(
        None, background=True, prepared_job=Prepared(), supervisor=Owner(), queue_only=True
    )

    assert status == "queued-status"
    assert calls == [runner_submit._leave_queued]
