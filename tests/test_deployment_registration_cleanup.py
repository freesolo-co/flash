"""Incomplete hosted registrations are revoked without disturbing verified siblings."""

from threading import Event
from types import SimpleNamespace

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.supervise import transitions
from flash.serve.contract.errors import RegistrationRejected, ServingError
from flash.serve.deployment import deploy
from flash.server.platform import deployment_state
from flash.server.platform.internal_client import run_serving_org_id
from flash.server.routes import serving_completion
from tests._helpers.deployments import seed_ready_checkpoint
from tests.test_serve import _stub_adapter_config
from tests.test_server_api import _bearer, _identity_for_token, _login, _make_run
from tests.test_server_api import api as api


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("registration_started", [False, True])
def test_failure_before_smoke_cleans_only_attempted_registration(
    api, monkeypatch, cleanup_fails, registration_started
):
    key = _login()
    run_id = _make_run(api, key, "done")
    checkpoint_id, sibling = f"{run_id}/final", f"{run_id}/step-20"
    seed_ready_checkpoint(run_id, sibling)
    before = status.get_status(run_id)
    undeploys = []

    def fail_deploy(**kwargs):
        if registration_started:
            kwargs["before_register"]({})
        raise ServingError(
            "checkpoint readiness timed out" if registration_started else "bad artifact"
        )

    def undeploy(target, *, org_id, timeout_s=None):
        undeploys.append((target, org_id))
        if cleanup_fails:
            raise ServingError("cleanup unavailable")
        return {"disabled_checkpoints": [target]}

    monkeypatch.setattr(deploy, "deploy_adapter", fail_deploy)
    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    response = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"checkpoint_id": checkpoint_id},
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    pending = registration_started and cleanup_fails
    assert response.json()["state"] == ("cleanup_pending" if pending else "failed")
    assert undeploys == (
        [(checkpoint_id, run_serving_org_id(before))] if registration_started else []
    )
    if pending:
        assert response.json()["retryable"] is True
        assert "cleanup unavailable" in response.json()["error"]
    assert status.get_status(run_id).deployments[sibling] == before.deployments[sibling]


@pytest.mark.parametrize("initial_state", ["queued", "smoke_testing", "cleanup_pending"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_recovery_fences_and_removes_unfinished_registration(
    api, monkeypatch, initial_state, cleanup_fails
):
    key = _login()
    run_id = _make_run(api, key, "done")
    checkpoint_id, sibling = f"{run_id}/step-20", f"{run_id}/final"
    seed_ready_checkpoint(run_id, sibling)
    current = status.get_status(run_id)
    attempt = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": 20,
        "generation": 1,
        "state": initial_state,
        "registration_started": True,
        "requested_at": 1.0,
    }
    current.deployments[checkpoint_id] = attempt
    state._save_status(current)
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": run_id}])
    reports, undeploys = [], []
    monkeypatch.setattr(deployment_state, "_enqueue_deployment_report", reports.append)

    def undeploy(target, *, org_id, timeout_s=None):
        undeploys.append((target, org_id))
        fenced = status.get_status(run_id).deployments[target]
        assert fenced["state"] == "cleanup_pending"
        assert fenced["generation"] >= attempt["generation"] + (initial_state != "cleanup_pending")
        late = {**attempt, "state": "ready", "verified_at": 123.0}
        assert transitions.mark_checkpoint_deployed(run_id, late).deployments[target] == fenced
        if cleanup_fails:
            raise ServingError("cleanup unavailable")
        return {"disabled_checkpoints": [target]}

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    assert serving_completion.recover_deployments() == 1
    assert undeploys == []
    serving_completion.cleanup_recovered_deployments(Event())
    saved = status.get_status(run_id)
    assert saved.deployments[sibling] == current.deployments[sibling]
    assert saved.deployments[checkpoint_id]["state"] == (
        "cleanup_pending" if cleanup_fails else "undeployed"
    )
    assert undeploys == [(checkpoint_id, run_serving_org_id(current))]
    assert reports[-1].deployments == saved.deployments
    if cleanup_fails:
        assert saved.deployments[checkpoint_id]["retryable"] is True
        cleanup_fails = False
        assert serving_completion.recover_deployments() == 1
        serving_completion.cleanup_recovered_deployments(Event())
        assert status.get_status(run_id).deployments[checkpoint_id]["state"] == "undeployed"
    assert serving_completion.recover_deployments() == 0


def test_recovery_without_org_retains_cleanup_pending(api, monkeypatch):
    key = _login()
    run_id = _make_run(api, key, "done")
    checkpoint_id = f"{run_id}/final"
    transitions.mark_deployment_pending(
        run_id,
        {
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": None,
            "generation": 1,
            "state": "queued",
            "registration_started": True,
        },
    )
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": run_id}])
    monkeypatch.setattr(serving_completion, "run_serving_org_id", lambda _status: "")
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda *_args, **_kwargs: pytest.fail("unscoped cleanup")
    )
    assert serving_completion.recover_deployments() == 1
    serving_completion.cleanup_recovered_deployments(Event())
    record = status.get_status(run_id).deployments[checkpoint_id]
    assert record["state"] == "cleanup_pending"
    assert "no organization scope" in record["error"]


@pytest.mark.parametrize("error_type", [RegistrationRejected, ServingError])
@pytest.mark.parametrize("status_code", [401, 403, 409, 422])
def test_only_registration_rejection_skips_cleanup(api, monkeypatch, error_type, status_code):
    key = _login()
    run_id = _make_run(api, key, "done")
    deleted = []

    def fail(**kwargs):
        kwargs["before_register"]({})
        raise error_type("rejected", status_code=status_code)

    monkeypatch.setattr(deploy, "deploy_adapter", fail)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **_: deleted.append(target))
    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert deleted == ([] if error_type is RegistrationRejected else [f"{run_id}/final"])


def test_recovery_uses_persisted_registration_org(api, monkeypatch):
    key = _login()
    run_id = _make_run(api, key, "done")
    original = status.get_status(run_id)
    original.billing_context = {}
    original.platform_context = {}
    state._save_status(original)

    def fail(**kwargs):
        kwargs["before_register"]({})
        raise ServingError("ambiguous POST")

    def unavailable(*args, **kwargs):
        raise ServingError("offline")

    monkeypatch.setattr(deploy, "deploy_adapter", fail)
    monkeypatch.setattr(deploy, "undeploy_adapter", unavailable)
    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": f"{run_id}/final"}, headers=_bearer(key)
    )
    assert response.status_code == 200
    record = status.get_status(run_id).deployments[f"{run_id}/final"]
    assert record["serving_org_id"] == _identity_for_token(key)["org_id"]
    monkeypatch.setattr(serving_completion, "run_serving_org_id", lambda _: "wrong-current-org")
    deleted = []
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **kw: deleted.append((target, kw))
    )
    serving_completion.recover_deployments()
    serving_completion.cleanup_recovered_deployments(Event())
    assert deleted == [(f"{run_id}/final", {"org_id": record["serving_org_id"], "timeout_s": 5.0})]


def test_recovery_worker_respects_stop_and_budget(api, monkeypatch):
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": "missing"}])
    monkeypatch.setattr(
        serving_completion.run_status, "get_status", lambda _: pytest.fail("read past budget")
    )
    stop = Event()
    stop.set()
    assert serving_completion.cleanup_recovered_deployments(stop) == 0
    assert serving_completion.cleanup_recovered_deployments(Event(), budget_seconds=0) == 0


@pytest.fixture
def blocked_recovery(monkeypatch):
    started, release = Event(), Event()

    def cleanup(stop):
        started.set()
        while not release.is_set() and not stop.wait(0.01):
            pass

    monkeypatch.setattr(serving_completion, "recover_deployments", lambda: 1)
    monkeypatch.setattr(serving_completion, "retry_deployment_cleanup", cleanup)
    try:
        yield started
    finally:
        release.set()


def test_startup_serves_requests_while_remote_cleanup_is_blocked(blocked_recovery, api):
    assert blocked_recovery.wait(2)
    assert api.get("/v1/me", headers=_bearer(_login())).status_code == 200


@pytest.mark.parametrize("status_code", [None, 408, 503])
def test_ambiguous_registration_preserves_mismatched_binding(
    api, monkeypatch, tmp_path, status_code
):
    key = _login()
    run_id = _make_run(api, key, "done")
    checkpoint_id, sibling = f"{run_id}/final", f"{run_id}/step-20"
    seed_ready_checkpoint(run_id, sibling)
    before = status.get_status(run_id).deployments[sibling]
    _stub_adapter_config(monkeypatch, tmp_path)
    monkeypatch.setattr(deploy, "resolve_artifact_revision", lambda _: "a" * 40)
    existing = {}
    reads, deletes = [], []

    def registered(*args, **kwargs):
        reads.append(checkpoint_id)
        return existing or None

    def ambiguous(method, url, **kwargs):
        assert method == "POST"
        persisted = status.get_status(run_id).deployments[checkpoint_id]
        assert persisted["registration_started"] is True
        assert persisted["registration_identity"] == kwargs["json"]
        existing.update({**kwargs["json"], "artifact_digest": "another-artifact"})
        raise ServingError("registration response lost", status_code=status_code)

    monkeypatch.setattr(deploy, "_registered_adapter", registered)
    monkeypatch.setattr(deploy.transport, "serving_request", ambiguous)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **_: deletes.append(target))
    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": checkpoint_id}, headers=_bearer(key)
    )
    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert reads == [checkpoint_id, checkpoint_id]
    assert deletes == []
    assert "different immutable identity" in response.json()["error"]
    assert status.get_status(run_id).deployments[sibling] == before


@pytest.mark.parametrize("interruption", ["failure", "budget", "locked"])
def test_cleanup_worker_retries_without_another_restart(api, monkeypatch, interruption):
    key = _login()
    run_id = _make_run(api, key, "done")
    targets = [f"{run_id}/final", f"{run_id}/step-20"]
    for target in targets:
        seed_ready_checkpoint(run_id, target)
        transitions.mark_deployment_cleanup_pending(run_id, "interrupted", target)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(serving_completion, "time", SimpleNamespace(monotonic=lambda: clock.now))
    deleted, waits = [], []
    stop = Event()
    lock = serving_completion.deployment_locks._deploy_lock(run_id)
    if interruption == "locked":
        lock.acquire()

    def undeploy(target, **kwargs):
        assert 0 < kwargs["timeout_s"] <= 5
        deleted.append(target)
        if len(deleted) == 1:
            if interruption == "failure":
                raise ServingError("temporarily unavailable")
            if interruption == "budget":
                clock.now += 30

    def wait(delay):
        waits.append(delay)
        assert len(waits) == 1, "cleanup did not finish on the next pass"
        if interruption == "locked":
            lock.release()
        return False

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    monkeypatch.setattr(stop, "wait", wait)
    try:
        assert serving_completion.retry_deployment_cleanup(stop) == 2
    finally:
        if interruption == "locked" and not waits:
            lock.release()
    assert waits == [1.0]
    assert len(deleted) == (3 if interruption == "failure" else 2)
    assert all(status.get_status(run_id).deployments[t]["state"] == "undeployed" for t in targets)


def test_cleanup_retry_backoff_is_bounded_and_stoppable(api, monkeypatch):
    key = _login()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_cleanup_pending(run_id, "interrupted", target)
    stop, waits, attempts = Event(), [], []

    def unavailable(*args, **kwargs):
        attempts.append(args)
        raise ServingError("offline")

    def wait(delay):
        waits.append(delay)
        if len(waits) == 7:
            stop.set()
        return stop.is_set()

    monkeypatch.setattr(deploy, "undeploy_adapter", unavailable)
    monkeypatch.setattr(stop, "wait", wait)
    assert serving_completion.retry_deployment_cleanup(stop) == 0
    assert waits == [1, 2, 4, 8, 16, 30, 30]
    assert len(attempts) == len(waits)
    assert status.get_status(run_id).deployments[target]["state"] == "cleanup_pending"


def test_cleanup_retry_rechecks_state_after_backoff(api, monkeypatch):
    key = _login()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_cleanup_pending(run_id, "interrupted", target)
    stop, deleted = Event(), []

    def unavailable(checkpoint_id, **kwargs):
        deleted.append(checkpoint_id)
        raise ServingError("offline")

    def wait(delay):
        seed_ready_checkpoint(run_id, target)
        return False

    monkeypatch.setattr(deploy, "undeploy_adapter", unavailable)
    monkeypatch.setattr(stop, "wait", wait)
    assert serving_completion.retry_deployment_cleanup(stop) == 0
    assert deleted == [target]
    assert status.get_status(run_id).deployments[target]["state"] == "ready"
