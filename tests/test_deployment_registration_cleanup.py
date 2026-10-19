"""Incomplete hosted registrations are revoked without disturbing verified siblings."""

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.supervise import transitions
from flash.serve.contract.errors import ServingError
from flash.serve.deployment import deploy
from flash.server.platform import deployment_state
from flash.server.platform.internal_client import run_serving_org_id
from flash.server.routes import serving_completion
from tests._helpers.deployments import seed_ready_checkpoint
from tests.test_server_api import _bearer, _login, _make_run
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
            kwargs["before_register"]()
        raise ServingError(
            "checkpoint readiness timed out" if registration_started else "bad artifact"
        )

    def undeploy(target, *, org_id):
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
        "requested_at": 1.0,
    }
    current.deployments[checkpoint_id] = attempt
    state._save_status(current)
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": run_id}])
    reports, undeploys = [], []
    monkeypatch.setattr(deployment_state, "_enqueue_deployment_report", reports.append)

    def undeploy(target, *, org_id):
        undeploys.append((target, org_id))
        fenced = status.get_status(run_id).deployments[target]
        assert fenced["state"] == "cleanup_pending"
        assert fenced["generation"] > attempt["generation"]
        late = {**attempt, "state": "ready", "verified_at": 123.0}
        assert transitions.mark_checkpoint_deployed(run_id, late).deployments[target] == fenced
        if cleanup_fails:
            raise ServingError("cleanup unavailable")
        return {"disabled_checkpoints": [target]}

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    assert serving_completion.recover_deployments() == 1
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
        },
    )
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": run_id}])
    monkeypatch.setattr(serving_completion, "run_serving_org_id", lambda _status: "")
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda *_args, **_kwargs: pytest.fail("unscoped cleanup")
    )
    assert serving_completion.recover_deployments() == 1
    record = status.get_status(run_id).deployments[checkpoint_id]
    assert record["state"] == "cleanup_pending"
    assert "no organization scope" in record["error"]
