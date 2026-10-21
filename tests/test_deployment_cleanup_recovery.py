"""Restart ownership, fair cleanup scheduling, and bounded retry diagnostics."""

from threading import Event
from types import SimpleNamespace

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.supervise import transitions
from flash.serve.contract.errors import ServingError
from flash.serve.deployment import deploy
from flash.server.routes import serving_completion
from tests._helpers.deployments import seed_ready_checkpoint
from tests.test_server_api import _bearer, _login, _make_run
from tests.test_server_api import api as api


@pytest.mark.parametrize(
    "stage",
    [
        "before_registration",
        "matching_binding",
        "conflicting_binding",
        "unavailable_registry",
        "slow_registry",
    ],
)
def test_restart_uses_durable_registration_intent(api, monkeypatch, stage):
    key = _login()
    run_id = _make_run(api, key, "done")
    target, sibling = f"{run_id}/final", f"{run_id}/step-20"
    seed_ready_checkpoint(run_id, sibling)
    snapshot = []
    identity = {
        "org_id": "org-test",
        "run_id": run_id,
        "checkpoint_step": None,
        "model_id": target,
        "artifact_digest": "expected-artifact",
    }

    def interrupted(**kwargs):
        assert status.get_status(run_id).deployments[target]["registration_started"] is False
        if stage != "before_registration":
            identity["org_id"] = kwargs["org_id"]
            kwargs["before_register"](identity)
        snapshot.append(status.get_status(run_id))
        raise ServingError("interrupted")

    # Restore exactly what was durable at the interruption, before normal error handling ran.
    with monkeypatch.context() as patch:
        patch.setattr(deploy, "deploy_adapter", interrupted)
        patch.setattr(serving_completion, "_cleanup_failed_deployment", lambda *a, **kw: True)
        response = api.post(
            f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": target}, headers=_bearer(key)
        )
        assert response.status_code == 200
    state._save_status(snapshot[0])
    reads, deletes = [], []
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(serving_completion, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def registered(org_id, checkpoint_id, **kwargs):
        reads.append((org_id, checkpoint_id))
        assert kwargs["timeout_s"] <= 5
        if stage == "unavailable_registry":
            raise ServingError("registry unavailable")
        if stage == "slow_registry":
            clock.now += 6
        return {
            **identity,
            "artifact_digest": "other" if stage == "conflicting_binding" else "expected-artifact",
        }

    monkeypatch.setattr(deploy, "_registered_adapter", registered)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **kw: deletes.append(target))
    assert serving_completion.recover_deployments() == 1
    serving_completion.cleanup_recovered_deployments(Event())
    recovered = status.get_status(run_id)
    if stage == "before_registration":
        assert reads == deletes == []
        assert recovered.deployments[target]["state"] == "failed"
    else:
        assert snapshot[0].deployments[target]["registration_identity"] == identity
        assert reads == [(identity["org_id"], target)]
        assert deletes == ([target] if stage == "matching_binding" else [])
        assert recovered.deployments[target]["state"] == (
            "cleanup_pending"
            if stage in {"unavailable_registry", "slow_registry"}
            else "undeployed"
        )
    assert recovered.deployments[sibling] == snapshot[0].deployments[sibling]
    late = {
        **snapshot[0].deployments[target],
        "state": "ready",
        "verified_at": 123.0,
        "deployment_generation": "generation-a",
    }
    assert (
        transitions.mark_checkpoint_deployed(run_id, late).deployments[target]
        == recovered.deployments[target]
    )


def test_registration_does_not_start_if_intent_cannot_be_persisted(api, monkeypatch):
    key = _login()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    original = transitions.mark_deployment_pending

    def save(run_id, record, **kwargs):
        if record.get("registration_started"):
            raise OSError("cannot persist registration intent")
        return original(run_id, record, **kwargs)

    def register(**kwargs):
        kwargs["before_register"]({})
        pytest.fail("POST must not follow a failed intent write")

    monkeypatch.setattr(transitions, "mark_deployment_pending", save)
    monkeypatch.setattr(deploy, "deploy_adapter", register)
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda *a, **kw: pytest.fail("unexpected DELETE")
    )
    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": target}, headers=_bearer(key)
    )
    assert response.json()["state"] == "failed"
    assert "cannot persist registration intent" in response.json()["error"]
    assert status.get_status(run_id).deployments[target]["registration_started"] is False


@pytest.mark.parametrize("multiple_runs", [False, True])
def test_cleanup_reaches_later_checkpoints_despite_repeated_timeouts(
    api, monkeypatch, multiple_runs
):
    key = _login()
    run_id = _make_run(api, key, "done")
    targets = []
    for i in range(7):
        if multiple_runs and i:
            run_id = _make_run(api, key, "done")
        target = f"{run_id}/step-{i + 1}"
        targets.append((run_id, target))
        seed_ready_checkpoint(run_id, target)
        transitions.mark_deployment_cleanup_pending(run_id, "original failure", target)
    targets.sort()
    last_run, last_target = targets[-1]
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(serving_completion, "time", SimpleNamespace(monotonic=lambda: clock.now))
    stop, deleted, waits = Event(), [], []

    def undeploy(target, **kwargs):
        deleted.append(target)
        clock.now += kwargs["timeout_s"]
        if target != last_target:
            raise ServingError("request timed out")

    def wait(delay):
        waits.append(delay)
        assert last_target in deleted, "later checkpoints were starved by earlier failures"
        assert clock.now >= 35, "the old whole-pass budget must not skip the seventh checkpoint"
        stop.set()
        return True

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    monkeypatch.setattr(stop, "wait", wait)
    serving_completion.retry_deployment_cleanup(stop)
    assert status.get_status(last_run).deployments[last_target]["state"] == "undeployed"
    assert waits == [1.0]


@pytest.mark.parametrize("legacy_composite", [False, True])
def test_cleanup_retry_keeps_original_cause_and_only_latest_failure(
    api, monkeypatch, legacy_composite
):
    key = _login()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_cleanup_pending(run_id, "readiness timed out", target)
    if legacy_composite:
        current = status.get_status(run_id)
        current.deployments[target].pop("deployment_error")
        current.deployments[target]["error"] += "; checkpoint cleanup failed: old" * 20
        state._save_status(current)
    stop, attempts, waits = Event(), [], []

    def unavailable(*args, **kwargs):
        attempts.append(args)
        raise ServingError(f"offline-{len(attempts)}")

    def wait(delay):
        waits.append(delay)
        record = status.get_status(run_id).deployments[target]
        assert record["deployment_error"] == "readiness timed out"
        assert record["cleanup_error"] == f"offline-{len(attempts)}"
        assert (
            record["error"]
            == f"readiness timed out; checkpoint cleanup failed: offline-{len(attempts)}"
        )
        if len(attempts) == 25:
            # Restart recovery must not replace the original cause with a generic restart error.
            serving_completion.recover_deployments()
        if len(attempts) == 50:
            stop.set()
        return stop.is_set()

    monkeypatch.setattr(deploy, "undeploy_adapter", unavailable)
    monkeypatch.setattr(stop, "wait", wait)
    serving_completion.retry_deployment_cleanup(stop)
    assert waits == [1, 2, 4, 8, 16] + [30] * 45
    assert len(attempts) == 50
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **kw: None)
    serving_completion.cleanup_recovered_deployments(Event())
    record = status.get_status(run_id).deployments[target]
    assert record["state"] == "undeployed"
    assert not {"error", "deployment_error", "cleanup_error"}.intersection(record)
