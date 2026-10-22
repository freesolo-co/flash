"""Incomplete hosted registrations are revoked without disturbing verified siblings."""

from queue import Queue
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
from tests.test_server_api import _bearer, _identity_for_token, _make_run, _new_user_token
from tests.test_server_api import api as api


def _deploy(api, key, run_id, checkpoint_id):
    response = api.post(
        f"/v1/runs/{run_id}/deploy", json={"checkpoint_id": checkpoint_id}, headers=_bearer(key)
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert "registration_identity" not in result
    assert "serving_org_id" not in result
    return result


def test_retired_ready_deployments_are_revoked_and_failed_cleanup_is_retried(api, monkeypatch):
    key = _new_user_token()
    retired, supported = [_make_run(api, key, "done") for _ in range(2)]
    for run_id in (retired, supported):
        seed_ready_checkpoint(run_id, f"{run_id}/final")
    current = status.get_status(retired)
    current.spec = {**current.spec, "algorithm": "opsd"}
    state._save_status(current)
    sibling = status.get_status(supported).deployments
    monkeypatch.setattr(
        serving_completion.db, "all_runs", lambda: [{"run_id": r} for r in (retired, supported)]
    )
    monkeypatch.setattr(deployment_state, "_enqueue_deployment_report", lambda _: None)
    calls = []

    def undeploy(target, *, org_id, timeout_s):
        assert target == f"{retired}/final"
        assert org_id == run_serving_org_id(current)
        assert timeout_s == 5.0
        assert status.get_status(retired).deployments[target]["state"] == "undeploying"
        calls.append(target)
        if len(calls) == 1:
            raise ServingError("temporary serving outage")
        return {"disabled_checkpoints": [target]}

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    assert serving_completion.recover_deployments() == 1
    assert not calls
    assert serving_completion.cleanup_pending_deployments(Event())
    pending = status.get_status(retired).deployments[f"{retired}/final"]
    assert pending["state"] == "undeploying"
    assert "temporary serving outage" in pending["error"]
    assert serving_completion.recover_deployments() == 1
    assert not serving_completion.cleanup_pending_deployments(Event())
    assert status.get_status(retired).deployments[f"{retired}/final"]["state"] == "undeployed"
    assert status.get_status(supported).deployments == sibling
    assert len(calls) == 2


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("registration_started", [False, True])
def test_failure_before_smoke_cleans_only_attempted_registration(
    api, monkeypatch, cleanup_fails, registration_started
):
    key = _new_user_token()
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
    result = _deploy(api, key, run_id, checkpoint_id)
    pending = registration_started and cleanup_fails
    assert result["state"] == ("undeploying" if pending else "failed")
    assert undeploys == (
        [(checkpoint_id, run_serving_org_id(before))] if registration_started else []
    )
    if pending:
        assert result["retryable"] is True
        assert "cleanup unavailable" in result["error"]
    assert status.get_status(run_id).deployments[sibling] == before.deployments[sibling]


@pytest.mark.parametrize("initial_state", ["deploying", "undeploying"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_recovery_fences_and_removes_unfinished_registration(
    api, monkeypatch, initial_state, cleanup_fails
):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    checkpoint_id, sibling = f"{run_id}/step-20", f"{run_id}/final"
    seed_ready_checkpoint(run_id, sibling)
    current = status.get_status(run_id)
    attempt = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": 20,
        "generation": 1,
        "state": initial_state,
        "registration_identity": {},
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
        assert fenced["state"] == "undeploying"
        assert fenced["generation"] >= attempt["generation"] + (initial_state != "undeploying")
        late = {
            **attempt,
            "state": "ready",
            "verified_at": 123.0,
            "deployment_generation": "generation-a",
        }
        assert transitions.mark_checkpoint_deployed(run_id, late).deployments[target] == fenced
        if cleanup_fails:
            raise ServingError("cleanup unavailable")
        return {"disabled_checkpoints": [target]}

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    assert serving_completion.recover_deployments() == 1
    assert undeploys == []
    serving_completion.cleanup_pending_deployments(Event())
    saved = status.get_status(run_id)
    assert saved.deployments[sibling] == current.deployments[sibling]
    assert saved.deployments[checkpoint_id]["state"] == (
        "undeploying" if cleanup_fails else "undeployed"
    )
    assert undeploys == [(checkpoint_id, run_serving_org_id(current))]
    assert reports[-1].deployments == saved.deployments
    if cleanup_fails:
        assert saved.deployments[checkpoint_id]["retryable"] is True
        cleanup_fails = False
        assert serving_completion.recover_deployments() == 1
        serving_completion.cleanup_pending_deployments(Event())
        assert status.get_status(run_id).deployments[checkpoint_id]["state"] == "undeployed"
    assert serving_completion.recover_deployments() == 0


def test_recovery_without_org_retains_cleanup_pending(api, monkeypatch):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    checkpoint_id = f"{run_id}/final"
    transitions.mark_deployment_pending(
        run_id,
        {
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": None,
            "generation": 1,
            "state": "deploying",
            "registration_identity": {},
        },
    )
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": run_id}])
    monkeypatch.setattr(serving_completion, "run_serving_org_id", lambda _status: "")
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda *_args, **_kwargs: pytest.fail("unscoped cleanup")
    )
    assert serving_completion.recover_deployments() == 1
    serving_completion.cleanup_pending_deployments(Event())
    record = status.get_status(run_id).deployments[checkpoint_id]
    assert record["state"] == "undeploying"
    assert "no organization scope" in record["error"]


@pytest.mark.parametrize("error_type", [RegistrationRejected, ServingError])
@pytest.mark.parametrize("status_code", [401, 403, 409, 422])
def test_only_registration_rejection_skips_cleanup(api, monkeypatch, error_type, status_code):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    deleted = []

    def fail(**kwargs):
        kwargs["before_register"]({})
        raise error_type("rejected", status_code=status_code)

    monkeypatch.setattr(deploy, "deploy_adapter", fail)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **_: deleted.append(target))
    result = _deploy(api, key, run_id, f"{run_id}/final")
    assert result["state"] == "failed"
    assert deleted == ([] if error_type is RegistrationRejected else [f"{run_id}/final"])


def test_recovery_uses_persisted_registration_org(api, monkeypatch):
    key = _new_user_token()
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
    _deploy(api, key, run_id, f"{run_id}/final")
    record = status.get_status(run_id).deployments[f"{run_id}/final"]
    assert record["serving_org_id"] == _identity_for_token(key)["org_id"]
    monkeypatch.setattr(serving_completion, "run_serving_org_id", lambda _: "wrong-current-org")
    deleted = []
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **kw: deleted.append((target, kw))
    )
    serving_completion.recover_deployments()
    serving_completion.cleanup_pending_deployments(Event())
    assert deleted == [(f"{run_id}/final", {"org_id": record["serving_org_id"], "timeout_s": 5.0})]


def test_recovery_worker_respects_stop(api, monkeypatch):
    monkeypatch.setattr(serving_completion.db, "all_runs", lambda: [{"run_id": "missing"}])
    monkeypatch.setattr(
        serving_completion.run_status, "get_status", lambda _: pytest.fail("read after shutdown")
    )
    stop = Event()
    stop.set()
    assert serving_completion.cleanup_pending_deployments(stop) is False


@pytest.fixture
def blocked_recovery(monkeypatch):
    started, release = Event(), Event()

    def cleanup(stop):
        started.set()
        while not release.is_set() and not stop.wait(0.01):
            pass

    monkeypatch.setattr(serving_completion, "recover_deployments", lambda: 1)
    monkeypatch.setattr(serving_completion, "run_deployment_cleanup_worker", cleanup)
    try:
        yield started
    finally:
        release.set()


def test_startup_serves_requests_while_remote_cleanup_is_blocked(blocked_recovery, api):
    assert blocked_recovery.wait(2)
    assert api.get("/v1/me", headers=_bearer(_new_user_token())).status_code == 200


@pytest.fixture
def controlled_cleanup(monkeypatch):
    retry = serving_completion.run_deployment_cleanup_worker
    waits, wake, finished = Queue(), Event(), Event()

    def worker(stop):
        stop_wait = stop.wait

        def wait(delay):
            waits.put(delay)
            while not stop_wait(0.01):
                if wake.is_set():
                    wake.clear()
                    return False
            return True

        monkeypatch.setattr(stop, "wait", wait)
        try:
            retry(stop)
        finally:
            finished.set()

    monkeypatch.setattr(serving_completion, "run_deployment_cleanup_worker", worker)
    yield waits, wake
    assert finished.wait(2), "cleanup worker did not stop with the API"


def test_cleanup_worker_handles_live_failures_after_idle(controlled_cleanup, api, monkeypatch):
    waits, wake = controlled_cleanup
    assert waits.get(timeout=2) == 30.0  # Clean startup still owns an idle worker.
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    offline = True
    deleted = []

    def register(**kwargs):
        kwargs["before_register"]({})
        raise ServingError("registration response lost")

    def undeploy(checkpoint_id, **kwargs):
        deleted.append(checkpoint_id)
        if offline:
            raise ServingError("backend offline")

    monkeypatch.setattr(deploy, "deploy_adapter", register)
    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    for _ in range(2):  # A drained queue must not end the worker's lifetime.
        offline = True
        assert _deploy(api, key, run_id, target)["state"] == "undeploying"
        wake.set()
        assert waits.get(timeout=2) == 1.0
        assert status.get_status(run_id).deployments[target]["state"] == "undeploying"
        offline = False
        wake.set()
        assert waits.get(timeout=2) == 30.0
        assert status.get_status(run_id).deployments[target]["state"] == "undeployed"
    assert deleted == [target] * 6


@pytest.mark.parametrize("status_code", [None, 408, 503])
def test_ambiguous_registration_preserves_mismatched_binding(
    api, monkeypatch, tmp_path, status_code
):
    key = _new_user_token()
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
        assert persisted["registration_identity"] == kwargs["json"]
        existing.update({**kwargs["json"], "artifact_digest": "another-artifact"})
        raise ServingError("registration response lost", status_code=status_code)

    monkeypatch.setattr(deploy, "fetch_registered_checkpoint", registered)
    monkeypatch.setattr(deploy.transport, "serving_request", ambiguous)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **_: deletes.append(target))
    result = _deploy(api, key, run_id, checkpoint_id)
    assert result["state"] == "failed"
    assert reads == [checkpoint_id, checkpoint_id]
    assert deletes == []
    assert "different immutable identity" in result["error"]
    assert status.get_status(run_id).deployments[sibling] == before


@pytest.mark.parametrize("interruption", ["failure", "locked"])
def test_cleanup_worker_retries_without_another_restart(api, monkeypatch, interruption):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    targets = [f"{run_id}/final", f"{run_id}/step-20"]
    for target in targets:
        seed_ready_checkpoint(run_id, target)
        transitions.mark_deployment_undeploying(run_id, "interrupted", target)
    deleted, waits = [], []
    stop = Event()
    lock = serving_completion.deployment_locks._deploy_lock(run_id)
    if interruption == "locked":
        lock.acquire()

    def undeploy(target, **kwargs):
        assert 0 < kwargs["timeout_s"] <= 5
        deleted.append(target)
        if len(deleted) == 1 and interruption == "failure":
            raise ServingError("temporarily unavailable")

    def wait(delay):
        waits.append(delay)
        if len(waits) == 2:
            assert delay == 30.0, "cleanup did not finish on the next pass"
            return True
        if interruption == "locked":
            lock.release()
        return False

    monkeypatch.setattr(deploy, "undeploy_adapter", undeploy)
    monkeypatch.setattr(stop, "wait", wait)
    try:
        serving_completion.run_deployment_cleanup_worker(stop)
    finally:
        if interruption == "locked" and not waits:
            lock.release()
    assert waits == [1.0, 30.0]
    assert len(deleted) == (3 if interruption == "failure" else 2)
    assert all(status.get_status(run_id).deployments[t]["state"] == "undeployed" for t in targets)


def test_cleanup_retry_rechecks_state_after_backoff(api, monkeypatch):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_undeploying(run_id, "interrupted", target)
    stop, deleted = Event(), []

    def unavailable(checkpoint_id, **kwargs):
        deleted.append(checkpoint_id)
        raise ServingError("offline")

    def wait(delay):
        if delay == 30.0:
            return True
        seed_ready_checkpoint(run_id, target)
        return False

    monkeypatch.setattr(deploy, "undeploy_adapter", unavailable)
    monkeypatch.setattr(stop, "wait", wait)
    serving_completion.run_deployment_cleanup_worker(stop)
    assert deleted == [target]
    assert status.get_status(run_id).deployments[target]["state"] == "ready"


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
    key = _new_user_token()
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
        assert "registration_identity" not in status.get_status(run_id).deployments[target]
        if stage != "before_registration":
            identity["org_id"] = kwargs["org_id"]
            kwargs["before_register"](identity)
        snapshot.append(status.get_status(run_id))
        raise ServingError("interrupted")

    # Restore exactly what was durable at the interruption, before normal error handling ran.
    with monkeypatch.context() as patch:
        patch.setattr(deploy, "deploy_adapter", interrupted)
        patch.setattr(serving_completion, "_cleanup_failed_deployment", lambda *a, **kw: True)
        _deploy(api, key, run_id, target)
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

    monkeypatch.setattr(deploy, "fetch_registered_checkpoint", registered)
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda target, **kw: deletes.append(target))
    assert serving_completion.recover_deployments() == 1
    serving_completion.cleanup_pending_deployments(Event())
    recovered = status.get_status(run_id)
    if stage == "before_registration":
        assert reads == deletes == []
        assert recovered.deployments[target]["state"] == "failed"
    else:
        assert snapshot[0].deployments[target]["registration_identity"] == identity
        assert reads == [(identity["org_id"], target)]
        assert deletes == ([target] if stage == "matching_binding" else [])
        assert recovered.deployments[target]["state"] == (
            "undeploying"
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
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    original = transitions.mark_deployment_pending

    def save(run_id, record, **kwargs):
        if "registration_identity" in record:
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
    result = _deploy(api, key, run_id, target)
    assert result["state"] == "failed"
    assert "cannot persist registration intent" in result["error"]
    assert "registration_identity" not in status.get_status(run_id).deployments[target]


@pytest.mark.parametrize("multiple_runs", [False, True])
def test_cleanup_reaches_later_checkpoints_despite_repeated_timeouts(
    api, monkeypatch, multiple_runs
):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    targets = []
    for i in range(7):
        if multiple_runs and i:
            run_id = _make_run(api, key, "done")
        target = f"{run_id}/step-{i + 1}"
        targets.append((run_id, target))
        seed_ready_checkpoint(run_id, target)
        transitions.mark_deployment_undeploying(run_id, "original failure", target)
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
    serving_completion.run_deployment_cleanup_worker(stop)
    assert status.get_status(last_run).deployments[last_target]["state"] == "undeployed"
    assert waits == [1.0]


@pytest.mark.parametrize("legacy_composite", [False, True])
def test_cleanup_retry_keeps_original_cause_and_only_latest_failure(
    api, monkeypatch, legacy_composite
):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    target = f"{run_id}/final"
    seed_ready_checkpoint(run_id, target)
    transitions.mark_deployment_undeploying(run_id, "readiness timed out", target)
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
    serving_completion.run_deployment_cleanup_worker(stop)
    assert waits == [1, 2, 4, 8, 16] + [30] * 45
    assert len(attempts) == 50
    monkeypatch.setattr(deploy, "undeploy_adapter", lambda *a, **kw: None)
    serving_completion.cleanup_pending_deployments(Event())
    record = status.get_status(run_id).deployments[target]
    assert record["state"] == "undeployed"
    assert not {"error", "deployment_error", "cleanup_error"}.intersection(record)
