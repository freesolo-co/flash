"""Checkpoint cancellation remains independent of the retired training providers."""

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.deploy as runner_deploy
import flash.runner.supervise.recovery as runner_recovery
import flash.runner.supervise.transitions as runner_transitions
import flash.serve.contract.errors as serving_errors
from tests._helpers import deployments as deployment_fixtures
from tests._helpers.deployments import only_deployment, records


def _run_spec(run_id: str):
    from flash.core.spec import JobSpec

    return JobSpec.from_dict({"gpu": {"type": "H100"}, "run_id": run_id})


def _checkpoint_id(run_id: str, step: int, sha: str = "a") -> str:
    del sha
    return f"{run_id}/step-{step}"


def _ready_checkpoint(run_id: str, step: int, *, remote: dict | None = None) -> dict:
    spec = _run_spec(run_id)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )
    deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "checkpoint_id": _checkpoint_id(run_id, step),
        "checkpoint_step": step,
    }
    deployment_fixtures.complete_deployment(run_id, deployment)
    return runner_status.get_status(run_id).deployments[deployment["checkpoint_id"]]


def test_checkpoint_completion_rejects_newer_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-stale"
    previous = _ready_checkpoint(run_id, 40)
    runner_transitions.mark_undeployed(run_id, previous["checkpoint_id"])
    newer = {**previous, "state": "deploying", "generation": 3, "requested_at": 200.0}
    runner_transitions.mark_deployment_pending(run_id, newer)
    out = runner_transitions.mark_checkpoint_deployed(run_id, previous)
    assert out.deployments[previous["checkpoint_id"]] == newer


def test_cancel_revokes_inflight_checkpoint_deployment(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-inflight"
    spec = _run_spec(run_id)
    revision = _checkpoint_id(run_id, 40)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployments=records(
                run_id,
                {
                    "state": "deploying",
                    "checkpoint_id": revision,
                    "checkpoint_step": 40,
                },
            ),
            billing_context={"org_id": "org-a"},
        )
    )
    undeployed = []
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda target, *, org_id: undeployed.append((org_id, target)),
    )

    out = runner_deploy.cancel_run(run_id)

    assert undeployed == [("org-a", revision)]
    assert only_deployment(out)["state"] == "undeployed"
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset()


def test_cancel_backend_success_local_commit_failure_is_not_backend_uncertainty(
    tmp_path, monkeypatch
):
    import flash.serve.deployment.deploy as deploy
    from flash.runner.supervise.deploy import DeploymentStatePersistenceError

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-local-persistence-failure"
    spec = _run_spec(run_id)
    revision = f"{run_id}/final"
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="running",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployments=records(run_id, {"state": "ready", "checkpoint_id": revision}),
        )
    )
    backend_calls = []
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)
    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **_: backend_calls.append(target)
    )
    monkeypatch.setattr(
        runner_transitions,
        "mark_undeployed",
        lambda _run_id, _checkpoint_id=None: (_ for _ in ()).throw(
            OSError("status store unavailable")
        ),
    )

    with pytest.raises(DeploymentStatePersistenceError) as excinfo:
        runner_deploy.cancel_run(run_id)

    assert not isinstance(excinfo.value, runner_deploy.DeploymentCleanupError)
    assert excinfo.value.backend_outcome == "confirmed"
    assert "backend disablement was confirmed" in str(excinfo.value)
    assert backend_calls == [f"{run_id}/final"]
    failed = runner_status.get_status(run_id)
    assert failed.state == "cancelled"
    assert only_deployment(failed)["state"] == "undeploying"
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset()


def test_repeated_cancel_preserves_checkpoint_serving(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-repeat"
    deployment = _ready_checkpoint(run_id, 40)
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("repeated cancel must not revoke the checkpoint"),
    )

    first = runner_deploy.cancel_run(run_id)
    second = runner_deploy.cancel_run(run_id)

    assert first.state == second.state == "cancelled"
    assert only_deployment(first) == only_deployment(second) == deployment
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset({deployment["checkpoint_id"]})


def test_cancel_preserved_checkpoint_keeps_verified_ready_siblings(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "flash-checkpoint-prune"
    preserved = _ready_checkpoint(run_id, 40, remote=None)
    older_revisions = {
        _checkpoint_id(run_id, 20, "b"),
        f"{run_id}/final",
    }
    for revision in older_revisions:
        deployment_fixtures.seed_ready_checkpoint(run_id, revision)
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _target: pytest.fail("the preserved checkpoint must remain serving"),
    )

    out = runner_deploy.cancel_run(run_id)

    assert out.state == "cancelled"
    assert out.deployments[preserved["checkpoint_id"]] == preserved
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset(
        {*older_revisions, preserved["checkpoint_id"]}
    )


def test_cancel_double_undeploy_failure_revokes_authority_and_is_retryable(tmp_path, monkeypatch):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    from flash.core.spec import JobSpec

    run_id = "flash-dep-revoke"
    spec = JobSpec.from_dict({"gpu": {"type": "H100"}, "run_id": run_id})
    revision = f"{run_id}/final"
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=run_id,
            state="deployed",
            spec=spec.to_dict(),
            platform_context={"org_id": "org-1"},
            deployments=records(
                run_id,
                {
                    "state": "ready",
                    "checkpoint_id": revision,
                    "endpoint_name": "https://serve.example",
                },
            ),
        )
    )
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset()

    attempts = []

    def fail_undeploy(target, **_):
        attempts.append(target)
        raise serving_errors.ServingError("backend unavailable")

    monkeypatch.setattr(deploy, "undeploy_adapter", fail_undeploy)
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)

    with pytest.raises(runner_deploy.DeploymentCleanupError) as excinfo:
        runner_deploy.cancel_run(run_id)

    assert excinfo.value.retryable is True
    assert attempts == [revision]
    failed = runner_status.get_status(run_id)
    assert failed.state == "cancelled"
    assert only_deployment(failed)["state"] == "undeploying"
    assert only_deployment(failed)["retryable"] is True
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset()

    monkeypatch.setattr(
        deploy, "undeploy_adapter", lambda target, **_: attempts.append(target) or {}
    )
    retried = runner_deploy.cancel_run(run_id)

    assert attempts == [revision, revision]
    assert retried.state == "cancelled"
    assert only_deployment(retried)["state"] == "undeployed"
    assert deployment_fixtures.ready_checkpoints(run_id) == frozenset()


def test_cancel_preserves_ready_siblings_and_retries_each_unfinished_checkpoint(
    tmp_path, monkeypatch
):
    import flash.serve.deployment.deploy as deploy

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    run_id = "cancel-inventory"
    ready = _ready_checkpoint(run_id, 20)
    status = runner_status.get_status(run_id)
    status.platform_context = {"org_id": "org-a"}
    runner_state._save_status(status)
    for step in (40, 60):
        runner_transitions.mark_deployment_pending(
            run_id,
            {
                "checkpoint_id": f"{run_id}/step-{step}",
                "checkpoint_step": step,
                "state": "deploying",
                "generation": 1,
                "requested_at": 1.0,
            },
        )
    attempts = []

    def revoke(target, **_):
        attempts.append(target)
        if target.endswith("step-40"):
            raise serving_errors.ServingError("backend unavailable")

    monkeypatch.setattr(deploy, "undeploy_adapter", revoke)
    monkeypatch.setattr(runner_recovery, "_gc_run_sandboxes", lambda _spec: None)
    with pytest.raises(runner_deploy.DeploymentCleanupError, match="backend unavailable"):
        runner_deploy.cancel_run(run_id)
    saved = runner_status.get_status(run_id)
    assert saved.deployments[ready["checkpoint_id"]] == ready
    assert saved.deployments[f"{run_id}/step-40"]["state"] == "undeploying"
    assert saved.deployments[f"{run_id}/step-60"]["state"] == "undeployed"
    assert set(attempts) == {f"{run_id}/step-40", f"{run_id}/step-60"}
