"""A superseded smoke result must neither publish readiness nor remove its replacement."""

from types import SimpleNamespace

from flash.server.routes import serving_completion as completion


def test_superseded_verification_does_not_undeploy_the_new_generation(monkeypatch):
    deployment = {"checkpoint_id": "run-1/final", "generation": 1, "state": "queued"}
    state = SimpleNamespace(deployments={"run-1/final": deployment})
    monkeypatch.setattr(completion.JobSpec, "from_dict", lambda _: SimpleNamespace())
    monkeypatch.setattr(completion.run_status, "get_status", lambda _: state)
    monkeypatch.setattr(completion, "_assert_deployment_fence", lambda *a: None)
    monkeypatch.setattr(
        completion.runner_transitions,
        "mark_deployment_pending",
        lambda run_id, current, **k: SimpleNamespace(deployments={"run-1/final": current}),
    )
    monkeypatch.setattr(
        completion.deployment_state, "_report_persisted_transition", lambda *a, **k: None
    )
    monkeypatch.setattr(
        completion.serving_smoke, "_run_deployment_smoke", lambda *a, **k: {"verified_at": 1.0}
    )

    def deploy(**kwargs):
        kwargs["before_ready"](
            completion.serving_deploy.DeploymentReadinessFacts(
                "run-1/final", "run-1/final", frozenset(), False
            )
        )
        raise completion.serving_deploy.DeploymentGenerationChanged("superseded")

    monkeypatch.setattr(completion.serving_deploy, "deploy_adapter", deploy)
    failures = []
    monkeypatch.setattr(completion, "_record_deployment_failure", lambda *a: failures.append(a))

    def forbidden(*a, **k):
        raise AssertionError(
            "superseded attempt must not publish readiness or delete its replacement"
        )

    monkeypatch.setattr(completion, "_cleanup_failed_smoke", forbidden)
    monkeypatch.setattr(completion, "_commit_ready_deployment", forbidden)
    completion._finish_deployment_unlocked(
        run_id="run-1",
        spec_dict={},
        is_checkpoint=False,
        deploy_kwargs={"org_id": "org-1"},
        deployment=deployment,
        prev_state="succeeded",
    )
    assert len(failures) == 1
