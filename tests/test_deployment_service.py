"""The deployment service's command contract, driven entirely through fake collaborators.

These prove the decisions the service makes -- lock ownership, CAS guards, reporting gates, dry-run
inertness, error translation -- without a serving backend, a store, or FastAPI.
"""

from __future__ import annotations

import pytest

from flash.serve.deploy import ServingError
from flash.server.domain.deployment_ports import (
    DeploymentConflict,
    DeploymentNotFound,
    DeploymentUnavailable,
    DeploymentUpstreamError,
    InvalidDeploymentRequest,
)
from flash.server.domain.deployments import (
    ChatCommand,
    DeployCommand,
    ExportCommand,
)
from flash.server.platform.deployment_jobs import DeploymentJobStartError
from tests._helpers.deployment import CALLER, FakeStatus, build_harness

_SHA = "a" * 40
_RUN = "flash-1-abcd"

_SPEC = {
    "run_id": _RUN,
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "project": "11111111-1111-1111-1111-111111111111",
    "train": {"hf_repo": "owner/runs", "lora_rank": 32},
    "environment": {"id": "ns/proj/env"},
}


def _status(**overrides) -> FakeStatus:
    fields = {"run_id": _RUN, "state": "done", "spec": dict(_SPEC), "deployment": None}
    fields.update(overrides)
    return FakeStatus(**fields)


def _deploy(harness, **payload):
    return harness.service.deploy(DeployCommand(run_id=_RUN, caller=CALLER, payload=dict(payload)))


# ---- lock ownership --------------------------------------------------------------------


def test_a_contended_deploy_conflicts_without_releasing_a_lock_it_never_took():
    """The 409 path never acquired the lock, so it must not release one."""
    harness = build_harness(status=_status(), acquirable=False)

    with pytest.raises(DeploymentConflict, match="another operation is in progress"):
        _deploy(harness)

    # exactly one attempt, and no release: releasing a lock this request never held would hand
    # the mutex to a caller that is not the owner.
    assert harness.lock_for(_RUN).events == ["acquire-failed"]
    assert harness.lock_for(_RUN).held is False


def test_a_contended_deploy_names_the_busy_state_when_the_record_shows_one():
    harness = build_harness(status=_status(deployment={"state": "smoke_testing"}), acquirable=False)

    with pytest.raises(DeploymentConflict, match="already has a deployment in smoke_testing"):
        _deploy(harness)


def test_a_started_job_takes_lock_ownership_from_the_request():
    """The request must NOT release: the job holds the lock until the lifecycle ends."""
    harness = build_harness(status=_status())

    _deploy(harness)

    lock = harness.lock_for(_RUN)
    assert lock.events == ["acquire"]
    assert lock.held is True
    # the job was handed the very lock the request took, so it can release it later.
    assert harness.jobs.started[0]["deploy_lock"] is lock


def test_a_job_that_never_started_returns_lock_ownership_to_the_request():
    harness = build_harness(
        status=_status(), start_error=DeploymentJobStartError("deployment jobs are shutting down")
    )

    with pytest.raises(DeploymentUnavailable) as excinfo:
        _deploy(harness)

    assert excinfo.value.detail["code"] == "deployment_job_unavailable"
    assert excinfo.value.detail["retryable"] is True
    lock = harness.lock_for(_RUN)
    assert lock.events == ["acquire", "release"]
    assert lock.held is False


def test_a_start_that_raises_after_launching_the_thread_leaves_the_lock_with_the_job():
    """Ownership must move BEFORE `start` is called, not after it returns.

    `start` is not atomic: it can launch the background thread and then fail on its way out
    (the live-set bookkeeping after the thread is running). The job is then alive and WILL
    release the lock when the lifecycle ends. Setting ownership from `start`'s return value
    instead of before the call leaves it False on that path, so the request releases too --
    two releases of one mutex, and the next deploy for this run acquires a lock the previous
    lifecycle still thinks it holds.

    Only a non-DeploymentJobStartError proves the ordering: the declared start error is the
    contract for "the job never ran", and correctly returns ownership to the request.
    """
    harness = build_harness(status=_status())

    def start_then_raise(target, *args, **kwargs):
        del target, args, kwargs
        raise RuntimeError("live-set bookkeeping failed after the thread was already running")

    harness.jobs.start = start_then_raise

    with pytest.raises(RuntimeError, match="live-set bookkeeping failed"):
        _deploy(harness)

    lock = harness.lock_for(_RUN)
    assert lock.events == ["acquire"]
    assert lock.held is True


def test_an_unexpected_failure_after_job_start_does_not_double_release_the_lock():
    """Ownership moves BEFORE the response is built, so a later error cannot steal it back.

    Releasing a lock the job is about to release would corrupt the mutex for every later deploy.
    """
    harness = build_harness(status=_status())

    def explode(*_args, **_kwargs):
        raise RuntimeError("status read blew up after the job started")

    original_start = harness.jobs.start

    def start_then_break(target, *args, **kwargs):
        original_start(target, *args, **kwargs)
        harness.runs.get_status = explode
        return True  # claim a synchronous run so the response path reads status

    harness.jobs.start = start_then_break

    with pytest.raises(RuntimeError, match="status read blew up"):
        _deploy(harness)

    lock = harness.lock_for(_RUN)
    assert lock.events == ["acquire"]
    assert lock.held is True


def test_a_failed_job_start_is_recorded_before_the_error_is_raised():
    harness = build_harness(status=_status(), start_error=DeploymentJobStartError("shutting down"))

    with pytest.raises(DeploymentUnavailable):
        _deploy(harness)

    assert "mark_failed" in harness.deployments.kinds()
    failed = harness.deployments.first_write("mark_failed")
    assert failed["state"] == "failed"
    assert failed["retryable"] is True
    assert "deployment job could not start" in failed["error"]


# ---- CAS and attempt ownership ---------------------------------------------------------


def test_the_queue_write_carries_the_run_state_it_read_as_its_cas_guard():
    harness = build_harness(status=_status(state="done"))

    _deploy(harness)

    args = dict(harness.deployments.first_write("mark_pending_args"))
    assert args["expect_state"] == "done"


def test_a_run_that_moves_during_deploy_conflicts_instead_of_queueing():
    harness = build_harness(status=_status())
    harness.deployments.results["mark_pending"] = FakeStatus(
        run_id=_RUN, state="cancelled", deployment={"state": "undeployed"}
    )

    with pytest.raises(DeploymentConflict, match="became 'cancelled' during deploy"):
        _deploy(harness)


def test_the_queued_record_carries_the_verification_generation_it_fenced_on():
    harness = build_harness(status=_status())
    harness.deployments.generation = 7

    _deploy(harness)

    queued = harness.jobs.started[0]["deployment"]
    assert queued["verification_generation"] == 7


def test_a_busy_recent_attempt_blocks_a_new_deploy():
    import time

    harness = build_harness(
        status=_status(deployment={"state": "queued", "updated_at": time.time()})
    )

    with pytest.raises(DeploymentConflict, match="already has a deployment in queued"):
        _deploy(harness)


def test_a_stale_busy_attempt_can_be_taken_over():
    import time

    harness = build_harness(
        status=_status(deployment={"state": "queued", "updated_at": time.time() - 10_000})
    )

    _deploy(harness)

    assert harness.jobs.started, "a stale attempt must be replaceable"


# ---- dry run ---------------------------------------------------------------------------


def test_a_dry_run_asks_the_gateway_but_persists_reports_and_starts_nothing():
    harness = build_harness(status=_status())

    result = _deploy(harness, dry_run=True)

    assert result["state"] == "dry_run"
    assert harness.serving.calls == ["deploy_adapter"]
    assert harness.serving.deploy_kwargs[0]["dry_run"] is True
    assert harness.deployments.writes == []
    assert harness.reporter.transitions == []
    assert harness.jobs.started == []
    # and the lock it took is given back
    assert harness.lock_for(_RUN).held is False


def test_a_dry_run_rejects_an_invalid_spec_as_an_invalid_request():
    harness = build_harness(status=_status())
    harness.serving.deploy_error = ValueError("lora rank 999 is not servable")

    with pytest.raises(InvalidDeploymentRequest, match="not servable"):
        _deploy(harness, dry_run=True)


# ---- validation ------------------------------------------------------------------------


def test_verify_false_is_refused_before_anything_is_queued():
    harness = build_harness(status=_status())

    with pytest.raises(InvalidDeploymentRequest, match="verify=false is not supported"):
        _deploy(harness, verify=False)

    assert harness.deployments.writes == []
    assert harness.serving.calls == []


def test_a_non_boolean_flag_is_refused():
    harness = build_harness(status=_status())

    with pytest.raises(InvalidDeploymentRequest, match="dry_run must be a boolean"):
        _deploy(harness, dry_run="true")


def test_a_run_without_an_hf_repo_cannot_be_deployed():
    spec = dict(_SPEC)
    spec["train"] = {"lora_rank": 32}
    harness = build_harness(status=_status(spec=spec))

    with pytest.raises(DeploymentConflict, match=r"has no \[train\].hf_repo"):
        _deploy(harness)


def test_an_unfinished_run_cannot_deploy_its_final_adapter():
    harness = build_harness(status=_status(state="running"))

    with pytest.raises(DeploymentConflict, match="only finished runs"):
        _deploy(harness)


def test_a_missing_checkpoint_step_is_not_found():
    harness = build_harness(status=_status(), checkpoints=[{"step": 20}])

    with pytest.raises(DeploymentNotFound, match="no deployable checkpoint at step 40"):
        _deploy(harness, step=40)


def test_an_alias_reconciliation_failure_is_a_structured_upstream_error():
    harness = build_harness(
        status=_status(deployment={"state": "reconciling", "activation_outcome_unknown": True})
    )
    harness.serving.alias_target = "someone-elses-run@final." + _SHA

    with pytest.raises(DeploymentUpstreamError) as excinfo:
        _deploy(harness)

    assert excinfo.value.detail["code"] == "alias_reconciliation_failed"
    assert excinfo.value.detail["run_id"] == _RUN
    assert excinfo.value.detail["retryable"] is True


# ---- reporting -------------------------------------------------------------------------


def test_the_queue_transition_is_reported_only_after_the_write_landed():
    harness = build_harness(status=_status())

    _deploy(harness)

    assert harness.reporter.transitions, "a landed queue write must be reported"
    assert all(persisted for _prev, _cur, persisted in harness.reporter.transitions)


# ---- undeploy --------------------------------------------------------------------------


def test_undeploy_marks_undeployed_and_returns_the_revocation_counts():
    harness = build_harness(status=_status(deployment={"state": "deployed"}))

    result = harness.service.undeploy(_RUN, CALLER)

    assert harness.serving.calls == ["undeploy_adapter"]
    assert "mark_undeployed" in harness.deployments.kinds()
    assert result["disabled_aliases"] == 1
    assert result["disabled_revisions"] == 2
    assert result["serving_deregistered"] is True
    assert harness.lock_for(_RUN).held is False


def test_a_revocation_failure_is_recorded_and_reported_as_a_structured_upstream_error():
    harness = build_harness(status=_status(deployment={"state": "deployed"}))
    harness.serving.undeploy_error = ServingError("serving refused the revocation")

    with pytest.raises(DeploymentUpstreamError) as excinfo:
        harness.service.undeploy(_RUN, CALLER)

    assert excinfo.value.detail["code"] == "deployment_revocation_failed"
    assert "mark_revocation_failed" in harness.deployments.kinds()
    # the lock is released even though the operation failed
    assert harness.lock_for(_RUN).held is False


# ---- export ----------------------------------------------------------------------------


def test_export_requires_a_repository_and_a_token():
    harness = build_harness(status=_status())

    with pytest.raises(InvalidDeploymentRequest, match="repository is required"):
        harness.service.export(ExportCommand(run_id=_RUN, caller=CALLER, payload={}))
    with pytest.raises(InvalidDeploymentRequest, match="hf_token is required"):
        harness.service.export(
            ExportCommand(run_id=_RUN, caller=CALLER, payload={"repository": "owner/name"})
        )


def test_export_rejects_a_repository_that_is_not_owner_slash_name():
    harness = build_harness(status=_status())

    with pytest.raises(InvalidDeploymentRequest, match="form 'owner/name'"):
        harness.service.export(
            ExportCommand(
                run_id=_RUN, caller=CALLER, payload={"repository": "just-a-name", "hf_token": "t"}
            )
        )


def test_export_returns_the_copied_location_and_releases_the_lock():
    harness = build_harness(status=_status())

    result = harness.service.export(
        ExportCommand(
            run_id=_RUN, caller=CALLER, payload={"repository": "owner/name", "hf_token": "tok"}
        )
    )

    assert result.to_dict() == {
        "run_id": _RUN,
        "adapter_id": _RUN,
        "repository": "owner/name",
        "url": "https://hf/x",
        "source": _RUN,
    }
    assert harness.artifacts.exports[0]["dest_repo"] == "owner/name"
    assert harness.lock_for(_RUN).held is False


def test_a_checkpoint_export_names_the_step_in_its_result():
    harness = build_harness(status=_status(), checkpoints=[{"step": 20}])

    result = harness.service.export(
        ExportCommand(
            run_id=_RUN,
            caller=CALLER,
            payload={"repository": "owner/name", "hf_token": "tok", "step": 20},
        )
    )

    assert result.to_dict()["step"] == 20
    assert result.to_dict()["source"] == f"{_RUN}/step-20"


# ---- chat ------------------------------------------------------------------------------


def _ready_deployment(revision: str) -> dict:
    return {"state": "deployed", "adapter_revision": revision, "run_id": _RUN}


def test_chat_authorization_comes_from_the_verified_ledger_not_the_status_record():
    """A record can claim ready for a revision the ledger never vouched for."""
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = []  # ledger vouches for nothing

    with pytest.raises(DeploymentConflict, match="has no active deployment"):
        harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload={}))


def test_chat_serves_the_bare_alias_when_the_ledger_vouches_for_the_ready_revision():
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    plan = harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload={}))

    assert plan.serving_model == _RUN
    assert plan.max_tokens == 512
    assert plan.temperature == 0.0
    assert plan.stream is False


def test_chat_pins_an_explicitly_named_verified_revision():
    revision = f"{_RUN}@step-20.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    plan = harness.service.plan_chat(
        ChatCommand(run_id=_RUN, caller=CALLER, payload={"adapter_revision": revision})
    )

    assert plan.serving_model == revision


def test_chat_refuses_a_revision_that_never_passed_a_smoke():
    revision = f"{_RUN}@step-20.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = []

    with pytest.raises(DeploymentConflict, match="has not passed a successful deployment smoke"):
        harness.service.plan_chat(
            ChatCommand(run_id=_RUN, caller=CALLER, payload={"adapter_revision": revision})
        )


def test_chat_refuses_a_revision_that_belongs_to_another_run():
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    with pytest.raises(InvalidDeploymentRequest, match="belongs to run other-run"):
        harness.service.plan_chat(
            ChatCommand(
                run_id=_RUN,
                caller=CALLER,
                payload={"adapter_revision": f"other-run@final.{_SHA}"},
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": float("inf")},
        {"temperature": float("nan")},
        {"messages": "not-a-list"},
    ],
)
def test_chat_refuses_malformed_sampling_and_messages(payload):
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    with pytest.raises(InvalidDeploymentRequest):
        harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload=payload))


def test_chat_refuses_both_a_revision_and_a_step():
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    with pytest.raises(InvalidDeploymentRequest, match="not both"):
        harness.service.plan_chat(
            ChatCommand(
                run_id=_RUN, caller=CALLER, payload={"adapter_revision": revision, "step": 20}
            )
        )


def test_a_cancelled_run_with_no_deployment_is_told_to_deploy_a_checkpoint():
    harness = build_harness(status=_status(state="cancelled"))

    with pytest.raises(DeploymentConflict, match="was cancelled; deploy a checkpoint"):
        harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload={}))


def test_a_busy_deployment_tells_the_caller_to_check_progress():
    harness = build_harness(status=_status(deployment={"state": "queued"}))

    with pytest.raises(DeploymentConflict, match="deployment is queued"):
        harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload={}))


def test_a_failed_deployment_surfaces_its_error():
    harness = build_harness(
        status=_status(deployment={"state": "failed", "error": "smoke never answered"})
    )

    with pytest.raises(DeploymentConflict, match="smoke never answered"):
        harness.service.plan_chat(ChatCommand(run_id=_RUN, caller=CALLER, payload={}))


def test_chat_passes_the_resolved_plan_through_to_serving():
    revision = f"{_RUN}@final.{_SHA}"
    harness = build_harness(status=_status(deployment=_ready_deployment(revision)))
    harness.deployments.revisions = [revision]

    plan = harness.service.plan_chat(
        ChatCommand(
            run_id=_RUN,
            caller=CALLER,
            payload={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        )
    )
    harness.service.chat(plan)

    call = harness.serving.chat_calls[0]
    assert call["run_id"] == _RUN
    assert call["max_tokens"] == 8
    assert call["messages"] == [{"role": "user", "content": "hi"}]


# ---- listing ---------------------------------------------------------------------------


def test_the_listing_omits_undeployed_and_dry_run_records():
    harness = build_harness(status=_status())
    harness.runs.rows = [{"run_id": _RUN}]
    harness.runs.status = _status(deployment={"state": "undeployed"})

    assert harness.service.list_deployments(CALLER, scope=None) == []

    harness.runs.status = _status(deployment={"state": "deployed"})
    assert len(harness.service.list_deployments(CALLER, scope=None)) == 1


def test_the_listing_redacts_internal_fields_from_each_record():
    harness = build_harness(status=_status())
    harness.runs.rows = [{"run_id": _RUN}]
    harness.runs.status = _status(
        deployment={
            "state": "deployed",
            "run_id": _RUN,
            "url": "https://stale.example",
            "verification_generation": 3,
            "previous_deployment": {"state": "ready"},
        }
    )

    listed = harness.service.list_deployments(CALLER, scope=None)[0]["deployment"]

    assert "previous_deployment" not in listed
    assert "verification_generation" not in listed
    assert "url" not in listed


def test_the_read_endpoint_reports_undeployed_for_a_run_with_no_record():
    harness = build_harness(status=_status(deployment=None))

    assert harness.service.get_deployment(_RUN, CALLER)["state"] == "undeployed"
