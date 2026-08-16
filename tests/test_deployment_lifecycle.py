"""The background half of a deployment attempt: fencing, activation, commit, and recovery.

The ordering these pin is the whole point of the module:
  fence -> smoke -> fence again -> activate -> coordinated ready commit
"""

from __future__ import annotations

import time

import pytest

from flash.serve.deploy import (
    ActivationOutcomeUnknown,
    AdapterConfigMissing,
    AliasThinkingSilent,
    ServingError,
)
from tests._helpers.deployment import FakeStatus, build_harness

_SHA = "a" * 40
_RUN = "flash-1-abcd"
_REVISION = f"{_RUN}@final.{_SHA}"

_SPEC = {
    "run_id": _RUN,
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "project": "11111111-1111-1111-1111-111111111111",
    "train": {"hf_repo": "owner/runs", "lora_rank": 32},
    "environment": {"id": "ns/proj/env"},
}


def _queued(requested_at: float = 1000.0, **extra) -> dict:
    record = {
        "run_id": _RUN,
        "state": "queued",
        "requested_at": requested_at,
        "verification_generation": 1,
        "adapter_hf_prefix": "sft/run/seed0/adapter",
        "openai_model": _RUN,
    }
    record.update(extra)
    return record


def _status(deployment: dict | None, state: str = "done"):
    return FakeStatus(run_id=_RUN, state=state, spec=dict(_SPEC), deployment=deployment)


def _finish(harness, deployment: dict, *, is_checkpoint: bool = False, prev_state: str = "done"):
    harness.service.lifecycle.finish(
        run_id=_RUN,
        spec_dict=dict(_SPEC),
        is_checkpoint=is_checkpoint,
        deploy_kwargs={"run_id": _RUN, "model": _SPEC["model"]},
        deployment=deployment,
        prev_state=prev_state,
    )


def _harness_for_attempt(**kwargs):
    harness = build_harness(**kwargs)
    harness.serving.activation = (_REVISION, _RUN)
    # the smoke is exercised in test_deployment_smoke; here it is a recorded step so the ORDER
    # around it is what gets asserted.
    harness.smoke_calls = []

    def fake_smoke(run_id, spec, *, serving_model, expected_checkpoint):
        harness.smoke_calls.append((run_id, serving_model, expected_checkpoint))
        return {"verify_kind": "fixed_prompt", "thinking_tag": False}

    harness.service.lifecycle.run_smoke = fake_smoke
    return harness


# ---- attempt ownership -----------------------------------------------------------------


def test_a_superseded_attempt_exits_without_mutating_anything():
    """A newer attempt owns the record; this one must not write, report, or call serving."""
    harness = _harness_for_attempt(status=_status(_queued(requested_at=2000.0)))

    _finish(harness, _queued(requested_at=1000.0))

    assert harness.deployments.writes == []
    assert harness.reporter.transitions == []
    assert harness.serving.calls == []


def test_an_attempt_whose_record_left_a_busy_state_exits_without_mutating():
    harness = _harness_for_attempt(status=_status({**_queued(), "state": "undeployed"}))

    _finish(harness, _queued())

    assert harness.deployments.writes == []
    assert harness.serving.calls == []


# ---- ordering --------------------------------------------------------------------------


def test_the_attempt_smokes_before_activation_and_commits_ready_after():
    harness = _harness_for_attempt(status=_status(_queued()))

    _finish(harness, _queued())

    assert harness.smoke_calls == [(_RUN, _REVISION, _RUN)]
    kinds = harness.deployments.kinds()
    # two progress writes (smoke_testing, reconciling) then the ready commit
    assert kinds == ["mark_pending", "mark_pending", "mark_deployed"]
    assert harness.serving.calls == ["deploy_adapter"]


def test_the_activation_fence_is_checked_before_smoke_and_again_before_activation():
    """A cancel can land while smoke is blocked, so one fence is not enough."""
    harness = _harness_for_attempt(status=_status(_queued()))
    checks: list[str] = []
    real_fence = harness.service.lifecycle.assert_activation_fence

    def counting_fence(run_id, deployment, is_checkpoint, prev_state):
        checks.append("fence")
        return real_fence(run_id, deployment, is_checkpoint, prev_state)

    harness.service.lifecycle.assert_activation_fence = counting_fence

    def smoke_between_fences(run_id, spec, *, serving_model, expected_checkpoint):
        checks.append("smoke")
        return {"thinking_tag": False}

    harness.service.lifecycle.run_smoke = smoke_between_fences

    _finish(harness, _queued())

    assert checks == ["fence", "smoke", "fence"]


def test_a_cancel_that_lands_while_smoke_is_blocked_stops_the_activation():
    harness = _harness_for_attempt(status=_status(_queued()))

    def cancel_during_smoke(run_id, spec, *, serving_model, expected_checkpoint):
        # the record is revoked while the smoke is in flight
        harness.runs.status = _status({**_queued(), "state": "undeployed"})
        return {"thinking_tag": False}

    harness.service.lifecycle.run_smoke = cancel_during_smoke

    _finish(harness, _queued())

    assert "mark_deployed" not in harness.deployments.kinds()
    assert "mark_failed" in harness.deployments.kinds()
    failed = harness.deployments.first_write("mark_failed")
    assert "superseded before alias activation" in failed["error"]


def test_a_changed_verification_generation_stops_the_activation():
    harness = _harness_for_attempt(status=_status(_queued()))
    harness.deployments.generation = 99  # the ledger was revoked and re-issued

    _finish(harness, _queued())

    failed = harness.deployments.first_write("mark_failed")
    assert "verification generation changed" in failed["error"]
    assert "mark_deployed" not in harness.deployments.kinds()


def test_a_run_that_changes_state_before_activation_is_fenced_out():
    harness = _harness_for_attempt(status=_status(_queued(), state="cancelled"))

    _finish(harness, _queued(), prev_state="done")

    failed = harness.deployments.first_write("mark_failed")
    assert "run state changed from 'done' to 'cancelled'" in failed["error"]


# ---- the coordinated ready commit ------------------------------------------------------


def test_the_ready_commit_carries_the_verification_generation():
    """Ledger membership and the status write must land together, under the ledger lock."""
    harness = _harness_for_attempt(status=_status(_queued()))

    _finish(harness, _queued())

    args = dict(harness.deployments.first_write("mark_deployed_args"))
    assert args["verification_generation"] == 1
    assert args["expect_state"] == "done"


def test_a_checkpoint_commit_uses_the_checkpoint_write_and_its_own_state_guard():
    harness = _harness_for_attempt(status=_status(_queued()))

    _finish(harness, _queued(), is_checkpoint=True)

    assert "mark_checkpoint_deployed" in harness.deployments.kinds()
    args = dict(harness.deployments.first_write("mark_checkpoint_deployed_args"))
    assert args["verification_generation"] == 1


def test_a_lost_ready_commit_is_reconciled_and_never_clobbers_a_newer_record(capsys):
    """The alias is already live; a lost CAS must be logged, not overwritten."""
    harness = _harness_for_attempt(status=_status(_queued()))
    # the commit does not land, and by reconcile time a newer actor owns the record
    harness.deployments.results["mark_deployed"] = FakeStatus(
        run_id=_RUN, state="done", deployment={"state": "undeployed"}
    )

    _finish(harness, _queued())

    output = capsys.readouterr().out
    assert "deployment_record_diverged" in output
    assert "serving alias left as activated" in output


def test_the_ready_record_is_redacted_before_it_is_persisted():
    harness = _harness_for_attempt(status=_status(_queued()))

    _finish(harness, _queued(previous_deployment={"state": "ready"}))

    committed = harness.deployments.first_write("mark_deployed")
    assert "previous_deployment" not in committed
    assert "verification_generation" not in committed


# ---- failures --------------------------------------------------------------------------


def test_an_ambiguous_activation_is_persisted_as_reconciling_and_flagged_unknown():
    harness = _harness_for_attempt(status=_status(_queued()))
    # the smoke ran and the ACTIVATION is what went ambiguous, so it is raised after the hook
    harness.serving.activation_error = ActivationOutcomeUnknown(
        _RUN, _REVISION, detail="serving did not answer the activation"
    )

    _finish(harness, _queued())

    recorded = harness.deployments.first_write("mark_failed")
    assert recorded["state"] == "reconciling"
    assert recorded["activation_outcome_unknown"] is True


def test_a_failure_before_activation_preserves_the_previous_alias():
    harness = _harness_for_attempt(status=_status(_queued()))
    harness.serving.deploy_error = ServingError("registration refused")

    _finish(harness, _queued())

    failed = harness.deployments.first_write("mark_failed")
    assert failed["state"] == "failed"
    assert "previous working alias was preserved" in failed["detail"]
    assert "alias_activation_confirmed" not in failed


def test_a_missing_final_adapter_names_the_checkpoints_that_could_be_deployed_instead():
    harness = _harness_for_attempt(
        status=_status(_queued()), checkpoints=[{"step": 20}, {"step": 40}]
    )
    harness.serving.deploy_error = AdapterConfigMissing("no adapter_config.json")

    _finish(harness, _queued())

    failed = harness.deployments.first_write("mark_failed")
    assert "flash models deploy flash-1-abcd/step-40" in failed["error"]
    assert "available steps: 20, 40" in failed["error"]


def test_a_post_activation_failure_leaves_the_alias_live_and_records_that_it_activated():
    """Never revert: a revert could clobber a NEWER deployment that already took the alias."""
    harness = _harness_for_attempt(status=_status(_queued()))
    spec_with_thinking = dict(_SPEC)
    spec_with_thinking["train"] = {**_SPEC["train"], "thinking": True}

    harness.service.lifecycle.run_smoke = lambda *a, **k: {"thinking_tag": True}

    def silent_alias(run_id, spec, revision, checkpoint):
        raise AliasThinkingSilent(run_id, revision, detail="alias returned no reasoning_content")

    harness.service.lifecycle.verify_alias_thinking = silent_alias

    harness.service.lifecycle.finish(
        run_id=_RUN,
        spec_dict={**_SPEC, "thinking": True},
        is_checkpoint=False,
        deploy_kwargs={"run_id": _RUN},
        deployment=_queued(),
        prev_state="done",
    )

    failed = harness.deployments.first_write("mark_failed")
    assert failed["alias_activation_confirmed"] is True
    assert failed["alias_thinking_tag"] is False
    assert "mark_deployed" not in harness.deployments.kinds()


# ---- reporting -------------------------------------------------------------------------


def test_nothing_is_reported_for_a_write_that_did_not_land():
    harness = _harness_for_attempt(status=_status(_queued()))
    # the ready commit is lost AND the reconcile retry is lost too
    harness.deployments.results["mark_deployed"] = FakeStatus(
        run_id=_RUN, state="done", deployment={"state": "undeployed"}
    )

    _finish(harness, _queued())

    committed_reports = [
        current for _prev, current, persisted in harness.reporter.transitions if persisted
    ]
    assert all(
        (current.deployment or {}).get("state") != "ready" for current in committed_reports
    ), "a ready state that never persisted must never be reported"


# ---- startup recovery ------------------------------------------------------------------


def test_recovery_fails_a_busy_record_regardless_of_how_recent_it_looks():
    """Holding the flock proves no owner survives, however fresh the timestamp is."""
    harness = build_harness(
        status=_status({**_queued(), "state": "smoke_testing", "updated_at": time.time()})
    )
    harness.runs.rows = [{"run_id": _RUN}]

    assert harness.service.recover_deployments() == 1

    failed = harness.deployments.first_write("mark_failed")
    assert failed["state"] == "failed"
    assert "interrupted by control-plane restart" in failed["error"]


def test_recovery_skips_a_record_whose_lock_another_replica_still_holds():
    harness = build_harness(
        status=_status({**_queued(), "state": "smoke_testing"}), acquirable=False
    )
    harness.runs.rows = [{"run_id": _RUN}]

    assert harness.service.recover_deployments() == 0
    assert harness.deployments.writes == []


def test_recovery_preserves_an_unknown_activation_as_reconciling():
    harness = build_harness(
        status=_status({**_queued(), "state": "reconciling", "activation_outcome_unknown": True})
    )
    harness.runs.rows = [{"run_id": _RUN}]

    assert harness.service.recover_deployments() == 1

    failed = harness.deployments.first_write("mark_failed")
    assert failed["state"] == "reconciling"


def test_recovery_retires_a_ready_record_whose_spec_can_no_longer_be_parsed():
    harness = build_harness(
        status=FakeStatus(
            run_id=_RUN,
            state="deployed",
            spec={"algorithm": "an-algorithm-that-was-removed"},
            deployment={"state": "deployed", "run_id": _RUN},
        )
    )
    harness.runs.rows = [{"run_id": _RUN}]

    assert harness.service.recover_deployments() == 1

    failed = harness.deployments.first_write("mark_failed")
    assert "no longer supported" in failed["error"]


def test_recovery_leaves_a_servable_ready_record_alone():
    harness = build_harness(status=_status({"state": "deployed", "run_id": _RUN}))
    harness.runs.rows = [{"run_id": _RUN}]

    assert harness.service.recover_deployments() == 0
    assert harness.deployments.writes == []


def test_recovery_releases_the_lock_it_took_for_every_row():
    harness = build_harness(status=_status({**_queued(), "state": "queued"}))
    harness.runs.rows = [{"run_id": _RUN}]

    harness.service.recover_deployments()

    lock = harness.lock_for(_RUN)
    assert lock.held is False
    assert lock.events == ["acquire", "release"]


# ---- status replay ---------------------------------------------------------------------


def test_replay_reports_each_persisted_status_and_stops_when_asked():
    from threading import Event

    harness = build_harness(status=_status(None))
    harness.runs.rows = [{"run_id": _RUN}, {"run_id": "flash-2-efgh"}]

    assert harness.service.replay_status_reports() == 2
    assert len(harness.reporter.reports) == 2

    stop = Event()
    stop.set()
    harness.reporter.reports.clear()
    assert harness.service.replay_status_reports(stop) == 0


def test_replay_delivers_each_status_sequentially_not_through_the_live_reporter():
    """Startup replay must block per item rather than queueing onto the async reporter.

    A historical backlog handed to the shared asynchronous reporter would sit ahead of live
    deployment updates, so every replayed status has to go through the sequential path.
    """
    harness = build_harness(status=_status(None))
    harness.runs.rows = [{"run_id": _RUN}, {"run_id": "flash-2-efgh"}]

    assert harness.service.replay_status_reports() == 2

    assert len(harness.reporter.sequential_reports) == 2


def test_replay_skips_a_row_it_cannot_read_instead_of_aborting():
    harness = build_harness(status=_status(None))
    harness.runs.rows = [{"run_id": "missing"}, {"run_id": _RUN}]
    calls = {"n": 0}
    real_get = harness.runs.get_status

    def flaky(run_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("unreadable record")
        return real_get(run_id)

    harness.runs.get_status = flaky

    assert harness.service.replay_status_reports() == 1


# ---- lock release by the job -----------------------------------------------------------


def test_the_job_releases_the_lock_it_was_handed_even_when_the_attempt_raises():
    harness = _harness_for_attempt(status=_status(_queued()))
    lock = harness.lock_for(_RUN)
    lock.acquire()

    def explode(**_kwargs):
        raise RuntimeError("lifecycle blew up")

    harness.service.lifecycle.finish = explode

    with pytest.raises(RuntimeError, match="lifecycle blew up"):
        harness.service.lifecycle.finish_locked(deploy_lock=lock, run_id=_RUN)

    assert lock.held is False
    assert lock.events == ["acquire", "release"]
