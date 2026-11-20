"""A run whose final attempt failed for the platform is classed `infra`, whatever came before.

Shape of html-glm flash-1795106917-5ad63061 (2026-11-19, Mithril Spot H200, max_count=1): earlier
attempts ended on Spot preemption and boot timeouts, then the final attempt's provider create could
not be reconciled (``UnreconciledCreateError``). The run failed at 00:20:36 UTC on 2026-11-20, before
#1835 (merged 01:12:50 UTC, released as 1.4.58) mapped that create to ``unreconciled_create``; the
submit path still returned ``job_failed``, so the run was labelled ``failure_class="job"``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.state as runner_state


def _spot_spec(run_id: str):
    from flash.core.spec import GpuSpec, JobSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-9B-Base",
        algorithm="sft",
        gpu=GpuSpec(provider="spot", spot=True, max_count=1, max_retries=5),
    )


def _chosen():
    from flash.providers.core.base import Candidate

    return Candidate("spot", "H200", 3.2, 141, 1)


def _submit_unreconciled(monkeypatch, spec):
    """Drive the real submit mapping with a provider whose create cannot be reconciled."""
    from flash.providers.core import registry
    from flash.providers.core.base import UnreconciledCreateError
    from flash.runner.lifecycle import deadlines
    from flash.runner.supervise import execution

    class _Provider:
        def submit_attempt(self, *_args, **_kwargs):
            raise UnreconciledCreateError(
                "Mithril launch reconciliation unavailable; refusing duplicate allocation",
                cause="transport error or timeout",
            )

    monkeypatch.setattr(registry, "get_provider", lambda _name: _Provider())
    monkeypatch.setattr(deadlines, "_worker_deadline_at", lambda *_args: None)
    monkeypatch.setattr(execution, "_handle_callback", lambda *_args: None)
    ctx = SimpleNamespace(spec=spec, log=None, source_snapshot=None)
    prepared = (SimpleNamespace(attempt=2), None, None, None)
    candidate_plan = ((_chosen(),), _chosen(), spec, spec)
    return execution._submit_provider(ctx, prepared, candidate_plan)


def _decide(run_id, claim, failure, *, remote=None):
    from flash.runner.lifecycle import status as runner_status
    from flash.runner.supervise.retry_decision import FailureObservation

    if remote is not None:
        assert runner_attempts.persist_claimed_remote(run_id, claim, remote)
        remote = runner_status._load_status_json(run_id)["remote"]
    plan = runner_attempts.decide_attempt_failure(
        run_id,
        claim_token=claim.token,
        expected_remote=remote,
        observation=FailureObservation(failure, chosen=_chosen(), candidates=(_chosen(),)),
        attempt=claim.attempt,
    )
    if plan.retry:
        # the supervisor confirms teardown before it reserves the replacement attempt.
        status = runner_status.get_status(run_id)
        status.remote = None
        runner_state._save_status(status)
    return plan


def test_unreconciled_create_on_submit_is_the_infra_failure(monkeypatch):
    from flash.runner.supervise.retry_decision import UNRECONCILED_CREATE_FAILURE, failure_class

    result, not_started = _submit_unreconciled(monkeypatch, _spot_spec("flash-unreconciled"))
    assert result.failure == UNRECONCILED_CREATE_FAILURE
    assert failure_class(result.failure) == "infra"
    # never retried: a second create could run beside an unseen first one.
    assert not_started is False


def test_final_infra_attempt_after_preemption_and_boot_timeout_is_infra(monkeypatch, tmp_path):
    from flash.runner.lifecycle import status as runner_status

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spot_spec("flash-1795106917-5ad63061")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    for attempt, failure in enumerate(("job_preempted", "boot_timeout")):
        claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
        assert claim.attempt == attempt
        remote = {"provider": "spot", "job_id": f"bid_{attempt}", "attempt": attempt}
        assert _decide(spec.run_id, claim, failure, remote=remote).retry
        assert runner_status.get_status(spec.run_id).failure_class is None

    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    assert claim.attempt == 2
    result, _ = _submit_unreconciled(monkeypatch, spec)
    plan = _decide(spec.run_id, claim, result.failure)
    assert not plan.retry
    status = runner_status.get_status(spec.run_id)
    assert status.failure_class == "infra"
    status.state = "failed"
    assert status.to_dict()["failure_class"] == "infra"


@pytest.mark.parametrize("failure", ["job_failed", "oom"])
def test_final_job_attempt_after_infra_retries_stays_job(monkeypatch, tmp_path, failure):
    """Earlier infra retries never relabel a run whose own final attempt failed."""
    from flash.runner.lifecycle import status as runner_status

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spot_spec(f"flash-final-{failure}")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    remote = {"provider": "spot", "job_id": "bid_0", "attempt": 0}
    assert _decide(spec.run_id, claim, "boot_timeout", remote=remote).retry
    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    remote = {"provider": "spot", "job_id": "bid_1", "attempt": 1}
    plan = _decide(spec.run_id, claim, failure, remote=remote)
    assert not plan.retry
    assert runner_status.get_status(spec.run_id).failure_class == "job"
