"""Focused retry policy, snapshot, and atomic decision tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from flash.core.spec import GpuSpec, JobSpec
from flash.providers.core.base import Candidate
from flash.runner.lifecycle import attempts, state, status
from flash.runner.lifecycle.attempts import decide_attempt_failure
from flash.runner.supervise.retry_decision import (
    FailureObservation,
    PersistedRetryDecision,
    RetryPlan,
    RetryState,
    _candidate_usable_vram_gb,
    _strictly_larger_candidates,
    transition_failure,
)


def _spec(run_id: str = "retry-decision", *, max_retries: int = 2) -> JobSpec:
    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=max_retries),
    )


def _candidate(provider: str, gpu: str, vram: float, count: int = 1) -> Candidate:
    return Candidate(provider, gpu, 1.0, vram, count)


def _observe(failure: str, chosen=None, candidates=(), *, cache=False) -> FailureObservation:
    return FailureObservation(failure, chosen, candidates, cache)


@pytest.mark.parametrize(
    "failure", ["stalled", "job_preempted", "poll_error", "no_capacity", "oom"]
)
def test_candidate_bound_failures_select_only_strictly_larger_candidates(failure):
    small = _candidate("runpod", "A100 PCIe", 80)
    equal = _candidate("lambda", "H100", 80)
    larger = _candidate("vast", "RTX Pro 6000", 96)
    retry_state, plan = transition_failure(
        RetryState(2, 2, 1),
        _observe(failure, small, (small, equal, larger)),
        attempt=0,
    )

    survivors, chosen = retry_state.select_candidate((equal, larger))

    assert plan.retry
    assert survivors == (larger,)
    assert chosen is larger


def test_candidate_bound_failure_stops_when_only_equal_vram_remains():
    failed = _candidate("runpod", "A100 PCIe", 80)
    equal = _candidate("lambda", "H100", 80)
    retry_state, plan = transition_failure(
        RetryState(2, 2, 1),
        _observe("stalled", failed, (failed, equal)),
        attempt=0,
    )

    assert not plan.retry
    assert retry_state.select_candidate((equal,))[1] is None


def test_retry_selection_preserves_allocator_order_among_larger_candidates():
    failed = _candidate("runpod", "RTX 4090", 24)
    first = _candidate("vast", "H100", 80)
    second = _candidate("lambda", "RTX Pro 6000", 96)
    retry_state, _ = transition_failure(
        RetryState(2, 2, 1),
        _observe("stalled", failed, (failed, first, second)),
        attempt=0,
    )

    assert retry_state.select_candidate((first, second)) == ((first, second), first)


def test_candidate_less_poll_error_retries_but_no_capacity_stops():
    state_after_poll, poll_plan = transition_failure(
        RetryState(2, 2, 1),
        _observe("poll_error", None, None),
        attempt=0,
    )
    state_after_capacity, capacity_plan = transition_failure(
        RetryState(2, 2, 1),
        _observe("no_capacity", None, None),
        attempt=0,
    )

    assert poll_plan.retry
    assert state_after_poll.infra_used == 1
    assert not capacity_plan.retry
    assert state_after_capacity.infra_used == 0


def test_cache_fallback_is_exact_one_shot_and_consumption_is_derived():
    chosen = _candidate("lambda", "H100", 80, 2)
    retry_state, plan = transition_failure(
        RetryState(2, 2, 1),
        _observe("poll_error", chosen, (chosen,), cache=True),
        attempt=0,
    )

    assert plan.retry
    assert retry_state.drop_weight_cache
    assert retry_state.cache_retry_shape == ("lambda", "H100", 2)
    assert "cache_used" not in retry_state.to_snapshot()
    assert retry_state.select_candidate((chosen,))[1] is chosen


def test_cache_retry_shape_does_not_fall_through_when_exact_shape_disappears():
    exact = _candidate("runpod", "H100", 80, 2)
    larger = _candidate("runpod", "B200", 180)
    retry_state = replace(
        RetryState(2, 2, 1),
        drop_weight_cache=True,
        cache_retry_shape=("runpod", "H100", 2),
    )

    assert retry_state.select_candidate((larger,)) == ((larger,), None)
    assert retry_state.select_candidate((exact, larger)) == ((exact, larger), exact)


def test_queue_grace_stays_ordinary_until_cache_fallback_is_consumed():
    chosen = _candidate("runpod", "H100", 80)
    retry_state = RetryState(0, 1, 1)

    assert not retry_state.on_last_gpu(chosen, (chosen,), cache_fallback_available=True)
    assert retry_state.on_last_gpu(chosen, (chosen,), cache_fallback_available=False)


def test_retry_floor_uses_executed_width_scale():
    failed = _candidate("runpod", "RTX 4090", 24, 2)
    larger = _candidate("runpod", "A100 SXM 40GB", 40)
    retry_state, plan = transition_failure(
        RetryState(0, 1, 0),
        _observe("oom", failed, (failed, larger)),
        attempt=0,
    )

    assert plan.retry
    assert retry_state.usable_vram_floor == pytest.approx(_candidate_usable_vram_gb(failed))
    assert _strictly_larger_candidates((larger,), retry_state.usable_vram_floor) == (larger,)


def test_snapshot_round_trip_uses_one_nested_decision_and_is_frozen():
    spec = _spec()
    decision = PersistedRetryDecision(0, "job_failed", RetryPlan(False, "not retrying"))
    retry_state = replace(RetryState.initial_for_spec(spec), last_decision=decision)

    snapshot = retry_state.to_snapshot()
    restored = RetryState.from_snapshot(spec, snapshot)

    assert snapshot["last_decision"] == {
        "attempt": 0,
        "failure": "job_failed",
        "plan": {"retry": False, "action": "not retrying", "infra_retry_ordinal": None},
    }
    assert restored.persisted_plan(0) == decision.plan
    with pytest.raises(FrozenInstanceError):
        restored.infra_used = 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snapshot: snapshot.update(last_decision=[]),
        lambda snapshot: snapshot["last_decision"].update(attempt=True),
        lambda snapshot: snapshot["last_decision"]["plan"].update(retry=1),
        lambda snapshot: snapshot.update(cache_retry_shape=["runpod", "H100"]),
        lambda snapshot: snapshot.update(drop_weight_cache="yes"),
    ],
)
def test_snapshot_validation_rejects_invalid_typed_state(mutation):
    spec = _spec()
    snapshot = replace(
        RetryState.initial_for_spec(spec),
        last_decision=PersistedRetryDecision(0, "job_failed", RetryPlan(False, "not retrying")),
    ).to_snapshot()
    mutation(snapshot)

    with pytest.raises(ValueError, match="persisted retry state"):
        RetryState.from_snapshot(spec, snapshot)


def test_atomic_retry_cas_has_one_owner_and_reuses_persisted_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spec("atomic-retry-cas")
    state._save_status(state.RunStatus(spec.run_id, "running", spec.to_dict()))
    claim = attempts.reserve_verified_attempt_launch(spec.run_id)
    assert claim is not None
    chosen = _candidate("runpod", "RTX 4090", 24)

    winner = decide_attempt_failure(
        spec.run_id,
        claim_token=claim.token,
        expected_remote=None,
        observation=_observe("stalled", chosen, (chosen, _candidate("runpod", "H100", 80))),
        attempt=0,
    )
    stale = decide_attempt_failure(
        spec.run_id,
        claim_token=claim.token,
        expected_remote=None,
        observation=_observe("job_failed", chosen, (chosen,)),
        attempt=0,
    )

    assert winner is not None
    assert winner.retry
    assert stale is None
    persisted = RetryState.from_snapshot(
        spec,
        status._load_status_json(spec.run_id)[state._RETRY_STATE_KEY],
    )
    assert persisted.last_decision is not None
    assert persisted.last_decision.failure == "stalled"
