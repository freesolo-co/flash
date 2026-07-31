import inspect

from flash.engine.worker.heartbeat import (
    _latest_named_reward_metrics,
    _mean_named_reward_metrics,
)
from flash.engine.worker.rl_train import run_rl_train, score_single_turn


def test_named_reward_metrics_are_averaged_across_completions() -> None:
    breakdowns = [
        {"success": 1.0, "quality": 0.4, "total": 0.7},
        {"success": 0.0, "quality": 0.8, "total": 0.6},
        {"success": 1.0, "quality": 1.0, "total": 0.9},
    ]

    assert _mean_named_reward_metrics(breakdowns) == {
        "success": 2.0 / 3.0,
        "quality": 2.2 / 3.0,
    }


def test_missing_named_metric_counts_as_zero_across_scored_completions() -> None:
    breakdowns = [{"success": 1.0, "total": 1.0}, {"total": 0.0}]

    assert _mean_named_reward_metrics(breakdowns) == {"success": 0.5}


def test_failed_scoring_attempt_counts_as_zero() -> None:
    breakdowns = [{"success": 1.0, "total": 1.0}, None]

    assert _mean_named_reward_metrics(breakdowns) == {"success": 0.5}


def test_named_metrics_repeat_until_new_breakdowns_replace_them() -> None:
    pending = [{"success": 1.0, "total": 1.0}]
    pending.extend([{"success": 0.0, "total": 0.0}, None])
    latest: dict[str, float] = {}

    expected = {"success": 1.0 / 3.0}
    assert _latest_named_reward_metrics(pending, latest) == expected
    assert pending == []
    assert _latest_named_reward_metrics(pending, latest) == expected

    pending.extend([{"success": 1.0, "total": 1.0}])
    assert _latest_named_reward_metrics(pending, latest) == {"success": 1.0}


def test_non_finite_named_metric_counts_as_zero() -> None:
    breakdowns = [
        {"m": 1.0, "total": 1.0},
        {"m": float("nan"), "total": 1.0},
        {"m": 0.0, "total": 1.0},
    ]

    assert _mean_named_reward_metrics(breakdowns) == {"m": 1.0 / 3.0}


def test_plain_scalar_rewards_produce_no_named_metrics() -> None:
    assert _mean_named_reward_metrics([None, None]) == {}
    assert _mean_named_reward_metrics([{"total": 0.4}, {"total": 0.8}]) == {}


def test_scoring_validates_total_before_aggregating_breakdown() -> None:
    source = inspect.getsource(score_single_turn)
    total_conversion = 'r = float(breakdown.get("total", 0.0))'
    aggregate_append = "breakdowns.append(breakdown)"

    assert source.count(aggregate_append) == 1
    assert source.index(total_conversion) < source.index(aggregate_append)
    # a failed grading must still occupy a slot, or it drops out of the denominator and biases
    # every named metric high.
    assert "breakdowns.append(None)" in source


def test_reward_metrics_reach_the_step_heartbeat() -> None:
    """verl's trainer is out of process and cannot host trl's TrainerCallback, so the per-name
    breakdowns travel through the reward-observability buffer into the rl_step liveness fields."""
    source = inspect.getsource(run_rl_train)

    assert "breakdowns=breakdowns" in source
    assert "observability.record(" in source
    assert "observability.heartbeat_fields()" in source
    assert "**_reward_observability()" in source
