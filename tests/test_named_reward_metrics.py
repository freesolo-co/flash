import inspect

from flash.engine.worker.io.heartbeat import RewardObservabilityBuffer
from flash.engine.worker.train.entry.rl_train import run_rl_train
from flash.engine.worker.train.entry.rl_train_runner import (
    _ingest_step_metrics,
    _start_reward_runtime,
)
from flash.engine.worker.train.rl.rollout.single_turn import score_single_turn


def _published_reward_metrics(breakdowns: list[dict[str, float] | None]) -> dict[str, float]:
    """Average named reward components the way production does: through the live buffer.

    ``RewardObservabilityBuffer`` is the only thing that averages these on the verl path, so the
    semantics below (a missing name counts as zero, a failed grading still occupies a denominator
    slot, a non-finite value is masked to zero) are asserted through it rather than through a
    standalone helper no caller reaches. One ``record`` per completion mirrors ``score_single_turn``,
    which appends a 0-or-1 element accumulator per completion; ``close_generation`` then seals the
    generation and ``heartbeat_fields`` reads back what a heartbeat would actually ship.
    """
    buffer = RewardObservabilityBuffer()
    for index, breakdown in enumerate(breakdowns):
        buffer.record(f"prompt-{index}", f"completion-{index}", 0.0, [breakdown])
    buffer.close_generation(1)
    return buffer.heartbeat_fields().get("reward_metrics", {})


def test_named_reward_metrics_are_averaged_across_completions() -> None:
    breakdowns = [
        {"success": 1.0, "quality": 0.4, "total": 0.7},
        {"success": 0.0, "quality": 0.8, "total": 0.6},
        {"success": 1.0, "quality": 1.0, "total": 0.9},
    ]

    assert _published_reward_metrics(breakdowns) == {
        "success": 2.0 / 3.0,
        "quality": 2.2 / 3.0,
    }


def test_missing_named_metric_counts_as_zero_across_scored_completions() -> None:
    breakdowns = [{"success": 1.0, "total": 1.0}, {"total": 0.0}]

    assert _published_reward_metrics(breakdowns) == {"success": 0.5}


def test_failed_scoring_attempt_counts_as_zero() -> None:
    breakdowns = [{"success": 1.0, "total": 1.0}, None]

    assert _published_reward_metrics(breakdowns) == {"success": 0.5}


def test_non_finite_named_metric_counts_as_zero() -> None:
    breakdowns = [
        {"m": 1.0, "total": 1.0},
        {"m": float("nan"), "total": 1.0},
        {"m": 0.0, "total": 1.0},
    ]

    assert _published_reward_metrics(breakdowns) == {"m": 1.0 / 3.0}


def test_plain_scalar_rewards_produce_no_named_metrics() -> None:
    assert _published_reward_metrics([None, None]) == {}
    assert _published_reward_metrics([{"total": 0.4}, {"total": 0.8}]) == {}


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
    source = "\n".join(
        inspect.getsource(fn) for fn in (run_rl_train, _start_reward_runtime, _ingest_step_metrics)
    )

    assert (
        "observability.record(message_prompts[int(index)], solution_str, score, breakdowns)"
        in source
    )
    assert "observability.heartbeat_fields()" in source
    assert "**_reward_observability()" in source
