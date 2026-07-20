import inspect

from flash.engine.worker.rl import _mean_named_reward_metrics, run_rl


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


def test_reward_fn_validates_total_before_aggregating_breakdown() -> None:
    source = inspect.getsource(run_rl)
    total_conversion = 'r = float(breakdown.get("total", 0.0))'
    aggregate_append = "breakdowns.append(breakdown)"

    assert source.count(aggregate_append) == 1
    assert source.index(total_conversion) < source.index(aggregate_append)
