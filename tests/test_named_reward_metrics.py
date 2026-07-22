import inspect

from flash.engine.worker.rl import (
    _latest_named_reward_metrics,
    _mean_named_reward_metrics,
    run_rl,
)


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


def test_reward_fn_validates_total_before_aggregating_breakdown() -> None:
    source = inspect.getsource(run_rl)
    total_conversion = 'r = float(breakdown.get("total", 0.0))'
    aggregate_append = "breakdowns.append(breakdown)"

    assert source.count(aggregate_append) == 1
    assert source.index(total_conversion) < source.index(aggregate_append)


def test_reward_fn_accumulates_breakdowns_across_microbatches() -> None:
    source = inspect.getsource(run_rl)

    assert "pending_named_breakdowns.extend(breakdowns)" in source
    assert "_latest_named_reward_metrics(" in source
    # reconciled with #591 log-follow: the rl_step heartbeat merges reward metrics
    # (**hb_cb.latest_fields()) alongside metrics_last; verify latest_fields is still emitted
    assert "hb_cb.latest_fields()" in source
    assert "breakdowns.append(None)" in source
