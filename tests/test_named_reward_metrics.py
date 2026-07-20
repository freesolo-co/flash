from flash.engine.worker.rl import _mean_named_reward_metrics


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


def test_plain_scalar_rewards_produce_no_named_metrics() -> None:
    assert _mean_named_reward_metrics([None, None]) == {}
    assert _mean_named_reward_metrics([{"total": 0.4}, {"total": 0.8}]) == {}
