"""Cost estimator: GRPO reward-grader latency (a single generalizable average). No network."""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost, seconds_per_step
from flash.cost.facts import AVG_REWARD_SECONDS_PER_COMPLETION, reward_seconds_per_completion


def test_single_average_is_generalizable():
    # No env classification: every run gets the same average unless it pins its own value.
    assert reward_seconds_per_completion() == AVG_REWARD_SECONDS_PER_COMPLETION


def test_override_wins_and_clamps():
    assert reward_seconds_per_completion(override=1.5) == 1.5
    assert reward_seconds_per_completion(override=-3.0) == 0.0  # clamped to >= 0


def test_environment_does_not_change_cost():
    # The env slug no longer drives reward latency, so two GRPO runs differing only in
    # environment cost identically -- a novel env is never mis-tiered.
    a = estimate_cost(RunConfig("openbmb/MiniCPM5-1B", "grpo", 20, environment="acme/code-judge"))
    b = estimate_cost(RunConfig("openbmb/MiniCPM5-1B", "grpo", 20, environment="acme/novel-xyz"))
    assert a.total_usd == pytest.approx(b.total_usd)


def test_explicit_override_flows_into_cost():
    fast = RunConfig(
        "openbmb/MiniCPM5-1B", "grpo", 10, batch_size=8, group_size=4,
        reward_seconds_per_completion=0.0,
    )
    slow = RunConfig(
        "openbmb/MiniCPM5-1B", "grpo", 10, batch_size=8, group_size=4,
        reward_seconds_per_completion=5.0,
    )
    assert seconds_per_step(slow, "RTX 5090") > seconds_per_step(fast, "RTX 5090")


def test_sft_has_no_reward_term():
    # SFT has no reward rollout, so an override doesn't change its per-step time.
    a = seconds_per_step(
        RunConfig("openbmb/MiniCPM5-1B", "sft", 10, reward_seconds_per_completion=5.0), "RTX 5090"
    )
    b = seconds_per_step(
        RunConfig("openbmb/MiniCPM5-1B", "sft", 10, reward_seconds_per_completion=0.0), "RTX 5090"
    )
    assert a == pytest.approx(b)


def test_heavy_override_bounded_by_concurrency():
    # A heavy per-completion latency raises cost but concurrency bounds it (ceil(comps/16)
    # waves, not comps x latency), so a small run stays under the wall cap.
    base = {"batch_size": 16, "group_size": 4}
    light = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 20, reward_seconds_per_completion=0.05, **base)
    )
    heavy = estimate_cost(
        RunConfig("openbmb/MiniCPM5-1B", "grpo", 20, reward_seconds_per_completion=3.0, **base)
    )
    assert heavy.seconds_per_step > light.seconds_per_step
    assert heavy.total_usd > light.total_usd
    assert any("reward" in n.lower() for n in heavy.notes)
