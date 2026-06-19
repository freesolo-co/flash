"""Cost estimator: GRPO reward-function latency. No network."""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost, seconds_per_step
from flash.cost.rewards import REWARD_TIERS, reward_seconds_per_completion


def test_override_wins_and_clamps():
    assert reward_seconds_per_completion("anything", override=1.5) == 1.5
    assert reward_seconds_per_completion("anything", override=-3.0) == 0.0


def test_env_keyword_tiers():
    assert reward_seconds_per_completion("primeintellect/code-contests") == REWARD_TIERS["heavy"]
    assert reward_seconds_per_completion("acme/llm-judge") == REWARD_TIERS["heavy"]
    assert reward_seconds_per_completion("freesolo-co/linkd-search") == REWARD_TIERS["medium"]
    assert reward_seconds_per_completion("acme/sentiment-sft") == REWARD_TIERS["light"]
    assert reward_seconds_per_completion("primeintellect/hendrycks-math") == REWARD_TIERS["trivial"]
    assert reward_seconds_per_completion("freesolo-co/flash-bench") == REWARD_TIERS["trivial"]


def test_unknown_env_uses_default():
    assert reward_seconds_per_completion(None) == reward_seconds_per_completion("totally/novel")


def test_reward_dominates_with_a_heavy_env():
    base = RunConfig("openbmb/MiniCPM5-1B", "grpo", 20, batch_size=16, group_size=4)
    trivial = estimate_cost(
        RunConfig(**{**base.__dict__, "environment": "primeintellect/hendrycks-math"})
    )
    heavy = estimate_cost(RunConfig(**{**base.__dict__, "environment": "acme/code-judge"}))
    assert heavy.seconds_per_step > 10 * trivial.seconds_per_step
    assert heavy.total_usd > trivial.total_usd
    assert any("reward" in n.lower() for n in heavy.notes)


def test_explicit_override_flows_into_cost():
    fast = RunConfig(
        "openbmb/MiniCPM5-1B",
        "grpo",
        10,
        batch_size=8,
        group_size=4,
        reward_seconds_per_completion=0.0,
    )
    slow = RunConfig(
        "openbmb/MiniCPM5-1B",
        "grpo",
        10,
        batch_size=8,
        group_size=4,
        reward_seconds_per_completion=5.0,
    )
    assert seconds_per_step(slow, "RTX 5090") > seconds_per_step(fast, "RTX 5090")


def test_sft_ignores_environment_reward():
    # SFT has no reward rollout, so the environment doesn't change its per-step time.
    a = seconds_per_step(
        RunConfig("openbmb/MiniCPM5-1B", "sft", 10, environment="acme/code-judge"), "RTX 5090"
    )
    b = seconds_per_step(
        RunConfig("openbmb/MiniCPM5-1B", "sft", 10, environment="acme/math"), "RTX 5090"
    )
    assert a == pytest.approx(b)
