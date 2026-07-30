"""Cost estimator: GRPO reward-grader latency (a single generalizable average). No network."""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.analytical import seconds_per_step
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
    a = estimate_cost(RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 20, environment="acme/code-judge"))
    b = estimate_cost(RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 20, environment="acme/novel-xyz"))
    assert a.total_usd == pytest.approx(b.total_usd)


def test_explicit_override_flows_into_cost():
    fast = RunConfig(
        "Qwen/Qwen3.5-0.8B",
        "grpo",
        10,
        batch_size=8,
        group_size=4,
        reward_seconds_per_completion=0.0,
    )
    slow = RunConfig(
        "Qwen/Qwen3.5-0.8B",
        "grpo",
        10,
        batch_size=8,
        group_size=4,
        reward_seconds_per_completion=5.0,
    )
    assert seconds_per_step(slow, "RTX 5090") > seconds_per_step(fast, "RTX 5090")


def test_sft_has_no_reward_term():
    # SFT has no reward rollout, so an override doesn't change its per-step time.
    a = seconds_per_step(
        RunConfig("Qwen/Qwen3.5-0.8B", "sft", 10, reward_seconds_per_completion=5.0), "RTX 5090"
    )
    b = seconds_per_step(
        RunConfig("Qwen/Qwen3.5-0.8B", "sft", 10, reward_seconds_per_completion=0.0), "RTX 5090"
    )
    assert a == pytest.approx(b)


def test_heavy_override_scales_with_every_completion():
    # A heavier per-completion latency raises cost, and it does so per COMPLETION: both grpo
    # backends score a step's completions one at a time, so nothing bounds the reward wall below
    # the full sum.
    base = {"batch_size": 16, "group_size": 4}
    light = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 20, reward_seconds_per_completion=0.05, **base)
    )
    heavy = estimate_cost(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 20, reward_seconds_per_completion=3.0, **base)
    )
    assert heavy.seconds_per_step > light.seconds_per_step
    assert heavy.total_usd > light.total_usd
    assert any("reward" in n.lower() for n in heavy.notes)


def test_reward_wall_is_serial_over_completions():
    """The reward term must equal completions x latency, not a divided wave count.

    Both single-turn grpo backends score serially, and the verl bridge takes an explicit lock to
    keep it that way for envs whose scorers are not thread-safe. Pricing a concurrency divisor here
    understated the reward term by that divisor at every batch/group shape. Asserted as an exact
    identity against seconds_per_step so a reintroduced divisor cannot pass.
    """
    base = {"batch_size": 8, "group_size": 4, "completion_len": 512, "seq_len": 1024}
    free = seconds_per_step(
        RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, reward_seconds_per_completion=0.0, **base),
        "RTX 5090",
    )
    for latency in (0.25, 1.0, 3.0):
        priced = seconds_per_step(
            RunConfig(
                "Qwen/Qwen3.5-0.8B", "grpo", 10, reward_seconds_per_completion=latency, **base
            ),
            "RTX 5090",
        )
        completions = base["batch_size"] * base["group_size"]
        assert priced - free == pytest.approx(completions * latency, rel=1e-6)


def test_reward_wall_grows_with_group_size():
    """Doubling the completions per step doubles the reward wall.

    Deliberately sized SMALL (8 and 16 completions). A wave-count model rounds both up to the same
    single wave and prices them identically, so this discriminates; picking 32 vs 64 completions
    would NOT -- there the wave count doubles too and the ratio holds under either model, leaving a
    test that cannot fail against the bug it names.
    """
    base = {"batch_size": 2, "completion_len": 512, "seq_len": 1024}
    free = seconds_per_step(
        RunConfig(
            "Qwen/Qwen3.5-0.8B", "grpo", 10, group_size=4, reward_seconds_per_completion=0.0, **base
        ),
        "RTX 5090",
    )
    small = seconds_per_step(
        RunConfig(
            "Qwen/Qwen3.5-0.8B", "grpo", 10, group_size=4, reward_seconds_per_completion=1.0, **base
        ),
        "RTX 5090",
    )
    free_big = seconds_per_step(
        RunConfig(
            "Qwen/Qwen3.5-0.8B", "grpo", 10, group_size=8, reward_seconds_per_completion=0.0, **base
        ),
        "RTX 5090",
    )
    big = seconds_per_step(
        RunConfig(
            "Qwen/Qwen3.5-0.8B", "grpo", 10, group_size=8, reward_seconds_per_completion=1.0, **base
        ),
        "RTX 5090",
    )
    # isolate the reward term from the rollout/update work, which also grows with group size.
    assert (big - free_big) == pytest.approx(2.0 * (small - free), rel=1e-6)
