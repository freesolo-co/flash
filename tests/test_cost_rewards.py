"""Cost estimator: GRPO reward-grader latency (a single generalizable average). No network."""

from __future__ import annotations

import pytest

from flash.cost import RunConfig, estimate_cost
from flash.cost.analytical import seconds_per_step
from flash.cost.facts import AVG_REWARD_SECONDS_PER_COMPLETION, reward_seconds_per_completion


def test_single_average_is_generalizable():
    # No env classification: every run gets the same average unless it pins its own value.
    assert reward_seconds_per_completion() == AVG_REWARD_SECONDS_PER_COMPLETION


def test_unmeasured_grading_is_not_charged_on_top_of_the_step_floor():
    """An unmeasured grader adds nothing beyond the floor, which was fitted to include grading.

    Pins the VALUE, not the constant: the assertion above compares the accessor to the constant it
    returns, so it holds at any default and never protected this. The old 1.0s default put
    ``completions x 1.0s`` beside a step floor already fitted with measured reward applied, which
    double-charged grading -- 32s of a 32-completion step, and 0.699x geometric bias over the 64
    grpo arms of the 2026-08-01 campaign against 0.995x at 0.0.
    """
    assert AVG_REWARD_SECONDS_PER_COMPLETION == 0.0

    base = {"batch_size": 8, "group_size": 4}
    default = RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, **base)
    explicit_zero = RunConfig(
        "Qwen/Qwen3.5-0.8B", "grpo", 10, reward_seconds_per_completion=0.0, **base
    )
    # the default run must cost exactly what a run that measured "no grading cost" costs.
    assert seconds_per_step(default, "RTX 5090") == pytest.approx(
        seconds_per_step(explicit_zero, "RTX 5090")
    )
    # and a measured slow grader is still charged, on top of the floor.
    slow = RunConfig("Qwen/Qwen3.5-0.8B", "grpo", 10, reward_seconds_per_completion=2.0, **base)
    assert seconds_per_step(slow, "RTX 5090") == pytest.approx(
        seconds_per_step(default, "RTX 5090") + 2.0 * 8 * 4
    )


def test_a_declared_slow_judge_reaches_the_quote():
    """A run whose reward() calls an external judge can price it, end to end from [train].

    The 0.0 default is right for the population it was fitted on (local scorers, 0.0001-0.001s), but
    nothing measures a seconds-long judge before the quote is persisted, so without a declared path
    that run is billed from an underestimate. Goes through runconfig_from_spec rather than RunConfig
    directly -- an in-process RunConfig could always carry the value, and it was the SPEC path that
    had no way to supply it.

    The declared value is per completion AFTER the caller's own batching, because the quote
    multiplies it by every completion in the step while the default adapter scores one call per
    prompt group and runs those groups concurrently. TRAINING.md documents that unit; 3.0 here is a
    magnitude the arithmetic can be checked against, not a recommended setting for a 3s judge.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec

    def _spec(**train):
        return JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "project": "11111111-1111-1111-1111-111111111111",
                "environment": {"id": "acme/env"},
                "train": {"batch_size": 8, "group_size": 4, "max_steps": 10, **train},
            }
        )

    default = runconfig_from_spec(_spec())
    assert default.reward_seconds_per_completion is None

    judge = runconfig_from_spec(_spec(reward_seconds_per_completion=3.0))
    assert judge.reward_seconds_per_completion == 3.0
    # and it must actually move the price: 32 completions x 3s on top of the floor.
    assert seconds_per_step(judge, "H200") == pytest.approx(
        seconds_per_step(default, "H200") + 3.0 * 8 * 4
    )


def test_a_failed_reward_probe_does_not_discard_the_declaration():
    """An all-failed reward probe is not a measurement, so the declaration must survive it.

    ``reward_failures == reward_samples`` is a state __post_init__ permits and trustworthy() does
    not check, so a profile whose every probe failed still arrives here carrying its ~0s latency.
    Treating that as evidence would silently discard the declared judge wall and underquote exactly
    the slow or unavailable grader the knob exists for.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec

    def _run(profile):
        spec = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "project": "11111111-1111-1111-1111-111111111111",
                "environment": {"id": "acme/env"},
                "train": {
                    "batch_size": 8,
                    "group_size": 4,
                    "max_steps": 10,
                    "reward_seconds_per_completion": 3.0,
                },
            }
        )
        object.__setattr__(spec, "_test_rollout", profile)
        return spec

    class _Profile:
        reward_seconds_per_completion = 0.0004
        reward_samples = 3
        reward_failures = 3  # every probe failed
        completion_tokens_mean = 180.5
        prompt_tokens_mean = 95.0

    import flash.cost.spec as cost_spec

    original = cost_spec._rollout_profile
    try:
        cost_spec._rollout_profile = lambda spec: getattr(spec, "_test_rollout", None)
        # all probes failed -> not a measurement -> the declaration is kept
        assert runconfig_from_spec(_run(_Profile())).reward_seconds_per_completion == 3.0

        class _OneSucceeded(_Profile):
            reward_failures = 2  # 3 samples, 1 real success

        # a single successful sample IS a measurement, and it wins over the declaration
        assert runconfig_from_spec(
            _run(_OneSucceeded())
        ).reward_seconds_per_completion == pytest.approx(0.0004)
    finally:
        cost_spec._rollout_profile = original


@pytest.mark.parametrize("algorithm", ["sft", "opd"])
def test_a_declared_judge_is_rejected_where_there_is_no_reward_function(algorithm):
    """Only grpo calls reward(). sft trains on dataset completions and opd distils against a
    teacher, so accepting the knob there would price a wall neither algorithm ever pays."""
    from flash.schema import ConfigError, spec_from_dict

    with pytest.raises(ConfigError, match="reward_seconds_per_completion"):
        spec_from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": algorithm,
                "project": "11111111-1111-1111-1111-111111111111",
                "environment": {"id": "acme/env"},
                "train": {"reward_seconds_per_completion": 3.0},
            }
        )


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
