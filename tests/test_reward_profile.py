"""Warm-up reward-latency profiling. No network, no gpu."""

from __future__ import annotations

import pytest

from flash.engine.reward_profile import gpu_idle_fraction, profile_reward_latency


def _sleeper(seconds: float, *, record: list[int] | None = None):
    """A scorer that costs a known amount of wall time."""

    def score_one(index: int, completion: str) -> float:
        if record is not None:
            record.append(index)
        # busy-wait rather than sleep: sleep resolution is coarse enough on a loaded box to make
        # a 5ms assertion flaky, and this test asserts on measured durations.
        import time

        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass
        return 1.0

    return score_one


def _samples(n: int, text: str = "a real completion") -> list[tuple[int, str]]:
    return [(i, text) for i in range(n)]


def test_measures_actual_latency():
    profile = profile_reward_latency(_sleeper(0.02), _samples(6), max_samples=3)
    assert profile.samples == 3
    assert profile.seconds_per_completion == pytest.approx(0.02, abs=0.015)
    assert profile.trustworthy


def test_warmup_call_is_discarded():
    """The first grading pays one-off setup cost and must not enter the statistic.

    Asserted via call COUNT: profiling for 3 samples must invoke the scorer 4 times, and the
    extra one must be the first.
    """
    seen: list[int] = []
    profile = profile_reward_latency(_sleeper(0.001, record=seen), _samples(10), max_samples=3)
    assert profile.samples == 3
    assert len(seen) == 4  # 3 measured + 1 discarded warm-up
    assert seen == [0, 1, 2, 3]


def test_blank_completions_are_reported_degenerate_not_trusted():
    """Grading empty text does not exercise the grader, so the reading must not be trusted.

    This is the trap the profiler exists to avoid: a regex grader returns instantly on blank input,
    so a profile built from blanks would report a near-zero latency for an env that is actually
    slow, and the cost model would then price 80% of the step as free.
    """
    profile = profile_reward_latency(_sleeper(0.001), _samples(6, text="   "), max_samples=3)
    assert profile.samples == 3
    assert profile.degenerate
    assert not profile.trustworthy
    assert "DEGENERATE" in profile.describe()


def test_mixed_completions_are_not_degenerate():
    """Only an ALL-blank sample set is degenerate; one real completion is enough to exercise it."""
    mixed = [(0, ""), (1, ""), (2, "real output"), (3, "")]
    profile = profile_reward_latency(_sleeper(0.001), mixed, max_samples=3)
    assert not profile.degenerate
    assert profile.trustworthy


def test_scorer_errors_are_counted_never_raised():
    """A profiler must not be able to fail the run it is trying to price."""

    def exploding(index: int, completion: str) -> float:
        raise RuntimeError("grader is down")

    profile = profile_reward_latency(exploding, _samples(6), max_samples=3)
    assert profile.samples == 0
    assert profile.failures == 4  # every attempted call, warm-up included
    assert not profile.trustworthy
    assert profile.seconds_per_completion == 0.0


def test_partial_failure_still_yields_a_reading():
    calls = {"n": 0}

    def flaky(index: int, completion: str) -> float:
        calls["n"] += 1
        if calls["n"] == 2:  # fails one MEASURED call (call 1 is the warm-up)
            raise RuntimeError("transient")
        return 1.0

    profile = profile_reward_latency(flaky, _samples(8), max_samples=3)
    assert profile.failures == 1
    assert profile.samples == 2
    assert profile.trustworthy  # failures < samples, and real text was graded


def test_budget_bounds_a_slow_grader():
    """A pathologically slow grader must delay training by a KNOWN ceiling.

    Sized so the budget expires mid-run: with a 0.05s grader and a 0.06s budget, the warm-up plus
    one measured call is all that fits, so max_samples cannot be reached.
    """
    profile = profile_reward_latency(_sleeper(0.05), _samples(20), max_samples=10, budget_s=0.06)
    assert profile.samples < 10  # budget stopped it early


def test_empty_or_disabled_inputs_are_safe():
    assert profile_reward_latency(_sleeper(0.0), [], max_samples=3).samples == 0
    assert profile_reward_latency(_sleeper(0.0), _samples(3), max_samples=0).samples == 0
    assert profile_reward_latency(_sleeper(0.0), _samples(3), budget_s=0.0).samples == 0


def test_median_not_mean_resists_one_outlier():
    """One slow call (a retried http request, a cold cache) must not move the reading much.

    A mean over [fast, fast, very-slow] is dragged toward the outlier; a median is not. Uses a
    scorer whose third measured call is 100x slower than the rest.
    """
    calls = {"n": 0}

    def spiky(index: int, completion: str) -> float:
        calls["n"] += 1
        import time

        cost = 0.10 if calls["n"] == 4 else 0.005
        end = time.perf_counter() + cost
        while time.perf_counter() < end:
            pass
        return 1.0

    profile = profile_reward_latency(spiky, _samples(8), max_samples=3)
    assert profile.samples == 3
    # mean would be ~0.037s; the median stays near the typical 0.005s call.
    assert profile.seconds_per_completion < 0.03


def test_gpu_idle_fraction_matches_the_cost_models_split():
    """The utilization number must agree with what the cost model prices.

    Cross-checked against flash.cost.analytical.step_seconds_split for the same shape, so the
    profiler and the estimator cannot drift into disagreeing about the same run.
    """
    from flash.cost import RunConfig
    from flash.cost.analytical import step_seconds_split

    config = RunConfig(
        "Qwen/Qwen3.5-4B",
        "grpo",
        100,
        batch_size=8,
        group_size=4,
        completion_len=512,
        seq_len=1024,
        reward_seconds_per_completion=1.0,
    )
    gpu_s, fixed_s = step_seconds_split(config, "A100 PCIe")
    completions = 8 * 4
    # the model's fixed half also carries per-step overhead, so compare against the reward term it
    # actually added rather than against fixed_s as a whole.
    free = step_seconds_split(
        RunConfig(
            "Qwen/Qwen3.5-4B",
            "grpo",
            100,
            batch_size=8,
            group_size=4,
            completion_len=512,
            seq_len=1024,
            reward_seconds_per_completion=0.0,
        ),
        "A100 PCIe",
    )
    reward_only = fixed_s - free[1]
    assert reward_only == pytest.approx(completions * 1.0, rel=1e-6)
    mine = gpu_idle_fraction(1.0, completions, gpu_s)
    assert mine == pytest.approx(reward_only / (gpu_s + reward_only), rel=1e-6)
    assert mine > 0.5  # a 1s grader really does idle the gpu for most of the step


def test_gpu_idle_fraction_edges():
    assert gpu_idle_fraction(0.0, 32, 10.0) == 0.0
    assert gpu_idle_fraction(1.0, 0, 10.0) == 0.0
    assert gpu_idle_fraction(1.0, 32, 0.0) == 1.0  # all wall time, no gpu work
    assert gpu_idle_fraction(-5.0, 32, 10.0) == 0.0  # negative latency clamped


def test_worker_hook_profiles_against_reference_completions_and_never_raises(capsys):
    """The verl worker hook must exercise the real scorer and stay non-fatal.

    Covers the integration, not just the helper: it asserts the hook grades the env's own
    reference completions (not blank text, which would profile as degenerate and teach us
    nothing), reports the per-step cost, and swallows a scorer that explodes.
    """
    from flash.engine.worker.rl_verl import _log_reward_profile

    class Env:
        def sft_completion(self, example):
            return [{"role": "assistant", "content": f"reference for {example['id']}"}]

    graded: list[str] = []

    def score_one(index: int, completion: str) -> float:
        graded.append(completion)
        return 1.0

    examples = [{"id": i} for i in range(4)]
    _log_reward_profile(Env(), score_one, examples, 32)
    out = capsys.readouterr().out

    assert graded, "hook never called the scorer"
    assert all(t.startswith("reference for") for t in graded), graded
    assert "reward profile:" in out
    assert "DEGENERATE" not in out  # real reference text was graded
    assert "per step" in out
    assert "32 completions" in out

    def exploding(index: int, completion: str) -> float:
        raise RuntimeError("grader down")

    _log_reward_profile(Env(), exploding, examples, 32)  # must not raise

    class BrokenEnv:
        def sft_completion(self, example):
            raise RuntimeError("no reference available")

    _log_reward_profile(BrokenEnv(), score_one, examples, 32)  # must not raise
