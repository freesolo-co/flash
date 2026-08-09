"""Warm-up reward-latency profiling. No network, no gpu."""

from __future__ import annotations

import threading
import time

import pytest

from flash.engine.profiling.reward_profile import gpu_idle_fraction, profile_reward_latency


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
    assert profile.samples == 0  # nothing real was graded, so there is no reading
    assert profile.degenerate
    assert not profile.trustworthy
    assert "unmeasured" in profile.describe()


def test_blank_completions_are_excluded_from_a_mixed_set():
    """A blank timing must not enter the statistic even when other samples are real.

    The earlier version of this asserted only that a mixed set is NOT degenerate, which let blank
    timings into the median: three instant blanks and one slow real grading produce a fast median
    that still reports `trustworthy`, which is exactly the under-reading the degenerate flag exists
    to prevent. Blanks are dropped before timing, so only real completions are measured.

    The scorer here behaves like a real grader -- it returns immediately on blank input and costs
    20ms on real text -- because that is what makes the two models disagree on the NUMBER and not
    just the sample count. Keeping blanks yields a median near 0 while still reporting trustworthy;
    dropping them yields ~20ms.
    """

    def realistic(index: int, completion: str) -> float:
        if not completion.strip():
            return 0.0  # a regex grader finds nothing in blank text and returns at once
        end = time.perf_counter() + 0.02
        while time.perf_counter() < end:
            pass
        return 1.0

    mixed = [(0, ""), (1, "  "), (2, ""), (3, "real one"), (4, "real two"), (5, "real three")]
    profile = profile_reward_latency(realistic, mixed, max_samples=3)
    assert not profile.degenerate
    assert profile.trustworthy
    # 3 real references, one spent as the warm-up: its setup cost is not steady-state grading, and
    # at these counts it cannot be averaged away.
    assert profile.samples == 2
    assert profile.seconds_per_completion == pytest.approx(0.02, abs=0.008)


def test_a_hung_grader_cannot_outlast_the_budget():
    """The budget must bound time spent INSIDE a call, not just the gap before one.

    Checking the deadline only between calls means one grader that never returns blocks training
    forever, which is the opposite of the bounded-delay promise. A hung call is counted as a
    failure rather than a data point: it proves grading is at least this slow, not how slow.
    """
    release = threading.Event()

    def hangs(index: int, completion: str) -> float:
        release.wait(30.0)  # never set during the test
        return 1.0

    started = time.perf_counter()
    try:
        profile = profile_reward_latency(hangs, _samples(6), max_samples=3, budget_s=0.15)
        elapsed = time.perf_counter() - started
        assert elapsed < 2.0, f"budget did not bound the in-flight call ({elapsed:.1f}s)"
        assert profile.failures >= 1
        assert not profile.trustworthy
    finally:
        release.set()  # let the abandoned worker thread exit


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
    # a 1s grader idles a real share of the step, but NOT most of it. this used to assert > 0.5,
    # which held only because the step model was missing its floor: with old_log_prob, weight sync
    # and checkpointing unpriced, gpu_s was small enough that a fictitious reward wall looked
    # dominant. those phases are real gpu work, so the honest split leaves the grader a minority.
    assert 0.2 < mine < 0.5


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
    from flash.engine.worker.rl_train import _log_reward_profile

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
    assert "unmeasured" not in out  # real reference text was graded
    assert "per step" in out
    assert "32 completions" in out

    def exploding(index: int, completion: str) -> float:
        raise RuntimeError("grader down")

    _log_reward_profile(Env(), exploding, examples, 32)  # must not raise

    class BrokenEnv:
        def sft_completion(self, example):
            raise RuntimeError("no reference available")

    _log_reward_profile(BrokenEnv(), score_one, examples, 32)  # must not raise


def test_worker_hook_skips_an_env_whose_scorer_is_not_thread_safe(capsys):
    """reward_thread_safe = False means the scorer keeps mutable state, so it must NOT be profiled.

    Profiling grades real completions on the SAME live env training will use. For an opted-out env
    (flash/envs/adapter.py documents the contract) that can advance a counter, consume an api
    quota or warm a cache before the first rollout, changing the rewards training then receives.
    A measurement that moves the thing it measures is worse than a missing measurement.
    """
    from flash.engine.worker.rl_train import _log_reward_profile

    class StatefulEnv:
        reward_thread_safe = False

        def __init__(self):
            self.calls = 0

        def sft_completion(self, example):
            return [{"role": "assistant", "content": "reference"}]

    env = StatefulEnv()

    def score_one(index: int, completion: str) -> float:
        env.calls += 1
        return 1.0

    _log_reward_profile(env, score_one, [{"id": i} for i in range(4)], 32)
    out = capsys.readouterr().out

    assert env.calls == 0, "profiling touched a scorer that declared itself unsafe to race"
    assert "reward_thread_safe" in out  # the skip is reported, not silent


def test_worker_hook_uses_the_envs_assistant_text_semantics(capsys):
    """Reference text must come from flash.content.multimodal.assistant_completion_text.

    A hand-rolled ''.join over every message's `content` breaks two ways: openai-style text blocks
    (`[{"type": "text", "text": "4"}]`) stringify into a python repr, and non-assistant turns get
    concatenated into the graded text. Both hand the grader something no rollout would produce.
    """
    from flash.engine.worker.rl_train import _log_reward_profile

    class BlockEnv:
        def sft_completion(self, example):
            return [
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": [{"type": "text", "text": "4"}]},
            ]

    graded: list[str] = []

    def score_one(index: int, completion: str) -> float:
        graded.append(completion)
        return 1.0

    _log_reward_profile(BlockEnv(), score_one, [{"id": i} for i in range(4)], 32)

    assert graded, "hook never called the scorer"
    assert set(graded) == {"4"}, graded  # not "{'type': 'text', ...}", not the user turn


def test_a_hung_sft_completion_cannot_outlast_the_hook_budget(monkeypatch):
    """Reference extraction is user code too, so it shares the profiler's deadline.

    FreesoloEnvAdapter.sft_completion delegates to the env's own hook, which can block on i/o
    exactly like a grader can. It runs BEFORE the timing phase, so bounding only the timing phase
    leaves the startup delay this hook adds unbounded -- the ceiling would be advertised but not
    held. The budget is patched down so the test costs a fraction of a second rather than 30.
    """
    from flash.engine.worker import rl_train

    monkeypatch.setattr(rl_train, "_PROFILE_BUDGET_S", 0.15)
    release = threading.Event()

    class HangingEnv:
        def sft_completion(self, example):
            release.wait(30.0)  # a blocking fetch that never comes back
            return [{"role": "assistant", "content": "4"}]

    def score_one(index: int, completion: str) -> float:
        return 1.0

    started = time.perf_counter()
    try:
        rl_train._log_reward_profile(HangingEnv(), score_one, [{"id": i} for i in range(4)], 32)
        elapsed = time.perf_counter() - started
        assert elapsed < 2.0, f"extraction ran outside the budget ({elapsed:.1f}s)"
    finally:
        release.set()  # let the abandoned worker thread exit


def test_profiling_does_not_pollute_the_training_sample_buffer():
    """The profiler must not be handed training's own _score closure.

    _score appends to recent_samples, the buffer the per-step completion dump reads, so profiling
    through it would seed the training log with gradings that never came from a rollout.

    Asserted against the worker's SOURCE rather than by calling it. Both closures live inside
    run_rl_train, which needs a model, a dataset and a verl interpreter to reach -- and the obvious
    alternative (build a fake scorer in the test and check it left a local list empty) is
    unfailable: it asserts on a fixture the test itself wrote, and passes with the bug restored.
    This is the weaker kind of test, so it is scoped to the one line that decides the wiring.
    """
    import inspect

    from flash.engine.worker.rl_train import run_rl_train

    source = inspect.getsource(run_rl_train)
    call = source[source.index("_log_reward_profile(") :]
    call = call[: call.index(")")]
    assert "_score_for_profile" in call, call
    assert "env, _score," not in call, call  # training's buffered closure, not the profiling one


def test_score_single_turn_can_propagate_errors_for_the_profiler():
    """raise_on_error lets the profiler tell a real 0.0 from a failed grading.

    Training keeps the swallow-to-0.0 contract so one bad completion never kills a run. But the
    profiler needs the opposite: if every grading raises and it sees 0.0 returned quickly, it
    reports a fast, confident latency for a grader that is entirely down.
    """
    from flash.engine.worker.rl_train import score_single_turn

    class Env:
        def reward(self, graded, ex, state):
            raise RuntimeError("grader is down")

    kwargs = {
        "tok": None,
        "thinking": False,
        "prompt_opened_thinking": False,
        "think_penalty": 0.0,
    }
    # training path: swallowed, scored 0.0
    assert score_single_turn(Env(), "text", {}, **kwargs) == 0.0
    # profiling path: propagated
    with pytest.raises(RuntimeError):
        score_single_turn(Env(), "text", {}, raise_on_error=True, **kwargs)


def test_a_single_reference_still_produces_a_reading():
    """One retained prompt is a valid grpo shape and must not profile as unmeasured.

    resolve_grpo_prompts_per_step (flash/engine/worker/train/rl/config.py) explicitly supports a
    single-prompt dataset. Slicing the sample list would spend that one reference on the discarded
    warm-up, leaving no timed call and reporting that nothing graded -- false, since grading
    succeeded. Below the warm-up threshold the first call is kept AS the measurement.
    """
    seen: list[int] = []
    profile = profile_reward_latency(_sleeper(0.002, record=seen), _samples(1), max_samples=3)

    assert profile.samples == 1, "the single reference produced no measured call"
    # graded exactly once: a repeat would time the scorer's cache, not the scorer.
    assert seen == [0]
    assert profile.trustworthy
    assert "no sample graded successfully" not in profile.describe()


def test_references_are_never_graded_twice():
    """A repeat measures the scorer's cache, not the scorer.

    An expensive grader -- the kind whose idle cost is worth knowing -- is the likeliest to memoize
    by (example, completion). Filling the call plan by repeating inputs would serve every repeat
    from that cache and publish a near-zero latency for a grader that trains against novel
    completions at full price. The error points the dangerous way: it reads as "grading is free",
    which packs more runs onto a card that cannot carry them.
    """
    seen: list[int] = []
    profile = profile_reward_latency(_sleeper(0.002, record=seen), _samples(2), max_samples=3)

    assert len(seen) == len(set(seen)), f"an example was graded more than once: {seen}"
    # two references, one spent as the warm-up. cycling would have produced 4 gradings from 2 inputs.
    assert profile.samples == 1
    assert profile.trustworthy


def test_the_warmup_is_discarded_whenever_a_second_reference_exists():
    """The cold call is spent as setup as soon as there is anything else to measure.

    The warm-up rule is not abandoned by the single-reference case -- it is suspended for it alone.
    Given references to spare the first is discarded exactly as before, and given exactly two it is
    STILL discarded, because two measured calls is the worst case for a cold outlier rather than a
    safe one.
    """
    seen: list[int] = []
    profile = profile_reward_latency(_sleeper(0.002, record=seen), _samples(4), max_samples=3)
    assert len(seen) == 4  # warm-up + 3 measured
    assert profile.samples == 3, "the warm-up was counted as a measurement"

    seen.clear()
    profile = profile_reward_latency(_sleeper(0.002, record=seen), _samples(2), max_samples=3)
    assert seen == [0, 1], "both references should still be graded, one of them as the warm-up"
    assert profile.samples == 1, "with a second reference to measure, the cold call is not a sample"


def test_a_cold_first_call_cannot_set_a_two_reference_reading():
    """A median of two values is their MEAN, so a cold outlier would land squarely in the middle.

    This is the case the warm-up exists for and the one where skipping it does the most damage: a
    scorer paying 100x setup on its first call and running fast afterwards would, if both calls were
    measured, publish roughly half the setup cost as its steady-state grading latency -- and mark it
    trustworthy. Above two measured calls the median leaves an outlier at an end; at exactly two
    there is no end for it to sit at.
    """
    calls = {"n": 0}

    def slow_setup_then_fast(index: int, completion: str) -> float:
        calls["n"] += 1
        end = time.perf_counter() + (0.1 if calls["n"] == 1 else 0.001)
        while time.perf_counter() < end:
            pass
        return 1.0

    profile = profile_reward_latency(slow_setup_then_fast, _samples(2), max_samples=3)

    assert profile.trustworthy
    # the ~1ms warm call, not the ~50ms average of a 100ms setup and a 1ms grading.
    assert profile.seconds_per_completion < 0.02, (
        f"the cold call set the reading: {profile.seconds_per_completion:.4f}s"
    )


def test_an_outlier_timeout_voids_the_whole_reading():
    """A hang that outruns every finished call cannot be outvoted by the fast ones.

    Warm-up and two measured calls return in ~2ms, then the fourth never comes back. Reporting the
    2ms median would advertise the quick half of a grader that also has a half taking at least
    5000x longer -- precisely the understatement this module exists to prevent.
    """
    release = threading.Event()
    calls = {"n": 0}

    def fast_then_hangs(index: int, completion: str) -> float:
        calls["n"] += 1
        if calls["n"] <= 3:
            end = time.perf_counter() + 0.002
            while time.perf_counter() < end:
                pass
            return 1.0
        release.wait(30.0)  # never set during the test
        return 1.0

    try:
        profile = profile_reward_latency(fast_then_hangs, _samples(8), max_samples=4, budget_s=0.5)
        assert profile.timed_out
        assert not profile.trustworthy, "a fast median was published despite an unbounded call"
        assert profile.seconds_per_completion == 0.0
        assert "unmeasured" in profile.describe()
        # the message must not claim nothing graded: two calls did.
        assert "no sample graded successfully" not in profile.describe()
    finally:
        release.set()


def test_one_slow_call_does_not_license_every_later_hang():
    """A hang must be judged against a typical call, not against the slowest one seen.

    Comparing to max(durations) lets a single slow call raise the bar above anything the remaining
    budget could fit -- after which NO hang can exceed it, and every one reads as ordinary budget
    exhaustion. Here one measured call takes ~150ms and the rest ~1ms, leaving the hang less than
    150ms of budget, so a max-based rule would publish the ~1ms median for a grader whose last call
    never returned at all.
    """
    release = threading.Event()
    calls = {"n": 0}

    def one_slow_then_fast_then_hangs(index: int, completion: str) -> float:
        calls["n"] += 1
        if calls["n"] >= 5:
            release.wait(30.0)  # never set during the test
            return 1.0
        # call 1 is the discarded warm-up; call 2 is a MEASURED ~150ms outlier that lands in
        # durations, then two ~1ms calls. the budget then leaves less than 150ms for the hang, so a
        # max-based rule cannot fire and would publish the ~1ms median.
        end = time.perf_counter() + (0.15 if calls["n"] == 2 else 0.001)
        while time.perf_counter() < end:
            pass
        return 1.0

    try:
        profile = profile_reward_latency(
            one_slow_then_fast_then_hangs, _samples(8), max_samples=5, budget_s=0.23
        )
        assert profile.timed_out, "an early slow call masked a genuine hang"
        assert not profile.trustworthy
    finally:
        release.set()


def test_a_uniformly_slow_grader_still_reports_its_latency():
    """Budget exhaustion mid-call is not evidence of an anomaly, so it must not void the reading.

    A grader slow enough that the budget expires partway through a later call is exactly the case
    worth pricing -- it is the one that idles the gpu. Voiding here would mean the slower a grader
    gets, the less likely the profiler is to say anything about it. The abandoned call ran no
    longer than the calls that finished, so the completed samples remain representative.
    """
    # ~0.05s per call against a 0.17s budget: warm-up + 2 measured fit, the next is cut off
    # mid-flight having run less than a completed call did.
    profile = profile_reward_latency(_sleeper(0.05), _samples(8), max_samples=6, budget_s=0.17)

    assert not profile.timed_out, "budget exhaustion was mistaken for an anomalous call"
    assert profile.samples >= 1
    assert profile.trustworthy
    assert profile.seconds_per_completion == pytest.approx(0.05, abs=0.03)


def test_hook_timeout_message_reports_references_already_gathered(monkeypatch, capsys):
    """The skip reason must not claim nothing was gathered when something was.

    The deadline is shared, so the hook returns either way -- what is being fixed is the stated
    reason. A reader told "no reference completion could be gathered" goes looking for an env that
    returns nothing, when the real fault is a hook that returned twice and then stalled.
    """
    from flash.engine.worker import rl_train

    monkeypatch.setattr(rl_train, "_PROFILE_BUDGET_S", 0.3)
    release = threading.Event()
    calls = {"n": 0}

    class SlowThirdEnv:
        def sft_completion(self, example):
            calls["n"] += 1
            if calls["n"] >= 3:
                release.wait(30.0)  # stalls only after two references are in hand
            return [{"role": "assistant", "content": "4"}]

    def score_one(index: int, completion: str) -> float:
        return 1.0

    try:
        rl_train._log_reward_profile(SlowThirdEnv(), score_one, [{"id": i} for i in range(4)], 32)
        out = capsys.readouterr().out
        assert "did not return within" in out, out
        assert "2 usable reference completion(s) gathered" in out, out
        assert "no reference completion could be gathered" not in out, out
    finally:
        release.set()


def test_hook_timeout_message_does_not_count_failed_references(monkeypatch, capsys):
    """A failed call still occupies a slot, and counting it would claim a reference it never had.

    Failed attempts append an empty placeholder to keep example indices aligned with the samples
    list. Reporting that raw length would say references were gathered when every attempt failed --
    as misleading as the "no reference could be gathered" message this replaced, in the opposite
    direction.
    """
    from flash.engine.worker import rl_train

    monkeypatch.setattr(rl_train, "_PROFILE_BUDGET_S", 0.3)
    release = threading.Event()
    calls = {"n": 0}

    class FailsThenStallsEnv:
        def sft_completion(self, example):
            calls["n"] += 1
            if calls["n"] >= 3:
                release.wait(30.0)
            raise RuntimeError("hook is broken")  # the first two produce nothing usable

    def score_one(index: int, completion: str) -> float:
        return 1.0

    try:
        rl_train._log_reward_profile(
            FailsThenStallsEnv(), score_one, [{"id": i} for i in range(4)], 32
        )
        out = capsys.readouterr().out
        assert "did not return within" in out, out
        assert "0 usable reference completion(s) gathered" in out, out
    finally:
        release.set()


def test_hook_timeout_message_does_not_count_blank_references(monkeypatch, capsys):
    """A whitespace-only completion is dropped by the profiler, so it is not a gathered reference.

    ``profile_reward_latency`` rejects these with ``text.strip()`` -- grading an empty string times
    the early return, not the grader. A count that admitted them would name a reference the profiler
    would have discarded, sending a reader after the wrong hook.
    """
    from flash.engine.worker import rl_train

    monkeypatch.setattr(rl_train, "_PROFILE_BUDGET_S", 0.3)
    release = threading.Event()
    calls = {"n": 0}

    class BlankThenStallsEnv:
        def sft_completion(self, example):
            calls["n"] += 1
            if calls["n"] >= 3:
                release.wait(30.0)
            return "   \n\t "  # succeeds, but nothing here can be profiled

    def score_one(index: int, completion: str) -> float:
        return 1.0

    try:
        rl_train._log_reward_profile(
            BlankThenStallsEnv(), score_one, [{"id": i} for i in range(4)], 32
        )
        out = capsys.readouterr().out
        assert "did not return within" in out, out
        assert "0 usable reference completion(s) gathered" in out, out
    finally:
        release.set()
