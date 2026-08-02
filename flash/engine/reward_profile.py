"""Warm-up measurement of an env's real per-completion grading latency.

Single-turn grpo grades a step's completions one at a time (the verl bridge holds a lock to keep it
that way), so grading is pure wall-clock the gpu spends idle. On an A100 PCIe at the default 64x8
shape that is 46.8% of every step at a 1s grader and 72.5% at a 3s judge.

The cost model has to price that from a single ``AVG_REWARD_SECONDS_PER_COMPLETION = 0.05`` guess
covering graders whose real span is ~0.01s (regex) to ~3s (llm judge). This measures the actual
value on the actual env before training starts.

That default prices GRADING ONLY. The rollout wall a step also pays is priced separately by
``rollout_step_seconds``, so a measured latency shrinks the grading term without deleting the rest
of the step -- which is exactly what a single conflated constant used to do.

What this measurement does NOT do yet: reach the quote. It runs on the worker after submission and
lands in run notes, while the persisted estimate is built at submit time from a ``RunConfig`` whose
``reward_seconds_per_completion`` no caller sets. So a judge env is still quoted at the default.
Wiring this value through is the fix; it belongs with the submit path, not here.

Three things this deliberately does NOT do:

- it does not re-implement scoring. it takes the caller's own ``score_one`` callable, so it times
  the exact path training will run. a profiler with its own scoring copy measures the copy.
- it does not time blank completions. an empty string does not exercise a grader (a regex finds
  nothing and returns immediately; a judge gets a trivial input), so a blank timing is not a
  cheap sample of real grading, it is a measurement of something else. blanks are dropped before
  the clock starts rather than averaged in, because a median over a mixed set silently inherits
  their speed while still reporting itself trustworthy.
- it does not wait indefinitely. a hung grader is bounded by the same budget as a slow one, so
  the ceiling this promises holds even when a call never returns. the caller gathering the sample
  completions is usually calling user code too, so ``call_bounded`` is public: the ceiling has to
  cover everything the profiler causes to run, not only the part it times.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# one discarded warm-up call: the first grading pays import, connection-setup and cache-fill costs
# that no later completion pays, and at n=3 that single outlier would dominate a mean.
_WARMUP_CALLS = 1


@dataclass(frozen=True)
class RewardProfile:
    """A measured per-completion grading latency, plus what it is safe to conclude from it."""

    seconds_per_completion: float
    samples: int
    degenerate: bool
    failures: int
    timed_out: bool = False

    @property
    def trustworthy(self) -> bool:
        """Whether this reading should be allowed to move a cost estimate or a schedule.

        A degenerate profile had no real completion to grade, so it measured nothing about this
        env's grader. A profile with no successful sample has nothing behind its number at all.

        A profile flagged ``timed_out`` is untrustworthy no matter how many samples succeeded
        around it. That flag is set only for a call abandoned at the deadline that had ALREADY run
        longer than a TYPICAL call that finished -- an outlier of unknown size, which the median of
        the fast calls would paper over. A grader that is uniformly slow enough to exhaust the
        budget mid-call is not flagged, because there the measured calls are representative and the
        only thing that ran out was time.
        """
        return (
            self.samples > 0
            and not self.degenerate
            and not self.timed_out
            and self.failures < self.samples
        )

    def describe(self) -> str:
        if self.degenerate:
            return "reward profile: no non-blank reference completion to grade, latency unmeasured"
        if self.timed_out:
            # distinct from "nothing graded": samples may have succeeded, and saying none did
            # would be false. what makes this unusable is the call that never came back.
            return (
                f"reward profile: a grading exceeded the budget after {self.samples} sample(s), "
                "latency unmeasured"
            )
        if not self.samples:
            return f"reward profile: no sample graded successfully ({self.failures} failed)"
        base = f"reward profile: {self.seconds_per_completion:.3f}s/completion over {self.samples} samples"
        if self.failures:
            base += f" ({self.failures} sample(s) failed)"
        return base


def call_bounded(fn: Callable[[], object], timeout_s: float) -> tuple[bool | None, float, object]:
    """Run ``fn`` and stop waiting after ``timeout_s``. Returns (ok, elapsed, value).

    ``ok`` is True on success, False if it raised, and None if it was still running at the
    deadline; ``value`` is ``fn``'s result on success and None otherwise. A stuck call cannot be
    killed in python, so it is abandoned on a daemon thread: it stops blocking us and it cannot
    hold interpreter exit open. Callers must only reach here for envs that declare a thread-safe
    scorer -- see the guard in the verl worker -- since this moves the call off the calling thread.

    Public because the caller that gathers the sample completions is calling user code too, and
    an unbounded step before a bounded one leaves the pair unbounded.
    """
    outcome: dict[str, object] = {}

    def _run() -> None:
        try:
            result = fn()
        except BaseException:  # the caller only needs pass/fail, not the type
            outcome["ok"] = False
        else:
            outcome["ok"] = True
            outcome["value"] = result

    thread = threading.Thread(target=_run, daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout_s)
    elapsed = time.perf_counter() - started
    if thread.is_alive():
        return None, elapsed, None
    return bool(outcome.get("ok", False)), elapsed, outcome.get("value")


def profile_reward_latency(
    score_one: Callable[[int, str], float],
    samples: list[tuple[int, str]],
    *,
    max_samples: int = 3,
    budget_s: float = 30.0,
) -> RewardProfile:
    """Time real gradings to learn this env's per-completion cost.

    ``score_one(index, completion) -> float`` must be the same callable training will use, and it
    must PROPAGATE scoring errors -- a callable that swallows them to 0.0 turns a grader that is
    failing on every call into a fast, confident, wrong reading.

    ``samples`` are (example_index, completion_text) pairs drawn from the run's own data. Blank
    texts are dropped before timing: grading an empty string measures the early-return, not the
    grader. Each surviving reference is graded at most ONCE -- a memoizing scorer would serve a
    repeat from its cache, and timing that would report grading as nearly free. The first call is a
    discarded warm-up whenever a second reference exists, since its setup cost is not steady-state
    grading; a run holding a single reference keeps it as the measurement rather than spending its
    only one on setup.

    Bounded twice over -- by ``max_samples`` and by ``budget_s``, the latter covering time spent
    INSIDE a call as well as between calls -- so a slow or hung grader delays training by a known
    ceiling rather than an unknown one. Never raises: a profiler that can fail the run it is trying
    to price is worse than no profiler, so scorer failures are counted and reported. A call
    abandoned at the deadline having already outrun a typical completed call voids the profile
    rather than being outvoted by the fast ones -- see ``RewardProfile.trustworthy``.

    Returns the MEDIAN, not the mean. At these sample counts one slow outlier (a retried http call,
    a cold cache) moves a mean far more than it moves the truth.
    """
    if max_samples <= 0 or budget_s <= 0:
        return RewardProfile(0.0, 0, degenerate=False, failures=0)

    # a blank completion is not a cheap grading, it is a different operation. dropping these up
    # front is what keeps a mixed set from reporting a blank-fast median as trustworthy.
    real = [(index, text) for index, text in samples if text and text.strip()]
    if not real:
        return RewardProfile(0.0, 0, degenerate=True, failures=0)

    durations: list[float] = []
    failures = 0
    started = time.perf_counter()

    # regrading an input is not a second sample of it. a scorer that memoizes by (example,
    # completion) -- which the expensive graders most worth profiling are the likeliest to do --
    # serves every repeat from its cache, so repeats would time the cache and publish a near-zero
    # latency for a grader that trains against novel completions at full cost. that error points the
    # dangerous way: it reads as "grading is free", which packs more runs onto a card that cannot
    # carry them.
    #
    # so each reference is graded exactly once, and the warm-up is spent whenever a second reference
    # exists to measure instead. the cold call is not a sample of steady-state grading: it carries
    # import, connection setup and cache fill that no later call pays again, and it cannot be
    # averaged away at these counts. two measured calls are the worst case for that -- a median of
    # two is their MEAN, so a scorer with 5s of setup and 10ms warm calls would publish 2.5s and
    # persist it as trustworthy. from three measured calls up the cold one sits at an end where it
    # cannot set the reading, but by then it has already been discarded anyway.
    #
    # a lone reference is the one exception: spending it would leave nothing measured at all. there
    # the first call IS the measurement, overstating by the setup it carries -- the safe direction,
    # since overstating grading cost packs FEWER runs onto a card rather than more.
    warmup_calls = _WARMUP_CALLS if len(real) > 1 else 0
    planned = real[: max_samples + warmup_calls]

    for call, (index, completion) in enumerate(planned):
        remaining = budget_s - (time.perf_counter() - started)
        if remaining <= 0:
            break
        # bind the loop vars into the lambda: the call runs on another thread, so a late-binding
        # closure could read the next iteration's sample.
        ok, elapsed, _ = call_bounded(
            lambda i=index, text=completion: score_one(i, text), remaining
        )
        if ok is None:
            # abandoned at the deadline. all this tells us is that the call took AT LEAST
            # ``elapsed``, so what it means depends entirely on how that compares to what already
            # finished:
            #
            # - longer than a TYPICAL completed call: it is not behaving like the others, and the
            #   median of those others would advertise the quick half of a grader that also has a
            #   slow half. voided.
            # - no slower than typical: the budget simply ran out mid-call, carrying no evidence of
            #   an anomaly, so the measured calls stand. voiding here would mean a uniformly slow
            #   grader -- 9s/completion fits three calls in a 30s budget but not four -- could never
            #   produce a profile at all, and slow graders are exactly the ones whose idle cost is
            #   worth knowing.
            #
            # compared against the median rather than the max: one slow call raises a max above
            # anything the remaining budget could fit, after which NO hang can exceed it and every
            # subsequent one reads as ordinary budget exhaustion. durations of 50ms, 1ms, 1ms then a
            # hang would have published the 1ms median. the median moves only if most calls are
            # slow, which is precisely the uniformly-slow case that must stay measurable.
            failures += 1
            if not durations or elapsed > statistics.median(durations):
                return RewardProfile(
                    0.0, len(durations), degenerate=False, failures=failures, timed_out=True
                )
            break
        if not ok:
            # a grader that raises on a sample tells us nothing about its latency.
            failures += 1
            continue
        if call < warmup_calls:
            continue  # warm-up: paid once, never paid again, so not representative
        durations.append(elapsed)

    if not durations:
        return RewardProfile(0.0, 0, degenerate=False, failures=failures)

    return RewardProfile(
        seconds_per_completion=statistics.median(durations),
        samples=len(durations),
        degenerate=False,
        failures=failures,
    )


def gpu_idle_fraction(seconds_per_completion: float, completions: int, gpu_seconds: float) -> float:
    """Share of a grpo step the gpu spends waiting on serial grading.

    This is the number the profile exists to produce: it converts a per-completion latency into
    the utilization answer, and it is what makes an 80% idle step visible instead of implicit.
    """
    reward_s = max(0.0, seconds_per_completion) * max(0, completions)
    total = gpu_seconds + reward_s
    return 0.0 if total <= 0 else reward_s / total
