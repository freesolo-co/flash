"""Warm-up measurement of an env's real per-completion grading latency.

Both single-turn grpo backends grade a step's completions one at a time (the trl reward_fn walks
them in a loop; the verl bridge holds a lock to keep it that way), so grading is pure wall-clock
the gpu spends idle. On an A100 PCIe at the default 64x8 shape that is 80.8% of every step at a 1s
grader and 92.7% at a 3s judge.

The cost model has to price that from a single ``AVG_REWARD_SECONDS_PER_COMPLETION = 1.0`` guess
covering graders whose real span is ~0.01s (regex) to ~3s (llm judge). This measures the actual
value on the actual env before training starts.

Two things this deliberately does NOT do:

- it does not re-implement scoring. it takes the caller's own ``score_one`` callable, so it times
  the exact path training will run. a profiler with its own scoring copy measures the copy.
- it does not trust a fast reading it cannot justify. grading an empty completion is not grading a
  real one (a regex finds nothing and returns immediately; a judge gets a trivial input), so a
  profile built from blank text understates latency. that case is reported as ``degenerate``
  instead of being returned as a confident number.
"""

from __future__ import annotations

import statistics
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

    @property
    def trustworthy(self) -> bool:
        """Whether this reading should be allowed to move a cost estimate or a schedule.

        A degenerate profile timed blank completions and understates real grading. A profile with
        no successful sample has nothing behind its number at all.
        """
        return self.samples > 0 and not self.degenerate and self.failures < self.samples

    def describe(self) -> str:
        if not self.samples:
            return "reward profile: no sample graded successfully"
        base = f"reward profile: {self.seconds_per_completion:.3f}s/completion over {self.samples} samples"
        if self.degenerate:
            base += " (DEGENERATE: completions were blank, real grading will be slower)"
        if self.failures:
            base += f" ({self.failures} sample(s) raised)"
        return base


def profile_reward_latency(
    score_one: Callable[[int, str], float],
    samples: list[tuple[int, str]],
    *,
    max_samples: int = 3,
    budget_s: float = 30.0,
) -> RewardProfile:
    """Time real gradings to learn this env's per-completion cost.

    ``score_one(index, completion) -> float`` must be the same callable training will use.
    ``samples`` are (example_index, completion_text) pairs drawn from the run's own data.

    Bounded twice over -- by ``max_samples`` and by ``budget_s`` -- so a pathologically slow grader
    delays training by a known ceiling rather than an unknown one. Never raises: a profiler that
    can fail the run it is trying to price is worse than no profiler, so scorer errors are counted
    and reported.

    Returns the MEDIAN, not the mean. At these sample counts one slow outlier (a retried http call,
    a cold cache) moves a mean far more than it moves the truth.
    """
    if not samples or max_samples <= 0 or budget_s <= 0:
        return RewardProfile(0.0, 0, degenerate=False, failures=0)

    durations: list[float] = []
    graded_texts: list[str] = []
    failures = 0
    started = time.perf_counter()

    for call, (index, completion) in enumerate(samples[: max_samples + _WARMUP_CALLS]):
        if time.perf_counter() - started >= budget_s:
            break
        call_started = time.perf_counter()
        try:
            score_one(index, completion)
        except Exception:
            # a grader that raises on a sample tells us nothing about its latency, and the
            # exception belongs to training's own error handling, not to a measurement helper.
            failures += 1
            continue
        elapsed = time.perf_counter() - call_started
        if call < _WARMUP_CALLS:
            continue  # warm-up: paid once, never paid again, so not representative
        durations.append(elapsed)
        graded_texts.append(completion)

    if not durations:
        return RewardProfile(0.0, 0, degenerate=False, failures=failures)

    # blank completions do not exercise the grader, so the reading is reported but flagged rather
    # than silently trusted.
    degenerate = all(not text.strip() for text in graded_texts)
    return RewardProfile(
        seconds_per_completion=statistics.median(durations),
        samples=len(durations),
        degenerate=degenerate,
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
