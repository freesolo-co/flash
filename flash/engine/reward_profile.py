"""measure an env's real per-completion grading latency before training.

use the caller's actual scorer and real nonblank completions; a separate scorer or blank input measures
a different path. calls share a hard budget through ``call_bounded`` so slow or hung user code cannot
stall startup. the result replaces the cost model's single latency guess.
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
        """return whether this profile may affect cost or scheduling.

        require a real completion and a successful sample. a timed-out call invalidates the profile
        only when it already exceeded the typical completed call; uniform slow budget exhaustion
        remains representative.
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
    """run ``fn`` for at most ``timeout_s`` and return ``(ok, elapsed, value)``.

    ``ok`` is true on success, false on exception, and none at the deadline. timed-out calls continue
    only on daemon threads, so callers must use thread-safe scorers. this is public because reference
    gathering also calls user code and must share the same bound.
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
    """time real gradings to estimate this env's per-completion cost.

    ``score_one`` must be the training scorer and must propagate errors. samples are unique, nonblank
    run references; repeat timing can hit scorer caches. discard the first call as warm-up when another
    reference exists, but measure it when it is the only sample.

    bound work by ``max_samples`` and ``budget_s`` and never raise. a deadline outlier can invalidate
    the profile; return the median to resist cold or retried calls.
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

    # grade each reference once because repeats can hit scorer caches and understate training cost.
    # discard the cold call when another reference remains; with one reference, keep it because an
    # overestimate is safer than publishing no measurement.
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
            # a deadline call invalidates the profile only when it already exceeds the median completed
            # call, indicating an unmeasured slow tail. otherwise the budget simply expired during a
            # uniformly slow workload. use the median, not max, so one outlier cannot mask later hangs.
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
