"""Fail a GRPO run early once it has provably produced no learning signal.

GRPO's advantage is each rollout's reward minus its group's mean. When every rollout in every
group gets the same reward -- most often because the reward function raises on every call and the
worker scores each failure 0.0 -- every advantage is 0, the policy loss is identically 0, and the
optimizer applies a zero update. verl still completes every step, so without this guard the run
holds its GPU to the step horizon and only then trips the publish-time refusal
(``checkpoints._check_grpo_had_a_gradient``). This guard reaches the same verdict after
``zero_signal_abort_steps`` optimizer steps instead.

It only arms while the run has never shown a positive actor gradient. A run that learned and then
saturated (every group solved, so later advantages are 0) is a healthy run that the publish check
accepts, and it must not be killed here.
"""

from __future__ import annotations

import math
import threading

# small enough to stop a broken reward within minutes, large enough that a hard but working
# binary task does not trip it by chance: at a 5% success rate, group_size 4 and 3 prompts per
# step, a step has no varied group with p ~= 0.54, and 8 such steps in a row has p < 0.01.
DEFAULT_ZERO_SIGNAL_ABORT_STEPS = 8


def resolve_zero_signal_abort_steps(train) -> int:
    """the researcher's threshold, the default when unset, and 0 to disable the guard."""
    requested = getattr(train, "zero_signal_abort_steps", None)
    return DEFAULT_ZERO_SIGNAL_ABORT_STEPS if requested is None else int(requested)


class RewardFaultTally:
    """count reward calls and the ones the scorer turned into a 0.0 because they failed.

    score_single_turn deliberately keeps a raising reward function from killing the run, which is
    exactly what hides a reward that fails on EVERY call. this tally keeps that evidence so the
    zero-signal verdict can name the real cause instead of only its symptom.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scored = 0
        self.faults = 0
        self.last_fault = ""

    def record_scored(self, count: int) -> None:
        with self._lock:
            self.scored += int(count)

    def record_fault(self, exc: BaseException) -> None:
        detail = " ".join(f"{type(exc).__name__}: {exc}".split())
        with self._lock:
            self.faults += 1
            self.last_fault = detail[:120]

    def snapshot(self) -> tuple[int, int, str]:
        with self._lock:
            return self.scored, self.faults, self.last_fault


class ZeroLearningSignalError(RuntimeError):
    """the run has shown no actor gradient for the configured number of steps.

    a RuntimeError, so the worker reports it as a deterministic job failure rather than a
    retriable infrastructure fault: rerunning the same reward on the same data cannot help.
    """


class ZeroLearningSignalGuard:
    def __init__(
        self,
        abort_after: int,
        *,
        resume_step: int = 0,
        prior_positive_step: int | None = None,
        faults: RewardFaultTally | None = None,
    ) -> None:
        self.abort_after = int(abort_after)
        self.resume_step = int(resume_step)
        self.faults = faults
        # a resumed attempt whose checkpoint already carries a positive gradient has learned.
        self._seen_positive = prior_positive_step is not None
        self._zero_steps: dict[int, tuple[float | None, float | None, float | None]] = {}

    @property
    def armed(self) -> bool:
        return self.abort_after > 0 and not self._seen_positive

    def observe(
        self,
        step: int,
        *,
        grad_norm: float | None,
        reward_mean: float | None = None,
        advantage_min: float | None = None,
        advantage_max: float | None = None,
    ) -> None:
        """account one optimizer step; raise once the run has provably learned nothing."""
        step = int(step)
        if not self.armed or step <= self.resume_step:
            return
        if not isinstance(grad_norm, float) or not math.isfinite(grad_norm):
            # no parseable gradient is not evidence of a zero one; the publish check owns it.
            return
        if grad_norm > 0.0:
            self._seen_positive = True
            self._zero_steps.clear()
            return
        # keyed by step: verl may reprint a step line, and a repeat must not count twice.
        self._zero_steps[step] = (reward_mean, advantage_min, advantage_max)
        if len(self._zero_steps) >= self.abort_after:
            raise ZeroLearningSignalError(self.diagnosis())

    def diagnosis(self) -> str:
        steps = sorted(self._zero_steps)
        span = f"steps {steps[0]}-{steps[-1]}" if len(steps) > 1 else f"step {steps[0]}"
        rows = [self._zero_steps[step] for step in steps]
        rewards = [r for r, _, _ in rows if isinstance(r, float)]
        flat_advantage = all(lo == 0.0 and hi == 0.0 for _, lo, hi in rows if lo is not None)
        # the worker's error heartbeat keeps ~500 chars, so the cause must fit right after this.
        head = (
            f"grpo has no learning signal: actor grad norm was 0 on all {len(steps)} steps so far "
            f"({span}), so the adapter cannot change; stopped early. "
        )
        scored, faults, last_fault = self.faults.snapshot() if self.faults else (0, 0, "")
        if faults and faults == scored:
            cause = (
                f"every reward call raised ({faults}/{scored}) and was scored 0.0 (last: "
                f"{last_fault}). fix the environment's reward/score function. "
            )
        elif faults:
            cause = (
                f"{faults}/{scored} reward calls raised and were scored 0.0 (last: {last_fault}); "
                "the rest gave every rollout in each group the same reward. "
            )
        elif flat_advantage:
            constant = (
                f" (mean reward {rewards[0]:.4g} on every step)"
                if rewards and len(set(rewards)) == 1
                else ""
            )
            cause = (
                "all rollouts in every group received the same reward"
                f"{constant}, so every advantage is 0. check the reward function, or raise "
                "group_size/temperature or use easier prompts. "
            )
        else:
            cause = (
                "advantages were nonzero but no gradient reached the adapter; check whether every "
                "completion is masked (e.g. truncated with mask_truncated_completions). "
            )
        return head + cause + "disable with [train] zero_signal_abort_steps = 0."
