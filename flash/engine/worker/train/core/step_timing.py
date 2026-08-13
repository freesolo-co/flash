"""Steady-state per-step timing, and what it says about finishing inside the wall deadline.

An operator deciding whether to commit hours of GPU time has two numbers before this module, and
neither is reliable alone. The pre-flight quote is derived rather than measured -- dry-run never
executes ``environment.py``, so it cannot model rollout cost at all. And the first observed step is
warmup: step 0 carries engine init, the first weight sync and cache population, none of which
repeat. A measured GRPO run showed step 0 = 515s against a 92s steady state, a factor of 5.6.

That factor is not a small conservative error. Extrapolated over 188 steps it predicted 26.9h, above
the platform's own 24h ``max_wall_seconds`` default, when the run actually needed 4.9h -- so the
projection inverted the decision, arguing for cutting ``max_examples`` on a run that fit comfortably.
The only trustworthy figure is the inter-step delta, which until now required a smoke run and
hand-diffing heartbeat timestamps.

So the rule this module exists to enforce: **a per-step cost that includes one-time initialisation is
not a per-step cost**. Every number here is derived from completed step-to-step spans and never from
the span that ends at the first step line.

That exclusion is structural rather than a filter. verl prints a step's metric line when the step
COMPLETES, so on the measured run the first line landed at 19:52:41 (515s after training began) and
the second at 19:54:13. Timing between consecutive lines therefore measures 92s and never sees the
515s, because the warmup falls before the first line rather than between two of them. A rule that
instead dropped "the first interval" would be both wrong and lossy: it would discard a real steady-
state step while still admitting warmup on any path whose first line arrives earlier.
"""

from __future__ import annotations

import itertools
import statistics


def step_intervals(step_line_times: list[float]) -> list[float]:
    """Wall-clock length of each COMPLETED step, from the times its step lines arrived.

    n step lines bound n-1 whole steps. The span before the first line is not one of them: it holds
    engine init, weight load and cudagraph capture, which a later step never pays again -- and
    including it is exactly the 5.6x error this module exists to prevent. Nothing is known about the
    span after the last line either, since the run may still be inside that step.

    This is the shared definition; ``flash.engine.worker.train.rl.verl_config._step_intervals``
    delegates here so the RL post-run metadata and the live heartbeat cannot drift apart on what a
    step is.
    """
    return [
        later - earlier for earlier, later in itertools.pairwise(step_line_times) if later > earlier
    ]


class StepClock:
    """Records when each step line arrived, and answers what a step now costs.

    SFT and OPD keep no per-step timing of their own -- they measure whole-child wall time only --
    so this holds the timestamps for all three trainers rather than each growing its own list and
    its own idea of which spans count.

    Bounded: a long run publishes tens of thousands of steps, and only recent ones inform a
    steady-state rate. The cap keeps the newest, so a rate that drifts over a long run tracks what
    the trainer is doing now instead of averaging in its first hour.
    """

    # enough spans for a stable median while staying small; the median is over completed steps, so
    # this is ~2x the window a projection actually needs.
    _RETAINED_STEP_LINES = 256

    def __init__(self) -> None:
        self._times: list[float] = []

    def record(self, now: float) -> None:
        """Note that a step line arrived at ``now``."""
        self._times.append(float(now))
        del self._times[: -self._RETAINED_STEP_LINES]

    def intervals(self) -> list[float]:
        return step_intervals(self._times)

    def step_seconds(self) -> float | None:
        """Steady-state seconds per step, or None before one whole step has been measured."""
        return steady_state_step_seconds(self.intervals())


def steady_state_step_seconds(step_intervals_s: list[float]) -> float | None:
    """Typical seconds per step, or None when no whole step has been measured yet.

    Takes the MEDIAN, matching ``_measured_idle_fraction``'s reading of the same intervals: a step
    that happens to publish a checkpoint, or one that pays a lazy recompilation, is a real outlier
    that a mean would smear across the projection.

    The caller supplies intervals from ``step_intervals``, which bounds n steps with n+1 step lines
    and therefore already excludes the warmup span before the first one. Nothing here can recover
    that exclusion if a caller passes raw timestamps instead, so callers must not.
    """
    finite = [float(gap) for gap in step_intervals_s if gap > 0]
    if not finite:
        return None
    return statistics.median(finite)


def projected_remaining_seconds(
    step_intervals_s: list[float],
    *,
    current_step: int,
    total_steps: int,
) -> float | None:
    """Seconds of training still to come, at the measured steady-state rate.

    None when the rate is not yet measurable or the horizon is unknown -- an unknown projection has
    to stay absent rather than default to zero, because a zero here reads as "about to finish" on
    every surface that consumes it.

    Counts only steps not yet run. Steps already completed are spent whether or not they were slow,
    so folding them back in would re-charge the run for its own warmup.
    """
    per_step = steady_state_step_seconds(step_intervals_s)
    if per_step is None:
        return None
    if total_steps <= 0:
        return None
    remaining_steps = max(0, int(total_steps) - max(0, int(current_step)))
    return per_step * remaining_steps


# fraction of the remaining wall allowance a projection may consume before the run is called at
# risk. below 1.0 deliberately: a projection that exactly equals the allowance has no room for the
# final checkpoint upload, which runs synchronously after the last step and can take minutes on a
# large model. warning only once the projection has already passed the deadline would leave nothing
# actionable -- by then the wall is reached mid-training and the run dies holding a paid GPU.
_WALL_RISK_FRACTION = 0.9


def wall_deadline_risk(
    step_intervals_s: list[float],
    *,
    current_step: int,
    total_steps: int,
    remaining_wall_seconds: float | None,
) -> dict[str, float] | None:
    """Report a steady-state projection that will not fit the run's remaining wall allowance.

    None when it fits, or when either side is unmeasurable: this warning exists to be acted on, so a
    guess would spend the user's attention on a number that never justified it.

    Both sides are measured against the SAME clock -- what remains to be trained versus what remains
    to be spent -- so the comparison stays valid on a resumed run, where absolute step counts and
    session wall time disagree.
    """
    projected = projected_remaining_seconds(
        step_intervals_s, current_step=current_step, total_steps=total_steps
    )
    if projected is None:
        return None
    if remaining_wall_seconds is None or remaining_wall_seconds <= 0:
        return None
    if projected <= remaining_wall_seconds * _WALL_RISK_FRACTION:
        return None
    return {
        "projected_remaining_s": projected,
        "remaining_wall_s": float(remaining_wall_seconds),
    }


def step_timing_fields(
    clock: StepClock,
    *,
    current_step: int,
    total_steps: int,
    remaining_wall_seconds: float | None,
) -> dict[str, float | bool]:
    """The step-timing fragment of one step heartbeat: ``{}`` until a whole step has been measured.

    Absent rather than zero or null before then. A renderer distinguishes "not measured yet" from a
    measurement only by the key missing, and step 0 -- the one reading that must never be published
    as a step duration -- is exactly the window where these keys are absent.

    ``step_duration_s`` is the steady-state median, not the duration of the step just finished: one
    step that paid a checkpoint upload is not what the next 180 will cost, and this number's whole
    purpose is to be extrapolated from.
    """
    intervals = clock.intervals()
    per_step = steady_state_step_seconds(intervals)
    if per_step is None:
        return {}
    fields: dict[str, float | bool] = {"step_duration_s": round(per_step, 3)}
    projected = projected_remaining_seconds(
        intervals, current_step=current_step, total_steps=total_steps
    )
    if projected is not None:
        fields["projected_remaining_s"] = round(projected, 1)
    risk = wall_deadline_risk(
        intervals,
        current_step=current_step,
        total_steps=total_steps,
        remaining_wall_seconds=remaining_wall_seconds,
    )
    if risk is not None:
        # the wall the projection is measured against, so a reader can see the comparison rather
        # than trust the flag. only stamped when it is at risk: on a healthy run it is noise.
        fields["remaining_wall_s"] = round(risk["remaining_wall_s"], 1)
        fields["wall_deadline_at_risk"] = True
    return fields
