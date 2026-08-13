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

Two other spans between step lines are not steps either, for the same reason. A step number that
verl REPRINTS -- on a validation pass, or replaying a resume step -- had no optimizer update between
the two lines. And a span in which this process BLOCKED, waiting on a synchronous heartbeat upload
that retries until it commits, contains work no step pays for. Both are excluded at ``StepClock``,
so every trainer gets the same answer.

The pace and the projection then part ways deliberately: ``step_duration_s`` is the median, what a
step costs, while a projection of the next N steps is a sum and takes an amortized rate that charges
for the checkpoint saves the median suppresses.
"""

from __future__ import annotations

import itertools
import statistics

# how long a call inside the stdout loop must take before the span containing it stops being a step.
# an uncommitted heartbeat returns in microseconds under the throttle, while the paths that actually
# block -- a failed upload, or a 30s wait on the upload lock -- are orders of magnitude above this.
_BLOCKING_CALL_THRESHOLD_S = 1.0


def step_intervals(step_line_times: list[float]) -> list[float]:
    """Wall-clock length of each COMPLETED step, from the times its step lines arrived.

    n step lines bound n-1 whole steps. The span before the first line is not one of them: it holds
    engine init, weight load and cudagraph capture, which a later step never pays again -- and
    including it is exactly the 5.6x error this module exists to prevent. Nothing is known about the
    span after the last line either, since the run may still be inside that step.

    This is the raw definition, and ``StepClock`` is what callers actually keep: it applies the other
    exclusions (reprinted step numbers, spans containing a blocking upload) on top. RL's post-run
    ``step_intervals`` metadata reads that same clock rather than a parallel unfiltered list, so a
    finished run cannot report a step cost the operator never saw while it was running.
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
        # closed runs of consecutive step lines, split where blocking work made a span unmeasurable.
        self._segments: list[list[float]] = []
        self._last_step: int | None = None
        self._break_after_last = False
        # a reprint's own arrival, held as the start of the next segment (see record).
        self._pending_baseline: float | None = None

    def note_blocking_work(self) -> None:
        """Declare that the caller is about to block, so the span in progress is not a step.

        The stdout consumer timestamps a step line when it READS it, so anything that blocks that
        loop defers the next timestamp and lands in the next interval. RL's first metric line is
        followed by a forced heartbeat that retries until the upload commits, which on a flaky HF
        can hold the loop for a long time -- and the resulting interval would be published as what a
        step costs.

        Dropping that one span is the honest answer rather than subtracting an estimate of the
        block: the step was still running while we blocked, so what we could subtract is not what
        the step cost. Same principle as the warmup exclusion -- a span that includes work no step
        pays is not a step -- and it costs one interval on a run that has many.
        """
        self._break_after_last = True

    def note_if_blocked(self, elapsed_s: float) -> None:
        """Break the span when a call took long enough to have blocked the reader.

        Measured rather than inferred from a success flag. A heartbeat that returns "not committed"
        may have skipped instantly under the throttle OR waited out a 30s upload lock and failed, and
        those are opposite cases: the first must not break the span, the second must. Timing the call
        separates them without asking ``heartbeat()`` to report why it declined -- and it also covers
        a slow upload that DID commit.

        The threshold is deliberately well below a plausible step so that ordinary call overhead
        never splits a segment; anything above it is long enough to distort an interval either way.
        """
        if elapsed_s >= _BLOCKING_CALL_THRESHOLD_S:
            self._break_after_last = True

    def record(self, now: float, step: int | None = None) -> None:
        """Note that a step line for ``step`` arrived at ``now``.

        A repeated step number closes the segment instead of extending it. verl reprints a step on a
        validation pass and a resumed run replays its resume step -- ``append_step_metrics`` dedupes
        exactly these repeats for the metrics backlog -- and no optimizer update happened between the
        two lines. Timing that span would measure the validation pass or the resume init and publish
        it as the cost of a step, which is the same class of error as counting warmup.

        The repeat's OWN timestamp becomes the baseline for the step that follows. Merely discarding
        it would leave the pre-repeat timestamp in place, so the next real step's interval would
        still swallow the whole replay: a resume printing step 5 at t=0 and again at t=50, then step
        6 at t=142, must measure 92s for step 6 and not 142s.

        Only an IMMEDIATE repeat is skipped, not every non-advancing number. The step patterns these
        callers scan with differ -- SFT and OPD use ``run_verl_training``'s looser ``step:\\s*(\\d+)``,
        which also matches ``global_step:9`` inside a checkpoint path, where RL's gate requires a
        word boundary. A monotonic rule would let one such spurious high number suppress every real
        step after it and freeze the published pace at whatever it last measured, which is worse
        than the repeat it set out to exclude: a wrong number that never corrects itself.

        ``step`` is optional because a caller that cannot identify the step is better off timing
        every line than timing none; only a caller that KNOWS a number repeated can skip it.
        """
        if step is not None:
            if self._last_step is not None and int(step) == self._last_step:
                # the span up to here is not a step, but this timestamp is where the next one starts.
                self._break_after_last = True
                self._pending_baseline = float(now)
                return
            self._last_step = int(step)
        if self._break_after_last and self._times:
            # this line closes a span that is not a step -- it held blocking work, or a reprint --
            # so it opens a new segment instead of extending the current one.
            self._segments.append(self._times)
            # a reprint leaves a baseline: the repeat's own arrival is when the next step began, so
            # the new segment starts there rather than discarding it and measuring from this line
            # back to before the repeat. blocking work leaves none, because the step was already
            # running while we blocked and no timestamp in that span bounds it.
            self._times = [self._pending_baseline] if self._pending_baseline is not None else []
        self._break_after_last = False
        self._pending_baseline = None
        self._times.append(float(now))
        self._trim()

    def _trim(self) -> None:
        """Keep the newest ``_RETAINED_STEP_LINES`` timestamps across all segments."""
        del self._times[: -self._RETAINED_STEP_LINES]
        budget = self._RETAINED_STEP_LINES - len(self._times)
        kept: list[list[float]] = []
        for segment in reversed(self._segments):
            if budget <= 1:
                break
            # a segment shorter than two lines bounds no step, so it is dropped rather than kept.
            trimmed = segment[-budget:]
            if len(trimmed) >= 2:
                kept.append(trimmed)
                budget -= len(trimmed)
        self._segments = list(reversed(kept))

    def intervals(self) -> list[float]:
        """Every measured step span, across segments split by blocking work."""
        spans: list[float] = []
        for segment in (*self._segments, self._times):
            spans.extend(step_intervals(segment))
        return spans

    def step_seconds(self) -> float | None:
        """Steady-state seconds per step, or None before one whole step has been measured."""
        return steady_state_step_seconds(self.intervals())


def steady_state_step_seconds(step_intervals_s: list[float]) -> float | None:
    """Typical seconds per step, or None when no whole step has been measured yet.

    Takes the MEDIAN, matching ``_measured_idle_fraction``'s reading of the same intervals: a step
    that happens to publish a checkpoint, or one that pays a lazy recompilation, is a real outlier
    that a mean would smear across every step the operator is trying to reason about.

    This is what a step COSTS, and it is what ``step_duration_s`` publishes. Projecting a sum of
    future steps is a different question with a different estimator -- see
    ``_amortized_step_seconds``, which charges for the saves this deliberately ignores.

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
    per_step = _amortized_step_seconds(step_intervals_s)
    if per_step is None:
        return None
    if total_steps <= 0:
        return None
    remaining_steps = max(0, int(total_steps) - max(0, int(current_step)))
    return per_step * remaining_steps


def _amortized_step_seconds(step_intervals_s: list[float]) -> float | None:
    """Seconds per step for projecting a SUM of future steps, or None if nothing is measured.

    The median answers "what does a step cost"; this answers "what will the next N cost", and those
    want different estimators. Checkpoint saves are real recurring work -- they land between step
    lines on the save schedule -- but the median deliberately suppresses them as outliers. Projecting
    with it therefore under-counts every future save, which matters most for the wall warning, whose
    entire job is to notice a run that will not fit.

    The mean is the correct estimator for a sum: an occasional expensive step is amortized across
    the horizon at the rate it actually occurs, with no save schedule to thread through or keep in
    sync. It is capped at twice the median so a single pathological span -- a long upload retry, a
    stall -- cannot inflate the projection the way raw warmup once did.
    """
    finite = [float(gap) for gap in step_intervals_s if gap > 0]
    if not finite:
        return None
    typical = statistics.median(finite)
    return min(statistics.fmean(finite), typical * _PROJECTION_OUTLIER_CAP)


# how far above the typical step the amortized rate may sit. saves and recompiles should raise a
# projection; one stalled span should not double it.
_PROJECTION_OUTLIER_CAP = 2.0


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
