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

# how much slower than a pipe read an arrival must be to count as one the reader waited for. pipe
# reads and real steps sit orders of magnitude apart (~0.001s against 0.1s and up), so a wide
# multiple separates them without needing to know the run's scale.
_PIPE_GAP_MULTIPLE = 20.0

# how much longer than the blocking call the next arrival may be and still be treated as output the
# call held back. a line buffered during a call is already sitting in the pipe when the call returns,
# so it is read essentially at that instant; a line the reader genuinely waited for arrives later on
# the run's own schedule. the slack is small because those two are what must be told apart.
_BLOCK_ARRIVAL_SLACK = 1.05


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

    # the drain terminator for the one case the arrival-gap test cannot decide: a block declared
    # before any step has been measured, where there is no pace to size the gap against. once a pace
    # exists the gap ends the drain instead and this does not apply -- see _draining, where capping a
    # measurable drain was itself what let a long backlog take the median.
    _MAX_DRAINED_LINES = 64

    def __init__(self) -> None:
        self._times: list[float] = []
        # closed runs of consecutive step lines, split where blocking work made a span unmeasurable.
        self._segments: list[list[float]] = []
        self._last_step: int | None = None
        self._break_after_last = False
        # a reprint's own arrival, held as the start of the next segment (see record).
        self._pending_baseline: float | None = None
        # set by a block and cleared by the first line the reader genuinely waited for; while set,
        # back-to-back arrivals are discarded as pipe backlog (see _draining).
        self._draining_backlog = False
        self._drained = 0
        # the pipe speed this drain is arriving at, learned from its own gaps (see _draining).
        self._drain_pipe_s: float | None = None
        self._last_arrival: float | None = None
        # blocking time inside the current span that could not be judged when it happened, because
        # no pace was measured yet. resolved at the next step line (see _resolve_unjudged_block).
        self._unjudged_block_s = 0.0
        # when that call returned, so the next arrival can be measured from the call's END rather
        # than from the previous step line, which loses where in the step the call sat.
        self._unjudged_ended_at: float | None = None

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

        Breaking the ONE span is not enough. The child keeps training while the reader is blocked,
        so every step it completes meanwhile queues in the pipe and is then read back-to-back at
        pipe speed. Those arrivals are the reader catching up, not steps costing microseconds, and
        left in they drag the median toward zero -- with the drain outnumbering the real lines the
        published pace collapsed to 0.10s against a true 92s. So the clock also enters a drain (see
        ``_draining``) and discards the burst.
        """
        self._break_after_last = True
        self._draining_backlog = True
        self._drained = 0
        self._drain_pipe_s = None
        # a replay's baseline is the pre-block instant, so carrying it across would start the next
        # segment before the block and measure the very upload this call exists to exclude -- a
        # 900s stall behind a repeated step published 496s/step against a true 92s.
        self._pending_baseline = None

    def note_if_blocked(self, elapsed_s: float, ended_at: float | None = None) -> None:
        """Break the span when a call took long enough to have blocked the reader.

        ``ended_at`` is when the call returned, on the same ``time.monotonic()`` base as ``record``.
        Callers pass it because a duration alone loses WHERE in the step the call sat, and that is
        what decides whether the next line was buffered (see ``_resolve_unjudged_block``).

        Measured rather than inferred from a success flag. A heartbeat that returns "not committed"
        may have skipped instantly under the throttle OR waited out a 30s upload lock and failed, and
        those are opposite cases: the first must not break the span, the second must. Timing the call
        separates them without asking ``heartbeat()`` to report why it declined -- and it also covers
        a slow upload that DID commit.

        Sized against the measured pace when there is one, for the same reason ``_draining`` is: a
        fixed 1s threshold only ever asks "is this call slow in absolute terms", and on a sub-second
        workload a call can stall the reader for SEVERAL steps while staying under it. A 0.9s upload
        on a 0.1s run queues ~9 lines, and with the span left intact they drain in as ordinary
        intervals -- 0.001s published against a true 0.1s. That is worst early in a run, which is
        exactly when RL's forced first-metrics upload fires and when the median has nothing clean to
        outvote it.

        Half the measured pace is the same floor the drain uses, so both halves of this mechanism
        answer "did the reader wait" the same way.

        Before any pace exists the question cannot be answered HERE, and that is precisely the case
        that matters: RL's forced first-metrics upload fires on the FIRST step line, so
        ``intervals()`` is empty at the only moment it is asked. Falling back to the fixed 1s let a
        0.9s upload on a 0.1s run slip under -- ~9 lines queued, the drain was never armed, and the
        median collapsed to 0.001s against a true 0.1s.

        So a sub-threshold call is not discarded, it is DEFERRED: the elapsed time is held and
        judged at the next step line, where the span it landed in is finally known (see
        ``_resolve_unjudged_block``). Deferring rather than guessing is what keeps ordinary call
        overhead from splitting a segment on a run that has not yet shown what a step costs.
        """
        measured = steady_state_step_seconds(self.intervals())
        if measured is not None:
            if elapsed_s >= min(_BLOCKING_CALL_THRESHOLD_S, measured / 2):
                self.note_blocking_work()
            return
        if not self._times:
            # nothing has been timed yet, so this call sits in the warmup that precedes the first
            # step line -- which is already excluded whole, and no interval can contain it. SFT and
            # OPD drive this callback from the child stream with last_hb=0, so it fires on startup
            # output long before any optimizer line; deferring there let a 0.9s upload that had
            # ALREADY FINISHED be charged to the second step's span, arming a drain that discarded
            # 39 genuine steps and published no pace at all. holding it would also be judging it
            # against a span it never touched.
            return
        if elapsed_s > 0:
            # no pace yet: hold it for the next step line to judge against the span it lands in.
            # length alone does not decide it, however long the call was -- a 2s upload that returns
            # mid-step on a 92s run delayed nothing, and charging it a segment cost the only
            # interval a short run had. the next arrival is the evidence either way.
            self._unjudged_block_s += float(elapsed_s)
            if ended_at is not None:
                # when the call ENDED, not just how long it ran. the two are not interchangeable: a
                # call occupying the tail of a step (0.09s to 0.99s of a 0.1s step) buffers output
                # just as surely as one covering the whole span, but a span measured from the
                # previous step line reads 0.991s against a 0.9s call and concludes nothing was
                # held back. span minus duration cannot recover this -- it conflates time before
                # the call started with time after it ended -- so the instant itself is required.
                self._unjudged_ended_at = float(ended_at)

    def _resolve_unjudged_block(self, span_s: float, now: float | None = None) -> None:
        """Judge a deferred block now that the span containing it is known.

        The question is whether the call held this line BACK, and the span it landed in answers it.
        A line buffered during the call is already sitting in the pipe when the call returns, so it
        is read at essentially that instant: its span is bounded by the call. A line the reader
        genuinely waited for arrives later, on the run's own schedule, and the span runs past the
        call by however long the step still had to go.

        Occupying half the span is NOT that test. A 0.09s call inside a 0.1s step clears it while
        buffering nothing -- the next line still arrived on time -- and declaring a block there
        armed a drain that then discarded every genuine arrival: 40 real steps thrown away and no
        pace published at all, because with no measured pace the drain's own floor could not
        recognise a 0.1s wait either. Overlapping a step is ordinary; delaying one is the defect.

        Nothing is retroactively unpublished: the block is declared before this line is recorded, so
        a span that really was contaminated is closed rather than measured, exactly as it would have
        been had the pace been known in time.
        """
        pending, self._unjudged_block_s = self._unjudged_block_s, 0.0
        ended_at, self._unjudged_ended_at = self._unjudged_ended_at, None
        if pending <= 0:
            return
        if ended_at is not None and now is not None:
            # measure from when the call RETURNED. a line already queued is read essentially at that
            # instant, so a gap far below the call's own length is buffered output whatever the call
            # cost. this is the phase the duration alone cannot carry: a 0.9s call sitting at the
            # END of a 0.1s step buffers nine lines, but the span from the previous step line reads
            # 0.991s and looked like the reader had waited -- so the drain never armed and the burst
            # published 0.001s against a true 0.1s.
            if float(now) - ended_at <= pending * (_BLOCK_ARRIVAL_SLACK - 1.0):
                self.note_blocking_work()
            return
        if span_s > 0 and span_s <= pending * _BLOCK_ARRIVAL_SLACK:
            self.note_blocking_work()

    def record(self, now: float, step: int | None = None) -> None:
        """Note that a step line for ``step`` arrived at ``now``.

        ``now`` must come from ``time.monotonic()``. Only differences are ever read from these
        timestamps, and a wall clock can be corrected mid-run by NTP or a VM resync: a backward
        correction produces a negative span that is silently dropped, and a forward one inflates the
        pace, the ETA and the wall-limit warning together. The run's remaining wall allowance is the
        one thing here that must stay on ``time.time()``, because its deadline is an absolute instant
        supplied from outside this process.

        A repeated step number closes the segment instead of extending it. verl reprints a step on a
        validation pass and a resumed run replays its resume step -- ``append_step_metrics`` dedupes
        exactly these repeats for the metrics backlog -- and no optimizer update happened between the
        two lines. Timing that span would measure the validation pass or the resume init and publish
        it as the cost of a step, which is the same class of error as counting warmup.

        The repeat's OWN timestamp becomes the baseline for the step that follows. Merely discarding
        it would leave the pre-repeat timestamp in place, so the next real step's interval would
        still swallow the whole replay: a resume printing step 5 at t=0 and again at t=50, then step
        6 at t=142, must measure 92s for step 6 and not 142s.

        Only an IMMEDIATE repeat is skipped, not every non-advancing number. A monotonic rule would
        let one out-of-order or spurious number suppress every real step after it and freeze the
        published pace at whatever it last measured, which is worse than the repeat it set out to
        exclude: a wrong number that never corrects itself.

        ``step`` is optional because a caller that cannot identify the step is better off timing
        every line than timing none; only a caller that KNOWS a number repeated can skip it.
        """
        if self._unjudged_block_s:
            if self._draining_backlog:
                # this burst is already being excluded, so a sub-threshold call inside it is
                # ordinary overhead. judging it anyway would compare it against a PIPE-SPEED span,
                # which it always wins, re-arming the drain on every queued line and resetting the
                # evidence that ends it -- the drain would never terminate and the run would
                # publish nothing at all.
                self._unjudged_block_s = 0.0
            elif self._times:
                # a block held from before any pace existed. the span it landed in ends HERE, so it
                # can finally be judged -- and it must happen before _draining is consulted, so a
                # confirmed block makes THIS line the head of its drain. judging it afterwards was
                # one line too late: the drain was armed behind a line already recorded as a segment
                # head, so that head and the next queued arrival bounded a pipe-speed interval, and
                # that interval then became the pace the rest of the drain was sized against.
                self._resolve_unjudged_block(float(now) - self._times[-1], float(now))
        if self._draining(now):
            return
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
        self._last_arrival = float(now)
        self._times.append(float(now))
        self._trim()

    def _draining(self, now: float) -> bool:
        """True while this line is one the reader is catching up on rather than one it waited for.

        A blocked reader leaves the child's step lines queued in the pipe, and they then arrive
        back-to-back at pipe speed. Timing those measures the drain, not the steps: with the burst
        outnumbering the real lines the published pace collapsed to 0.10s against a true 92s.

        A burst is identified by its arrival gap, not by counting lines: the whole point is that the
        reader was not waiting, so an inter-arrival below the drain floor is a line that was already
        sitting in the pipe. The first line whose gap exceeds it is one the reader genuinely waited
        for, which ends the drain -- so a block that buffered nothing costs a single line rather than
        a fixed quota.

        The floor is the measured pace when there is one, not a fixed 1s. A fixed floor cannot tell a
        0.4s real step from a 0.001s pipe arrival -- both are "below threshold" -- so on a fast run
        every genuine wait read as more backlog and only the line COUNT could end the drain. Half the
        measured pace separates them by two orders of magnitude, and it is self-scaling: a 92s run
        and a 0.4s run get the same test.

        ``_MAX_DRAINED_LINES`` then bounds only the case the gap test cannot decide -- a block that
        lands before any step has been measured, where there is no pace to compare a gap against.
        With a pace in hand the cap must NOT fire: a 30s upload lock on a 0.4s run queues ~75 lines,
        and cutting the drain at 64 admitted the rest of the burst at pipe speed. Once those
        outnumbered the clean intervals they took the median and the published pace collapsed to
        0.001s against a true 0.4s -- the same failure the drain exists to prevent, reintroduced by
        its own bound. Ending on the gap instead keeps the guarantee (a genuine wait always arrives,
        and its gap always clears half the pace) without capping how much backlog may be discarded.
        """
        if not self._draining_backlog:
            return False
        previous = self._last_arrival
        self._last_arrival = float(now)
        if previous is None or self._drained == 0:
            # the FIRST line after the block is the head of the burst, and its own gap spans the
            # block itself, so that gap says nothing about whether a backlog follows. it is kept
            # (it opens the new segment) and the question is deferred to the line after it.
            self._drained = 1
            return False
        gap = float(now) - previous
        measured = steady_state_step_seconds(self.intervals())
        if measured is not None:
            floor = min(_BLOCKING_CALL_THRESHOLD_S, measured / 2)
        elif self._drain_pipe_s is not None:
            # no pace yet, so the burst supplies its own yardstick. the drain's first confirmed
            # back-to-back gap IS this pipe's read speed, and a line the reader genuinely waited for
            # arrives orders of magnitude later than that. sizing against it keeps the test
            # scale-free where the absolute floor could not be: 1.0s is a claim about what a STEP
            # costs, and applying it to an ARRIVAL gap silently asserts every run is slower than a
            # second. on a 0.1s workload that made every real wait look like more backlog, so only
            # the line count could end the drain -- and the count ends it mid-burst, admitting the
            # rest at pipe speed. the multiple is wide because the two populations are far apart.
            floor = min(_BLOCKING_CALL_THRESHOLD_S, self._drain_pipe_s * _PIPE_GAP_MULTIPLE)
        else:
            floor = _BLOCKING_CALL_THRESHOLD_S
        if gap >= floor:
            # the reader waited for this one, so the backlog is gone and normal timing resumes.
            self._draining_backlog = False
            return False
        if self._drain_pipe_s is None or gap < self._drain_pipe_s:
            # the fastest confirmed arrival is the cleanest read of the pipe speed: a queued line
            # delayed by scheduler noise only ever reads slower than the pipe, never faster.
            self._drain_pipe_s = gap
        if (
            measured is not None
            and self._drain_pipe_s * _PIPE_GAP_MULTIPLE >= floor
            and self._drained >= self._MAX_DRAINED_LINES
        ):
            # these arrivals are not pipe-fast: the fastest one in this whole drain is within an
            # order of magnitude of the floor itself, so they look like real steps sitting under a
            # stale bar rather than buffered output. a real backlog is FINITE, so one that long
            # without a single genuinely fast read means the PACE is wrong, not that more output is
            # queued -- a run settling at 0.4s after a block taken at a 1.0s pace discarded 200 real
            # steps while still publishing 1.0s, with the ETA and wall warning overstated to match.
            #
            # the distinguishability guard is what keeps this from reintroducing the collapse the
            # bound itself once caused: a genuine 30s lock on a 0.4s run drains at 0.001s, which is
            # far below the floor, so its 75 queued lines are still discarded in full rather than
            # cut off at 64 and admitted as ordinary intervals.
            self._draining_backlog = False
            return False
        if (
            measured is None
            and self._drain_pipe_s * _PIPE_GAP_MULTIPLE >= _BLOCKING_CALL_THRESHOLD_S
            and self._drained >= self._MAX_DRAINED_LINES
        ):
            # neither test can end this drain: no pace was ever measured, and the arrivals are too
            # slow for their own gap to separate a pipe read from a real step -- a 0.2s "burst" is
            # indistinguishable from a 0.2s workload. the count is the only terminator left, and a
            # bounded skew beats publishing nothing for the rest of the run.
            #
            # it must NOT fire when the gap test can decide. a drain reading at 0.001s can be ended
            # by evidence, and cutting it at 64 instead admits the rest of the burst as ordinary
            # intervals -- the same collapse the bound was added to prevent, reintroduced by its own
            # terminator. a 0.9s upload on a 0.1s workload queues past the count long before the
            # first real wait arrives.
            self._draining_backlog = False
            return False
        if self._drained == 1:
            # this line arrived back-to-back, which is the confirmation the deferred head was
            # buffered too: it was read at drain speed rather than waited for, so it bounds no step.
            # left in place it becomes the start of a PARTIAL interval, closed by the first line the
            # reader genuinely waits for -- 26s against a true 92s. the median absorbs that from the
            # third interval on, but a block in a run's first steps has nothing to absorb it with.
            # dropping it lets that first waited-for line open the segment instead, which is what the
            # reprint path already does with its own baseline.
            self._times.clear()
        self._drained += 1
        return True

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
    guess would spend the user's attention on a number that never justified it. An allowance of zero
    is measured, not unmeasurable -- the deadline has simply elapsed -- and is the one case where
    this fires hardest.

    Both sides are measured against the SAME clock -- what remains to be trained versus what remains
    to be spent -- so the comparison stays valid on a resumed run, where absolute step counts and
    session wall time disagree.
    """
    projected = projected_remaining_seconds(
        step_intervals_s, current_step=current_step, total_steps=total_steps
    )
    if projected is None:
        return None
    if remaining_wall_seconds is None:
        return None
    # only None is unmeasurable. `_remaining_worker_wall_seconds` CLAMPS an elapsed deadline to 0.0,
    # so a zero allowance is a measurement -- the strongest one this warning can make -- and dropping
    # it silenced the row at exactly the moment the work definitively cannot fit: it fired at 1s
    # remaining and went quiet at 0s. a negative value cannot come from the clamp, so it can only be
    # a caller passing something this function has no basis to reason about.
    if remaining_wall_seconds < 0:
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
