"""Per-step timing: what a step costs, what that projects, and what it must never be measured from.

The incident these pin: a GRPO run whose first step took 515s against a 92s steady state. Reading
the run's pace off that first step predicted 26.9h -- past the 24h wall default -- for a run that
actually needed 4.9h. Every test here exists to keep the published number on the 92s side of that.
"""

from __future__ import annotations

import time

import pytest

from flash.cli.ui.heartbeat import _heartbeat_pairs, step_timing_pairs
from flash.engine.worker.train.core import step_timing

# the measured run, as the worker sees it, with training starting at t=0. verl prints a step's
# metric line when the step COMPLETES, so training start is not itself a step line: the first line
# lands 515s in, the second 92s after that.
_STEP0_LINE = 515.0
_STEP1_LINE = 607.0
_HORIZON = 188


def test_the_warmup_span_is_never_one_of_the_measured_intervals():
    """The 5.6x error, pinned at its source.

    The span from training start to the first step line holds engine init, the first weight sync and
    cache population. It is excluded structurally -- it falls BEFORE the first line rather than
    between two of them -- so no filter has to recognise it and none can be forgotten.
    """
    clock = step_timing.StepClock()
    clock.record(_STEP0_LINE)
    clock.record(_STEP1_LINE)

    assert clock.intervals() == [92.0]
    assert 515.0 not in clock.intervals()
    assert clock.step_seconds() == 92.0


def test_nothing_is_published_until_a_whole_step_has_been_measured():
    """One step line bounds no step, so there is no honest number to show yet.

    Absent rather than zero: a renderer tells "not measured yet" from a measurement only by the key
    being missing, and this is exactly the window where the only available reading is warmup.
    """
    clock = step_timing.StepClock()
    assert clock.step_seconds() is None
    assert (
        step_timing.step_timing_fields(
            clock, current_step=0, total_steps=_HORIZON, remaining_wall_seconds=86400.0
        )
        == {}
    )

    clock.record(_STEP0_LINE)
    assert clock.step_seconds() is None
    assert (
        step_timing.step_timing_fields(
            clock, current_step=0, total_steps=_HORIZON, remaining_wall_seconds=86400.0
        )
        == {}
    )


def test_the_projection_matches_the_run_that_actually_happened():
    """4.9h, not the 26.9h that step 0 predicted -- the whole point of the measurement."""
    clock = step_timing.StepClock()
    clock.record(_STEP0_LINE)
    clock.record(_STEP1_LINE)

    fields = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=24 * 3600
    )
    assert fields["step_duration_s"] == 92.0
    projected_hours = fields["projected_remaining_s"] / 3600
    assert 4.5 < projected_hours < 5.0, projected_hours
    # and the run fits, so nothing warns. warning on a healthy run is the failure that teaches
    # users to ignore the row.
    assert "wall_deadline_at_risk" not in fields


def test_a_step_that_cannot_finish_in_the_wall_allowance_warns():
    """The case the warning exists for: 188 steps at 600s is 31h against a 24h wall."""
    clock = step_timing.StepClock()
    clock.record(0.0)
    clock.record(600.0)

    fields = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=24 * 3600
    )
    assert fields["wall_deadline_at_risk"] is True
    assert fields["remaining_wall_s"] == 86400.0
    assert fields["projected_remaining_s"] / 3600 > 24


def test_the_warning_needs_both_sides_measured():
    """A run with no configured deadline still reports its pace, and never guesses at a risk."""
    clock = step_timing.StepClock()
    clock.record(0.0)
    clock.record(600.0)

    fields = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=None
    )
    assert fields["step_duration_s"] == 600.0
    assert "wall_deadline_at_risk" not in fields
    assert "remaining_wall_s" not in fields

    # an unknown horizon is the same: the pace is measured, the projection is not invented.
    unknown_horizon = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=0, remaining_wall_seconds=24 * 3600
    )
    assert unknown_horizon["step_duration_s"] == 600.0
    assert "projected_remaining_s" not in unknown_horizon
    assert "wall_deadline_at_risk" not in unknown_horizon


def test_only_the_steps_still_to_come_are_projected():
    """Completed steps are spent, whether or not they were slow.

    Folding them back in would re-charge the run for its own warmup -- the same error one level up.
    """
    intervals = [90.0, 90.0]
    early = step_timing.projected_remaining_seconds(intervals, current_step=10, total_steps=100)
    late = step_timing.projected_remaining_seconds(intervals, current_step=90, total_steps=100)
    assert early == 90.0 * 90
    assert late == 90.0 * 10
    # and a run past its horizon projects nothing left rather than a negative span.
    assert (
        step_timing.projected_remaining_seconds(intervals, current_step=120, total_steps=100) == 0.0
    )


def test_the_rate_resists_a_single_expensive_step():
    """A step that publishes a checkpoint is a real outlier, not the run's pace.

    The median is what makes the projection survive it; a mean would smear that one step across
    every remaining one.
    """
    clock = step_timing.StepClock()
    for arrival in (0.0, 92.0, 184.0, 900.0, 992.0, 1084.0):
        clock.record(arrival)
    assert clock.step_seconds() == 92.0


def test_a_replayed_step_number_is_not_timed_as_a_step():
    """verl reprints a step on a validation pass, and a resumed run replays its resume step.

    ``append_step_metrics`` dedupes exactly these repeats for the metrics backlog. No optimizer
    update happened between the two lines, so timing them would publish the validation pass as the
    cost of a step -- the same class of error as counting warmup.
    """
    clock = step_timing.StepClock()
    clock.record(0.0, 0)
    clock.record(92.0, 1)
    clock.record(140.0, 1)  # the reprint: 48s that no step paid for
    assert clock.intervals() == [92.0]
    assert clock.step_seconds() == 92.0

    # a caller that cannot identify the step still times every line: timing all is better than none.
    unnumbered = step_timing.StepClock()
    unnumbered.record(0.0)
    unnumbered.record(92.0)
    assert unnumbered.intervals() == [92.0]


def test_a_spurious_step_number_does_not_suppress_every_later_step():
    """The dedup skips an IMMEDIATE repeat, never every non-advancing number.

    SFT and OPD scan with ``run_verl_training``'s looser ``step:\\s*(\\d+)``, which also matches
    ``global_step:9`` inside a checkpoint path where RL's gate requires a word boundary. Under a
    monotonic rule one such spurious high number would silently drop every real step after it and
    freeze the published pace -- a wrong number that never corrects itself, which is worse than the
    reprint the dedup exists to exclude.
    """
    clock = step_timing.StepClock()
    for arrival, step in ((0.0, 1), (92.0, 9), (184.0, 2), (276.0, 3), (368.0, 4)):
        clock.record(arrival, step)
    # steps 2, 3 and 4 are real and must still be measured despite the bogus 9.
    assert clock.intervals() == [92.0, 92.0, 92.0, 92.0]
    assert clock.step_seconds() == 92.0

    # a resumed run replaying its resume step is still excluded: that repeat is immediate. the
    # replay's own arrival becomes the baseline, so step 6 measures 142-50 and not 142-0 -- merely
    # dropping the repeat would leave the stale timestamp and charge step 6 for the replay too.
    resumed = step_timing.StepClock()
    resumed.record(0.0, 5)
    resumed.record(50.0, 5)
    resumed.record(142.0, 6)
    assert resumed.intervals() == [92.0]


def test_a_reprint_becomes_the_baseline_for_the_step_after_it():
    """Dropping a repeat is not enough -- the NEXT step must not be charged for it either.

    Discarding the repeat and keeping the earlier timestamp leaves the following interval spanning
    the whole replay: the pre-repeat line to the next real step. The repeat's own arrival is when
    that step actually began, so it becomes the baseline instead.
    """
    resumed = step_timing.StepClock()
    resumed.record(0.0, 5)
    resumed.record(50.0, 5)  # the replay: 50s of resume init, no optimizer update
    resumed.record(142.0, 6)
    assert resumed.intervals() == [92.0]  # 142-50, not 142-0

    # a mid-run validation reprint behaves the same, and the steps around it stay measurable.
    validated = step_timing.StepClock()
    for arrival, step in ((0.0, 1), (92.0, 2), (140.0, 2), (232.0, 3), (324.0, 4)):
        validated.record(arrival, step)
    assert validated.intervals() == [92.0, 92.0, 92.0]
    assert validated.step_seconds() == 92.0


def test_blocking_work_leaves_no_baseline_the_way_a_reprint_does():
    """The two breaks are not the same, and only one carries a timestamp forward.

    A reprint marks a boundary the next step starts from. Blocking work does not: the step was
    already running while we blocked, so no timestamp in that span bounds it and seeding one would
    invent a start time. The span is simply dropped.
    """
    clock = step_timing.StepClock()
    clock.record(0.0, 0)
    clock.note_blocking_work()
    clock.record(300.0, 1)  # 92s of step plus a slow upload
    clock.record(392.0, 2)
    assert clock.intervals() == [92.0]  # only the clean span, no 300.0 and no invented start


def test_a_span_containing_blocking_work_is_not_timed():
    """The stdout consumer timestamps a step line when it READS it.

    RL's first metric line is followed by a forced heartbeat that retries until it commits, which
    can hold that loop for minutes. The span is dropped rather than published, and the steps around
    it still measure -- one lost interval on a run that has many.
    """
    clock = step_timing.StepClock()
    clock.record(0.0, 0)
    clock.note_blocking_work()
    clock.record(300.0, 1)  # 92s of step plus a slow upload retry
    clock.record(392.0, 2)
    assert clock.intervals() == [92.0]
    assert clock.step_seconds() == 92.0


def test_breaking_on_every_step_would_leave_nothing_measured():
    """Why SFT and OPD break on a MEASURED block, not on every step.

    Their ``on_step`` runs once per step inside the stdout loop. Declaring a block unconditionally
    splits the record at every line, so no segment ever holds two -- and a trainer that published a
    pace would go silent.
    """
    every_step = step_timing.StepClock()
    for index in range(10):
        every_step.record(float(index) * 92.0, index)
        every_step.note_blocking_work()
    assert every_step.intervals() == []
    assert every_step.step_seconds() is None

    # breaking only on the one call that actually blocked keeps the rest measurable.
    occasional = step_timing.StepClock()
    for index in range(10):
        occasional.record(float(index) * 92.0, index)
        occasional.note_if_blocked(30.0 if index == 4 else 0.0002)
    assert occasional.step_seconds() == 92.0
    assert len(occasional.intervals()) == 8  # nine gaps, less the one that held the upload


def test_a_blocked_call_is_detected_by_duration_not_by_its_result():
    """``heartbeat()`` returning "not committed" does not mean it returned quickly.

    A throttled skip returns in microseconds; a failed upload, or one that waited out the 30s upload
    lock, blocks the reader and ALSO reports not-committed. Keying the break on the result would skip
    it in exactly the case that distorts an interval, and a flaky HF path repeats that every step
    until the inflated span is the median rather than an outlier the median drops.
    """
    clock = step_timing.StepClock()
    # a throttled no-op must not split the segment...
    clock.record(0.0, 1)
    clock.note_if_blocked(0.0001)
    clock.record(92.0, 2)
    assert clock.intervals() == [92.0]

    # ...while a failed upload that waited on the lock must, even though it committed nothing.
    clock.note_if_blocked(30.0)
    clock.record(214.0, 3)  # 92s of step plus the 30s wait
    clock.record(306.0, 4)
    assert clock.intervals() == [92.0, 92.0]
    assert clock.step_seconds() == 92.0


def test_the_projection_amortizes_saves_that_the_pace_excludes():
    """The two numbers answer different questions and need different estimators.

    ``step_duration_s`` is what a step costs, so it stays median and ignores the save. The
    projection is a SUM of future steps, and saves are real recurring work on the save schedule --
    under-counting them is worst for the wall warning, whose whole job is spotting a run that will
    not fit.
    """
    intervals = [92.0] * 9 + [400.0]  # one save every ten steps
    assert step_timing.steady_state_step_seconds(intervals) == 92.0

    projected = step_timing.projected_remaining_seconds(intervals, current_step=0, total_steps=100)
    assert projected is not None
    # the median alone would claim 9200s and never charge for the nine remaining saves.
    assert projected > 92.0 * 100
    assert projected == pytest.approx(12280.0)

    # but one pathological span cannot double the projection the way raw warmup once did.
    stalled = [92.0] * 9 + [9000.0]
    capped = step_timing.projected_remaining_seconds(stalled, current_step=0, total_steps=100)
    assert capped == pytest.approx(92.0 * 2 * 100)


def test_the_retained_window_is_bounded_and_keeps_the_recent_steps():
    """A long run must not grow this without bound, and must track its CURRENT rate.

    Keeping the oldest instead would report an hour-old pace for the rest of the run.
    """
    clock = step_timing.StepClock()
    for index in range(5000):
        clock.record(float(index) * 10.0)
    assert len(clock.intervals()) < 300
    assert clock.step_seconds() == 10.0

    # the run's rate changes. once the slow steps fill the window, the reading is theirs alone --
    # an unbounded record would still be averaging in the fast first hour.
    for offset in range(1, clock._RETAINED_STEP_LINES + 1):
        clock.record(50000.0 + offset * 200.0)
    assert clock.step_seconds() == 200.0


def test_the_window_stays_bounded_when_blocking_work_splits_it():
    """Splitting on blocking work must not become a way to retain unbounded state.

    A long run can block many times, and each split leaves a closed segment behind. The cap covers
    every segment together rather than each one, so the total is what a bounded window promises.
    """
    clock = step_timing.StepClock()
    for index in range(20000):
        if index % 7 == 0:
            clock.note_blocking_work()
        clock.record(float(index) * 10.0, index)

    retained = sum(len(segment) for segment in clock._segments) + len(clock._times)
    assert retained <= clock._RETAINED_STEP_LINES
    assert clock.step_seconds() == 10.0

    # and the degenerate case -- blocking before EVERY line, so no segment ever bounds a step --
    # publishes nothing rather than a wrong number, and still keeps nothing around.
    always_blocked = step_timing.StepClock()
    for index in range(500):
        always_blocked.note_blocking_work()
        always_blocked.record(float(index) * 10.0, index)
    assert always_blocked.intervals() == []
    assert always_blocked.step_seconds() is None


def test_the_panel_shows_nothing_until_a_step_is_measured():
    """No row at all rather than a placeholder: step 0 is precisely when there is nothing true."""
    heartbeat = {"stage": "rl_step", "step": 0, "ts": time.time()}
    assert step_timing_pairs(heartbeat, running=True) == []


def test_the_panel_reports_pace_and_the_projection():
    heartbeat = {
        "stage": "rl_step",
        "step": 12,
        "ts": time.time(),
        "step_duration_s": 92.0,
        "projected_remaining_s": 17204.0,
    }
    rows = dict(step_timing_pairs(heartbeat, running=True))
    assert "92s/step" in rows["pace"]
    assert "4.8h" in rows["pace"]
    assert "wall limit" not in rows


def test_a_per_step_cost_keeps_the_precision_a_comparison_needs():
    """92s and 149s must not both render "2m": that hides a 60% difference in what the run costs."""

    def pace(seconds: float) -> str:
        heartbeat = {"stage": "rl_step", "step": 3, "ts": time.time(), "step_duration_s": seconds}
        return dict(step_timing_pairs(heartbeat, running=True))["pace"]

    assert pace(92.0) != pace(149.0)
    assert "92s" in pace(92.0)
    assert "149s" in pace(149.0)

    # a sub-second pace keeps a decimal for the same reason: rounding a measured step to "0s/step"
    # reads as no measurement at all, beside a projection that is plainly nonzero.
    assert "0.4s" in pace(0.4)
    assert not pace(0.4).startswith("0s")


def test_the_panel_warns_when_the_rate_will_not_fit_the_wall():
    heartbeat = {
        "stage": "rl_step",
        "step": 12,
        "ts": time.time(),
        "step_duration_s": 600.0,
        "projected_remaining_s": 112200.0,
        "remaining_wall_s": 86400.0,
        "wall_deadline_at_risk": True,
    }
    rows = dict(step_timing_pairs(heartbeat, running=True))
    warning = rows["wall limit"]
    assert "24.0h" in warning
    # says what survives, so the reading is actionable rather than merely alarming. checkpoints
    # already published at the run's save steps are NOT lost when the wall cuts training off.
    assert "checkpoints" in warning
    assert "relaunch" in warning


def test_a_finished_run_shows_no_projection():
    """The real duration is on the record by then; projecting over it would be a worse answer."""
    heartbeat = {
        "stage": "rl_step",
        "step": 188,
        "ts": time.time(),
        "step_duration_s": 92.0,
        "projected_remaining_s": 0.0,
    }
    assert step_timing_pairs(heartbeat, running=False) == []


def test_a_junk_duration_is_ignored_rather_than_rendered():
    """The field crosses a network boundary from the worker, so the panel cannot assume it is sane."""
    for bad in (0, -5, "fast", None, True, float("inf"), float("nan")):
        heartbeat = {"stage": "rl_step", "step": 3, "ts": time.time(), "step_duration_s": bad}
        assert step_timing_pairs(heartbeat, running=True) == [], bad


def test_a_superseded_attempts_pace_is_not_shown_as_this_runs():
    """That rate was measured on a worker that no longer exists, possibly on different hardware."""
    status = {
        "state": "running",
        "remote": {"attempt": 2},
        "last_heartbeat": {
            "stage": "rl_step",
            "step": 12,
            "attempt": 1,
            "ts": time.time() - 30,
            "step_duration_s": 92.0,
            "projected_remaining_s": 17204.0,
        },
    }
    assert "pace" not in dict(_heartbeat_pairs(status))

    live = {**status, "last_heartbeat": {**status["last_heartbeat"], "attempt": 2}}
    assert "pace" in dict(_heartbeat_pairs(live))


def test_the_checkpoint_heartbeat_carries_the_pace_it_would_otherwise_blank():
    """A save must not erase the measured pace from live status.

    An upload REPLACES the published snapshot, and ``checkpoint_uploaded`` is unthrottled while
    arming the 900s throttle the step stages share. Publishing it without timing would blank the
    pace and then block every ping that could restore it, for up to the throttle interval after
    every save -- on exactly the long runs this measurement exists for.
    """
    from flash.engine.worker.io import heartbeat as worker_heartbeat

    assert worker_heartbeat.step_timing_fields_now() == {}
    measured = {"step_duration_s": 92.0, "projected_remaining_s": 17204.0}
    with worker_heartbeat.publishing_step_timing(lambda: measured):
        assert worker_heartbeat.step_timing_fields_now() == measured
    # scoped: a finished trainer's clock must not keep answering for a later stage.
    assert worker_heartbeat.step_timing_fields_now() == {}
    assert worker_heartbeat.LATEST_STEP_TIMING_FIELDS == []


def test_an_observability_read_never_breaks_a_checkpoint_upload():
    """This runs inside the upload's success path, so it fails closed to no fields.

    A checkpoint that uploaded correctly must not be reported as failed because a pace could not be
    read, and a malformed value must not be splatted into the heartbeat payload.
    """
    from flash.engine.worker.io import heartbeat as worker_heartbeat

    def raises() -> dict:
        raise RuntimeError("clock exploded")

    with worker_heartbeat.publishing_step_timing(raises):
        assert worker_heartbeat.step_timing_fields_now() == {}

    with worker_heartbeat.publishing_step_timing(lambda: "not a dict"):
        assert worker_heartbeat.step_timing_fields_now() == {}

    assert worker_heartbeat.LATEST_STEP_TIMING_FIELDS == []


def test_the_upload_actually_sends_the_pace_on_the_checkpoint_ping(monkeypatch, tmp_path):
    """The wiring, driven through the real upload rather than the registry alone.

    ``upload_resume_checkpoint`` is where the blanking would happen, so a test of the registry in
    isolation would still pass with the call site publishing nothing.
    """
    import flash.engine.worker as worker
    from flash.engine.worker.io import heartbeat as worker_heartbeat
    from flash.engine.worker.io import hf as worker_hf

    sent: list[tuple[str, dict]] = []

    class Api:
        def upload_folder(self, **_kwargs):
            return None

        def list_repo_files(self, **_kwargs):
            return []

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker, "hf_api", lambda: Api())
    monkeypatch.setattr(worker, "heartbeat", lambda stage, **kw: sent.append((stage, kw)))

    measured = {"step_duration_s": 92.0, "projected_remaining_s": 17204.0}
    with worker_heartbeat.publishing_step_timing(lambda: measured):
        assert worker_hf.upload_resume_checkpoint(4, str(tmp_path))

    uploaded = [kw for stage, kw in sent if stage == "checkpoint_uploaded"]
    assert uploaded, [stage for stage, _ in sent]
    assert uploaded[0]["step_duration_s"] == 92.0
    assert uploaded[0]["projected_remaining_s"] == 17204.0


def test_every_step_timestamp_comes_from_a_clock_that_cannot_jump():
    """These timestamps are only ever read as differences, so a wall clock is the wrong source.

    An NTP correction or a VM resync mid-run moves ``time.time()``: backwards produces a negative
    span that is silently dropped, forwards inflates the pace, the ETA and the wall-limit warning at
    once -- and a long training run is exactly where a resync has time to happen.

    Asserted at the call sites because the clock takes a float and cannot tell which one produced
    it. Read off disk rather than through ``inspect``: these runner modules are imported via their
    parents and importing one directly raises on a circular import.

    ``_remaining_worker_wall_seconds`` is deliberately not covered here -- its deadline is an
    absolute instant supplied from outside the process, so it is the one comparison that needs the
    wall clock.
    """
    import ast
    from pathlib import Path

    import flash

    runners = ("rl_train_runner.py", "sft_train_runner.py", "opd_train_runner.py")
    worker_dir = Path(flash.__file__).parent / "engine" / "worker"
    checked = 0
    for name in runners:
        path = worker_dir / name
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"record", "note_if_blocked"} or not node.args:
                continue
            # the receiver has to be the step clock: `.record(` alone also matches unrelated calls.
            receiver = node.func.value
            named = (
                receiver.attr
                if isinstance(receiver, ast.Attribute)
                else getattr(receiver, "id", "")
            )
            if named != "step_clock":
                continue
            timestamp = ast.dump(node.args[0])
            assert "monotonic" in timestamp, (
                f"{name}:{node.lineno} times a step off a clock that can jump: {timestamp}"
            )
            checked += 1
    # every trainer records and two of them also time their heartbeat; a passing assertion loop that
    # visited nothing would be the failure this guards against.
    assert checked == 5, checked
