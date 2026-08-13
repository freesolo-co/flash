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


def test_an_elapsed_deadline_still_warns_rather_than_going_quiet():
    """A zero allowance is measured, not unmeasurable, and is the strongest case this warning has.

    ``_remaining_worker_wall_seconds`` clamps an elapsed deadline to 0.0, so treating <= 0 as
    "unknown" silenced the row at exactly the moment the work definitively cannot fit: it fired at
    1s remaining and went quiet at 0s, with hours of projected work still to run.
    """
    clock = step_timing.StepClock()
    clock.record(0.0)
    clock.record(600.0)

    elapsed = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=0.0
    )
    assert elapsed["wall_deadline_at_risk"] is True
    assert elapsed["remaining_wall_s"] == 0.0
    assert elapsed["projected_remaining_s"] > 0

    # one second earlier must not be the only moment it warns.
    nearly = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=1.0
    )
    assert nearly["wall_deadline_at_risk"] is True


def test_a_negative_allowance_is_not_a_measurement():
    """The clamp cannot produce one, so it can only be a caller with nothing meaningful to say."""
    clock = step_timing.StepClock()
    clock.record(0.0)
    clock.record(600.0)

    fields = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=_HORIZON, remaining_wall_seconds=-5.0
    )
    assert fields["step_duration_s"] == 600.0
    assert "wall_deadline_at_risk" not in fields
    assert "remaining_wall_s" not in fields


def test_the_panel_omits_a_zero_wall_instead_of_naming_it():
    """The row must not read "against 0s of wall time left" -- it drops the clause and warns anyway."""
    pairs = dict(
        step_timing_pairs(
            {
                "step_duration_s": 92.0,
                "projected_remaining_s": 16376.0,
                "remaining_wall_s": 0.0,
                "wall_deadline_at_risk": True,
            },
            running=True,
        )
    )

    assert "0s of wall time left" not in pairs["wall limit"]
    # a zero allowance can never "only just fit", so it takes the cutoff wording.
    assert "expected to be cut off" in pairs["wall limit"]


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
    # every trainer records, and SFT and OPD each time TWO uploads: the one on the step line and the
    # one the child stream fires from a non-step line. a passing assertion loop that visited nothing
    # would be the failure this guards against.
    assert checked == 7, checked


def test_a_failed_optional_checkpoint_does_not_blank_the_pace_either(monkeypatch, tmp_path):
    """The failure ping is unthrottled too, so it arms the same throttle the success ping does.

    An optional checkpoint that exhausts its retries does not stop the run: training continues, and
    a payload without the timing would blank the pace of a run that is still going -- then block
    every step ping that could restore it for up to the throttle interval.
    """
    import flash.engine.worker as worker
    from flash.engine.worker.io import heartbeat as worker_heartbeat
    from flash.engine.worker.io import hf as worker_hf

    sent: list[tuple[str, dict]] = []

    class Api:
        def upload_folder(self, **_kwargs):
            raise ConnectionError("hf unavailable")

        def list_repo_files(self, **_kwargs):
            return []

    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker, "hf_api", lambda: Api())
    monkeypatch.setattr(worker, "heartbeat", lambda stage, **kw: sent.append((stage, kw)))
    monkeypatch.setattr(worker_hf, "_CKPT_UPLOAD_BACKOFF_S", 0.0)

    measured = {"step_duration_s": 92.0, "projected_remaining_s": 17204.0}
    with worker_heartbeat.publishing_step_timing(lambda: measured):
        assert not worker_hf.upload_resume_checkpoint(4, str(tmp_path))

    failed = [kw for stage, kw in sent if stage == "checkpoint_upload_failed"]
    assert failed, [stage for stage, _ in sent]
    assert failed[0]["step_duration_s"] == 92.0
    assert failed[0]["projected_remaining_s"] == 17204.0


def test_a_checkpoint_path_is_not_timed_as_a_step(tmp_path):
    """SFT and OPD used to scan stdout with a looser pattern than the gate their siblings use.

    ``step:\\s*(\\d+)`` also matches ``global_step:9`` inside a checkpoint path and reads step 1 out
    of ``timing/step:1.25``, so a save or a metrics line could fabricate a step no optimizer update
    produced -- splitting one real interval into two shorter samples and biasing the pace, the ETA
    and the wall-risk warning low. Both now gate on ``verl_step_number``.
    """
    import os
    import sys

    from flash.engine.worker import backend_common

    lines = [
        "step:1 - train/loss:0.5",
        "Saving checkpoint to /ckpt/global_step:9/actor",
        "timing/step:1.25",
        "val/global_step:12",
        "step:2 - train/loss:0.4",
    ]
    script = "import sys\n" + "".join(f"print({line!r}, flush=True)\n" for line in lines)

    seen: list[int] = []
    code = backend_common.run_verl_training(
        [sys.executable, "-c", script], env=dict(os.environ), on_step=seen.append
    )
    assert code == 0
    assert seen == [1, 2], seen


def _replay(clock, timeline):
    """Drive the clock the way a trainer's on_step does: record, heartbeat, then note the duration."""
    for arrival, step, heartbeat_s in timeline:
        clock.record(arrival, step)
        clock.note_if_blocked(heartbeat_s)


def test_the_backlog_a_block_leaves_behind_is_not_timed_as_fast_steps():
    """Breaking the one blocked span is not enough -- the burst behind it is the bigger error.

    The child keeps training while the reader is blocked, so the steps it completes meanwhile queue
    in the pipe and are then read back-to-back at pipe speed. Those arrivals measure the drain, not
    the steps. Left in, they do not merely add noise: once the burst outnumbers the real lines it
    takes the median, and the published pace collapsed to 0.10s against a true 92s -- a 920x
    under-report that makes the ETA and the wall warning useless in the direction that matters.
    """
    clock = step_timing.StepClock()
    _replay(
        clock,
        [
            (0.0, 1, 0.001),
            (92.0, 2, 0.001),
            (184.0, 3, 900.0),  # a 900s block; the child completes steps 4-12 meanwhile
            *[(1084.0 + i * 0.1, 4 + i, 0.001) for i in range(9)],  # the drain, at pipe speed
            (1176.9, 13, 0.001),  # the first line the reader actually waited for
            (1268.9, 14, 0.001),
        ],
    )

    assert clock.step_seconds() == 92.0
    assert not [gap for gap in clock.intervals() if gap < 1.0], clock.intervals()


def test_the_head_of_the_burst_cannot_end_the_drain_by_its_own_gap():
    """The first line after a block spans the block itself, so its gap proves nothing.

    Reading it as "the reader waited, so there is no backlog" ends the drain before it starts and
    admits the entire burst -- which is exactly how the first attempt at this fix failed.
    """
    clock = step_timing.StepClock()
    _replay(
        clock,
        [
            (0.0, 1, 0.001),
            (92.0, 2, 900.0),
            (992.0, 3, 0.001),  # head of the burst: a 900s gap, but a backlog follows it
            (992.1, 4, 0.001),
            (992.2, 5, 0.001),
            (1084.2, 6, 0.001),
            (1176.2, 7, 0.001),
        ],
    )

    assert clock.step_seconds() == 92.0
    assert not [gap for gap in clock.intervals() if gap < 1.0], clock.intervals()


def test_a_confirmed_backlog_discards_the_head_it_deferred_judgement_on():
    """The burst head bounds no step either, once a back-to-back line proves it was buffered.

    Its own gap spans the block, so the drain cannot judge it on arrival and keeps it. But when the
    next line arrives back-to-back, that is the confirmation the head was read at drain speed rather
    than waited for. Left in place it starts a PARTIAL interval, closed by the first line the reader
    genuinely waits for: 26s against a true 92s.

    The median absorbs that from the third interval on, so this only bites a run that blocks in its
    first steps -- which has nothing to absorb it with, and is exactly when an operator is deciding
    whether to let hours of GPU time run. It understates the pace, so the wall warning under-fires.
    """
    clock = step_timing.StepClock()
    _replay(
        clock,
        [
            (0.0, 1, 0.001),
            (92.0, 2, 0.001),
            (184.0, 3, 60.0),  # blocked; the child completes step 4 meanwhile
            (250.0, 4, 0.001),  # head of the burst: kept, pending confirmation
            (250.1, 5, 0.001),  # back-to-back -- confirms the head was buffered too
            (276.0, 6, 0.001),  # the first line genuinely waited for: opens the new segment
            (368.0, 7, 0.001),
        ],
    )

    # 276.0 -> 368.0 only. the 26s partial (250.1 -> 276.0) must not be in here.
    assert clock.intervals() == [92.0, 92.0, 92.0]
    assert clock.step_seconds() == 92.0


def test_an_early_block_cannot_understate_the_pace_with_nothing_to_average_it_out():
    """The same defect where it actually hurts: too few intervals for the median to hide it.

    With one clean interval on either side the bad sample is a third of the sample set, and the
    median lands halfway between wrong and right rather than on either.
    """
    clock = step_timing.StepClock()
    _replay(
        clock,
        [
            (0.0, 1, 0.001),
            (92.0, 2, 60.0),
            (158.0, 3, 0.001),  # burst head
            (158.1, 4, 0.001),  # confirms the backlog
            (184.0, 5, 0.001),  # partial span, 25.9s, if the head survives
            (276.0, 6, 0.001),
        ],
    )

    # the median is asserted at TWO intervals on purpose. with three it lands on 92.0 even when the
    # partial is present, so a median-only assertion here passes against the bug it is meant to
    # catch; the interval set is what actually distinguishes the two.
    assert clock.intervals() == [92.0, 92.0]
    assert clock.step_seconds() == 92.0


def test_a_block_that_buffered_nothing_costs_one_interval_not_a_quota():
    """The drain ends on evidence, not a fixed count, so a quiet block is cheap.

    Discarding a fixed number of lines after every block would throw away real steps on the common
    case where the block was short enough to buffer nothing.
    """
    clock = step_timing.StepClock()
    _replay(
        clock,
        [
            (0.0, 1, 0.001),
            (92.0, 2, 5.0),  # blocked 5s: far too short to buffer a 92s step
            (184.0, 3, 0.001),
            (276.0, 4, 0.001),
            (368.0, 5, 0.001),
        ],
    )

    # the blocked span itself is dropped; every step after it is measured normally.
    assert clock.intervals() == [92.0, 92.0, 92.0]
    assert clock.step_seconds() == 92.0


def test_the_drain_always_ends_so_a_fast_run_is_never_silenced():
    """A run whose real steps are faster than the block threshold must still publish a pace.

    Without the bound such a run looks like one unending burst and reports nothing at all for the
    rest of its life, which is worse than a slightly skewed median.
    """
    clock = step_timing.StepClock()
    fast = [(0.0, 1, 0.001), (0.2, 2, 900.0)]
    # every subsequent line arrives faster than the blocking threshold, so nothing ever looks
    # like a line the reader waited for.
    fast += [(0.4 + i * 0.2, 3 + i, 0.001) for i in range(400)]
    _replay(clock, fast)

    measured = clock.step_seconds()
    assert measured is not None, "the bound must let a fast run resume publishing"
    assert measured == pytest.approx(0.2, abs=0.01), measured
    # asserted on the drain itself, not only on the published number: without the bound the clock
    # stays in the drain forever and keeps discarding lines, and a run long enough to leave two
    # stragglers outside it would still answer 0.2 while measuring almost nothing.
    assert not clock._draining_backlog, "the drain must end rather than run for the whole run"
    assert clock._drained <= step_timing.StepClock._MAX_DRAINED_LINES
    # the retained window is what bounds this, not the drain: 256 timestamps bound 255 spans.
    assert len(clock.intervals()) == 255, len(clock.intervals())


def test_a_thin_margin_is_not_reported_as_a_certain_cutoff():
    """The 90% trigger covers two situations, and only one of them is a cutoff.

    At 95 minutes projected against 100 remaining the warning fires, but the measured horizon still
    fits -- what the margin threatens is the final checkpoint upload, which runs after the last step.
    Telling that operator training "is expected to be cut off" asserts something the measurement does
    not show, and a row that overstates its case on runs that go on to finish is how it gets ignored
    on the runs that do not.
    """
    thin = {
        "step_duration_s": 92.0,
        "projected_remaining_s": 95 * 60,
        "remaining_wall_s": 100 * 60,
        "wall_deadline_at_risk": True,
        "from_current_attempt": True,
    }
    warning = dict(step_timing_pairs(thin, running=True))["wall limit"]
    assert "only just fit" in warning
    assert "expected to be cut off" not in warning
    assert "final checkpoint upload" in warning


def test_a_projection_that_really_overruns_still_says_so():
    """The original wording has to survive for the case it was written for: 31h against a 24h wall."""
    over = {
        "step_duration_s": 600.0,
        "projected_remaining_s": 31 * 3600,
        "remaining_wall_s": 24 * 3600,
        "wall_deadline_at_risk": True,
        "from_current_attempt": True,
    }
    warning = dict(step_timing_pairs(over, running=True))["wall limit"]
    assert "do not fit" in warning
    assert "expected to be cut off" in warning
    assert "only just fit" not in warning


def test_an_unnamed_wall_falls_back_to_the_stronger_warning():
    """Without both numbers the panel cannot prove the projection fits, so it must not claim it does.

    The worker sends ``remaining_wall_s`` only alongside the flag, but a payload can lose it (an
    older worker, a junk value rejected by _finite_positive). Softening the warning on missing
    evidence would be the wrong default: the flag itself still means the run is at risk.
    """
    no_wall = {
        "step_duration_s": 92.0,
        "projected_remaining_s": 95 * 60,
        "wall_deadline_at_risk": True,
        "from_current_attempt": True,
    }
    warning = dict(step_timing_pairs(no_wall, running=True))["wall limit"]
    assert "do not fit" in warning
    assert "only just fit" not in warning


def test_a_block_discards_a_pending_replay_baseline():
    """The two breaks can overlap, and the block has to win.

    A validation reprint sets its own arrival as the baseline the next step starts from. If the
    heartbeat issued for that same callback then blocks, that baseline is a pre-block instant:
    carrying it forward starts the next segment BEFORE the upload and so measures the very stall
    note_blocking_work exists to exclude. Seen end to end it published 496.5s/step against a
    true 92.0 -- worse than not measuring at all, because it looks like a measurement.
    """
    clock = step_timing.StepClock()
    clock.record(100.0, 5)
    clock.record(192.0, 6)
    clock.record(200.0, 6)  # validation reprint -> baseline 200.0
    assert clock._pending_baseline == 200.0
    clock.note_if_blocked(900.0)  # the heartbeat for that callback stalled
    assert clock._pending_baseline is None
    clock.record(1101.0, 7)
    assert clock.intervals() == [92.0]  # not [92.0, 901.0]
    assert clock.step_seconds() == 92.0


def test_the_rl_projection_reads_the_step_from_the_shared_gate():
    """The projection's current step decides how many steps remain, so a bad one is not cosmetic.

    _execute_rl_child used to scan with a looser `step:\\s*(\\d+)` of its own, which reads 1 out of
    `timing/step:1.25` and 9 out of a `global_step:9` checkpoint path. After a real step 20 that
    resets the published step to 1, and the next liveness or checkpoint ping carries a remaining
    time computed for 99 steps instead of 80 -- with a wall-limit warning that fires off it.
    """
    import inspect

    from flash.engine.worker import rl_train
    from flash.engine.worker.verl.child_io import verl_step_number

    src = " ".join(inspect.getsource(rl_train._execute_rl_child).split())
    assert "parsed_step = verl_step_number(line)" in src
    assert 'progress["step"] = parsed_step' in src
    assert "step_re" not in src, "a second, looser step scan is back"

    # and the gate itself rejects the shapes that caused the reset.
    assert verl_step_number("step:20 - critic/rewards/mean:0.5") == 20
    for noise in (
        "(TaskRunner pid=1) timing/step:1.25",
        "val/global_step:12 something",
        "saved to /ckpt/global_step:9/actor",
    ):
        assert verl_step_number(noise) is None, noise

    # the projection is what a contaminated step corrupts: same clock, same pace, wrong step.
    clock = step_timing.StepClock()
    for i in range(6):
        clock.record(float(i) * 92.0)
    truth = step_timing.step_timing_fields(
        clock, current_step=20, total_steps=100, remaining_wall_seconds=10_000.0
    )
    contaminated = step_timing.step_timing_fields(
        clock, current_step=1, total_steps=100, remaining_wall_seconds=10_000.0
    )
    assert truth["projected_remaining_s"] == 80 * 92.0
    assert not truth.get("wall_deadline_at_risk")
    assert contaminated["projected_remaining_s"] == 99 * 92.0
    assert contaminated["wall_deadline_at_risk"] is True  # the false warning


def test_the_child_stream_heartbeat_is_timed_like_the_step_one():
    """Both uploads run inside run_verl_training's reader loop, so both defer the next timestamp.

    The child ping fires from a NON-step line, which is exactly why it needs its own guard: the
    on_step path never sees this call, so its note_if_blocked cannot cover it. Without one, a
    900s stall on this ping lands whole inside the next interval and is published as a step cost.
    """
    import ast
    from pathlib import Path

    import flash

    # read off disk, not through inspect: importing these runners directly raises on a circular
    # import (see test_every_step_timestamp_comes_from_a_clock_that_cannot_jump).
    worker_dir = Path(flash.__file__).parent / "engine" / "worker"
    for name in ("sft_train_runner.py", "opd_train_runner.py"):
        body = None
        for node in ast.walk(ast.parse((worker_dir / name).read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == "child_heartbeat":
                body = " ".join(ast.unparse(node).split())
        assert body is not None, f"{name} has no child_heartbeat"
        assert "started = time.monotonic()" in body, name
        assert "note_if_blocked(time.monotonic() - started)" in body, name

    # the guard's effect: the stalled span is dropped rather than published as what a step costs.
    clock = step_timing.StepClock()
    clock.record(0.0, 1)
    clock.record(92.0, 2)
    clock.note_if_blocked(900.0)  # the child ping blocked on a non-step line
    clock.record(1084.0, 3)  # 900s upload + 92s step
    clock.record(1176.0, 4)
    assert clock.intervals() == [92.0, 92.0]
    assert clock.step_seconds() == 92.0


def test_the_wall_warning_never_recommends_an_inert_knob():
    """`max_examples` does not shorten a `max_steps` run, so advising it buys a doomed relaunch.

    `resolve_update_horizon` returns the configured `max_steps` whenever it is positive and ignores
    the derived example-based horizon entirely. An operator who follows advice to cut `max_examples`
    on such a run pays for a relaunch projected to hit the same wall cutoff. The panel cannot tell
    the two configurations apart -- the heartbeat carries no horizon provenance -- so the guidance
    has to name the knob that shortens the run under EITHER configuration.
    """
    from flash.engine.plan.steps import resolve_update_horizon

    # the premise, asserted rather than assumed: max_steps wins and max_examples is inert.
    assert resolve_update_horizon(1000, 40) == 40
    assert resolve_update_horizon(1000, None) == 1000

    for remaining, wall in ((95 * 60, 100 * 60), (300 * 60, 100 * 60)):
        beat = {
            "step_duration_s": 92.0,
            "projected_remaining_s": remaining,
            "remaining_wall_s": wall,
            "wall_deadline_at_risk": True,
            "from_current_attempt": True,
        }
        warning = dict(step_timing_pairs(beat, running=True))["wall limit"]
        assert "max_examples" not in warning, warning
        assert "step horizon" in warning, warning


def test_a_long_upload_keeps_publishing_the_measured_pace():
    """The upload daemon commits real heartbeats for the whole save, not just one at the end.

    `checkpoint_uploading` is a keepalive liveness wrap, so it publishes every tick while the upload
    runs -- and an upload REPLACES the published snapshot. Without the timing fields those ticks
    blank pace, ETA and wall-risk for the entire duration of a save, which on a large model is
    minutes, and the throttle they arm holds that blank state afterwards.
    """
    import ast
    from pathlib import Path

    import flash

    source = (Path(flash.__file__).parent / "engine" / "worker" / "io" / "hf.py").read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "liveness_heartbeat"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "checkpoint_uploading"
    ]
    assert len(calls) == 1, f"expected one checkpoint_uploading wrap, found {len(calls)}"
    fields = {kw.arg: kw.value for kw in calls[0].keywords}.get("fields")
    assert fields is not None, "the upload daemon publishes without step timing"
    assert isinstance(fields, ast.Name), ast.dump(fields)
    assert fields.id == "_step_timing_fields_now", fields.id


def test_timing_stays_registered_through_the_final_checkpoint_drain():
    """The last save is published during teardown, after the training block has exited.

    Each trainer drains its uploader/watcher in a `finally` that runs AFTER the step-heartbeat
    block. That drain uploads the final checkpoint, whose `checkpoint_uploaded` ping is unthrottled
    and reads the timing registry. If the registration ends with the training block, the run's last
    save publishes with no pace and arms the 900s throttle behind it -- so the final measurement is
    blanked on the ping most likely to be the last one anybody reads.
    """
    import ast
    from pathlib import Path

    import flash

    worker = Path(flash.__file__).parent / "engine" / "worker"
    cases = (
        ("rl_train.py", "run_rl_train", "publishing_step_timing"),
        ("sft_train.py", "run_sft_train", "publishing_step_timing"),
        ("opd_train_runner.py", "_run_child", "publishing_step_timing"),
    )
    for filename, funcname, registrar in cases:
        tree = ast.parse((worker / filename).read_text())
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == funcname),
            None,
        )
        assert fn is not None, f"{filename} has no {funcname}"
        registrations = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == registrar)
                or (isinstance(n.func, ast.Attribute) and n.func.attr == registrar)
            )
        ]
        assert len(registrations) == 1, f"{funcname}: {len(registrations)} registrations"
        reg = registrations[0]
        # every Try whose finally drains a watcher/uploader must be INSIDE the registration's block,
        # which is true exactly when the registration starts before the Try does.
        drains = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Try)
            and n.finalbody
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "stop"
                for stmt in n.finalbody
                for c in ast.walk(stmt)
            )
        ]
        assert drains, f"{funcname}: no draining finally found"
        for drain in drains:
            assert reg.lineno < drain.lineno, (
                f"{funcname}: registration at line {reg.lineno} starts after the drain at "
                f"{drain.lineno}, so the final checkpoint publishes with no pace"
            )
