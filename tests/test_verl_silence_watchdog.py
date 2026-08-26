from __future__ import annotations

import json

import pytest

from flash.engine.worker.io.progress import RewardObservabilityBuffer
from flash.engine.worker.teacher.client import TeacherClient
from flash.engine.worker.verl import diagnostics
from flash.engine.worker.verl.parent_work import ParentWorkGauge


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _Proc:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


def _observer(monkeypatch, *, baseline=0, gauge=None):
    clock = _Clock()
    monkeypatch.setattr(diagnostics.time, "monotonic", clock)
    monkeypatch.setattr(diagnostics, "VERL_CHILD_SILENCE_SECONDS", 10.0)
    monkeypatch.setattr(diagnostics.sys, "_current_frames", dict)
    proc = _Proc()
    tail = diagnostics.ChildOutputTail()
    observer = diagnostics.VerlChildSilenceObserver(
        tail,
        baseline_step=baseline,
        parent_work=gauge,
        child_alive=lambda: proc.poll() is None,
    )
    return clock, proc, tail, observer


def _diagnostic_payload(captured: str) -> dict:
    line = next(line for line in captured.splitlines() if line.startswith("VERL DIAGNOSTIC "))
    return json.loads(line.removeprefix("VERL DIAGNOSTIC "))


def test_repeated_frozen_lines_emit_one_diagnostic_without_lifecycle_failure(monkeypatch, capsys):
    clock, proc, tail, observer = _observer(monkeypatch)
    tail.record("Training Progress\n")
    observer.observe_line("Training Progress")
    tail.record("frozen line\n")
    observer.observe_line("frozen line")
    first = tail.written
    for _ in range(3):
        tail.record("frozen line\n")
        observer.observe_line("frozen line")
    assert tail.tail()[-4:] == ["frozen line"] * 4
    assert tail.written == first

    clock.advance(10.1)
    observer.check()
    payload = _diagnostic_payload(capsys.readouterr().out)
    assert payload["kind"] == "verl_child_silence_diagnostic"
    assert payload["elapsed_s"] == 10.1
    assert payload["child_tail"][-4:] == ["frozen line"] * 4
    assert proc.poll() is None

    observer.check()
    assert capsys.readouterr().out == ""


def test_fresh_and_resumed_runs_arm_only_at_training_boundary(monkeypatch, capsys):
    clock, _proc, _tail, fresh = _observer(monkeypatch, baseline=0)
    fresh.observe_step(0)
    clock.advance(20)
    fresh.check()
    assert capsys.readouterr().out == ""
    fresh.observe_line("Training Progress: 0%")
    clock.advance(10.1)
    fresh.check()
    assert _diagnostic_payload(capsys.readouterr().out)["elapsed_s"] == 10.1

    clock, _proc, _tail, resumed = _observer(monkeypatch, baseline=7)
    resumed.observe_step(7)
    clock.advance(20)
    resumed.check()
    assert capsys.readouterr().out == ""
    resumed.observe_step(8)
    clock.advance(10.1)
    resumed.check()
    assert _diagnostic_payload(capsys.readouterr().out)["elapsed_s"] == 10.1


def test_ordinary_setup_output_does_not_arm_the_observer(monkeypatch, capsys):
    clock, _proc, tail, observer = _observer(monkeypatch, baseline=0)
    for line in (
        "INFO downloading model shards",
        "WARNING flash attention kernel unavailable",
        "loading checkpoint shards: 100%",
    ):
        tail.record(line + "\n")
        observer.observe_line(line)
    clock.advance(60)
    observer.check()
    assert capsys.readouterr().out == ""

    observer.observe_line("Training Progress: 0%")
    clock.advance(10.1)
    observer.check()
    assert _diagnostic_payload(capsys.readouterr().out)["elapsed_s"] == 10.1


def test_line_progress_parent_completion_and_busy_work_reset_silence(monkeypatch, capsys):
    gauge = ParentWorkGauge()
    clock, proc, tail, observer = _observer(monkeypatch, gauge=gauge)
    observer.observe_line("Training Progress")
    clock.advance(9)
    tail.record("new line")
    observer.check()
    clock.advance(9)
    gauge.complete()
    observer.check()
    clock.advance(9)
    with gauge.busy():
        observer.check()
        clock.advance(20)
        observer.check()
    assert capsys.readouterr().out == ""
    assert proc.poll() is None
    clock.advance(10.1)
    observer.check()
    payload = _diagnostic_payload(capsys.readouterr().out)
    assert payload["parent_completed"] == 1
    assert payload["parent_depth"] == 0


def test_new_activity_allows_a_later_diagnostic(monkeypatch, capsys):
    clock, _proc, tail, observer = _observer(monkeypatch)
    observer.observe_line("Training Progress")
    clock.advance(10.1)
    observer.check()
    assert _diagnostic_payload(capsys.readouterr().out)["elapsed_s"] == 10.1

    tail.record("new child output")
    observer.observe_line("new child output")
    clock.advance(10.1)
    observer.check()
    assert _diagnostic_payload(capsys.readouterr().out)["elapsed_s"] == 10.1


def test_child_exit_suppresses_silence_diagnostic(monkeypatch, capsys):
    clock, proc, _tail, observer = _observer(monkeypatch)
    observer.observe_line("Training Progress")
    proc.returncode = 1
    clock.advance(20)
    observer.check()
    assert capsys.readouterr().out == ""


def test_parent_work_gauge_releases_depth_in_finally():
    gauge = ParentWorkGauge()

    def fail_while_busy():
        with gauge.busy():
            assert gauge.snapshot().depth == 1
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        fail_while_busy()
    assert gauge.snapshot().depth == 0
    gauge.complete()
    assert gauge.snapshot().completed == 1


def test_reward_observability_lifetime_count_and_busy_depth_survive_generations():
    buffer = RewardObservabilityBuffer(generation_size=1)
    buffer.record("prompt-1", "completion-1", 1.0)
    buffer.close_generation(1)
    buffer.record("prompt-2", "completion-2", 2.0)
    buffer.close_generation(2)
    assert buffer.progress_fields()["reward_completions"] == 2
    with buffer.parent_work.busy():
        assert buffer.progress_fields()["reward_grading_depth"] == 1
    assert buffer.progress_fields()["reward_grading_depth"] == 0


def test_teacher_score_many_calls_completion_callback_per_result():
    client = TeacherClient.__new__(TeacherClient)
    client._score_one = lambda prompt, completion: (prompt, completion)
    completed = []
    scored = client.score_many(
        [("p1", "c1"), ("p2", "c2")],
        on_scored=lambda: completed.append(True),
    )
    assert scored == [("p1", "c1"), ("p2", "c2")]
    assert completed == [True, True]
