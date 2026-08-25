from __future__ import annotations

import inspect
import threading
import time

import pytest

import flash.engine.worker.train.entry.rl_train_runner as rl_train_runner
from flash.engine.worker.io import heartbeat
from flash.engine.worker.io.heartbeat import RewardObservabilityBuffer
from flash.engine.worker.teacher.client import TeacherClient
from flash.engine.worker.train.entry import backend_common
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


def _watchdog(monkeypatch, *, baseline=0, gauge=None):
    clock = _Clock()
    monkeypatch.setattr(diagnostics.time, "monotonic", clock)
    monkeypatch.setattr(diagnostics, "VERL_CHILD_SILENCE_SECONDS", 10.0)
    torn_down = []
    proc = _Proc()
    tail = diagnostics.ChildOutputTail()
    watchdog = diagnostics.VerlChildSilenceWatchdog(
        tail,
        baseline_step=baseline,
        parent_work=gauge,
        child_alive=lambda: proc.poll() is None,
        teardown=lambda: torn_down.append("teardown"),
    )
    return clock, proc, tail, watchdog, torn_down


def test_repeated_frozen_lines_are_retained_without_counting_as_progress(monkeypatch):
    clock, _proc, tail, watchdog, torn_down = _watchdog(monkeypatch)
    tail.record("Training Progress\n")
    watchdog.observe_line("Training Progress")
    tail.record("frozen line\n")
    watchdog.observe_line("frozen line")
    first = tail.written
    for _ in range(3):
        tail.record("frozen line\n")
        watchdog.observe_line("frozen line")
    assert tail.tail()[-4:] == ["frozen line"] * 4
    assert tail.written == first

    clock.advance(10.1)
    watchdog.check()
    assert torn_down == ["teardown"]
    with pytest.raises(RuntimeError, match="verl_child_silence"):
        watchdog.raise_if_failed()
    watchdog.check()
    assert torn_down == ["teardown"]


def test_fresh_and_resumed_runs_arm_only_at_training_boundary(monkeypatch):
    clock, _proc, _tail, fresh, torn_down = _watchdog(monkeypatch, baseline=0)
    fresh.observe_step(0)
    clock.advance(20)
    fresh.check()
    assert torn_down == []
    fresh.observe_line("Training Progress: 0%")
    clock.advance(10.1)
    fresh.check()
    assert torn_down == ["teardown"]

    clock, _proc, _tail, resumed, torn_down = _watchdog(monkeypatch, baseline=7)
    resumed.observe_step(7)
    clock.advance(20)
    resumed.check()
    assert torn_down == []
    resumed.observe_step(8)
    clock.advance(10.1)
    resumed.check()
    assert torn_down == ["teardown"]


def test_ordinary_setup_output_does_not_arm_the_watchdog(monkeypatch):
    """only the fit-loop banner arms; setup chatter must not.

    a long model download or engine build emits plenty of lines and then goes quiet for minutes. if
    any line armed the watchdog, that ordinary silence would tear down a healthy run.
    """
    clock, _proc, tail, watchdog, torn_down = _watchdog(monkeypatch, baseline=0)
    for line in (
        "INFO downloading model shards",
        "WARNING flash attention kernel unavailable",
        "loading checkpoint shards: 100%",
    ):
        tail.record(line + "\n")
        watchdog.observe_line(line)
    clock.advance(60)
    watchdog.check()
    assert torn_down == []
    assert watchdog._failure is None

    # the banner is the boundary: the same silence now counts.
    watchdog.observe_line("Training Progress: 0%")
    clock.advance(10.1)
    watchdog.check()
    assert torn_down == ["teardown"]


def test_line_progress_parent_completion_and_busy_work_reset_silence(monkeypatch):
    gauge = ParentWorkGauge()
    clock, _proc, tail, watchdog, torn_down = _watchdog(monkeypatch, gauge=gauge)
    watchdog.observe_line("Training Progress")
    clock.advance(9)
    tail.record("new line")
    watchdog.check()
    clock.advance(9)
    gauge.complete()
    watchdog.check()
    clock.advance(9)
    with gauge.busy():
        watchdog.check()
        clock.advance(20)
        watchdog.check()
    assert torn_down == []
    clock.advance(10.1)
    watchdog.check()
    assert torn_down == ["teardown"]


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


def test_off_thread_field_sampling_continues_while_heartbeat_upload_blocks(monkeypatch):
    monkeypatch.setattr(heartbeat, "_LIVENESS_TICK_S", 0.01)
    monkeypatch.setattr(heartbeat, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.2)
    diagnostics_entered = threading.Event()
    diagnostics_release = threading.Event()
    calls = []
    caller_threads = set()

    def blocked_diagnostics(*, include_torch):
        diagnostics_entered.set()
        diagnostics_release.wait(timeout=1)
        return {}

    def fields():
        calls.append(time.monotonic())
        caller_threads.add(threading.get_ident())
        return {"count": len(calls)}

    monkeypatch.setattr(heartbeat.worker_perf, "gpu_diagnostics", blocked_diagnostics)
    monkeypatch.setattr(heartbeat, "heartbeat", lambda *_args, **_kwargs: None)
    with heartbeat.liveness_heartbeat(
        "rl_step",
        fields=fields,
        sample_off_thread=True,
    ):
        assert diagnostics_entered.wait(1)
        time.sleep(0.08)
        assert len(calls) >= 3
        diagnostics_release.set()
    assert len(caller_threads) == 1


def test_reward_observability_lifetime_count_and_busy_depth_survive_generations():
    buffer = RewardObservabilityBuffer(generation_size=1)
    buffer.record("prompt-1", "completion-1", 1.0)
    buffer.close_generation(1)
    buffer.record("prompt-2", "completion-2", 2.0)
    buffer.close_generation(2)
    assert buffer.heartbeat_fields()["reward_completions"] == 2
    with buffer.parent_work.busy():
        assert buffer.heartbeat_fields()["reward_grading_depth"] == 1
    assert buffer.heartbeat_fields()["reward_grading_depth"] == 0


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


def test_authoritative_tail_classification_precedes_generic_silence():
    backend_source = inspect.getsource(backend_common.run_verl_training)
    assert backend_source.index("raise_for_classified_verl_exit") < backend_source.index(
        "silence_watchdog.raise_if_failed"
    )
    grpo_source = inspect.getsource(rl_train_runner._GrpoSubprocessStream.wait_and_classify)
    assert grpo_source.index("raise_for_classified_verl_exit") < grpo_source.index(
        "self._silence_watchdog.raise_if_failed"
    )
