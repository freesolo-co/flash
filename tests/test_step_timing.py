import json
import math

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.io.hf as worker_hf
from flash.cli.ui.heartbeat import _heartbeat_pairs, _step_timing_pairs
from flash.engine.worker.train.core.lifecycle.step_timing import StepTiming
from flash.engine.worker.train.entry import rl_train_runner
from flash.engine.worker.train.entry.rl_train_runner import _ingest_step_metrics, _StepMetricState


def _line(step: int, duration: float, *, reward: float = 0.4) -> str:
    return f"step:{step} - critic/rewards/mean:{reward} - timing_s/step:{duration}"


def test_warmup_is_excluded_and_incident_projects_steady_pace() -> None:
    timing = StepTiming()
    timing.record_duration(515.0)

    assert timing.heartbeat_fields(current_step=1, total_steps=190, remaining_wall_s=86400) == {}

    timing.record_duration(92.0)
    fields = timing.heartbeat_fields(current_step=2, total_steps=190, remaining_wall_s=86400)

    assert fields["step_duration_s"] == 92.0
    assert fields["projected_remaining_s"] == pytest.approx(17296.0)
    assert fields["projected_remaining_s"] / 3600 == pytest.approx(4.8, abs=0.01)
    assert "wall_deadline_at_risk" not in fields


def test_median_resists_outlier_while_projection_uses_mean() -> None:
    timing = StepTiming()
    for duration in (500.0, 92.0, 92.0, 900.0):
        timing.record_duration(duration)

    fields = timing.heartbeat_fields(current_step=4, total_steps=14, remaining_wall_s=None)

    assert fields["step_duration_s"] == 92.0
    assert fields["projected_remaining_s"] == pytest.approx(((92 + 92 + 900) / 3) * 10)


def test_overflowing_diagnostic_is_omitted_through_step_ingestion(monkeypatch) -> None:
    calls = []

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return True

    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", heartbeat)
    monkeypatch.setattr(
        rl_train_runner._worker_state, "_remaining_worker_wall_seconds", lambda: 20000.0
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    inp = {"max_completion": 512, "steps": 190}

    for step, duration in ((1, 500.0), (2, 1e308), (3, 1e308)):
        state.progress["step"] = step
        _ingest_step_metrics(
            _line(step, duration),
            inp,
            state,
            lambda: rl_train_runner._step_timing_fields(inp, state),
        )

    assert len(calls) == 1
    assert "step_duration_s" not in calls[0][1]
    assert "projected_remaining_s" not in calls[0][1]
    assert (
        state.step_timing.heartbeat_fields(
            current_step=3,
            total_steps=190,
            remaining_wall_s=20000.0,
        )
        == {}
    )


def test_retains_only_the_latest_64_post_warmup_samples() -> None:
    timing = StepTiming()
    timing.record_duration(500.0)
    for duration in range(2, 68):
        timing.record_duration(float(duration))

    fields = timing.heartbeat_fields(current_step=0, total_steps=1, remaining_wall_s=None)

    assert fields["step_duration_s"] == 35.5
    assert fields["projected_remaining_s"] == 35.5


def test_overrun_flag_uses_actual_remaining_wall_allowance() -> None:
    timing = StepTiming()
    timing.record_duration(500.0)
    timing.record_duration(100.0)

    safe = timing.heartbeat_fields(current_step=2, total_steps=4, remaining_wall_s=200.0)
    over = timing.heartbeat_fields(current_step=2, total_steps=4, remaining_wall_s=199.0)
    exhausted = timing.heartbeat_fields(current_step=2, total_steps=4, remaining_wall_s=0.0)
    unknown = timing.heartbeat_fields(current_step=2, total_steps=4, remaining_wall_s=None)

    assert "wall_deadline_at_risk" not in safe
    assert over["wall_deadline_at_risk"] is True
    assert exhausted["wall_deadline_at_risk"] is True
    assert "wall_deadline_at_risk" not in unknown


def test_non_step_and_validation_lines_do_not_consume_warmup() -> None:
    state = _StepMetricState(sent_first_metrics=True)
    inp = {"max_completion": 512, "steps": 3}
    for line in (
        "global_step:1 - critic/rewards/mean:0.4 - timing_s/step:900.0",
        "validation metrics: val-core/reward/mean:0.4 - timing_s/step:800.0",
        "step:1 - val-core/reward/mean:0.4 - timing_s/step:700.0",
    ):
        _ingest_step_metrics(line, inp, state, dict)

    state.progress["step"] = 1
    _ingest_step_metrics(_line(1, 500.0), inp, state, dict)
    assert (
        state.step_timing.heartbeat_fields(current_step=1, total_steps=3, remaining_wall_s=1000)
        == {}
    )

    state.progress["step"] = 2
    _ingest_step_metrics(_line(2, 92.0), inp, state, dict)
    assert (
        state.step_timing.heartbeat_fields(current_step=2, total_steps=3, remaining_wall_s=1000)[
            "step_duration_s"
        ]
        == 92.0
    )


def test_step_metrics_keep_numeric_census_fields_and_leave_exact_steps_for_terminal_notes() -> None:
    state = _StepMetricState(sent_first_metrics=True)
    state.host_census = {
        "available": 1,
        "peak_processes": 7,
        "steps": [{"optimizer_step": 1, "processes": 7}],
    }
    _ingest_step_metrics(_line(1, 92.0), {"max_completion": 512}, state, dict)

    metrics = state.metrics_last[-1]
    assert metrics["host_census/available"] == 1
    assert metrics["host_census/peak_processes"] == 7
    assert "host_census/steps" not in metrics


def test_first_usable_pace_is_forced_after_first_metrics_commit(monkeypatch) -> None:
    calls = []

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return True

    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", heartbeat)
    monkeypatch.setattr(
        rl_train_runner._worker_state, "_remaining_worker_wall_seconds", lambda: 20000.0
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    inp = {"max_completion": 512, "steps": 190}

    def observability():
        return {
            "reward_metrics": {"quality": 0.5},
            **rl_train_runner._step_timing_fields(inp, state),
        }

    state.progress["step"] = 1
    _ingest_step_metrics(_line(1, 515.0), inp, state, observability)
    state.progress["step"] = 2
    _ingest_step_metrics(_line(2, 92.0), inp, state, observability)
    state.progress["step"] = 3
    _ingest_step_metrics(_line(3, 93.0), inp, state, observability)

    assert len(calls) == 2
    assert calls[0][1]["force"] is True
    assert "step_duration_s" not in calls[0][1]
    assert calls[1][1]["force"] is True
    assert calls[1][1]["step_duration_s"] == 92.0
    assert calls[1][1]["projected_remaining_s"] == pytest.approx(17296.0)
    assert state.sent_first_metrics is True
    assert state.sent_first_timing is True


def test_first_timing_bypasses_force_floor_and_retries_with_real_wrapper(monkeypatch) -> None:
    import importlib

    heartbeat_module = importlib.import_module("flash.engine.worker.io.heartbeat")

    now = {"value": 1000.0}
    outcomes = iter((True, False, True))
    uploads = []

    def upload(local, *_args, **_kwargs):
        with open(local) as f:
            uploads.append(json.load(f))
        return next(outcomes)

    monkeypatch.setattr(heartbeat_module.time, "time", lambda: now["value"])
    monkeypatch.setattr(worker_hf, "hf_upload_file", upload)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_FORCE_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 999.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_FORCED_UPLOAD", 0.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_COMMITTED_STEP", 0)
    monkeypatch.setattr(worker_heartbeat, "_HB_PROGRESS_SEQ", 0)
    monkeypatch.setattr(worker_heartbeat, "_HB_PROGRESS_UPLOADED_SEQ", 0)
    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", worker_heartbeat.heartbeat)
    monkeypatch.setattr(
        rl_train_runner._worker_state, "_remaining_worker_wall_seconds", lambda: 20000.0
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    inp = {"max_completion": 512, "steps": 190}

    def observability():
        return rl_train_runner._step_timing_fields(inp, state)

    state.progress["step"] = 1
    _ingest_step_metrics(_line(1, 515.0), inp, state, observability)
    assert state.sent_first_metrics is True
    assert worker_heartbeat._HB_LAST_FORCED_UPLOAD == 1000.0

    now["value"] = 1001.0
    state.progress["step"] = 2
    _ingest_step_metrics(_line(2, 92.0), inp, state, observability)
    assert state.sent_first_timing is False
    assert worker_heartbeat._HB_LAST_COMMITTED_STEP == 1
    assert worker_heartbeat._HB_LAST_FORCED_UPLOAD == 1000.0

    now["value"] = 1002.0
    state.progress["step"] = 3
    _ingest_step_metrics(_line(3, 93.0), inp, state, observability)
    assert state.sent_first_timing is True

    now["value"] = 1003.0
    state.progress["step"] = 4
    _ingest_step_metrics(_line(4, 94.0), inp, state, observability)
    assert worker_heartbeat.heartbeat("rl_step", step=4, force=True) is False

    assert [payload["step"] for payload in uploads] == [1, 2, 3]
    assert "step_duration_s" not in uploads[0]
    assert uploads[1]["step_duration_s"] == 92.0
    assert uploads[2]["step_duration_s"] == 92.5


def test_failed_first_timing_commit_retries_on_next_step(monkeypatch) -> None:
    calls = []
    outcomes = iter([True, False, True])

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return next(outcomes)

    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", heartbeat)
    monkeypatch.setattr(
        rl_train_runner._worker_state, "_remaining_worker_wall_seconds", lambda: 20000.0
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    inp = {"max_completion": 512, "steps": 190}

    def observability():
        return {
            "reward_metrics": {"quality": 0.5},
            **rl_train_runner._step_timing_fields(inp, state),
        }

    state.progress["step"] = 1
    _ingest_step_metrics(_line(1, 515.0), inp, state, observability)
    assert state.sent_first_metrics is True
    assert state.sent_first_timing is False

    state.progress["step"] = 2
    _ingest_step_metrics(_line(2, 92.0), inp, state, observability)
    assert state.sent_first_metrics is True
    assert state.sent_first_timing is False

    state.progress["step"] = 3
    _ingest_step_metrics(_line(3, 93.0), inp, state, observability)
    assert state.sent_first_timing is True

    state.progress["step"] = 4
    _ingest_step_metrics(_line(4, 94.0), inp, state, observability)

    assert len(calls) == 3
    assert [fields["step"] for _, fields in calls] == [1, 2, 3]
    assert "step_duration_s" not in calls[0][1]
    assert calls[1][1]["step_duration_s"] == 92.0
    assert calls[2][1]["step_duration_s"] == 92.5


def test_forced_first_metrics_retry_carries_pace_once_available(monkeypatch) -> None:
    calls = []

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return len(calls) > 1

    monkeypatch.setattr(rl_train_runner._worker_heartbeat, "heartbeat", heartbeat)
    monkeypatch.setattr(
        rl_train_runner._worker_state, "_remaining_worker_wall_seconds", lambda: 20000.0
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = _StepMetricState()
    inp = {"max_completion": 512, "steps": 190}

    def observability():
        return {
            "reward_metrics": {"quality": 0.5},
            **rl_train_runner._step_timing_fields(inp, state),
        }

    state.progress["step"] = 1
    _ingest_step_metrics(_line(1, 515.0), inp, state, observability)
    assert state.sent_first_metrics is False
    assert state.sent_first_timing is False

    state.progress["step"] = 2
    _ingest_step_metrics(_line(2, 92.0), inp, state, observability)
    state.progress["step"] = 3
    _ingest_step_metrics(_line(3, 93.0), inp, state, observability)

    assert len(calls) == 2
    assert calls[0][1]["force"] is True
    assert "step_duration_s" not in calls[0][1]
    assert calls[1][1]["reward_metrics"] == {"quality": 0.5}
    assert calls[1][1]["step_duration_s"] == 92.0
    assert calls[1][1]["projected_remaining_s"] == pytest.approx(17296.0)
    assert state.sent_first_metrics is True
    assert state.sent_first_timing is True


def test_current_running_rl_attempt_renders_compact_pace_and_one_warning() -> None:
    heartbeat = {
        "stage": "rl_step",
        "step_duration_s": 92.0,
        "projected_remaining_s": 17296.0,
        "wall_deadline_at_risk": True,
    }

    pairs = _step_timing_pairs(heartbeat, running=True, current_attempt=True)

    assert pairs[:2] == [("median pace", "92s/step"), ("mean ETA", "~4.8h left")]
    warnings = [value for label, value in pairs if label == "warning"]
    assert warnings == ["mean-based remaining-time projection exceeds the run's wall time left"]


def test_pace_is_suppressed_for_superseded_finished_and_non_rl_heartbeats() -> None:
    heartbeat = {
        "stage": "rl_step",
        "step_duration_s": 92.0,
        "projected_remaining_s": 17296.0,
    }

    assert (
        _step_timing_pairs(
            {**heartbeat, "stage": "rl_finalizing"}, running=True, current_attempt=True
        )
        == []
    )

    superseded = {
        "state": "running",
        "remote": {"attempt": 2},
        "last_heartbeat": {**heartbeat, "attempt": 1},
    }
    finished = {"state": "done", "last_heartbeat": heartbeat}
    assert "median pace" not in dict(_heartbeat_pairs(superseded))
    assert "mean ETA" not in dict(_heartbeat_pairs(superseded))
    assert "median pace" not in dict(_heartbeat_pairs(finished))
    assert "mean ETA" not in dict(_heartbeat_pairs(finished))


@pytest.mark.parametrize("bad", [None, True, 0, -1, math.nan, math.inf, "92", 10**1000])
def test_cli_rejects_invalid_numeric_pace_fields(bad) -> None:
    pairs = _step_timing_pairs(
        {
            "stage": "rl_step",
            "step_duration_s": bad,
            "projected_remaining_s": 100.0,
            "wall_deadline_at_risk": True,
        },
        running=True,
        current_attempt=True,
    )

    assert pairs == []


def test_warning_requires_a_valid_projection_and_literal_true_flag() -> None:
    base = {"stage": "rl_step", "step_duration_s": 92.0}

    assert dict(_step_timing_pairs(base, running=True, current_attempt=True)) == {
        "median pace": "92s/step"
    }
    assert "warning" not in dict(
        _step_timing_pairs(
            {**base, "projected_remaining_s": 100.0, "wall_deadline_at_risk": "true"},
            running=True,
            current_attempt=True,
        )
    )
    assert "warning" not in dict(
        _step_timing_pairs(
            {**base, "projected_remaining_s": 0.0, "wall_deadline_at_risk": True},
            running=True,
            current_attempt=True,
        )
    )
