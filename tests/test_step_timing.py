import math

import pytest

from flash.cli.ui.heartbeat import _heartbeat_pairs, _step_timing_pairs
from flash.engine.worker import rl_train_runner
from flash.engine.worker.rl_train_runner import _ingest_step_metrics, _StepMetricState
from flash.engine.worker.train.core.step_timing import StepTiming


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


def test_forced_first_metrics_retry_carries_pace_once_available(monkeypatch) -> None:
    calls = []

    def heartbeat(stage, **fields):
        calls.append((stage, fields))
        return len(calls) > 1

    monkeypatch.setattr(rl_train_runner._w, "heartbeat", heartbeat)
    monkeypatch.setattr(rl_train_runner._w, "_remaining_worker_wall_seconds", lambda: 20000.0)
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

    assert len(calls) == 2
    assert calls[0][1]["force"] is True
    assert "step_duration_s" not in calls[0][1]
    assert calls[1][1]["reward_metrics"] == {"quality": 0.5}
    assert calls[1][1]["step_duration_s"] == 92.0
    assert calls[1][1]["projected_remaining_s"] == pytest.approx(17296.0)


def test_current_running_rl_attempt_renders_compact_pace_and_one_warning() -> None:
    heartbeat = {
        "stage": "rl_step",
        "step_duration_s": 92.0,
        "projected_remaining_s": 17296.0,
        "wall_deadline_at_risk": True,
    }

    pairs = _step_timing_pairs(heartbeat, running=True, current_attempt=True)

    assert pairs[0] == ("pace", "92s/step · ~4.8h left")
    warnings = [value for label, value in pairs if label == "warning"]
    assert len(warnings) == 1
    assert "exceeds" in warnings[0]


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
    assert "pace" not in dict(_heartbeat_pairs(superseded))
    assert "pace" not in dict(_heartbeat_pairs(finished))


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
        "pace": "92s/step"
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
