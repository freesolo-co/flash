"""Timing is diagnostic: concurrent ranks, retries and failures cannot become step evidence."""

import json
from types import SimpleNamespace

import pytest

from flash.cli.ui.heartbeat import _training_timing_pairs
from flash.engine.worker.runtime import phases
from flash.engine.worker.train.entry.backend_common import verl_step_number


def event(name, state, span="a" * 32, **kwargs):
    return dict(phase=name, state=state, span=span, **kwargs)


def test_overlapping_ranks_are_not_summed_and_first_update_is_attempt_local(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(phases.time, "monotonic", lambda: now[0])
    timings = phases.PhaseTimings(started=10)
    timings.observe(event("model_initialization", "started"))
    timings.observe(event("model_initialization", "started", span="b" * 32))
    now[0] = 20
    assert timings.snapshot()["active_phase_elapsed_s"] == {"model_initialization": 10}
    timings.observe(event("model_initialization", "completed", seconds=10))
    timings.observe(event("model_initialization", "completed", span="b" * 32, seconds=8))
    timings.observe(event("first_update", "completed", seconds=3))
    result = timings.snapshot()
    assert result["longest_phase_s"]["model_initialization"] == 10
    assert result["worker_to_first_update_s"] == 10
    now[0] = 30
    timings.observe(event("first_update", "completed", seconds=4))
    assert timings.snapshot()["worker_to_first_update_s"] == 10
    assert phases.PhaseTimings().snapshot()["longest_phase_s"] == {}


def test_failure_releases_span_but_never_records_success(monkeypatch, capsys):
    now = iter([10.0, 10.0, 15.0])
    context = SimpleNamespace(phase_timings=phases.PhaseTimings(started=0))
    monkeypatch.setattr(phases.time, "monotonic", lambda: next(now))
    with (
        pytest.raises(ValueError, match="original error"),
        phases.training_phase("first_update", context=context, step=19),
    ):
        raise ValueError("original error")
    records = [
        json.loads(line.split(phases.PREFIX)[1]) for line in capsys.readouterr().out.splitlines()
    ]
    assert [r["state"] for r in records] == ["started", "failed"]
    assert records[-1]["seconds"] == 5
    assert context.phase_timings.active == {}
    assert context.phase_timings.longest == {}
    assert context.phase_timings.first_update_ready_s is None
    for record in records:
        assert verl_step_number(phases.PREFIX + json.dumps(record)) is None


def test_child_records_survive_ray_prefix_and_reject_malformed_values():
    timings = phases.PhaseTimings()
    timings.observe_line(
        "(TaskRunner pid=12) "
        + phases.PREFIX
        + json.dumps(event("compilation_and_warmup", "completed", seconds=17))
    )
    assert timings.snapshot()["longest_phase_s"] == {"compilation_and_warmup": 17}
    for raw in ["[]", "{}", '{"phase": []}', "{bad", "null", '"string"']:
        timings.observe_line(phases.PREFIX + raw)
    for value in [float("nan"), float("inf"), -1, True, "17", None, 10**1000]:
        timings.observe(event("first_update", "completed", seconds=value))
    assert "first_update" not in timings.longest
    for i in range(200):
        timings.observe(event("model_initialization", "started", span=f"{i:032x}"))
    assert len(timings.active) == 128


def test_heartbeat_and_metadata_share_measured_timings(worker_context, monkeypatch):
    from flash.engine.worker.io import heartbeat, hf
    from flash.engine.worker.train.core.lifecycle import finalize

    payloads = []
    monkeypatch.setattr(hf, "hf_upload_file", lambda *a, **k: True)
    monkeypatch.setattr(
        heartbeat,
        "_console_heartbeat_snapshot",
        lambda payload, *args, **kw: payloads.append(payload) or "{}",
    )
    worker_context.phase_timings.observe(event("first_update", "completed", seconds=1020))
    heartbeat.heartbeat("rl_step", step=1, context=worker_context)
    assert payloads[-1]["training_timing"]["longest_phase_s"]["first_update"] == 1020
    metrics = []
    monkeypatch.setattr(finalize, "_publish_training_completion", lambda m, **kw: metrics.append(m))
    monkeypatch.setattr(worker_context, "require_active_env", lambda: SimpleNamespace(id="test"))
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)
    finalize.write_train_meta(
        "rl", "/tmp/adapter", "model", 1030, 21, 0, 0, {}, context=worker_context
    )
    assert metrics[0].notes["training_timing"]["longest_phase_s"]["first_update"] == 1020
    assert "excludes trainer initialization" in metrics[0].notes["setup_seconds_scope"]


def test_status_panel_uses_actual_phases_and_labels_snapshot_age():
    pairs = _training_timing_pairs(
        {
            "training_timing": {
                "active_phase_elapsed_s": {"compilation_and_warmup": 60, "unknown": 7},
                "worker_to_first_update_s": 1020,
                "longest_phase_s": {"first_update": 900, "steady_update": 9},
            }
        }
    )
    assert any(
        key == "compilation / warmup" and "at last heartbeat" in value for key, value in pairs
    )
    assert {key for key, _ in pairs} == {
        "compilation / warmup",
        "worker to first update",
        "longest first update",
        "longest later update",
    }
    assert _training_timing_pairs({"training_timing": []}) == []
