"""Metered billing: charge = measured training seconds x rented card $/hr x rented cards.

Setup (model download, data loading, trainer/model/vLLM init) and final publication never bill.
"""

import json
import os

import pytest

from flash.engine.worker.runtime import phases
from flash.runner.accounting import costs
from flash.runner.lifecycle import state as runner_state
from flash.runner.lifecycle import status as status_ops
from flash.runner.lifecycle.state import RunStatus
from tests.test_charge_pricing import _spec
from tests.test_spot_billing import spot_spec

H100_MODAL = 4.42512


def _event(name, state, span="a" * 32, **kw):
    return dict(phase=name, state=state, span=span, **kw)


def _metrics(tmp_path, monkeypatch, spec, **fields):
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path))
    dest = runner_state.artifacts_dir(spec)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "metrics.json"), "w") as handle:
        json.dump(fields, handle)


def _status(spec, *, heartbeat=None, quote=50.0, state="done"):
    return RunStatus(
        run_id=spec.run_id,
        state=state,
        spec=spec.to_dict(),
        estimated_cost_usd=quote,
        last_heartbeat=heartbeat,
    )


# ---------------------------------------------------------------------------- worker meter


def test_meter_starts_at_first_update_and_stops_at_loop_exit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(phases.time, "monotonic", lambda: now[0])
    timings = phases.PhaseTimings(started=0)
    # model download + init happen before the first update: never metered.
    timings.observe(_event("model_download", "started", span="d" * 32))
    now[0] = 600
    timings.observe(_event("model_download", "completed", span="d" * 32, seconds=600))
    assert "billable_train_s" not in timings.snapshot()
    assert timings.billable_train_seconds() is None
    now[0] = 900
    timings.observe(_event("first_update", "started"))
    now[0] = 1000
    timings.observe(_event("first_update", "completed", seconds=100))
    timings.observe(_event("steady_update", "started", span="b" * 32))
    now[0] = 4500
    assert timings.snapshot()["billable_train_s"] == 3600
    timings.stop_billing()
    now[0] = 9000  # final publication: not metered
    assert timings.snapshot()["billable_train_s"] == 3600
    timings.stop_billing()  # idempotent
    assert timings.billable_train_seconds() == 3600


def test_meter_ignores_a_second_first_update_in_the_same_attempt(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(phases.time, "monotonic", lambda: now[0])
    timings = phases.PhaseTimings(started=0)
    timings.observe(_event("first_update", "started"))
    now[0] = 20
    timings.observe(_event("first_update", "started", span="c" * 32))  # another rank
    now[0] = 70
    assert timings.billable_train_seconds() == 60


def test_stop_before_training_never_meters():
    timings = phases.PhaseTimings()
    timings.stop_billing()
    assert timings.billable_train_seconds() is None
    assert "billable_train_s" not in timings.snapshot()


def test_sft_runner_meters_from_first_update_to_child_exit(worker_context, monkeypatch):
    """The real SFT runner path: child stdout drives the meter, child exit freezes it."""
    from types import SimpleNamespace

    from flash.engine.worker.train.entry import sft_train, sft_train_runner

    now = [100.0]
    monkeypatch.setattr(phases.time, "monotonic", lambda: now[0])

    def fake_child(child, callbacks, on_line, *, context):
        now[0] = 700.0  # trainer/model init inside the child: not billed
        line = phases.PREFIX + json.dumps(_event("first_update", "started"))
        context.phase_timings.observe_line(line)
        now[0] = 4300.0  # one hour of optimizer updates
        return 0

    class _Sampler:
        def start(self):
            return self

        def stop_gb(self):
            now[0] = 9000.0  # post-loop work (sampler stop, publication): not billed
            return 1.0

    monkeypatch.setattr(sft_train_runner, "_invoke_sft_child", fake_child)
    monkeypatch.setattr(sft_train, "_NvidiaSmiPeakSampler", _Sampler)
    monkeypatch.setattr(
        sft_train_runner,
        "liveness_heartbeat",
        lambda *a, **k: __import__("contextlib").nullcontext(),
    )
    watcher = SimpleNamespace(start=lambda: None, stop=lambda require_complete: None)
    child = SimpleNamespace(resume_step=0, watcher=watcher)
    progress = SimpleNamespace(values={"step": 0}, expected_components=(), runtime_verified=True)
    sft_train_runner._run_sft_child(child, 10, progress, lambda line: None, context=worker_context)
    assert worker_context.phase_timings.billable_train_seconds() == pytest.approx(3600.0)
    now[0] = 99999.0
    assert worker_context.phase_timings.snapshot()["billable_train_s"] == pytest.approx(3600.0)


def test_heartbeat_and_terminal_metrics_carry_the_meter(worker_context, monkeypatch):
    from types import SimpleNamespace

    from flash.engine.worker.io import artifacts as hf
    from flash.engine.worker.io import heartbeat
    from flash.engine.worker.train.core.lifecycle import finalize

    now = [0.0]
    monkeypatch.setattr(phases.time, "monotonic", lambda: now[0])
    payloads = []
    monkeypatch.setattr(hf, "upload_artifact_file", lambda *a, **k: True)
    monkeypatch.setattr(
        heartbeat,
        "_console_heartbeat_snapshot",
        lambda payload, *args, **kw: payloads.append(payload) or "{}",
    )
    worker_context.phase_timings.observe(_event("first_update", "started"))
    now[0] = 120.0
    heartbeat.heartbeat("rl_step", step=1, context=worker_context)
    assert payloads[-1]["training_timing"]["billable_train_s"] == 120.0
    worker_context.phase_timings.stop_billing()
    now[0] = 500.0
    metrics = []
    monkeypatch.setattr(finalize, "_publish_training_completion", lambda m, **kw: metrics.append(m))
    monkeypatch.setattr(worker_context, "require_active_env", lambda: SimpleNamespace(id="test"))
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)
    finalize.write_train_meta(
        "rl", "/tmp/adapter", "model", 130, 21, 0, 0, {}, context=worker_context
    )
    assert metrics[0].notes["training_timing"]["billable_train_s"] == 120.0


# ---------------------------------------------------------------------------- charge


def test_completed_run_bills_metered_seconds_not_the_quote(tmp_path, monkeypatch):
    spec = _spec()
    _metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=20,
        allocated_provider="modal",
        allocated_gpu="H100",
        allocated_gpu_count=1,
        notes={"training_timing": {"billable_train_s": 7200.0}},
    )
    charge = costs._status_estimated_charge(_status(spec, quote=105.6), spec, fallback=999.0)
    assert charge == pytest.approx(2 * H100_MODAL)


def test_metered_charge_uses_rented_card_count(tmp_path, monkeypatch):
    spec = _spec()
    _metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=5,
        allocated_provider="modal",
        allocated_gpu="H100",
        allocated_gpu_count=4,
        notes={"training_timing": {"billable_train_s": 3600.0}},
    )
    charge = costs._status_estimated_charge(_status(spec), spec, fallback=999.0)
    assert charge == pytest.approx(4 * H100_MODAL)


def test_metered_charge_sums_prior_attempts(tmp_path, monkeypatch):
    spec = _spec()
    _metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=20,
        allocated_provider="modal",
        allocated_gpu="H100",
        allocated_gpu_count=1,
        notes={"training_timing": {"billable_train_s": 1800.0}},
    )
    hb = {"attempt": 2, "billable_train_s_prior": 1800.0, "training_timing": {}}
    charge = costs._status_estimated_charge(_status(spec, heartbeat=hb), spec, fallback=999.0)
    assert charge == pytest.approx(H100_MODAL)


def test_spot_run_bills_spot_rate(tmp_path, monkeypatch):
    spec = spot_spec("H100")
    _metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=20,
        allocated_provider="modal",  # stale/legacy stamp: the authored Spot choice wins
        allocated_gpu="H100",
        allocated_gpu_count=1,
        notes={"training_timing": {"billable_train_s": 36000.0}},
    )
    charge = costs._status_estimated_charge(_status(spec), spec, fallback=999.0)
    assert charge == pytest.approx(10 * 2.00)  # fixed Spot H100 rate


def test_spot_run_resumed_on_another_class_bills_usage_weighted_rates(tmp_path, monkeypatch):
    spec = spot_spec("H100")
    _metrics(
        tmp_path,
        monkeypatch,
        spec,
        step=20,
        allocated_provider="spot",
        allocated_gpu="H100",
        allocated_gpu_count=1,
        notes={"training_timing": {"billable_train_s": 1800.0}},
    )
    # half the steps ran on H100 ($2.00), half on H200 ($2.50) after a preemption.
    hb = {
        "attempt": 1,
        "billable_train_s_prior": 1800.0,
        "spot_usage_by_gpu": {"H100": 10, "H200": 10},
        "training_timing": {},
    }
    charge = costs._status_estimated_charge(_status(spec, heartbeat=hb), spec, fallback=999.0)
    assert charge == pytest.approx(1.0 * 2.25)


def test_cancel_bills_heartbeat_meter_with_pre_teardown_handle(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path))  # no metrics yet
    hb = {
        "attempt": 0,
        "stage": "sft_step",
        "step": 3,
        "training_timing": {"billable_train_s": 900},
    }
    remote = {"provider": "modal", "allocated_gpu": "H100", "allocated_gpu_count": 2}
    charge = costs.cancelled_charge_usd(
        _status(spec, heartbeat=hb, state="cancelled"), spec, steps=3, rented_remote=remote
    )
    assert charge == pytest.approx(0.25 * H100_MODAL * 2)


def test_cancel_mid_first_update_bills_measured_time(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path))
    hb = {"attempt": 0, "stage": "sft_step", "training_timing": {"billable_train_s": 36}}
    remote = {"provider": "modal", "allocated_gpu": "H100", "allocated_gpu_count": 1}
    charge = costs.cancelled_charge_usd(
        _status(spec, heartbeat=hb, state="cancelled"), spec, steps=0, rented_remote=remote
    )
    assert charge == pytest.approx(0.01 * H100_MODAL)


def test_legacy_worker_without_meter_keeps_quote_pricing(tmp_path, monkeypatch):
    spec = _spec()
    _metrics(tmp_path, monkeypatch, spec, step=20, wall_seconds=1.0)
    assert costs.metered_training_seconds(_status(spec), spec) is None
    charge = costs._status_estimated_charge(_status(spec, quote=12.34), spec, fallback=999.0)
    assert charge == pytest.approx(12.34)


# ---------------------------------------------------------------------------- attempt carry


def test_status_carries_prior_attempt_meter_across_retries(tmp_path, monkeypatch):
    spec = _spec()
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path), raising=False)
    status = _status(spec, state="running", heartbeat=None)
    saved = {}
    monkeypatch.setattr(status_ops, "get_status", lambda run_id: status)
    monkeypatch.setattr(runner_state, "_save_status_unlocked", lambda s: saved.update(s=s))
    monkeypatch.setattr(status_ops.os.path, "exists", lambda path: True)
    monkeypatch.setattr(status_ops.reporting, "_report_status", lambda s: None, raising=False)

    def beat(attempt, billable):
        hb = {"attempt": attempt, "stage": "sft_step", "training_timing": {}}
        if billable is not None:
            hb["training_timing"]["billable_train_s"] = billable
        status_ops.record_heartbeat(spec.run_id, hb)
        status.last_heartbeat = saved["s"].last_heartbeat

    beat(0, 100.0)
    beat(0, 400.0)
    beat(1, None)  # replacement attempt still initializing
    assert status.last_heartbeat["billable_train_s_prior"] == 400.0
    beat(1, 50.0)
    beat(2, 10.0)
    assert status.last_heartbeat["billable_train_s_prior"] == 450.0
    assert costs.metered_training_seconds(status, spec) == pytest.approx(460.0)


# ---------------------------------------------------------------------------- estimator


def test_image_sft_quotes_vision_encoding_per_example():
    from dataclasses import replace

    from flash.cost.analytical import SFT_IMAGE_S_PER_EXAMPLE, estimate_cost, sft_image_seconds
    from flash.cost.types import RunConfig

    text = RunConfig(
        "Qwen/Qwen3.5-9B",
        "sft",
        1000,
        seq_len=8192,
        batch_size=1,
        gpu_type="H100",
        train_tokens=4_000_000,
        supervised_train_tokens=100_000,
        sft_packing_mode="packed",
        sft_packed_blocks=1000,
        sft_retained_examples=1000,
        max_wall_seconds=10**7,
    )
    image = replace(text, sft_packing_mode="exact-unpacked")
    assert sft_image_seconds(text) == 0.0
    assert sft_image_seconds(image) == pytest.approx(1000 * SFT_IMAGE_S_PER_EXAMPLE)
    gap = estimate_cost(image).train_seconds - estimate_cost(text).train_seconds
    assert gap == pytest.approx(1000 * SFT_IMAGE_S_PER_EXAMPLE)
