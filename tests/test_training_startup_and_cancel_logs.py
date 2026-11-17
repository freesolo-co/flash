"""Cold-start visibility and user-cancel labelling in run logs and run status; no paid resources."""

from __future__ import annotations

import io
import time

import pytest

import flash.providers._lifecycle.instances.poll_instance as poll_instance
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.supervise.deploy as runner_deploy
from flash.core.spec import JobSpec
from flash.providers._lifecycle.instances.poll_instance import (
    InstancePollAdapter,
    poll_instance_job,
)
from flash.providers._lifecycle.instances.startup import StartupTracker
from flash.runner.lifecycle.startup import COLD_START_NOTICE, record_startup_stage

RUN_ID = "flash-startup-test"
LAUNCH_TS = 1000.0


class _StopPolling(Exception):
    """Ends a scripted poll after its last tick."""


@pytest.fixture
def clock(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    now = [LAUNCH_TS]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(
        poll_instance, "pause", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    return now


def _save_run(state: str = "running") -> None:
    spec = JobSpec(run_id=RUN_ID, model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=RUN_ID, state=state, spec=spec.to_dict())
    )


def _adapter(fetch_instance) -> InstancePollAdapter:
    def nothing(**_kwargs):
        return None

    return InstancePollAdapter(
        instance_id="bid_test",
        run_id=RUN_ID,
        current_attempt=0,
        launch_ts=LAUNCH_TS,
        done_reader=nothing,
        marker_reader=nothing,
        metrics_reader=nothing,
        fetch_instance=fetch_instance,
        poll_error_exceptions=(),
        status_field="status",
        running_status="running",
        dead_states=frozenset({"stopped", "error", "preempting"}),
        missing_dead_threshold=3,
        early_liveness_alive=lambda: True,
        read_current_error=lambda: None,
        stamp_cost_and_notes=lambda *_a, **_kw: None,
        failure_detail=lambda _marker: "Spot worker stopped without a successful terminal marker",
        load_timeout_detail=lambda status, elapsed: f"{status} for {elapsed}",
        first_liveness_detail=lambda elapsed, limit: f"{elapsed} > {limit}",
    )


def _poll(adapter, log, heartbeat_reader=None):
    return poll_instance_job(
        adapter,
        log=log,
        interval_s=15.0,
        heartbeat_reader=heartbeat_reader,
        setup_grace_s=10_000,
        stall_after_s=10_000,
        first_liveness_s=10_000,
        load_timeout_s=10_000,
    )


def test_startup_phase_is_logged_and_in_status_until_first_training_heartbeat(clock):
    _save_run()
    # tick -> (instance status, heartbeat). 8 quiet model-load ticks (120 s) earn one reminder.
    script = [
        ("starting", None),
        ("running", None),
        ("running", {"stage": "sft_model_load", "attempt": 0, "ts": LAUNCH_TS + 30}),
        *[("running", {"stage": "sft_model_load", "attempt": 0, "ts": LAUNCH_TS + 30})] * 9,
        ("running", {"stage": "sft_step", "step": 1, "attempt": 0, "ts": LAUNCH_TS + 200}),
    ]
    tick = [-1]
    startup_seen = []

    def fetch():
        tick[0] += 1
        startup_seen.append(runner_status.get_status(RUN_ID).to_dict()["startup"])
        if tick[0] >= len(script):
            raise _StopPolling
        return {"status": script[tick[0]][0]}

    def heartbeat():
        return script[min(tick[0], len(script) - 1)][1]

    log = io.StringIO()
    with pytest.raises(_StopPolling):
        _poll(_adapter(fetch), log, heartbeat)
    text = log.getvalue()

    assert COLD_START_NOTICE in text
    assert "10-12 minutes" in COLD_START_NOTICE
    assert "not a stall" in COLD_START_NOTICE
    assert "starting: booting instance; instance bid_test is starting (0m00s since launch)" in text
    assert "starting: instance running; pulling the worker image and starting the worker" in text
    assert "starting: loading model weights; worker stage=sft_model_load" in text
    assert "starting: still loading model weights" in text
    assert "started: first training heartbeat (stage=sft_step step=1)" in text
    # the notice and stage lines come before the first training heartbeat, never after it
    assert text.index(COLD_START_NOTICE) < text.index("started: first training heartbeat")

    # status: no stage before the first poll, then each observed stage, cleared by training
    assert startup_seen[0] is None
    assert startup_seen[1]["stage"] == "booting_instance"
    assert startup_seen[2]["stage"] == "starting_worker"
    assert startup_seen[3]["stage"] == "loading_model"
    assert startup_seen[3]["detail"].startswith("starting: loading model weights")
    assert startup_seen[3]["typical_cold_start"] == "10-12 minutes"
    assert startup_seen[-1] is None


def test_startup_record_is_hidden_once_the_run_is_terminal(clock):
    _save_run()
    record_startup_stage(RUN_ID, "booting_instance", attempt=0)
    assert runner_status.get_status(RUN_ID).to_dict()["startup"]["stage"] == "booting_instance"
    status = runner_status.get_status(RUN_ID)
    status.state = "failed"
    runner_state._save_status(status)
    assert runner_status.get_status(RUN_ID).to_dict()["startup"] is None


def test_reattached_poll_does_not_narrate_a_finished_cold_start(clock):
    _save_run()
    status = runner_status.get_status(RUN_ID)
    status.last_heartbeat = {"stage": "sft_step", "step": 40, "attempt": 0, "ts": LAUNCH_TS}
    runner_state._save_status(status)
    lines = []
    tracker = StartupTracker(
        run_id=RUN_ID, instance_id="bid", attempt=0, launch_ts=LAUNCH_TS, say=lines.append
    )
    tracker.observe_instance("running", running=True, dead=False)
    assert not tracker.active
    assert lines == []


def test_user_cancel_is_reported_as_cancel_not_preemption(clock):
    _save_run(state="cancelled")
    log = io.StringIO()
    # the cancel's teardown makes the provider answer `preempting` for the stopping bid
    result = _poll(_adapter(lambda: {"status": "preempting"}), log)
    text = log.getvalue()
    assert not result.ok
    assert result.detail == "cancelled by user request"
    assert "run cancelled by user request; instance bid_test is being torn down" in text
    assert "preempting" not in text
    assert "instance gone" not in text


def test_real_preemption_is_still_reported_as_preemption(clock):
    _save_run()
    log = io.StringIO()
    result = _poll(_adapter(lambda: {"status": "preempting"}), log)
    text = log.getvalue()
    assert result.failure == "job_preempted"
    assert "instance bid_test: preempting" in text
    assert "cancelled" not in text


def test_cancel_run_logs_user_cancel_and_records_reason(clock):
    _save_run()
    status = runner_deploy.cancel_run(RUN_ID)
    assert status.state == "cancelled"
    assert status.cancel_reason.startswith("cancelled by user request at ")
    with open(runner_state.runs_file_path(RUN_ID, ".log"), encoding="utf-8") as handle:
        text = handle.read()
    assert "cancelled by user request at " in text
    assert "no instance is running yet" in text
    assert "preempt" not in text
    public = status.to_dict()
    assert public["cancel_reason"] == status.cancel_reason
    assert public["startup"] is None


def test_cli_surfaces_startup_stage_and_cancel_reason():
    from flash.cli.commands.ops.runs import _log_follow_progress
    from flash.cli.ui.render import run_status

    starting = {
        "run_id": RUN_ID,
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
        "startup": {
            "stage": "booting_instance",
            "detail": "starting: booting instance",
            "typical_cold_start": "10-12 minutes",
        },
    }
    _, progress = _log_follow_progress(starting, "running")
    assert "starting: booting instance" in progress
    panel = run_status(starting)
    assert "starting: booting instance" in panel
    assert "cold start typically takes 10-12 minutes" in panel

    cancelled = {
        **starting,
        "state": "cancelled",
        "startup": None,
        "cancel_reason": "cancelled by user request at 2026-11-16 12:00:00 UTC",
    }
    assert "cancelled by user request at 2026-11-16 12:00:00 UTC" in run_status(cancelled)
