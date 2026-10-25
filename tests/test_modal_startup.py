"""Queued reservations do not accrue worker liveness time or estimated compute cost."""

import json

import pytest

from flash.core.spec import GpuSpec, JobSpec, TrainSpec
from flash.providers.core.base import JobHandle
from flash.providers.modal import api, jobs, startup
from tests._helpers.modal import modal_remote


@pytest.fixture
def polling(monkeypatch):
    clock = [100.0]
    artifacts = {}
    paths = []
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        jobs.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def reader(repo, path, **kwargs):
        paths.append(path)
        return lambda **kw: artifacts.get(path.rsplit("/", 1)[-1])

    monkeypatch.setattr(jobs, "make_hf_text_reader", reader)
    monkeypatch.setattr(startup, "make_hf_text_reader", reader)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *a, **k: lambda **kw: None)
    monkeypatch.setattr(api, "get_instance", lambda *a, **k: {"status": "pending"})
    raw = modal_remote(started_ts=100.0, hourly_usd=3.6)
    spec = JobSpec(
        run_id=raw["run_id"],
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo=raw["hf_repo"]),
        gpu=GpuSpec(type="L4"),
    )
    return clock, artifacts, paths, JobHandle.from_dict(raw), spec


def test_queue_wait_uses_capacity_budget_not_worker_liveness(polling):
    clock, _, _, handle, spec = polling
    result = jobs.poll(handle, spec, deadline_at=10000)
    assert result.failure == "no_capacity"
    assert 1800 < clock[0] - 100 < 1900


def test_shutdown_of_reattached_poll_preserves_worker(monkeypatch, polling):
    import threading

    from flash._internal.background import Supervisor
    from flash.providers.modal.provider import PROVIDER

    _, _, _, handle, spec = polling
    observed = threading.Event()
    monkeypatch.setattr(
        api, "get_instance", lambda *a, **kw: observed.set() or {"status": "pending"}
    )
    monkeypatch.setattr(api, "terminate", lambda *_: pytest.fail("shutdown destroyed worker"))
    monkeypatch.setattr(
        "flash.providers.modal.checkpoint_upload.cleanup",
        lambda *_: pytest.fail("shutdown destroyed checkpoint uploads"),
    )
    owner = Supervisor()
    try:
        owner.submit(
            ("run", spec.run_id),
            "training",
            lambda: PROVIDER.poll_attempt(handle, spec, _deadline_at=10000),
        )
        assert observed.wait(1)
        assert owner.close(1), "shutdown must interrupt the reattached poll's wait"
    finally:
        owner.close(1)


@pytest.mark.parametrize("reattach", [False, True])
@pytest.mark.parametrize("log_enabled", [False, True])
def test_long_queue_then_bootstrap_can_finish_with_queue_excluded_from_cost(
    monkeypatch, polling, reattach, log_enabled, tmp_path
):
    clock, artifacts, paths, handle, spec = polling
    if reattach:
        clock[0] = 1400

    def instance(*args, **kwargs):
        if clock[0] >= 1400:
            artifacts["modal_started_attempt0.json"] = json.dumps(
                {"run_id": spec.run_id, "attempt": 0, "ts": 1400}
            )
            artifacts["console_sft_attempt0.txt"] = "loading model"
        if clock[0] >= 2600:
            artifacts["modal_attempt0.json"] = json.dumps(
                {
                    "run_id": spec.run_id,
                    "attempt": 0,
                    "ts": 2600,
                    "ok": True,
                    "error": "",
                    "retriable": False,
                }
            )
            artifacts["metrics.json"] = json.dumps({"train_tokens": 10})
        return {"status": "pending"}

    monkeypatch.setattr(api, "get_instance", instance)
    log_path = tmp_path / "poll.log"
    with log_path.open("w") as log:
        result = jobs.poll(handle, spec, log=log if log_enabled else None, deadline_at=10000)
        # The runner supplies a file, and startup must be logged only once across poll ticks.
        assert log_path.read_text().count(
            "allocation and bootstrap: 1300.0s to worker start"
        ) == int(log_enabled)
    assert result.ok
    assert result.metrics["cost_usd"] == pytest.approx(1.2)
    assert result.metrics["notes"]["cost_basis"] == "bootstrap_resource_estimate"
    assert f"sft/{spec.run_id}/console_sft_attempt0.txt" in paths


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "other"},
        {"attempt": 1},
        {"attempt": False},
        {"ts": 99},
        {"ts": 201},
        {"ts": float("nan")},
        {"ts": True},
    ],
)
def test_start_marker_rejects_wrong_identity_or_invalid_time(monkeypatch, change):
    marker = {"run_id": "flash-test", "attempt": 0, "ts": 150, **change}
    monkeypatch.setattr(startup.time, "time", lambda: 200)
    monkeypatch.setattr(startup, "make_hf_text_reader", lambda *a, **k: lambda: json.dumps(marker))
    assert startup.start_time_reader(modal_remote(started_ts=100))() is None


def test_bootstrap_publishes_start_identity_before_setup(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import huggingface_hub

    from flash.providers._lifecycle.bootstrapping import bootstrap

    uploads = []

    def upload(**kwargs):
        uploads.append(
            (kwargs["path_in_repo"], json.loads(Path(kwargs["path_or_fileobj"]).read_text()))
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(upload_file=upload))
    monkeypatch.setattr(bootstrap.time, "time", lambda: 150)
    bootstrap._publish_started(
        {
            "run_id": "queued-run",
            "attempt": 3,
            "hf_prefix": "sft/queued-run",
            "hf_repo": "org/artifacts",
            "env": {},
            "deadline_at": 200,
            "run_created_at": 100,
            "run_max_wall_seconds": 100,
        }
    )
    assert uploads == [
        (
            "sft/queued-run/modal_started_attempt3.json",
            {"run_id": "queued-run", "attempt": 3, "ts": 150},
        )
    ]


@pytest.mark.parametrize("returncode", [1, 2, 42, 127])
def test_nonzero_process_exit_is_not_preemption_when_failure_upload_is_missing(
    monkeypatch, polling, returncode
):
    _, _, _, handle, spec = polling
    monkeypatch.setattr(
        api, "get_instance", lambda *_a, **_k: {"status": "terminated", "returncode": returncode}
    )
    result = jobs.poll(handle, spec, deadline_at=10000)
    assert result.failure == "job_failed"
    assert f"exit code {returncode}" in result.detail


@pytest.mark.parametrize("returncode", [None, 0, -1, 137, 143, True, "1"])
def test_unknown_or_signal_exit_keeps_host_loss_recovery(monkeypatch, polling, returncode):
    _, _, _, handle, spec = polling
    monkeypatch.setattr(
        api, "get_instance", lambda *_a, **_k: {"status": "terminated", "returncode": returncode}
    )
    assert jobs.poll(handle, spec, deadline_at=10000).failure == "job_preempted"


def test_explicit_retriable_worker_evidence_overrides_process_exit(monkeypatch, polling):
    _, _, _, handle, spec = polling
    monkeypatch.setattr(
        api, "get_instance", lambda *_a, **_k: {"status": "terminated", "returncode": 1}
    )
    monkeypatch.setattr(
        jobs,
        "heartbeat_reader_for",
        lambda *_a, **_k: (
            lambda **_kw: {
                "stage": "error_sft",
                "ts": 100,
                "attempt": 0,
                "retriable": True,
            }
        ),
    )
    assert jobs.poll(handle, spec, deadline_at=10000).failure == "job_preempted"
