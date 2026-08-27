from __future__ import annotations

import copy
import json

import pytest

import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.core.spec import JobSpec
from flash.runner.lifecycle.protocol import AttemptRecord
from flash.runner.lifecycle.state import RunStatus
from flash.server.domain.registry import runs
from tests._helpers.source_snapshot import valid_source_snapshot

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64


def _spec(*, algorithm: str = "sft") -> JobSpec:
    return JobSpec.from_dict(
        {
            "run_id": f"lifecycle-{algorithm}",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": algorithm,
            "project": _PROJECT_ID,
            "train": {"epochs": 1, "hf_repo": "org/artifacts"},
            "gpu": {"max_wall_seconds": 3600},
        }
    )


def _remote(*, attempt: int = 0, endpoint_id: str = "endpoint-1") -> dict:
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": "job-1",
        "attempt": attempt,
        "fence": attempt + 1,
        "started_ts": 1.0,
    }


def _attempt(*, attempt: int = 0, fence: int | None = None) -> dict:
    fence = attempt + 1 if fence is None else fence
    return AttemptRecord(
        attempt_id=attempt,
        fence=fence,
        state="active",
        reserved_at=1.0,
        grant_deadline_at=30.0,
        work_deadline_at=3600.0,
        result_deadline_at=4500.0,
        run_deadline_at=3600.0,
    ).to_dict()


def _status(spec: JobSpec) -> RunStatus:
    return RunStatus(
        run_id=spec.run_id,
        state="queued",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        effective_preparation={"worker_spec": spec.to_internal_dict()},
        source_snapshot=valid_source_snapshot(),
        attempt=_attempt(),
    )


def _progress(
    run_id: str,
    *,
    attempt: int = 0,
    fence: int | None = None,
    sequence: int = 1,
    training_entered: bool = True,
    completed_steps: object = 1,
) -> dict:
    fence = attempt + 1 if fence is None else fence
    return {
        "run_id": run_id,
        "attempt_id": attempt,
        "fence": fence,
        "sequence": sequence,
        "training_entered": training_entered,
        "completed_steps": completed_steps,
        "receipt": {
            "path": f"training/{run_id}/{attempt}-{fence}-{sequence}.json",
            "revision": "a" * 40,
            "digest": f"{sequence:064x}",
        },
    }


def _report(monkeypatch, status: RunStatus) -> dict:
    bodies = []
    monkeypatch.setattr(runs, "_post", lambda _path, body: bodies.append(body) or True)
    assert runs.record_training_run(status=status) is True
    return bodies[-1]


def test_training_report_projects_only_conservative_lifecycle_booleans(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec()
    status = _status(spec)

    runner_state._save_status(status)
    assert _report(monkeypatch, status)["lifecycle"] == {
        "started": False,
        "progressed": False,
        "artifactsComplete": False,
        "cleanupComplete": False,
    }

    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)
    assert _report(monkeypatch, status)["lifecycle"] == {
        "started": True,
        "progressed": False,
        "artifactsComplete": False,
        "cleanupComplete": False,
    }

    assert runner_status.record_progress(
        status.run_id,
        _progress(status.run_id),
        attempt_id=0,
        fence=1,
    )
    status = runner_status.get_status(status.run_id)
    assert _report(monkeypatch, status)["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": False,
        "cleanupComplete": False,
    }

    status.state = "done"
    status.artifacts_dir = "/private/artifacts/lifecycle-sft"
    status.source_verified_attempt = 0
    runner_state._save_status(status)
    body = _report(monkeypatch, status)
    assert body["adapterRef"] == spec.run_id
    assert body["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": True,
        "cleanupComplete": False,
    }
    assert all(isinstance(value, bool) for value in body["lifecycle"].values())

    status.lifecycle_started_attempt = 0
    status.lifecycle_progressed_attempt = 0
    status.remote["allocated_gpu"] = "A100 PCIe"
    status.cleanup_confirmed_remote = status.remote
    status.realized_cost_remote = status.remote
    status.remote = None
    runner_state._save_status(status, _cleanup_remotes=None)
    public = status.to_dict()
    assert "lifecycle_started_attempt" not in public
    assert "lifecycle_progressed_attempt" not in public
    assert "cleanup_confirmed_remote" not in public
    assert "realized_cost_remote" not in public
    report = _report(monkeypatch, status)
    assert report["gpuType"] == "A100 PCIe"
    assert report["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": True,
        "cleanupComplete": True,
    }
    assert "lastHeartbeat" not in report
    assert "gpuStatus" not in report

    runner_costs.record_realized_cost(
        status.run_id,
        realized_cost_usd=1.0,
        reconciled_at=10.0,
    )
    reconciled = runner_status.get_status(status.run_id)
    assert reconciled.realized_cost_remote is None
    assert _report(monkeypatch, reconciled)["lifecycle"]["cleanupComplete"] is True


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_progress_projects_each_training_algorithm(monkeypatch, tmp_path, algorithm):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / algorithm))
    spec = _spec(algorithm=algorithm)
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)
    assert runner_status.record_progress(
        status.run_id,
        _progress(status.run_id),
        attempt_id=0,
        fence=1,
    )

    lifecycle = _report(monkeypatch, runner_status.get_status(status.run_id))["lifecycle"]
    assert lifecycle["started"] is True
    assert lifecycle["progressed"] is True


@pytest.mark.parametrize(
    ("attempt", "completed_steps", "training_entered"),
    [
        (0, 0, True),
        (0, 1, False),
        (0, True, True),
        (0, 1.0, True),
        (1, 1, True),
    ],
)
def test_progress_requires_exact_attempt_and_positive_integer_work(
    monkeypatch,
    tmp_path,
    attempt,
    completed_steps,
    training_entered,
):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / str(attempt)))
    spec = _spec()
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)
    accepted = runner_status.record_progress(
        status.run_id,
        _progress(
            status.run_id,
            attempt=attempt,
            completed_steps=completed_steps,
            training_entered=training_entered,
        ),
        attempt_id=attempt,
        fence=attempt + 1,
    )
    if attempt != 0:
        assert accepted is False

    lifecycle = _report(monkeypatch, runner_status.get_status(status.run_id))["lifecycle"]
    assert lifecycle["started"] is True
    assert lifecycle["progressed"] is False


def test_progress_evidence_is_monotonic_across_later_observations(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec(algorithm="opd")
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)

    assert runner_status.record_progress(
        spec.run_id,
        _progress(spec.run_id),
        attempt_id=0,
        fence=1,
    )
    assert runner_status.record_progress(
        spec.run_id,
        _progress(spec.run_id, sequence=2, completed_steps=1),
        attempt_id=0,
        fence=1,
    )

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.lifecycle_progressed_attempt == 0
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is True


def test_progress_remains_true_across_provider_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec()
    status = _status(spec)
    status.state = "running"
    status.remote = _remote(attempt=0)
    status.lifecycle_started_attempt = 0
    runner_state._save_status(status)
    assert runner_status.record_progress(
        spec.run_id,
        _progress(spec.run_id),
        attempt_id=0,
        fence=1,
    )

    with runner_state._status_guard(spec.run_id):
        persisted = runner_status.get_status(spec.run_id)
        persisted.attempt = _attempt(attempt=1)
        persisted.remote = _remote(attempt=1, endpoint_id="endpoint-2")
        runner_state._save_status_unlocked(persisted)

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.lifecycle_started_attempt == 0
    assert persisted.lifecycle_progressed_attempt == 0
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is True


def test_lifecycle_fails_closed_without_exact_durable_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spec()
    status = _status(spec)
    status.state = "done"
    status.remote = _remote()
    status.lifecycle_started_attempt = 0
    status.lifecycle_progressed_attempt = 0
    status.artifacts_dir = "/private/artifacts/lifecycle-sft"
    status.source_verified_attempt = 0
    runner_state._save_status(status)

    stale = copy.deepcopy(status)
    status.error = "newer durable write"
    runner_state._save_status(status)

    assert _report(monkeypatch, stale)["lifecycle"] == {
        "started": False,
        "progressed": False,
        "artifactsComplete": False,
        "cleanupComplete": False,
    }


def test_cleanup_requires_no_active_remote_and_an_empty_private_set(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spec()
    status = _status(spec)
    status.state = "cancelled"
    status.remote = _remote()
    runner_state._save_status(status, _cleanup_remotes=[_remote()])

    assert _report(monkeypatch, status)["lifecycle"]["cleanupComplete"] is False

    status.remote = None
    runner_state._save_status(status)
    assert _report(monkeypatch, status)["lifecycle"]["cleanupComplete"] is False

    raw = runner_status._load_status_json(status.run_id)
    raw[runner_state._CLEANUP_REMOTES_KEY] = {"invalid": "shape"}
    path = runner_state.runs_file_path(status.run_id, ".json")
    with open(path, "w") as handle:
        json.dump(raw, handle)
    status.report_sequence = raw["report_sequence"]
    assert _report(monkeypatch, status)["lifecycle"]["cleanupComplete"] is False
