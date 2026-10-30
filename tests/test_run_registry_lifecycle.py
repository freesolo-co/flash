from __future__ import annotations

import copy
import json

import pytest

import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import RunStatus
from flash.server.domain.registry import runs
from tests._helpers.source_snapshot import valid_source_snapshot

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
from tests._helpers.modal import modal_remote


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
    return modal_remote(instance_id=endpoint_id, attempt=attempt)


def _status(spec: JobSpec) -> RunStatus:
    return RunStatus(
        run_id=spec.run_id,
        state="queued",
        spec=spec.to_dict(),
        platform_context={"org_id": "org-1"},
        effective_preparation={"worker_spec": spec.to_internal_dict()},
        source_snapshot=valid_source_snapshot(),
    )


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
    body = _report(monkeypatch, status)
    assert body["lifecycle"] == {
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

    runner_status.record_heartbeat(
        status.run_id,
        {"attempt": 0, "stage": "sft_step", "step": 1},
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
    assert "adapterRef" not in body
    assert body["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": True,
        "cleanupComplete": False,
    }
    assert set(body["lifecycle"]) == {
        "started",
        "progressed",
        "artifactsComplete",
        "cleanupComplete",
    }
    assert all(isinstance(value, bool) for value in body["lifecycle"].values())

    status.state = "deployed"
    runner_state._save_status(status)
    deployed_lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert deployed_lifecycle["artifactsComplete"] is True
    assert deployed_lifecycle["cleanupComplete"] is False

    status.state = "done"
    status.lifecycle_started_attempt = 0
    status.lifecycle_progressed_attempt = 0
    status.remote["allocated_gpu"] = "A100 SXM"
    status.cleanup_confirmed_remote = status.remote
    status.billing_remote = status.remote
    status.remote = None
    runner_state._save_status(status, _cleanup_remotes=None)
    assert "lifecycle_started_attempt" not in status.to_dict()
    assert "lifecycle_progressed_attempt" not in status.to_dict()
    assert "cleanup_confirmed_remote" not in status.to_dict()
    assert "billing_remote" not in status.to_dict()
    report = _report(monkeypatch, status)
    assert report["gpuType"] == "A100 SXM"
    assert report["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": True,
        "cleanupComplete": True,
    }

    status.state = "deployed"
    runner_state._save_status(status)
    assert _report(monkeypatch, status)["lifecycle"]["cleanupComplete"] is True

    runner_costs.record_billing_state(status.run_id, billing_state="charged")
    charged = runner_status.get_status(status.run_id)
    assert charged.billing_remote is None
    assert _report(monkeypatch, charged)["lifecycle"] == {
        "started": True,
        "progressed": True,
        "artifactsComplete": True,
        "cleanupComplete": True,
    }


@pytest.mark.parametrize(
    ("algorithm", "stage"),
    [("sft", "sft_step"), ("grpo", "rl_step")],
)
def test_progress_projects_each_training_algorithm(
    monkeypatch,
    tmp_path,
    algorithm,
    stage,
):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / algorithm))
    spec = _spec(algorithm=algorithm)
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)
    runner_status.record_heartbeat(
        status.run_id,
        {"attempt": 0, "stage": stage, "step": 1},
    )
    status = runner_status.get_status(status.run_id)

    lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert lifecycle["started"] is True
    assert lifecycle["progressed"] is True


@pytest.mark.parametrize(
    ("algorithm", "heartbeat"),
    [
        ("sft", {"attempt": 0, "stage": "sft_step", "step": 0}),
        ("sft", {"attempt": 0, "stage": "done", "step": 0}),
        ("sft", {"attempt": 1, "stage": "done", "step": 1}),
        ("sft", {"attempt": 0, "stage": "sft_step", "step": True}),
        ("sft", {"attempt": 0, "stage": "sft_step", "step": 1.0}),
        ("sft", {"attempt": 1, "stage": "sft_step", "step": 1}),
        ("sft", {"attempt": 0, "stage": "rl_step", "step": 1}),
        ("grpo", {"attempt": 0, "stage": "sft_step", "step": 1}),
    ],
)
def test_progress_requires_exact_algorithm_attempt_and_positive_integer_step(
    monkeypatch,
    tmp_path,
    algorithm,
    heartbeat,
):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / algorithm))
    spec = _spec(algorithm=algorithm)
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)
    runner_status.record_heartbeat(status.run_id, heartbeat)
    status = runner_status.get_status(status.run_id)

    lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert lifecycle["started"] is True
    assert lifecycle["progressed"] is False


def test_terminal_done_heartbeat_preserves_exact_progress_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec(algorithm="grpo")
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)

    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 0, "stage": "done", "step": 1},
    )

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.lifecycle_progressed_attempt == 0
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is True


def test_checkpoint_heartbeat_preserves_exact_progress_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec(algorithm="grpo")
    status = _status(spec)
    status.state = "running"
    status.remote = _remote()
    runner_state._save_status(status)

    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 0, "stage": "rl_step", "step": 1},
    )
    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 0, "stage": "checkpoint_uploaded", "step": 1},
    )

    persisted = runner_status.get_status(spec.run_id)
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is True
    assert "lifecycle_progress" not in persisted.to_dict()["last_heartbeat"]


def test_progress_remains_true_across_provider_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec(algorithm="sft")
    status = _status(spec)
    status.state = "running"
    status.remote = _remote(attempt=0)
    status.lifecycle_started_attempt = 0
    runner_state._save_status(status)

    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 0, "stage": "sft_step", "step": 1},
    )
    assert runner_status._update(
        spec.run_id,
        "running",
        remote=_remote(attempt=1, endpoint_id="endpoint-2"),
        lifecycle_started_attempt=1,
    )
    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 1, "stage": "booting"},
    )

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.lifecycle_started_attempt == 0
    assert persisted.lifecycle_progressed_attempt == 0
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is True


def test_checkpoint_heartbeat_rejects_stale_or_mismatched_progress_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    spec = _spec(algorithm="grpo")
    status = _status(spec)
    status.state = "running"
    status.remote = _remote(attempt=1)
    runner_state._save_status(status)

    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 0, "stage": "rl_step", "step": 1},
    )
    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 1, "stage": "checkpoint_uploaded", "step": 1},
    )
    persisted = runner_status.get_status(spec.run_id)
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is False

    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 1, "stage": "sft_step", "step": 1},
    )
    runner_status.record_heartbeat(
        spec.run_id,
        {"attempt": 1, "stage": "checkpoint_uploaded", "step": 1},
    )
    persisted = runner_status.get_status(spec.run_id)
    assert _report(monkeypatch, persisted)["lifecycle"]["progressed"] is False


def test_lifecycle_fails_closed_without_exact_durable_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = _spec()
    status = _status(spec)
    status.state = "done"
    status.remote = _remote()
    status.last_heartbeat = {"attempt": 0, "stage": "sft_step", "step": 1}
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

    lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert lifecycle["cleanupComplete"] is False

    status.remote = None
    runner_state._save_status(status)
    lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert lifecycle["cleanupComplete"] is False

    raw = runner_status._load_status_json(status.run_id)
    raw[runner_state._CLEANUP_REMOTES_KEY] = {"invalid": "shape"}
    path = runner_state.runs_file_path(status.run_id, ".json")
    with open(path, "w") as handle:
        json.dump(raw, handle)
    status.report_sequence = raw["report_sequence"]
    lifecycle = _report(monkeypatch, status)["lifecycle"]
    assert lifecycle["cleanupComplete"] is False


def test_status_report_sends_durable_sequence_for_backend_ordering(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    status = _status(_spec())
    runner_state._save_status(status)
    first = _report(monkeypatch, status)
    assert first["reportSequence"] == 1
    assert "report_sequence" not in status.to_dict()

    # Reload from disk as another process or a restarted reporter would do.
    reloaded = runner_status.get_status(status.run_id)
    runner_state._save_status(reloaded)
    assert _report(monkeypatch, reloaded)["reportSequence"] == 2
    assert _report(monkeypatch, status)["reportSequence"] == 1


@pytest.mark.parametrize("sequence", [0, None, True, -1, "7", 1.5, {"corrupt": True}])
def test_status_report_rejects_invalid_sequence_before_http_post(monkeypatch, sequence):
    status = _status(_spec())
    status.report_sequence = sequence
    monkeypatch.setattr(
        runs, "_post", lambda *_args: pytest.fail("invalid sequence must not reach the backend")
    )
    assert runs.record_training_run(status=status) is False


def test_legacy_reconciliation_record_preserves_pending_customer_charge(monkeypatch, tmp_path):
    from flash.server.billing import charges

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    status = _status(_spec())
    status.state = "done"
    status.cost_usd = 2.5
    status.billing_context = {"org_id": "org-1"}
    status.billing_state = "pending"
    runner_state._save_status(status)
    path = runner_state.runs_file_path(status.run_id, ".json")
    with open(path) as file:
        raw = json.load(file)
    remote = {**_remote(), "allocated_gpu": "A100 SXM"}
    raw.pop("billing_remote")
    raw.update(realized_cost_remote=remote, realized_cost_usd=1.0, reconciled_at=10.0)
    with open(path, "w") as file:
        json.dump(raw, file)

    loaded = runner_status.get_status(status.run_id)
    assert loaded.billing_remote == remote
    assert not {"realized_cost_usd", "reconciled_at", "billing_remote"} & loaded.to_dict().keys()
    report = _report(monkeypatch, loaded)
    assert "realizedCostUsd" not in report
    assert report["gpuType"] == "A100 SXM"
    assert report["lifecycle"]["started"] is True

    requests = []
    monkeypatch.setattr(charges, "_post_billing", lambda **kwargs: requests.append(kwargs) or {})
    charges.charge_completed_run(internal_key="internal-test", status=loaded)
    assert requests[0]["body"]["costCents"] == 250
    assert requests[0]["body"]["gpu"] == "A100 SXM"
    assert requests[0]["body"]["provider"] == "modal"

    runner_costs.record_billing_state(status.run_id, billing_state="charged")
    saved = runner_status._load_status_json(status.run_id)
    assert saved["billing_remote"] is None
    assert not {"realized_cost_usd", "reconciled_at", "realized_cost_remote"} & saved.keys()
    assert runner_status.get_status(status.run_id).billing_state == "charged"


def test_cleared_billing_handle_is_not_restored_from_legacy_field():
    raw = runner_state._status_storage_dict(_status(_spec()))
    raw["realized_cost_remote"] = _remote()
    assert runner_status._run_status_from_json(raw).billing_remote is None
