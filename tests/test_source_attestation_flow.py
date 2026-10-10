from __future__ import annotations

import copy

import pytest

import flash.runner.accounting.costs as runner_costs
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
from flash.snapshot.archive import (
    PUBLIC_PROVENANCE_KEY,
    TERMINAL_ATTESTATION_KEY,
    source_attestation,
)
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


def _status(*, attempt: int = 2, source_snapshot: dict | None = SOURCE_SNAPSHOT):
    from flash.runner.lifecycle.state import RunStatus

    return RunStatus(
        run_id="run-1",
        state="running",
        spec={},
        source_snapshot=source_snapshot,
        remote={"attempt": attempt},
    )


def _metrics(*, attempt: int = 2, descriptor: dict = SOURCE_SNAPSHOT) -> dict:
    return {
        "wall_seconds": 1.0,
        TERMINAL_ATTESTATION_KEY: source_attestation(
            descriptor,
            run_id="run-1",
            attempt=attempt,
        ),
    }


def test_terminal_source_metrics_require_exact_attempt_bound_evidence() -> None:
    from flash.runner.lifecycle.status import validate_terminal_source_metrics

    sanitized, verified_attempt = validate_terminal_source_metrics(_status(), _metrics())
    assert verified_attempt == 2
    assert TERMINAL_ATTESTATION_KEY not in sanitized
    assert sanitized[PUBLIC_PROVENANCE_KEY] == {
        "format_version": 1,
        "sha256": "a" * 64,
        "verified": True,
        "verified_attempt": 2,
    }


@pytest.mark.parametrize("case", ["missing", "digest", "revision", "stale"])
def test_terminal_source_metrics_reject_missing_mismatched_and_stale_evidence(case: str) -> None:
    from flash.runner.lifecycle.status import validate_terminal_source_metrics

    metrics = _metrics()
    if case == "missing":
        metrics.pop(TERMINAL_ATTESTATION_KEY)
    elif case == "stale":
        metrics = _metrics(attempt=1)
    else:
        metrics = copy.deepcopy(metrics)
        field = "sha256" if case == "digest" else "revision"
        metrics[TERMINAL_ATTESTATION_KEY][field] = "c" * len(
            metrics[TERMINAL_ATTESTATION_KEY][field]
        )
    with pytest.raises(RuntimeError):
        validate_terminal_source_metrics(_status(), metrics)


def test_descriptorless_attempt_stays_unverified_even_with_forged_evidence() -> None:
    from flash.runner.lifecycle.status import validate_terminal_source_metrics

    sanitized, verified_attempt = validate_terminal_source_metrics(
        _status(source_snapshot=None),
        _metrics(),
    )
    assert verified_attempt is None
    assert TERMINAL_ATTESTATION_KEY not in sanitized
    assert PUBLIC_PROVENANCE_KEY not in sanitized


def test_public_status_exposes_only_safe_source_projection() -> None:
    status = _status()
    status.source_verified_attempt = 2
    status.last_heartbeat = {
        "stage": "sft_step",
        PUBLIC_PROVENANCE_KEY: {
            "sha256": SOURCE_SNAPSHOT["sha256"],
            "verified": True,
            "verified_attempt": 99,
        },
    }
    public = status.to_dict()
    assert public[PUBLIC_PROVENANCE_KEY]["sha256"] == SOURCE_SNAPSHOT["sha256"]
    assert public[PUBLIC_PROVENANCE_KEY]["verified_attempt"] == 2
    assert PUBLIC_PROVENANCE_KEY not in public["last_heartbeat"]
    assert status.last_heartbeat[PUBLIC_PROVENANCE_KEY]["verified_attempt"] == 99
    rendered = repr(public)
    assert SOURCE_SNAPSHOT["archive_path"] not in rendered
    assert SOURCE_SNAPSHOT["revision"] not in rendered
    assert "source_snapshot" not in public


def test_recovery_completion_requires_attestation_before_done(monkeypatch, tmp_path) -> None:
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec.from_dict(
        {
            "run_id": "run-1",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "org/repo"},
            "gpu": {"max_wall_seconds": 3600},
        }
    )
    status = runner_state.RunStatus(
        run_id="run-1",
        state="running",
        spec=spec.to_dict(),
        source_snapshot=SOURCE_SNAPSHOT,
    )
    runner_state._save_status(status, _next_attempt=3)
    monkeypatch.setattr(runner_status, "_persist_metrics", lambda _spec, _metrics: 1.0)
    monkeypatch.setattr(runner_costs, "_status_estimated_charge", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(runner_state, "artifacts_dir", lambda _spec: "/artifacts")
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)

    with pytest.raises(RuntimeError):
        runner_reconciliation._compare_and_complete_remote(
            "run-1", None, spec, {"wall_seconds": 1.0}
        )
    assert runner_status.get_status("run-1").state == "running"

    assert (
        runner_reconciliation._compare_and_complete_remote("run-1", None, spec, _metrics()) is True
    )
    completed = runner_status.get_status("run-1")
    assert completed.state == "done"
    assert completed.source_verified_attempt == 2


def test_attach_freezes_top_level_descriptor(monkeypatch, tmp_path) -> None:
    from flash.core.spec import JobSpec
    from flash.runner.supervise.attach import _build_attach_context

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec.from_dict(
        {
            "run_id": "run-1",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "org/repo"},
        }
    )
    remote = {
        "provider": "runpod",
        "endpoint_id": "endpoint",
        "endpoint_name": "name",
        "key_fingerprint": "c" * 64,
        "job_id": "job",
        "attempt": 2,
        "started_ts": 100.0,
        "launch_claim_token": "token-2",
    }
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="run-1",
            state="running",
            spec=spec.to_dict(),
            source_snapshot=SOURCE_SNAPSHOT,
            remote=remote,
        )
    )

    context = _build_attach_context(spec, remote)
    assert context.source_snapshot == SOURCE_SNAPSHOT
    assert context.recovered_attempt == 2
    assert context.launch_claim_token == "token-2"
    # the launch token authorizes the attempt and must not ride into the polled provider handle.
    assert "launch_claim_token" not in context.handle.to_dict()


def test_descriptorless_replacement_is_blocked() -> None:
    from flash.runner.lifecycle.status import source_snapshot_from_status

    with pytest.raises(RuntimeError, match="descriptor-less attempts cannot be replaced"):
        source_snapshot_from_status(_status(source_snapshot=None), required=True)
