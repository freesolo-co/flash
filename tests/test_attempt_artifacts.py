from __future__ import annotations

import json

import pytest

from flash.providers.artifacts import attempts
from flash.runner.lifecycle.protocol import (
    ProgressRecord,
    ResultManifest,
    canonical_bytes,
    digest_record,
    progress_path,
    result_path,
)

SOURCE = {
    "kind": "flash-source-snapshot",
    "format_version": 1,
    "archive_path": "source/" + "a" * 64 + "/flash-source.zip",
    "sha256": "a" * 64,
    "size": 1,
    "revision": "b" * 40,
}
ATTESTATION = {
    "kind": "flash-source-attestation",
    "format_version": 1,
    "sha256": "a" * 64,
    "revision": "b" * 40,
    "run_id": "run-1",
    "attempt": 2,
    "fence": 9,
}


def _progress(sequence: int, previous: ProgressRecord | None = None) -> ProgressRecord:
    return ProgressRecord(
        run_id="run-1",
        phase_namespace="rl",
        attempt_id=2,
        fence=9,
        sequence=sequence,
        previous_digest=digest_record(previous.to_dict()) if previous else None,
        occurred_at=100.0 + sequence,
        kind="attempt_started" if sequence == 1 else "progressed",
        phase="boot" if sequence == 1 else "rl_step",
        training_entered=sequence > 1,
        completed_steps=max(0, sequence - 1),
    )


def _result(**updates) -> ResultManifest:
    values = {
        "run_id": "run-1",
        "phase_namespace": "rl",
        "attempt_id": 2,
        "fence": 9,
        "outcome": "succeeded",
        "failure_class": None,
        "started_at": 100.0,
        "finished_at": 120.0,
        "training_entered": True,
        "completed_steps": 1,
        "metrics": {"step": 1},
        "checkpoint": {},
        "artifacts": {"adapter": "published"},
        "source_attestation": ATTESTATION,
        "diagnostics": {},
    }
    values.update(updates)
    return ResultManifest(**values)


def test_reads_latest_verified_progress_and_single_result(monkeypatch) -> None:
    first = _progress(1)
    second = _progress(2, first)
    result = _result()
    payloads = {
        progress_path(first): canonical_bytes(first.to_dict()),
        progress_path(second): canonical_bytes(second.to_dict()),
        result_path(result): canonical_bytes(result.to_dict()),
    }
    monkeypatch.setattr(attempts, "_repo_snapshot", lambda _repo: ("c" * 40, list(payloads)))
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )
    monkeypatch.setattr(attempts.time, "time", lambda: 130.0)

    observed = attempts.read_attempt_artifacts(
        "org/repo",
        phase="rl",
        run_id="run-1",
        attempt_id=2,
        fence=9,
        source_snapshot=SOURCE,
    )

    assert observed.progress["sequence"] == 2
    assert observed.progress["observed_at"] == 130.0
    assert observed.result["outcome"] == "succeeded"
    assert observed.result["receipt"]["path"] == result_path(result)


def test_rejects_conflicting_results(monkeypatch) -> None:
    first = _result(outcome="failed", failure_class="worker", metrics={})
    second = _result(outcome="failed", failure_class="oom", metrics={})
    payloads = {
        result_path(first): canonical_bytes(first.to_dict()),
        result_path(second): canonical_bytes(second.to_dict()),
    }
    monkeypatch.setattr(attempts, "_repo_snapshot", lambda _repo: ("c" * 40, list(payloads)))
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )

    with pytest.raises(attempts.AttemptArtifactError, match="conflicting"):
        attempts.read_attempt_artifacts(
            "org/repo",
            phase="rl",
            run_id="run-1",
            attempt_id=2,
            fence=9,
            source_snapshot=SOURCE,
        )


def test_stale_fence_records_are_not_visible(monkeypatch) -> None:
    stale = ProgressRecord(
        **{
            **_progress(1).to_dict(),
            "fence": 8,
        }
    )
    payloads = {progress_path(stale): json.dumps(stale.to_dict()).encode()}
    monkeypatch.setattr(attempts, "_repo_snapshot", lambda _repo: ("c" * 40, list(payloads)))
    monkeypatch.setattr(
        attempts,
        "_download_bytes",
        lambda _repo, path, *, revision: payloads[path],
    )

    observed = attempts.read_attempt_artifacts(
        "org/repo",
        phase="rl",
        run_id="run-1",
        attempt_id=2,
        fence=9,
        source_snapshot=SOURCE,
    )

    assert observed.progress is None
    assert observed.result is None
