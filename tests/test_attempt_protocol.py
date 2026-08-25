from __future__ import annotations

import math

import pytest

from flash.runner.lifecycle.protocol import (
    AttemptRecord,
    ProgressRecord,
    ResultManifest,
    bounded_json,
    digest_record,
    progress_path,
    receipt,
    result_path,
)


def _attempt(**updates) -> AttemptRecord:
    values = {
        "attempt_id": 2,
        "fence": 9,
        "state": "reserved",
        "reserved_at": 100.0,
        "grant_deadline_at": 120.0,
        "work_deadline_at": 180.0,
        "run_deadline_at": 200.0,
        "result_deadline_at": 230.0,
    }
    values.update(updates)
    return AttemptRecord(**values)


def _progress(**updates) -> ProgressRecord:
    values = {
        "run_id": "flash-1",
        "phase_namespace": "rl",
        "attempt_id": 2,
        "fence": 9,
        "sequence": 1,
        "previous_digest": None,
        "occurred_at": 110.0,
        "kind": "attempt_started",
        "phase": "boot",
        "training_entered": False,
        "completed_steps": 0,
    }
    values.update(updates)
    return ProgressRecord(**values)


def _result(**updates) -> ResultManifest:
    values = {
        "run_id": "flash-1",
        "phase_namespace": "rl",
        "attempt_id": 2,
        "fence": 9,
        "outcome": "succeeded",
        "failure_class": None,
        "started_at": 110.0,
        "finished_at": 170.0,
        "training_entered": True,
        "completed_steps": 3,
        "metrics": {"step": 3},
        "checkpoint": {"final": "ok"},
        "artifacts": {"adapter": "sha256:abc"},
        "source_attestation": {"run_id": "flash-1"},
        "diagnostics": {},
    }
    values.update(updates)
    return ResultManifest(**values)


def test_attempt_deadlines_are_fixed_and_monotonic() -> None:
    record = _attempt()
    assert AttemptRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="not monotonic"):
        _attempt(work_deadline_at=240.0)


def test_progress_chain_requires_monotonic_cumulative_work() -> None:
    first = _progress()
    second = _progress(
        sequence=2,
        previous_digest=digest_record(first.to_dict()),
        occurred_at=130.0,
        kind="progressed",
        phase="rl_step",
        training_entered=True,
        completed_steps=1,
    )
    assert second.follows(first)
    assert not _progress(
        sequence=2,
        previous_digest=digest_record(first.to_dict()),
        occurred_at=130.0,
        kind="progressed",
        phase="rl_step",
        training_entered=True,
        completed_steps=0,
    ).follows(second)
    assert progress_path(second).endswith(f"-{digest_record(second.to_dict())}.json")


def test_result_is_digest_addressed_and_success_requires_metrics() -> None:
    manifest = _result()
    assert ResultManifest.from_dict(manifest.to_dict()) == manifest
    assert result_path(manifest).endswith(f"/{digest_record(manifest.to_dict())}.json")
    with pytest.raises(ValueError, match="requires final metrics"):
        _result(metrics={})
    with pytest.raises(ValueError, match="failure class"):
        _result(outcome="failed", failure_class=None)


def test_receipt_and_bounded_values_reject_untrusted_shapes() -> None:
    digest = "a" * 64
    assert receipt("rl/flash-1/result/x.json", "revision", digest)["digest"] == digest
    with pytest.raises(ValueError, match="sha256"):
        receipt("x", "r", "bad")
    sanitized = bounded_json({"bad": math.inf, "rows": list(range(100))})
    assert sanitized["bad"] is None
    assert len(sanitized["rows"]) == 64
