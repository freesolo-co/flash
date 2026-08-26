from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.providers.artifacts.attempts import AttemptArtifacts
from flash.providers.core.base import PollResult
from flash.runner.supervise import recovery

ATTEMPT = {
    "attempt_id": 2,
    "fence": 9,
    "state": "active",
    "reserved_at": 1.0,
    "grant_deadline_at": 2.0,
    "work_deadline_at": 3.0,
    "result_deadline_at": 5.0,
    "run_deadline_at": 4.0,
    "provider": "runpod",
    "provider_contract": None,
    "resource": None,
    "allocation": None,
    "progress_receipt": None,
    "result_receipt": None,
    "cleanup": {},
    "schema_version": 1,
}


def _install_seams(monkeypatch, *, handle: dict, result: dict | None) -> None:
    import flash.providers.artifacts.attempts as attempt_artifacts
    import flash.runner.lifecycle.status as status_ops

    monkeypatch.setattr(
        recovery,
        "_canonical_provider_handle",
        lambda _handle: SimpleNamespace(to_dict=lambda: handle),
    )
    monkeypatch.setattr(
        status_ops,
        "get_status",
        lambda _run_id: SimpleNamespace(attempt=ATTEMPT),
    )
    monkeypatch.setattr(
        status_ops,
        "effective_spec_from_status",
        lambda _status: SimpleNamespace(
            train=SimpleNamespace(hf_repo="org/repo"),
            phase="rl",
        ),
    )
    monkeypatch.setattr(
        status_ops,
        "source_snapshot_from_status",
        lambda _status, required: {"source": "snapshot"},
    )
    monkeypatch.setattr(
        attempt_artifacts,
        "read_attempt_artifacts",
        lambda *_args, **_kwargs: AttemptArtifacts("r", 10.0, None, result),
    )
    monkeypatch.setattr(attempt_artifacts, "persist_attempt_artifacts", lambda *_args: None)
    monkeypatch.setattr(
        attempt_artifacts,
        "poll_result_from_manifest",
        lambda projection: PollResult(
            projection["ok"],
            metrics=projection.get("metrics"),
            failure=projection.get("failure"),
            detail=projection.get("detail"),
        ),
    )


def test_recovery_reads_only_current_fenced_result(monkeypatch) -> None:
    projection = {"ok": True, "metrics": {"step": 3}}
    handle = {"attempt": 2, "fence": 9}
    _install_seams(monkeypatch, handle=handle, result=projection)

    result = recovery._attempt_result("run-1", handle)

    assert result.ok
    assert result.metrics == {"step": 3}


def test_recovery_preserves_current_fenced_failure(monkeypatch) -> None:
    projection = {"ok": False, "failure": "oom", "detail": "cuda out of memory"}
    handle = {"attempt": 2, "fence": 9}
    _install_seams(monkeypatch, handle=handle, result=projection)

    result = recovery._attempt_result("run-1", handle)

    assert not result.ok
    assert result.failure == "oom"
    assert result.detail == "cuda out of memory"


def test_recovery_rejects_stale_fence_before_artifact_read(monkeypatch) -> None:
    handle = {"attempt": 2, "fence": 8}
    _install_seams(monkeypatch, handle=handle, result=None)

    with pytest.raises(RuntimeError, match="current fenced attempt"):
        recovery._attempt_result("run-1", handle)
