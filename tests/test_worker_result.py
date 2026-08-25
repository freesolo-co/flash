from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.worker.io import result as result_io
from flash.runner.lifecycle.protocol import ResultManifest, result_path

ATTESTATION = {
    "kind": "flash-source-attestation",
    "format_version": 1,
    "sha256": "a" * 64,
    "revision": "b" * 40,
    "run_id": "run-1",
    "attempt": 2,
    "fence": 9,
}


def _manifest(**updates) -> ResultManifest:
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


def _set_identity(monkeypatch) -> None:
    monkeypatch.setattr(result_io.state, "HF_REPO", "org/repo")
    monkeypatch.setattr(result_io.state, "RUN_ID", "run-1")
    monkeypatch.setattr(result_io.state, "PHASE", "rl")
    monkeypatch.setattr(result_io.state, "ATTEMPT", 2)
    monkeypatch.setattr(result_io.state, "FENCE", 9)
    monkeypatch.setattr(result_io.hf_io, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(result_io.hf_io, "_sleep_with_hf_deadline", lambda _delay: True)


def test_exactly_once_publish_adopts_matching_concurrent_result(monkeypatch, tmp_path) -> None:
    _set_identity(monkeypatch)
    proposed = _manifest(finished_at=121.0)
    existing = _manifest(finished_at=120.0)
    result_prefix = result_path(existing).rsplit("/", 1)[0] + "/"

    class Api:
        def __init__(self) -> None:
            self.revision = "c" * 40
            self.created = 0

        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha=self.revision)

        def list_repo_files(self, **_kwargs):
            return [] if self.created == 0 else [result_path(existing)]

        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == "c" * 40
            assert kwargs["operations"][0].path_in_repo.startswith(result_prefix)
            self.created += 1
            self.revision = "d" * 40
            raise RuntimeError("parent commit changed")

    api = Api()
    monkeypatch.setattr(result_io.hf_io, "hf_api", lambda: api)
    monkeypatch.setattr(result_io, "_download_result", lambda _path, *, revision: existing)
    local = tmp_path / "result.json"
    local.write_text("{}")

    observed = result_io._publish_exactly_once(proposed, str(local))

    assert observed == existing
    assert api.created == 1


def test_exactly_once_publish_rejects_conflicting_existing_result(monkeypatch, tmp_path) -> None:
    _set_identity(monkeypatch)
    proposed = _manifest()
    existing = _manifest(outcome="failed", failure_class="worker", metrics={})

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **_kwargs):
            return [result_path(existing)]

        def create_commit(self, **_kwargs):
            raise AssertionError("conflicting result must be rejected before upload")

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)
    monkeypatch.setattr(result_io, "_download_result", lambda _path, *, revision: existing)
    local = tmp_path / "result.json"
    local.write_text("{}")

    with pytest.raises(RuntimeError, match="conflicting terminal result"):
        result_io._publish_exactly_once(proposed, str(local))
