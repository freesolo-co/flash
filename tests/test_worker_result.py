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


def test_exact_fence_terminal_lookup_validates_source_identity(monkeypatch) -> None:
    _set_identity(monkeypatch)
    existing = _manifest()

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **_kwargs):
            return [result_path(existing)]

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)
    monkeypatch.setattr(result_io, "_download_result", lambda _path, *, revision: existing)
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)

    assert result_io.read_existing_terminal_result() == existing

    monkeypatch.setattr(
        result_io,
        "_source_attestation",
        lambda: {**ATTESTATION, "revision": "d" * 40},
    )
    with pytest.raises(RuntimeError, match="source identity"):
        result_io.read_existing_terminal_result()


def test_exact_fence_terminal_lookup_fails_closed_on_malformed_result(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub

    _set_identity(monkeypatch)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json")

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **_kwargs):
            return ["rl/run-1/attempts/2-9/result/not-a-result.json"]

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(malformed))

    with pytest.raises(RuntimeError, match="manifest is malformed"):
        result_io.read_existing_terminal_result()


def test_exact_fence_terminal_lookup_transport_failure_is_retriable(monkeypatch) -> None:
    _set_identity(monkeypatch)

    class Api:
        def repo_info(self, **_kwargs):
            raise OSError("temporary transport failure")

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)

    with pytest.raises(result_io.hf_io.RetriableInfraError, match="terminal result lookup"):
        result_io.read_existing_terminal_result()


def test_exact_fence_terminal_lookup_fails_closed_on_conflicts(monkeypatch) -> None:
    _set_identity(monkeypatch)
    first = _manifest()
    second = _manifest(outcome="failed", failure_class="worker", metrics={})

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **_kwargs):
            return [result_path(first), result_path(second)]

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)

    with pytest.raises(RuntimeError, match="conflicting result manifests"):
        result_io.read_existing_terminal_result()


def test_worker_boot_adopts_existing_terminal_result_before_setup(monkeypatch) -> None:
    from flash.engine.worker.entry import worker

    existing = _manifest()
    calls = []
    monkeypatch.setattr(worker.hf_io, "_disable_xet_upload_staging", lambda: calls.append("xet"))
    monkeypatch.setattr(
        worker.result_io,
        "read_existing_terminal_result",
        lambda: calls.append("result") or existing,
    )
    monkeypatch.setattr(
        worker,
        "_preflight_gpu_occupancy_for_spec",
        lambda: pytest.fail("gpu occupancy must not run"),
    )
    monkeypatch.setattr(
        worker.kernel_warmup,
        "load_mega_cache",
        lambda: pytest.fail("kernel cache must not load"),
    )
    monkeypatch.setattr(
        worker.sft_entry,
        "run_sft",
        lambda: pytest.fail("handler must not run"),
    )
    monkeypatch.setattr(worker.state, "RUN_MODE", "sft")

    worker._run_worker_mode()

    assert calls == ["xet", "result"]


def test_worker_boot_terminal_lookup_failure_does_not_start_setup(monkeypatch) -> None:
    from flash.engine.worker.entry import worker

    monkeypatch.setattr(worker.hf_io, "_disable_xet_upload_staging", lambda: None)
    monkeypatch.setattr(
        worker.result_io,
        "read_existing_terminal_result",
        lambda: (_ for _ in ()).throw(RuntimeError("conflicting result manifests")),
    )
    monkeypatch.setattr(
        worker,
        "_preflight_gpu_occupancy_for_spec",
        lambda: pytest.fail("gpu occupancy must not run"),
    )
    monkeypatch.setattr(
        worker.kernel_warmup,
        "load_mega_cache",
        lambda: pytest.fail("kernel cache must not load"),
    )
    monkeypatch.setattr(
        worker.sft_entry,
        "run_sft",
        lambda: pytest.fail("handler must not run"),
    )
    monkeypatch.setattr(worker.state, "RUN_MODE", "sft")

    with pytest.raises(RuntimeError, match="conflicting"):
        worker._run_worker_mode()


def test_cancelled_result_uses_latest_current_fence_progress(monkeypatch) -> None:
    _set_identity(monkeypatch)
    captured = []
    monkeypatch.setattr(
        result_io,
        "_latest_local_progress",
        lambda: {
            "training_entered": True,
            "completed_steps": 4,
            "metrics": {"loss": 0.25},
            "checkpoint": {"step": 4},
        },
    )
    monkeypatch.setattr(
        result_io,
        "publish_result",
        lambda **kwargs: captured.append(kwargs) or _manifest(),
    )

    result_io.publish_cancelled_result(started_at=100.0)

    assert captured == [
        {
            "outcome": "cancelled",
            "failure_class": None,
            "started_at": 100.0,
            "training_entered": True,
            "completed_steps": 4,
            "metrics": {"loss": 0.25},
            "checkpoint": {"step": 4},
            "artifacts": {"console": "console_rl.txt"},
            "diagnostics": {"error": "worker attempt cancelled"},
        }
    ]


def test_worker_surfaces_required_result_publication_failure(monkeypatch) -> None:
    from flash.engine.worker.entry import worker

    class ResultPublicationError(RuntimeError):
        pass

    original = ValueError("training failed")
    publication = ResultPublicationError("result transport failed")
    monkeypatch.setattr(worker, "_run_worker_mode", lambda: (_ for _ in ()).throw(original))
    monkeypatch.setattr(worker.hf_io, "hf_upload_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker.backend_common, "collect_ray_failure_logs", lambda **_kwargs: "")
    monkeypatch.setattr(
        worker.result_io,
        "publish_result",
        lambda **_kwargs: (_ for _ in ()).throw(publication),
    )
    monkeypatch.setattr(worker.progress_io, "publish_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(worker.worker_perf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)

    with pytest.raises(ResultPublicationError) as exc_info:
        worker.main()

    assert exc_info.value is publication
    assert exc_info.value.__cause__ is original


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
