"""Orchestrator end-to-end path with the Flash submission mocked (no RunPod calls)."""

from __future__ import annotations

import importlib
import json
import logging
import os
import tempfile

import pytest

import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.providers._lifecycle.net import worker as provider_worker
from tests._helpers.source_snapshot import valid_source_snapshot

SOURCE_SNAPSHOT = valid_source_snapshot()


def test_run_job_persists_flash_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.providers._lifecycle.net.worker as flash_train

        importlib.reload(flash_train)
        # Storage roots are fixed constants now; redirect via monkeypatch (auto-restored).
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner_state, "RESULTS_DIR", os.path.join(tmp, "results"))
        publication_events = []
        monkeypatch.setattr(
            provider_worker,
            "publish_source_snapshot",
            lambda _repo: publication_events.append("published") or SOURCE_SNAPSHOT,
        )
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        captured = {}

        def fake_submit(spec, log=None, **kwargs):
            from flash.snapshot.archive import TERMINAL_ATTESTATION_KEY, source_attestation

            captured["gpu"] = spec.gpu.type
            captured["seed"] = spec.seed
            captured["source_snapshot"] = kwargs["source_snapshot"]
            persisted = runner_status.get_status(spec.run_id)
            assert persisted.source_snapshot == SOURCE_SNAPSHOT
            claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
            assert claim is not None
            attempt = claim.attempt
            runner_attempts.release_launch_claim(spec.run_id, claim)
            return {
                "arm": "runpod",
                "phase": spec.phase,
                "seed": spec.seed,
                "wall_seconds": 3600.0,
                "trained_eval_acc": 0.7,
                "base_eval_acc": 0.3,
                "cost_usd": 0.0,
                "allocated_gpu": "RTX 4090",
                "notes": {},
                TERMINAL_ATTESTATION_KEY: source_attestation(
                    SOURCE_SNAPSHOT,
                    run_id=spec.run_id,
                    attempt=attempt,
                ),
            }

        # Stub the submit/poll path (the seam that used to be the in-process
        # offline shortcut) so the run completes without provisioning a GPU.
        # _run_training resolves it from the canonical supervise lifecycle owner.
        monkeypatch.setattr(runner_lifecycle, "_run_attempts_supervised", fake_submit)

        spec = JobSpec(
            run_id="flash-run",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=2),
            gpu=GpuSpec(type=""),
            seed=123,
        )
        status = runner_submit.submit_job(spec, dry_run=False, background=False)

        assert status.state == "done", status.error
        from flash.providers.runpod.client.pricing import hourly_rate

        # the full run charges the persisted accepted quote; measured cost lands in metrics.json.
        assert status.cost_usd == status.estimated_cost_usd, status.cost_usd
        assert publication_events == ["published"]
        assert captured["gpu"] == ""
        assert captured["seed"] == 123
        assert captured["source_snapshot"] == SOURCE_SNAPSHOT
        assert status.source_verified_attempt == 0

        # Metrics are namespaced by run id so same-phase runs cannot collide.
        metrics_path = os.path.join(tmp, "results", "runpod", "rl", status.run_id, "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            m = json.load(f)
        assert abs(m["cost_usd"] - hourly_rate("RTX 4090")) < 1e-6  # measured: 1h on the 4090
        assert m["trained_eval_acc"] == 0.7
        assert m["notes"]["gpu"] == "RTX 4090"


def test_source_publication_failure_is_generic_at_submission_and_api_boundary(monkeypatch, caplog):
    from flash.core.spec import JobSpec, TrainSpec
    from flash.server.routes.runs import _submit_failure_http_error

    spec = JobSpec(
        run_id="source-publication-failure",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="private-org/private-run-repo"),
    )
    prepared = runner_submit.PreparedJob(public_spec=spec, worker_spec=spec, estimated_cost_usd=1.0)
    secret = "hf_private_token_123456"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(
        provider_worker,
        "publish_source_snapshot",
        lambda _repo: (_ for _ in ()).throw(
            RuntimeError(
                f"403 from https://resolver.internal/private-org/private-run-repo "
                f"at source/{'a' * 64}/flash-source.zip?token={secret} revision={'b' * 40}"
            )
        ),
    )

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(runner_submit.SourceSnapshotPublicationError) as raised,
    ):
        runner_submit.submit_job(spec, prepared_job=prepared)

    public_error = _submit_failure_http_error(raised.value)
    assert public_error.status_code == 503
    assert public_error.detail == "managed source publication failed; retry the submission later"
    rendered = str(raised.value) + str(public_error.detail)
    for private_value in (
        "private-org/private-run-repo",
        "resolver.internal",
        "flash-source.zip",
        "b" * 40,
        secret,
        "403",
    ):
        assert private_value not in rendered
    assert secret not in caplog.text
    assert "managed source publication failed" in caplog.text


def _install_snapshot_hub(monkeypatch, api_type, download):
    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = api_type
    fake_hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)


def test_publish_source_snapshot_forces_private_and_captures_commit(monkeypatch, tmp_path):
    import hashlib
    import types

    from flash.providers._lifecycle.net import worker

    archive = b"deterministic-source-archive"
    archive_file = tmp_path / "archive.zip"
    archive_file.write_bytes(archive)
    calls = {"info": 0, "settings": [], "upload": []}

    class _Missing(Exception):
        response = types.SimpleNamespace(status_code=404)

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            calls["info"] += 1
            return types.SimpleNamespace(sha="1" * 40, private=False)

        def update_repo_settings(self, **kwargs):
            calls["settings"].append(kwargs)

        def upload_file(self, **kwargs):
            calls["upload"].append(kwargs)
            return types.SimpleNamespace(oid="c" * 40)

    def download(*, revision, **_kwargs):
        if revision == "1" * 40:
            raise _Missing("absent")
        assert revision == "c" * 40
        return str(archive_file)

    _install_snapshot_hub(monkeypatch, _Api, download)
    monkeypatch.setattr("flash.snapshot.archive.build_source_archive", lambda **_kwargs: archive)
    monkeypatch.setattr(
        "flash.snapshot.archive.read_verified_archive", lambda data, _desc: {"ok": data}
    )

    descriptor = worker.publish_source_snapshot("owner/run-artifacts")

    assert descriptor["sha256"] == hashlib.sha256(archive).hexdigest()
    assert descriptor["revision"] == "c" * 40
    assert calls["settings"][0]["private"] is True
    assert len(calls["upload"]) == 1
    assert calls["upload"][0]["path_in_repo"] == descriptor["archive_path"]


def test_publish_source_snapshot_creates_missing_repo(monkeypatch, tmp_path):
    import types

    from flash.providers._lifecycle.net import worker

    archive = b"archive"
    archive_file = tmp_path / "archive.zip"
    archive_file.write_bytes(archive)
    calls = {"info": 0, "create": [], "settings": []}

    class _RepositoryNotFoundError(Exception):
        pass

    _RepositoryNotFoundError.__name__ = "RepositoryNotFoundError"

    class _MissingEntry(Exception):
        response = types.SimpleNamespace(status_code=404)

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, **kwargs):
            calls["info"] += 1
            if calls["info"] == 1:
                raise _RepositoryNotFoundError("missing")
            return types.SimpleNamespace(sha="1" * 40, private=True)

        def create_repo(self, repo, **kwargs):
            calls["create"].append((repo, kwargs))

        def update_repo_settings(self, **kwargs):
            calls["settings"].append(kwargs)

        def upload_file(self, **_kwargs):
            return types.SimpleNamespace(oid="c" * 40)

    def download(*, revision, **_kwargs):
        if revision == "1" * 40:
            raise _MissingEntry("absent")
        return str(archive_file)

    _install_snapshot_hub(monkeypatch, _Api, download)
    monkeypatch.setattr("flash.snapshot.archive.build_source_archive", lambda **_kwargs: archive)
    monkeypatch.setattr("flash.snapshot.archive.read_verified_archive", lambda *_args: {})

    worker.publish_source_snapshot("owner/new-env-artifacts")

    assert calls["create"] == [
        ("owner/new-env-artifacts", {"repo_type": "dataset", "exist_ok": True, "private": True})
    ]
    assert calls["settings"][0]["private"] is True


def test_hf_call_honors_retry_after(monkeypatch):
    import flash.providers._lifecycle.net.worker as flash_train

    sleeps: list[float] = []
    logs: list[tuple[str, tuple]] = []

    class _Response:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "17"}

    class _RateLimited(Exception):
        response = _Response()

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited("slow down")
        return "ok"

    monkeypatch.setattr("flash.providers._lifecycle.net.worker.time.sleep", sleeps.append)
    monkeypatch.setattr(flash_train.logger, "warning", lambda msg, *args: logs.append((msg, args)))

    assert flash_train._hf_call(flaky, "upload") == "ok"
    assert sleeps == [17.0]
    assert logs


def test_hf_call_caps_http_date_retry_after(monkeypatch):
    import flash.providers._lifecycle.net.worker as flash_train

    sleeps: list[float] = []

    class _Response:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}

    class _RateLimited(Exception):
        response = _Response()

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimited("slow down")
        return "ok"

    monkeypatch.setattr("flash.providers._lifecycle.net.worker.time.sleep", sleeps.append)
    monkeypatch.setattr(flash_train.logger, "warning", lambda *_args: None)

    assert flash_train._hf_call(flaky, "upload") == "ok"
    assert sleeps == [60.0]


def test_publish_source_snapshot_reuses_verified_archive_at_exact_head(monkeypatch, tmp_path):
    import types

    from flash.providers._lifecycle.net import worker

    archive = b"same-archive"
    archive_file = tmp_path / "archive.zip"
    archive_file.write_bytes(archive)
    calls = {"downloads": [], "uploads": 0}

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, **_kwargs):
            return types.SimpleNamespace(sha="d" * 40, private=True)

        def update_repo_settings(self, **_kwargs):
            pass

        def upload_file(self, **_kwargs):
            calls["uploads"] += 1
            raise AssertionError("verified content-addressed archives must be reused")

    def download(**kwargs):
        calls["downloads"].append(kwargs)
        return str(archive_file)

    _install_snapshot_hub(monkeypatch, _Api, download)
    monkeypatch.setattr("flash.snapshot.archive.build_source_archive", lambda **_kwargs: archive)
    monkeypatch.setattr("flash.snapshot.archive.read_verified_archive", lambda *_args: {})

    descriptor = worker.publish_source_snapshot("owner/run-artifacts")

    assert descriptor["revision"] == "d" * 40
    assert calls["uploads"] == 0
    assert calls["downloads"][0]["revision"] == "d" * 40


def test_publish_source_snapshot_rereads_concurrent_winner(monkeypatch, tmp_path):
    import types

    from flash.providers._lifecycle.net import worker

    archive = b"same-archive"
    archive_file = tmp_path / "archive.zip"
    archive_file.write_bytes(archive)
    heads = iter(["1" * 40, "e" * 40])

    class _Missing(Exception):
        response = types.SimpleNamespace(status_code=404)

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, **_kwargs):
            return types.SimpleNamespace(sha=next(heads), private=True)

        def update_repo_settings(self, **_kwargs):
            pass

        def upload_file(self, **_kwargs):
            raise RuntimeError("concurrent commit conflict")

    def download(*, revision, **_kwargs):
        if revision == "1" * 40:
            raise _Missing("absent")
        assert revision == "e" * 40
        return str(archive_file)

    _install_snapshot_hub(monkeypatch, _Api, download)
    monkeypatch.setattr("flash.snapshot.archive.build_source_archive", lambda **_kwargs: archive)
    monkeypatch.setattr("flash.snapshot.archive.read_verified_archive", lambda *_args: {})

    descriptor = worker.publish_source_snapshot("owner/run-artifacts")
    assert descriptor["revision"] == "e" * 40


def test_run_job_background_swallows_exception(monkeypatch):
    """The daemon-thread entrypoint must NOT let _run_job's exception escape (it would surface as
    an alarming 'Exception in thread' traceback for every failed run); the synchronous _run_job
    keeps raising for its callers. Terminal state is already persisted before the raise, so the
    wrapper just logs and returns."""

    calls = {"n": 0}

    def boom(spec):
        calls["n"] += 1
        raise RuntimeError("seed 0 failed after retries: job_failed")

    # The wrapper dispatches through the package-level _run_job that tests patch.
    monkeypatch.setattr(runner_lifecycle, "_run_job", boom)
    spec = type("S", (), {"run_id": "bg-run"})()
    # Must not raise — the wrapper swallows it (state already persisted by _run_job_inner).
    runner_lifecycle._run_job_background(spec)
    assert calls["n"] == 1


def test_run_job_background_persists_failed_when_not_yet_terminal(monkeypatch, caplog):
    """If _run_job crashes BEFORE _run_job_inner persisted a terminal state (e.g. an import/resolve
    error), the daemon wrapper must still record a terminal `failed` (via terminal-sticky _update) so
    the run doesn't hang non-terminal forever."""
    import os
    import tempfile

    from flash.runner.lifecycle.state import RunStatus

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner_state, "RESULTS_DIR", os.path.join(tmp, "results"))
        os.makedirs(runner_state.RUNS_DIR, exist_ok=True)
        # a non-terminal (queued) run, as submit_job persists before dispatching the daemon thread
        runner_state._save_status(RunStatus(run_id="bg-fail", state="queued", spec={}))
        raw_message = "crashed before persisting terminal state"

        def boom(spec):
            raise RuntimeError(raw_message)

        monkeypatch.setattr(runner_lifecycle, "_run_job", boom)
        caplog.set_level(logging.WARNING, logger=runner_lifecycle.__name__)
        spec = type("S", (), {"run_id": "bg-fail"})()
        runner_lifecycle._run_job_background(spec)  # must not raise

        status = runner_status.get_status("bg-fail")
        assert status.state == "failed"
        safe_detail = "RuntimeError: background run failed"
        assert status.error == safe_detail
        assert raw_message not in (status.error or "")
        assert f"background run bg-fail ended in error: {safe_detail}" in caplog.messages
        assert raw_message not in caplog.text


def test_run_job_background_does_not_clobber_persisted_failure(monkeypatch):
    """If the run is ALREADY terminal (e.g. _run_job_inner persisted the real, detailed failure
    before the re-raise), the wrapper must NOT overwrite its error with the caught exception."""
    import os
    import tempfile

    from flash.runner.lifecycle.state import RunStatus

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner_state, "RESULTS_DIR", os.path.join(tmp, "results"))
        os.makedirs(runner_state.RUNS_DIR, exist_ok=True)
        # already terminal, carrying the REAL failure detail _run_job_inner persisted
        runner_state._save_status(
            RunStatus(run_id="bg-done", state="failed", spec={}, error="real seed failure detail")
        )

        def boom(spec):
            raise RuntimeError("generic wrapper-level error")

        monkeypatch.setattr(runner_lifecycle, "_run_job", boom)
        spec = type("S", (), {"run_id": "bg-done"})()
        runner_lifecycle._run_job_background(spec)  # must not raise

        status = runner_status.get_status("bg-done")
        assert status.state == "failed"
        assert status.error == "real seed failure detail"  # NOT clobbered by the wrapper
