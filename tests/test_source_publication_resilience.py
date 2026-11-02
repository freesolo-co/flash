"""Managed source publication: retries, private-repo write skipping, deferral and safe errors."""

from __future__ import annotations

import sys
import types

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.core.spec import JobSpec, TrainSpec
from flash.providers._lifecycle.net import worker
from tests._helpers.source_snapshot import valid_source_snapshot

REPO = "private-org/private-run-repo"
HEAD = "1" * 40
COMMIT = "c" * 40


class _HTTPError(Exception):
    def __init__(self, status, message="upstream error"):
        super().__init__(message)
        self.response = types.SimpleNamespace(status_code=status, headers={})


def _install_hub(monkeypatch, api, download):
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = lambda token=None: api
    fake.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setattr("flash.snapshot.archive.build_source_archive", lambda **_k: b"archive")
    monkeypatch.setattr("flash.snapshot.archive.read_verified_archive", lambda *_a: {})
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)


class _Api:
    """Fake HfApi: private repo with no snapshot at HEAD; upload behaviour is scripted."""

    def __init__(self, upload_errors=(), private=True):
        self.upload_errors = list(upload_errors)
        self.private = private
        self.settings = []
        self.uploads = 0

    def repo_info(self, **_kwargs):
        return types.SimpleNamespace(sha=HEAD, private=self.private)

    def update_repo_settings(self, **kwargs):
        self.settings.append(kwargs)

    def upload_file(self, **_kwargs):
        self.uploads += 1
        if self.upload_errors:
            raise self.upload_errors.pop(0)
        return types.SimpleNamespace(oid=COMMIT)


def _download_present_at_commit(tmp_path):
    archive_file = tmp_path / "archive.zip"
    archive_file.write_bytes(b"archive")

    def download(*, revision, **_kwargs):
        if revision != COMMIT:
            raise _HTTPError(404, "absent")
        return str(archive_file)

    return download


def test_already_private_repo_skips_settings_write(monkeypatch, tmp_path):
    api = _Api(private=True)
    _install_hub(monkeypatch, api, _download_present_at_commit(tmp_path))

    descriptor = worker.publish_source_snapshot(REPO)

    assert descriptor["revision"] == COMMIT
    assert api.settings == []


def test_public_repo_is_still_forced_private(monkeypatch, tmp_path):
    api = _Api(private=False)
    _install_hub(monkeypatch, api, _download_present_at_commit(tmp_path))

    worker.publish_source_snapshot(REPO)

    assert [s["private"] for s in api.settings] == [True]


@pytest.mark.parametrize(
    "transient",
    [ConnectionResetError("reset by peer"), _HTTPError(412, "commit race"), _HTTPError(409)],
)
def test_transport_errors_and_commit_races_are_retried(monkeypatch, tmp_path, transient):
    api = _Api(upload_errors=[transient])
    _install_hub(monkeypatch, api, _download_present_at_commit(tmp_path))

    descriptor = worker.publish_source_snapshot(REPO)

    assert descriptor["revision"] == COMMIT
    assert api.uploads == 2


def test_persistent_rate_limit_raises_safe_step_error(monkeypatch, tmp_path):
    secret = "hf_private_token_123456"
    api = _Api(upload_errors=[_HTTPError(429, f"{REPO} token={secret}")] * 50)
    _install_hub(monkeypatch, api, _download_present_at_commit(tmp_path))

    with pytest.raises(worker.SourcePublicationStepError) as raised:
        worker.publish_source_snapshot(REPO)

    assert str(raised.value) == "upload source snapshot failed (rate limited)"
    assert raised.value.status_code == 429
    for private_value in (REPO, secret, "429"):
        assert private_value not in str(raised.value)


def test_non_retryable_rejection_fails_fast(monkeypatch, tmp_path):
    api = _Api(upload_errors=[_HTTPError(403, "forbidden")])
    _install_hub(monkeypatch, api, _download_present_at_commit(tmp_path))

    with pytest.raises(worker.SourcePublicationStepError) as raised:
        worker.publish_source_snapshot(REPO)

    assert str(raised.value) == "upload source snapshot failed (rejected)"
    assert api.uploads == 1


def _queued_run(monkeypatch, tmp_path, publish):
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="deferred-source",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo=REPO),
    )
    prepared = runner_submit.PreparedJob(public_spec=spec, worker_spec=spec, estimated_cost_usd=1.0)
    monkeypatch.setattr(worker, "publish_source_snapshot", publish)
    status = runner_submit.submit_job(spec, prepared_job=prepared, queue_only=True)
    return spec, status


def test_deferred_snapshot_is_published_once_before_launch(monkeypatch, tmp_path):
    snapshot = valid_source_snapshot()
    calls = []
    outcomes = [RuntimeError("hub down"), RuntimeError("hub down"), snapshot]

    def publish(_repo):
        calls.append(_repo)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    spec, status = _queued_run(monkeypatch, tmp_path, publish)
    assert status.state == "queued"
    assert runner_status.get_status(spec.run_id).source_snapshot is None

    sleeps = []
    descriptor = runner_submit.ensure_source_snapshot(spec, sleep=sleeps.append)

    assert descriptor == runner_status.get_status(spec.run_id).source_snapshot
    assert len(calls) == 3  # one failed at submit, one failed retry, one success
    assert sleeps == [10.0]
    # already persisted: no further publication
    assert runner_submit.ensure_source_snapshot(spec, sleep=sleeps.append) == descriptor
    assert len(calls) == 3


def test_deferred_publication_gives_up_with_safe_terminal_error(monkeypatch, tmp_path):
    step_error = worker.SourcePublicationStepError("upload source snapshot", _HTTPError(503))

    def publish(_repo):
        raise step_error

    spec, _ = _queued_run(monkeypatch, tmp_path, publish)
    monkeypatch.setattr(runner_submit, "SOURCE_PUBLICATION_LAUNCH_DEADLINE_S", 0.0)

    with pytest.raises(runner_submit.SourceSnapshotPublicationError) as raised:
        runner_submit.ensure_source_snapshot(spec, sleep=lambda _s: None)

    expected = (
        "managed source publication failed: upload source snapshot failed (service unavailable)"
    )
    assert str(raised.value) == expected
    assert runner_lifecycle._terminal_failure_detail(raised.value) == expected


def test_record_source_snapshot_keeps_first_descriptor(monkeypatch, tmp_path):
    first = valid_source_snapshot()

    def publish(_repo):
        raise RuntimeError("hub down")

    spec, _ = _queued_run(monkeypatch, tmp_path, publish)
    assert runner_status.record_source_snapshot(spec.run_id, first) == first
    other = {**first, "revision": "d" * 40}
    assert runner_status.record_source_snapshot(spec.run_id, other) == first
    assert runner_status.get_status(spec.run_id).source_snapshot == first


def test_client_submit_uses_long_timeout(monkeypatch):
    from flash.client import http

    seen = {}

    def fake_request(self, method, path, **kwargs):
        seen.update(method=method, path=path, timeout=kwargs.get("timeout"))
        return {"run_id": "flash-1"}

    monkeypatch.setattr(http.ApiClient, "_request", fake_request)
    client = http.ApiClient("https://flash.example", "key")
    client.create_run({"project": "3b935f75-b1b1-4f1b-9508-25255e4b2938", "model": "m"})

    assert seen == {
        "method": "POST",
        "path": "/v1/runs",
        "timeout": http.SUBMIT_CLIENT_TIMEOUT_SECONDS,
    }
    assert http.SUBMIT_CLIENT_TIMEOUT_SECONDS > worker.SOURCE_PUBLICATION_DEADLINE_S
