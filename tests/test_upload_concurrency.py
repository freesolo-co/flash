"""Parallel file transfer must preserve the Hub's single atomic folder commit."""

import pytest
from huggingface_hub import HfApi

from flash.engine.worker.io.transfer import upload_api


@pytest.mark.parametrize("value", [None, "12", "0", "17", "bad", "1.5"])
def test_folder_commit_uses_fixed_parallelism(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv("FLASH_HF_UPLOAD_THREADS", raising=False)
    else:
        monkeypatch.setenv("FLASH_HF_UPLOAD_THREADS", value)
    (tmp_path / "weights.bin").write_bytes(b"weights")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    calls = []
    monkeypatch.setattr(HfApi, "create_commit", lambda self, *a, **kw: calls.append(kw))
    api = upload_api(token=None)
    api.upload_folder(folder_path=tmp_path, repo_id="test/private", repo_type="dataset")
    assert len(calls) == 1
    assert calls[0]["num_threads"] == 8
    assert calls[0]["repo_id"] == "test/private"
    assert len(calls[0]["operations"]) == 1


def test_explicit_commit_thread_count_is_preserved(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    calls = []
    monkeypatch.setattr(HfApi, "create_commit", lambda self, *a, **kw: calls.append(kw))
    upload_api(token=None).create_commit("test/private", operations=[], num_threads=2)
    assert calls[0]["num_threads"] == 2


@pytest.mark.parametrize("value", ["17", "bad"])
def test_runtime_secret_cannot_break_worker_uploads(monkeypatch, value):
    import time
    from dataclasses import replace

    from flash.providers.modal import api, jobs
    from tests._helpers.source_snapshot import valid_source_snapshot
    from tests.test_modal_training_lifecycle import spec

    monkeypatch.delenv("FLASH_HF_UPLOAD_THREADS", raising=False)
    monkeypatch.setattr("flash.runtime_capsule.build_capsule", lambda _: (b"capsule", {}))
    job = spec()
    job = replace(job, environment=replace(job.environment, secrets=("FLASH_HF_UPLOAD_THREADS",)))
    calls = []
    monkeypatch.setattr(HfApi, "create_commit", lambda self, *a, **kw: calls.append(kw))

    def capture(**kwargs):
        # Reproduce the worker receiving a value from the job's runtime secrets.
        monkeypatch.setenv(
            "FLASH_HF_UPLOAD_THREADS", kwargs["payload"]["env"]["FLASH_HF_UPLOAD_THREADS"]
        )
        upload_api(token=None).create_commit("test/private", operations=[])
        assert calls[0]["num_threads"] == 8
        raise RuntimeError("captured before rental")

    monkeypatch.setattr(api, "create", capture)
    with pytest.raises(RuntimeError, match="captured before rental"):
        jobs.submit_attempt(
            job,
            runtime_secrets={"FLASH_HF_UPLOAD_THREADS": value},
            source_snapshot=valid_source_snapshot(),
            deadline_at=time.time() + 60,
        )
