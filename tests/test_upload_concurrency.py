"""Parallel file transfer must preserve the Hub's single atomic folder commit."""

import pytest
from huggingface_hub import HfApi

from flash.engine.worker.io.transfer import upload_api, upload_threads


@pytest.mark.parametrize("value", ["0", "17", "bad", "1.5"])
def test_concurrency_rejects_invalid_bounds(monkeypatch, value):
    monkeypatch.setenv("FLASH_HF_UPLOAD_THREADS", value)
    with pytest.raises(ValueError, match="1 to 16"):
        upload_threads()


def test_folder_commit_uses_bounded_parallelism(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_HF_UPLOAD_THREADS", "12")
    (tmp_path / "weights.bin").write_bytes(b"weights")
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    calls = []
    monkeypatch.setattr(HfApi, "create_commit", lambda self, *a, **kw: calls.append(kw))
    api = upload_api(token=None)
    api.upload_folder(folder_path=tmp_path, repo_id="test/private", repo_type="dataset")
    assert len(calls) == 1
    assert calls[0]["num_threads"] == 12
    assert calls[0]["repo_id"] == "test/private"
    assert len(calls[0]["operations"]) == 1


def test_explicit_commit_thread_count_is_preserved(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    calls = []
    monkeypatch.setattr(HfApi, "create_commit", lambda self, *a, **kw: calls.append(kw))
    upload_api(token=None).create_commit("test/private", operations=[], num_threads=2)
    assert calls[0]["num_threads"] == 2


def test_controller_validates_and_forwards_upload_threads(monkeypatch):
    import time

    from flash.providers.modal import api, jobs
    from tests._helpers.source_snapshot import valid_source_snapshot
    from tests.test_modal_training_lifecycle import spec

    monkeypatch.setenv("FLASH_HF_UPLOAD_THREADS", "4")
    monkeypatch.setattr("flash.runtime_capsule.build_capsule", lambda _: (b"capsule", {}))

    def capture(**kwargs):
        assert kwargs["payload"]["env"]["FLASH_HF_UPLOAD_THREADS"] == "4"
        raise RuntimeError("captured before rental")

    monkeypatch.setattr(api, "create", capture)
    with pytest.raises(RuntimeError, match="captured before rental"):
        jobs.submit_attempt(
            spec(), source_snapshot=valid_source_snapshot(), deadline_at=time.time() + 60
        )
    monkeypatch.setenv("FLASH_HF_UPLOAD_THREADS", "17")
    with pytest.raises(ValueError, match="1 to 16"):
        jobs.submit_attempt(
            spec(), source_snapshot=valid_source_snapshot(), deadline_at=time.time() + 60
        )
