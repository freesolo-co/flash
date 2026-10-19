"""Offline tokenizer snapshots retain their own completeness contract."""

import json

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError

from flash.serve.runtime.hub import engine_tokenizer_path
from flash.serve.runtime.types import EngineConfig


def test_offline_tokenizer_snapshot_does_not_require_unused_model_weights(tmp_path, monkeypatch):
    revision = "b" * 40
    storage = tmp_path / "models--tokenizer--model"
    snapshot = storage / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}")
    trees = storage / "trees"
    trees.mkdir()
    (trees / f"{revision}.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {
                    name: {"size": 2, "blob_id": "a" * 40}
                    for name in ("tokenizer.json", "model-00001-of-00018.safetensors")
                },
            }
        )
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    config = EngineConfig(
        model="served/model",
        tokenizer_model="tokenizer/model",
        model_revision="a" * 40,
        tokenizer_revision=revision,
    )
    with pytest.raises(IncompleteSnapshotError):
        snapshot_download(
            "tokenizer/model", revision=revision, cache_dir=tmp_path, local_files_only=True
        )
    assert engine_tokenizer_path(config) == str(snapshot)
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(IncompleteSnapshotError):
        engine_tokenizer_path(config)


@pytest.mark.parametrize("offline", ["0", "1"])
def test_shared_model_tokenizer_keeps_the_full_snapshot_contract(monkeypatch, offline):
    monkeypatch.setenv("HF_HUB_OFFLINE", offline)
    config = EngineConfig(
        model="served/model", model_revision="a" * 40, tokenizer_revision="a" * 40
    )
    assert engine_tokenizer_path(config) == "served/model"


def test_online_tokenizer_remains_a_repository_id(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert (
        engine_tokenizer_path(EngineConfig(model="served/model", tokenizer_model="tokenizer/model"))
        == "tokenizer/model"
    )
