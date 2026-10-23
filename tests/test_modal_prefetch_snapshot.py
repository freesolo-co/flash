"""Execute the exact CPU staging program against complete and damaged cache snapshots."""

import builtins
import json
import os
from types import SimpleNamespace

import pytest

from flash.providers.modal import prefetch

PIN = "a" * 40


@pytest.fixture
def staging(monkeypatch, tmp_path):
    calls = []
    snapshot = tmp_path / PIN
    snapshot.mkdir()
    monkeypatch.setenv("FLASH_BASE_MODEL", json.dumps(["org/model", PIN]))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    def run(repair=None, *, check_transport=False):
        def download(model, **kwargs):
            calls.append((model, kwargs))
            if kwargs.get("force_download") and repair:
                repair(snapshot)
            return str(snapshot)

        def import_module(name, *args, **kwargs):
            if name == "huggingface_hub":
                if check_transport:
                    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
                return SimpleNamespace(snapshot_download=download)
            return builtins.__import__(name, *args, **kwargs)

        namespace = {"__builtins__": dict(vars(builtins), __import__=import_module)}
        exec(prefetch._SCRIPT, namespace)

    return snapshot, calls, run


def test_config_only_snapshot_is_repaired_at_the_exact_revision(staging):
    snapshot, calls, run = staging
    (snapshot / "config.json").write_text("{}")
    run(lambda path: (path / "model.safetensors").write_bytes(b"weights"))
    assert len(calls) == 2
    assert calls[1][1] == calls[0][1] | {"force_download": True}
    assert calls[1][0] == calls[0][0] == "org/model"
    assert calls[1][1]["revision"] == PIN
    assert calls[1][1]["cache_dir"] == "/weight-cache/hf-cache/hub"
    assert calls[1][1]["token"] == "test-token"


@pytest.mark.parametrize("damage", ["missing_shard", "dangling_shard", "malformed_index"])
def test_partial_sharded_snapshot_is_repaired(staging, damage):
    snapshot, calls, run = staging
    index = snapshot / "model.safetensors.index.json"
    complete_index = json.dumps(
        {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
    )
    index.write_text("{" if damage == "malformed_index" else complete_index)
    (snapshot / "model-1.safetensors").write_bytes(b"weights")
    if damage == "dangling_shard":
        (snapshot / "model-2.safetensors").symlink_to(snapshot / "missing-blob")

    def repair(path):
        index.write_text(complete_index)
        second = path / "model-2.safetensors"
        second.unlink(missing_ok=True)
        second.write_bytes(b"weights")

    run(repair)
    assert len(calls) == 2
    assert calls[-1][1]["force_download"] is True


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
def test_complete_snapshot_needs_no_repair(staging, weight_name):
    snapshot, calls, run = staging
    (snapshot / weight_name).write_bytes(b"weights")
    run()
    assert len(calls) == 1
    assert not calls[0][1].get("force_download")


def test_unrepairable_snapshot_stops_after_one_forced_download(staging):
    snapshot, calls, run = staging
    (snapshot / "adapter_model.safetensors").write_bytes(b"not base weights")
    with pytest.raises(RuntimeError, match="weights"):
        run()
    assert len(calls) == 2
    assert calls[-1][1]["force_download"] is True


def test_xet_is_disabled_before_importing_huggingface(staging):
    snapshot, calls, run = staging
    (snapshot / "model.safetensors").write_bytes(b"weights")
    run(check_transport=True)
    assert len(calls) == 1
