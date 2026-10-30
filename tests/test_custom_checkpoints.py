"""Custom state uses the shared publisher, transport, and checkpoint listing."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.custom.checkpoints import RunCheckpoints
from flash.engine.worker.io import hf


@pytest.fixture
def store(tmp_path, monkeypatch, worker_context):
    durable = tmp_path / "durable"
    durable.mkdir()

    def upload(**kwargs):
        assert kwargs["ignore_patterns"] == []
        shutil.copytree(kwargs["folder_path"], durable / kwargs["path_in_repo"])
        return SimpleNamespace(oid="commit")

    def fetch(*, local_dir, **kwargs):
        prefix = kwargs["allow_patterns"][0].removesuffix("/**")
        shutil.copytree(durable / prefix, Path(local_dir) / prefix)
        return str(local_dir)

    worker_context.config = replace(
        worker_context.config, run_id="run", phase="custom", hf_repo="private/repo"
    )
    monkeypatch.setattr(hf, "hf_api", lambda: SimpleNamespace(upload_folder=upload))
    monkeypatch.setattr(hf, "_write_deployable_provenance", Mock())
    monkeypatch.setattr(hf, "_sleep_with_hf_deadline", lambda *a, **k: False)
    monkeypatch.setattr("flash.engine.worker.io.heartbeat.heartbeat", Mock())
    monkeypatch.setattr("huggingface_hub.snapshot_download", fetch)
    return RunCheckpoints(worker_context)


def publish(store, root, step, name=None):
    directory = root / (name or f"step-{step}")
    directory.mkdir(parents=True)
    (directory / "state.pt").write_bytes(f"optimizer-{step}".encode())
    (directory / "algorithm.json").write_text('{"cursor": 4}')
    (directory / "adapter").mkdir()
    (directory / "adapter" / "adapter_config.json").write_text("{}")
    (directory / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    store.publish(directory, {"step": step})
    return directory


def test_many_saves_keep_latest_locally_and_all_steps_in_standard_listing(store, tmp_path):
    from flash.runner.results.checkpoints import _checkpoints_from_files
    from flash.schema import spec_from_dict

    root = tmp_path / "checkpoints"
    for step in range(1, 21):
        publish(store, root, step)
        assert [path.name for path in root.iterdir()] == [f"step-{step}"]
    durable = tmp_path / "durable"
    files = [str(p.relative_to(durable)) for p in durable.rglob("*") if p.is_file()]
    spec = spec_from_dict(
        {"model": "Qwen/Qwen3.5-9B", "algorithm": "custom", "custom": {"file": "algorithm.py"}},
        run_id="run",
    )
    rows = _checkpoints_from_files(spec, files)
    assert [row["step"] for row in rows] == list(range(1, 21))
    assert (durable / "custom/run/checkpoints/step-1/state.pt").read_bytes() == b"optimizer-1"
    assert len(store.published) == 20


def test_failed_publication_preserves_previous_local_checkpoint(store, tmp_path, monkeypatch):
    first = publish(store, tmp_path / "checkpoints", 1)
    monkeypatch.setattr(
        hf,
        "hf_api",
        lambda: SimpleNamespace(upload_folder=Mock(side_effect=RuntimeError("storage offline"))),
    )
    with pytest.raises(hf.RetriableInfraError, match="upload failed"):
        publish(store, tmp_path / "checkpoints", 2)
    assert first.is_dir()
    assert set(store.published) == {1}


def test_large_native_state_uses_shared_chunk_transport_and_restores(store, tmp_path, monkeypatch):
    from flash.engine.support import checkpoint_transport

    monkeypatch.setattr(checkpoint_transport, "PART_BYTES", 8)
    publish(store, tmp_path / "checkpoints", 1)
    durable = tmp_path / "durable/custom/run/checkpoints/step-1"
    assert not (durable / "state.pt").exists()
    assert (durable / checkpoint_transport.MANIFEST).is_file()

    def restore(source):
        assert (source / "state.pt").read_bytes() == b"optimizer-1"
        assert json.loads((source / "algorithm.json").read_text()) == {"cursor": 4}
        return {"checkpoint": json.loads((source / "metadata.json").read_text())}

    result = store.restore(
        SimpleNamespace(restore_checkpoint=restore),
        {
            "repo": "private/repo",
            "revision": "commit",
            "path": "custom/run/checkpoints/step-1",
        },
    )
    assert result["checkpoint"]["step"] == 1


def test_evicted_checkpoint_name_remains_reserved(store, tmp_path):
    from flash.custom.validation import validate_operation

    root = tmp_path / "checkpoints"
    publish(store, root, 1, name="manual")
    publish(store, root, 2)
    backend = SimpleNamespace(checkpoint_dir=root, checkpoints=store, pending_backward=0)
    with pytest.raises(ValueError, match="checkpoint already exists"):
        validate_operation(backend, "save_checkpoint", {"name": "manual"})


@pytest.mark.parametrize("name", ["step-1", "step-10", "step-999"])
def test_explicit_managed_saves_cannot_claim_automatic_names(store, tmp_path, name):
    from flash.custom.validation import validate_operation

    backend = SimpleNamespace(checkpoint_dir=tmp_path, checkpoints=store, pending_backward=0)
    with pytest.raises(ValueError, match="reserved for automatic"):
        validate_operation(backend, "save_checkpoint", {"name": name})


@pytest.mark.parametrize("chunked", [False, True])
def test_resume_validates_pinned_shared_artifacts_before_gpu_allocation(
    tmp_path, monkeypatch, chunked
):
    from flash.custom.managed import resolve_resume
    from flash.engine.support import checkpoint_transport
    from flash.runner.lifecycle import preparation, status
    from flash.schema import spec_from_dict

    source = spec_from_dict(
        {"model": "Qwen/Qwen3.5-9B", "algorithm": "custom", "custom": {"file": "algorithm.py"}},
        run_id="source",
    )
    source = replace(
        source, model_revision="a" * 40, train=replace(source.train, hf_repo="private/repo")
    )
    target = replace(
        source, run_id="target", custom={**source.custom, "resume_from": "source/step-1"}
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "adapter").mkdir()
    (root / "adapter/adapter_config.json").write_text("{}")
    (root / "adapter/adapter_model.safetensors").write_bytes(b"weights")
    (root / "state.pt").write_bytes(b"x" * 2048)
    (root / "algorithm.json").write_text("{}")
    (root / "metadata.json").write_text(
        json.dumps(
            {"format": "flash-custom-v1", "step": 1, "model_revision": source.model_revision}
        )
    )
    monkeypatch.setattr(checkpoint_transport, "PART_BYTES", 1024 if chunked else 4096)
    monkeypatch.setattr(status, "get_status", lambda run: object())
    monkeypatch.setattr(status, "effective_spec_from_status", lambda record: source)
    monkeypatch.setattr(preparation, "_warmstart_source_is_authorized", lambda *a, **k: True)
    prefix = "custom/source/checkpoints/step-1"
    with checkpoint_transport.upload_directory(str(root)) as staged:
        staged = Path(staged)
        files = [f"{prefix}/{p.relative_to(staged)}" for p in staged.rglob("*") if p.is_file()]
        api = SimpleNamespace(
            repo_info=lambda *a, **k: SimpleNamespace(sha="b" * 40),
            list_repo_files=lambda *a, **k: files,
        )
        monkeypatch.setattr("huggingface_hub.HfApi", lambda **kw: api)

        def download(repo, path, **kwargs):
            assert kwargs["revision"] == "b" * 40
            return str(staged / path.removeprefix(prefix + "/"))

        monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
        resolved = resolve_resume(target, owner_org_id="org", owner_key_id=1)
        assert resolved.custom["resolved_resume"] == {
            "repo": "private/repo",
            "revision": "b" * 40,
            "path": prefix,
        }
        files.remove(f"{prefix}/adapter/adapter_model.safetensors")
        with pytest.raises(ValueError, match="incomplete"):
            resolve_resume(target, owner_org_id="org", owner_key_id=1)
