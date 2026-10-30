"""Durable checkpoint retention and temporary exports without GPU dependencies."""

import asyncio
import io
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.custom.checkpoints import RunCheckpoints
from flash.custom.session import TrainingSession


@pytest.fixture
def store(tmp_path, monkeypatch):
    import huggingface_hub

    from flash.engine.worker.io import hf

    durable = tmp_path / "durable"
    durable.mkdir()

    def upload(source, path, **kwargs):
        shutil.copytree(source, durable / path)
        return True

    def fetch(*, local_dir, **kwargs):
        prefix = kwargs["allow_patterns"][0].removesuffix("/**")
        step = prefix.rsplit("/", 1)[-1]
        shutil.copytree(durable / "checkpoints" / step, Path(local_dir) / prefix)
        return str(local_dir)

    monkeypatch.setattr(hf, "hf_upload_folder", upload)
    monkeypatch.setattr(hf, "_write_deployable_provenance", Mock())
    monkeypatch.setattr("flash.engine.worker.io.heartbeat.heartbeat", Mock())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    context = SimpleNamespace(
        config=SimpleNamespace(run_id="run", phase="custom", hf_repo="private/repo")
    )
    return RunCheckpoints(context)


def publish(store, root, step, name=None):
    directory = root / (name or f"step-{step}")
    directory.mkdir(parents=True)
    (directory / "state.pt").write_bytes(f"optimizer-{step}".encode())
    (directory / "adapter").mkdir()
    (directory / "adapter" / "weights").write_bytes(b"adapter")
    store.publish(directory, {"step": step})
    return directory


def test_many_saves_keep_only_latest_local_snapshot_and_rehydrate_old_exports(store, tmp_path):
    from fastapi.testclient import TestClient

    from flash.custom.server import create_app

    root = tmp_path / "checkpoints"
    for step in range(1, 21):
        publish(store, root, step)
        assert [path.name for path in root.iterdir()] == [f"step-{step}"]
    backend = SimpleNamespace(checkpoints=store, checkpoint_dir=root)
    session = TrainingSession(backend)
    response = TestClient(create_app(session, "s" * 40)).get(
        "/v1/custom/checkpoints/step-1", headers={"Authorization": "Bearer " + "s" * 40}
    )
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("state.pt") == b"optimizer-1"
        assert json.loads(archive.read("metadata.json"))["step"] == 1
    assert session.active_exports == 0
    assert [path.name for path in root.iterdir()] == ["step-20"]
    assert len(store.published) == 20


def test_failed_publication_preserves_previous_local_checkpoint(store, tmp_path, monkeypatch):
    from flash.engine.worker.io import hf

    first = publish(store, tmp_path / "checkpoints", 1)
    monkeypatch.setattr(hf, "hf_upload_folder", Mock(side_effect=RuntimeError("storage offline")))
    with pytest.raises(RuntimeError, match="storage offline"):
        publish(store, tmp_path / "checkpoints", 2)
    assert first.is_dir()
    assert set(store.published) == {1}


def test_export_is_bounded_and_cleans_up_after_disconnect(store, tmp_path, monkeypatch):
    import tempfile

    from fastapi import HTTPException

    from flash.custom.exports import checkpoint_response

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root = tmp_path / "checkpoints"
    publish(store, root, 1)
    session = TrainingSession(SimpleNamespace(checkpoints=store, checkpoint_dir=root))

    async def run():
        response = checkpoint_response(session, "step-1")
        assert session.active_exports == 1
        with pytest.raises(HTTPException) as caught:
            checkpoint_response(session, "step-1")
        assert caught.value.status_code == 409
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        assert session.active_exports == 0
        assert not list(tmp_path.glob("flash-custom-export-*"))

    asyncio.run(run())


def test_export_cleanup_when_connection_fails_before_first_chunk(store, tmp_path, monkeypatch):
    import tempfile

    from starlette.requests import ClientDisconnect

    from flash.custom.exports import checkpoint_response

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root = tmp_path / "checkpoints"
    publish(store, root, 1)
    session = TrainingSession(SimpleNamespace(checkpoints=store, checkpoint_dir=root))

    async def disconnected(*args):
        raise OSError("connection closed")

    async def run():
        response = checkpoint_response(session, "step-1")
        with pytest.raises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, disconnected, disconnected
            )
        assert session.active_exports == 0
        assert not list(tmp_path.glob("flash-custom-export-*"))

    asyncio.run(run())


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
