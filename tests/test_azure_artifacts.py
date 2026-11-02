"""Atomic publication, immutable reads, and integrity across the Azure cutover."""

import hashlib
import io
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceModifiedError, ResourceNotFoundError

from flash.artifacts import api
from flash.artifacts.manifest import ArtifactNotFoundError, repo_prefix
from flash.artifacts.storage import BlobArtifacts

REPO = "Freesolo-Co/flashrun-test"


class MemoryContainer:
    def __init__(self):
        self.blobs = {}
        self.sequence = 0
        self.before_head = None
        self.fail_object = False

    def get_container_properties(self):
        return SimpleNamespace(public_access=None)

    def get_blob_client(self, name):
        container = self

        class Blob:
            def upload_blob(self, data, *, overwrite=False, etag=None, metadata=None, **kwargs):
                if "/objects/" in name and container.fail_object:
                    raise OSError("interrupted upload")
                if name.endswith("/HEAD") and container.before_head:
                    callback, container.before_head = container.before_head, None
                    callback()
                current = container.blobs.get(name)
                if current and not overwrite:
                    raise ResourceExistsError("exists")
                if etag is not None and (current is None or current[1] != etag):
                    raise ResourceModifiedError("etag mismatch")
                raw = data.read() if hasattr(data, "read") else data
                container.sequence += 1
                container.blobs[name] = (raw, str(container.sequence), metadata or {})

            def download_blob(self):
                if name not in container.blobs:
                    raise ResourceNotFoundError("missing")
                raw, etag, _ = container.blobs[name]
                return SimpleNamespace(
                    size=len(raw), properties=SimpleNamespace(etag=etag), chunks=lambda: iter([raw])
                )

            def get_blob_properties(self):
                raw, _, metadata = container.blobs[name]
                return SimpleNamespace(size=len(raw), metadata=metadata)

        return Blob()

    def list_blobs(self, *, name_starts_with):
        return [
            SimpleNamespace(name=name) for name in self.blobs if name.startswith(name_starts_with)
        ]


@pytest.fixture
def azure(monkeypatch):
    container = MemoryContainer()

    @contextmanager
    def open_container(token=None):
        yield container

    monkeypatch.setattr(api, "open_container", open_container)
    return container, api.ArtifactStore(), BlobArtifacts(container)


def upload(store, name, value=b"test"):
    return store.upload_file(repo_id=REPO, path_in_repo=name, path_or_fileobj=io.BytesIO(value)).oid


def test_partial_upload_never_exposes_checkpoint(azure, tmp_path):
    container, client, store = azure
    before = upload(client, "sft/run/heartbeat.json")
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    container.fail_object = True
    with pytest.raises(OSError, match="interrupted"):
        client.upload_folder(
            repo_id=REPO, folder_path=folder, path_in_repo="sft/run/checkpoints/step-1/adapter"
        )
    assert store.manifest(REPO)[1] == before
    assert client.list_repo_files(REPO) == ["sft/run/heartbeat.json"]


def test_concurrent_heartbeat_does_not_drop_checkpoint(azure):
    container, client, _ = azure
    upload(client, "initial")
    container.before_head = lambda: upload(client, "heartbeat", b"alive")
    upload(client, "adapter", b"weights")
    assert set(client.list_repo_files(REPO)) == {"initial", "heartbeat", "adapter"}


def test_pinned_revision_survives_overwrite_and_pruning(azure, tmp_path):
    _, client, _ = azure
    revision = upload(client, "checkpoint/state", b"first")
    upload(client, "checkpoint/state", b"second")
    client.delete_folder(repo_id=REPO, path_in_repo="checkpoint")
    assert not client.file_exists(REPO, "checkpoint/state")
    path = api.download_file(REPO, "checkpoint/state", revision=revision, local_dir=tmp_path)
    assert Path(path).read_bytes() == b"first"


def test_corrupt_download_never_replaces_valid_local_file(azure, tmp_path):
    container, client, _ = azure
    upload(client, "adapter", b"weights")
    target = tmp_path / "adapter"
    target.write_bytes(b"valid previous download")
    key = f"{repo_prefix(REPO)}/objects/{hashlib.sha256(b'weights').hexdigest()}"
    container.blobs[key] = (b"corrupt", "etag", {})
    with pytest.raises(ValueError, match="checksum"):
        api.download_file(REPO, "adapter", local_dir=tmp_path)
    assert target.read_bytes() == b"valid previous download"
    assert not list(tmp_path.glob(".artifact-*"))


@pytest.mark.parametrize("path", ["../secret", "/secret", "a/../b", "a//b", "a\\b", "a/./b"])
def test_rejects_unsafe_paths_before_network(azure, path):
    container, client, _ = azure
    with pytest.raises(ValueError, match="invalid artifact path"):
        upload(client, path)
    assert not container.blobs


def test_rejects_download_through_symlink(azure, tmp_path):
    _, client, _ = azure
    upload(client, "directory/file")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "cache"
    destination.mkdir()
    (destination / "directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        api.download_file(REPO, "directory/file", local_dir=destination)
    assert not list(outside.iterdir())


def test_missing_revision_never_resets_head(azure):
    container, client, _ = azure
    revision = upload(client, "original")
    del container.blobs[f"{repo_prefix(REPO)}/revisions/{revision}.json"]
    with pytest.raises(ArtifactNotFoundError):
        upload(client, "new")
    assert container.blobs[f"{repo_prefix(REPO)}/HEAD"][0] == revision.encode()


def test_manifest_integrity_is_checked(azure):
    container, client, _ = azure
    revision = upload(client, "original")
    key = f"{repo_prefix(REPO)}/revisions/{revision}.json"
    raw, etag, metadata = container.blobs[key]
    container.blobs[key] = (raw.replace(b"original", b"tampered"), etag, metadata)
    with pytest.raises(ValueError, match="manifest checksum"):
        client.list_repo_files(REPO)


def test_snapshot_is_pinned_once(azure, tmp_path):
    _, client, store = azure
    upload(client, "a", b"original-a")
    revision = upload(client, "b", b"original-b")
    upload(client, "a", b"new-a")
    api.download_snapshot(REPO, revision=revision, local_dir=tmp_path)
    assert (tmp_path / "a").read_bytes() == b"original-a"
    assert (tmp_path / "b").read_bytes() == b"original-b"
    assert store.manifest(REPO)[1] != revision


def test_worker_capability_never_contains_account_key(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from flash.artifacts.config import ENV_KEYS, runtime_environment

    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(
        "FLASH_ARTIFACT_BLOB_CONNECTION_STRING",
        "DefaultEndpointsProtocol=https;AccountName=exampletest;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
    )
    monkeypatch.setenv("FLASH_ARTIFACT_BLOB_CONTAINER", "artifacts-dev")
    result = runtime_environment()
    assert set(result) == {"FLASH_ARTIFACT_BLOB_SAS_URL"}
    value = result["FLASH_ARTIFACT_BLOB_SAS_URL"]
    assert "dGVzdA==" not in value
    url = urlsplit(value)
    assert url.path == "/artifacts-dev"
    permissions = parse_qs(url.query)["sp"][0]
    assert set(permissions) == set("rcwl")


def test_upload_uses_same_bytes_for_digest_and_transfer(azure, tmp_path, monkeypatch):
    container, client, _ = azure
    source = tmp_path / "live-log"
    source.write_bytes(b"first")
    original = container.get_blob_client

    def get_blob(name):
        blob = original(name)
        upload_blob = blob.upload_blob

        def mutate_then_upload(data, **kwargs):
            if "/objects/" in name:
                source.write_bytes(b"later appended output")
            return upload_blob(data, **kwargs)

        blob.upload_blob = mutate_then_upload
        return blob

    monkeypatch.setattr(container, "get_blob_client", get_blob)
    client.upload_file(repo_id=REPO, path_in_repo="log", path_or_fileobj=source)
    downloaded = api.download_file(REPO, "log", local_dir=tmp_path / "download")
    assert Path(downloaded).read_bytes() == b"first"


def test_custom_attempt_receives_derived_session_token_only(monkeypatch):
    import time

    from flash.artifacts.config import ENV_KEYS
    from flash.custom.managed import session_token
    from flash.providers._lifecycle.instances.instance import build_payload
    from flash.schema import spec_from_dict
    from tests._helpers.source_snapshot import valid_source_snapshot

    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "controller-signing-key")
    spec = spec_from_dict(
        {"model": "Qwen/Qwen3.5-0.8B", "algorithm": "custom", "custom": {"file": "algorithm.py"}}
    )
    payload = build_payload(
        spec,
        2,
        arm="modal",
        source_snapshot=valid_source_snapshot(),
        deadline_at=time.time() + 3600,
    )
    assert payload["env"]["FLASH_CUSTOM_SESSION_TOKEN"] == session_token(spec.run_id, 2)
    assert payload["env"]["FLASH_CUSTOM_SESSION_TOKEN"] != session_token(spec.run_id, 1)
    assert "controller-signing-key" not in str(payload)
    assert "HF_TOKEN" not in payload["env"]


def test_supplied_sas_cannot_redirect_development_to_production(monkeypatch):
    from flash.artifacts.config import runtime_environment

    monkeypatch.setenv("FLASH_ARTIFACT_BLOB_CONTAINER", "artifacts-dev")
    monkeypatch.setenv(
        "FLASH_ARTIFACT_BLOB_SAS_URL", "https://test.blob.core.windows.net/artifacts?sig=secret"
    )
    with pytest.raises(ValueError, match="destination does not match") as error:
        runtime_environment()
    assert "sig=secret" not in str(error.value)


def test_namespace_initialization_preserves_existing_head(azure):
    _, client, _ = azure
    empty = client.create_repo(REPO).oid
    assert client.repo_info(REPO).sha == empty
    updated = upload(client, "checkpoint", b"weights")
    assert client.create_repo(REPO).oid == updated
    assert client.list_repo_files(REPO) == ["checkpoint"]


def test_namespace_initialization_rejects_public_container(azure, monkeypatch):
    container, client, _ = azure
    monkeypatch.setattr(
        container, "get_container_properties", lambda: SimpleNamespace(public_access="blob")
    )
    with pytest.raises(ValueError, match="public access"):
        client.create_repo(REPO)
    assert not container.blobs
