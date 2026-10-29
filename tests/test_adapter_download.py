"""Authenticated downloads keep artifact credentials server-side and identify exact weights."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.runner.lifecycle import state as runner_state
from flash.runner.lifecycle import status as runner_status
from flash.serve.contract.errors import ServingError
from flash.serve.deployment import download
from flash.server.routes import adapter_download as route
from tests.test_export import _PLAIN_CONFIG, _PLAIN_WEIGHTS
from tests.test_server_api import _bearer, _finished_run, _new_user_token

pytest_plugins = ["tests._helpers.server_api_plugin"]


@pytest.fixture
def builder(monkeypatch):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        path = kwargs["directory"] / "adapter.zip"
        path.write_bytes(b"test archive")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(route, "build_archive", build)
    return calls


def test_download_final_uses_effective_model_revision_and_cleans_up(api, builder):
    key = _new_user_token()
    run_id = _finished_run(api, key)
    status = runner_status.get_status(run_id)
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle import preparation as runner_preparation

    snapshot = status.effective_preparation
    snapshot["worker_spec"]["model_revision"] = "a" * 40
    snapshot["worker_spec"]["model_revision_auto"] = True
    snapshot["preparation_digest"] = runner_preparation._preparation_digest(
        JobSpec.from_dict(status.spec),
        JobSpec.from_dict(snapshot["worker_spec"]),
        snapshot.get("adapter_identity"),
    )
    runner_state._save_status(status)
    response = api.get(
        f"/v1/runs/{run_id}/adapter",
        params={"checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert response.status_code == 200, response.text
    assert response.content == b"test archive"
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-flash-checkpoint-id"] == f"{run_id}/final"
    assert response.headers["x-flash-run-id"] == run_id
    assert (
        response.headers["x-flash-adapter-sha256"] == hashlib.sha256(response.content).hexdigest()
    )
    assert response.headers["cache-control"] == "private, no-store"
    assert len(builder) == 1
    call = builder[0]
    assert call["source_repo"] == status.effective_preparation["worker_spec"]["train"]["hf_repo"]
    assert call["base_model_revision"] == "a" * 40
    assert call["source_subfolder"].endswith("/adapter")
    assert not call["directory"].exists()


def test_download_requires_owner_and_explicit_same_run_checkpoint(api, builder):
    key = _new_user_token()
    run_id = _finished_run(api, key)
    url = f"/v1/runs/{run_id}/adapter"
    params = {"checkpoint_id": f"{run_id}/final"}
    assert api.get(url, params=params).status_code == 401
    assert api.get(url, params=params, headers=_bearer(_new_user_token())).status_code == 404
    assert api.get(url, headers=_bearer(key)).status_code == 422
    for checkpoint in ("other/final", f"{run_id}/step-bad", "../final"):
        assert (
            api.get(url, params={"checkpoint_id": checkpoint}, headers=_bearer(key)).status_code
            == 400
        )
    assert builder == []


def test_download_step_requires_known_artifact_and_final_requires_finished_run(
    api, builder, monkeypatch
):
    from flash.runner.lifecycle.status import effective_spec_from_status
    from flash.runner.results import checkpoints

    key = _new_user_token()
    run_id = _finished_run(api, key)
    status = runner_status.get_status(run_id)
    status.state = "running"
    runner_state._save_status(status)
    monkeypatch.setattr(checkpoints, "list_checkpoints", lambda spec: [{"step": 12}])
    url = f"/v1/runs/{run_id}/adapter"
    for target, expected in (("final", 409), ("step-9", 404), ("step-12", 200)):
        response = api.get(
            url, params={"checkpoint_id": f"{run_id}/{target}"}, headers=_bearer(key)
        )
        assert response.status_code == expected, response.text
    assert len(builder) == 1
    assert builder[0]["source_subfolder"] == (
        checkpoints.checkpoint_adapter_prefix(effective_spec_from_status(status), 12) + "/adapter"
    )


@pytest.mark.parametrize(
    ("failure", "code"), [(ServingError, 502), (OSError, 502), (ValueError, 404)]
)
def test_download_failure_is_sanitized_and_cleans_workspace(api, monkeypatch, failure, code):
    key = _new_user_token()
    run_id = _finished_run(api, key)
    directories = []

    def fail(**kwargs):
        directories.append(kwargs["directory"])
        raise failure("private/repo hf-secret signed-url")

    monkeypatch.setattr(route, "build_archive", fail)
    response = api.get(
        f"/v1/runs/{run_id}/adapter",
        params={"checkpoint_id": f"{run_id}/final"},
        headers=_bearer(key),
    )
    assert response.status_code == code
    assert "private/repo" not in response.text
    assert "hf-secret" not in response.text
    assert directories
    assert all(not p.exists() for p in directories)


def test_response_disconnect_cleans_archive(monkeypatch):
    workspace = tempfile.TemporaryDirectory()
    path = Path(workspace.name) / "adapter.zip"
    path.write_bytes(b"archive")

    async def disconnected(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(route.FileResponse, "__call__", disconnected)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(route.AdapterResponse(path, workspace)({}, None, None))
    assert not path.parent.exists()


def test_archive_contains_only_normalized_adapter_and_hash_manifest(tmp_path, monkeypatch):
    adapter = tmp_path / "source"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(_PLAIN_CONFIG)
    (adapter / "adapter_model.safetensors").write_bytes(_PLAIN_WEIGHTS)
    (adapter / "trainer-secret.json").write_text("hf-secret")
    monkeypatch.setattr(download, "_fetch_adapter", lambda *args: (adapter, "b" * 40))
    archive, digest = download.build_archive(
        source_repo="private/repo",
        source_subfolder="run/adapter",
        run_id="run1",
        checkpoint_id="run1/step-12",
        base_model="Qwen/Qwen3.5-9B",
        base_model_revision="a" * 40,
        directory=tmp_path,
    )
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "manifest.json",
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["schema"] == "flash-adapter-download-v1"
        assert manifest["checkpoint_id"] == "run1/step-12"
        assert manifest["source_revision"] == "b" * 40
        for name, entry in manifest["files"].items():
            assert entry["bytes"] == len(bundle.read(name))
            assert entry["sha256"] == hashlib.sha256(bundle.read(name)).hexdigest()
        config = json.loads(bundle.read("adapter_config.json"))
        assert config["base_model_name_or_path"] == manifest["base_model"]
        assert config["revision"] == manifest["base_model_revision"]
        assert "private/repo" not in json.dumps(manifest)
        assert "hf-secret" not in archive.read_bytes().decode(errors="ignore")


def test_fetch_pins_source_revision_and_downloads_only_checkpoint_files(tmp_path, monkeypatch):
    import huggingface_hub as hub

    class RepoFile:
        def __init__(self, name):
            self.path = f"run/step-12/adapter/{name}"
            self.size = 32

    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="b" * 40)
    api.list_repo_tree.return_value = [
        RepoFile(n)
        for n in ("adapter_config.json", "adapter_model.safetensors", "trainer-secret.json")
    ]
    factory = Mock(return_value=api)
    fetch = Mock()
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    monkeypatch.setattr(hub, "RepoFile", RepoFile)
    monkeypatch.setattr(hub, "HfApi", factory)
    monkeypatch.setattr(hub, "hf_hub_download", fetch)
    path, revision = download._fetch_adapter("private/repo", "run/step-12/adapter", tmp_path)
    factory.assert_called_once_with(token="server-secret")
    assert path == tmp_path / "run/step-12/adapter"
    assert revision == "b" * 40
    api.list_repo_tree.assert_called_once_with(
        repo_id="private/repo",
        repo_type="dataset",
        revision=revision,
        path_in_repo="run/step-12/adapter",
        recursive=False,
    )
    assert fetch.call_count == 2
    for call in fetch.call_args_list:
        assert call.kwargs["revision"] == revision
        assert call.kwargs["token"] == "server-secret"
        assert call.kwargs["filename"].startswith("run/step-12/adapter/")


@pytest.mark.parametrize("prefix", ["../adapter", "/run/adapter", "run//adapter"])
def test_fetch_rejects_invalid_prefixes_before_network(tmp_path, monkeypatch, prefix):
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    with pytest.raises(ValueError, match="directory"):
        download._fetch_adapter("private/repo", prefix, tmp_path)
