"""`flash export` core: the HF-to-HF adapter copy and the destination-token resolution.

These run offline by faking ``huggingface_hub`` (the copy is a download-then-upload, so we record
the calls instead of touching the Hub).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _install_fake_hub(monkeypatch, *, download, hf_api):
    """Inject a fake ``huggingface_hub`` module exposing HfApi + snapshot_download."""
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = hf_api
    fake.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_export_adapter_reads_source_with_operator_token_writes_dest_with_user_token(monkeypatch):
    calls: dict = {}

    def fake_snapshot_download(*, repo_id, repo_type, allow_patterns, local_dir, token):
        calls["download"] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "allow_patterns": allow_patterns,
            "token": token,
        }
        # Materialize the adapter folder exactly where export_adapter looks for it.
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            calls["dest_token"] = token

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            calls["create_repo"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message):
            calls["upload"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "files": sorted(p.name for p in Path(folder_path).iterdir()),
            }

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)

    from flash.serve.export import export_adapter

    url = export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        source_token="hf_operator",
        private=True,
    )
    assert url == "https://huggingface.co/me/adapters"
    # Read the PRIVATE source dataset repo with the operator token, only the adapter subfolder.
    assert calls["download"]["repo_id"] == "org/test-runs"
    assert calls["download"]["repo_type"] == "dataset"
    assert calls["download"]["allow_patterns"] == ["rl/run-x/seed0/adapter/*"]
    assert calls["download"]["token"] == "hf_operator"
    # Write the user's MODEL repo with the user's token; create it (private) if missing.
    assert calls["dest_token"] == "hf_user"
    assert calls["create_repo"] == {
        "repo_id": "me/adapters",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert calls["upload"]["repo_id"] == "me/adapters"
    assert calls["upload"]["repo_type"] == "model"
    assert set(calls["upload"]["files"]) == {"adapter_config.json", "adapter_model.safetensors"}


def test_export_adapter_falls_back_to_hf_token_env_for_source(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    seen: dict = {}

    def fake_snapshot_download(*, repo_id, repo_type, allow_patterns, local_dir, token):
        seen["token"] = token
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def upload_folder(self, **kw):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
    )
    assert seen["token"] == "hf_from_env"  # no source_token -> HF_TOKEN


def test_export_adapter_raises_value_error_when_source_is_empty(monkeypatch):
    def fake_snapshot_download(*, local_dir, **kw):
        # Download succeeds but the adapter subfolder has no files (nothing matched).
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ValueError, match="no adapter artifacts"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
        )


def test_export_adapter_wraps_download_failure_in_serving_error(monkeypatch):
    from flash.serve.deploy import ServingError

    def fake_snapshot_download(**kw):
        raise RuntimeError("401 Unauthorized")

    class FakeHfApi:
        def __init__(self, token=None):
            pass

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    with pytest.raises(ServingError, match="could not download adapter"):
        export_adapter(
            source_repo="org/test-runs",
            source_subfolder="rl/run-x/seed0/adapter",
            dest_repo="me/adapters",
            dest_token="hf_user",
        )


def test_resolve_hf_token_priority_explicit_then_env_then_dotenv(tmp_path, monkeypatch):
    from flash.client.runtime_secrets import resolve_hf_token

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    # Nothing set anywhere -> None.
    assert resolve_hf_token(None) is None

    # A local .env supplies the token when the env doesn't.
    (tmp_path / ".env").write_text('HF_TOKEN="hf_from_dotenv"\n')
    assert resolve_hf_token(None) == "hf_from_dotenv"

    # The process environment wins over .env.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert resolve_hf_token(None) == "hf_from_env"

    # An explicit value (the --api-key flag) wins over everything.
    assert resolve_hf_token("hf_explicit") == "hf_explicit"
