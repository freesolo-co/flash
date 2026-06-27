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

        def list_repo_files(self, *, repo_id, repo_type):
            return []  # brand-new repo -> no orphans to clean

        def update_repo_settings(self, *, repo_id, repo_type, private):
            calls["update_settings"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
            }

        def upload_folder(
            self, *, repo_id, repo_type, folder_path, commit_message, delete_patterns=None
        ):
            calls["upload"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "files": sorted(p.name for p in Path(folder_path).iterdir()),
                "delete_patterns": delete_patterns,
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
    # The repo is created PRIVATE, then visibility is enforced after the upload commits (create_repo(
    # exist_ok=True) won't change an existing repo's visibility).
    assert calls["create_repo"]["private"] is True
    assert calls["update_settings"] == {
        "repo_id": "me/adapters",
        "repo_type": "model",
        "private": True,
    }
    # Stale adapter artifacts are cleared by STATIC, adapter-scoped patterns (never the user's
    # unrelated files), so a re-export of the same format is a no-op delete.
    assert calls["upload"]["delete_patterns"] == ["adapter_model*", "adapter_config.json"]


def test_export_clears_stale_adapter_weights_without_touching_user_files(monkeypatch):
    """A re-export into a repo holding a previous, differently-serialized adapter must clear the stale
    adapter weights (so a leftover ``.bin`` can't be loaded next to the new ``.safetensors``) WITHOUT
    deleting the user's unrelated files. The deletion is by STATIC adapter-scoped patterns, not a
    ``existing - uploaded`` mirror, and needs no ``list_repo_files`` call (so a listing failure can't
    silently skip cleanup either)."""
    calls: dict = {}
    listed = {"called": False}

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"new-weights")
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            pass

        def list_repo_files(self, *, repo_id, repo_type):
            listed["called"] = True  # must NOT be called: cleanup is pattern-based, not listing-based
            return []

        def update_repo_settings(self, **kw):
            pass

        def upload_folder(self, *, folder_path, delete_patterns, **kw):
            calls["files"] = sorted(p.name for p in Path(folder_path).iterdir())
            calls["delete_patterns"] = delete_patterns

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
    )
    # Static, adapter-scoped delete patterns: ``adapter_model*`` clears a stale ``.bin`` (and sharded /
    # index variants); ``adapter_config.json`` clears the old config. A hypothetical ``extra_weights.pt``
    # or any other user file matches NEITHER pattern, so it is never deleted — the data-loss the old
    # ``existing - uploaded`` mirror caused.
    assert calls["delete_patterns"] == ["adapter_model*", "adapter_config.json"]
    assert "extra_weights.pt" not in calls["delete_patterns"]
    assert not listed["called"], "stale cleanup must not depend on list_repo_files"


def test_export_public_visibility_is_deferred_until_after_upload(monkeypatch):
    """``private=False`` must not expose an empty/partial PUBLIC repo: the repo is created/ensured
    private, the adapter is uploaded, and only THEN is it flipped public — so a failed upload never
    leaves an empty public repo behind."""
    order: list = []

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            order.append(("create_repo", private))

        def update_repo_settings(self, *, repo_id, repo_type, private):
            order.append(("update_settings", private))

        def upload_folder(self, **kw):
            order.append(("upload", None))

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        private=False,
    )
    # Created private, uploaded, THEN made public — never public before the content lands.
    assert order == [("create_repo", True), ("upload", None), ("update_settings", False)]


def test_export_private_is_enforced_before_upload(monkeypatch):
    """``private=True`` (default) into a PRE-EXISTING public repo must lock it down BEFORE the upload:
    create_repo(exist_ok=True) won't change an existing repo's visibility, so the weights would
    otherwise commit while the repo is still public. Visibility private must be set before upload."""
    order: list = []

    def fake_snapshot_download(*, local_dir, **kw):
        adapter = Path(local_dir) / "rl/run-x/seed0/adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        return str(local_dir)

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            order.append(("create_repo", private))

        def update_repo_settings(self, *, repo_id, repo_type, private):
            order.append(("update_settings", private))

        def upload_folder(self, **kw):
            order.append(("upload", None))

    _install_fake_hub(monkeypatch, download=fake_snapshot_download, hf_api=FakeHfApi)
    from flash.serve.export import export_adapter

    export_adapter(
        source_repo="org/test-runs",
        source_subfolder="rl/run-x/seed0/adapter",
        dest_repo="me/adapters",
        dest_token="hf_user",
        private=True,
    )
    # Locked private BEFORE the upload (no post-upload visibility flip needed for a private export).
    assert order == [("create_repo", True), ("update_settings", True), ("upload", None)]


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

        def update_repo_settings(self, **kw):
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

    # The huggingface_hub aliases are deliberately NOT accepted: only HF_TOKEN is.
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alias")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_alias2")
    assert resolve_hf_token(None) is None
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    # A local .env supplies the token when the env doesn't.
    (tmp_path / ".env").write_text('HF_TOKEN="hf_from_dotenv"\n')
    assert resolve_hf_token(None) == "hf_from_dotenv"

    # The process environment wins over .env.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert resolve_hf_token(None) == "hf_from_env"

    # An explicit value (the --api-key flag) wins over everything.
    assert resolve_hf_token("hf_explicit") == "hf_explicit"


def test_hf_api_missing_extra_raises_runtime_error_not_serving_error(monkeypatch):
    # A missing huggingface_hub extra is an internal misconfiguration (-> 500), NOT an upstream
    # gateway/transport failure (ServingError -> 502). _hf_api must raise a plain RuntimeError so the
    # route lets it surface as a 500 rather than a misleading 502.
    import builtins

    from flash.serve import export
    from flash.serve.deploy import ServingError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as ei:
        export._hf_api()
    assert not isinstance(ei.value, ServingError)  # NOT the 502 path
    assert "huggingface_hub" in str(ei.value)
