"""Preflight checks: operator credentials and the client login check."""

from __future__ import annotations

import pytest

import flash.providers.preflight as pf


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "RUNPOD_API_KEY",
        "FLASH_ENV_BLOB_CONNECTION_STRING",
        "FLASH_ENV_PG_URL",
        "HF_REPO",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_azure(monkeypatch):
    monkeypatch.setenv("FLASH_ENV_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;x=y")
    monkeypatch.setenv("FLASH_ENV_PG_URL", "postgres://u:p@h/db")


def test_preflight_lists_all_missing(clean_env):
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "RUNPOD_API_KEY" in msg
    assert "FLASH_ENV_BLOB_CONNECTION_STRING" in msg
    assert "FLASH_ENV_PG_URL" in msg
    assert "HF_REPO" not in msg
    assert "HF_TOKEN" in msg
    assert "operator" in msg


def test_preflight_passes_when_present(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    _set_azure(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    pf.check_run_preflight()


def test_preflight_require_hf_false_still_needs_provider_keys(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    _set_azure(monkeypatch)
    pf.check_run_preflight(require_hf=False)
    with pytest.raises(pf.PreflightError):
        pf.check_run_preflight(require_hf=True)


def test_runpod_key_is_env_only(clean_env):
    from flash.providers.runpod.auth import load_api_key

    assert load_api_key() is None


def test_preflight_requires_runpod_by_default(clean_env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    _set_azure(monkeypatch)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "RUNPOD_API_KEY" in str(excinfo.value)
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    pf.check_run_preflight()


def test_preflight_passes_with_required_operator_keys(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    _set_azure(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf")
    pf.check_run_preflight()


def test_preflight_always_requires_shared_hf_and_azure(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "HF_REPO" not in msg
    assert "HF_TOKEN" in msg
    assert "FLASH_ENV_BLOB_CONNECTION_STRING" in msg
    assert "FLASH_ENV_PG_URL" in msg


def test_client_requires_login(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    import importlib

    import flash.client.config as client_config

    importlib.reload(client_config)
    try:
        from flash.client import ClientError
        from flash.client.http import client_from_config

        with pytest.raises(ClientError, match="flash login"):
            client_from_config()
        client = client_from_config(require_key=False)
        assert client.api_key is None
    finally:
        monkeypatch.undo()
        importlib.reload(client_config)
