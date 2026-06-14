"""Preflight checks: operator credentials (server-side) and the client login check."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import autoslm.flash.preflight as pf


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("RUNPOD_API_KEY", "HF_REPO", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)


# -- operator preflight (the control plane fails fast at startup) ----------------------


def test_preflight_lists_all_missing(clean_env):
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "RUNPOD_API_KEY" in msg
    assert "HF_REPO" in msg
    assert "HUGGINGFACE_TOKEN" in msg
    assert "operator" in msg  # the message targets the operator, not end users


def test_preflight_passes_when_present(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("HF_REPO", "org/autoslm-runs")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_token")
    pf.check_run_preflight()  # should not raise


def test_preflight_require_hf_false_only_needs_key(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    pf.check_run_preflight(require_hf=False)  # no HF_REPO needed
    with pytest.raises(pf.PreflightError):
        pf.check_run_preflight(require_hf=True)


def test_runpod_key_is_env_only(clean_env):
    # ~/.autoslm/config.json holds the AutoSLM key; it must never be read as a RunPod key.
    from autoslm.flash.auth import load_api_key

    assert load_api_key() is None


# -- client preflight (`slm <cmd>` without a key fails with a login hint) --------------


def test_client_requires_login(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUTOSLM_API_KEY", raising=False)
    import importlib

    import autoslm.client.config as client_config

    importlib.reload(client_config)
    try:
        from autoslm.client import ClientError
        from autoslm.client.http import client_from_config

        with pytest.raises(ClientError, match="slm login"):
            client_from_config()
        client = client_from_config(require_key=False)
        assert client.api_key is None
    finally:
        monkeypatch.undo()
        importlib.reload(client_config)
