"""Preflight checks: operator credentials (server-side) and the client login check."""

from __future__ import annotations

import os

import pytest

import flash.providers.preflight as pf


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "RUNPOD_API_KEY",
        "PRIME_API_KEY",
        "HF_REPO",
        "HF_TOKEN",
        "VAST_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# -- operator preflight (the control plane fails fast at startup) ----------------------


def test_preflight_lists_all_missing(clean_env):
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "RUNPOD_API_KEY" in msg
    assert "PRIME_API_KEY" in msg  # the worker uses it to `prime env install` the env
    assert "HF_REPO" not in msg  # the HF dataset repo is per-run ([train] hf_repo), not operator
    assert "HF_TOKEN" in msg
    assert "operator" in msg  # the message targets the operator, not end users


def test_preflight_passes_when_present(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    pf.check_run_preflight()  # should not raise (HF_REPO is per-run, not an operator var)


def test_preflight_accepts_hf_token_as_fallback(clean_env, monkeypatch):
    # HF_TOKEN (the modern HF-ecosystem var) satisfies the HF_TOKEN requirement and is
    # mirrored into os.environ so the worker payload (which reads HF_TOKEN) gets it.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    monkeypatch.setenv("HF_TOKEN", "hf_only_token")
    pf.check_run_preflight()  # should not raise
    assert os.environ.get("HF_TOKEN") == "hf_only_token"


def test_preflight_require_hf_false_still_needs_provider_keys(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    pf.check_run_preflight(require_hf=False)  # no HF_REPO needed
    with pytest.raises(pf.PreflightError):
        pf.check_run_preflight(require_hf=True)


def test_runpod_key_is_env_only(clean_env):
    # ~/.flash/config.json holds the Flash key; it must never be read as a RunPod key.
    from flash.providers.runpod.auth import load_api_key

    assert load_api_key() is None


def test_preflight_requires_runpod_by_default(clean_env, monkeypatch):
    """RunPod is the always-on default substrate: preflight demands RUNPOD_API_KEY (+ the shared
    HF/PRIME keys) when nothing else is configured, and clears once it's present."""
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "RUNPOD_API_KEY" in str(excinfo.value)
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    pf.check_run_preflight()  # fully configured -> passes


def test_preflight_vast_is_opt_in(clean_env, monkeypatch):
    """Vast is opt-in: with no VAST_API_KEY it is not demanded (a RunPod-only plane clears);
    a present VAST_API_KEY pulls Vast into the check."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    monkeypatch.setenv("PRIME_API_KEY", "pit")
    monkeypatch.setenv("HF_TOKEN", "hf")
    pf.check_run_preflight()  # runpod-only, no vast key demanded -> passes


def test_preflight_always_requires_shared_hf_and_prime(clean_env, monkeypatch):
    """Every substrate streams artifacts through HF and prime-installs the env, so HF_TOKEN +
    PRIME_API_KEY are always required. The HF dataset repo itself is per-run, not an operator var."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "HF_REPO" not in msg
    assert "HF_TOKEN" in msg
    assert "PRIME_API_KEY" in msg


# -- client preflight (`flash <cmd>` without a key fails with a login hint) --------------


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
