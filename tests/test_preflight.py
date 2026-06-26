"""Preflight checks: operator credentials and the client login check."""

from __future__ import annotations

import pytest

import flash.providers.preflight as pf
import flash.providers.runpod.keys as runpod_keys

# Every operator credential a managed control plane requires at deploy.
_ALL_REQUIRED = (
    "RUNPOD_API_KEY",
    "LAMBDA_API_KEY",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "FREESOLO_INTERNAL_KEY",
)


def _set_runpod(monkeypatch, value: str) -> None:
    """Set RUNPOD_API_KEY and drop the cached key pool so key_count() re-reads it."""
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    runpod_keys.reset()


def _full_config(monkeypatch) -> None:
    """A complete, deploy-ready operator config (>= 2 RunPod accounts + every other credential)."""
    _set_runpod(monkeypatch, "rp-a,rp-b")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp")
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HF_REPO", *_ALL_REQUIRED):
        monkeypatch.delenv(var, raising=False)
    runpod_keys.reset()  # don't let a previously-cached pool leak in
    yield
    runpod_keys.reset()  # ...or out


def test_preflight_lists_all_missing(clean_env):
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    for var in _ALL_REQUIRED:
        assert var in msg, f"{var} should be reported missing"
    assert "HF_REPO" not in msg  # per-run, not operator config
    assert "operator" in msg


def test_preflight_passes_with_full_config(clean_env, monkeypatch):
    _full_config(monkeypatch)
    pf.check_run_preflight()  # no raise


def test_preflight_requires_two_runpod_keys(clean_env, monkeypatch):
    _full_config(monkeypatch)
    # A single-account pool is rejected (it can't reap/fail over across accounts).
    _set_runpod(monkeypatch, "only-one")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "RUNPOD_API_KEY" in msg
    assert "found 1" in msg
    # Two accounts -> accepted.
    _set_runpod(monkeypatch, "rp-a,rp-b")
    pf.check_run_preflight()


def test_preflight_rejects_duplicate_runpod_keys(clean_env, monkeypatch):
    # The same key twice satisfies a raw count of 2 but authenticates to ONE account, so reap/failover
    # still can't cross to a second — preflight must count DISTINCT keys and reject it.
    _full_config(monkeypatch)
    _set_runpod(monkeypatch, "rp-a,rp-a")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "RUNPOD_API_KEY" in msg
    assert "found 1" in msg  # one DISTINCT account, despite two raw entries
    # Whitespace around a dup doesn't make it distinct either.
    _set_runpod(monkeypatch, "rp-a, rp-a ")
    with pytest.raises(pf.PreflightError):
        pf.check_run_preflight()


def test_preflight_rejects_empty_runpod_pool(clean_env, monkeypatch):
    # RUNPOD_API_KEY present but parsing to NO usable keys (","-only / whitespace) is "missing", not
    # a silently-accepted empty pool — keys() drops the empties so the count is 0.
    _full_config(monkeypatch)
    for empty in (",", "   ", " , , "):
        _set_runpod(monkeypatch, empty)
        with pytest.raises(pf.PreflightError) as excinfo:
            pf.check_run_preflight()
        assert "RUNPOD_API_KEY" in str(excinfo.value), f"{empty!r} should flag RUNPOD as missing"


@pytest.mark.parametrize(
    "var",
    ["LAMBDA_API_KEY", "FREESOLO_INTERNAL_KEY", "GITHUB_TOKEN", "HF_TOKEN"],
)
def test_preflight_rejects_whitespace_only_credentials(clean_env, monkeypatch, var):
    # A whitespace-only secret would pass a bare truthiness check but fail later when the provider
    # actually authenticates with it. Preflight strips before deciding presence, so it's caught now.
    _full_config(monkeypatch)
    monkeypatch.setenv(var, "   ")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert var in str(excinfo.value)


def test_preflight_requires_each_provider_and_internal_key(clean_env, monkeypatch):
    # Dropping any single required credential from an otherwise-complete config fails the deploy.
    for missing in ("LAMBDA_API_KEY", "FREESOLO_INTERNAL_KEY"):
        _full_config(monkeypatch)
        monkeypatch.delenv(missing, raising=False)
        with pytest.raises(pf.PreflightError) as excinfo:
            pf.check_run_preflight()
        assert missing in str(excinfo.value)


def test_preflight_requires_hf_unconditionally(clean_env, monkeypatch):
    # HF is always required (no require_hf escape hatch): a config complete except HF_TOKEN fails,
    # and only HF is flagged (GitHub is present).
    _full_config(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "HF_TOKEN" in msg
    assert "GITHUB_TOKEN" not in msg


def test_runpod_key_is_env_only(clean_env):
    from flash.providers.runpod.auth import load_api_key

    assert load_api_key() is None


def test_preflight_always_requires_shared_hf_and_github(clean_env, monkeypatch):
    # Complete EXCEPT the shared HF/GitHub tokens -> both still flagged (HF_REPO is per-run, not here).
    _set_runpod(monkeypatch, "rp-a,rp-b")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "HF_REPO" not in msg
    assert "HF_TOKEN" in msg
    assert "GITHUB_TOKEN" in msg


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
