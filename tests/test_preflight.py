"""Preflight checks: operator credentials and the client login check."""

from __future__ import annotations

import os

import pytest

import flash.providers.preflight as pf
import flash.providers.runpod.keys as runpod_keys

# Credentials every control plane needs regardless of which GPU substrate it runs on.
_ALWAYS_REQUIRED = (
    "HF_TOKEN",
    "FREESOLO_INTERNAL_KEY",
)
# Any ONE of these enables a GPU provider and satisfies the substrate floor.
_PROVIDER_KEYS = ("RUNPOD_API_KEY", "LAMBDA_API_KEY", "VAST_API_KEY")


def _set_runpod(monkeypatch, value: str) -> None:
    """Set RUNPOD_API_KEY and drop the cached key pool so key_count() re-reads it."""
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    runpod_keys.reset()


def _minimal_config(monkeypatch) -> None:
    """The smallest config that boots: one provider + the always-required credentials."""
    _set_runpod(monkeypatch, "rp-a,rp-b")
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HF_REPO", "GITHUB_TOKEN", *_ALWAYS_REQUIRED, *_PROVIDER_KEYS):
        monkeypatch.delenv(var, raising=False)
    runpod_keys.reset()  # don't let a previously-cached pool leak in
    _clear_provider_cache()
    yield
    runpod_keys.reset()  # ...or out
    _clear_provider_cache()


def _clear_provider_cache() -> None:
    """Drop the per-name Provider singletons so is_configured() re-reads the env."""
    import flash.providers as providers

    providers._get_provider.cache_clear()


def test_preflight_lists_all_missing(clean_env):
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    for var in _ALWAYS_REQUIRED:
        assert var in msg, f"{var} should be reported missing"
    # The substrate floor names EVERY way to satisfy it, not just the one Freesolo uses.
    for var in _PROVIDER_KEYS:
        assert var in msg, f"{var} should be offered as a way to configure a provider"
    assert "HF_REPO" not in msg  # per-run, not operator config
    assert "operator" in msg


def test_preflight_passes_with_minimal_config(clean_env, monkeypatch):
    _minimal_config(monkeypatch)
    pf.check_run_preflight()  # no raise


@pytest.mark.parametrize("provider_var", _PROVIDER_KEYS)
def test_preflight_accepts_any_single_provider(clean_env, monkeypatch, provider_var):
    """A self-hosted plane picks its own substrate: ONE configured provider is enough.

    This is the whole point of the floor - requiring RunPod AND Lambda locked self-hosters out of
    a working single-provider deployment.
    """
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")
    if provider_var == "RUNPOD_API_KEY":
        _set_runpod(monkeypatch, "rp-a")
    else:
        monkeypatch.setenv(provider_var, "key")
    _clear_provider_cache()
    pf.check_run_preflight()  # no raise

    from flash.providers import available_providers

    expected = {
        "RUNPOD_API_KEY": "runpod",
        "LAMBDA_API_KEY": "lambda",
        "VAST_API_KEY": "vast",
    }[provider_var]
    assert available_providers() == (expected,)


def test_preflight_rejects_zero_providers(clean_env, monkeypatch):
    """Credentials complete but NO GPU substrate -> nothing can ever be allocated."""
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")
    _clear_provider_cache()
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "no GPU provider is configured" in str(excinfo.value)


def test_runpod_unconfigured_is_not_available(clean_env):
    """RunPod must be gated on its key like every other provider.

    It previously reported itself configured unconditionally, so a Lambda-only plane ranked
    RunPod classes it could never provision and died at submit instead of allocating on Lambda.
    """
    from flash.providers import available_providers

    assert "runpod" not in available_providers()


def test_single_runpod_account_is_allowed_with_warning(clean_env, monkeypatch, caplog):
    """One RunPod account still trains runs; it only loses cross-account failover.

    Refusing to boot on it blocked every operator with a single RunPod account.
    """
    _minimal_config(monkeypatch)
    _set_runpod(monkeypatch, "only-one")
    _clear_provider_cache()
    with caplog.at_level("WARNING"):
        pf.check_run_preflight()  # no raise
    assert any("RUNPOD_API_KEY" in r.getMessage() for r in caplog.records)


def test_empty_runpod_pool_does_not_enable_runpod(clean_env, monkeypatch):
    """RUNPOD_API_KEY parsing to NO usable keys is 'unconfigured', not an empty enabled pool."""
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")
    from flash.providers import available_providers

    for empty in (",", "   ", " , , "):
        _set_runpod(monkeypatch, empty)
        _clear_provider_cache()
        assert "runpod" not in available_providers(), f"{empty!r} must not enable runpod"
        with pytest.raises(pf.PreflightError) as excinfo:
            pf.check_run_preflight()
        assert "no GPU provider is configured" in str(excinfo.value)


@pytest.mark.parametrize("var", list(_ALWAYS_REQUIRED))
def test_preflight_rejects_whitespace_only_credentials(clean_env, monkeypatch, var):
    # A whitespace-only secret would pass a bare truthiness check but fail later when the provider
    # actually authenticates with it. Preflight strips before deciding presence, so it's caught now.
    _minimal_config(monkeypatch)
    monkeypatch.setenv(var, "   ")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert var in str(excinfo.value)


def test_preflight_requires_hf_unconditionally(clean_env, monkeypatch):
    # HF is always required (no require_hf escape hatch): artifacts stream through HF on every
    # provider, so a config complete except HF_TOKEN fails.
    _minimal_config(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "HF_TOKEN" in str(excinfo.value)


def test_github_token_is_optional_with_warning(clean_env, monkeypatch, caplog):
    """GITHUB_TOKEN gates private env repos and `env push`, not the ability to run a job.

    Public GitHub environment refs load without it, so it must not block startup.
    """
    _minimal_config(monkeypatch)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with caplog.at_level("WARNING"):
        pf.check_run_preflight()  # no raise
    assert any("GITHUB_TOKEN" in r.getMessage() for r in caplog.records)


def test_runpod_key_is_env_only(clean_env):
    from flash.providers.runpod.auth import load_api_key

    assert load_api_key() is None


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


@pytest.mark.parametrize("var", ["HF_TOKEN", "GITHUB_TOKEN"])
@pytest.mark.parametrize("padded", ["  hf_tok  ", "hf_tok\n", "\thf_tok", "hf_tok "])
def test_preflight_normalizes_credentials_it_accepted_after_stripping(
    clean_env, monkeypatch, var, padded
):
    """Preflight must leave the environment holding the value it JUDGED, not the padded original.

    `_present` strips before deciding a credential is set, so a token with a trailing newline (a
    routine artifact of a copied `.env`) passes startup. The consumers then read `os.environ`
    raw and `HfApi(token=...)` does not strip either -- the padding reaches the wire as
    `Authorization: Bearer <sp><sp>hf_...`, HF rejects it, and the first submit dies creating the
    artifact repo long after the plane reported healthy.

    Asserted on `os.environ` rather than on "preflight did not raise" because not raising was
    already true while the bug was live: the accept-decision and the value consumers read were
    two different things, and only the second one is what breaks.
    """
    _minimal_config(monkeypatch)
    monkeypatch.setenv(var, padded)
    pf.check_run_preflight()  # no raise: a stray newline is a typo, not a missing credential
    assert os.environ[var] == "hf_tok"


def test_preflight_leaves_a_clean_credential_untouched(clean_env, monkeypatch):
    """The normalizer must be a no-op on the normal case, including for an UNSET variable --
    it must not materialize an empty GITHUB_TOKEN that then reads as configured-but-blank."""
    _minimal_config(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_tok")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    pf.check_run_preflight()
    assert os.environ["HF_TOKEN"] == "hf_tok"
    assert "GITHUB_TOKEN" not in os.environ


def test_preflight_still_rejects_a_whitespace_only_hf_token(clean_env, monkeypatch):
    """Normalizing must not turn a whitespace-only credential into an accepted one: stripping it
    yields "", which is absent, and preflight must still refuse to boot."""
    _minimal_config(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "   \n ")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "HF_TOKEN" in str(excinfo.value)
