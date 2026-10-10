"""Preflight checks: operator credentials and the client login check."""

from __future__ import annotations

import os

import pytest

import flash.providers.core.preflight as pf
import flash.providers.runpod.client.auth as runpod_keys
from flash._internal.channel import CLI_NAME

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
    for var in (
        "HF_REPO",
        "GITHUB_TOKEN",
        "FLASH_VAST_RESULT_ORIGINS",
        *_ALWAYS_REQUIRED,
        *_PROVIDER_KEYS,
    ):
        monkeypatch.delenv(var, raising=False)
    runpod_keys.reset()  # don't let a previously-cached pool leak in
    _clear_provider_cache()
    yield
    runpod_keys.reset()  # ...or out
    _clear_provider_cache()


def _clear_provider_cache() -> None:
    """Drop the per-name Provider singletons so is_configured() re-reads the env."""
    from flash.providers.core import registry as providers

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

    from flash.providers.core.registry import available_providers

    expected = {
        "RUNPOD_API_KEY": "runpod",
        "LAMBDA_API_KEY": "lambda",
        "VAST_API_KEY": "vast",
    }[provider_var]
    assert available_providers() == (expected,)


def test_preflight_validates_vast_result_origins_when_vast_is_configured(clean_env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fsk")
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.setenv("FLASH_VAST_RESULT_ORIGINS", "http://signed-secret.example.com")
    _clear_provider_cache()

    with pytest.raises(pf.PreflightError) as exc_info:
        pf.require_operator_config()
    detail = str(exc_info.value)
    assert "FLASH_VAST_RESULT_ORIGINS" in detail
    assert "exact canonical HTTPS origins" in detail
    assert "signed-secret.example.com" not in detail


def test_preflight_validates_present_vast_result_origins_without_vast(clean_env, monkeypatch):
    _minimal_config(monkeypatch)
    monkeypatch.setenv("FLASH_VAST_RESULT_ORIGINS", "https://user:secret@logs.example.com")

    with pytest.raises(pf.PreflightError) as exc_info:
        pf.require_operator_config()
    detail = str(exc_info.value)
    assert "FLASH_VAST_RESULT_ORIGINS" in detail
    assert "user:secret" not in detail


def test_preflight_accepts_blank_vast_result_origins_as_default(clean_env, monkeypatch):
    _minimal_config(monkeypatch)
    monkeypatch.setenv("FLASH_VAST_RESULT_ORIGINS", "")
    pf.require_operator_config()


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
    from flash.providers.core.registry import available_providers

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
    from flash.providers.core.registry import available_providers

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


def test_require_operator_config_does_not_log_the_advisory_summary(clean_env, monkeypatch, caplog):
    """`require_operator_config` is the refusing half only, so a second caller validates silently.

    Paired against the full `check_run_preflight` on the SAME config: without that pair, a test
    asserting "no records" would also pass if the config simply had nothing to warn about, and the
    split would be unproven. This config warns twice (one RunPod account, no GITHUB_TOKEN) and logs
    the provider summary, so all three lines are on the table.
    """
    _minimal_config(monkeypatch)
    _set_runpod(monkeypatch, "only-one")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    _clear_provider_cache()

    # at_level on the "flash" logger, not the root: the provider summary is INFO, and an earlier
    # test that calls configure_logging leaves this logger pinned at WARNING, which filters the
    # record before the root level is ever consulted. Raising only the root passes alone and fails
    # behind that test.
    with caplog.at_level("INFO", logger="flash"):
        pf.require_operator_config()
    assert caplog.records == []

    with caplog.at_level("INFO", logger="flash"):
        pf.check_run_preflight()
    logged = [r.getMessage() for r in caplog.records]
    assert any("RUNPOD_API_KEY" in m for m in logged)
    assert any("GITHUB_TOKEN" in m for m in logged)
    assert any("GPU provider(s) configured" in m for m in logged)


def test_require_operator_config_still_refuses_a_missing_credential(clean_env, monkeypatch):
    """Dropping the advisory phase must not drop the check itself.

    `run_server` calls this half purely to avoid double-logging what the lifespan copy already
    prints; if it also skipped validation, the early call would be decoration and the operator
    would be back to reading a PreflightError out of an ASGI startup traceback.
    """
    _minimal_config(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.require_operator_config()
    assert "HF_TOKEN" in str(excinfo.value)


def test_runpod_key_is_env_only(clean_env):
    from flash.providers.runpod.client.auth import load_api_key

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

        with pytest.raises(ClientError, match=rf"{CLI_NAME} login"):
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
