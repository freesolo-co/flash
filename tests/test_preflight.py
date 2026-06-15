"""Preflight checks: operator credentials (server-side) and the client login check."""

from __future__ import annotations

import pytest

import autoslm.providers.preflight as pf


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "RUNPOD_API_KEY",
        "PRIME_API_KEY",
        "HF_REPO",
        "HUGGINGFACE_TOKEN",
        "VAST_API_KEY",
        "AUTOSLM_PROVIDERS",
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
    assert "HUGGINGFACE_TOKEN" in msg
    assert "operator" in msg  # the message targets the operator, not end users


def test_preflight_passes_when_present(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_token")
    pf.check_run_preflight()  # should not raise (HF_REPO is per-run, not an operator var)


def test_preflight_require_hf_false_still_needs_provider_keys(clean_env, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("PRIME_API_KEY", "pit-key")
    pf.check_run_preflight(require_hf=False)  # no HF_REPO needed
    with pytest.raises(pf.PreflightError):
        pf.check_run_preflight(require_hf=True)


def test_runpod_key_is_env_only(clean_env):
    # ~/.autoslm/config.json holds the AutoSLM key; it must never be read as a RunPod key.
    from autoslm.providers.runpod.auth import load_api_key

    assert load_api_key() is None


def test_preflight_vast_only_pin_ignores_runpod(clean_env, monkeypatch):
    """Fix #7: AUTOSLM_PROVIDERS=vast is a Vast-only control plane — preflight must NOT
    demand RUNPOD_API_KEY, only the pinned provider's key (+ the shared HF requirement)."""
    monkeypatch.setenv("AUTOSLM_PROVIDERS", "vast")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_token")
    monkeypatch.setenv(
        "PRIME_API_KEY", "pit-key"
    )  # shared run infra (worker prime-installs the env)
    # No RUNPOD_API_KEY set, but vast is the only pinned provider -> runpod not required.
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "VAST_API_KEY" in msg  # the pinned provider's key is demanded
    assert "RUNPOD_API_KEY" not in msg  # ...but RunPod's is not, since it's deselected
    # supplying the vast key clears preflight (HF already present)
    monkeypatch.setenv("VAST_API_KEY", "vk")
    pf.check_run_preflight()


def test_preflight_vast_only_still_requires_hf(clean_env, monkeypatch):
    """Fix #7: even a Vast-only plane needs the SHARED HF write token + PRIME key (every
    substrate streams artifacts through HF), so they are checked regardless of the provider
    pin. The HF dataset repo itself is per-run ([train] hf_repo), not an operator var."""
    monkeypatch.setenv("AUTOSLM_PROVIDERS", "vast")
    monkeypatch.setenv("VAST_API_KEY", "vk")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    msg = str(excinfo.value)
    assert "HF_REPO" not in msg  # per-run, not an operator requirement
    assert "HUGGINGFACE_TOKEN" in msg
    assert "PRIME_API_KEY" in msg
    assert "RUNPOD_API_KEY" not in msg


def test_preflight_runpod_pin_ignores_vast_key(clean_env, monkeypatch):
    """Fix #7: pinning runpod-only means a stray VAST_API_KEY does not pull vast into
    preflight (the pin is authoritative)."""
    monkeypatch.setenv("AUTOSLM_PROVIDERS", "runpod")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    monkeypatch.setenv(
        "PRIME_API_KEY", "pit-key"
    )  # shared run infra (worker prime-installs the env)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf")
    monkeypatch.setenv("VAST_API_KEY", "")  # set-but-empty: vast not really configured
    pf.check_run_preflight()  # runpod-only pin, fully configured -> passes


# -- AUTOSLM_PROVIDERS pin parsing (parses-to-nothing => unset, not disable-all) -------


def test_pinned_provider_names_unset_is_none(clean_env):
    from autoslm.providers import pinned_provider_names

    assert pinned_provider_names() is None


def test_pinned_provider_names_blank_is_none(clean_env, monkeypatch):
    # A blank/whitespace/comma-only pin parses to NO known names. It must be treated as
    # UNSET (None) — the documented "no pin / default" contract — NOT an empty set (which
    # downstream would treat as an authoritative pin disabling EVERY provider) and NOT the
    # full set (which would force every provider's credentials in preflight).
    from autoslm.providers import pinned_provider_names

    for blank in (",", "  ", " , ,"):
        monkeypatch.setenv("AUTOSLM_PROVIDERS", blank)
        assert pinned_provider_names() is None, f"blank {blank!r} must be treated as unset"


def test_pinned_provider_names_all_unknown_is_none(clean_env, monkeypatch):
    # Only-unknown names (a typo) drop to nothing after filtering -> treated as unset (None)
    # rather than silently pinning to the full set or to nothing.
    from autoslm.providers import pinned_provider_names

    monkeypatch.setenv("AUTOSLM_PROVIDERS", "rnupod, vastt")
    assert pinned_provider_names() is None


def test_pinned_provider_names_drops_unknown_keeps_known(clean_env, monkeypatch):
    from autoslm.providers import pinned_provider_names

    monkeypatch.setenv("AUTOSLM_PROVIDERS", "runpod, bogus")
    assert pinned_provider_names() == {"runpod"}


def test_blank_pin_does_not_skip_preflight(clean_env, monkeypatch):
    # The bug downstream: a blank pin -> empty set -> _preflight_provider_names() selects no
    # providers -> check_run_preflight silently passes with NO credentials. Treating it as
    # unset (None) restores the default checks (runpod's key demanded), without forcing
    # Vast's key (which a full-set fallback would).
    monkeypatch.setenv("AUTOSLM_PROVIDERS", " , ")
    with pytest.raises(pf.PreflightError) as excinfo:
        pf.check_run_preflight()
    assert "RUNPOD_API_KEY" in str(excinfo.value)


def test_blank_pin_does_not_force_vast_key(clean_env, monkeypatch):
    # A parses-to-nothing pin must behave like UNSET, not like the full provider set: a
    # RunPod-only operator (RunPod creds present, Vast opt-out) must clear preflight WITHOUT
    # a VAST_API_KEY. A full-set fallback would wrongly demand it.
    monkeypatch.setenv("AUTOSLM_PROVIDERS", "rnupod, vastt")  # all-typo -> parses to nothing
    monkeypatch.setenv("RUNPOD_API_KEY", "rp")
    monkeypatch.setenv("PRIME_API_KEY", "pit")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf")
    pf.check_run_preflight()  # no VAST_API_KEY, but vast is not demanded -> passes


# -- client preflight (`slm <cmd>` without a key fails with a login hint) --------------


def test_client_requires_login(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
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
