"""Standalone mode is a deployment shape, not a permission downgrade.

``FLASH_STANDALONE`` exists so a self-hosted plane -- one with no Freesolo backend behind it --
can run at all: the normal paths validate every bearer token, project, and environment against a
SaaS backend the self-hoster does not have. The risk in that seam is obvious, so these tests pin
the direction it fails in: standalone must accept FEWER credentials than managed mode, never more.
"""

from __future__ import annotations

import pytest

from flash.server import auth


@pytest.fixture(autouse=True)
def _clear_verify_state():
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()
    yield
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_standalone_is_on_for_the_documented_spellings(monkeypatch, value: str) -> None:
    monkeypatch.setenv(auth.STANDALONE_ENV, value)
    assert auth.standalone() is True


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "no", "off", "maybe"])
def test_standalone_is_off_by_default_and_for_anything_else(monkeypatch, value: str) -> None:
    """Managed mode is the default. A typo'd value must not silently relax project ownership."""
    monkeypatch.setenv(auth.STANDALONE_ENV, value)
    assert auth.standalone() is False

    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert auth.standalone() is False


def test_standalone_rejects_an_external_token_instead_of_trusting_it(monkeypatch) -> None:
    """The failure direction that matters.

    Managed mode verifies an external bearer token against the backend. Standalone has no backend
    to ask -- and the tempting shortcut, treating "cannot verify" as "accept", would turn every
    self-hosted plane into an open one. Unverifiable means rejected; the operator key is the only
    credential standalone honours.
    """
    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")

    def _boom(*_args, **_kwargs):
        raise AssertionError("standalone must not call the Freesolo backend to verify a token")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    assert auth.authenticate("Bearer some-external-user-key") is None
    assert auth.authenticate("Bearer ") is None
    assert auth.authenticate(None) is None


def test_standalone_accepts_the_operator_key_as_internal(monkeypatch, tmp_path) -> None:
    """The operator key is the whole trust boundary of a standalone plane, so it must still work."""
    from flash.server import db

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    # Resolving the key registers it, so point the db at tmp rather than the operator's real one.
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))

    def _boom(*_args, **_kwargs):
        raise AssertionError("the operator key is resolved locally, with no backend call")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    row = auth.authenticate("Bearer operator-key")
    assert row is not None
    assert row["auth_kind"] == "internal"


def test_standalone_disables_backend_reporting_at_the_shared_gate(monkeypatch) -> None:
    """A standalone plane SETS the internal key -- it is how its own clients authenticate.

    Every best-effort reporter (billing, checkpoint registration, environment recording) gates on
    that key, so without this they would all POST the operator's key to api.freesolo.co and log a
    warning per run. One gate, so the reporters cannot drift apart.
    """
    from flash.server._internal_client import internal_key

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert internal_key() == "operator-key"

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert internal_key() is None


def test_standalone_disables_the_backend_polling_loops(monkeypatch) -> None:
    """Cost reconciliation and charge retry poll on a timer against a backend that isn't there."""
    from flash.server.billing_retry import charge_retry_enabled
    from flash.server.reconcile import reconcile_enabled

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert reconcile_enabled() is True
    assert charge_retry_enabled() is True

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert reconcile_enabled() is False
    assert charge_retry_enabled() is False


def test_standalone_disables_the_artifact_gc_sweep(monkeypatch) -> None:
    """The GC is the one background loop gated on HF_TOKEN rather than the internal key, so it
    survived the first pass and shipped the operator's key to serve.freesolo.co on every startup --
    it confirms the live set against the hosted serving registry before deleting. It can only ever
    delete inside the hardcoded Freesolo-Co/flashrun-* allowlist, which a self-hoster's token does
    not own, so standalone loses nothing by skipping it."""
    from flash.server.repo_cleanup import repo_cleanup_enabled

    monkeypatch.setenv("HF_TOKEN", "hf-operator-token")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert repo_cleanup_enabled() is True

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert repo_cleanup_enabled() is False


def test_standalone_falls_back_to_the_in_process_slot_semaphore(monkeypatch) -> None:
    """The shared RunPod slot store is a backend table; standalone caps concurrency in-process."""
    from flash.providers.runpod import slots

    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "operator-key")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)
    assert slots.internal_key() == "operator-key"

    monkeypatch.setenv(auth.STANDALONE_ENV, "1")
    assert slots.internal_key() is None
