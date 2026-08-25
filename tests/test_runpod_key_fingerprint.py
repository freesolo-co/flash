"""CPU contracts for non-secret RunPod account ownership fingerprints."""

from __future__ import annotations

import pytest


def _reset_pool(monkeypatch, value: str) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    from flash.providers.runpod.client import auth

    auth.reset()


def test_key_fingerprint_is_stable_and_non_revealing():
    from flash.providers.runpod.client import api

    secret = "rpk-supersecret-value-123"
    fingerprint = api.key_fingerprint(secret)
    assert fingerprint == api.key_fingerprint(secret)
    assert secret not in fingerprint
    assert fingerprint.startswith("rpk-")
    assert len(fingerprint) == 68
    assert api.key_fingerprint("a-different-key") != fingerprint


def test_key_lookup_rejects_unknown_fingerprint_without_leaking_credentials(monkeypatch):
    from flash.providers.runpod.client import api

    keys = ["secretA", "secretB"]
    monkeypatch.setattr(api._keys, "keys", lambda: keys)

    with pytest.raises(api.RunpodApiError, match="exactly one") as exc_info:
        api._key_for_fingerprint("rpk-" + "0" * 64)

    assert all(key not in str(exc_info.value) for key in keys)


def test_key_lookup_rejects_colliding_configured_fingerprints(monkeypatch):
    from flash.providers.runpod.client import api

    keys = ["secretA", "secretB"]
    fingerprint = "rpk-" + "a" * 64
    monkeypatch.setattr(api._keys, "keys", lambda: keys)
    monkeypatch.setattr(api, "key_fingerprint", lambda _key: fingerprint)

    with pytest.raises(api.RunpodApiError, match="exactly one") as exc_info:
        api._key_for_fingerprint(fingerprint)

    assert all(key not in str(exc_info.value) for key in keys)


def test_repeated_identical_pool_key_still_resolves_its_fingerprint(monkeypatch):
    from flash.providers.runpod.client import api

    _reset_pool(monkeypatch, "secretA,secretB,secretA")

    assert api._key_for_fingerprint(api.key_fingerprint("secretA")) == "secretA"
    assert api._key_for_fingerprint(api.key_fingerprint("secretB")) == "secretB"
