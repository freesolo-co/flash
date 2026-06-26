"""list_endpoints_by_key identifies pool accounts by a NON-secret fingerprint — the raw
RUNPOD_API_KEY never appears in its return value (a stray log of a ``{key: ...}`` dict or a
``failed`` list would otherwise leak a live credential). The reaper passes the fingerprint back to
``*_for_fingerprint`` helpers, which resolve it to the real key INSIDE the api module.

CPU-only; the per-key REST call is mocked.
"""

from __future__ import annotations

import pytest


def _reset_pool(monkeypatch, value):
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    from flash.providers.runpod import keys

    keys.reset()


def test_key_fingerprint_is_stable_and_non_revealing():
    from flash.providers.runpod import api

    secret = "rpk-supersecret-value-123"
    fp = api.key_fingerprint(secret)
    assert fp == api.key_fingerprint(secret)  # stable across calls
    assert secret not in fp  # never embeds the secret
    assert fp.startswith("rpk-")
    assert api.key_fingerprint("a-different-key") != fp  # distinguishes accounts


def test_list_endpoints_by_key_returns_fingerprints_not_raw_keys(monkeypatch):
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA,secretB")

    def fake_req(key, url, **kw):
        if key == "secretA":
            return [{"id": "ep-a", "name": "flash-x"}]
        raise api.RunpodApiError("403 for secretB")  # this account fails to list this cycle

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", fake_req)
    by_fp, failed = api.list_endpoints_by_key()

    # The raw secrets must appear NOWHERE in the return value (the credential-leak guard).
    blob = repr(by_fp) + repr(failed)
    assert "secretA" not in blob
    assert "secretB" not in blob
    # Accounts are identified by fingerprint instead.
    assert by_fp == {api.key_fingerprint("secretA"): [{"id": "ep-a", "name": "flash-x"}]}
    assert failed == [api.key_fingerprint("secretB")]


def test_fingerprint_helpers_resolve_to_the_owning_key(monkeypatch):
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA,secretB")
    seen = {}
    monkeypatch.setattr(
        api, "delete_endpoint_for_key", lambda eid, key: seen.update(delete=(eid, key)) or True
    )
    monkeypatch.setattr(
        api,
        "endpoint_health_for_key",
        lambda eid, key: seen.update(health=(eid, key)) or {"ok": True},
    )

    fp_b = api.key_fingerprint("secretB")
    assert api.delete_endpoint_for_fingerprint("ep-1", fp_b) is True
    assert seen["delete"] == ("ep-1", "secretB")  # resolved to the real owning key, internally
    api.endpoint_health_for_fingerprint("ep-1", fp_b)
    assert seen["health"] == ("ep-1", "secretB")

    with pytest.raises(api.RunpodApiError):
        api.delete_endpoint_for_fingerprint("ep-1", "rpk-no-such-account")  # no pool key matches
