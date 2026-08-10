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
    from flash.providers.runpod import auth

    auth.reset()


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
        lambda eid, key, *, deadline_at=None: (
            seen.update(health=(eid, key, deadline_at)) or {"ok": True}
        ),
    )

    fp_b = api.key_fingerprint("secretB")
    assert api.delete_endpoint_for_fingerprint("ep-1", fp_b) is True
    assert seen["delete"] == ("ep-1", "secretB")  # resolved to the real owning key, internally
    api.endpoint_health_for_fingerprint("ep-1", fp_b, deadline_at=123.0)
    assert seen["health"] == ("ep-1", "secretB", 123.0)

    with pytest.raises(api.RunpodApiError):
        api.delete_endpoint_for_fingerprint("ep-1", "rpk-no-such-account")  # no pool key matches


def test_submit_status_cancel_and_delete_keep_owning_key_after_rotation(monkeypatch):
    from flash.providers.runpod import api, auth

    _reset_pool(monkeypatch, "secretA,secretB")
    owner = api.key_fingerprint("secretA")
    calls = []

    def request(key, url, **kwargs):
        calls.append((key, url, kwargs.get("method", "GET")))
        if url.endswith("/run"):
            return {"id": "job-1"}
        if "/status/" in url:
            return {"status": "IN_PROGRESS"}
        if "/cancel/" in url:
            return {"id": "job-1", "status": "CANCELLED"}
        return {}

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", request)
    auth.advance_key()
    assert auth.active_key() == "secretB"

    assert (
        api.submit_job(
            "ep-1",
            {"x": 1},
            key_fingerprint=owner,
            deadline_at=4_000_000_000.0,
        )
        == "job-1"
    )
    assert (
        api.job_status(
            "ep-1",
            "job-1",
            key_fingerprint=owner,
            deadline_at=4_000_000_000.0,
        )["status"]
        == "IN_PROGRESS"
    )
    assert api.cancel_job("ep-1", "job-1", key_fingerprint=owner)["status"] == "CANCELLED"
    assert api.delete_endpoint_for_fingerprint("ep-1", owner) is True

    assert [key for key, _url, _method in calls] == ["secretA"] * 4
