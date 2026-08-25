"""list_endpoints_by_key identifies pool accounts by a NON-secret fingerprint — the raw
RUNPOD_API_KEY never appears in its return value (a stray log of a ``{key: ...}`` dict or a
``failed`` list would otherwise leak a live credential). The reaper passes the fingerprint back to
``*_for_fingerprint`` helpers, which resolve it to the real key INSIDE the api module.

CPU-only; the per-key REST call is mocked.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

import flash.providers.runpod.client.api as runpod_api
from flash.providers.runpod.client.api import list_endpoints as _real_list_endpoints


def _reset_pool(monkeypatch, value):
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    from flash.providers.runpod.client import auth

    auth.reset()


def test_key_fingerprint_is_stable_and_non_revealing():
    from flash.providers.runpod.client import api

    secret = "rpk-supersecret-value-123"
    fp = api.key_fingerprint(secret)
    assert fp == api.key_fingerprint(secret)  # stable across calls
    assert secret not in fp  # never embeds the secret
    assert fp.startswith("rpk-")
    assert len(fp) == 68
    assert api.key_fingerprint("a-different-key") != fp  # distinguishes accounts


def test_the_prefix_form_is_what_a_deployed_release_writes_not_history():
    """The 16-char shape is a LIVE producer's output, so its resolver is not deletable as legacy.

    Pinned because it was already deleted once in this branch as dead compatibility code and
    restored a day later. `dev`'s `key_fingerprint` is `sha256(...).hexdigest()[:12]`; only this
    branch widened it to the full digest. Every endpoint a currently deployed release creates
    therefore persists this shape, and a cleanup record whose owner cannot be resolved leaves a
    live RunPod endpoint billing with nothing able to tear it down.
    """
    from flash.providers.runpod.client import api

    deployed_shape = api.key_fingerprint("some-pool-key")[:16]
    assert api._is_prefix_key_fingerprint(deployed_shape)
    assert not api._is_valid_key_fingerprint(deployed_shape)


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
    """A duplicated credential is a config typo, not an ownership ambiguity.

    ``RUNPOD_API_KEY`` is a comma-separated pool, so the same key pasted twice yields two
    exact fingerprint matches. Ownership is still unambiguous (equal sha256 => equal key),
    and refusing to resolve it would break submit/poll/cancel/delete for that account.
    """
    from flash.providers.runpod.client import api

    _reset_pool(monkeypatch, "secretA,secretB,secretA")

    assert api._key_for_fingerprint(api.key_fingerprint("secretA")) == "secretA"
    assert api._key_for_fingerprint(api.key_fingerprint("secretB")) == "secretB"


def test_prefix_fingerprint_owner_listing_deletes_through_authenticated_path(monkeypatch):
    from flash.providers.runpod.client import api
    from flash.runner.supervise.recovery import _delete_runpod_endpoint

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api,
        "list_endpoints_by_key",
        lambda **_kwargs: pytest.fail(
            "the persisted prefix must resolve before inventory fallback"
        ),
    )
    calls = []

    def request(request_key, url, **kwargs):
        calls.append((request_key, url, kwargs.get("method", "GET")))
        if url.endswith("/endpoints"):
            return [{"id": "ep-legacy"}]
        return {}

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", request)

    _delete_runpod_endpoint({"endpoint_id": "ep-legacy", "key_fingerprint": full_fingerprint[:16]})

    assert calls == [
        (key, f"{api.REST_BASE}/endpoints", "GET"),
        (key, f"{api.REST_BASE}/endpoints/ep-legacy", "DELETE"),
    ]


def test_prefix_fingerprint_already_gone_retires_cleanup_record(monkeypatch):
    from flash.providers.runpod.client import api
    from flash.runner.supervise.recovery import _delete_runpod_endpoint

    key = "legacy-owner"
    other_key = "other-account"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key, other_key])
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    def gone_everywhere(_request_key, url, **_kwargs):
        if url.endswith("/endpoints"):
            return []
        raise api.RunpodApiError("not found") from urllib.error.HTTPError(
            url, 404, "not found", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", gone_everywhere)

    _delete_runpod_endpoint({"endpoint_id": "ep-legacy", "key_fingerprint": full_fingerprint[:16]})


def test_prefix_fingerprint_rejects_ambiguous_prefix_owners(monkeypatch):
    from flash.providers.runpod.client import api

    keys = ["owner-a", "owner-b"]
    fingerprints = {
        "owner-a": "rpk-" + "a" * 12 + "1" * 52,
        "owner-b": "rpk-" + "a" * 12 + "2" * 52,
    }
    monkeypatch.setattr(api._keys, "keys", lambda: keys)
    monkeypatch.setattr(api, "key_fingerprint", fingerprints.__getitem__)
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: pytest.fail("ambiguous ownership must not query an account"),
    )

    with pytest.raises(api.RunpodApiError, match="expected exactly one"):
        api.resolve_prefix_key_fingerprint("ep-legacy", "rpk-" + "a" * 12)


def test_prefix_fingerprint_rejects_endpoint_live_under_other_account(monkeypatch):
    from flash.providers.runpod.client import api

    key = "legacy-owner"
    other_key = "other-account"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key, other_key])
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    def alive_under_other_account(request_key, url, **_kwargs):
        if url.endswith("/endpoints"):
            return [] if request_key == key else [{"id": "ep-legacy"}]
        raise api.RunpodApiError("not found") from urllib.error.HTTPError(
            url, 404, "not found", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", alive_under_other_account)

    with pytest.raises(api.RunpodApiError, match="not owned by the fingerprint prefix match"):
        api.resolve_prefix_key_fingerprint("ep-legacy", full_fingerprint[:16])


def test_rotated_sole_replacement_with_same_48_bit_prefix_cannot_confirm_absence(monkeypatch):
    from flash.providers.runpod.client import api

    digests = {
        b"secretA": "a" * 12 + "1" * 52,
        b"secretB": "a" * 12 + "2" * 52,
    }

    class Digest:
        def __init__(self, value):
            self.value = value

        def hexdigest(self):
            return self.value

    monkeypatch.setattr(api.hashlib, "sha256", lambda value: Digest(digests[value]))
    owner = api.key_fingerprint("secretA")
    monkeypatch.setattr(api._keys, "keys", lambda: ["secretB"])

    def wrong_account_404(*_args, **_kwargs):
        error = urllib.error.HTTPError(
            "https://rest.runpod.io/v1/endpoints/ep-owner-a",
            404,
            "not found",
            {},
            io.BytesIO(b""),
        )
        raise api.RunpodApiError("not found") from error

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", wrong_account_404)

    with pytest.raises(api.RunpodApiError, match="exactly one"):
        api.endpoint_absent_for_fingerprint("ep-owner-a", owner)


def test_strict_handle_rejects_truncated_owner_identity(monkeypatch):
    from flash.providers.runpod.client import api
    from flash.providers.runpod.execution.jobs import JobHandle

    monkeypatch.setattr(api._keys, "keys", list)
    payload = {
        "provider": "runpod",
        "endpoint_id": "ep-1",
        "endpoint_name": "flash-test",
        "key_fingerprint": "rpk-" + "a" * 12,
        "attempt": 0,
        "started_ts": 1.0,
    }
    with pytest.raises(ValueError, match="key fingerprint is invalid"):
        JobHandle.from_dict(payload)

    payload["key_fingerprint"] = "rpk-" + "a" * 64
    assert JobHandle.from_dict(payload).key_fingerprint == payload["key_fingerprint"]


def test_list_endpoints_by_key_returns_fingerprints_not_raw_keys(monkeypatch):
    from flash.providers.runpod.client import api

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
    from flash.providers.runpod.client import api

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
        api.delete_endpoint_for_fingerprint("ep-1", "rpk-" + "0" * 64)  # no pool key matches


@pytest.mark.parametrize(
    "job_id",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty-string"),
        pytest.param({"nested": "private"}, id="dict"),
        pytest.param(["private"], id="list"),
        pytest.param(b"job-private-bytes", id="bytes"),
        pytest.param(True, id="bool"),
        pytest.param(7, id="integer"),
        pytest.param(7.5, id="float"),
    ],
)
def test_submit_job_rejects_invalid_id_without_exposing_provider_response(monkeypatch, job_id):
    from flash.providers.runpod.client import api

    key = "secretA"
    private_canary = "provider-private-response-canary"
    response = {"private": private_canary}
    if job_id is not None:
        response["id"] = job_id
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(
        api.RunpodApiError, match="submit_job: response did not contain a valid job id"
    ) as exc_info:
        runpod_api.submit_job(
            "ep-1",
            {"x": 1},
            key_fingerprint=api.key_fingerprint(key),
            deadline_at=4_000_000_000.0,
        )

    assert str(exc_info.value) == "submit_job: response did not contain a valid job id"
    assert private_canary not in str(exc_info.value)


def test_submit_job_returns_nonempty_string_id(monkeypatch):
    from flash.providers.runpod.client import api

    key = "secretA"
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: {"id": "job-1"},
    )

    assert (
        runpod_api.submit_job(
            "ep-1",
            {"x": 1},
            key_fingerprint=api.key_fingerprint(key),
            deadline_at=4_000_000_000.0,
        )
        == "job-1"
    )


def test_submit_status_cancel_and_delete_keep_owning_key_after_rotation(monkeypatch):
    from flash.providers.runpod.client import api, auth

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
        runpod_api.submit_job(
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
