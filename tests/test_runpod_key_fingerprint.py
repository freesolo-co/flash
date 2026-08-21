"""list_endpoints_by_key identifies pool accounts by a NON-secret fingerprint — the raw
RUNPOD_API_KEY never appears in its return value (a stray log of a ``{key: ...}`` dict or a
``failed`` list would otherwise leak a live credential). The reaper passes the fingerprint back to
``*_for_fingerprint`` helpers, which resolve it to the real key INSIDE the api module.

CPU-only; the per-key REST call is mocked.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from flash.providers.runpod.api import list_endpoints as _real_list_endpoints


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
    assert len(fp) == 68
    assert api.key_fingerprint("a-different-key") != fp  # distinguishes accounts


def test_key_lookup_rejects_unknown_fingerprint_without_leaking_credentials(monkeypatch):
    from flash.providers.runpod import api

    keys = ["secretA", "secretB"]
    monkeypatch.setattr(api._keys, "keys", lambda: keys)

    with pytest.raises(api.RunpodApiError, match="exactly one") as exc_info:
        api._key_for_fingerprint("rpk-" + "0" * 64)

    assert all(key not in str(exc_info.value) for key in keys)


def test_key_lookup_rejects_colliding_configured_fingerprints(monkeypatch):
    from flash.providers.runpod import api

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
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA,secretB,secretA")

    assert api._key_for_fingerprint(api.key_fingerprint("secretA")) == "secretA"
    assert api._key_for_fingerprint(api.key_fingerprint("secretB")) == "secretB"


def test_repeated_identical_pool_key_still_resolves_a_legacy_prefix(monkeypatch):
    """The legacy-prefix resolver counts distinct credentials for the same reason."""
    from flash.providers.runpod import api

    key = "legacy-owner"
    _reset_pool(monkeypatch, f"{key},{key}")
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: [{"id": "ep-legacy"}],
    )

    full_fingerprint = api.key_fingerprint(key)
    resolved = api.resolve_legacy_key_fingerprint("ep-legacy", full_fingerprint[:16])

    assert resolved == full_fingerprint


def test_rotated_sole_replacement_with_same_48_bit_prefix_cannot_confirm_absence(monkeypatch):
    from flash.providers.runpod import api

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
    from flash.providers.runpod import api
    from flash.providers.runpod.jobs import JobHandle

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


def test_legacy_handle_with_confirmed_owner_supports_poll_cancel_and_destroy(monkeypatch):
    import types

    from flash.providers import base
    from flash.providers.runpod import RunpodProvider, api, jobs

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    legacy_fingerprint = full_fingerprint[:16]
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: [{"id": "ep-legacy"}],
    )
    polled = []
    monkeypatch.setattr(
        jobs,
        "poll_job",
        lambda handle, **_kwargs: polled.append(handle) or base.PollResult(ok=True),
    )
    cancelled = []
    monkeypatch.setattr(
        api,
        "cancel_job",
        lambda endpoint_id, job_id, *, key_fingerprint: (
            cancelled.append((endpoint_id, job_id, key_fingerprint))
            or {"id": job_id, "status": "CANCELLED"}
        ),
    )
    destroyed = []
    monkeypatch.setattr(
        api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, fingerprint: destroyed.append((endpoint_id, fingerprint)) or True,
    )
    handle = base.JobHandle.from_dict(
        {
            "provider": "runpod",
            "endpoint_id": "ep-legacy",
            "endpoint_name": "flash-legacy",
            "key_fingerprint": legacy_fingerprint,
            "job_id": "job-legacy",
            "attempt": 0,
            "started_ts": 1.0,
        }
    )
    spec = types.SimpleNamespace(
        phase="sft",
        run_id="missing-persisted-run",
        seed=0,
        train=types.SimpleNamespace(hf_repo=None),
    )
    provider = RunpodProvider()

    assert provider.poll(handle, spec, 0).ok is True
    provider.cancel(handle)
    provider.destroy(handle)

    assert polled[0].key_fingerprint == full_fingerprint
    assert cancelled == [("ep-legacy", "job-legacy", full_fingerprint)]
    assert destroyed == [("ep-legacy", full_fingerprint)]


def test_legacy_fingerprint_rejects_zero_prefix_matches(monkeypatch):
    from flash.providers.runpod import api

    monkeypatch.setattr(api._keys, "keys", lambda: ["other-key"])

    with pytest.raises(api.RunpodApiError, match="exactly one"):
        api.resolve_legacy_key_fingerprint("ep-legacy", "rpk-" + "0" * 12)


def test_legacy_fingerprint_rejects_multiple_prefix_matches(monkeypatch):
    from flash.providers.runpod import api

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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not list")),
    )

    with pytest.raises(api.RunpodApiError, match="exactly one"):
        api.resolve_legacy_key_fingerprint("ep-legacy", "rpk-" + "a" * 12)


def test_legacy_fingerprint_upgrades_an_already_deleted_endpoint(monkeypatch):
    """A cleanup record whose endpoint is already gone must still migrate.

    A process that dies between deleting the endpoint and clearing its durable cleanup record
    leaves the endpoint absent from every account's listing. Refusing the upgrade there stranded
    the record forever, because `_drain_cleanup_remotes` never reached the authenticated absence
    check.
    """
    from flash.providers.runpod import api

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key, "other-account"])

    def gone_everywhere(_key, url, **_kwargs):
        if url.endswith("/endpoints"):
            return []
        raise api.RunpodApiError("not found") from urllib.error.HTTPError(
            url, 404, "not found", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", gone_everywhere)
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    assert (
        api.resolve_legacy_key_fingerprint("ep-legacy", full_fingerprint[:16]) == full_fingerprint
    )


def test_legacy_fingerprint_refuses_an_endpoint_alive_under_another_pool_account(monkeypatch):
    """A 404 under the matching key is invisibility, not deletion.

    RunPod answers 404 both for an endpoint that no longer exists and for one that exists under a
    different account, so the matching key's own lookup cannot separate them. Upgrading here would
    bind the record to the wrong credential, and teardown under it would read 404, report success,
    and leave the real endpoint billing. Only the pool-wide view settles this.
    """
    from flash.providers.runpod import api

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key, "other-account"])

    def alive_only_under_the_other_account(request_key, url, **_kwargs):
        if url.endswith("/endpoints"):
            return [] if request_key == key else [{"id": "ep-legacy"}]
        # the matching key's own detail lookup 404s, because the endpoint really is invisible to
        # THIS credential. that 404 alone would read as "deleted", so only the pool-wide listing
        # can reject here -- which is exactly the signal under test.
        raise api.RunpodApiError("not found") from urllib.error.HTTPError(
            url, 404, "not found", {}, io.BytesIO(b"")
        )

    monkeypatch.setattr(
        api._CLIENT, "request_with_retries_for_key", alive_only_under_the_other_account
    )
    # the `_offline` fixture stubs `list_endpoints` to an empty fleet so no test reaches the real
    # API; restore the genuine aggregator here, since the pool-wide view is the behaviour under test.
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(api.RunpodApiError, match="not owned"):
        api.resolve_legacy_key_fingerprint("ep-legacy", full_fingerprint[:16])


def test_legacy_fingerprint_refuses_when_absence_cannot_be_confirmed(monkeypatch):
    """A partial fleet view is not proof of deletion, so the upgrade must still refuse."""
    from flash.providers.runpod import api

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    monkeypatch.setattr(api._keys, "keys", lambda: [key, "other-account"])

    def the_other_account_is_unreachable(request_key, _url, **_kwargs):
        if request_key == key:
            return []
        raise api.RunpodApiError("boom")

    monkeypatch.setattr(
        api._CLIENT, "request_with_retries_for_key", the_other_account_is_unreachable
    )
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(api.RunpodApiError, match="owner unconfirmed"):
        api.resolve_legacy_key_fingerprint("ep-legacy", full_fingerprint[:16])


def test_rotated_sole_legacy_prefix_match_cannot_claim_endpoint(monkeypatch):
    from flash.providers.runpod import api

    key = "rotated-key"
    full_fingerprint = "rpk-" + "a" * 12 + "2" * 52
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(api, "key_fingerprint", lambda _key: full_fingerprint)
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: [{"id": "ep-rotated-owner"}],
    )
    monkeypatch.setattr(api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(api.RunpodApiError, match="not owned"):
        api.resolve_legacy_key_fingerprint("ep-original-owner", "rpk-" + "a" * 12)


def test_legacy_fingerprint_is_persisted_for_active_and_cleanup_handles(tmp_path, monkeypatch):
    import flash.runner as runner
    from flash.core.spec import JobSpec
    from flash.providers.runpod import api, jobs

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    legacy_fingerprint = full_fingerprint[:16]
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep-legacy",
        "endpoint_name": "flash-legacy",
        "key_fingerprint": legacy_fingerprint,
        "job_id": "job-legacy",
        "attempt": 0,
        "started_ts": 1.0,
    }
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path))
    runner._save_status(
        runner.RunStatus(
            run_id="legacy-persisted",
            state="running",
            spec=JobSpec(
                run_id="legacy-persisted",
                model="Qwen/Qwen3.5-4B",
                algorithm="sft",
            ).to_dict(),
            remote=dict(remote),
        ),
        _cleanup_remotes=[dict(remote)],
    )
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    listings = []

    def list_owned(*_args, **_kwargs):
        listings.append("listed")
        return [{"id": "ep-legacy"}]

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", list_owned)

    jobs.migrate_persisted_legacy_key_fingerprints("legacy-persisted")
    jobs.migrate_persisted_legacy_key_fingerprints("legacy-persisted")

    with open(runner.runs_file_path("legacy-persisted", ".json"), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["remote"]["key_fingerprint"] == full_fingerprint
    assert stored["cleanup_remotes"][0]["key_fingerprint"] == full_fingerprint
    assert listings == ["listed"]


def test_unresolvable_cleanup_record_still_migrates_the_resolvable_ones(tmp_path, monkeypatch):
    """One unverifiable historical record must not strand the rest of the teardown.

    Cancellation migrates fingerprints BEFORE it reads the current remote, so an all-or-nothing
    migration meant a single stale cleanup entry -- a credential that left the pool, or an
    ownership lookup that is merely unreachable -- aborted the whole cancellation and left the
    active worker and every sibling resource billing. The resolvable records must still be
    upgraded, and the unverifiable one must still be reported rather than silently skipped.
    """
    import flash.runner as runner
    from flash.core.spec import JobSpec
    from flash.providers.runpod import api, jobs

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)

    def record(endpoint_id, fingerprint):
        return {
            "provider": "runpod",
            "endpoint_id": endpoint_id,
            "endpoint_name": f"flash-{endpoint_id}",
            "key_fingerprint": fingerprint,
            "job_id": None,
            "attempt": 0,
            "started_ts": 1.0,
        }

    # the active remote resolves; the stale cleanup record's credential is no longer in the pool.
    stranger_fingerprint = "rpk-" + "b" * 12
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path))
    runner._save_status(
        runner.RunStatus(
            run_id="partial-migration",
            state="running",
            spec=JobSpec(
                run_id="partial-migration",
                model="Qwen/Qwen3.5-4B",
                algorithm="sft",
            ).to_dict(),
            remote=record("ep-live", full_fingerprint[:16]),
        ),
        _cleanup_remotes=[record("ep-stale", stranger_fingerprint)],
    )
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT, "request_with_retries_for_key", lambda *_a, **_k: [{"id": "ep-live"}]
    )

    with pytest.raises(ValueError, match="persisted RunPod key fingerprint is invalid"):
        jobs.migrate_persisted_legacy_key_fingerprints("partial-migration")

    with open(runner.runs_file_path("partial-migration", ".json"), encoding="utf-8") as handle:
        stored = json.load(handle)
    # the resolvable record was upgraded despite the unresolvable sibling raising afterwards.
    assert stored["remote"]["key_fingerprint"] == full_fingerprint
    assert stored["cleanup_remotes"][0]["key_fingerprint"] == stranger_fingerprint


def test_confirmed_legacy_handle_cancel_reaches_teardown(tmp_path, monkeypatch):
    import flash.runner as runner
    from flash.core.spec import JobSpec
    from flash.providers.runpod import api
    from flash.runner.supervise import lifecycle
    from tests._helpers.runner import provisioned_status

    key = "legacy-owner"
    full_fingerprint = api.key_fingerprint(key)
    legacy_fingerprint = full_fingerprint[:16]
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep-legacy",
        "endpoint_name": "flash-legacy",
        "key_fingerprint": legacy_fingerprint,
        "job_id": "job-legacy",
        "attempt": 0,
        "started_ts": 1.0,
    }
    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX 5090"},
            "run_id": "legacy-cancel",
        }
    )
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path))
    runner._save_status(provisioned_status(runner, spec, state="running", remote=remote))
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: [{"id": "ep-legacy"}],
    )
    teardown = []
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda handle, run_id: teardown.append((handle.to_dict(), run_id)) or True,
    )
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)

    result = runner.cancel_run(spec.run_id)

    assert result.state == "cancelled"
    assert result.remote is None
    assert teardown[0][0]["key_fingerprint"] == full_fingerprint
    assert teardown[0][1] == spec.run_id


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
    from flash.providers.runpod import api

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
        api.submit_job(
            "ep-1",
            {"x": 1},
            key_fingerprint=api.key_fingerprint(key),
            deadline_at=4_000_000_000.0,
        )

    assert str(exc_info.value) == "submit_job: response did not contain a valid job id"
    assert private_canary not in str(exc_info.value)


def test_submit_job_returns_nonempty_string_id(monkeypatch):
    from flash.providers.runpod import api

    key = "secretA"
    monkeypatch.setattr(api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: {"id": "job-1"},
    )

    assert (
        api.submit_job(
            "ep-1",
            {"x": 1},
            key_fingerprint=api.key_fingerprint(key),
            deadline_at=4_000_000_000.0,
        )
        == "job-1"
    )


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
