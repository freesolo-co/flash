"""Control-plane API: freesolo bearer auth, multi-tenant isolation (CPU-only).

User auth is freesolo API keys only (no native key system). Tests run offline: the `api`
fixture monkeypatches ``auth._freesolo_verify`` to accept any token shaped like a freesolo
user key, so each distinct token resolves to its own run-ownership identity via
``db.ensure_external_key``. All runs are dry-run so nothing touches the network; operator
env vars are dummies (the startup preflight only checks presence).
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import sqlite3
import time

import pytest

from flash import runner as _orch
from flash.server import db as _db_mod

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SPEC = {
    "model": "Qwen/Qwen3.5-4B",
    "algorithm": "grpo",
    "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
    "train": {"steps": 1, "hf_repo": "org/test-runs"},
    "gpu": {"type": "RTX 5090"},
}

# Tokens shaped like a verified freesolo user key. The fixture's stub verify accepts any
# token with this prefix, so each distinct one is a distinct authenticated user.
_USER_PREFIX = "fslo-user-"
_counter = itertools.count()


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _login() -> str:
    """A fresh, distinct freesolo user token (accepted by the fixture's stub verify)."""
    return f"{_USER_PREFIX}{next(_counter)}"


def _identity_for_token(token: str) -> dict[str, str]:
    if not token.startswith(_USER_PREFIX):
        return {}
    suffix = token.removeprefix(_USER_PREFIX)
    return {
        "email": f"user-{suffix}@example.com",
        "key_prefix": "fslo_test",
        "org_id": f"org-{suffix}",
        "org_slug": f"org-{suffix}",
    }


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Full operator config so the app's startup preflight passes (>= 2 RunPod accounts + Lambda +
    # the shared tokens + the internal key); see tests/test_preflight.py for the gate.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    # runpod.keys caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.keys as runpod_keys

    runpod_keys.reset()
    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    # The storage roots are fixed constants (not env-configurable); redirect them to tmp for
    # test isolation by patching the module attributes after reload.
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)
    # The new preflight requires the Lambda key above, which also makes
    # `configured_providers()` treat it as live — so the startup lifespan's `recover_runs()` and
    # the orphan-sweep loop would dispatch real `sweep_orphans()` (Lambda list calls) and
    # break test hermeticity. These API tests don't exercise orphan reaping, so stub the provider set
    # to empty: preflight still passes on the keys, but startup stays CPU-only with no network. (Both
    # call sites do a function-local `from flash.providers import configured_providers`, so patching
    # the package attribute covers them.)
    import flash.providers as providers_mod
    import flash.providers.runpod.train.endpoints as rp_endpoints
    import flash.server.run_registry as run_registry

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [], raising=False)
    # The dummy FREESOLO_INTERNAL_KEY also enables the best-effort backend reporting path: a dry-run
    # /v1/runs submit carries an org_id, so runner.submit_job() -> _report_status() ->
    # run_registry._post() would urllib-POST the real backend (or wait out its 10s timeout). Stub the
    # single network choke-point so these offline tests stay hermetic (same as the billing fixture).
    monkeypatch.setattr(run_registry, "_post", lambda *a, **k: False, raising=False)
    # ...and that same key makes create_app() startup run the RunPod slot-store reconcile
    # (reconcile_endpoint_slots() -> runpod.slots.reconcile() urllib POST). No-op it at the entry.
    monkeypatch.setattr(
        rp_endpoints, "reconcile_endpoint_slots", lambda *a, **k: None, raising=False
    )
    # Offline auth: a token is a valid freesolo USER key iff it has the test prefix. This stub
    # replaces the real network verify.
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    monkeypatch.setattr(auth_mod, "_cached_identity", _identity_for_token)
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_me(api):
    key = _login()
    me = api.get("/v1/me", headers=_bearer(key))
    assert me.status_code == 200
    # A verified freesolo user key resolves to the Freesolo identity returned by verify.
    assert me.json()["email"] == f"user-{key.removeprefix(_USER_PREFIX)}@example.com"
    assert me.json()["key_prefix"] == "fslo_test"
    assert me.json()["org_slug"] == f"org-{key.removeprefix(_USER_PREFIX)}"


def test_requests_without_key_are_rejected(api):
    assert api.get("/v1/runs").status_code == 401
    # A token that doesn't verify with freesolo is rejected.
    assert api.get("/v1/runs", headers=_bearer("not-a-freesolo-key")).status_code == 401
    assert api.get("/v1/models", headers=_bearer("nope")).status_code == 401
    assert api.get("/v1/health").status_code == 200  # health stays open


def test_internal_key_authenticates_as_service_identity(api, monkeypatch):
    # With FREESOLO_INTERNAL_KEY configured, the shared internal key works as a bearer and
    # owns the runs it submits — the freesolo SDK authenticates with the same credential the
    # platform uses. It is matched BEFORE the freesolo user-key verify, so it never hits the
    # backend.
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-secret")
    r = api.post(
        "/v1/runs",
        json={"spec": SPEC, "dry_run": True},
        headers=_bearer("fslo-internal-secret"),
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    # owns its run (run_owner resolves to the provisioned service identity)
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer("fslo-internal-secret")).status_code == 200
    # a token that is neither the internal key nor a verified freesolo key is rejected
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer("wrong-internal")).status_code == 401
    # the internal key is stored hashed, like any other (never persisted in the clear)
    with sqlite3.connect(_db_mod.db_path()) as conn:
        prefixes = [row[0] for row in conn.execute("SELECT key_prefix FROM api_keys").fetchall()]
    assert "internal" in prefixes


def test_internal_key_rejected_when_unconfigured(api):
    # Without FREESOLO_INTERNAL_KEY set, the would-be internal key is just an unknown token
    # that doesn't verify with freesolo and gets 401 — no implicit acceptance.
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-secret")).status_code == 401


def test_freesolo_user_key_authenticates(api, monkeypatch):
    # A user who `flash login`s with a freesolo key sends it as the bearer. With the token
    # verified by the backend it authenticates and resolves to a stable per-token identity
    # (its own run-ownership row).
    import flash.server.auth as auth_mod

    auth_mod._verify_cache.clear()
    calls = {"n": 0}

    def fake_verify(token):
        calls["n"] += 1
        return token == "fslo-user-good"

    monkeypatch.setattr(auth_mod, "_freesolo_verify", fake_verify)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: (
            {"email": "user-good@example.com", "key_prefix": "fslo_good", "org_slug": "acme"}
            if token == "fslo-user-good"
            else {}
        ),
    )

    row = auth_mod.authenticate("Bearer fslo-user-good")
    assert row is not None
    assert row["email"] == "user-good@example.com"
    # An unverified token returns None (401).
    assert auth_mod.authenticate("Bearer fslo-user-bad") is None
    # The same key resolves to the same identity across requests (stable per-token row).
    again = auth_mod.authenticate("Bearer fslo-user-good")
    assert again["id"] == row["id"]


def test_freesolo_user_key_without_org_slug_is_rejected(api, monkeypatch):
    # A verified external key must include an org slug. Do not fall back to email or
    # token-derived namespaces for env publishing.
    import flash.server.auth as auth_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {"email": "user@example.com", "key_prefix": "fslo_noorg"},
    )

    assert auth_mod.authenticate("Bearer fslo-no-org") is None


def test_freesolo_user_key_without_email_authenticates_with_org_slug(api, monkeypatch):
    import flash.server.auth as auth_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {"key_prefix": "fslo_noemail", "org_slug": "acme", "org_id": "org-acme"},
    )

    row = auth_mod.authenticate("Bearer fslo-no-email")
    assert row is not None
    assert row["org_slug"] == "acme"
    assert not row.get("email")


def test_freesolo_user_key_disabled_is_401_not_500(api, monkeypatch):
    # A freesolo key that verifies with the backend but whose db row was disabled (revoked)
    # must be rejected as 401 (authenticate -> None), not raise a 500.
    import sqlite3

    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)
    monkeypatch.setattr(
        auth_mod,
        "_cached_identity",
        lambda token: {
            "email": "revoked@example.com",
            "key_prefix": "fslo_revoked",
            "org_slug": "acme",
        },
    )

    assert auth_mod.authenticate("Bearer fslo-revoked") is not None  # provisioned on first use
    with sqlite3.connect(db_mod.db_path()) as conn:
        conn.execute(
            "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?",
            (db_mod.hash_key("fslo-revoked"),),
        )
    # Verified by freesolo but disabled in the db -> None (401), never a raised 500.
    assert auth_mod.authenticate("Bearer fslo-revoked") is None


def test_freesolo_verify_does_not_cache_network_errors(monkeypatch):
    # A transient network error must NOT be cached as a rejection, or a valid key would be
    # locked out for the whole TTL. The next call (backend recovered) must succeed.
    import urllib.error

    import flash.server.auth as auth_mod

    # Use the real _freesolo_verify (not the fixture stub) and let it touch the (patched) net.
    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"fail": True}

    def flaky(req, timeout=None):
        if state["fail"]:
            raise urllib.error.URLError("connection timed out")
        return _Resp()

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", flaky)
    assert auth_mod._freesolo_verify("tok") is False  # transient failure
    assert "tok" not in auth_mod._verify_cache  # NOT cached
    state["fail"] = False
    assert auth_mod._freesolo_verify("tok") is True  # recovers immediately


def test_freesolo_verify_5xx_transient_but_4xx_cached(monkeypatch):
    # A backend 5xx/429 is a transient hiccup (urllib raises HTTPError for these too): it must
    # NOT be cached, so a valid key recovers immediately. A definitive 4xx (401/403) IS cached.
    import urllib.error

    import flash.server.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"code": 503}

    def responder(req, timeout=None):
        code = state["code"]
        if code == 200:
            return _Resp()
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", responder)
    # 5xx -> transient, not cached
    assert auth_mod._freesolo_verify("tok") is False
    assert "tok" not in auth_mod._verify_cache
    # 429 -> transient, not cached
    state["code"] = 429
    assert auth_mod._freesolo_verify("tok") is False
    assert "tok" not in auth_mod._verify_cache
    # backend recovers -> immediately verified (no stale negative cached)
    state["code"] = 200
    assert auth_mod._freesolo_verify("tok") is True

    # a definitive 401 IS cached as a rejection (no repeated backend round-trips)
    auth_mod._verify_cache.clear()
    state["code"] = 401
    assert auth_mod._freesolo_verify("bad") is False
    assert auth_mod._verify_cache.get("bad", (None,))[0] is False


def test_freesolo_verify_negative_short_ttl_positive_long_ttl(monkeypatch):
    # A negative verdict (a 401 may be a TRANSIENT backend auth-lookup outage, not a real
    # rejection) gets the short negative TTL so a valid key isn't locked out for 5 minutes;
    # a positive keeps the long TTL.
    import urllib.error

    import flash.server.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    state = {"code": 401}

    def responder(req, timeout=None):
        code = state["code"]
        if code == 200:
            return _Resp()
        raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", responder)

    # Negative (401) -> cached with the SHORT negative TTL.
    now = time.time()
    assert auth_mod._freesolo_verify("neg") is False
    neg_exp = auth_mod._verify_cache["neg"][2]
    assert neg_exp <= now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    # ...and definitely shorter than the long TTL.
    assert neg_exp < now + auth_mod._VERIFY_CACHE_TTL_S

    # Positive -> cached with the LONG TTL.
    auth_mod._verify_cache.clear()
    state["code"] = 200
    now = time.time()
    assert auth_mod._freesolo_verify("pos") is True
    pos_exp = auth_mod._verify_cache["pos"][2]
    assert pos_exp > now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    assert pos_exp <= now + auth_mod._VERIFY_CACHE_TTL_S + 1

    # The negative entry expires after the short TTL: simulate the clock advancing past
    # _VERIFY_CACHE_NEG_TTL_S (but not the long TTL) and confirm the negative is treated as
    # expired while a same-age positive would still be live.
    auth_mod._verify_cache.clear()
    base = time.time()
    auth_mod._verify_cache["neg"] = (False, {}, base + auth_mod._VERIFY_CACHE_NEG_TTL_S)
    auth_mod._verify_cache["pos"] = (True, {}, base + auth_mod._VERIFY_CACHE_TTL_S)
    later = base + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1.0  # past neg TTL, well under pos TTL
    assert auth_mod._verify_cache["neg"][2] <= later  # negative entry has expired
    assert auth_mod._verify_cache["pos"][2] > later  # positive entry is still live


def test_freesolo_verify_rejects_oversized_token(monkeypatch):
    # An oversized bearer must be rejected before it touches the cache or the network, so it
    # can't bloat _verify_cache (keyed by the raw token) or send a huge Authorization header.
    import flash.server.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    def boom(*a, **k):
        raise AssertionError("oversized token must not reach the network")

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", boom)
    huge = "x" * (auth_mod._MAX_TOKEN_LEN + 1)
    assert auth_mod._freesolo_verify(huge) is False
    assert huge not in auth_mod._verify_cache


def test_freesolo_user_key_unverified_when_backend_unreachable(api, monkeypatch):
    # When the backend verify can't be reached (the offline test harness makes urlopen fail),
    # _freesolo_verify returns False and authenticate yields None — an unverifiable key is
    # never admitted.
    import urllib.error

    import flash.server.auth as auth_mod

    auth_mod._verify_cache.clear()
    # Drop the fixture's stub so the real _freesolo_verify runs, and make the backend
    # unreachable (offline): the verify can't be reached.
    importlib.reload(auth_mod)
    monkeypatch.setattr(
        auth_mod.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    assert auth_mod.authenticate("Bearer unknown-token") is None


def test_freesolo_verify_cache_prevents_second_call(monkeypatch):
    # The in-process cache means a second authenticate for the same token doesn't re-hit the
    # backend within the TTL (positives and negatives are both cached).
    import flash.server.auth as auth_mod

    # Use the real _freesolo_verify (not the fixture stub) and let it touch the (patched) net.
    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1

        class _Resp:
            status = 200

            def read(self):
                return (
                    b'{"email":"cached@example.com","key_prefix":"fslo_cached","org_slug":"acme"}'
                )

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", fake_urlopen)

    first = auth_mod.authenticate("Bearer fslo-cached")
    second = auth_mod.authenticate("Bearer fslo-cached")
    assert first is not None
    assert second is not None
    assert calls["n"] == 1  # second authenticate served from cache, no second backend call


def test_freesolo_verify_cache_is_bounded_and_prunes_expired(monkeypatch):
    # The verify cache keys by the raw bearer token, so a stream of distinct tokens could
    # grow it without bound. Each write prunes expired entries and caps the cache size.
    import time

    import flash.server.auth as auth_mod

    importlib.reload(auth_mod)
    auth_mod._verify_cache.clear()

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", lambda req, timeout=None: _Resp())

    # An already-expired entry must be removed on the next write (no longer reachable).
    auth_mod._verify_cache["stale"] = (True, {}, time.time() - 1)
    auth_mod._freesolo_verify("fresh-token")
    assert "stale" not in auth_mod._verify_cache
    assert "fresh-token" in auth_mod._verify_cache

    # Verifying many distinct (live) tokens never grows the cache past the cap.
    monkeypatch.setattr(auth_mod, "_VERIFY_CACHE_MAX", 8)
    auth_mod._verify_cache.clear()
    for i in range(50):
        auth_mod._freesolo_verify(f"tok-{i}")
        assert len(auth_mod._verify_cache) <= auth_mod._VERIFY_CACHE_MAX
    assert len(auth_mod._verify_cache) <= auth_mod._VERIFY_CACHE_MAX
    auth_mod._verify_cache.clear()


def test_keys_are_hashed_at_rest(api):
    key = _login()
    # Authenticate once so the key's row is provisioned.
    assert api.get("/v1/me", headers=_bearer(key)).status_code == 200
    with sqlite3.connect(_db_mod.db_path()) as conn:
        rows = conn.execute("SELECT key_hash, key_prefix FROM api_keys").fetchall()
    assert rows
    for key_hash, _prefix in rows:
        assert key_hash != key
        assert len(key_hash) == 64  # sha256 hex
    with open(_db_mod.db_path(), "rb") as f:
        raw = f.read()
    assert key.encode() not in raw


def test_run_lifecycle_and_tenant_isolation(api):
    key_a, key_b = _login(), _login()
    created = api.post("/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key_a))
    assert created.status_code == 200, created.text
    run_id = created.json()["run_id"]
    assert created.json()["state"] == "dry_run"

    # Owner sees it (status, list); the other tenant gets 404s and an empty list.
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key_a)).status_code == 200
    assert [r["run_id"] for r in api.get("/v1/runs", headers=_bearer(key_a)).json()["runs"]] == [
        run_id
    ]
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer(key_b)).status_code == 404
    assert api.get("/v1/runs", headers=_bearer(key_b)).json()["runs"] == []
    assert api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key_b)).status_code == 404
    assert api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(key_b)).status_code == 404


def test_runtime_secret_validation_and_non_persistence(api):
    key = _login()
    bad = api.post(
        "/v1/runs",
        json={
            "spec": SPEC,
            "dry_run": True,
            "runtime_secrets": {"RUNPOD_API_KEY": "must-stay-platform-side"},
        },
        headers=_bearer(key),
    )
    assert bad.status_code == 400
    assert "unsupported runtime secret" in bad.json()["detail"]

    created = api.post(
        "/v1/runs",
        json={
            "spec": SPEC,
            "dry_run": True,
            "runtime_secrets": {"WANDB_API_KEY": "user-wandb-key"},
        },
        headers=_bearer(key),
    )
    assert created.status_code == 200, created.text
    body = created.json()
    dumped = json.dumps(body)
    assert "user-wandb-key" not in dumped
    assert "runtime_secrets" not in dumped
    assert "WANDB_API_KEY" not in body["spec"].get("worker_env", {})

    env_secret_spec = {
        **SPEC,
        "environment": {
            **SPEC["environment"],
            "secrets": ["SERPAPI_API_KEY"],
        },
    }
    created = api.post(
        "/v1/runs",
        json={
            "spec": env_secret_spec,
            "dry_run": True,
            "runtime_secrets": {"SERPAPI_API_KEY": "serp-user-key"},
        },
        headers=_bearer(key),
    )
    assert created.status_code == 200, created.text
    body = created.json()
    dumped = json.dumps(body)
    assert "serp-user-key" not in dumped
    assert "runtime_secrets" not in dumped
    assert body["spec"]["environment"]["secrets"] == ["SERPAPI_API_KEY"]

    missing = api.post(
        "/v1/runs",
        json={"spec": env_secret_spec, "runtime_secrets": {}},
        headers=_bearer(key),
    )
    assert missing.status_code == 400
    assert "missing runtime secret" in missing.json()["detail"]


def test_logs_offset_paging(api):
    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    log_path = os.path.join(_orch.RUNS_DIR, f"{run_id}.log")
    with open(log_path, "w") as f:
        f.write("line one\n")
    page = api.get(f"/v1/runs/{run_id}/logs", headers=_bearer(key)).json()
    assert page["logs"] == "line one\n"
    assert page["state"] == "dry_run"
    with open(log_path, "a") as f:
        f.write("line two\n")
    page2 = api.get(f"/v1/runs/{run_id}/logs?offset={page['offset']}", headers=_bearer(key)).json()
    assert page2["logs"] == "line two\n"


def test_worker_output_route(api, monkeypatch):
    # /worker surfaces the train-subprocess stdout/traceback from the run's HF repo (operator
    # token, server-side). Best-effort: no artifacts -> empty dict; present -> passed through.
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    # Stub _worker_artifacts to {} BEFORE the first request: the real impl would hf_hub_download from
    # the run's HF repo (slow/flaky network in a unit test). This keeps the "no artifacts -> {}"
    # assertion fully offline/deterministic.
    monkeypatch.setattr(app_mod, "_worker_artifacts", lambda spec: {})
    empty = api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(key)).json()
    assert empty["run_id"] == run_id
    assert empty["worker"] == {}

    monkeypatch.setattr(
        app_mod, "_worker_artifacts", lambda spec: {"console_sft.txt": "real worker stdout\n"}
    )
    got = api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(key)).json()
    assert got["worker"] == {"console_sft.txt": "real worker stdout\n"}

    # Another user can't read it (same ownership gate as /logs).
    other = _login()
    assert api.get(f"/v1/runs/{run_id}/worker", headers=_bearer(other)).status_code == 404


def test_latest_error_artifact_name_picks_highest_attempt(monkeypatch):
    """The logs fetcher resolves the newest attempt-scoped error file, so a retried-then-failed run
    surfaces the FINAL attempt's traceback, not attempt0's stale one."""
    import huggingface_hub

    from flash.server._runtime import _latest_error_artifact_name

    prefix = "sft/run-1/seed0"
    listed = [
        f"{prefix}/console_sft.txt",
        f"{prefix}/error_sft_attempt0.txt",
        f"{prefix}/error_sft_attempt2.txt",
        f"{prefix}/error_sft_attempt1.txt",
        f"{prefix}/heartbeat.json",
        "other/run/error_sft_attempt9.txt",  # different prefix -> ignored
    ]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return listed

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    assert _latest_error_artifact_name("org/repo", prefix, "sft") == "error_sft_attempt2.txt"


def test_latest_error_artifact_name_defaults_when_unlistable(monkeypatch):
    """If the repo can't be listed, fall back to attempt0 rather than failing the logs fetch."""
    import huggingface_hub

    from flash.server._runtime import _latest_error_artifact_name

    class _BoomApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            raise RuntimeError("HF down")

    monkeypatch.setattr(huggingface_hub, "HfApi", _BoomApi)
    assert _latest_error_artifact_name("org/repo", "rl/r/seed0", "rl") == "error_rl_attempt0.txt"


def test_worker_artifacts_fetches_console_and_latest_attempt_error(monkeypatch, tmp_path):
    """The fetcher pulls the worker console plus the NEWEST attempt-scoped error file
    (error_<phase>_attempt<N>.txt) — on a retried run only the highest attempt is the real crash."""
    import types

    import huggingface_hub

    from flash.server._runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl.txt": "worker console\n",
        "rl/r1/error_rl_attempt0.txt": "stale first-attempt traceback\n",
        "rl/r1/error_rl_attempt1.txt": "TRACEBACK latest\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["console_rl.txt"] == "worker console\n"
    # The newest attempt's traceback surfaces; the superseded attempt0 is not fetched.
    assert out["error_rl_attempt1.txt"] == "TRACEBACK latest\n"
    assert "error_rl_attempt0.txt" not in out


def test_worker_artifacts_prefers_latest_attempt_console(monkeypatch, tmp_path):
    """When console output is attempt-scoped, /worker should show the current attempt's tail."""
    import types

    import huggingface_hub

    from flash.server._runtime import _worker_artifacts

    spec = types.SimpleNamespace(
        phase="rl",
        run_id="r1",
        train=types.SimpleNamespace(hf_repo="org/repo"),
    )
    content = {
        "rl/r1/console_rl_attempt0.txt": "stale console\n",
        "rl/r1/console_rl_attempt2.txt": "current console\n",
        "rl/r1/error_rl_attempt2.txt": "current traceback\n",
    }

    def fake_dl(repo_id, repo_type, filename, token=None, force_download=False):
        if filename not in content:
            raise FileNotFoundError(filename)
        p = tmp_path / filename.replace("/", "_")
        p.write_text(content[filename])
        return str(p)

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, repo_type):
            return list(content)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    out = _worker_artifacts(spec)
    assert out["console_rl_attempt2.txt"] == "current console\n"
    assert out["error_rl_attempt2.txt"] == "current traceback\n"
    assert "console_rl_attempt0.txt" not in out


def test_local_env_path_rejected(api):
    # Managed runs accept Freesolo environment ids; local [environment] paths are rejected.
    key = _login()
    bad = {**SPEC, "environment": {"id": "custom", "path": "/home/user/env.py"}}
    r = api.post("/v1/runs", json={"spec": bad, "dry_run": True}, headers=_bearer(key))
    assert r.status_code == 400
    assert "not supported on the managed service" in r.json()["detail"]


def test_bad_spec_is_400(api):
    key = _login()
    r = api.post("/v1/runs", json={"spec": {"algorithm": "grpo"}}, headers=_bearer(key))
    assert r.status_code == 400
    assert "model" in r.json()["detail"]


def test_non_object_spec_fields_get_targeted_400(api):
    # A falsy NON-object JSON value (""/0/[]/false) for spec / spec.environment / runtime_secrets
    # must 400 with the intended "must be a JSON object" message, not get coerced to {} (which
    # would surface a misleading downstream error like "config must set [environment] id").
    key = _login()
    for bad_spec in ("", 0, [], False):
        r = api.post("/v1/runs", json={"spec": bad_spec}, headers=_bearer(key))
        assert r.status_code == 400, (bad_spec, r.text)
        assert "spec must be a JSON object" in r.json()["detail"], (bad_spec, r.text)

    for bad_env in ("", 0, [], False):
        r = api.post(
            "/v1/runs",
            json={"spec": {**SPEC, "environment": bad_env}},
            headers=_bearer(key),
        )
        assert r.status_code == 400, (bad_env, r.text)
        assert "spec.environment must be a JSON object" in r.json()["detail"], (bad_env, r.text)

    for bad_secrets in ("", 0, [], False):
        r = api.post(
            "/v1/runs",
            json={"spec": SPEC, "dry_run": True, "runtime_secrets": bad_secrets},
            headers=_bearer(key),
        )
        assert r.status_code == 400, (bad_secrets, r.text)
        assert "runtime_secrets must be a JSON object" in r.json()["detail"], (bad_secrets, r.text)


def test_deploy_dry_run(api):
    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    dep = api.post(f"/v1/runs/{run_id}/deploy", json={"dry_run": True}, headers=_bearer(key))
    assert dep.status_code == 200, dep.text
    assert dep.json()["state"] == "dry_run"
    assert "mode" not in dep.json()
    # Dry-run deploys never show up as active deployments.
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_deploy_serving_error_is_clean_502(api, monkeypatch):
    """A serving-backend failure during deploy surfaces as a clean 502 with the upstream
    reason — NOT an unhandled 500 + traceback. This is the main user-facing behavior change
    of the route: ServingError (raised by deploy_adapter when freesolo serving rejects the
    registration or is unreachable) is translated to HTTPException(502)."""
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    # Make the run real-deployable: flip its persisted state to "done" (a finished run with
    # trained adapter artifacts). Ownership lives in the DB, so this only changes the gate.
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    # The serving backend rejects the registration (e.g. upstream 5xx). deploy_adapter is
    # imported into the app namespace, so patch it there.
    def boom(**kwargs):
        raise ServingError("serving backend unreachable: no engine for base model")

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 502, resp.text
    # The 502 carries the upstream reason verbatim, not a generic/unhandled-500 body.
    assert "serving backend unreachable" in resp.json()["detail"]
    # The failed deploy left no active deployment record behind.
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_deploy_attributes_adapter_to_run_owning_org(api, monkeypatch):
    """The adapter is registered under the RUN's owning org (its persisted billing_context) so
    serving can authorize external chat by org — not merely whatever key initiated the deploy."""
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.billing_context = {"org_id": "run-owner-org"}
    runner._save_status(status)

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return Deployment(
            run_id=run_id,
            model=kwargs["model"],
            adapter_hf_prefix="x/adapter",
            gpu="RTX 5090",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            state="ready",
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", capture)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    # The run's owning org (billing_context) is what's attributed, not the bare caller key.
    assert seen["org_id"] == "run-owner-org"


def test_deploy_falls_back_to_platform_context_org(api, monkeypatch):
    """An internal/operator deploy has no billing_context but persists the org in
    platform_context; the adapter must still be attributed to that run-owning org."""
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.billing_context = None
    status.platform_context = {"org_id": "platform-org"}
    runner._save_status(status)

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return Deployment(
            run_id=run_id,
            model=kwargs["model"],
            adapter_hf_prefix="x/adapter",
            gpu="RTX 5090",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            state="ready",
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", capture)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    assert seen["org_id"] == "platform-org"


def test_chat_streams_deployed_run(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    runner.mark_deployed(run_id, {"state": "ready", "endpoint_name": "https://serve.example"})

    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield "hi"
        yield " there"

    monkeypatch.setattr(app_mod, "serve_chat_stream", fake_stream)

    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    ) as resp:
        text = resp.read().decode()

    assert resp.status_code == 200, text
    assert text == "hi there"
    assert seen["run_id"] == run_id
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_uses_saved_thinking_flag_not_payload_override(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs",
        json={"spec": {**SPEC, "thinking": True}, "dry_run": True},
        headers=_bearer(key),
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    runner.mark_deployed(run_id, {"state": "ready", "endpoint_name": "https://serve.example"})

    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["thinking"] is True
    assert "enable_thinking" not in seen


def test_chat_serves_cancelled_run_with_active_checkpoint_deployment(api, monkeypatch):
    """A run cancelled mid-RL can deploy a per-step checkpoint (stays `cancelled`, listed active by
    /v1/deployments). The chat route must SERVE that live adapter, not 409 on the cancelled state."""
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "cancelled"  # cancelled, but with a live checkpoint deployment
    status.deployment = {"state": "ready", "endpoint_name": "https://serve.example"}
    runner._save_status(status)

    monkeypatch.setattr(app_mod, "serve_chat_stream", lambda **k: iter(["hi", " there"]))
    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        headers=_bearer(key),
    ) as resp:
        text = resp.read().decode()
    assert resp.status_code == 200, text
    assert text == "hi there"


def test_chat_cancelled_run_without_deployment_is_409(api):
    """A cancelled run with no active deployment still 409s, pointing the user at `flash deploy`."""
    import flash.runner as runner

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "cancelled"
    runner._save_status(status)

    r = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_bearer(key),
    )
    assert r.status_code == 409
    assert "deploy a checkpoint" in r.json()["detail"]


def test_chat_rejects_non_finite_sampling_params_with_400(api, monkeypatch):
    """JSON `1e400`/`Infinity` parses to float('inf'); `int(inf)` raises OverflowError (an
    ArithmeticError, NOT TypeError/ValueError) which used to escape the guard -> 500. A non-finite
    max_tokens or temperature must be a clean 400, per the route's own bad-values-are-400 contract."""
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    runner.mark_deployed(run_id, {"state": "ready", "endpoint_name": "https://serve.example"})
    monkeypatch.setattr(app_mod, "serve_chat_stream", lambda **k: iter(["hi"]))

    headers = {**_bearer(key), "content-type": "application/json"}
    for body in (
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1e400}',
        b'{"messages": [{"role": "user", "content": "hi"}], "temperature": 1e400}',
    ):
        r = api.post(f"/v1/runs/{run_id}/chat", content=body, headers=headers)
        assert r.status_code == 400, (body, r.status_code, r.text)


def test_undeploy_serving_error_is_clean_502(api, monkeypatch):
    """An undeploy that hits a serving-backend failure surfaces as a clean 502 (same as deploy),
    not an unhandled 500: ServingError from undeploy_adapter is translated to HTTPException(502)."""
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    def boom(_run_id):
        raise ServingError("serving backend unreachable: could not delete endpoint")

    monkeypatch.setattr(app_mod, "undeploy_adapter", boom)

    resp = api.delete(f"/v1/runs/{run_id}/deploy", headers=_bearer(key))
    assert resp.status_code == 502, resp.text
    assert "serving backend unreachable" in resp.json()["detail"]


def test_mark_deployed_allows_done_but_not_cancelled(monkeypatch, tmp_path):
    # A finished run (state="done") MUST be deployable: mark_deployed has to record the
    # deployment and flip to "deployed". But a cancelled/failed run must never be flipped
    # to "deployed" (a /cancel racing deployment persisted the terminal state).
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "run_id": "dep-1"}
    runner._save_status(runner.RunStatus(run_id="dep-1", state="done", spec=spec, remote=None))
    out = runner.mark_deployed("dep-1", {"endpoint_name": "e"})
    assert out.state == "deployed"
    assert out.deployment == {"endpoint_name": "e"}

    # cancelled is sticky: the deploy must be refused, state preserved.
    runner._save_status(
        runner.RunStatus(
            run_id="dep-2", state="cancelled", spec={**spec, "run_id": "dep-2"}, remote=None
        )
    )
    out2 = runner.mark_deployed("dep-2", {"endpoint_name": "e2"})
    assert out2.state == "cancelled"
    assert out2.deployment is None


def test_mark_deployed_expect_state_cas_blocks_undeploy_race(monkeypatch, tmp_path):
    # Redeploy finalization must NOT clobber an undeploy that raced in mid-warmup: the
    # undeploy wrote `done`/undeployed and deleted the endpoint, so a final mark_deployed
    # that still expects "deployed" must refuse to re-advertise the deleted endpoint.
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "run_id": "dep-3"}
    runner._save_status(
        runner.RunStatus(
            run_id="dep-3",
            state="deployed",
            spec=spec,
            remote=None,
            deployment={"endpoint_name": "e"},
        )
    )
    # undeploy races in: endpoint torn down, run back to done/undeployed.
    undone = runner.mark_undeployed("dep-3")
    assert undone.state == "done"
    assert undone.deployment["state"] == "undeployed"
    # the deploy that was warming finalizes expecting "deployed" -> refused.
    out = runner.mark_deployed("dep-3", {"endpoint_name": "e2"}, expect_state="deployed")
    assert out.state == "done"
    assert out.deployment["state"] == "undeployed"  # not re-advertised


def test_mark_checkpoint_deployed_refuses_dry_run(monkeypatch, tmp_path):
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "run_id": "dep-dry"}
    runner._save_status(runner.RunStatus(run_id="dep-dry", state="dry_run", spec=spec, remote=None))
    out = runner.mark_checkpoint_deployed("dep-dry", {"endpoint_name": "e"})
    assert out.state == "dry_run"
    assert out.deployment is None


def test_mark_deployed_legacy_finished_at_backfill_only_on_done_transition(monkeypatch, tmp_path):
    # The legacy finished_at backfill (for runs that went `done` before finished_at existed) must
    # run ONLY on the done->deployed transition, where updated_at == training teardown. On an
    # already-`deployed` run (the CAS finalization with expect_state="deployed"), updated_at is the
    # DEPLOY time, so stamping finished_at from it would reintroduce the instance over-billing this
    # whole change fixes.
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "run_id": "dep-leg"}

    # (1) done -> deployed: legacy run, finished_at=None, updated_at == teardown -> backfilled.
    teardown = 1_000.0
    runner._save_status(
        runner.RunStatus(
            run_id="dep-leg",
            state="done",
            spec=spec,
            remote=None,
            updated_at=teardown,
            finished_at=None,
        )
    )
    out = runner.mark_deployed("dep-leg", {"endpoint_name": "e"})
    assert out.state == "deployed"
    assert out.finished_at == teardown  # frozen to the real teardown time
    assert out.updated_at > teardown  # the deploy bumped updated_at past teardown

    # (2) already-`deployed` legacy run whose finished_at was never backfilled: a CAS-finalization
    # re-call must NOT turn the deploy-time updated_at into finished_at.
    deploy_time = 5_000.0
    runner._save_status(
        runner.RunStatus(
            run_id="dep-leg2",
            state="deployed",
            spec={**spec, "run_id": "dep-leg2"},
            remote=None,
            updated_at=deploy_time,
            finished_at=None,
            deployment={"endpoint_name": "e"},
        )
    )
    out2 = runner.mark_deployed("dep-leg2", {"endpoint_name": "e2"}, expect_state="deployed")
    assert out2.state == "deployed"
    assert out2.finished_at is None  # NOT stamped from the deploy-time updated_at

    # (3) reconciled-then-deployed legacy `done` run: record_realized_cost bumped updated_at to the
    # reconcile time, so the backfill must NOT freeze that (later) stamp as teardown.
    runner._save_status(
        runner.RunStatus(
            run_id="dep-leg3",
            state="done",
            spec={**spec, "run_id": "dep-leg3"},
            remote=None,
            updated_at=9_000.0,  # reconcile-time bump, well after teardown
            finished_at=None,
            reconciled_at=8_500.0,
        )
    )
    out3 = runner.mark_deployed("dep-leg3", {"endpoint_name": "e3"})
    assert out3.state == "deployed"
    assert out3.finished_at is None  # not frozen from the reconcile-bumped updated_at


def test_deploy_lock_is_usable_and_weakly_cleaned():
    # threading.Lock() isn't weak-referenceable, so the per-run lock must be a wrapper that
    # both works as a context manager AND can live in the WeakValueDictionary (the raw lock
    # would TypeError on the first deploy). It must also re-enter and serialize.
    import gc

    from flash.server import app as app_mod

    lk = app_mod._deploy_lock("run-xyz")
    assert app_mod._deploy_lock("run-xyz") is lk  # same lock for the same run while alive
    with lk:
        pass
    with app_mod._deploy_lock("run-xyz"):  # re-acquirable after release
        pass
    # once nothing references it, the weak entry is dropped (no unbounded growth).
    del lk
    gc.collect()
    assert "run-xyz" not in dict(app_mod._DEPLOY_LOCKS)


def test_recover_runs_resubmits_no_handle_run(monkeypatch, tmp_path):
    # A recoverable run with no persisted handle (crash in the submit->on_handle window,
    # before any worker was provisioned) must NOT be lost on a control-plane restart: its
    # reconstructable RunPod endpoint is GC'd (so it doesn't hold worker quota), then the run
    # is RESUBMITTED from scratch — there is no remote work to reattach to, so a fresh job is
    # the only way to preserve the session.
    import threading

    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "nohandle-1",
    }
    runner._save_status(
        runner.RunStatus(run_id="nohandle-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-1"}])
    gced = []
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: gced.append(s.run_id))
    # Capture the resubmit instead of provisioning a real GPU; recover_runs resolves _run_job
    # via a function-local `from flash.runner import _run_job`, so patching the package attr wins.
    resubmitted = []
    done = threading.Event()

    def fake_run_job(s):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    # Codex MtzrJ: a handle-less run may have left a phantom instance from a non-idempotent create
    # (Vast PUT /asks) that surfaces via eventual consistency. Recovery must force-reap the run's label
    # across instance providers RIGHT BEFORE resubmitting, so a phantom isn't left writing the same
    # seed-scoped artifacts as the fresh worker. Capture the gc-by-run.
    reaped = []

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            reaped.append(s.run_id)

        def sweep_orphans(self, **k):
            return []

    import flash.providers as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "no-handle recovery must launch a resubmit thread"
    assert gced == ["nohandle-1"], "no-handle recovery must GC the reconstructable endpoint first"
    assert resubmitted == ["nohandle-1"], "no-handle run must be resubmitted, not failed"
    assert reaped == ["nohandle-1"], "must force-reap the run's instance-provider label before resubmit"
    # The resubmit GC's the orphaned endpoint and re-runs the job; the run is NOT failed.
    assert runner.get_status("nohandle-1").state != "failed"


def test_recover_runs_defers_resubmit_when_instance_not_confirmed_reaped(monkeypatch, tmp_path):
    # Codex: a handle-less run's force-reap before resubmit is best-effort — Vast's gc
    # (destroy_run_instances) returns an empty list rather than raising when a DELETE is unconfirmed
    # (success:false / network). If an instance for this run is STILL present after gc, recovery must
    # NOT launch a second worker over it (double-write the same seed-scoped HF artifacts); it defers the
    # run for a later recovery/sweep. run_instances_remaining([...]) reports the survivor.
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "phantom-1",
    }
    runner._save_status(
        runner.RunStatus(run_id="phantom-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "phantom-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    monkeypatch.setattr(runner, "_run_job", lambda s: resubmitted.append(s.run_id))

    reaped = []

    class _FakeVast:
        name = "vast"

        def gc(self, s):  # unconfirmed DELETE -> destroys nothing, returns no error
            reaped.append(s.run_id)

        def run_instances_remaining(self, run_id):  # the phantom is STILL there after gc
            return [4242]

        def sweep_orphans(self, **k):
            return []

    import flash.providers as providers_mod
    import flash.server._runtime as rt

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])
    # Disable the background retry budget so the defer is a clean no-op for this assertion (no lingering
    # daemon thread polling a torn-down tmp db); the reschedule behavior has its own test below.
    monkeypatch.setattr(rt, "_DEFERRED_RECOVERY_MAX_RETRIES", 0)

    app_mod.recover_runs()

    assert reaped == ["phantom-1"], "must still attempt the force-reap"
    assert resubmitted == [], "must NOT resubmit while an instance for the run may still be live"
    assert runner.get_status("phantom-1").state != "failed", "deferred, not failed (later recovery retries)"


def test_recover_runs_defers_when_recorded_provider_unconfigurable(monkeypatch, tmp_path):
    # Codex: a handle-less run's lost create could have left a Vast phantom on a provider whose creds
    # were dropped before the restart (VAST_API_KEY removed). configured_providers() then omits Vast, so
    # iterating only the configured set would silently return "clear" and resubmit a SECOND worker while
    # the phantom keeps billing + writing the same HF prefix. The guard must FAIL CLOSED for an instance
    # provider RECORDED as available at submit (submitted_instance_providers) that it can no longer
    # enumerate. Scoping to the recorded set is what keeps a never-Vast plane recoverable (it never
    # records Vast); here the run recorded Vast, so its recovery defers.
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "unconf-1",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="unconf-1",
            state="provisioning",
            spec=spec,
            remote=None,
            submitted_instance_providers=["vast"],  # Vast was configured when this run was submitted
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "unconf-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    monkeypatch.setattr(runner, "_run_job", lambda s: resubmitted.append(s.run_id))

    import flash.providers as providers_mod
    import flash.server._runtime as rt

    # Vast is no longer configured -> omitted from configured_providers(); the real get_provider("vast")
    # still exposes run_instances_remaining, so the recorded-but-unconfigurable provider can't be
    # enumerated -> the guard must fail closed rather than declare clear.
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [])
    monkeypatch.setattr(rt, "_DEFERRED_RECOVERY_MAX_RETRIES", 0)

    app_mod.recover_runs()

    assert resubmitted == [], "must NOT resubmit while an uncheckable Vast phantom may still bill"
    assert runner.get_status("unconf-1").state != "failed", "deferred, not failed (later restart retries)"


def test_recover_runs_resubmits_when_no_capability_provider_recorded(monkeypatch, tmp_path):
    # The fail-closed must stay SCOPED: a handle-less run on a plane that never configured Vast records
    # no Vast in submitted_instance_providers, so it can't have left a Vast phantom and must still
    # recover (resubmit) even though get_provider("vast") exposes the capability. Guards against the
    # over-broad "any unconfigured capability provider blocks" regression that would strand RunPod/Lambda
    # -only deployments.
    import threading

    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "novast-1",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="novast-1",
            state="provisioning",
            spec=spec,
            remote=None,
            submitted_instance_providers=[],  # no instance provider was available at submit
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "novast-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(runner, "_run_job", lambda s: (resubmitted.append(s.run_id), done.set()))

    import flash.providers as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "a run that never recorded Vast must still recover on a Vast-less plane"
    assert resubmitted == ["novast-1"]


def test_recover_runs_ignores_newly_configured_unrecorded_provider(monkeypatch, tmp_path):
    # A provider enabled after submit cannot have owned that run's pre-handle create. Its listing outage
    # must not strand recovery when submitted_instance_providers explicitly says it was not available.
    import threading

    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "newvast-1",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="newvast-1",
            state="provisioning",
            spec=spec,
            remote=None,
            submitted_instance_providers=[],
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "newvast-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(runner, "_run_job", lambda s: (resubmitted.append(s.run_id), done.set()))

    class _NewVast:
        name = "vast"

        def gc(self, s):
            raise AssertionError("newly configured unrecorded provider must not be reaped")

        def run_instances_remaining(self, run_id):
            raise AssertionError("newly configured unrecorded provider must not block recovery")

        def sweep_orphans(self, **k):
            return []

    import flash.providers as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_NewVast()])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "newly configured unrecorded Vast must not block recovery"
    assert resubmitted == ["newvast-1"]


def test_recover_runs_deferred_resubmit_retries_until_clear(monkeypatch, tmp_path):
    # Codex: a deferred handle-less resubmit must not be stranded until the next control-plane restart —
    # a bounded background retry re-confirms the reap and resubmits once it becomes safe (the phantom is
    # gone / the listing recovers). Here run_instances_remaining reports the box present on the first
    # check, then clear -> the background loop resubmits on the retry.
    import threading

    import flash.runner as runner
    import flash.server._runtime as rt
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "retry-1",
    }
    runner._save_status(
        runner.RunStatus(run_id="retry-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "retry-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()
    monkeypatch.setattr(
        runner, "_run_job", lambda s: (resubmitted.append(s.run_id), done.set())
    )
    monkeypatch.setattr(rt, "_DEFERRED_RECOVERY_RETRY_S", 0.01)  # fast background retry

    calls = {"n": 0}

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            pass

        def run_instances_remaining(self, run_id):
            calls["n"] += 1
            return [4242] if calls["n"] == 1 else []  # present once, then cleared

        def sweep_orphans(self, **k):
            return []

    import flash.providers as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()  # first check sees the box -> defers + schedules the background retry

    assert done.wait(timeout=5), "the background retry must resubmit once the run is confirmed clear"
    assert resubmitted == ["retry-1"]


def test_recover_runs_resubmits_when_instance_confirmed_clear(monkeypatch, tmp_path):
    # The confirmation gate must not block the normal case: when run_instances_remaining returns []
    # (confirmed no instance for this run remains), the handle-less run resubmits as before.
    import threading

    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "clear-1",
    }
    runner._save_status(
        runner.RunStatus(run_id="clear-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "clear-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    resubmitted = []
    done = threading.Event()

    def fake_run_job(s):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    class _FakeVast:
        name = "vast"

        def gc(self, s):
            pass

        def run_instances_remaining(self, run_id):  # confirmed clear
            return []

        def sweep_orphans(self, **k):
            return []

    import flash.providers as providers_mod

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeVast()])

    app_mod.recover_runs()

    assert done.wait(timeout=5), "a confirmed-clear run must still resubmit"
    assert resubmitted == ["clear-1"]


def test_recover_runs_bad_spec_is_isolated_not_fatal(monkeypatch, tmp_path):
    # Fault isolation: if the FIRST recoverable run's persisted spec is malformed (e.g. a
    # unsupported `environment.path`, which makes JobSpec.from_dict raise), recovery of that one
    # run must be skipped — it must NOT abort recover_runs() and thereby skip recovery of
    # every OTHER in-flight run and the orphan sweep that follows. Here run #1 has a bad spec
    # and run #2 has a valid no-handle spec: assert run #2 is still resubmitted AND the orphan
    # sweep still runs (the bad spec didn't take down the whole recovery pass).
    import threading

    import flash.providers as providers_mod
    import flash.providers.runpod.train as runpod_train
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    # Run #1: a malformed spec — local `environment.path` makes from_dict raise.
    bad_spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "environment": {"path": "/legacy/local/env"},
        "train": {"steps": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "bad-1",
    }
    # Run #2: a valid no-handle spec — must still be recovered (resubmitted) despite run #1.
    good_spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"steps": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "good-2",
    }
    runner._save_status(
        runner.RunStatus(run_id="bad-1", state="provisioning", spec=bad_spec, remote=None)
    )
    runner._save_status(
        runner.RunStatus(run_id="good-2", state="provisioning", spec=good_spec, remote=None)
    )
    # Order matters: the bad run is iterated FIRST, so an unguarded parse would abort here.
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "bad-1"}, {"run_id": "good-2"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)

    # A malformed spec can't be parsed into a JobSpec, so the good-spec branch's
    # `_gc_run_endpoints(spec)` is unavailable — yet the aborted attempt may still have
    # registered its uniquely-named RunPod endpoint before crashing, which the no-op RunPod
    # `sweep_orphans` won't reap. recover_runs must instead derive the endpoint name from the
    # RAW persisted status (gpu.type + run_id, no spec parse) and `terminate_endpoint` it.
    # The malformed path imports it via `from flash.providers.runpod.train import
    # terminate_endpoint`, so patch the attribute on that module and record the call args.
    terminated = []
    monkeypatch.setattr(
        runpod_train,
        "terminate_endpoint",
        lambda gpu_type, run_id=None: terminated.append((gpu_type, run_id)) or [],
    )

    # The orphan sweep must still run after the loop. recover_runs resolves it via a
    # function-local `from flash.providers import configured_providers`, so patch the
    # package attr; record that sweep_orphans fired.
    swept = threading.Event()

    class _FakeProvider:
        # known_labels is part of the real sweep_orphans signature (multi-plane guard); accept it so the
        # actual call prov.sweep_orphans(active_labels=..., known_labels=...) doesn't TypeError (which
        # the recovery suppress would swallow, silently skipping the sweep this test asserts fired).
        def sweep_orphans(self, active_labels=None, known_labels=None):
            swept.set()
            return []

    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [_FakeProvider()])

    resubmitted = []
    done = threading.Event()

    def fake_run_job(s):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    app_mod.recover_runs()

    assert done.wait(timeout=5), "the valid run must still be resubmitted despite a prior bad spec"
    assert resubmitted == ["good-2"], "only the valid run resubmits; the malformed one is skipped"
    assert swept.is_set(), "a malformed spec must not abort the orphan sweep that follows the loop"

    # The malformed run must NOT be silently skipped and left recoverable (it would be
    # retried-then-skipped on every restart forever, invisible to the user). It must be
    # persisted as terminal `failed` with an operator-visible error note, so it surfaces to
    # the user AND drops out of the recoverable set (never re-attempted).
    bad_status = runner.get_status("bad-1")
    assert bad_status.state == "failed", "an unparseable persisted spec must be marked failed"
    assert bad_status.state in runner.TERMINAL_STATES, (
        "failed is terminal, so it won't recover again"
    )
    assert bad_status.state not in app_mod._RECOVERABLE, "the failed run leaves the recoverable set"
    assert bad_status.error, "the failed run must carry an operator-visible error note"
    assert "unrecoverable" in bad_status.error, (
        "the failure note must explain the malformed spec to the operator"
    )

    # Resource-leak guard: even though the spec couldn't be parsed, the malformed run's RunPod
    # endpoint must still be torn down — derived from the RAW persisted gpu.type + run_id, not
    # from a JobSpec — so a crash that registered an endpoint before persisting a handle can't
    # leak it (RunPod's `sweep_orphans` is a no-op and would never catch it). Best-effort GC
    # must run for the malformed run AND not have aborted the failed-marking / sweep / resubmit
    # above (all already asserted), proving it's properly suppressed and ordered.
    assert ("RTX 5090", "bad-1") in terminated, (
        "a malformed-spec run's endpoint must be GC'd by reconstructed name (raw gpu.type + "
        "run_id), since its spec can't be parsed and the RunPod orphan sweep is a no-op"
    )


def test_publish_env_endpoint_publishes_under_managed_account(api, monkeypatch):
    """POST /v1/envs publishes an uploaded package to the managed environment hub."""
    import base64
    import io
    import tarfile

    import flash.server.envs as envs_mod

    published_roots: list[str] = []
    monkeypatch.setattr(
        envs_mod,
        "_github_publish_once",
        lambda *, publish_root, **_kwargs: published_roots.append(publish_root),
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("environment.py", b"def load_environment(**kwargs): return None\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    key = _login()
    expected_root = f"org-{key.removeprefix(_USER_PREFIX)}/myenv"
    resp = api.post(
        "/v1/envs",
        headers=_bearer(key),
        json={"name": "MyEnv", "package_b64": pkg},
    )
    assert resp.status_code == 200
    ref = resp.json()["id"]
    assert ref == expected_root
    assert expected_root in published_roots

    # Unauthenticated requests are rejected.
    assert api.post("/v1/envs", json={"name": "e", "package_b64": pkg}).status_code in (401, 403)


def test_publish_env_ignores_legacy_is_new(api, monkeypatch):
    """Publish mode is determined by the explicit name and server publish id."""
    import base64
    import io
    import tarfile

    import flash.server.envs as envs_mod

    seen: dict = {}

    def fake_publish_package(*, package_b64, name, key):
        seen.update(package_b64=package_b64, name=name, key=key)
        return "key-1/e"

    monkeypatch.setattr(envs_mod, "publish_package", fake_publish_package)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for nm, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("environment.py", b"def load_environment(**kwargs): return None\n"),
        ):
            info = tarfile.TarInfo(nm)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    resp = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "e", "package_b64": pkg, "is_new": False},
    )
    assert resp.status_code == 200, resp.text
    assert seen["name"] == "e"
    assert seen["package_b64"] == pkg
    assert seen["key"]["org_slug"].startswith("org-")
    assert "is_new" not in seen


def test_publish_env_falsy_non_string_fields_are_not_coerced(api):
    """Regression: a present-but-falsy non-string `name`/`package_b64` (e.g. 0, False, []) must
    reach publish_package's type check and yield the *type* 400 — not be `or ""`-coerced to an
    empty string first (which would surface a different/misleading 400)."""
    # name = 0 -> hits the name type check, not "missing env name".
    r = api.post("/v1/envs", headers=_bearer(_login()), json={"name": 0, "package_b64": "x"})
    assert r.status_code == 400, r.text
    assert "name must be a string" in r.text.lower()
    # package_b64 = False (valid string name) -> hits the package type check.
    r2 = api.post("/v1/envs", headers=_bearer(_login()), json={"name": "e", "package_b64": False})
    assert r2.status_code == 400, r2.text
    assert "must be a base64 string" in r2.text.lower()


def test_delete_env_endpoint_removes_package(api, monkeypatch):
    """DELETE /v1/envs/{id} removes the package and reports it deleted."""
    import flash.server.envs as envs_mod
    from flash.server import environment_registry

    seen: dict = {}

    def fake_delete_package(*, slug, key):
        seen.update(slug=slug, key=key)
        return True

    monkeypatch.setattr(envs_mod, "delete_package", fake_delete_package)
    recorded: dict = {}
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda *, slug, key, org_id=None: recorded.update(slug=slug, org_id=org_id) or True,
    )

    resp = api.delete(
        "/v1/envs/acme/my-env",
        headers={**_bearer(_login()), "X-Freesolo-Org-Id": "org-acme"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": "acme/my-env", "deleted": True}
    assert seen["slug"] == "acme/my-env"
    assert recorded["slug"] == "acme/my-env"
    # the caller-supplied org (web UI delete) reaches the metadata-mirror drop.
    assert recorded["org_id"] == "org-acme"

    # Unauthenticated requests are rejected.
    assert api.delete("/v1/envs/acme/my-env").status_code in (401, 403)


def test_delete_env_endpoint_maps_publish_error_status(api, monkeypatch):
    """A namespace-authorization EnvPublishError surfaces as its HTTP status (403)."""
    import flash.server.envs as envs_mod

    def fake_delete_package(*, slug, key):
        raise envs_mod.EnvPublishError("not your namespace", status=403)

    monkeypatch.setattr(envs_mod, "delete_package", fake_delete_package)
    resp = api.delete("/v1/envs/someone-else/env", headers=_bearer(_login()))
    assert resp.status_code == 403, resp.text


def test_delete_env_endpoint_mirror_failure_is_non_fatal(api, monkeypatch):
    """A failing metadata-mirror delete must not turn a successful delete into a 500."""
    import flash.server.envs as envs_mod
    from flash.server import environment_registry

    monkeypatch.setattr(envs_mod, "delete_package", lambda *, slug, key: True)

    def boom(*, slug, key, org_id=None):
        raise RuntimeError("backend down")

    monkeypatch.setattr(environment_registry, "record_deleted_environment", boom)
    resp = api.delete("/v1/envs/acme/my-env", headers=_bearer(_login()))
    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted"] is True


def test_delete_env_endpoint_rejects_non_canonical_id(api, monkeypatch):
    """A non-canonical id (uppercase / trailing slash) is rejected 400 before any storage call."""
    import flash.server.envs as envs_mod
    from flash.server import environment_registry

    monkeypatch.setattr(
        envs_mod, "delete_package", lambda **_k: pytest.fail("storage must not be touched")
    )
    monkeypatch.setattr(
        environment_registry,
        "record_deleted_environment",
        lambda **_k: pytest.fail("mirror must not be touched"),
    )
    for bad in ("Acme/My-Env", "acme/my-env/"):
        resp = api.delete(f"/v1/envs/{bad}", headers=_bearer(_login()))
        assert resp.status_code == 400, resp.text


# --------------------------------------------------------------------------------------------
# Deployable RL checkpoints: list + deploy-by-step (incl. a run cancelled mid-RL).
# --------------------------------------------------------------------------------------------
_FAKE_CKPTS = [
    {
        "step": 40,
        "adapter_prefix": "rl/X/checkpoints/step-40",
        "subfolder": "rl/X/checkpoints/step-40/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
    {
        "step": 80,
        "adapter_prefix": "rl/X/checkpoints/step-80",
        "subfolder": "rl/X/checkpoints/step-80/adapter",
        "repo_id": "org/test-runs",
        "repo_type": "dataset",
    },
]


class _FakeDeployment:
    def __init__(self, adapter_prefix):
        self.adapter_prefix = adapter_prefix

    def to_dict(self):
        return {
            "state": "ready",
            "run_id": "X",
            "adapter_hf_prefix": f"{self.adapter_prefix}/adapter",
        }


def _make_run(api, key, state):
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    import flash.runner as runner

    status = runner.get_status(run_id)
    status.state = state
    runner._save_status(status)
    return run_id


def test_list_checkpoints_endpoint(api, monkeypatch):
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    r = api.get(f"/v1/runs/{run_id}/checkpoints", headers=_bearer(key))
    assert r.status_code == 200, r.text
    assert [c["step"] for c in r.json()["checkpoints"]] == [40, 80]


def test_deploy_specific_checkpoint_of_finished_run(api, monkeypatch):
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    captured = {}

    def fake_deploy(**kwargs):
        captured.update(kwargs)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    key = _login()
    run_id = _make_run(api, key, "done")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    # Served the step-40 checkpoint's adapter, not the run's final adapter.
    assert captured["adapter_prefix"].endswith("/checkpoints/step-40")
    assert r.json()["checkpoint_step"] == 40
    # A finished run flips to `deployed` as usual.
    import flash.runner as runner

    assert runner.get_status(run_id).state == "deployed"


def test_deploy_checkpoint_of_cancelled_run_keeps_terminal_state(api, monkeypatch):
    """The headline fix: a run cancelled mid-RL can deploy a checkpoint, and stays `cancelled`."""
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(app_mod, "deploy_adapter", lambda **k: _FakeDeployment(k["adapter_prefix"]))

    key = _login()
    run_id = _make_run(api, key, "cancelled")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 80}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    assert r.json()["checkpoint_step"] == 80
    # Training outcome preserved (NOT flipped to `deployed`)...
    assert runner.get_status(run_id).state == "cancelled"
    # ...but the serving deployment is recorded and listed as active.
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert any(d["run_id"] == run_id for d in deployments)


@pytest.mark.parametrize("state", ["queued", "provisioning", "running", "failed"])
def test_deploy_checkpoint_ignores_run_state_once_step_exists(api, monkeypatch, state):
    """A resolved checkpoint step proves the adapter exists, so run state does not gate serving it."""
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(app_mod, "deploy_adapter", lambda **k: _FakeDeployment(k["adapter_prefix"]))

    key = _login()
    run_id = _make_run(api, key, state)
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    assert r.json()["checkpoint_step"] == 40
    status = runner.get_status(run_id)
    assert status.state == state
    assert status.deployment["checkpoint_step"] == 40
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert any(d["run_id"] == run_id and d["state"] == state for d in deployments)


def test_deploy_checkpoint_promotes_if_run_finishes_during_registration(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "running")

    def fake_deploy(**kwargs):
        status = runner.get_status(run_id)
        status.state = "done"
        runner._save_status(status)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    status = runner.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment["checkpoint_step"] == 40


def test_deploy_checkpoint_rolls_back_if_final_deploy_wins_cas(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    rollbacks = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id: rollbacks.append(run_id))

    key = _login()
    run_id = _make_run(api, key, "done")

    def fake_deploy(**kwargs):
        runner.mark_deployed(run_id, {"state": "ready", "endpoint_name": "final"})
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 409, r.text
    assert rollbacks == [run_id]
    status = runner.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment == {"state": "ready", "endpoint_name": "final"}


def test_deploy_checkpoint_of_dry_run_run_is_409(api, monkeypatch):
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("dry-run run must not touch serving"),
    )

    key = _login()
    run_id = _make_run(api, key, "dry_run")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 409, r.text
    assert "dry-run runs cannot be deployed" in r.json()["detail"]


def test_deploy_checkpoint_preserves_finished_run_undeploy_cas(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    rollbacks = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id: rollbacks.append(run_id))

    key = _login()
    run_id = _make_run(api, key, "deployed")
    status = runner.get_status(run_id)
    status.deployment = {"state": "ready", "endpoint_name": "old"}
    runner._save_status(status)

    def fake_deploy(**kwargs):
        runner.mark_undeployed(run_id)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 409, r.text
    assert rollbacks == [run_id]
    status = runner.get_status(run_id)
    assert status.state == "done"
    assert status.deployment["state"] == "undeployed"


def test_undeploy_checkpoint_of_running_run_keeps_training_state(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    monkeypatch.setattr(app_mod, "deploy_adapter", lambda **k: _FakeDeployment(k["adapter_prefix"]))
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id: [run_id])

    key = _login()
    run_id = _make_run(api, key, "running")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text

    r = api.delete(f"/v1/runs/{run_id}/deploy", headers=_bearer(key))
    assert r.status_code == 200, r.text
    status = runner.get_status(run_id)
    assert status.state == "running"
    assert status.deployment["state"] == "undeployed"


def test_deploy_cancelled_run_without_step_is_409(api):
    """Without a step, a cancelled run is still undeployable (no final adapter)."""
    key = _login()
    run_id = _make_run(api, key, "cancelled")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert r.status_code == 409, r.text


def test_deploy_unknown_step_is_404_with_available(api, monkeypatch):
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 999}, headers=_bearer(key))
    assert r.status_code == 404, r.text
    assert "available: 40, 80" in r.json()["detail"]


def test_deploy_rejects_non_integer_step(api, monkeypatch):
    """A bool (True->1) or non-integer step must be rejected, not silently coerced. An all-digit string
    over Python's 4300-digit int()-conversion limit must also be a clean 400, not int()->uncaught 500."""
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    run_id = _make_run(api, key, "done")
    for bad in (True, 40.9, "40.9", "1" * 5000):
        r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": bad}, headers=_bearer(key))
        assert r.status_code == 400, f"{bad!r} -> {r.status_code} {r.text}"


def test_create_run_records_managed_environment_use(api, monkeypatch):
    import flash.server.environment_registry as registry

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_environment_use",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    spec = {**SPEC, "environment": {"id": "acme/my-env"}}
    key = _login()

    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": spec, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    assert calls
    assert calls[0]["slug"] == "acme/my-env"
    assert calls[0]["run_id"] == resp.json()["run_id"]
    assert calls[0]["key"]["org_id"] == f"org-{key.removeprefix(_USER_PREFIX)}"


def test_create_run_records_flash_training_run(api, monkeypatch):
    import flash.server.run_registry as registry

    calls: list[dict] = []
    monkeypatch.setattr(
        registry,
        "record_training_run",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    key = _login()

    resp = api.post(
        "/v1/runs",
        headers=_bearer(key),
        json={"spec": SPEC, "dry_run": True},
    )

    assert resp.status_code == 200, resp.text
    assert calls
    last = calls[-1]
    assert last["status"].run_id == resp.json()["run_id"]
    # Org attribution rides on the persisted platform_context (org_id/user_id/api_key_id),
    # which submit_job reports for us — create_run no longer double-POSTs with an explicit key.
    assert last["status"].platform_context["org_id"] == f"org-{key.removeprefix(_USER_PREFIX)}"


# --- export: copy a trained adapter to a user-owned HuggingFace repo ----------------------


def _finished_run(api, key) -> str:
    """Submit a run and flip it to `done` (a finished run with a trained final adapter)."""
    import flash.runner as runner

    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    return run_id


def test_export_copies_final_adapter_to_user_repo(api, monkeypatch):
    """A finished run's final adapter is read from its private artifact repo (operator side) and
    re-uploaded to the user's repo with the user's token; the response reports the source path."""
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    # The platform auto-assigns each run a per-run HF dataset repo under the OPERATOR's org, so
    # only the control plane (operator token) can read the source — read it back from the run.
    src_repo = runner.get_status(run_id).spec["train"]["hf_repo"]

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return "https://huggingface.co/me/adapters"

    monkeypatch.setattr(app_mod, "export_adapter", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/adapters", "hf_token": "hf_user"},
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["repository"] == "me/adapters"
    assert body["url"] == "https://huggingface.co/me/adapters"
    assert body["source"] == f"{src_repo}:rl/{run_id}/adapter"
    assert "step" not in body
    # Source = the run's private dataset repo + final-adapter subfolder; dest = the user's repo.
    assert seen["source_repo"] == src_repo
    assert seen["source_subfolder"] == f"rl/{run_id}/adapter"
    assert seen["dest_repo"] == "me/adapters"
    assert seen["dest_token"] == "hf_user"
    assert seen["private"] is True  # private by default
    assert seen["base_model"] == SPEC["model"]


def test_export_holds_deploy_lock_across_owned_run(api, monkeypatch):
    """The /export handler must take the per-run deploy lock FROM THE VERY TOP — before even the
    payload-shape validation — and keep it across owned_run/the artifact read (mirroring /deploy,
    which locks first). Taking the lock after body validation leaves a window for another deploy,
    undeploy, or export operation to interleave with the request. Assert the lock is already held during
    the payload validation (``_validate_hf_repo_id``, which runs first) AND by the time owned_run
    runs."""
    import flash.server.app as app_mod
    from flash.server.routes import serving as serving_routes

    key = _login()
    run_id = _finished_run(api, key)

    real_owned_run = serving_routes.owned_run
    real_validate = serving_routes._validate_hf_repo_id
    seen: dict = {}

    def checking_validate(repository):
        # The payload validation runs BEFORE owned_run; assert it too is inside the lock.
        lk = app_mod._deploy_lock(run_id)
        acquired = lk.acquire(blocking=False)
        seen["locked_during_validation"] = not acquired
        if acquired:
            lk.release()
        return real_validate(repository)

    def checking_owned_run(rid, k):
        # A non-blocking acquire from outside must FAIL (lock already held by the handler) — proving
        # the handler is INSIDE the `with _deploy_lock(...)` block by the time owned_run runs.
        lk = app_mod._deploy_lock(rid)
        acquired = lk.acquire(blocking=False)
        seen["locked_during_owned_run"] = not acquired
        if acquired:
            lk.release()
        return real_owned_run(rid, k)

    monkeypatch.setattr(serving_routes, "_validate_hf_repo_id", checking_validate)
    monkeypatch.setattr(serving_routes, "owned_run", checking_owned_run)
    monkeypatch.setattr(app_mod, "export_adapter", lambda **kw: "https://huggingface.co/me/a")

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf"},
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    assert seen["locked_during_validation"] is True
    assert seen["locked_during_owned_run"] is True


def test_export_public_flag_sets_private_false(api, monkeypatch):
    import flash.server.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    seen: dict = {}
    monkeypatch.setattr(
        app_mod,
        "export_adapter",
        lambda **kw: (seen.update(kw), "https://huggingface.co/me/a")[1],
    )
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "private": False},
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    assert seen["private"] is False


def test_export_validates_repository_and_token(api):
    key = _login()
    run_id = _finished_run(api, key)
    missing_repo = api.post(
        f"/v1/runs/{run_id}/export", json={"hf_token": "hf"}, headers=_bearer(key)
    )
    assert missing_repo.status_code == 400
    assert "repository" in missing_repo.json()["detail"]
    missing_token = api.post(
        f"/v1/runs/{run_id}/export", json={"repository": "me/a"}, headers=_bearer(key)
    )
    assert missing_token.status_code == 400
    assert "hf_token" in missing_token.json()["detail"]
    # SHAPE check: a HF repo id is EXACTLY two non-empty segments — reject no-slash AND the over-/
    # under-segmented forms that the old "at least one '/'" check let through (which would 404/400 deep
    # in hf_hub). These produce the "owner/name" shape error.
    for bad_repo in ("noslash", "owner/name/extra", "owner//name", "/name", "name/"):
        malformed = api.post(
            f"/v1/runs/{run_id}/export",
            json={"repository": bad_repo, "hf_token": "hf"},
            headers=_bearer(key),
        )
        assert malformed.status_code == 400, bad_repo
        assert "owner/name" in malformed.json()["detail"]
    # GRAMMAR check: two segments but NOT a valid HF repo id — embedded whitespace, a segment that
    # starts/ends with '-' or '.', a '--'/'..' run, or a >96-char name. The full Hub grammar
    # (huggingface_hub.validate_repo_id) must reject these FAST with a 400, not let export_adapter
    # download the private source adapter first and hit a wrapped 502 from create_repo.
    for bad_repo in (
        "owner/ name",
        "own er/name",
        "owner/na\tme",
        "owner/na me",
        "owner/-bad",
        "owner/bad-",
        "owner/.bad",
        "owner/bad--name",
        "owner/ba..d",
        "owner/" + "x" * 97,
    ):
        malformed = api.post(
            f"/v1/runs/{run_id}/export",
            json={"repository": bad_repo, "hf_token": "hf"},
            headers=_bearer(key),
        )
        assert malformed.status_code == 400, bad_repo
        assert "valid HuggingFace repo id" in malformed.json()["detail"], bad_repo


def test_export_unfinished_run_is_409(api, monkeypatch):
    """A run with no trained final adapter (never finished) can't be exported — and the HF copy
    is never attempted."""
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]

    def must_not_run(**kw):
        raise AssertionError("export_adapter must not run for an unfinished run")

    monkeypatch.setattr(app_mod, "export_adapter", must_not_run)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf"},
        headers=_bearer(key),
    )
    assert resp.status_code == 409, resp.text


def test_export_missing_artifacts_is_404(api, monkeypatch):
    import flash.server.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)

    def boom(**kw):
        raise ValueError("no adapter artifacts found at org/test-runs:... (nothing to export)")

    monkeypatch.setattr(app_mod, "export_adapter", boom)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf"},
        headers=_bearer(key),
    )
    assert resp.status_code == 404, resp.text
    assert "no adapter artifacts" in resp.json()["detail"]


def test_export_hf_failure_is_clean_502(api, monkeypatch):
    """An HF transport/permission failure (download or upload) surfaces as a clean 502 carrying the
    real reason — not an unhandled 500 (mirrors the deploy/undeploy ServingError handling)."""
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = _finished_run(api, key)

    def boom(**kw):
        raise ServingError("could not upload adapter to me/a: 403 Forbidden")

    monkeypatch.setattr(app_mod, "export_adapter", boom)
    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf"},
        headers=_bearer(key),
    )
    assert resp.status_code == 502, resp.text
    assert "could not upload" in resp.json()["detail"]


def test_export_step_targets_the_checkpoint_adapter(api, monkeypatch):
    """`--step N` exports the exact per-step checkpoint `flash deploy --step N` would serve; an
    unknown step 404s with the available list (resolved against published checkpoints)."""
    import flash.server.app as app_mod

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod,
        "list_checkpoints",
        lambda spec: [
            {
                "step": 40,
                "subfolder": f"rl/{run_id}/checkpoints/step-40/adapter",
                "repo_id": "org/test-runs",
                "repo_type": "dataset",
            }
        ],
    )
    seen: dict = {}
    monkeypatch.setattr(
        app_mod,
        "export_adapter",
        lambda **kw: (seen.update(kw), "https://huggingface.co/me/a")[1],
    )
    ok = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "step": 40},
        headers=_bearer(key),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["step"] == 40
    assert seen["source_subfolder"] == f"rl/{run_id}/checkpoints/step-40/adapter"
    assert seen["base_model"] == SPEC["model"]

    bad = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "step": 99},
        headers=_bearer(key),
    )
    assert bad.status_code == 404, bad.text
    assert "step 99" in bad.json()["detail"]
