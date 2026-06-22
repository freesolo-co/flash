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
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {"steps": 1, "seeds": [0], "hf_repo": "org/test-runs"},
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


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("PRIME_API_KEY", "pit-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
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
    # Offline auth: a token is a valid freesolo USER key iff it has the test prefix. This stub
    # replaces the real network verify.
    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: token.startswith(_USER_PREFIX))
    with TestClient(app_mod.create_app()) as client:
        yield client


def test_me(api):
    key = _login()
    me = api.get("/v1/me", headers=_bearer(key))
    assert me.status_code == 200
    # A verified freesolo user key resolves to a per-token "freesolo" identity.
    assert me.json()["email"] == "freesolo-user"
    assert me.json()["key_prefix"] == "freesolo"


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

    row = auth_mod.authenticate("Bearer fslo-user-good")
    assert row is not None
    assert row["email"] == "freesolo-user"
    # An unverified token returns None (401).
    assert auth_mod.authenticate("Bearer fslo-user-bad") is None
    # The same key resolves to the same identity across requests (stable per-token row).
    again = auth_mod.authenticate("Bearer fslo-user-good")
    assert again["id"] == row["id"]


def test_freesolo_user_key_disabled_is_401_not_500(api, monkeypatch):
    # A freesolo key that verifies with the backend but whose db row was disabled (revoked)
    # must be rejected as 401 (authenticate -> None), not raise a 500.
    import sqlite3

    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    auth_mod._verify_cache.clear()
    monkeypatch.setattr(auth_mod, "_freesolo_verify", lambda token: True)

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
    neg_exp = auth_mod._verify_cache["neg"][1]
    assert neg_exp <= now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    # ...and definitely shorter than the long TTL.
    assert neg_exp < now + auth_mod._VERIFY_CACHE_TTL_S

    # Positive -> cached with the LONG TTL.
    auth_mod._verify_cache.clear()
    state["code"] = 200
    now = time.time()
    assert auth_mod._freesolo_verify("pos") is True
    pos_exp = auth_mod._verify_cache["pos"][1]
    assert pos_exp > now + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1
    assert pos_exp <= now + auth_mod._VERIFY_CACHE_TTL_S + 1

    # The negative entry expires after the short TTL: simulate the clock advancing past
    # _VERIFY_CACHE_NEG_TTL_S (but not the long TTL) and confirm the negative is treated as
    # expired while a same-age positive would still be live.
    auth_mod._verify_cache.clear()
    base = time.time()
    auth_mod._verify_cache["neg"] = (False, base + auth_mod._VERIFY_CACHE_NEG_TTL_S)
    auth_mod._verify_cache["pos"] = (True, base + auth_mod._VERIFY_CACHE_TTL_S)
    later = base + auth_mod._VERIFY_CACHE_NEG_TTL_S + 1.0  # past neg TTL, well under pos TTL
    assert auth_mod._verify_cache["neg"][1] <= later  # negative entry has expired
    assert auth_mod._verify_cache["pos"][1] > later  # positive entry is still live


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
    auth_mod._verify_cache["stale"] = (True, time.time() - 1)
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


def test_local_env_path_rejected(api):
    # Prime Hub-only: a local [environment] path is rejected on the managed service.
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


def test_deploy_dry_run(api):
    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    dep = api.post(
        f"/v1/runs/{run_id}/deploy", json={"mode": "dev", "dry_run": True}, headers=_bearer(key)
    )
    assert dep.status_code == 200, dep.text
    assert dep.json()["state"] == "dry_run"
    assert dep.json()["mode"] == "dev"
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

    resp = api.post(
        f"/v1/runs/{run_id}/deploy", json={"mode": "dev"}, headers=_bearer(key)
    )
    assert resp.status_code == 502, resp.text
    # The 502 carries the upstream reason verbatim, not a generic/unhandled-500 body.
    assert "serving backend unreachable" in resp.json()["detail"]
    # The failed deploy left no active deployment record behind.
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_mark_deployed_allows_done_but_not_cancelled(monkeypatch, tmp_path):
    # A finished run (state="done") MUST be deployable: mark_deployed has to record the
    # deployment and flip to "deployed". But a cancelled/failed run must never be flipped
    # to "deployed" (a /cancel racing always-on provisioning persisted the terminal state).
    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo", "run_id": "dep-1"}
    runner._save_status(runner.RunStatus(run_id="dep-1", state="done", spec=spec, remote=None))
    out = runner.mark_deployed("dep-1", {"endpoint_name": "e", "mode": "dev"})
    assert out.state == "deployed"
    assert out.deployment == {"endpoint_name": "e", "mode": "dev"}

    # cancelled is sticky: the deploy must be refused, state preserved.
    runner._save_status(
        runner.RunStatus(
            run_id="dep-2", state="cancelled", spec={**spec, "run_id": "dep-2"}, remote=None
        )
    )
    out2 = runner.mark_deployed("dep-2", {"endpoint_name": "e2", "mode": "dev"})
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
            deployment={"endpoint_name": "e", "mode": "always-on"},
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


def test_recover_runs_gcs_no_handle_endpoints(monkeypatch, tmp_path):
    # A recoverable run with no persisted handle (crash between endpoint
    # registration and on_handle) must have its reconstructable RunPod endpoint
    # GC'd before being failed, so it doesn't hold worker quota until manual cleanup.
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
        "run_id": "nohandle-1",
    }
    runner._save_status(
        runner.RunStatus(run_id="nohandle-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-1"}])
    gced = []
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: gced.append(s.run_id))

    app_mod.recover_runs()

    assert gced == ["nohandle-1"], "no-handle recovery must GC the reconstructable endpoint"
    assert runner.get_status("nohandle-1").state == "failed"


def test_publish_env_endpoint_publishes_under_managed_account(api, monkeypatch):
    """POST /v1/envs publishes an uploaded package under FreeSolo's Prime account (the control
    plane's PRIME_API_KEY), namespaced per identity — so the user needs no Prime account."""
    import base64
    import io
    import tarfile

    import flash.server.envs as envs_mod

    # Stub the actual `prime env push` so the test doesn't hit Prime; echo the namespaced name.
    monkeypatch.setattr(
        envs_mod, "_prime_push", lambda env_dir, *, name, is_new: f"freesolo-co/{name}"
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in (
            ("pyproject.toml", b"[project]\nname='e'\n"),
            ("e/__init__.py", b"x=1\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    resp = api.post(
        "/v1/envs",
        headers=_bearer(_login()),
        json={"name": "MyEnv", "is_new": True, "package_b64": pkg},
    )
    assert resp.status_code == 200
    slug = resp.json()["id"]
    assert slug.startswith("freesolo-co/")  # published under the managed account
    assert slug.endswith("-myenv")  # per-identity namespaced + sanitized

    # Unauthenticated requests are rejected.
    assert api.post("/v1/envs", json={"name": "e", "package_b64": pkg}).status_code in (401, 403)


def test_publish_env_parses_is_new_robustly(api, monkeypatch):
    """`is_new` from the JSON body must be parsed as a real bool: the string "false"/"0" is False
    (a plain bool() would make any non-empty string True), and it defaults to True when absent."""
    import base64
    import io
    import tarfile

    import flash.server.envs as envs_mod

    seen: dict = {}

    def fake_push(env_dir, *, name, is_new):
        seen["is_new"] = is_new
        return f"freesolo-co/{name}"

    monkeypatch.setattr(envs_mod, "_prime_push", fake_push)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for nm, content in (("pyproject.toml", b"[project]\nname='e'\n"), ("e/__init__.py", b"x=1\n")):
            info = tarfile.TarInfo(nm)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    pkg = base64.b64encode(buf.getvalue()).decode()

    def _publish(body_extra):
        resp = api.post(
            "/v1/envs",
            headers=_bearer(_login()),
            json={"name": "e", "package_b64": pkg, **body_extra},
        )
        assert resp.status_code == 200, resp.text
        return seen["is_new"]

    # Falsey string forms -> False (the bug: bool("false") was True).
    assert _publish({"is_new": "false"}) is False
    assert _publish({"is_new": "0"}) is False
    assert _publish({"is_new": False}) is False
    # Truthy forms and absence -> True.
    assert _publish({"is_new": "true"}) is True
    assert _publish({"is_new": True}) is True
    assert _publish({}) is True  # default when absent


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
