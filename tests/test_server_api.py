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
    "train": {"epochs": 1, "max_examples": 1, "hf_repo": "org/test-runs"},
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
    monkeypatch.setenv("FLASH_DEPLOY_SYNC", "1")
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
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]},
    )
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


def test_dry_run_reports_schema_agreement_without_persisting_it(api) -> None:
    from flash.schema import train_schema_metadata

    metadata = {
        "version": "0.2.56",
        "fields": train_schema_metadata(),
        "authored_keys": sorted(SPEC["train"]),
    }
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True, "client_train_schema": metadata},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["train_schema_compatibility"] == {
        "status": "agreement",
        "client_only": [],
        "server_only": [],
        "introduced_in_differences": [],
    }
    status = api.get(f"/v1/runs/{body['run_id']}", headers=_bearer("fslo-internal-test")).json()
    assert "train_schema_compatibility" not in status


def test_dry_run_schema_disagreement_is_diagnostic_only(api) -> None:
    from flash.schema import train_schema_metadata

    fields = train_schema_metadata()
    fields.pop("teacher_model")
    fields.pop("structured_outputs")
    fields["epochs"] = "0.2.1"
    metadata = {
        "version": "0.2.55",
        "fields": fields,
        "authored_keys": sorted(SPEC["train"]),
    }
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": SPEC, "dry_run": True, "client_train_schema": metadata},
    )

    assert response.status_code == 200, response.text
    compatibility = response.json()["train_schema_compatibility"]
    assert compatibility["status"] == "disagreement"
    assert compatibility["client_only"] == []
    assert compatibility["server_only"] == ["structured_outputs", "teacher_model"]
    assert compatibility["introduced_in_differences"] == [
        {"key": "epochs", "client": "0.2.1", "server": "0.2.0"}
    ]


def test_missing_or_malformed_schema_metadata_does_not_change_parser_acceptance(api) -> None:
    payloads = [
        {"spec": SPEC, "dry_run": True},
        {
            "spec": SPEC,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.2.56",
                "fields": ["not", "a", "mapping"],
                "authored_keys": sorted(SPEC["train"]),
            },
        },
    ]

    for payload in payloads:
        response = api.post("/v1/runs", headers=_bearer("fslo-internal-test"), json=payload)
        assert response.status_code == 200, response.text
        assert "train_schema_compatibility" not in response.json()


def test_unknown_authored_train_key_enriches_parser_rejection_once(api, monkeypatch) -> None:
    import flash.server.routes.runs as runs_route
    from flash.schema import train_schema_metadata

    original_parse = runs_route._parse_spec
    calls = 0

    def counted_parse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(runs_route, "_parse_spec", counted_parse)
    monkeypatch.setattr(
        runs_route, "_runtime_secrets", lambda *_a, **_k: pytest.fail("secrets inspected")
    )
    monkeypatch.setattr(runs_route.db, "record_run", lambda *_a, **_k: pytest.fail("run persisted"))
    monkeypatch.setattr(
        runs_route._app, "submit_job", lambda *_a, **_k: pytest.fail("job submitted")
    )
    fields = train_schema_metadata()
    fields["future_knob"] = "0.3.0"
    spec = {**SPEC, "train": {**SPEC["train"], "future_knob": 1}}
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={
            "spec": spec,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.3.0",
                "fields": fields,
                "authored_keys": sorted(spec["train"]),
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unknown key(s): future_knob" in detail
    assert "future_knob (minimum released Flash version 0.3.0)" in detail
    assert "client/server [train] schemas disagree" in detail
    assert calls == 1
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_malformed_schema_metadata_does_not_enrich_parser_rejection(api) -> None:
    spec = {**SPEC, "train": {**SPEC["train"], "future_knob": 1}}
    response = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={
            "spec": spec,
            "dry_run": True,
            "client_train_schema": {
                "version": "0.3.0",
                "fields": {"future_knob": "0.3.0"},
                "authored_keys": ["future_knob", "future_knob"],
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unknown key(s): future_knob" in detail
    assert "minimum released Flash version" not in detail
    assert "schemas disagree" not in detail


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


def test_create_run_rejects_authored_warmstart_rank_before_prepare_or_persist(api, monkeypatch):
    import flash.server.app as app_mod

    calls = {"prepare": 0, "persist": 0}

    def unexpected_prepare(*args, **kwargs):
        calls["prepare"] += 1
        raise AssertionError("prepare_job must not run")

    def unexpected_persist(*args, **kwargs):
        calls["persist"] += 1
        raise AssertionError("record_run must not run")

    monkeypatch.setattr(app_mod, "prepare_job", unexpected_prepare)
    monkeypatch.setattr(app_mod.db, "record_run", unexpected_persist)
    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run",
            "lora_rank": 32,
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    assert (
        resp.json()["detail"]
        == "train.lora_rank cannot be set with train.init_from_adapter because source adapter "
        "rank metadata is authoritative"
    )
    assert calls == {"prepare": 0, "persist": 0}
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_preflights_init_adapter_rank_before_submit(api, monkeypatch):
    import flash.lora_rank as rank_mod
    import flash.runner as runner
    import flash.runner.checkpoints as checkpoints
    from flash.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "Freesolo-Co/source"},
        }
    )
    runner._save_status(runner.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
    monkeypatch.setattr(checkpoints, "adapter_artifact_exists", lambda spec, *, step: True)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 96,
            "lora_alpha": 192,
        },
    )

    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run",
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    assert "source 'source-run' could not be prepared" in resp.text
    assert "rank 96" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_dry_run_still_preflights_init_adapter_rank(api, monkeypatch):
    # A dry-run is a faithful server-side preview: it runs the SAME warm-start rank preflight as a
    # real submit, so a rank-mismatched adapter is rejected at --dry-run (400) instead of being
    # silently accepted and only failing at live submit. A rejected dry-run leaves no run behind.
    import flash.lora_rank as rank_mod
    import flash.runner as runner
    import flash.runner.checkpoints as checkpoints
    from flash.spec import JobSpec

    source = JobSpec.from_dict(
        {
            "run_id": "source-run",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"epochs": 1, "hf_repo": "Freesolo-Co/source"},
        }
    )
    runner._save_status(runner.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
    monkeypatch.setattr(checkpoints, "adapter_artifact_exists", lambda spec, *, step: True)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 96,
            "lora_alpha": 192,
        },
    )

    spec = {
        **SPEC,
        "train": {
            **SPEC["train"],
            "init_from_adapter": "source-run",
        },
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec, "dry_run": True},
    )

    assert resp.status_code == 400
    assert "source 'source-run' could not be prepared" in resp.text
    assert "rank 96" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


def test_create_run_redacts_internal_warmstart_preparation_error(api, monkeypatch):
    import flash.server.app as app_mod

    internal_ref = "private-owner/private-repo:sft/source-run/checkpoints/step-20"
    monkeypatch.setattr(
        app_mod,
        "prepare_job",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"failed to read {internal_ref}")),
    )
    spec = {
        **SPEC,
        "train": {**SPEC["train"], "init_from_adapter": "source-run/step-20"},
    }

    resp = api.post(
        "/v1/runs",
        headers=_bearer("fslo-internal-test"),
        json={"spec": spec},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "source-run/step-20" in detail
    assert "private-owner" not in resp.text
    assert "private-repo" not in resp.text
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-test")).json()["runs"] == []


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


def test_create_run_rejects_top_level_and_gpu_typos_as_400(api):
    key = _login()
    for spec, expected in (
        ({**SPEC, "model_revison": "main"}, "model_revison"),
        ({**SPEC, "gpu": {"exact_typ": "H100"}}, "exact_typ"),
    ):
        response = api.post(
            "/v1/runs",
            json={"spec": spec, "dry_run": True},
            headers=_bearer(key),
        )
        assert response.status_code == 400, response.text
        assert expected in response.json()["detail"]


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


def test_deploy_rejects_revision_pinned_base_model(api):
    import flash.runner as runner

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.spec["model_revision"] = "a" * 40
    runner._save_status(status)

    response = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True},
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "does not support revision-pinned base models" in response.json()["detail"]


def test_deploy_dry_run_does_not_reconcile_unknown_alias(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.deployment = {"state": "failed", "activation_outcome_unknown": True}
    runner._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "adapter_alias_target",
        lambda _run_id: pytest.fail("dry-run deploy must not read the serving alias"),
    )

    dep = api.post(f"/v1/runs/{run_id}/deploy", json={"dry_run": True}, headers=_bearer(key))

    assert dep.status_code == 200, dep.text
    assert dep.json()["state"] == "dry_run"
    assert runner.get_status(run_id).deployment == status.deployment


def test_public_run_routes_redact_private_and_legacy_deployment_fields(api, monkeypatch):
    import flash.runner as runner
    import flash.serve.deploy as deploy_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    revision = f"{run_id}@final." + "a" * 40
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "openai_base_url": "https://serve.example/v1",
        "url": "https://stale.example/v1",
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
        "adapter_revision": revision,
    }
    runner._save_status(status)
    runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=runner.verified_adapter_revision_generation(run_id),
    )

    responses = [
        api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json(),
        api.get("/v1/runs", headers=_bearer(key)).json()["runs"][0],
        api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"][0],
    ]
    for body in responses:
        deployment = body["deployment"]
        assert deployment["openai_base_url"] == "https://serve.example/v1"
        assert "url" not in deployment
        assert "previous_deployment" not in deployment

    persisted = runner.get_status(run_id).deployment
    assert persisted["previous_deployment"]["endpoint_name"] == "https://old.example"
    assert persisted["openai_base_url"] == "https://serve.example/v1"
    assert persisted["url"] == "https://stale.example/v1"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({revision})

    monkeypatch.setattr(deploy_mod, "undeploy_adapter", lambda target: [target])
    cancelled = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["deployment"]["state"] == "undeployed"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()
    assert api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"] == []


def test_deploy_uses_effective_warmstart_rank(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    public_spec = {**status.spec, "train": {**status.spec["train"]}}
    public_spec["train"].update({"init_from_adapter": "source-run", "lora_rank": 8})
    worker_spec = {**public_spec, "train": {**public_spec["train"]}}
    worker_spec["train"].update(
        {
            "init_from_adapter": "private-owner/private-repo:sft/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }
    )
    identity = {"digest": "immutable-v1"}
    status.spec = public_spec
    status.effective_preparation = {
        "worker_spec": worker_spec,
        "adapter_identity": identity,
        "preparation_digest": runner._preparation_digest(
            runner.JobSpec.from_dict(public_spec),
            runner.JobSpec.from_dict(worker_spec),
            identity,
        ),
    }
    runner._save_status(status)
    seen = {}

    def fake_deploy(**kwargs):
        seen.update(kwargs)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"dry_run": True},
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["lora_rank"] == 64
    public = api.get(f"/v1/runs/{run_id}", headers=_bearer(key)).json()
    assert "lora_rank" not in public["spec"]["train"]
    assert public["spec"]["train"]["init_from_adapter"] == "source-run"
    assert "effective_preparation" not in public


def test_deploy_serving_error_is_recorded_as_failed_deployment(api, monkeypatch):
    """A serving-backend failure during deploy is recorded on the deployment status."""
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
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "serving backend unreachable" in resp.json()["error"]
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert deployments[0]["deployment"]["state"] == "failed"
    assert "serving backend unreachable" in deployments[0]["deployment"]["error"]


def test_deploy_returns_deploying_before_background_job_finishes(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    started: dict = {}

    def fake_start(target, **kwargs):
        started.update({"target": target, **kwargs})
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("deploy_adapter must run in the background job"),
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "queued"
    assert resp.json()["verify"] is True
    assert started["run_id"] == run_id
    assert started["deploy_kwargs"]["adapter_prefix"].endswith(run_id)

    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "queued"
    deployments = api.get("/v1/deployments", headers=_bearer(key)).json()["deployments"]
    assert deployments[0]["deployment"]["state"] == "queued"


def test_deploy_rejects_verify_false_before_anything_registers(api, monkeypatch):
    # smoke verification is mandatory: an explicit opt-out is a 400 before queuing, and neither
    # serving registration nor alias activation is ever attempted.
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **kwargs: pytest.fail("verify=false must never reach deploy_adapter"),
    )
    monkeypatch.setattr(
        app_mod,
        "start_deployment_job",
        lambda *args, **kwargs: pytest.fail("verify=false must never queue a deployment"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"verify": False},
        headers=_bearer(key),
    )

    assert resp.status_code == 400, resp.text
    assert "verify=false is not supported" in resp.json()["detail"]
    assert not runner.get_status(run_id).deployment


def test_deploy_rechecks_run_state_before_alias_activation(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    def fake_deploy(**kwargs):
        latest = runner.get_status(run_id)
        latest.state = "cancelled"
        runner._save_status(latest)
        kwargs["before_activate"](f"{run_id}@final." + "a" * 40, run_id)
        pytest.fail("state recheck must block alias activation")

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "run state changed from 'done' to 'cancelled'" in resp.json()["error"]


def test_cancel_while_smoke_is_blocked_prevents_alias_activation(api, monkeypatch):
    import threading

    import flash.runner as runner
    import flash.serve.deploy as deploy_mod
    import flash.server.app as app_mod
    from flash.server.routes import serving

    key = _login()
    run_id = _make_run(api, key, "done")
    previous_revision = f"{run_id}@step-10." + "b" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://old.example",
            "adapter_revision": previous_revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    attempted_revision = f"{run_id}@final." + "a" * 40
    smoke_started = threading.Event()
    release_smoke = threading.Event()
    local_revoked = threading.Event()
    activations = []
    results: dict[str, object] = {}

    def blocked_smoke(*args, **kwargs):
        smoke_started.set()
        if not release_smoke.wait(timeout=5):
            raise TimeoutError("test did not release deployment smoke")
        return {"verified_at": time.time()}

    def fake_deploy(**kwargs):
        kwargs["before_activate"](attempted_revision, run_id)
        activations.append(attempted_revision)
        raise AssertionError("activation must not run after cancellation wins")

    def fake_undeploy(target):
        assert target == run_id
        assert runner.get_status(target).deployment["state"] == "revocation_failed"
        return {"run_id": run_id}

    real_mark_revocation_failed = runner.mark_deployment_revocation_failed

    def mark_pending_then_release(target, error):
        status = real_mark_revocation_failed(target, error)
        if "pending" in error:
            local_revoked.set()
        return status

    monkeypatch.setattr(serving, "_run_deployment_smoke", blocked_smoke)
    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(deploy_mod, "undeploy_adapter", fake_undeploy)
    monkeypatch.setattr(runner, "mark_deployment_revocation_failed", mark_pending_then_release)
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)

    deploy_thread = threading.Thread(
        target=lambda: results.setdefault(
            "deploy",
            api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key)),
        )
    )
    deploy_thread.start()
    assert smoke_started.wait(timeout=5)

    def cancel_target():
        try:
            results["cancel"] = runner.cancel_run(run_id)
        except BaseException as exc:
            results["cancel_error"] = exc

    cancel_thread = threading.Thread(target=cancel_target)
    cancel_thread.start()
    assert local_revoked.wait(timeout=5)
    release_smoke.set()
    deploy_thread.join(timeout=5)
    cancel_thread.join(timeout=5)

    assert not deploy_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert "cancel_error" not in results
    assert activations == []
    assert results["deploy"].status_code == 200
    final = runner.get_status(run_id)
    assert final.state == "cancelled"
    assert final.deployment["state"] == "undeployed"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


@pytest.mark.parametrize("attempt_kind", ["final", "checkpoint"])
def test_contended_cancel_revokes_activation_completed_after_predecessor_restore(
    api, monkeypatch, attempt_kind
):
    import threading

    import flash.runner as runner
    import flash.serve.deploy as deploy_mod
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment
    from flash.server.routes import serving

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    key = _login()
    initial_state = "done" if attempt_kind == "final" else "running"
    run_id = _make_run(api, key, initial_state)
    previous_revision = f"{run_id}@step-40." + "b" * 40
    previous = {
        "state": "ready",
        "endpoint_name": "https://old.example",
        "adapter_revision": previous_revision,
        "checkpoint_step": 40,
    }
    runner.mark_checkpoint_deployed(
        run_id,
        previous,
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    initial_generation = runner.verified_adapter_revision_generation(run_id)
    if attempt_kind == "final":
        attempted_revision = f"{run_id}@final." + "a" * 40
        expected_checkpoint = run_id
        payload = {}
        checkpoint_step = None
    else:
        attempted_revision = f"{run_id}@step-80." + "a" * 40
        expected_checkpoint = f"{run_id}/step-80"
        payload = {"step": 80}
        checkpoint_step = 80

    smoke_started = threading.Event()
    release_smoke = threading.Event()
    cancellation_snapshotted = threading.Event()
    activation_started = threading.Event()
    predecessor_restored = threading.Event()
    release_activation = threading.Event()
    live_alias = {"target": previous_revision}
    alias_reads = []
    undeploys = []
    results: dict[str, object] = {}

    def blocked_smoke(*args, **kwargs):
        smoke_started.set()
        if not release_smoke.wait(timeout=5):
            raise TimeoutError("test did not release deployment smoke")
        return {
            "verified_at": time.time(),
            "verify_kind": "fixed_prompt",
            "verify_turns": 1,
            "verify_latency_s": 0.1,
            "verify_finish_reason": "stop",
            "thinking_tag": False,
            "verify_sample": "4",
        }

    def fake_deploy(**kwargs):
        kwargs["before_activate"](attempted_revision, expected_checkpoint)
        activation_started.set()
        if not release_activation.wait(timeout=5):
            raise TimeoutError("test did not release alias activation")
        live_alias["target"] = attempted_revision
        return Deployment(
            run_id=run_id,
            model=SPEC["model"],
            adapter_hf_prefix=f"{kwargs['adapter_prefix']}/adapter",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            adapter_revision=attempted_revision,
            checkpoint_step=checkpoint_step,
        )

    real_mark_revocation_failed = runner.mark_deployment_revocation_failed

    def fence_after_activation_started(target, error):
        if "in-progress deployment" in error:
            cancellation_snapshotted.set()
            release_smoke.set()
            if not activation_started.wait(timeout=5):
                raise TimeoutError("worker did not cross the final activation fence")
        return real_mark_revocation_failed(target, error)

    real_mark_checkpoint_deployed = runner.mark_checkpoint_deployed

    def observe_restore(*args, **kwargs):
        status = real_mark_checkpoint_deployed(*args, **kwargs)
        owner = kwargs.get("owner_deployment")
        if isinstance(owner, dict) and owner.get("state") == "revocation_failed":
            assert kwargs["verification_generation"] == initial_generation + 1
            if status.deployment == previous:
                predecessor_restored.set()
        return status

    def alias_target(target):
        alias_reads.append(target)
        return live_alias["target"]

    monkeypatch.setattr(serving, "_run_deployment_smoke", blocked_smoke)
    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(runner, "mark_deployment_revocation_failed", fence_after_activation_started)
    monkeypatch.setattr(runner, "mark_checkpoint_deployed", observe_restore)
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy_mod, "adapter_alias_target", alias_target)
    monkeypatch.setattr(deploy_mod, "undeploy_adapter", lambda target: undeploys.append(target))

    deploy_thread = threading.Thread(
        target=lambda: results.setdefault(
            "deploy",
            api.post(
                f"/v1/runs/{run_id}/deploy",
                json=payload,
                headers=_bearer(key),
            ),
        )
    )
    deploy_thread.start()
    assert smoke_started.wait(timeout=5)

    def cancel_target():
        try:
            results["cancel"] = runner.cancel_run(run_id)
        except BaseException as exc:
            results["cancel_error"] = exc

    cancel_thread = threading.Thread(target=cancel_target)
    cancel_thread.start()
    assert cancellation_snapshotted.wait(timeout=5)
    assert predecessor_restored.wait(timeout=5)
    assert runner.verified_adapter_revision_generation(run_id) == initial_generation + 1
    assert runner.get_status(run_id).deployment == previous
    release_activation.set()
    deploy_thread.join(timeout=5)
    cancel_thread.join(timeout=5)

    assert not deploy_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert "cancel_error" not in results
    assert results["deploy"].status_code == 200
    assert alias_reads == [run_id]
    assert undeploys == [run_id]
    final = runner.get_status(run_id)
    assert final.state == "cancelled"
    assert final.deployment["state"] == "undeployed"
    assert final.deployment.get("adapter_revision") != attempted_revision
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


def test_cancel_local_persistence_failure_returns_structured_retryable_error(api, monkeypatch):
    import flash.runner as runner
    from flash.runner.deploy import DeploymentStatePersistenceError

    assert runner.DeploymentStatePersistenceError is DeploymentStatePersistenceError
    key = _login()
    run_id = _make_run(api, key, "running")
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    real_mark_undeployed = runner.mark_deployment_undeployed
    attempts = []

    def fail_once(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise OSError("generation store unavailable")
        return real_mark_undeployed(target)

    monkeypatch.setattr(runner, "mark_deployment_undeployed", fail_once)

    response = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "deployment_state_persistence_failed"
    assert detail["run_id"] == run_id
    assert detail["retryable"] is True
    assert detail["backend_outcome"] == "not_required"
    assert "backend revocation was not required" in detail["message"]
    assert runner.get_status(run_id).state == "running"

    retried = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert retried.status_code == 200
    assert attempts == [run_id, run_id]
    assert runner.get_status(run_id).state == "cancelled"


def test_cancel_double_undeploy_failure_returns_structured_retryable_error(api, monkeypatch):
    import flash.runner as runner
    import flash.serve.deploy as deploy_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    attempts = []

    def fail_undeploy(target):
        attempts.append(target)
        raise deploy_mod.ServingError("backend unavailable")

    monkeypatch.setattr(deploy_mod, "undeploy_adapter", fail_undeploy)
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: None)

    response = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["code"] == "deployment_revocation_failed"
    assert detail["run_id"] == run_id
    assert detail["retryable"] is True
    assert "backend unavailable" in detail["message"]
    assert attempts == [run_id]
    status = runner.get_status(run_id)
    assert status.state == "cancelled"
    assert status.deployment["state"] == "revocation_failed"
    assert status.deployment["retryable"] is True
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


def test_deploy_recovers_ambiguous_ready_persistence_after_activation(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment
    from flash.server.routes import serving

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        return Deployment(
            run_id=run_id,
            model=SPEC["model"],
            adapter_hf_prefix=f"{kwargs['adapter_prefix']}/adapter",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            adapter_revision=revision,
        )

    original_mark_deployed = serving.mark_deployed
    calls = {"count": 0}

    def persist_then_raise(*args, **kwargs):
        calls["count"] += 1
        result = original_mark_deployed(*args, **kwargs)
        raise OSError("status write acknowledgement lost")

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(serving, "mark_deployed", persist_then_raise)
    monkeypatch.setattr(
        app_mod, "serve_chat", lambda **kwargs: _smoke_chat_result(revision, run_id)
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert calls["count"] == 1
    assert resp.json()["state"] == "ready"
    assert resp.json()["adapter_revision"] == revision
    assert runner.get_status(run_id).state == "deployed"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({revision})


def test_commit_miss_with_same_attempt_retries_and_persists_ready(api, monkeypatch):
    # the run state moves under the cas guard (e.g. done -> deployed by a sibling write) while
    # this attempt still owns the deployment record: the ready commit must be retried against the
    # fresh state, never dropped silently after the serving alias already flipped.
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        # the guard was captured at state "done"; move the run so the first cas write misses
        latest = runner.get_status(run_id)
        latest.state = "deployed"
        runner._save_status(latest)
        return Deployment(
            run_id=run_id,
            model=SPEC["model"],
            adapter_hf_prefix=f"{kwargs['adapter_prefix']}/adapter",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            adapter_revision=revision,
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod, "serve_chat", lambda **kwargs: _smoke_chat_result(revision, run_id)
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "ready"
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "ready"
    assert deployment["adapter_revision"] == revision


def test_commit_miss_superseded_records_divergence_without_alias_revert(api, monkeypatch):
    # a newer actor (undeploy) took the record during activation: the lost commit must be
    # recorded as a divergence rather than dropped, and the serving alias must NOT be reverted
    # (post-promotion recovery reads the authoritative alias).
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import Deployment

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        # a concurrent undeploy supersedes the record after activation
        latest = runner.get_status(run_id)
        latest.state = "cancelled"
        latest.deployment = {**(latest.deployment or {}), "state": "undeployed"}
        runner._save_status(latest)
        return Deployment(
            run_id=run_id,
            model=SPEC["model"],
            adapter_hf_prefix=f"{kwargs['adapter_prefix']}/adapter",
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
            adapter_revision=revision,
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod, "serve_chat", lambda **kwargs: _smoke_chat_result(revision, run_id)
    )
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda *args, **kwargs: pytest.fail("commit reconciliation must never revert the alias"),
    )
    divergences = []
    printed = print

    def capture_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if "deployment_record_diverged" in text:
            divergences.append(text)
        printed(*args, **kwargs)

    import builtins

    monkeypatch.setattr(builtins, "print", capture_print)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert divergences, "lost commit must be logged as a divergence"
    # the newer actor's record is preserved: undeploy wrote "undeployed" and it stays
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "undeployed"


def test_post_activation_recovery_failure_logs_divergence(api, monkeypatch):
    import builtins

    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40

    class BrokenDeployment:
        def to_dict(self):
            raise RuntimeError("serialization failed after activation")

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        return BrokenDeployment()

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod, "serve_chat", lambda **kwargs: _smoke_chat_result(revision, run_id)
    )
    divergences = []
    real_print = print

    def capture_print(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        if "deployment_record_diverged" in text:
            divergences.append(text)
        real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", capture_print)

    response = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert response.status_code == 200, response.text
    assert divergences
    assert "ready-state recovery failed" in divergences[0]


def test_deploy_ignores_legacy_spec_gpu(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    legacy_spec = {**SPEC, "gpu": {"type": "RTX A6000"}}
    run_id = api.post(
        "/v1/runs", json={"spec": legacy_spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.remote = {"provider": "runpod", "allocated_gpu": "RTX 5090"}
    runner._save_status(status)

    seen: dict = {}

    def fake_start(target, **kwargs):
        seen.update({"target": target, **kwargs})
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "queued"
    assert "gpu" not in resp.json()
    assert "gpu_name" not in seen["deploy_kwargs"]


def test_deploy_forwards_structured_outputs_to_serving(api, monkeypatch):
    """The deploy route hands the run's [train].structured_outputs to deploy_adapter so serving can
    register it as the adapter's guided-decoding default (guided-decoding train/serve parity)."""
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    schema = {"type": "object", "properties": {"industries": {"type": "array"}}}
    so_spec = {**SPEC, "train": {**SPEC["train"], "structured_outputs": {"json": schema}}}
    run_id = api.post(
        "/v1/runs", json={"spec": so_spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    seen: dict = {}

    def fake_start(target, **kwargs):
        seen.update({"target": target, **kwargs})
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    forwarded = seen["deploy_kwargs"]["structured_outputs"]
    assert json.loads(forwarded) == {"json": schema}


def test_thinking_structured_deploy_rejects_verify_false_before_mutation(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    schema = {"type": "object", "required": ["answer"]}
    spec = {
        **SPEC,
        "thinking": True,
        "train": {**SPEC["train"], "structured_outputs": {"json": schema}},
    }
    run_id = api.post(
        "/v1/runs", json={"spec": spec, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    monkeypatch.setattr(
        app_mod,
        "start_deployment_job",
        lambda *_a, **_k: pytest.fail("deployment must not be queued"),
    )
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: pytest.fail("serving mutation must not be attempted"),
    )

    resp = api.post(
        f"/v1/runs/{run_id}/deploy",
        json={"verify": False},
        headers=_bearer(key),
    )

    assert resp.status_code == 400, resp.text
    assert "verify=false is not supported" in resp.json()["detail"]
    assert runner.get_status(run_id).deployment is None


def test_deploy_retry_takes_over_stale_busy_record(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.deployment = {"state": "deploying", "updated_at": 0.0, "requested_at": 0.0}
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, run_id),
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "ready"


def test_failed_smoke_revision_cannot_be_exact_chatted(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "d" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, run_id)
        pytest.fail("failed smoke must block alias activation")

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: (_ for _ in ()).throw(ServingError("smoke generation failed")),
    )

    deployment = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert deployment.status_code == 200, deployment.text
    assert deployment.json()["state"] == "failed"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()

    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("unverified revision must not reach serving"),
    )
    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": revision,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "has not passed a successful deployment smoke" in response.json()["detail"]


def test_failed_redeploy_restores_previous_ready_deployment(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = _make_run(api, key, "deployed")
    previous = {"state": "ready", "endpoint_name": "old", "adapter_hf_prefix": "rl/old/adapter"}
    status = runner.get_status(run_id)
    status.deployment = previous
    runner._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_k: (_ for _ in ()).throw(ServingError("new adapter failed smoke")),
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "ready"
    assert deployment["endpoint_name"] == "old"
    assert deployment["last_deploy_error"] == "new adapter failed smoke"
    assert resp.json()["endpoint_name"] == "old"


@pytest.mark.parametrize("deployment_state", ["undeployed", "revocation_failed"])
def test_redeploy_after_inactive_deployment_state_is_allowed(api, monkeypatch, deployment_state):
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = _make_run(api, key, "done")
    status = runner.get_status(run_id)
    status.deployment = {"state": deployment_state, "requested_at": 1.0}
    runner._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "deploy_adapter",
        lambda **_kwargs: (_ for _ in ()).throw(ServingError("new adapter failed smoke")),
    )

    response = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert response.status_code == 200, response.text
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "failed"
    assert deployment["requested_at"] != 1.0


def test_activation_unknown_preserves_previous_revision_for_retry_cas(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ActivationOutcomeUnknown, ServingError

    key = _login()
    run_id = _make_run(api, key, "done")
    previous_revision = f"{run_id}@step-10." + "b" * 40
    previous = {
        "state": "ready",
        "endpoint_name": "https://old.example",
        "adapter_revision": previous_revision,
    }
    runner.mark_deployed(
        run_id,
        previous,
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    attempted_revision = f"{run_id}@final." + "a" * 40
    expected_revisions = []
    alias_reads = []

    def fake_alias_target(alias_run_id):
        alias_reads.append(alias_run_id)
        return attempted_revision

    def fake_deploy(**kwargs):
        expected_revisions.append(kwargs["expected_adapter_revision"])
        if len(expected_revisions) == 2:
            retry_record = runner.get_status(run_id).deployment
            assert retry_record["previous_deployment"]["adapter_revision"] == attempted_revision
            assert retry_record["previous_deployment"]["state"] == "reconciling"
            raise ServingError("retry failed before alias activation")
        kwargs["before_activate"](attempted_revision, run_id)
        activating = runner.get_status(run_id).deployment
        assert activating["state"] == "reconciling"
        assert activating["activation_outcome_unknown"] is True
        if len(expected_revisions) == 1:
            raise ActivationOutcomeUnknown(run_id, attempted_revision)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "adapter_alias_target", fake_alias_target)
    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(attempted_revision, run_id),
    )

    response = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "reconciling"
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "reconciling"
    assert deployment["adapter_revision"] == attempted_revision
    assert deployment["activation_outcome_unknown"] is True
    assert deployment["previous_deployment"] == previous
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({previous_revision})

    retry = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert retry.status_code == 200, retry.text
    assert retry.json()["state"] == "failed"
    retry_failed = runner.get_status(run_id).deployment
    assert retry_failed["state"] == "failed"
    assert retry_failed["adapter_revision"] is None
    assert retry_failed["activation_outcome_unknown"] is True
    assert retry_failed["previous_deployment"]["adapter_revision"] == attempted_revision
    assert retry_failed["previous_deployment"]["state"] == "reconciling"

    final_retry = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert final_retry.status_code == 200, final_retry.text
    assert final_retry.json()["state"] == "ready"
    assert alias_reads == [run_id, run_id]
    assert expected_revisions == [previous_revision, attempted_revision, attempted_revision]


@pytest.mark.parametrize("retry_state", ["queued", "smoke_testing"])
def test_unknown_reconciliation_allows_one_retry_then_blocks_overlap(api, monkeypatch, retry_state):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    status = runner.get_status(run_id)
    status.deployment = {
        "state": "reconciling",
        "requested_at": time.time(),
        "activation_outcome_unknown": True,
    }
    runner._save_status(status)
    alias_reads = []
    jobs = []
    monkeypatch.setattr(
        app_mod,
        "adapter_alias_target",
        lambda target: alias_reads.append(target) or None,
    )
    monkeypatch.setattr(
        app_mod,
        "start_deployment_job",
        lambda target, **kwargs: jobs.append((target, kwargs)) or False,
    )

    first = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert first.status_code == 200, first.text
    queued = runner.get_status(run_id)
    assert queued.deployment["state"] == "queued"
    assert queued.deployment["activation_outcome_unknown"] is True
    if retry_state == "smoke_testing":
        queued.deployment["state"] = retry_state
        queued.deployment["updated_at"] = time.time()
        runner._save_status(queued)

    second = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert second.status_code == 409, second.text
    assert f"deployment in {retry_state} state" in second.json()["detail"]
    assert alias_reads == [run_id]
    assert len(jobs) == 1


def test_activation_unknown_synthetic_checkpoint_predecessor_survives_cancel(api, monkeypatch):
    import flash.runner as runner
    import flash.serve.deploy as deploy
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    attempted_revision = f"{run_id}@final." + "a" * 40
    stale_revision = f"{run_id}@step-10." + "b" * 40
    live_revision = f"{run_id}@step-20." + "c" * 40
    status = runner.get_status(run_id)
    status.deployment = {
        "state": "reconciling",
        "adapter_revision": attempted_revision,
        "activation_outcome_unknown": True,
        "previous_deployment": {
            "state": "ready",
            "adapter_revision": stale_revision,
            "checkpoint_step": 10,
        },
    }
    runner._save_status(status)
    runner.add_verified_adapter_revision(
        run_id,
        live_revision,
        expected_generation=runner.verified_adapter_revision_generation(run_id),
    )

    monkeypatch.setattr(app_mod, "adapter_alias_target", lambda _run_id: live_revision)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _run_id: live_revision)

    def fail_before_activation(**kwargs):
        assert kwargs["expected_adapter_revision"] == live_revision
        queued = runner.get_status(run_id).deployment
        predecessor = queued["previous_deployment"]
        assert predecessor == {
            "run_id": run_id,
            "adapter_revision": live_revision,
            "checkpoint_step": 20,
            "openai_model": run_id,
            "state": "ready",
        }
        raise deploy.ServingError("retry failed before alias activation")

    monkeypatch.setattr(app_mod, "deploy_adapter", fail_before_activation)

    retry = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert retry.status_code == 200, retry.text
    failed = runner.get_status(run_id).deployment
    assert failed["state"] == "failed"
    assert failed["activation_outcome_unknown"] is True
    assert failed["previous_deployment"]["checkpoint_step"] == 20

    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the verified live checkpoint must remain serving"),
    )

    cancelled = runner.cancel_run(run_id)

    assert cancelled.state == "cancelled"
    assert cancelled.deployment["state"] == "ready"
    assert cancelled.deployment["adapter_revision"] == live_revision
    assert cancelled.deployment["checkpoint_step"] == 20
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({live_revision})


def test_cancel_restores_owned_previous_checkpoint_and_bare_chat_authority(api, monkeypatch):
    import flash.runner as runner
    import flash.serve.deploy as deploy
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "running")
    previous_revision = f"{run_id}@step-40." + "d" * 40
    previous = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": previous_revision,
        "checkpoint_step": 40,
    }
    runner.mark_checkpoint_deployed(
        run_id,
        previous,
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    previous["requested_at"] = time.time() - 60
    status = runner.get_status(run_id)
    busy = {
        "state": "reconciling",
        "requested_at": time.time(),
        "adapter_revision": f"{run_id}@final." + "e" * 40,
        "activation_outcome_unknown": True,
        "previous_deployment": previous,
    }
    status.deployment = busy
    runner._save_status(status)
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(deploy, "adapter_alias_target", lambda _run_id: previous_revision)
    monkeypatch.setattr(
        deploy,
        "undeploy_adapter",
        lambda _run_id: pytest.fail("the restored checkpoint must remain serving"),
    )
    served = []

    def serve_chat(**kwargs):
        served.append(kwargs["run_id"])
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(app_mod, "serve_chat", serve_chat)

    cancelled = api.post(f"/v1/runs/{run_id}/cancel", headers=_bearer(key))

    assert cancelled.status_code == 200, cancelled.text
    restored = runner.get_status(run_id)
    assert restored.state == "cancelled"
    assert restored.deployment == previous
    assert restored.deployment != busy

    chat = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert chat.status_code == 200, chat.text
    assert served == [run_id]
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({previous_revision})


def test_failed_redeploy_after_registration_restores_previous_serving(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    previous = {"state": "ready", "endpoint_name": "old", "adapter_hf_prefix": "rl/old/adapter"}
    status = runner.get_status(run_id)
    status.deployment = previous
    runner._save_status(status)

    registered_prefixes = []

    def fake_deploy(**kwargs):
        registered_prefixes.append(kwargs["adapter_prefix"])
        kwargs["before_activate"]("immutable@final." + "a" * 40, run_id)
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **_k: {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
    )

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))

    assert resp.status_code == 200, resp.text
    assert registered_prefixes == [f"rl/{run_id}"]
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "ready"
    assert deployment["endpoint_name"] == "old"
    assert "smoke generation returned no content" in deployment["last_deploy_error"]


def test_deploy_ignores_stored_training_gpu(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.spec["gpu"]["type"] = "H200"
    status.effective_preparation = None
    runner._save_status(status)
    seen: dict = {}

    def fake_start(target, **kwargs):
        seen.update({"target": target, **kwargs})
        return False

    monkeypatch.setattr(app_mod, "start_deployment_job", fake_start)

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "queued"
    assert "gpu" not in resp.json()
    assert "gpu_name" not in seen["deploy_kwargs"]


def test_deploy_missing_run_level_adapter_points_at_checkpoint_steps(api, monkeypatch):
    """A run whose finalize never published the run-level <prefix>/adapter (but which streamed
    per-step deployable checkpoints) must not fail run-level deploy with an opaque 502 rank
    error: it returns a 409 telling the caller to `flash deploy <run>/step-N`."""
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import AdapterConfigMissing

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    def boom(**kwargs):
        raise AdapterConfigMissing(
            "could not verify adapter rank: failed to read org/repo:rl/x/adapter/adapter_config.json"
        )

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)
    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: [{"step": 10}, {"step": 40}])

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    detail = resp.json()["error"]
    assert "no run-level adapter" in detail
    assert f"flash deploy {run_id}/step-40" in detail
    assert "10, 40" in detail


def test_deploy_missing_adapter_without_checkpoints_stays_502(api, monkeypatch):
    """No checkpoints to point at -> keep the 502 with the upstream reason."""
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import AdapterConfigMissing

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)

    def boom(**kwargs):
        raise AdapterConfigMissing("could not verify adapter rank: failed to read org/repo:x")

    monkeypatch.setattr(app_mod, "deploy_adapter", boom)
    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: [])

    resp = api.post(f"/v1/runs/{run_id}/deploy", json={}, headers=_bearer(key))
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "failed"
    assert "failed to read" in resp.json()["error"]


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
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
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
            openai_model=run_id,
            endpoint_name="https://serve.example",
            openai_base_url="https://serve.example/v1",
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
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )

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


def test_chat_streams_verified_immutable_revision_unchanged(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}@final." + "a" * 40
    status = runner.get_status(run_id)
    status.state = "deployed"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": revision,
    }
    runner._save_status(status)
    generation = runner.verified_adapter_revision_generation(run_id)
    assert runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=generation,
    )
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield "verified"

    monkeypatch.setattr(app_mod, "serve_chat_stream", fake_stream)

    with api.stream(
        "POST",
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": revision,
            "stream": True,
        },
        headers=_bearer(key),
    ) as response:
        text = response.read().decode()

    assert response.status_code == 200, text
    assert text == "verified"
    assert seen["run_id"] == revision


def test_chat_ready_record_without_ledger_membership_rejects_revision(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}@final." + "b" * 40
    status = runner.get_status(run_id)
    status.state = "deployed"
    status.deployment = {
        "state": "ready",
        "endpoint_name": "https://serve.example",
        "adapter_revision": revision,
    }
    runner._save_status(status)

    assert "verification_generation" not in runner.get_status(run_id).deployment
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("unverified revision must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": revision,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "has not passed a successful deployment smoke" in response.json()["detail"]


def test_chat_bare_alias_rejects_status_only_ready_record(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "deployed")
    status = runner.get_status(run_id)
    status.deployment = {"state": "ready", "endpoint_name": "https://serve.example"}
    runner._save_status(status)
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("status-only ready record must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "no active deployment" in response.json()["detail"]


def test_chat_reconciling_alias_rejects_bare_and_allows_verified_revision(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    previous_revision = f"{run_id}@step-10." + "b" * 40
    previous = {
        "state": "ready",
        "endpoint_name": "https://old.example",
        "adapter_revision": previous_revision,
    }
    runner.mark_deployed(
        run_id,
        previous,
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    runner.mark_deployment_pending(
        run_id,
        {
            "state": "reconciling",
            "requested_at": time.time(),
            "updated_at": time.time(),
            "activation_outcome_unknown": True,
            "previous_deployment": previous,
        },
    )
    served_revisions = []

    def serve_chat(**kwargs):
        served_revisions.append(kwargs["run_id"])
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(app_mod, "serve_chat", serve_chat)

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "deployment is reconciling" in response.json()["detail"]
    assert served_revisions == []

    explicit = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": previous_revision,
        },
        headers=_bearer(key),
    )

    assert explicit.status_code == 200, explicit.text
    assert served_revisions == [previous_revision]
    assert runner.read_verified_adapter_revisions(run_id) == frozenset({previous_revision})


def test_chat_selects_immutable_revisions_independently(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revisions = [f"{run_id}@step-20." + "a" * 40, f"{run_id}@step-40." + "b" * 40]
    for revision in revisions:
        runner.mark_checkpoint_deployed(
            run_id,
            {
                "state": "ready",
                "endpoint_name": "https://serve.example",
                "adapter_revision": revision,
            },
            verification_generation=runner.verified_adapter_revision_generation(run_id),
        )
    assert runner.read_verified_adapter_revisions(run_id) == frozenset(revisions)
    seen = []

    def fake_chat(**kwargs):
        seen.append(kwargs["run_id"])
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    for revision in revisions:
        response = api.post(
            f"/v1/runs/{run_id}/chat",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "adapter_revision": revision,
            },
            headers=_bearer(key),
        )
        assert response.status_code == 200, response.text

    assert seen == revisions


def test_chat_rejects_cross_run_immutable_revision(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("cross-run revision must not reach serving"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "adapter_revision": "another-run@final." + "c" * 40,
        },
        headers=_bearer(key),
    )

    assert response.status_code == 400
    assert "belongs to run another-run" in response.json()["detail"]


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
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )

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


def test_chat_forwards_user_supplied_system_prompt(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    runner._save_status(status)
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )

    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(app_mod, "serve_chat", fake_chat)

    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hello"},
    ]
    resp = api.post(
        f"/v1/runs/{run_id}/chat",
        json={
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True},
            "enable_thinking": True,
        },
        headers=_bearer(key),
    )

    assert resp.status_code == 200, resp.text
    assert seen["messages"] == messages
    assert seen["thinking"] is False
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
    status.state = "cancelled"
    runner._save_status(status)
    revision = f"{run_id}@step-40." + "a" * 40
    runner.mark_checkpoint_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )

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


def test_chat_rejects_undeployed_record_with_previous_ready_deployment(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    status = runner.get_status(run_id)
    status.state = "done"
    status.deployment = {
        "state": "undeployed",
        "previous_deployment": {"state": "ready", "endpoint_name": "https://old.example"},
    }
    runner._save_status(status)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: pytest.fail("undeployed aliases must never be served"),
    )

    response = api.post(
        f"/v1/runs/{run_id}/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=_bearer(key),
    )

    assert response.status_code == 409
    assert "no active deployment" in response.json()["detail"]


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
    revision = f"{run_id}@final." + "a" * 40
    runner.mark_deployed(
        run_id,
        {
            "state": "ready",
            "endpoint_name": "https://serve.example",
            "adapter_revision": revision,
        },
        verification_generation=runner.verified_adapter_revision_generation(run_id),
    )
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
    import flash.runner as runner
    import flash.server.app as app_mod
    from flash.serve.deploy import ServingError

    key = _login()
    run_id = api.post(
        "/v1/runs", json={"spec": SPEC, "dry_run": True}, headers=_bearer(key)
    ).json()["run_id"]
    revision = f"{run_id}@final." + "e" * 40
    runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=runner.verified_adapter_revision_generation(run_id),
    )

    def boom(_run_id):
        raise ServingError("serving backend unreachable: could not delete endpoint")

    monkeypatch.setattr(app_mod, "undeploy_adapter", boom)

    resp = api.delete(f"/v1/runs/{run_id}/deploy", headers=_bearer(key))
    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "deployment_revocation_failed"
    assert detail["retryable"] is True
    assert "serving backend unreachable" in detail["message"]
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()
    deployment = runner.get_status(run_id).deployment
    assert deployment["state"] == "revocation_failed"
    assert deployment["retryable"] is True


def test_undeploy_without_status_projection_invalidates_orphaned_ledger(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}@final." + "f" * 40
    generation = runner.verified_adapter_revision_generation(run_id)
    assert runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=generation,
    )
    assert runner.get_status(run_id).deployment is None
    monkeypatch.setattr(
        app_mod,
        "undeploy_adapter",
        lambda target: {"run_id": target, "serving_deregistered": False},
    )

    response = api.delete(f"/v1/runs/{run_id}/deploy", headers=_bearer(key))

    assert response.status_code == 200, response.text
    assert runner.verified_adapter_revision_generation(run_id) == generation + 1
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()
    assert runner.get_status(run_id).deployment is None


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
    deployment = {
        "state": "ready",
        "endpoint_name": "e",
        "adapter_revision": "dep-1@final." + "a" * 40,
    }
    out = runner.mark_deployed(
        "dep-1",
        deployment,
        verification_generation=runner.verified_adapter_revision_generation("dep-1"),
    )
    assert out.state == "deployed"
    assert out.deployment == deployment

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
    out = runner.mark_deployed(
        "dep-leg",
        {
            "state": "ready",
            "endpoint_name": "e",
            "adapter_revision": "dep-leg@final." + "a" * 40,
        },
        verification_generation=runner.verified_adapter_revision_generation("dep-leg"),
    )
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
    out2 = runner.mark_deployed(
        "dep-leg2",
        {
            "state": "ready",
            "endpoint_name": "e2",
            "adapter_revision": "dep-leg2@final." + "b" * 40,
        },
        expect_state="deployed",
        verification_generation=runner.verified_adapter_revision_generation("dep-leg2"),
    )
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
    out3 = runner.mark_deployed(
        "dep-leg3",
        {
            "state": "ready",
            "endpoint_name": "e3",
            "adapter_revision": "dep-leg3@final." + "c" * 40,
        },
        verification_generation=runner.verified_adapter_revision_generation("dep-leg3"),
    )
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
        "train": {"epochs": 1, "max_examples": 1},
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
    assert reaped == ["nohandle-1"], (
        "must force-reap the run's instance-provider label before resubmit"
    )
    # The resubmit GC's the orphaned endpoint and re-runs the job; the run is NOT failed.
    assert runner.get_status("nohandle-1").state != "failed"


def test_recover_runs_drains_private_cleanup_for_terminal_run(monkeypatch, tmp_path):
    import flash.providers as providers_mod
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)
    run_id = "terminal-cleanup-recovery"
    remote = {
        "provider": "runpod",
        "endpoint_id": "endpoint-cleanup",
        "job_id": "job-cleanup",
        "attempt": 1,
    }
    runner._save_status(
        runner.RunStatus(run_id=run_id, state="cancelled", spec={"run_id": run_id}),
        _cleanup_remotes=[remote],
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": run_id}])
    drained = []
    monkeypatch.setattr(runner, "_drain_cleanup_remotes", lambda rid: drained.append(rid) or set())
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [])

    app_mod.recover_runs()

    assert drained == [run_id]



def test_recover_runs_blocks_expired_handleless_resubmit(monkeypatch, tmp_path):
    import flash.providers as providers_mod
    import flash.runner as runner
    import flash.server.db as db_mod
    from flash.spec import GpuSpec, JobSpec

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)
    spec = JobSpec(
        run_id="blocked-expired",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    created_at = 100.0
    deadline = created_at + float(spec.gpu.max_wall_seconds)
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=created_at,
        ),
        _run_deadline_at=deadline,
        _next_attempt=0,
    )
    monkeypatch.setattr(runner.time, "time", lambda: deadline + 1.0)
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    submitted = []
    monkeypatch.setattr(
        runner, "_run_job_background", lambda recovered: submitted.append(recovered.run_id)
    )
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [])

    app_mod.recover_runs()

    recovered = runner.get_status(spec.run_id)
    assert submitted == []
    assert recovered.state == "failed"
    assert "deadline exhausted" in recovered.error


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
        "train": {"epochs": 1, "max_examples": 1},
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
    assert runner.get_status("phantom-1").state != "failed", (
        "deferred, not failed (later recovery retries)"
    )


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
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "unconf-1",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="unconf-1",
            state="provisioning",
            spec=spec,
            remote=None,
            submitted_instance_providers=[
                "vast"
            ],  # Vast was configured when this run was submitted
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
    assert runner.get_status("unconf-1").state != "failed", (
        "deferred, not failed (later restart retries)"
    )


def test_recover_runs_resubmits_queued_run_despite_unconfigurable_vast(monkeypatch, tmp_path):
    # Codex: a run still `queued` never reached lifecycle's `provisioning` transition, so no provider
    # create (Vast's non-idempotent PUT /asks) was ever attempted and no phantom can exist. The phantom
    # guard (_confirm_run_clear) must be SKIPPED for it — otherwise a purely-queued run whose VAST_API_KEY
    # was dropped after submit fails closed on the unenumerable recorded Vast and defers forever. This is
    # identical to test_recover_runs_defers_when_recorded_provider_unconfigurable EXCEPT the state is
    # `queued` (never created) instead of `provisioning` (could have created) -> resubmit, not defer.
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
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "queued-1",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="queued-1",
            state="queued",  # never provisioned -> no create attempted -> no phantom possible
            spec=spec,
            remote=None,
            submitted_instance_providers=["vast"],  # Vast configured at submit, creds now gone
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "queued-1"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)

    resubmitted = []
    done = threading.Event()

    def fake_run_job(s):
        resubmitted.append(s.run_id)
        done.set()

    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    import flash.providers as providers_mod
    import flash.server._runtime as rt

    # Vast unconfigurable now: the OLD unconditional guard would fail closed here and defer forever. A
    # queued run must resubmit anyway, because it provably never created the phantom the guard protects
    # against. _confirm_run_clear must not even be consulted (Vast enumeration would raise/defer).
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [])
    monkeypatch.setattr(rt, "_DEFERRED_RECOVERY_MAX_RETRIES", 0)

    app_mod.recover_runs()

    assert done.wait(timeout=5), (
        "queued run must launch a resubmit thread, not defer on a phantom check"
    )
    assert resubmitted == ["queued-1"], (
        "a never-provisioned queued run resubmits despite unconfigurable Vast"
    )
    assert runner.get_status("queued-1").state != "failed"


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
        "train": {"epochs": 1, "max_examples": 1},
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

    assert done.wait(timeout=5), (
        "a run that never recorded Vast must still recover on a Vast-less plane"
    )
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
        "train": {"epochs": 1, "max_examples": 1},
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
        "train": {"epochs": 1, "max_examples": 1},
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
    monkeypatch.setattr(runner, "_run_job", lambda s: (resubmitted.append(s.run_id), done.set()))
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

    assert done.wait(timeout=5), (
        "the background retry must resubmit once the run is confirmed clear"
    )
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
        "train": {"epochs": 1, "max_examples": 1},
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


def test_recover_runs_reuses_verified_effective_snapshot_for_no_handle_resubmit(
    monkeypatch, tmp_path
):
    import threading

    import flash.lora_rank as rank_mod
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)

    public_spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {
            "epochs": 1,
            "max_examples": 1,
            "init_from_adapter": "source-run",
            "lora_rank": 8,
        },
        "gpu": {"type": "RTX 5090"},
        "run_id": "nohandle-warm",
    }
    worker_spec = {
        **public_spec,
        "train": {
            **public_spec["train"],
            "init_from_adapter": "org/source-runs:rl/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 32,
        },
    }
    identity = rank_mod.AdapterArtifactIdentity(
        "digest-v1", "config-v1", "adapter_model.safetensors", "weights-v1:123"
    )
    from flash.spec import JobSpec

    public_job = JobSpec.from_dict(public_spec)
    worker_job = JobSpec.from_dict(worker_spec)
    runner._save_status(
        runner.RunStatus(
            run_id="nohandle-warm",
            state="provisioning",
            spec=public_spec,
            remote=None,
            effective_preparation={
                "worker_spec": worker_spec,
                "adapter_identity": identity.to_dict(),
                "preparation_digest": runner._preparation_digest(
                    public_job, worker_job, identity.to_dict()
                ),
            },
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-warm"}])
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda s: None)
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 32,
            "lora_alpha": 64,
        },
    )
    monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)
    resubmitted: list[tuple[str, int]] = []
    done = threading.Event()

    def fake_run_job(s):
        resubmitted.append((s.train.init_from_adapter, s.train.lora_rank))
        done.set()

    monkeypatch.setattr(runner, "_run_job", fake_run_job)

    app_mod.recover_runs()

    assert done.wait(timeout=5), "no-handle recovery must launch a resubmit thread"
    assert resubmitted == [("org/source-runs:rl/source-run", 32)]


def test_recover_runs_rejects_warmstart_artifact_drift(monkeypatch, tmp_path):
    import flash.lora_rank as rank_mod
    import flash.runner as runner
    import flash.server.db as db_mod

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import flash.server.app as app_mod

    importlib.reload(app_mod)
    public_spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"init_from_adapter": "source-run", "lora_rank": 8},
        "run_id": "drifted-warm",
    }
    worker_spec = {
        **public_spec,
        "train": {
            **public_spec["train"],
            "init_from_adapter": "private-owner/private-repo:rl/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 32,
        },
    }
    from flash.spec import JobSpec

    public_job = JobSpec.from_dict(public_spec)
    worker_job = JobSpec.from_dict(worker_spec)
    original_identity = {
        "digest": "original",
        "config_sha256": "config-v1",
        "weight_filename": "adapter_model.safetensors",
        "weight_identity": "weights-v1:123",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="drifted-warm",
            state="provisioning",
            spec=public_spec,
            effective_preparation={
                "worker_spec": worker_spec,
                "adapter_identity": original_identity,
                "preparation_digest": runner._preparation_digest(
                    public_job, worker_job, original_identity
                ),
            },
        )
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "drifted-warm"}])
    monkeypatch.setattr(
        rank_mod,
        "load_hf_adapter_config",
        lambda *a, **k: {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3.5-4B",
            "r": 32,
            "lora_alpha": 64,
        },
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            "changed", "config-v2", "adapter_model.safetensors", "weights-v2:123"
        ),
    )
    cleaned = []
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda spec: cleaned.append(spec))
    monkeypatch.setattr(
        runner,
        "_run_job",
        lambda s: pytest.fail("drifted warm-start source must not be resubmitted"),
    )

    app_mod.recover_runs()

    status = runner.get_status("drifted-warm")
    assert status.state == "failed"
    assert len(cleaned) == 1
    assert cleaned[0].train.init_from_adapter == "source-run"
    assert "source-run" in (status.error or "")
    assert "private-owner" not in (status.error or "")
    assert "private-repo" not in (status.error or "")


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
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "bad-1",
    }
    # Run #2: a valid no-handle spec — must still be recovered (resubmitted) despite run #1.
    good_spec = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "train": {"epochs": 1, "max_examples": 1},
        "gpu": {"type": "RTX 5090"},
        "run_id": "good-2",
    }
    runner._save_status(
        runner.RunStatus(
            run_id="bad-1",
            state="provisioning",
            spec={**good_spec, "run_id": "bad-1"},
            remote=None,
        )
    )
    bad_raw = runner._load_status_json("bad-1")
    bad_raw["spec"] = bad_spec
    with open(runner.runs_file_path("bad-1", ".json"), "w") as file:
        json.dump(bad_raw, file)
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


def _smoke_chat_result(revision: str, checkpoint: str, content: str = "4") -> dict:
    # a serve_chat response that passes _smoke_provenance for the given immutable revision
    hf_revision = revision.rsplit(".", 1)[-1]
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "freesolo": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
        "_freesolo_headers": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
    }


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
    key = _login()
    run_id = _make_run(api, key, "done")
    revision = f"{run_id}@step-40." + "a" * 40

    def fake_deploy(**kwargs):
        captured.update(kwargs)
        kwargs["before_activate"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )

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

    key = _login()
    run_id = _make_run(api, key, "cancelled")
    revision = f"{run_id}@step-80." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, f"{run_id}/step-80")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-80"),
    )
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

    key = _login()
    run_id = _make_run(api, key, state)
    revision = f"{run_id}@step-40." + "a" * 40

    def fake_deploy(**kwargs):
        kwargs["before_activate"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )
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
    revision = f"{run_id}@step-40." + "a" * 40

    def fake_deploy(**kwargs):
        status = runner.get_status(run_id)
        status.state = "done"
        runner._save_status(status)
        kwargs["before_activate"](revision, f"{run_id}/step-40")
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)
    monkeypatch.setattr(
        app_mod,
        "serve_chat",
        lambda **kwargs: _smoke_chat_result(revision, f"{run_id}/step-40"),
    )

    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    status = runner.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment["checkpoint_step"] == 40


def test_deploy_checkpoint_preserves_final_deploy_that_wins_cas(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    undeploys = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id: undeploys.append(run_id))

    key = _login()
    run_id = _make_run(api, key, "done")

    final_deployment = {
        "state": "ready",
        "endpoint_name": "final",
        "adapter_revision": f"{run_id}@final." + "f" * 40,
    }

    def fake_deploy(**kwargs):
        runner.mark_deployed(
            run_id,
            final_deployment,
            verification_generation=runner.verified_adapter_revision_generation(run_id),
        )
        return _FakeDeployment(kwargs["adapter_prefix"])

    monkeypatch.setattr(app_mod, "deploy_adapter", fake_deploy)

    r = api.post(f"/v1/runs/{run_id}/deploy", json={"step": 40}, headers=_bearer(key))
    assert r.status_code == 200, r.text
    assert undeploys == []
    status = runner.get_status(run_id)
    assert status.state == "deployed"
    assert status.deployment == final_deployment


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


def test_deploy_checkpoint_preserves_concurrent_run_undeploy(api, monkeypatch):
    import flash.runner as runner
    import flash.server.app as app_mod

    monkeypatch.setattr(app_mod, "list_checkpoints", lambda spec: _FAKE_CKPTS)
    undeploys = []
    monkeypatch.setattr(app_mod, "undeploy_adapter", lambda run_id: undeploys.append(run_id))

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
    assert r.status_code == 200, r.text
    assert undeploys == []
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
    revision = f"{run_id}@step-40." + "f" * 40
    runner.add_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=runner.verified_adapter_revision_generation(run_id),
    )

    r = api.delete(f"/v1/runs/{run_id}/deploy", headers=_bearer(key))
    assert r.status_code == 200, r.text
    status = runner.get_status(run_id)
    assert status.state == "running"
    assert status.deployment["state"] == "undeployed"
    assert runner.read_verified_adapter_revisions(run_id) == frozenset()


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
    """A finished run's final adapter is read privately and exported with a public source ref."""
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
    assert body["source"] == run_id
    assert src_repo not in resp.text
    assert "step" not in body
    # the operator still reads the private source internally; only the response uses the public ref.
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
    """Step export targets the exact per-step checkpoint `flash deploy RUN_ID/step-N` would serve; an
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
    assert ok.json()["source"] == f"{run_id}/step-40"
    assert "org/test-runs" not in ok.text
    assert seen["source_subfolder"] == f"rl/{run_id}/checkpoints/step-40/adapter"
    assert seen["base_model"] == SPEC["model"]

    bad = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/a", "hf_token": "hf", "step": 99},
        headers=_bearer(key),
    )
    assert bad.status_code == 404, bad.text
    assert "step 99" in bad.json()["detail"]


# --- export: product-analytics report ------------------------------------------------------


def test_export_reports_product_analytics_event(api, monkeypatch):
    """A successful export fires the platform product-event reporter (best-effort) with the
    destination repo, url, and step; the report failing must never fail the export itself."""
    import flash.server.app as app_mod
    import flash.server.run_registry as run_registry

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod, "export_adapter", lambda **kw: "https://huggingface.co/me/adapters"
    )

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(run_registry, "record_model_exported", capture)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/adapters", "hf_token": "hf_user"},
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text
    assert seen["repository"] == "me/adapters"
    assert seen["url"] == "https://huggingface.co/me/adapters"
    assert seen["step"] is None
    assert seen["status"].run_id == run_id


def test_export_succeeds_even_when_analytics_report_raises(api, monkeypatch):
    import flash.server.app as app_mod
    import flash.server.run_registry as run_registry

    key = _login()
    run_id = _finished_run(api, key)
    monkeypatch.setattr(
        app_mod, "export_adapter", lambda **kw: "https://huggingface.co/me/adapters"
    )

    def boom(**_kwargs):
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(run_registry, "record_model_exported", boom)

    resp = api.post(
        f"/v1/runs/{run_id}/export",
        json={"repository": "me/adapters", "hf_token": "hf_user"},
        headers=_bearer(key),
    )
    assert resp.status_code == 200, resp.text


def test_record_model_exported_posts_allowlisted_event(monkeypatch):
    """The reporter posts the flash_model_exported event with org/user attribution and the
    export detail; no org in context disables the report entirely."""
    import flash.server.run_registry as run_registry

    posted: dict = {}
    monkeypatch.setattr(
        run_registry,
        "_post",
        lambda path, body: posted.update({"path": path, "body": body}) or True,
    )

    class _Status:
        def __init__(self):
            self.run_id = "run-9"
            self.platform_context = {"org_id": "org-A", "user_id": "user-1"}
            self.billing_context = None
            self.spec = {"model": "Qwen/Qwen3.5-0.8B"}

    ok = run_registry.record_model_exported(
        status=_Status(),
        repository="me/adapters",
        url="https://huggingface.co/me/adapters",
        step=120,
    )
    assert ok is True
    assert posted["path"] == "/api/flash/events/internal"
    assert posted["body"]["orgId"] == "org-A"
    assert posted["body"]["userId"] == "user-1"
    assert posted["body"]["event"] == "flash_model_exported"
    props = posted["body"]["properties"]
    assert props == {
        "run_id": "run-9",
        "repository": "me/adapters",
        "url": "https://huggingface.co/me/adapters",
        "step": 120,
        "model": "Qwen/Qwen3.5-0.8B",
    }

    class _NoOrg:
        def __init__(self):
            self.run_id = "run-9"
            self.platform_context = None
            self.billing_context = None
            self.spec = {}

    posted.clear()
    assert (
        run_registry.record_model_exported(
            status=_NoOrg(), repository="x/y", url="https://x", step=None
        )
        is False
    )
    assert posted == {}
