"""Control-plane API: key claiming, bearer auth, multi-tenant isolation (CPU-only).

All runs are dry-run so nothing touches the network; operator env vars are dummies
(the startup preflight only checks presence).
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm import orchestrator as _orch
from autoslm.server import db as _db_mod

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

SPEC = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "algorithm": "grpo",
    "environment": {"id": "gsm8k"},
    "train": {"steps": 1, "seeds": [0]},
    "gpu": {"type": "RTX 5090"},
}


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("HF_REPO", "org/test-runs")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-test")
    import autoslm.orchestrator as orchestrator
    import autoslm.server.db as db_mod

    importlib.reload(orchestrator)
    # The storage roots are fixed constants (not env-configurable); redirect them to tmp for
    # test isolation by patching the module attributes after reload.
    monkeypatch.setattr(orchestrator, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orchestrator, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))
    import autoslm.server.app as app_mod

    importlib.reload(app_mod)
    import autoslm.server.auth as auth_mod

    auth_mod._claims.clear()
    with TestClient(app_mod.create_app()) as client:
        yield client


def _claim(api, email=None) -> str:
    r = api.post("/v1/keys", json={"email": email} if email else {})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def test_claim_and_me(api):
    key = _claim(api, email="a@example.com")
    assert key.startswith("sk-autoslm-")
    me = api.get("/v1/me", headers=_bearer(key))
    assert me.status_code == 200
    assert me.json()["email"] == "a@example.com"
    assert me.json()["key_prefix"] == key[:15]


def test_requests_without_key_are_rejected(api):
    assert api.get("/v1/runs").status_code == 401
    assert api.get("/v1/runs", headers=_bearer("sk-autoslm-bogus")).status_code == 401
    assert api.get("/v1/models", headers=_bearer("not-even-the-right-shape")).status_code == 401
    assert api.get("/v1/health").status_code == 200  # health stays open


def test_internal_key_authenticates_as_service_identity(api, monkeypatch):
    # With FREESOLO_INTERNAL_KEY configured, the shared internal key works as a bearer
    # (no sk-autoslm- claim needed) and owns the runs it submits — the freesolo SDK
    # authenticates with the same credential the platform uses.
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
    # a token that is neither a minted key nor the configured internal key is rejected
    assert api.get(f"/v1/runs/{run_id}", headers=_bearer("wrong-internal")).status_code == 401
    # the internal key is stored hashed, like any other (never persisted in the clear)
    with sqlite3.connect(_db_mod.db_path()) as conn:
        prefixes = [row[0] for row in conn.execute("SELECT key_prefix FROM api_keys").fetchall()]
    assert "internal" in prefixes


def test_internal_key_rejected_when_unconfigured(api):
    # Without FREESOLO_INTERNAL_KEY set, the would-be internal key is just an unknown
    # token (it lacks the sk-autoslm- prefix) and gets 401 — no implicit acceptance.
    assert api.get("/v1/runs", headers=_bearer("fslo-internal-secret")).status_code == 401


def test_keys_are_hashed_at_rest(api):
    key = _claim(api)
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
    key_a, key_b = _claim(api), _claim(api)
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
    key = _claim(api)
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
    key = _claim(api)
    bad = {**SPEC, "environment": {"id": "custom", "path": "/home/user/env.py"}}
    r = api.post("/v1/runs", json={"spec": bad, "dry_run": True}, headers=_bearer(key))
    assert r.status_code == 400
    assert "not supported on the managed service" in r.json()["detail"]


def test_bad_spec_is_400(api):
    key = _claim(api)
    r = api.post("/v1/runs", json={"spec": {"algorithm": "grpo"}}, headers=_bearer(key))
    assert r.status_code == 400
    assert "model" in r.json()["detail"]


def test_deploy_dry_run(api):
    key = _claim(api)
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


def test_mark_deployed_allows_done_but_not_cancelled(monkeypatch, tmp_path):
    # A finished run (state="done") MUST be deployable: mark_deployed has to record the
    # deployment and flip to "deployed". But a cancelled/failed run must never be flipped
    # to "deployed" (a /cancel racing always-on provisioning persisted the terminal state).
    import autoslm.orchestrator as orchestrator

    importlib.reload(orchestrator)
    monkeypatch.setattr(orchestrator, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orchestrator, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3-4B-Instruct-2507", "algorithm": "grpo", "run_id": "dep-1"}
    orchestrator._save_status(
        orchestrator.RunStatus(run_id="dep-1", state="done", spec=spec, remote=None)
    )
    out = orchestrator.mark_deployed("dep-1", {"endpoint_name": "e", "mode": "dev"})
    assert out.state == "deployed"
    assert out.deployment == {"endpoint_name": "e", "mode": "dev"}

    # cancelled is sticky: the deploy must be refused, state preserved.
    orchestrator._save_status(
        orchestrator.RunStatus(
            run_id="dep-2", state="cancelled", spec={**spec, "run_id": "dep-2"}, remote=None
        )
    )
    out2 = orchestrator.mark_deployed("dep-2", {"endpoint_name": "e2", "mode": "dev"})
    assert out2.state == "cancelled"
    assert out2.deployment is None


def test_mark_deployed_expect_state_cas_blocks_undeploy_race(monkeypatch, tmp_path):
    # Redeploy finalization must NOT clobber an undeploy that raced in mid-warmup: the
    # undeploy wrote `done`/undeployed and deleted the endpoint, so a final mark_deployed
    # that still expects "deployed" must refuse to re-advertise the deleted endpoint.
    import autoslm.orchestrator as orchestrator

    importlib.reload(orchestrator)
    monkeypatch.setattr(orchestrator, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orchestrator, "RESULTS_DIR", str(tmp_path / "results"))

    spec = {"model": "Qwen/Qwen3-4B-Instruct-2507", "algorithm": "grpo", "run_id": "dep-3"}
    orchestrator._save_status(
        orchestrator.RunStatus(
            run_id="dep-3",
            state="deployed",
            spec=spec,
            remote=None,
            deployment={"endpoint_name": "e", "mode": "always-on"},
        )
    )
    # undeploy races in: endpoint torn down, run back to done/undeployed.
    undone = orchestrator.mark_undeployed("dep-3")
    assert undone.state == "done"
    assert undone.deployment["state"] == "undeployed"
    # the deploy that was warming finalizes expecting "deployed" -> refused.
    out = orchestrator.mark_deployed("dep-3", {"endpoint_name": "e2"}, expect_state="deployed")
    assert out.state == "done"
    assert out.deployment["state"] == "undeployed"  # not re-advertised


def test_deploy_lock_is_usable_and_weakly_cleaned():
    # threading.Lock() isn't weak-referenceable, so the per-run lock must be a wrapper that
    # both works as a context manager AND can live in the WeakValueDictionary (the raw lock
    # would TypeError on the first deploy). It must also re-enter and serialize.
    import gc

    from autoslm.server import app as app_mod

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


def test_claim_throttle(api):
    for _ in range(5):
        _claim(api)
    r = api.post("/v1/keys", json={})
    assert r.status_code == 429


def test_recover_runs_gcs_no_handle_endpoints(monkeypatch, tmp_path):
    # A recoverable run with no persisted handle (crash between endpoint
    # registration and on_handle) must have its reconstructable RunPod endpoint
    # GC'd before being failed, so it doesn't hold worker quota until manual cleanup.
    import autoslm.orchestrator as orchestrator
    import autoslm.server.db as db_mod

    importlib.reload(orchestrator)
    monkeypatch.setattr(orchestrator, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orchestrator, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "s.db"))
    import autoslm.server.app as app_mod

    importlib.reload(app_mod)

    spec = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "algorithm": "grpo",
        "train": {"steps": 1, "seeds": [0]},
        "gpu": {"type": "RTX 5090"},
        "run_id": "nohandle-1",
    }
    orchestrator._save_status(
        orchestrator.RunStatus(run_id="nohandle-1", state="provisioning", spec=spec, remote=None)
    )
    monkeypatch.setattr(app_mod.db, "all_runs", lambda: [{"run_id": "nohandle-1"}])
    gced = []
    monkeypatch.setattr(orchestrator, "_gc_run_endpoints", lambda s: gced.append(s.run_id))

    app_mod.recover_runs()

    assert gced == ["nohandle-1"], "no-handle recovery must GC the reconstructable endpoint"
    assert orchestrator.get_status("nohandle-1").state == "failed"


def test_valid_last_xff_hop_is_its_own_throttle_bucket(api):
    # A well-formed IP in the last X-Forwarded-For hop keys the throttle, so distinct trusted
    # client IPs each get their own claim budget rather than sharing one bucket.
    for i in range(3):
        r = api.post("/v1/keys", json={}, headers={"x-forwarded-for": f"203.0.113.{i}"})
        assert r.status_code == 200, r.text


def test_invalid_xff_hop_falls_back_to_peer_bucket(api):
    # Garbage / non-IP last hops are rejected and collapse to the socket-peer bucket, so a
    # client can't dodge the per-IP throttle by rotating spoofed XFF values.
    for i in range(5):
        r = api.post("/v1/keys", json={}, headers={"x-forwarded-for": f"not-an-ip-{i}"})
        assert r.status_code == 200, r.text
    r = api.post("/v1/keys", json={}, headers={"x-forwarded-for": "still-not-an-ip"})
    assert r.status_code == 429


def test_oversized_xff_hop_is_rejected(api):
    # An absurdly long last hop is bounded out before parsing (can't be a real IP literal) and
    # falls back to the peer bucket — no unbounded growth of the in-memory claims map.
    for _ in range(5):
        r = api.post("/v1/keys", json={}, headers={"x-forwarded-for": "9" * 4000})
        assert r.status_code == 200, r.text
    r = api.post("/v1/keys", json={}, headers={"x-forwarded-for": "8" * 4000})
    assert r.status_code == 429
