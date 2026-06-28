from __future__ import annotations

import importlib
import json
import sqlite3

import pytest

from flash.server import db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "state" / "server.db"))
    return db


def test_hash_key_is_stable_sha256_not_plaintext() -> None:
    digest = db.hash_key("fslo_secret")

    assert digest == db.hash_key("fslo_secret")
    assert digest != "fslo_secret"
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_external_key_provisioning_is_idempotent_and_hashed(isolated_db) -> None:
    row = isolated_db.ensure_external_key(
        "fslo_external_secret", key_prefix="fslo_abcd", email="user@example.com"
    )
    assert row is not None
    assert row["key_prefix"] == "fslo_abcd"
    assert row["email"] == "user@example.com"

    again = isolated_db.ensure_external_key(
        "fslo_external_secret", key_prefix="changed", email="changed@example.com"
    )
    assert again is not None
    assert again["id"] == row["id"]
    assert again["key_prefix"] == "fslo_abcd"
    assert again["email"] == "user@example.com"

    with sqlite3.connect(isolated_db.db_path()) as conn:
        stored = conn.execute("SELECT key_hash FROM api_keys").fetchone()[0]
    assert stored == isolated_db.hash_key("fslo_external_secret")
    assert stored != "fslo_external_secret"
    with open(isolated_db.db_path(), "rb") as f:
        assert b"fslo_external_secret" not in f.read()


def test_internal_key_provisioning_is_idempotent_and_hashed(isolated_db) -> None:
    first = isolated_db.ensure_internal_key("internal-secret")
    second = isolated_db.ensure_internal_key("internal-secret")

    assert second["id"] == first["id"]
    assert second["key_prefix"] == "internal"
    assert second["email"] == "internal@freesolo.co"
    with sqlite3.connect(isolated_db.db_path()) as conn:
        stored = conn.execute(
            "SELECT key_hash FROM api_keys WHERE id = ?", (first["id"],)
        ).fetchone()[0]
    assert stored == isolated_db.hash_key("internal-secret")
    assert stored != "internal-secret"


def test_internal_key_email_is_migrated_to_service_email(isolated_db) -> None:
    key_hash = isolated_db.hash_key("internal-secret")
    with isolated_db._connect() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, email, created_at) VALUES (?, ?, ?, ?)",
            (key_hash, "internal", "freesolo-internal", 1.0),
        )

    row = isolated_db.ensure_internal_key("internal-secret")

    assert row["email"] == "internal@freesolo.co"


def test_disabled_external_key_is_not_recreated_or_revived(isolated_db) -> None:
    row = isolated_db.ensure_external_key("fslo_revoked", key_prefix="fslo_rev")
    assert row is not None

    with sqlite3.connect(isolated_db.db_path()) as conn:
        conn.execute("UPDATE api_keys SET disabled = 1 WHERE id = ?", (row["id"],))

    assert isolated_db.lookup_key("fslo_revoked") is None
    assert isolated_db.ensure_external_key("fslo_revoked", key_prefix="fslo_new") is None
    with sqlite3.connect(isolated_db.db_path()) as conn:
        rows = conn.execute(
            "SELECT disabled, key_prefix FROM api_keys WHERE key_hash = ?",
            (isolated_db.hash_key("fslo_revoked"),),
        ).fetchall()
    assert rows == [(1, "fslo_rev")]


def test_lookup_key_updates_last_used_at_in_database(isolated_db, monkeypatch) -> None:
    times = iter([1000.0, 1001.0, 1234.0])
    monkeypatch.setattr(isolated_db.time, "time", lambda: next(times))

    row = isolated_db.ensure_external_key("fslo_time", key_prefix="fslo_time")
    assert row is not None
    found = isolated_db.lookup_key("fslo_time")
    assert found is not None
    assert found["id"] == row["id"]

    with sqlite3.connect(isolated_db.db_path()) as conn:
        last_used_at = conn.execute(
            "SELECT last_used_at FROM api_keys WHERE id = ?", (row["id"],)
        ).fetchone()[0]
    assert last_used_at == 1234.0


def test_run_ownership_lists_in_creation_order_and_delete_is_idempotent(
    isolated_db, monkeypatch
) -> None:
    owner_a = isolated_db.ensure_external_key("fslo_owner_a")
    owner_b = isolated_db.ensure_external_key("fslo_owner_b")
    assert owner_a is not None
    assert owner_b is not None

    created_at = iter([300.0, 100.0, 200.0])
    monkeypatch.setattr(isolated_db.time, "time", lambda: next(created_at))
    isolated_db.record_run("run-late", owner_a["id"])
    isolated_db.record_run("run-early", owner_a["id"])
    isolated_db.record_run("run-other", owner_b["id"])

    assert [r["run_id"] for r in isolated_db.runs_for_key(owner_a["id"])] == [
        "run-early",
        "run-late",
    ]
    assert isolated_db.run_owner("run-other") == owner_b["id"]
    assert {r["run_id"] for r in isolated_db.all_runs()} == {"run-late", "run-early", "run-other"}

    isolated_db.delete_run("run-early")
    isolated_db.delete_run("run-early")
    assert isolated_db.run_owner("run-early") is None
    assert [r["run_id"] for r in isolated_db.runs_for_key(owner_a["id"])] == ["run-late"]


def test_record_run_rejects_duplicate_run_id(isolated_db) -> None:
    owner_a = isolated_db.ensure_external_key("fslo_a")
    owner_b = isolated_db.ensure_external_key("fslo_b")
    assert owner_a is not None
    assert owner_b is not None

    isolated_db.record_run("run-dupe", owner_a["id"])
    with pytest.raises(sqlite3.IntegrityError):
        isolated_db.record_run("run-dupe", owner_b["id"])
    assert isolated_db.run_owner("run-dupe") == owner_a["id"]


def test_record_run_rejects_unknown_key_id(isolated_db) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        isolated_db.record_run("run-orphan", 999)
    assert isolated_db.all_runs() == []


def test_me_surfaces_verify_identity_fields_through_api(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test,rp-test-2")
    monkeypatch.setenv("LAMBDA_API_KEY", "lam-test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "fslo-internal-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    # runpod.keys caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.keys as runpod_keys

    runpod_keys.reset()

    import flash.runner as runner
    import flash.server.auth as auth_mod
    import flash.server.db as db_mod

    importlib.reload(runner)
    importlib.reload(auth_mod)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "server.db"))

    token = "fslo_abc123_secret"
    body = json.dumps(
        {
            "ok": True,
            "email": "user@example.com",
            "user_id": "user-123",
            "org": {"id": "org-456", "slug": "acme", "name": "Acme"},
            "api_key": {"id": "key-789", "key_prefix": "fslo_abc123"},
            "training_agent_job_id": "job-001",
            "project_id": "proj-002",
        }
    ).encode()
    seen: dict[str, str | None] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self) -> bytes:
            return body

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(auth_mod.urllib.request, "urlopen", fake_urlopen)

    import flash.providers as providers_mod
    import flash.providers.runpod.train.endpoints as rp_endpoints
    import flash.server.app as app_mod

    importlib.reload(app_mod)
    # The Lambda key above (required by the new preflight) makes configured_providers()
    # treat it as live, so create_app()'s lifespan recover_runs() would dispatch real sweep_orphans()
    # list calls at startup. This test only checks /v1/me, so stub the provider set to empty to keep it
    # hermetic. (Pre-PR this fixture set only RUNPOD_API_KEY, whose sweep is a no-op.)
    monkeypatch.setattr(providers_mod, "configured_providers", lambda: [], raising=False)
    # FREESOLO_INTERNAL_KEY also enables startup slot-store reconcile (urllib POST). No-op it so this
    # fixture's urlopen stub stays focused on /api/auth/verify.
    monkeypatch.setattr(
        rp_endpoints, "reconcile_endpoint_slots", lambda *a, **k: None, raising=False
    )
    with TestClient(app_mod.create_app()) as client:
        res = client.get("/v1/me", headers={"Authorization": f"Bearer {token}"})

    assert res.status_code == 200, res.text
    assert seen == {
        "url": "https://freesolo.test/api/auth/verify",
        "authorization": f"Bearer {token}",
    }
    assert res.json() == {
        "kind": "freesolo_api_key",
        "key_prefix": "fslo_abc123",
        "email": "user@example.com",
        "user_id": "user-123",
        "org_id": "org-456",
        "org_slug": "acme",
        "org_name": "Acme",
        "api_key_id": "key-789",
        "training_agent_job_id": "job-001",
        "project_id": "proj-002",
    }

    with sqlite3.connect(db_mod.db_path()) as conn:
        stored = conn.execute("SELECT key_hash, key_prefix, email FROM api_keys").fetchone()
    assert stored == (db_mod.hash_key(token), "fslo_abc123", "user@example.com")
