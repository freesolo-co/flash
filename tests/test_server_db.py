from __future__ import annotations

import contextlib
import importlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import flash.runner.lifecycle.state as runner_state
from flash.server.platform import db


def _initialize_fresh_database_process(path: str, barrier, results) -> None:
    from flash.server.platform import db as process_db

    real_connect = sqlite3.connect
    synchronized = False

    class SynchronizedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal synchronized
            if sql == "PRAGMA journal_mode=WAL" and not synchronized:
                synchronized = True
                barrier.wait(timeout=10)
            return super().execute(sql, parameters)

    def synchronized_connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=SynchronizedConnection)

    process_db.DB_PATH = path
    process_db.sqlite3.connect = synchronized_connect
    try:
        row = process_db.ensure_external_key(f"fslo_process_{os.getpid()}")
        results.put(None if row is not None else "key provisioning returned no row")
    except BaseException as exc:
        results.put(repr(exc))


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


def test_schema_is_initialized_once_across_many_requests(isolated_db, monkeypatch) -> None:
    executescript_calls = 0
    journal_mode_calls = 0
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def executescript(self, sql_script):
            nonlocal executescript_calls
            executescript_calls += 1
            return super().executescript(sql_script)

        def execute(self, sql, parameters=()):
            nonlocal journal_mode_calls
            if sql == "PRAGMA journal_mode=WAL":
                journal_mode_calls += 1
            return super().execute(sql, parameters)

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=TrackingConnection)

    monkeypatch.setattr(isolated_db.sqlite3, "connect", tracking_connect)

    for _ in range(20):
        assert isolated_db.ensure_external_key("fslo_repeated") is not None

    assert executescript_calls == 1
    assert journal_mode_calls == 1


@pytest.mark.parametrize("remove_parent", [False, True])
def test_deleted_database_reinitializes_schema_and_replaces_pooled_connection(
    isolated_db, remove_parent
) -> None:
    assert isolated_db.ensure_external_key("fslo_before_delete") is not None
    old_connection = isolated_db._connect()
    path = isolated_db.db_path()

    if remove_parent:
        shutil.rmtree(os.path.dirname(path))
    else:
        for candidate in (path, f"{path}-wal", f"{path}-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(candidate)

    assert isolated_db.ensure_external_key("fslo_after_delete") is not None
    new_connection = isolated_db._connect()

    assert new_connection is not old_connection
    with pytest.raises(sqlite3.ProgrammingError):
        old_connection.execute("SELECT 1")
    tables = {
        row[0]
        for row in new_connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {"api_keys", "runs"} <= tables


def test_schema_init_closes_locked_connection_before_backoff(isolated_db, monkeypatch) -> None:
    real_connect = sqlite3.connect
    failed_connections = []
    sleeps = []

    class LockedConnection(sqlite3.Connection):
        closed = False

        def execute(self, sql, parameters=()):
            if sql == "PRAGMA journal_mode=WAL":
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, parameters)

        def close(self):
            self.closed = True
            return super().close()

    def tracking_connect(*args, **kwargs):
        if not failed_connections:
            connection = real_connect(*args, **kwargs, factory=LockedConnection)
            failed_connections.append(connection)
            return connection
        return real_connect(*args, **kwargs)

    def record_sleep(delay):
        assert failed_connections[0].closed is True
        sleeps.append(delay)

    monkeypatch.setattr(isolated_db.sqlite3, "connect", tracking_connect)
    monkeypatch.setattr(isolated_db.time, "sleep", record_sleep)

    assert isolated_db.ensure_external_key("fslo_retry_after_close") is not None
    assert sleeps == [0.01]


def test_fresh_database_initialization_retries_process_lock_contention(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    process_count = 12
    barrier = context.Barrier(process_count)
    results = context.Queue()
    path = str(tmp_path / "state" / "server.db")
    processes = [
        context.Process(
            target=_initialize_fresh_database_process,
            args=(path, barrier, results),
        )
        for _ in range(process_count)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
        assert all(not process.is_alive() for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    errors = [results.get(timeout=5) for _ in processes]
    assert errors == [None] * process_count


def test_connection_is_reused_per_thread(isolated_db) -> None:
    main_connection = isolated_db._connect()
    assert isolated_db._connect() is main_connection

    barrier = threading.Barrier(3)
    thread_connections = []

    def connect_twice() -> None:
        first = isolated_db._connect()
        second = isolated_db._connect()
        thread_connections.append((first, second))
        barrier.wait(timeout=5)
        barrier.wait(timeout=5)

    threads = [threading.Thread(target=connect_twice) for _ in range(2)]
    for thread in threads:
        thread.start()

    barrier.wait(timeout=5)
    assert len(thread_connections) == 2
    assert all(first is second for first, second in thread_connections)
    assert thread_connections[0][0] is not thread_connections[1][0]
    assert all(first is not main_connection for first, _ in thread_connections)
    barrier.wait(timeout=5)

    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_last_used_at_writes_are_throttled(isolated_db, monkeypatch) -> None:
    update_calls = 0
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal update_calls
            if sql.startswith("UPDATE api_keys SET last_used_at"):
                update_calls += 1
            return super().execute(sql, parameters)

    def tracking_connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=TrackingConnection)

    monkeypatch.setattr(isolated_db.sqlite3, "connect", tracking_connect)
    times = iter([1000.0, 1001.0, 1010.0, 1050.0, 1062.0])
    monkeypatch.setattr(isolated_db.time, "time", lambda: next(times))

    row = isolated_db.ensure_external_key("fslo_throttled")
    assert row is not None
    assert isolated_db.lookup_key("fslo_throttled") is not None
    assert isolated_db.lookup_key("fslo_throttled") is not None
    assert isolated_db.lookup_key("fslo_throttled") is not None

    with real_connect(isolated_db.db_path()) as conn:
        last_used_at = conn.execute(
            "SELECT last_used_at FROM api_keys WHERE id = ?", (row["id"],)
        ).fetchone()[0]
    assert update_calls == 2
    assert last_used_at == 1062.0


def test_concurrent_threads_preserve_basic_read_write_correctness(isolated_db) -> None:
    barrier = threading.Barrier(2)

    def create_owned_run(index: int) -> tuple[int, int | None]:
        barrier.wait()
        row = isolated_db.ensure_external_key(f"fslo_thread_{index}")
        assert row is not None
        isolated_db.record_run(f"run-thread-{index}", row["id"])
        return row["id"], isolated_db.run_owner(f"run-thread-{index}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create_owned_run, range(2)))

    assert all(key_id == owner_id for key_id, owner_id in results)
    assert {row["run_id"] for row in isolated_db.all_runs()} == {
        "run-thread-0",
        "run-thread-1",
    }


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
    # runpod.auth caches the parsed pool on first read; reset so the startup preflight reads THIS
    # RUNPOD_API_KEY (the autouse _offline fixture also resets, but make the fixture self-contained).
    import flash.providers.runpod.client.auth as runpod_keys

    runpod_keys.reset()
    import flash.server.platform.auth as auth_mod
    import flash.server.platform.db as db_mod

    importlib.reload(auth_mod)
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
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

    import flash.server.asgi.app as app_mod
    from flash.providers.core import registry as providers_mod

    importlib.reload(app_mod)
    # The Lambda key above (required by the new preflight) makes configured_providers()
    # treat it as live, so create_app()'s lifespan recover_runs() would dispatch real sweep_orphans()
    # list calls at startup. This test only checks /v1/me, so stub the provider set to empty to keep it
    # hermetic. (Pre-PR this fixture set only RUNPOD_API_KEY, whose sweep is a no-op.)
    monkeypatch.setattr(providers_mod, "configured_providers", list, raising=False)
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
