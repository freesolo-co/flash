from __future__ import annotations

import pytest

from flash.providers.runpod import warm_pool
from flash.server import db
from flash.spec import JobSpec


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "state" / "server.db"))


def _spec(**overrides):
    data = {
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "sft",
        "environment": {"id": "env/demo"},
        "train": {"lora_rank": 8, "max_context_tokens": 2048, "learning_rate": 1e-5, "epochs": 1},
        "gpu": {"type": "H100", "disk_gb": 60, "keep_alive_seconds": 600},
        "run_id": "run-1",
    }
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(section), dict):
            data[section] = {**data[section], **value}
        else:
            data[section] = value
    return JobSpec.from_dict(data)


def _sig(spec):
    return warm_pool.compat_signature(spec, code_digest="deadbeef", worker_image="img:1")


# --- compat signature ---------------------------------------------------------------------------


def test_compat_signature_is_stable():
    assert _sig(_spec()) == _sig(_spec())


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "Qwen/Qwen3.5-9B"},
        {"train": {"lora_rank": 16}},
        {"train": {"max_context_tokens": 4096}},
        {"gpu": {"type": "H200"}},
        {"algorithm": "grpo"},
        {"environment": {"id": "env/other"}},  # environment name is a match key
    ],
)
def test_compat_signature_changes_when_gpu_or_env_determinant_changes(overrides):
    assert _sig(_spec()) != _sig(_spec(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"train": {"learning_rate": 5e-6}},
        {"train": {"epochs": 3}},
        {"run_id": "run-99"},
        {"environment": {"pip": ["cowsay"]}},  # extra pip changes neither gpu nor environment code
    ],
)
def test_compat_signature_ignores_non_resource_config(overrides):
    # things that change neither the gpu nor the environment code must still reuse the warm box
    assert _sig(_spec()) == _sig(_spec(**overrides))


def test_compat_signature_tracks_env_code_digest():
    a = warm_pool.compat_signature(_spec(), code_digest="aaaa", worker_image="img:1")
    b = warm_pool.compat_signature(_spec(), code_digest="bbbb", worker_image="img:1")
    assert a != b


# --- acquire ------------------------------------------------------------------------------------


def _register(spec, *, endpoint_id="ep-1", owner_key_id=7, owner_org_id="org-a", expiry_ts=1e12):
    db.register_warm_endpoint(
        endpoint_id=endpoint_id,
        name=f"flash-{endpoint_id}",
        owning_fingerprint="fp",
        owner_key_id=owner_key_id,
        owner_org_id=owner_org_id,
        compat_sig=warm_pool.compat_signature(spec, code_digest="deadbeef", worker_image="img:1"),
        gpu_type=spec.gpu.type,
        expiry_ts=expiry_ts,
    )


def _healthy(monkeypatch, in_flight=0, exists=True):
    def fake_health(endpoint_id, fp):
        if not exists:
            return {}
        return {"workers": {"idle": 1}, "jobs": {"inQueue": in_flight, "inProgress": 0}}

    monkeypatch.setattr(
        "flash.providers.runpod.api.endpoint_health_for_fingerprint", fake_health, raising=False
    )


def test_try_acquire_reuses_healthy_same_owner(isolated_db, monkeypatch):
    spec = _spec()
    _register(spec)
    _healthy(monkeypatch)
    got = warm_pool.try_acquire(
        spec, owner_key_id=7, owner_org_id="org-a", code_digest="deadbeef",
        worker_image="img:1", run_id="run-2", now=1.0,
    )
    assert got is not None and got["endpoint_id"] == "ep-1"


def test_try_acquire_none_for_different_owner(isolated_db, monkeypatch):
    spec = _spec()
    _register(spec, owner_org_id="org-a")
    _healthy(monkeypatch)
    assert (
        warm_pool.try_acquire(
            spec, owner_key_id=7, owner_org_id="org-b", code_digest="deadbeef",
            worker_image="img:1", run_id="run-2", now=1.0,
        )
        is None
    )


def test_try_acquire_releases_and_falls_back_when_endpoint_gone(isolated_db, monkeypatch):
    spec = _spec()
    _register(spec)
    _healthy(monkeypatch, exists=False)  # endpoint vanished
    assert (
        warm_pool.try_acquire(
            spec, owner_key_id=7, owner_org_id="org-a", code_digest="deadbeef",
            worker_image="img:1", run_id="run-2", now=1.0,
        )
        is None
    )
    # the stale record must be dropped so it never blocks a future deploy
    assert db.all_warm_endpoints() == []


# --- keep-warm gating + reap --------------------------------------------------------------------


def test_keep_warm_after_run_registers_only_when_enabled(isolated_db):
    on = _spec(gpu={"keep_alive_seconds": 600})
    kept = warm_pool.keep_warm_after_run(
        on, endpoint_id="ep-1", name="flash-ep-1", owning_fingerprint="fp",
        owner_key_id=7, owner_org_id="org-a", code_digest="deadbeef", worker_image="img:1", now=100.0,
    )
    assert kept is True
    rows = db.all_warm_endpoints()
    assert len(rows) == 1 and rows[0]["expiry_ts"] == pytest.approx(700.0)


def test_keep_warm_after_run_noop_when_disabled(isolated_db):
    off = _spec(gpu={"keep_alive_seconds": 0})
    kept = warm_pool.keep_warm_after_run(
        off, endpoint_id="ep-1", name="flash-ep-1", owning_fingerprint="fp",
        owner_key_id=7, owner_org_id="org-a", code_digest="deadbeef", worker_image="img:1", now=100.0,
    )
    assert kept is False
    assert db.all_warm_endpoints() == []


def test_reap_expired_deletes_and_drops(isolated_db, monkeypatch):
    spec = _spec()
    _register(spec, expiry_ts=100.0)
    deleted = []
    monkeypatch.setattr(
        "flash.providers.runpod.api.delete_endpoint_for_fingerprint",
        lambda eid, fp: deleted.append(eid) or True,
        raising=False,
    )
    assert warm_pool.reap_expired(now=200.0) == 1
    assert deleted == ["ep-1"]
    assert db.all_warm_endpoints() == []
