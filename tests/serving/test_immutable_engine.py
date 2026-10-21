from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.engine.lora_engine import _LoraEngineImpl, _LoraEntry
from flash.serving.src.engine.support import (
    _adapter_source_cache_dir,
    _adapter_source_ident,
)
from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload
from flash.serving.src.store.registry import AdapterRegistry

BASE_MODEL = "Qwen/Qwen3.5-9B"
RUN_ID = "flash-1234567890-abcdef12"


def _revision(sha: str, *, status: str = "ready") -> AdapterRecord:
    values = {
        "model_id": f"{RUN_ID}/step-20",
        "repo_id": "org/run",
        "org_id": "org-1",
        "base_model": BASE_MODEL,
        "subfolder": "checkpoints/step-20",
        "repo_type": "model",
        "checkpoint": f"{RUN_ID}/step-20",
        "private": True,
        "thinking": False,
        "status": status,
        "run_id": RUN_ID,
        "checkpoint_step": 20,
        "artifact_revision": sha,
        "artifact_digest": "b" * 64,
        "lora_rank": 16,
    }
    values["artifact_fingerprint"] = immutable_binding_fingerprint(
        {**values, "checkpoint_id": values["model_id"]}
    )
    return AdapterRecord.model_validate(values)


def test_distinct_hub_shas_have_distinct_source_and_cache_identity(tmp_path: Path) -> None:
    first = _revision("a" * 40)
    second = _revision("b" * 40)
    assert _adapter_source_ident(first) != _adapter_source_ident(second)
    assert _adapter_source_cache_dir(tmp_path, first) != _adapter_source_cache_dir(tmp_path, second)


def test_unregister_skips_cleanup_for_a_newer_deployment_generation() -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-new"})
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine.registry.upsert(record, revive=True)
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    evicted: list[tuple[str, str]] = []

    async def _evict(adapter_key: tuple[str, str]) -> None:
        evicted.append(adapter_key)

    engine._evict_loaded_lora = _evict

    async def _exercise() -> tuple[dict, dict, dict]:
        legacy = await engine._unregister(record.org_id, record.model_id)
        stale = await engine._unregister(record.org_id, record.model_id, "generation-old")
        removed = await engine._unregister(record.org_id, record.model_id, "generation-new")
        return legacy, stale, removed

    legacy, stale, removed = asyncio.run(_exercise())

    assert legacy["skipped_stale_generation"] is True
    assert stale["skipped_stale_generation"] is True
    assert removed["removed"] == record.model_id
    assert engine.registry.get(record.org_id, record.model_id) is None
    assert evicted == [record.storage_key]


def test_add_lora_exception_releases_new_entry_for_retry(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    engine = _LoraEngineImpl()
    calls = 0

    class _Engine:
        async def add_lora(self, _request: object) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("worker unavailable")
            return True

    engine.engine = _Engine()

    with pytest.raises(RuntimeError, match="worker unavailable"):
        asyncio.run(engine._add_lora_locked(record, tmp_path))

    assert record.storage_key not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.storage_key].state == "loaded"


def test_rejected_new_lora_releases_entry_and_reaches_engine_on_retry(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    engine = _LoraEngineImpl()
    outcomes = iter((False, True))
    calls = 0

    class _Engine:
        async def add_lora(self, _request: object) -> bool:
            nonlocal calls
            calls += 1
            return next(outcomes)

    engine.engine = _Engine()

    with pytest.raises(RuntimeError, match="vLLM rejected a new LoRA registration"):
        asyncio.run(engine._add_lora_locked(record, tmp_path))

    assert record.storage_key not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.storage_key].state == "loaded"


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        ("missing", "vLLM cannot confirm LoRA removal"),
        ("raises", "worker unavailable"),
        ("false", "vLLM rejected LoRA removal"),
    ],
)
def test_loaded_lora_failed_removal_retains_unconfirmed_entry(
    outcome: str, message: str, tmp_path: Path
) -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    engine = _LoraEngineImpl()
    engine._lora_entries = {
        record.storage_key: _LoraEntry(_adapter_source_ident(record), request, "loaded")
    }

    if outcome == "missing":
        engine.engine = object()
    else:

        class _Engine:
            async def remove_lora(self, _int_id: int) -> bool:
                if outcome == "raises":
                    raise RuntimeError("worker unavailable")
                return False

        engine.engine = _Engine()

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(engine._evict_loaded_lora(record.storage_key))

    entry = engine._entries()[record.storage_key]
    assert entry.lora_request is request
    assert entry.state == "unconfirmed"
    with pytest.raises(RuntimeError, match="registration is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)


def test_reserved_lora_evict_releases_without_engine_removal() -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    engine = _LoraEngineImpl()
    engine._lora_entries = {
        record.storage_key: _LoraEntry(_adapter_source_ident(record), request, "reserved")
    }
    removed: list[int] = []

    class _Engine:
        async def remove_lora(self, int_id: int) -> bool:
            removed.append(int_id)
            return True

    engine.engine = _Engine()

    asyncio.run(engine._evict_loaded_lora(record.storage_key))

    assert record.storage_key not in engine._entries()
    assert removed == []


def test_cached_lora_request_enforces_state_and_source_identity(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    source_ident = _adapter_source_ident(record)
    engine = _LoraEngineImpl()
    engine._lora_entries = {record.storage_key: _LoraEntry(source_ident, request, "loaded")}

    assert engine._cached_lora_request_locked(record, tmp_path) is request

    engine._lora_entries[record.storage_key] = _LoraEntry(source_ident, request, "unconfirmed")
    with pytest.raises(RuntimeError, match="registration is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)

    other_ident = ("other/repo", *source_ident[1:])
    engine._lora_entries[record.storage_key] = _LoraEntry(other_ident, request, "loaded")
    with pytest.raises(RuntimeError, match="previous LoRA removal is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)


def test_cold_unregister_uses_durable_disable_time(monkeypatch) -> None:
    stale_record = _revision("a" * 40).model_copy(
        update={
            "updated_at": "2026-07-14T00:00:00+00:00",
            "deployment_generation": "generation-old",
        }
    )
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    evicted: list[tuple[str, str]] = []

    async def _evict(adapter_key: tuple[str, str]) -> None:
        evicted.append(adapter_key)

    engine._evict_loaded_lora = _evict
    disabled = stale_record.model_copy(
        update={
            "status": "disabled",
            "deployment_generation": None,
            "updated_at": "2026-07-14T00:01:00+00:00",
        }
    )

    async def authoritative(org_id, model_id):
        assert (org_id, model_id) == stale_record.storage_key
        return disabled

    monkeypatch.setattr("flash.serving.src.store.undeploy.get_authoritative", authoritative)

    result = asyncio.run(
        engine._unregister(stale_record.org_id, stale_record.model_id, "generation-old")
    )

    assert result["removed"] == stale_record.model_id
    assert evicted == [stale_record.storage_key]
    engine.registry.upsert(stale_record)
    assert engine.registry.get(stale_record.org_id, stale_record.model_id) is None
    # even if the redeploy completed before this cleanup returned, its durable timestamp wins.
    newer = stale_record.model_copy(
        update={
            "updated_at": "2026-07-14T00:02:00+00:00",
            "deployment_generation": "generation-new",
        }
    )
    engine.registry.upsert(newer)
    assert engine.registry.get(newer.org_id, newer.model_id) == newer


def test_snapshot_download_receives_exact_hub_sha(monkeypatch, tmp_path: Path) -> None:
    record = _revision("a" * 40)
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        adapter_dir = local_dir / "checkpoints/step-20"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
        return str(local_dir)

    monkeypatch.setattr("flash.serving.src.engine.hub.download_adapter", snapshot_download)
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path)

    engine = _LoraEngineImpl()
    engine.registry = AdapterRegistry()
    engine.settings = SimpleNamespace(hf_api_key="token")
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine._lora_entries = {}

    path = asyncio.run(engine._ensure_adapter_local_locked(record))
    relative = path.relative_to(tmp_path)
    assert relative.parts[0] == "sources"
    assert relative.parts[-2:] == ("checkpoints", "step-20")
    assert calls[0]["repo_id"] == "org/run"
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["subfolder"] == "checkpoints/step-20"


def test_runtime_containment_rejects_schema_bypass(monkeypatch, tmp_path: Path) -> None:
    record = _revision("a" * 40).model_copy(update={"subfolder": "../escape"})
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("flash.serving.src.engine.hub.download_adapter", snapshot_download)
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path)

    engine = _LoraEngineImpl()
    engine.registry = AdapterRegistry()
    engine.settings = SimpleNamespace(hf_api_key="token")
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine._lora_entries = {}

    with pytest.raises(ValueError, match="escapes its exact-SHA source cache"):
        asyncio.run(engine._ensure_adapter_local_locked(record))
    assert calls == []


@pytest.mark.parametrize("status", ["ready", "disabled"])
def test_cold_unregister_does_not_tombstone_new_deployment(monkeypatch, status):
    record = _revision("a" * 40, status=status).model_copy(
        update={"deployment_generation": "generation-new", "updated_at": "2026-07-14T00:00:00Z"}
    )
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()

    async def authoritative(org_id, model_id):
        assert (org_id, model_id) == record.storage_key
        return record

    monkeypatch.setattr("flash.serving.src.store.undeploy.get_authoritative", authoritative)
    result = asyncio.run(engine._unregister(record.org_id, record.model_id, "generation-old"))
    assert result["skipped_stale_generation"] is True
    ready = record.model_copy(update={"status": "ready"})
    engine.registry.upsert(ready)
    assert engine.registry.get(ready.org_id, ready.model_id) == ready


@pytest.mark.parametrize("unavailable", [False, True])
def test_cold_unregister_without_authoritative_record_leaves_registry_untouched(
    monkeypatch, unavailable
):
    record = _revision("a" * 40)
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()

    async def authoritative(org_id, model_id):
        if unavailable:
            raise RuntimeError("storage unavailable")
        return

    monkeypatch.setattr("flash.serving.src.store.undeploy.get_authoritative", authoritative)
    call = engine._unregister(record.org_id, record.model_id, "generation-old")
    if unavailable:
        with pytest.raises(RuntimeError, match="storage unavailable"):
            asyncio.run(call)
    else:
        assert asyncio.run(call)["removed"] is None
    engine.registry.upsert(record)
    assert engine.registry.get(record.org_id, record.model_id) == record


def test_request_loaded_adapter_is_registered_once_and_evicted_on_undeploy(tmp_path):
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-one"})
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    resident = set()
    added = []
    removed = []

    class _Engine:
        async def add_lora(self, request):
            assert request.lora_int_id not in resident
            resident.add(request.lora_int_id)
            added.append(request.lora_int_id)
            return True

        async def remove_lora(self, int_id):
            resident.remove(int_id)
            removed.append(int_id)
            return True

    async def local_path(record):
        return tmp_path

    engine.engine = _Engine()
    engine._ensure_adapter_local_locked = local_path

    async def exercise():
        payload = internal_adapter_payload(record)
        first, _ = await engine._lora_request(record.model_id, payload)
        assert first.lora_int_id in resident
        second, _ = await engine._lora_request(record.model_id, payload)
        assert second is first
        await engine._unregister(record.org_id, record.model_id, "generation-one")
        assert not resident
        # the same immutable checkpoint can subsequently be deployed again.
        await engine._register(payload, "generation-two")
        assert resident == {first.lora_int_id}

    asyncio.run(exercise())
    assert len(added) == 2
    assert removed == added[:1]
