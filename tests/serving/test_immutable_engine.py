from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serving.src.engine_support import (
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _load_adapters_for_base,
)
from flash.serving.src.lora_engine import _LoraEngineImpl, _LoraEntry
from flash.serving.src.registry import AdapterRegistry
from flash.serving.src.schemas import AdapterRecord

BASE_MODEL = "Qwen/Qwen3.5-9B"
RUN_ID = "flash-1234567890-abcdef12"


def _revision(sha: str, *, status: str = "ready") -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{RUN_ID}@step-20.{sha}",
            "repo_id": "org/run",
            "org_id": "org-1",
            "base_model": BASE_MODEL,
            "subfolder": "checkpoints/step-20",
            "repo_type": "model",
            "checkpoint": f"{RUN_ID}/step-20",
            "private": True,
            "thinking": False,
            "status": status,
            "metadata": {
                "record_type": "revision",
                "run_id": RUN_ID,
                "checkpoint_step": 20,
                "hf_revision": sha,
            },
        }
    )


def _alias(target: AdapterRecord) -> AdapterRecord:
    return target.model_copy(
        update={
            "adapter_id": RUN_ID,
            "checkpoint": None,
            "metadata": {
                "record_type": "alias",
                "run_id": RUN_ID,
                "alias_of": target.adapter_id,
            },
        }
    )


def test_distinct_hub_shas_have_distinct_source_and_cache_identity(tmp_path: Path) -> None:
    first = _revision("a" * 40)
    second = _revision("b" * 40)
    assert _adapter_source_ident(first) != _adapter_source_ident(second)
    assert _adapter_source_cache_dir(tmp_path, first) != _adapter_source_cache_dir(tmp_path, second)


def test_engine_hydration_excludes_aliases_legacy_and_disabled(monkeypatch) -> None:
    ready = _revision("a" * 40)
    records = [
        ready,
        _alias(ready),
        AdapterRecord.model_validate(
            {
                "adapter_id": "legacy",
                "repo_id": "org/legacy",
                "base_model": BASE_MODEL,
                "thinking": False,
            }
        ),
        _revision("b" * 40, status="disabled"),
    ]
    monkeypatch.setattr("flash.serving.src.persistence.load_adapters", lambda settings: records)
    assert _load_adapters_for_base(object(), BASE_MODEL) == [ready]


def test_unregister_skips_cleanup_for_a_newer_deployment_generation() -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-new"})
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine.registry.upsert(record, revive=True)
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    evicted: list[str] = []

    async def _evict(adapter_id: str) -> None:
        evicted.append(adapter_id)

    engine._evict_loaded_lora = _evict

    async def _exercise() -> tuple[dict, dict, dict]:
        legacy = await engine._unregister(record.adapter_id)
        stale = await engine._unregister(record.adapter_id, "generation-old")
        removed = await engine._unregister(record.adapter_id, "generation-new")
        return legacy, stale, removed

    legacy, stale, removed = asyncio.run(_exercise())

    assert legacy["skipped_stale_generation"] is True
    assert stale["skipped_stale_generation"] is True
    assert removed["removed"] == record.adapter_id
    assert engine.registry.get(record.adapter_id) is None
    assert evicted == [record.adapter_id]


def test_add_lora_exception_releases_new_entry_for_retry(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    engine = _LoraEngineImpl()
    engine._pin_loras = False
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

    assert record.adapter_id not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.adapter_id].state == "loaded"


def test_rejected_new_lora_releases_entry_and_reaches_engine_on_retry(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    engine = _LoraEngineImpl()
    engine._pin_loras = False
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

    assert record.adapter_id not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.adapter_id].state == "loaded"


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
        record.adapter_id: _LoraEntry(_adapter_source_ident(record), request, "loaded")
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
        asyncio.run(engine._evict_loaded_lora(record.adapter_id))

    entry = engine._entries()[record.adapter_id]
    assert entry.lora_request is request
    assert entry.state == "unconfirmed"
    with pytest.raises(RuntimeError, match="registration is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)


def test_reserved_lora_evict_releases_without_engine_removal() -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    engine = _LoraEngineImpl()
    engine._lora_entries = {
        record.adapter_id: _LoraEntry(_adapter_source_ident(record), request, "reserved")
    }
    removed: list[int] = []

    class _Engine:
        async def remove_lora(self, int_id: int) -> bool:
            removed.append(int_id)
            return True

    engine.engine = _Engine()

    asyncio.run(engine._evict_loaded_lora(record.adapter_id))

    assert record.adapter_id not in engine._entries()
    assert removed == []


def test_cached_lora_request_enforces_state_and_source_identity(tmp_path: Path) -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    source_ident = _adapter_source_ident(record)
    engine = _LoraEngineImpl()
    engine._lora_entries = {record.adapter_id: _LoraEntry(source_ident, request, "loaded")}

    assert engine._cached_lora_request_locked(record, tmp_path) is request

    engine._lora_entries[record.adapter_id] = _LoraEntry(source_ident, request, "unconfirmed")
    with pytest.raises(RuntimeError, match="registration is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)

    other_ident = ("other/repo", source_ident[1], source_ident[2], source_ident[3])
    engine._lora_entries[record.adapter_id] = _LoraEntry(other_ident, request, "loaded")
    with pytest.raises(RuntimeError, match="previous LoRA removal is unconfirmed"):
        engine._cached_lora_request_locked(record, tmp_path)


def test_unregister_tombstones_missing_record_for_expected_generation() -> None:
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
    evicted: list[str] = []

    async def _evict(adapter_id: str) -> None:
        evicted.append(adapter_id)

    engine._evict_loaded_lora = _evict

    result = asyncio.run(engine._unregister(stale_record.adapter_id, "generation-old"))

    assert result["removed"] == stale_record.adapter_id
    assert evicted == [stale_record.adapter_id]
    engine.registry.upsert(stale_record)
    assert engine.registry.get(stale_record.adapter_id) is None


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

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path)

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
    assert calls[0]["allow_patterns"] == [
        "checkpoints/step-20/**",
        "checkpoints/step-20/*",
    ]


def test_runtime_containment_rejects_schema_bypass(monkeypatch, tmp_path: Path) -> None:
    record = _revision("a" * 40).model_copy(update={"subfolder": "../escape"})
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("flash.serving.src.settings.ADAPTER_CACHE_DIR", tmp_path)

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
