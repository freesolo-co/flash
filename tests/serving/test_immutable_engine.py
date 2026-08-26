from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serving.src.engine.lora_engine import _LoraEngineImpl, _LoraEntry
from flash.serving.src.engine.support import (
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _load_adapters_for_base,
    _replica_adapter_cache_dir,
)
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.registry import AdapterRegistry

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
    monkeypatch.setattr(
        "flash.serving.src.store.persistence.load_adapters", lambda settings: records
    )
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
    assert legacy["cleanup_scope"] == "replica_local"
    assert stale["cleanup_scope"] == "replica_local"
    assert removed["removed"] == record.adapter_id
    assert removed["cleanup_scope"] == "replica_local"
    assert removed["engine_replica_id"] == stale["engine_replica_id"]
    assert engine.registry.get(record.adapter_id) is None
    assert evicted == [record.adapter_id]


def test_unregister_never_removes_lora_during_blocked_stream_lifetime(tmp_path: Path) -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-1"})
    request = SimpleNamespace(lora_int_id=42, lora_name=record.adapter_id)
    release = asyncio.Event()
    iterator_closed = asyncio.Event()
    removed: list[int] = []

    class _Engine:
        def generate(self, *_args: object, **_kwargs: object):
            async def outputs():
                try:
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                index=0,
                                text="partial",
                                finish_reason=None,
                                token_ids=[1],
                            )
                        ],
                        prompt_token_ids=[1],
                        num_cached_tokens=0,
                    )
                    await release.wait()
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                index=0,
                                text="done",
                                finish_reason="stop",
                                token_ids=[2],
                            )
                        ],
                        prompt_token_ids=[1],
                        num_cached_tokens=0,
                    )
                finally:
                    iterator_closed.set()

            return outputs()

        async def remove_lora(self, int_id: int) -> bool:
            removed.append(int_id)
            return True

    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine.registry.upsert(record, revive=True)
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    engine.registry.set_local_path(record, tmp_path)
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine._lora_entries = {
        record.adapter_id: _LoraEntry(_adapter_source_ident(record), request, "loaded")
    }
    engine.engine = _Engine()
    engine.reasoning_parser = None
    engine._prompt_cache_size = 0

    async def ensure_local(_record: AdapterRecord) -> Path:
        return tmp_path

    async def prompt_input(*_args: object) -> dict[str, list[int]]:
        return {"prompt_token_ids": [1]}

    engine._ensure_adapter_local_locked = ensure_local  # type: ignore[method-assign]
    engine._prepare_prompt_input = prompt_input  # type: ignore[method-assign]

    async def scenario() -> tuple[dict[str, object], list[dict[str, object]]]:
        stream = engine._stream_generate({"adapter_id": record.adapter_id, "prompt": "hi"})
        ready = await anext(stream)
        cleanup = await engine._unregister(record.adapter_id, "generation-1")
        assert removed == []
        with pytest.raises(ValueError, match="Unknown adapter id"):
            await engine._lora_request(record.adapter_id)
        release.set()
        remaining = [event async for event in stream]
        return cleanup, remaining

    cleanup, remaining = asyncio.run(scenario())

    assert cleanup["cleanup_scope"] == "replica_local"
    assert cleanup["removed"] == record.adapter_id
    assert [event["type"] for event in remaining] == [
        "delta",
        "delta",
        "choice_finished",
        "final",
    ]
    assert iterator_closed.is_set()
    assert removed == []


def test_generation_rejects_forwarded_revision_material_that_differs_from_local_identity() -> None:
    local = _revision("a" * 40).model_copy(update={"updated_at": "2026-08-26T00:00:02Z"})
    forwarded = local.model_copy(
        update={"repo_id": "org/different", "updated_at": "2026-08-26T00:00:01Z"}
    )
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine.registry.upsert(local, revive=True)
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()

    with pytest.raises(ValueError, match="identity differs"):
        asyncio.run(engine._lora_request(local.adapter_id, forwarded.model_dump(by_alias=True)))


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


def test_loaded_lora_cleanup_is_replica_local_and_never_calls_remove_lora() -> None:
    record = _revision("a" * 40)
    request = SimpleNamespace(lora_int_id=42)
    engine = _LoraEngineImpl()
    engine._lora_entries = {
        record.adapter_id: _LoraEntry(_adapter_source_ident(record), request, "loaded")
    }
    removed: list[int] = []

    class _Engine:
        async def remove_lora(self, int_id: int) -> bool:
            removed.append(int_id)
            return True

    engine.engine = _Engine()

    asyncio.run(engine._evict_loaded_lora(record.adapter_id))

    assert engine._entries()[record.adapter_id].lora_request is request
    assert removed == []


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
    shared_snapshot = tmp_path / "hub" / "snapshots" / ("a" * 40)
    adapter_dir = shared_snapshot / "checkpoints/step-20"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(shared_snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "replicas")
    monkeypatch.setattr("flash.serving.src.store.settings.HF_HUB_CACHE_DIR", tmp_path / "hub")

    engine = _LoraEngineImpl()
    engine.registry = AdapterRegistry()
    engine.settings = SimpleNamespace(hf_api_key="token")
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._source_paths = {}
    engine._lora_entries = {}

    path = asyncio.run(engine._ensure_adapter_local_locked(record))
    relative = path.relative_to(tmp_path / "replicas")
    assert relative.parts[0].startswith("replica-")
    assert relative.parts[1] == "sources"
    assert relative.parts[-2:] == ("checkpoints", "step-20")
    assert calls[0]["repo_id"] == "org/run"
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["cache_dir"] == str(tmp_path / "hub")
    assert "local_dir" not in calls[0]
    assert calls[0]["allow_patterns"] == [
        "checkpoints/step-20/**",
        "checkpoints/step-20/*",
    ]


def test_two_engine_replicas_materialize_same_revision_without_shared_final_path(
    monkeypatch, tmp_path: Path
) -> None:
    record = _revision("a" * 40)
    barrier = threading.Barrier(2)
    cache_dirs: list[Path] = []
    shared_snapshot = tmp_path / "hub" / "snapshots" / ("a" * 40)
    adapter_dir = shared_snapshot / "checkpoints/step-20"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    def snapshot_download(**kwargs: object) -> str:
        cache_dirs.append(Path(str(kwargs["cache_dir"])))
        barrier.wait(timeout=2)
        return str(shared_snapshot)

    async def run() -> tuple[Path, Path]:
        engines = [_LoraEngineImpl(), _LoraEngineImpl()]
        for engine in engines:
            engine.registry = AdapterRegistry()
            engine.settings = SimpleNamespace(hf_api_key="token")
            engine._source_locks = {}
            engine._source_locks_guard = asyncio.Lock()
            engine._source_paths = {}
            engine._lora_entries = {}
        tasks = [
            asyncio.create_task(engine._ensure_adapter_local_locked(record)) for engine in engines
        ]
        return tuple(await asyncio.gather(*tasks))  # type: ignore[return-value]

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("flash.serving.src.store.settings.ADAPTER_CACHE_DIR", tmp_path / "replicas")
    monkeypatch.setattr("flash.serving.src.store.settings.HF_HUB_CACHE_DIR", tmp_path / "hub")

    first, second = asyncio.run(run())

    assert first != second
    assert cache_dirs == [tmp_path / "hub", tmp_path / "hub"]
    assert all(
        path.relative_to(tmp_path / "replicas").parts[0].startswith("replica-")
        for path in (first, second)
    )
    assert (first / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (second / "adapter_model.safetensors").read_bytes() == b"weights"


def test_replica_adapter_cache_identity_is_stable_and_distinct(tmp_path: Path) -> None:
    first = _replica_adapter_cache_dir(tmp_path, "replica-a")
    assert first == _replica_adapter_cache_dir(tmp_path, "replica-a")
    assert first != _replica_adapter_cache_dir(tmp_path, "replica-b")


def test_runtime_containment_rejects_schema_bypass(monkeypatch, tmp_path: Path) -> None:
    record = _revision("a" * 40).model_copy(update={"subfolder": "../escape"})
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
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
