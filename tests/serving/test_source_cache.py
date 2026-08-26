from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.engine.lora_lifecycle import AdapterCacheCapacityError, ReplicaSourceCache
from flash.serving.src.engine.support import _adapter_source_cache_dir
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.registry import AdapterRegistry

BASE_MODEL = "Qwen/Qwen3.5-9B"


def _ident(name: str) -> tuple[str, str, str, str | None]:
    return (f"org/{name}", "model", name * 40, None)


def _source(root: Path, name: str, size: int = 1) -> Path:
    directory = root / "sources" / name
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_bytes(b"x" * size)
    return directory


def _record(name: str) -> AdapterRecord:
    sha = name * 40
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"run@step-1.{sha}",
            "repo_id": f"org/{name}",
            "org_id": "org-1",
            "base_model": BASE_MODEL,
            "checkpoint": "run/step-1",
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": "run",
                "checkpoint_step": 1,
                "hf_revision": sha,
            },
        }
    )


def test_byte_cap_evicts_lru_inactive_and_preserves_live_sources(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=12)
        inactive = _source(root, "inactive")
        active = _source(root, "active")
        current = _source(root, "current")
        loaded = _source(root, "loaded")
        incoming = _source(root, "incoming")

        await cache.bind_current("inactive", _ident("i"), inactive)
        await cache.release_current("inactive")
        await cache.bind_current("active", _ident("a"), active)
        await cache.bind_current("current", _ident("c"), current)
        await cache.bind_current("loaded", _ident("l"), loaded)
        await cache.mark_loaded("loaded", _ident("l"), loaded)
        await cache.release_current("loaded")

        async with cache.active(_ident("a"), active):
            await cache.release_current("active")
            await cache.bind_current("incoming", _ident("n"), incoming)
            assert not inactive.exists()
            assert active.exists()
            assert current.exists()
            assert loaded.exists()
            assert incoming.exists()

    asyncio.run(scenario())


def test_failed_add_lora_removes_unreferenced_materialization(monkeypatch, tmp_path: Path) -> None:
    record = _record("a")
    root = tmp_path / "replica"
    source = _adapter_source_cache_dir(root, record)
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text("{}")
    (source / "adapter_model.safetensors").write_bytes(b"x")
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine._adapter_cache_dir = root
    engine._source_cache = ReplicaSourceCache(root, max_bytes=1024)
    engine._source_paths = {}
    engine._lora_entries = {}
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine._pin_loras = False
    engine.engine = SimpleNamespace(add_lora=_raise_add_lora)

    async def ensure(target: AdapterRecord) -> Path:
        await engine._bind_source_path(target, source)
        return source

    monkeypatch.setattr(engine, "_ensure_adapter_local_locked", ensure)

    with pytest.raises(RuntimeError, match="add failed"):
        asyncio.run(engine._register(record.model_dump(by_alias=True)))

    assert not source.exists()
    assert engine.registry.local_path(record) is None


async def _raise_add_lora(_request: Any) -> None:
    raise RuntimeError("add failed")


def test_reclaim_waits_for_active_lease_after_unregister(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=4)
        old = _source(root, "old")
        incoming = _source(root, "incoming")
        await cache.bind_current("adapter", _ident("o"), old)
        lease = cache.active(_ident("o"), old)
        await lease.__aenter__()
        await cache.release_current("adapter")
        with pytest.raises(AdapterCacheCapacityError):
            await cache.bind_current("incoming", _ident("n"), incoming)
        assert old.exists()
        await lease.__aexit__(None, None, None)
        incoming = _source(root, "incoming")
        await cache.bind_current("incoming", _ident("n"), incoming)
        assert not old.exists()
        assert incoming.exists()

    asyncio.run(scenario())


def test_byte_ceiling_fails_closed_when_only_live_sources_remain(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=4)
        loaded = _source(root, "loaded")
        incoming = _source(root, "incoming")
        await cache.bind_current("loaded", _ident("l"), loaded)
        await cache.mark_loaded("loaded", _ident("l"), loaded)
        await cache.release_current("loaded")
        with pytest.raises(AdapterCacheCapacityError):
            await cache.bind_current("incoming", _ident("n"), incoming)
        assert loaded.exists()
        assert not incoming.exists()

    asyncio.run(scenario())


def test_failed_same_id_replacement_keeps_old_binding_transactional(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=4)
        old = root / "sources" / "old"
        new = root / "sources" / "new"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        (old / "weights").write_bytes(b"old")
        (new / "weights").write_bytes(b"newer")
        await cache.bind_current("adapter", _ident("o"), old)

        with pytest.raises(AdapterCacheCapacityError):
            await cache.bind_current("adapter", _ident("n"), new)

        assert old.exists()
        assert not new.exists()
        assert cache._current["adapter"] == _ident("o")
        async with cache.active(_ident("o"), old):
            assert old.exists()

    asyncio.run(scenario())


def test_oversized_reservation_fails_before_copy_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=4)
        copied = False

        with pytest.raises(AdapterCacheCapacityError):
            async with cache.materializing(_ident("o"), root / "sources" / "oversized", 5):
                copied = True

        assert copied is False

    asyncio.run(scenario())


def test_concurrent_materializations_reserve_before_copy_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=4)
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        copied: list[str] = []

        async def first() -> None:
            async with cache.materializing(_ident("a"), root / "sources" / "first", 3):
                copied.append("first")
                first_entered.set()
                await release_first.wait()

        async def second() -> None:
            await first_entered.wait()
            with pytest.raises(AdapterCacheCapacityError):
                async with cache.materializing(_ident("b"), root / "sources" / "second", 3):
                    copied.append("second")
            release_first.set()

        await asyncio.gather(first(), second())
        assert copied == ["first"]

    asyncio.run(scenario())


def test_identical_source_concurrency_is_safe(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "replica"
        cache = ReplicaSourceCache(root, max_bytes=1024)
        source = _source(root, "shared")
        await asyncio.gather(
            cache.bind_current("first", _ident("s"), source),
            cache.bind_current("second", _ident("s"), source),
        )
        async with cache.active(_ident("s"), source):
            await asyncio.gather(
                cache.release_current("first"),
                cache.release_current("second"),
            )
            assert source.exists()

    asyncio.run(scenario())


def test_replica_roots_are_isolated_and_exit_removes_only_own_root(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_root = tmp_path / "replica-first"
        second_root = tmp_path / "replica-second"
        first_source = _source(first_root, "same")
        second_source = _source(second_root, "same")
        first = ReplicaSourceCache(first_root, max_bytes=1024)
        second = ReplicaSourceCache(second_root, max_bytes=1024)
        await first.bind_current("adapter", _ident("s"), first_source)
        await second.bind_current("adapter", _ident("s"), second_source)
        await first.close()
        assert not first_root.exists()
        assert second_root.exists()
        assert second_source.exists()

    asyncio.run(scenario())
