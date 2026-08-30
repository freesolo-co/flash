from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.engine.lora_engine import _LoraEngineImpl, _LoraEntry
from flash.serving.src.engine.lora_lifecycle import (
    AdapterCacheCapacityError,
    ReplicaSourceCache,
)
from flash.serving.src.engine.support import (
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _load_adapters_for_base,
    _replica_adapter_cache_dir,
)
from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload
from flash.serving.src.store.registry import AdapterRegistry

BASE_MODEL = "Qwen/Qwen3.5-9B"
RUN_ID = "flash-1234567890-abcdef12"


def _revision(sha: str, *, status: str = "ready") -> AdapterRecord:
    values = {
        "adapter_id": f"{RUN_ID}/step-20",
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
    values["artifact_fingerprint"] = immutable_binding_fingerprint(values)
    return AdapterRecord.model_validate(values)


def test_distinct_hub_shas_have_distinct_source_and_cache_identity(tmp_path: Path) -> None:
    first = _revision("a" * 40)
    second = _revision("b" * 40)
    assert _adapter_source_ident(first) != _adapter_source_ident(second)
    assert _adapter_source_cache_dir(tmp_path, first) != _adapter_source_cache_dir(tmp_path, second)


def test_engine_hydration_excludes_base_models_and_disabled_checkpoints(monkeypatch) -> None:
    ready = _revision("a" * 40)
    records = [
        ready,
        AdapterRecord.model_validate(
            {
                "adapter_id": BASE_MODEL,
                "repo_id": BASE_MODEL,
                "base_model": BASE_MODEL,
                "serve_base_model": True,
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
    evicted: list[tuple[str, str]] = []

    async def _evict(adapter_key: tuple[str, str]) -> None:
        evicted.append(adapter_key)

    engine._evict_loaded_lora = _evict

    async def _exercise() -> tuple[dict, dict, dict]:
        legacy = await engine._unregister(record.org_id, record.adapter_id)
        stale = await engine._unregister(record.org_id, record.adapter_id, "generation-old")
        removed = await engine._unregister(record.org_id, record.adapter_id, "generation-new")
        return legacy, stale, removed

    legacy, stale, removed = asyncio.run(_exercise())

    assert legacy["skipped_stale_generation"] is True
    assert stale["skipped_stale_generation"] is True
    assert removed["removed"] == record.adapter_id
    assert engine.registry.get(record.org_id, record.adapter_id) is None
    assert evicted == [record.storage_key]


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

    assert record.storage_key not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.storage_key].state == "loaded"


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

    assert record.storage_key not in engine._entries()
    asyncio.run(engine._add_lora_locked(record, tmp_path))
    assert calls == 2
    assert engine._entries()[record.storage_key].state == "loaded"


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
    evicted: list[tuple[str, str]] = []

    async def _evict(adapter_key: tuple[str, str]) -> None:
        evicted.append(adapter_key)

    engine._evict_loaded_lora = _evict

    result = asyncio.run(
        engine._unregister(stale_record.org_id, stale_record.adapter_id, "generation-old")
    )

    assert result["removed"] == stale_record.adapter_id
    assert evicted == [stale_record.storage_key]
    engine.registry.upsert(stale_record)
    assert engine.registry.get(stale_record.org_id, stale_record.adapter_id) is None


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
    # the immutable hub snapshot stays shared; only the final directory is replica-local.
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


async def _await_accept(accepted: asyncio.Event, *tasks: asyncio.Task) -> None:
    """Wait for the fake engine to accept, surfacing a task failure instead of deadlocking."""
    waiter = asyncio.ensure_future(accepted.wait())
    try:
        await asyncio.wait([waiter, *tasks], timeout=10, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
    for task in tasks:
        if task.done():
            task.result()
    if not accepted.is_set():
        raise AssertionError("engine.generate was never reached")


def _in_flight_engine(record: AdapterRecord, tmp_path: Path) -> tuple[_LoraEngineImpl, Path]:
    root = tmp_path / "replica"
    source = _adapter_source_cache_dir(root, record)
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text("")
    (source / "adapter_model.safetensors").write_bytes(b"old")
    engine = _LoraEngineImpl()
    engine.base_model = BASE_MODEL
    engine.registry = AdapterRegistry()
    engine.registry.upsert(record, revive=True)
    engine.registry.set_local_path(record, source)
    engine._adapter_cache_dir = root
    engine._source_cache = ReplicaSourceCache(root, max_bytes=1024)
    engine._source_paths = {_adapter_source_ident(record): source}
    engine._lora_entries = {}
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine._source_locks = {}
    engine._source_locks_guard = asyncio.Lock()
    engine._pin_loras = False
    return engine, source


def _single_output_engine(accepted: asyncio.Event, release: asyncio.Event) -> object:
    class _Engine:
        def generate(self, *_args: object, **_kwargs: object):
            accepted.set()

            async def outputs():
                await release.wait()
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            text="done",
                            finish_reason="stop",
                            token_ids=[2],
                            logprobs=None,
                        )
                    ],
                    prompt_token_ids=[1],
                    num_cached_tokens=0,
                )

            return outputs()

    return _Engine()


def _wire_generation(engine: _LoraEngineImpl, monkeypatch) -> None:
    engine.reasoning_parser = None
    engine._prompt_cache_size = 0

    async def prompt_input(*_args: object) -> dict[str, list[int]]:
        return {"prompt_token_ids": [1]}

    engine._prepare_prompt_input = prompt_input  # type: ignore[method-assign]
    monkeypatch.setattr(
        "flash.serving.src.engine.generation._sampling_params",
        lambda *_args: SimpleNamespace(),
    )


def test_in_flight_unregister_retains_id_and_first_output_marks_exact_source_loaded(
    monkeypatch, tmp_path: Path
) -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-1"})
    engine, source = _in_flight_engine(record, tmp_path)
    replacement = record.model_copy(update={"repo_id": "org/replacement"})
    key = record.storage_key

    accepted = asyncio.Event()
    release_output = asyncio.Event()
    engine.engine = _single_output_engine(accepted, release_output)
    _wire_generation(engine, monkeypatch)

    async def scenario() -> None:
        await engine._source_cache.bind_current(key, _adapter_source_ident(record), source)
        generation = asyncio.create_task(
            engine._generate(
                {"adapter_id": record.adapter_id, "prompt": "hi"},
                internal_adapter_payload(record),
            )
        )
        await _await_accept(accepted, generation)
        # undeploy while a request holds the adapter: the id and its weights must survive.
        await engine._unregister(record.org_id, record.adapter_id, "generation-1")
        retained = engine._entries()[key]
        assert retained.in_flight == 1
        assert retained.tombstoned is True
        assert source.exists()
        with pytest.raises(RuntimeError, match="previous LoRA removal is unconfirmed"):
            engine._cached_lora_request_locked(replacement, tmp_path / "replacement")
        release_output.set()
        result = await generation
        assert result["text"] == "done"
        loaded = engine._entries()[key]
        assert loaded.state == "loaded"
        assert loaded.tombstoned is True
        assert loaded.in_flight == 0
        assert source.exists()

    asyncio.run(scenario())


def test_lazy_redeploy_reactivates_retained_loaded_lora_after_in_flight_unregister(
    monkeypatch, tmp_path: Path
) -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-1"})
    engine, source = _in_flight_engine(record, tmp_path)
    key = record.storage_key

    accepted = asyncio.Event()
    release_output = asyncio.Event()
    engine.engine = _single_output_engine(accepted, release_output)
    _wire_generation(engine, monkeypatch)

    async def scenario() -> None:
        await engine._source_cache.bind_current(key, _adapter_source_ident(record), source)
        generation = asyncio.create_task(
            engine._generate(
                {"adapter_id": record.adapter_id, "prompt": "hi"},
                internal_adapter_payload(record),
            )
        )
        await _await_accept(accepted, generation)
        await engine._unregister(record.org_id, record.adapter_id, "generation-1")
        release_output.set()
        await generation
        assert engine._entries()[key].tombstoned is True

        # redeploying the SAME checkpoint must revive the retained weights rather than wait on a
        # confirmed vllm removal that no longer has a holder.
        engine.registry.upsert(record, revive=True)
        engine.registry.set_local_path(record, source)
        lora_request, resolved = await engine._lora_request(
            record.adapter_id, internal_adapter_payload(record)
        )
        assert resolved.adapter_id == record.adapter_id
        revived = engine._entries()[key]
        assert revived.tombstoned is False
        assert revived.state == "loaded"
        assert revived.lora_request is lora_request

    asyncio.run(scenario())


def test_two_in_flight_requests_both_release_before_deferred_cleanup(
    monkeypatch, tmp_path: Path
) -> None:
    record = _revision("a" * 40).model_copy(update={"deployment_generation": "generation-1"})
    engine, source = _in_flight_engine(record, tmp_path)
    key = record.storage_key

    accepted_count = 0
    both_accepted = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()

    class _Engine:
        def generate(self, *_args: object, **_kwargs: object):
            nonlocal accepted_count
            accepted_count += 1
            gate = release_first if accepted_count == 1 else release_second
            if accepted_count == 2:
                both_accepted.set()

            async def outputs():
                await gate.wait()
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            text="done",
                            finish_reason="stop",
                            token_ids=[2],
                            logprobs=None,
                        )
                    ],
                    prompt_token_ids=[1],
                    num_cached_tokens=0,
                )

            return outputs()

    engine.engine = _Engine()
    _wire_generation(engine, monkeypatch)

    async def scenario() -> None:
        await engine._source_cache.bind_current(key, _adapter_source_ident(record), source)
        payload = internal_adapter_payload(record)
        first = asyncio.create_task(
            engine._generate({"adapter_id": record.adapter_id, "prompt": "a"}, payload)
        )
        second = asyncio.create_task(
            engine._generate({"adapter_id": record.adapter_id, "prompt": "b"}, payload)
        )
        await _await_accept(both_accepted, first, second)
        assert engine._entries()[key].in_flight == 2

        release_first.set()
        await first
        # one holder released, the other still owns the adapter: cleanup must stay deferred.
        assert engine._entries()[key].in_flight == 1
        assert source.exists()

        release_second.set()
        await second
        assert engine._entries()[key].in_flight == 0

    asyncio.run(scenario())


def test_source_cache_refuses_a_materialization_over_its_byte_ceiling(tmp_path: Path) -> None:
    root = tmp_path / "replica"
    root.mkdir()
    cache = ReplicaSourceCache(root, max_bytes=64)
    live = root / "sources" / "live"
    live.mkdir(parents=True)
    (live / "adapter_model.safetensors").write_bytes(b"x" * 64)
    incoming = root / "sources" / "incoming"
    live_ident = ("org/live", "model", "a" * 40, "b" * 64, None)
    incoming_ident = ("org/incoming", "model", "c" * 40, "d" * 64, None)

    async def scenario() -> None:
        await cache.bind_current(("org-1", "run/step-1"), live_ident, live)
        with pytest.raises(AdapterCacheCapacityError):
            async with cache.materializing(incoming_ident, incoming, 64):
                pass
        # the live source is protected, so it must survive the refused reservation.
        assert live.exists()

    asyncio.run(scenario())


def test_source_cache_reclaims_only_unprotected_sources(tmp_path: Path) -> None:
    root = tmp_path / "replica"
    root.mkdir()
    cache = ReplicaSourceCache(root, max_bytes=64)
    idle = root / "sources" / "idle"
    idle.mkdir(parents=True)
    (idle / "adapter_model.safetensors").write_bytes(b"x" * 64)
    idle_ident = ("org/idle", "model", "a" * 40, "b" * 64, None)
    live = root / "sources" / "live"
    live.mkdir(parents=True)
    (live / "adapter_model.safetensors").write_bytes(b"y" * 64)
    live_ident = ("org/live", "model", "c" * 40, "d" * 64, None)

    async def scenario() -> None:
        await cache.bind_current(("org-1", "idle/step-1"), idle_ident, idle)
        await cache.release_current(("org-1", "idle/step-1"))
        await cache.bind_current(("org-1", "live/step-1"), live_ident, live)
        assert live.exists()
        assert not idle.exists()

    asyncio.run(scenario())


def test_source_cache_reclaims_a_source_once_its_loaded_reference_is_released(
    tmp_path: Path,
) -> None:
    """a loaded source must stop being protected once the engine drops its weights.

    `loaded` outliving the adapter is what pinned every served source for the life of the
    replica, so the budget filled with adapters nothing could reach.
    """

    root = tmp_path / "replica"
    root.mkdir()
    cache = ReplicaSourceCache(root, max_bytes=1024)
    source = root / "sources" / "served"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"z" * 64)
    ident = ("org/served", "model", "e" * 40, "f" * 64, None)
    key = ("org-1", "served/step-1")

    async def scenario() -> None:
        await cache.mark_loaded(key, ident, source)
        assert cache._states[ident].protected

        await cache.release_loaded(key, ident)
        assert not cache._states[ident].protected

        await cache.remove_if_unreferenced(ident)
        assert ident not in cache._states
        assert not source.exists()

    asyncio.run(scenario())


def test_source_cache_rejects_a_directory_outside_the_replica_root(tmp_path: Path) -> None:
    root = tmp_path / "replica"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cache = ReplicaSourceCache(root, max_bytes=1024)
    ident = ("org/escape", "model", "a" * 40, "b" * 64, None)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="outside this replica cache"):
            await cache.bind_current(("org-1", "run/step-1"), ident, outside)

    asyncio.run(scenario())


def test_exit_deletes_only_this_replicas_adapter_root(tmp_path: Path) -> None:
    mine = _replica_adapter_cache_dir(tmp_path, "replica-mine")
    theirs = _replica_adapter_cache_dir(tmp_path, "replica-theirs")
    for root in (mine, theirs):
        (root / "sources" / "s").mkdir(parents=True)
        (root / "sources" / "s" / "adapter_model.safetensors").write_bytes(b"w")

    engine = _LoraEngineImpl()
    engine._adapter_cache_dir = mine
    engine._source_cache = ReplicaSourceCache(mine, max_bytes=1024)

    asyncio.run(engine._exit())

    # container shutdown must not reclaim a sibling replica's materialization.
    assert not mine.exists()
    assert (theirs / "sources" / "s" / "adapter_model.safetensors").exists()
