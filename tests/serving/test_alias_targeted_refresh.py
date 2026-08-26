from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from fastapi import HTTPException

from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.lookup import AdapterLookup

BASE_MODEL = "Qwen/Qwen3.5-9B"


def _revision(run_id: str, sha: str) -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{sha}",
            "repo_id": f"org/{run_id}",
            "org_id": "org-1",
            "base_model": BASE_MODEL,
            "checkpoint": run_id,
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
            },
        }
    )


def _alias(target: AdapterRecord, **updates: Any) -> AdapterRecord:
    record = target.model_copy(
        deep=True,
        update={
            "adapter_id": target.run_id,
            "metadata": {
                "record_type": "alias",
                "run_id": target.run_id,
                "alias_of": target.adapter_id,
            },
        },
    )
    return record.model_copy(update=updates)


def test_repeated_alias_requests_use_targeted_reads_and_observe_move() -> None:
    first = _revision("run", "a" * 40)
    second = _revision("run", "b" * 40)
    state = {first.adapter_id: first, second.adapter_id: second, "run": _alias(first)}
    reads: list[str] = []
    reloads = 0

    def reload_records() -> list[AdapterRecord]:
        nonlocal reloads
        reloads += 1
        return list(state.values())

    def lookup_record(adapter_id: str) -> AdapterRecord | None:
        reads.append(adapter_id)
        return state.get(adapter_id)

    lookup = AdapterLookup(
        AdapterRouter([first, _alias(first)]),
        reload_records,
        lookup_record=lookup_record,
    )

    async def scenario() -> tuple[str, str]:
        initial = (await lookup.resolve("run"))[1].adapter_id
        state["run"] = _alias(second)
        moved = (await lookup.resolve("run"))[1].adapter_id
        return initial, moved

    assert asyncio.run(scenario()) == (first.adapter_id, second.adapter_id)
    assert reads == [
        "run",
        first.adapter_id,
        "run",
        "run",
        second.adapter_id,
        "run",
    ]
    assert reloads == 0


@pytest.mark.parametrize("invalid", ["missing", "disabled", "malformed", "target_missing"])
def test_invalid_alias_or_target_fails_closed(invalid: str) -> None:
    target = _revision("run", "a" * 40)
    alias = _alias(target)
    state: dict[str, AdapterRecord | None] = {"run": alias, target.adapter_id: target}
    if invalid == "missing":
        state["run"] = None
    elif invalid == "disabled":
        state["run"] = alias.model_copy(update={"status": "disabled"})
    elif invalid == "malformed":
        state["run"] = target
    else:
        state[target.adapter_id] = None

    router = AdapterRouter([alias, target])
    lookup = AdapterLookup(
        router,
        list,
        lookup_record=lambda adapter_id: state.get(adapter_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lookup.resolve("run"))

    assert exc_info.value.status_code == 404
    assert router.resolve("run") is None


def test_alias_move_between_exact_reads_never_returns_or_caches_stale_target() -> None:
    first = _revision("run", "a" * 40)
    second = _revision("run", "b" * 40)
    alias_first = _alias(first)
    alias_second = _alias(second)
    reads = 0

    def lookup_record(adapter_id: str) -> AdapterRecord | None:
        nonlocal reads
        reads += 1
        if reads == 1:
            return alias_first
        if reads == 2:
            return first
        if adapter_id == "run":
            return alias_second
        return second

    router = AdapterRouter([alias_first, first])
    lookup = AdapterLookup(router, list, lookup_record=lookup_record)

    resolved = asyncio.run(lookup.resolve("run"))

    assert resolved == (alias_second, second)
    assert router.resolve("run") == (alias_second, second)
    assert router.get(first.adapter_id) is None


def test_delayed_targeted_refresh_cannot_overwrite_newer_state() -> None:
    first = _revision("run", "a" * 40)
    second = _revision("run", "b" * 40)
    alias_first = _alias(first)
    alias_second = _alias(second)
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    def lookup_record(adapter_id: str) -> AdapterRecord | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            release_first.wait(timeout=2)
            return alias_first
        if adapter_id == first.adapter_id:
            return first
        if adapter_id == "run":
            return alias_second
        return second

    router = AdapterRouter([alias_first, first])
    lookup = AdapterLookup(router, list, lookup_record=lookup_record)

    async def scenario() -> tuple[str, str]:
        older = asyncio.create_task(lookup.resolve("run"))
        await asyncio.to_thread(first_started.wait, 2)
        newer = asyncio.create_task(lookup.resolve("run"))
        release_first.set()
        old_result = await older
        new_result = await newer
        return old_result[1].adapter_id, new_result[1].adapter_id

    old_id, new_id = asyncio.run(scenario())
    assert old_id == second.adapter_id
    assert new_id == second.adapter_id
    assert router.resolve("run")[1].adapter_id == second.adapter_id  # type: ignore[index]
    assert router.get(first.adapter_id) is None
