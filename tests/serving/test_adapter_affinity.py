"""Reserved-slot routing is sticky per adapter and bounded by each worker's LoRA slots."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from flash.serving.lambda_cloud.affinity import AdapterAffinity, rendezvous

SLOTS = list(range(1, 9))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_rendezvous_moves_only_the_adapters_of_a_removed_slot():
    keys = [f"org\0run-{i}/final" for i in range(400)]
    before = {key: rendezvous(key, SLOTS)[0] for key in keys}
    after = {key: rendezvous(key, [s for s in SLOTS if s != 3])[0] for key in keys}
    moved = {key for key in keys if before[key] != after[key]}
    assert moved == {key for key in keys if before[key] == 3}
    # Adapters spread over every slot.
    assert set(before.values()) == set(SLOTS)


@pytest.mark.anyio
async def test_one_adapter_stays_on_its_slot_and_spills_only_when_saturated():
    affinity = AdapterAffinity(max_adapters=16, spill_after=4)
    key = "org\0run-a/final"
    first, second = rendezvous(key, SLOTS)[:2]
    async with affinity.route(key, SLOTS, weight=3) as slot:
        assert slot == first
        async with affinity.route(key, SLOTS) as again:
            assert again == first  # 4 in flight now: saturated
            async with affinity.route(key, SLOTS) as spill:
                assert spill == second
                async with affinity.route(key, SLOTS) as next_one:
                    assert next_one == second  # still less loaded than the first
    assert affinity.load(first) == affinity.load(second) == 0
    assert not affinity.adapters(first)


@pytest.mark.anyio
async def test_a_worker_scores_at_most_max_adapters_and_others_wait_for_room():
    affinity = AdapterAffinity(max_adapters=2, spill_after=100)
    slots = [1]
    holders = [asyncio.Event() for _ in range(2)]
    release = asyncio.Event()

    async def hold(key, ready):
        async with affinity.route(key, slots):
            ready.set()
            await release.wait()

    tasks = [asyncio.create_task(hold(f"a{i}", holders[i])) for i in range(2)]
    for ready in holders:
        await ready.wait()
    assert affinity.adapters(1) == {"a0", "a1"}
    # The same adapter shares its slot; a third waits until one finishes.
    async with affinity.route("a0", slots) as slot:
        assert slot == 1
    waiting = asyncio.create_task(_route_once(affinity, "a2", slots))
    await asyncio.sleep(0.02)
    assert not waiting.done()
    release.set()
    assert await asyncio.wait_for(waiting, 1) == 1
    await asyncio.gather(*tasks)


@pytest.mark.anyio
async def test_a_full_preferred_slot_sends_a_new_adapter_to_its_next_slot():
    affinity = AdapterAffinity(max_adapters=1, spill_after=100)
    new = "new-adapter"
    preferred, fallback = rendezvous(new, [1, 2])
    async with affinity.route("busy", [preferred]), affinity.route(new, [1, 2]) as slot:
        assert slot == fallback


@pytest.mark.anyio
async def test_an_adapter_waits_at_most_wait_seconds_then_takes_its_first_slot():
    affinity = AdapterAffinity(max_adapters=1, spill_after=100, wait_seconds=0.05)
    async with affinity.route("busy", [1]):
        assert await asyncio.wait_for(_route_once(affinity, "late", [1]), 1) == 1


async def _route_once(affinity, key, slots):
    async with affinity.route(key, slots) as slot:
        return slot


# the serving pool -------------------------------------------------------------------------------


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module(load_modal_app_under_stub):
    modal_stub = MagicMock(name="modal")
    for name in ("concurrent", "method", "enter", "exit", "asgi_app"):
        getattr(modal_stub, name).side_effect = _passthrough_decorator
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    for name in ("cls", "function", "local_entrypoint"):
        getattr(app_mock, name).side_effect = _passthrough_decorator
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()
    return load_modal_app_under_stub(modal_stub)


class _Health:
    def __init__(self, usable):
        self.slots = usable
        self.round_robin = 0

    async def usable(self, _model):
        return list(self.slots)

    async def next_slot(self, _model):
        self.round_robin += 1
        return self.slots[self.round_robin % len(self.slots)]


@pytest.mark.anyio
async def test_pool_routes_a_checkpoint_to_its_sticky_slot_among_healthy_ones(
    modal_app_module, monkeypatch
):
    model = "Qwen/Qwen3.5-9B-Base"

    def engine_for(slot):
        return lambda base_model: SimpleNamespace(slot=slot)

    engines = tuple(engine_for(slot) for slot in SLOTS)
    monkeypatch.setattr(modal_app_module, "RESERVED_ENGINES", engines)
    monkeypatch.setattr(modal_app_module, "_reserved", lambda _model: True)
    pool = object.__new__(modal_app_module._ModalEnginePool)
    pool._reserved_health = _Health(SLOTS)
    pool._affinity = {}
    record = SimpleNamespace(model_id="run-a/final", org_id="org-1", serve_base_model=False)
    seen = set()
    for _ in range(5):
        async with pool._routed_engine(model, record) as engine:
            seen.add(engine.slot)
    key = modal_app_module._affinity_key(record)
    assert seen == {rendezvous(key, SLOTS)[0]}
    affinity = pool._affinity[model]
    # Read from the engine config: 9B Base holds 16 LoRAs, admits 63 requests per worker.
    assert (affinity.max_adapters, affinity.spill_after) == (16, 63)
    # A quarantined slot is not routable; its adapters move to their next-ranked slot.
    sticky = rendezvous(key, SLOTS)[0]
    pool._reserved_health.slots = [s for s in SLOTS if s != sticky]
    async with pool._routed_engine(model, record) as engine:
        assert engine.slot == rendezvous(key, SLOTS)[1]
    # Base-model serves have no adapter to keep warm and stay round-robin.
    base = SimpleNamespace(model_id=model, org_id=None, serve_base_model=True)
    async with pool._routed_engine(model, base) as engine:
        assert pool._reserved_health.round_robin == 1
