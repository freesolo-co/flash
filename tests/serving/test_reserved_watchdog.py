"""Reserved 9B Base H100s: independent registration, quarantine and automatic recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud import reserved_watchdog as wd
from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest


@pytest.fixture
def anyio_backend():
    return "asyncio"


class MemoryStore:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get(self, slot):
        return self.data.get(slot)

    async def put(self, slot, value):
        self.data[slot] = value


def workers():
    return [
        {"slot": i, "instance_id": f"inst-{i}", "ip": f"8.8.8.{i}", "revision": "a" * 40, "tls": {}}
        for i in range(1, 5)
    ]


def test_quarantine_after_consecutive_failures_and_readmit_on_success():
    state, restart = wd.advance(wd.SlotState(), False, 0, "timeout")
    assert state.healthy
    assert not restart
    state, restart = wd.advance(state, False, 120, "timeout")
    assert not state.healthy
    assert not restart
    state, restart = wd.advance(state, True, 240)
    assert state.healthy
    assert state.failures == 0
    assert state.unhealthy_since is None


def test_reboot_after_sustained_failure_then_cooldown():
    state = wd.SlotState()
    now = 0
    while True:
        state, restart = wd.advance(state, False, now, "timeout")
        if restart:
            break
        now += wd.WATCHDOG_PERIOD_SECONDS
    assert now >= wd.RESTART_AFTER_UNHEALTHY_SECONDS
    assert state.restarted_at == now
    # Still dead while booting: no second reboot until the cooldown elapses.
    state, again = wd.advance(state, False, now + wd.WATCHDOG_PERIOD_SECONDS)
    assert not again
    state, again = wd.advance(state, False, now + wd.RESTART_COOLDOWN_SECONDS)
    assert again
    # A recovered worker is readmitted and keeps its reboot history.
    state, _ = wd.advance(state, True, now + wd.RESTART_COOLDOWN_SECONDS + 120)
    assert state.healthy
    assert state.restarted_at == now + wd.RESTART_COOLDOWN_SECONDS


def test_routable_slots_fail_open():
    bad = wd.SlotState(healthy=False).dump()
    assert wd.routable_slots({4: bad}, (1, 2, 3, 4)) == [1, 2, 3]
    assert wd.routable_slots(dict.fromkeys(range(1, 5), bad), (1, 2, 3, 4)) == [1, 2, 3, 4]


def test_canary_matches_worker_schemas():
    body = wd.canary_request(reserved.MODEL)
    record = AdapterRecord.model_validate(body["record"])
    payload = GenerateRequest.model_validate(body["payload"])
    assert record.serve_base_model
    assert record.model_id == reserved.MODEL
    assert payload.max_tokens == 1
    assert payload.generation_id == body["generation_id"]
    assert body["canary"] is True


@pytest.mark.anyio
async def test_probe_detects_worker_that_answers_health_but_never_generates(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)

    async def call(path, body=None):
        if path == "/health":
            return {"ok": True, "engine_dead": False}
        await asyncio.sleep(10)
        return None

    transport = MagicMock(call=call)
    assert await wd.probe(transport, reserved.MODEL) == (False, "timeout")


@pytest.mark.anyio
async def test_run_once_quarantines_and_reboots_only_the_hung_worker(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)

    def factory(ip, tls):
        hung = ip.endswith(".4")

        async def call(path, body=None):
            if path == "/health":
                return {"ok": True, "engine_dead": False}
            if hung:
                await asyncio.sleep(10)
            return {"request_id": body["generation_id"]}

        return MagicMock(call=call, aclose=AsyncMock())

    store, api = MemoryStore(), AsyncMock()
    now = [0.0]
    for _ in range(int(wd.RESTART_AFTER_UNHEALTHY_SECONDS / wd.WATCHDOG_PERIOD_SECONDS) + 1):
        report = await wd.run_once(
            workers(), store, api, reserved.MODEL, transport_factory=factory, clock=lambda: now[0]
        )
        now[0] += wd.WATCHDOG_PERIOD_SECONDS
    assert all(report[s]["healthy"] for s in (1, 2, 3))
    assert not report[4]["healthy"]
    api.call.assert_awaited_once_with(
        "POST", "instance-operations/restart", {"instance_ids": ["inst-4"]}
    )


@pytest.mark.anyio
async def test_health_view_caches_and_keeps_last_view_on_store_error():
    store = MemoryStore({4: wd.SlotState(healthy=False).dump()})
    now = [0.0]
    view = wd.HealthView(store, range(1, 5), clock=lambda: now[0])
    assert await view.usable() == [1, 2, 3]
    store.data.clear()
    assert await view.usable() == [1, 2, 3]  # cached
    store.get = AsyncMock(side_effect=RuntimeError("dict unavailable"))
    now[0] += wd.ROUTER_CACHE_SECONDS
    assert await view.usable() == [1, 2, 3]


def _hung_factory(hung_suffixes):
    def factory(ip, tls):
        hung = ip.endswith(tuple(hung_suffixes))

        async def call(path, body=None):
            if path == "/health":
                return {"ok": True, "engine_dead": False}
            if hung:
                await asyncio.sleep(10)
            return {"request_id": body["generation_id"]}

        return MagicMock(call=call, aclose=AsyncMock())

    return factory


async def _run_until_reboot_due(factory, store, api):
    now = [0.0]
    reports = []
    for _ in range(int(wd.RESTART_AFTER_UNHEALTHY_SECONDS / wd.WATCHDOG_PERIOD_SECONDS) + 2):
        reports.append(
            await wd.run_once(
                workers(),
                store,
                api,
                reserved.MODEL,
                transport_factory=factory,
                clock=lambda: now[0],
            )
        )
        now[0] += wd.WATCHDOG_PERIOD_SECONDS
    return reports


@pytest.mark.anyio
async def test_no_reboot_when_every_worker_fails_the_probe(monkeypatch):
    # Fleet-wide saturation or a watchdog network fault: never reboot the whole fleet.
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    store, api = MemoryStore(), AsyncMock()
    reports = await _run_until_reboot_due(_hung_factory([".1", ".2", ".3", ".4"]), store, api)
    api.call.assert_not_awaited()
    assert all(not r["restart"] for report in reports for r in report.values())
    assert all(wd.SlotState.load(store.data[s]).restarted_at is None for s in range(1, 5))


@pytest.mark.anyio
async def test_at_most_one_reboot_per_cycle(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    store, api = MemoryStore(), AsyncMock()
    reports = await _run_until_reboot_due(_hung_factory([".3", ".4"]), store, api)
    assert all(sum(r["restart"] for r in report.values()) <= 1 for report in reports)
    # The deferred worker is rebooted on the following cycle, not dropped.
    rebooted = [c.args[2]["instance_ids"] for c in api.call.await_args_list]
    assert rebooted == [["inst-3"], ["inst-4"]]


@pytest.mark.anyio
async def test_health_view_does_not_block_routing_on_a_slow_store(monkeypatch):
    monkeypatch.setattr(wd, "ROUTER_REFRESH_TIMEOUT_SECONDS", 0.05)
    store = MemoryStore({4: wd.SlotState(healthy=False).dump()})
    now = [0.0]
    view = wd.HealthView(store, range(1, 5), clock=lambda: now[0])
    assert await view.usable() == [1, 2, 3]

    async def stalled(slot):
        await asyncio.sleep(10)

    store.get = stalled
    now[0] += wd.ROUTER_CACHE_SECONDS
    assert await asyncio.wait_for(view.usable(), 1) == [1, 2, 3]


def _pool(mod, engines, usable):
    pool = object.__new__(mod._ModalEnginePool)
    quarantined = wd.SlotState(healthy=False).dump()
    store = MemoryStore({s: quarantined for s in range(1, len(engines) + 1) if s not in usable})
    pool._reserved_health = wd.HealthView(store, range(1, len(engines) + 1))
    return pool


def _engine(register):
    instance = MagicMock()
    instance.register.remote.aio = register
    return MagicMock(return_value=instance)


@pytest.fixture
def mod(monkeypatch):
    import flash.serving.app.modal_app as modal_app

    return modal_app


@pytest.mark.anyio
async def test_register_promotes_when_one_worker_hangs(monkeypatch, mod):
    monkeypatch.setattr(reserved, "REGISTER_TIMEOUT_SECONDS", 0.05)

    async def hang(*args):
        await asyncio.sleep(10)

    ok = AsyncMock(return_value={"ok": True})
    engines = (_engine(ok), _engine(ok), _engine(ok), _engine(hang))
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    pool = _pool(mod, engines, [1, 2, 3, 4])
    monkeypatch.setattr(pool, "_record_payload", lambda record: {"model_id": "m"})
    await pool.register(reserved.MODEL, MagicMock(deployment_generation="g"))
    assert ok.await_count == 3


@pytest.mark.anyio
async def test_register_skips_quarantined_and_fails_only_if_all_fail(monkeypatch, mod):
    bad = AsyncMock(side_effect=RuntimeError("load failed"))
    skipped = AsyncMock()
    engines = (_engine(bad), _engine(bad), _engine(bad), _engine(skipped))
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    pool = _pool(mod, engines, [1, 2, 3])
    monkeypatch.setattr(pool, "_record_payload", lambda record: {"model_id": "m"})
    with pytest.raises(RuntimeError, match="load failed"):
        await pool.register(reserved.MODEL, MagicMock(deployment_generation="g"))
    skipped.assert_not_awaited()


@pytest.mark.anyio
async def test_routing_skips_quarantined_worker(monkeypatch, mod):
    engines = tuple(MagicMock() for _ in range(4))
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    pool = _pool(mod, engines, [1, 2, 3])
    picks = [await pool._engine_for(reserved.MODEL) for _ in range(9)]
    assert engines[3].call_count == 0
    assert all(e.call_count == 3 for e in engines[:3])
    assert len(picks) == 9


@pytest.mark.anyio
async def test_health_view_reports_fleet_down_only_when_every_worker_is_quarantined():
    quarantined = wd.SlotState(healthy=False).dump()
    store = MemoryStore(dict.fromkeys((1, 2, 3), quarantined))
    now = [0.0]
    view = wd.HealthView(store, range(1, 5), clock=lambda: now[0])
    assert not await view.fleet_down()
    store.data[4] = quarantined
    now[0] += wd.ROUTER_CACHE_SECONDS
    assert await view.fleet_down()
    assert await view.usable() == [1, 2, 3, 4]  # slot routing alone still fails open
    store.data[2] = wd.SlotState().dump()  # one worker passes a probe again
    now[0] += wd.ROUTER_CACHE_SECONDS
    assert not await view.fleet_down()


@pytest.mark.anyio
async def test_dead_reserved_fleet_routes_9b_base_to_modal_fallback(monkeypatch, mod):
    # 2026-11-19: the H100 reservation expired and all four workers vanished; failing open
    # into them queued every request until Modal refused inputs, so no Base call or adapter
    # deploy could succeed. With every slot quarantined the router must use the GPU fallback.
    engines = tuple(MagicMock() for _ in range(4))
    fallback_instance = MagicMock()
    fallback = MagicMock(return_value=fallback_instance)
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    monkeypatch.setattr(mod, "RESERVED_FALLBACK", fallback)
    pool = _pool(mod, engines, [])
    assert await pool._engine_for(reserved.MODEL) is fallback_instance
    assert all(e.call_count == 0 for e in engines)


@pytest.mark.anyio
async def test_dead_reserved_fleet_registers_adapters_on_modal_fallback(monkeypatch, mod):
    dead = AsyncMock(side_effect=RuntimeError("pending input queue limit"))
    engines = tuple(_engine(dead) for _ in range(4))
    fallback_register = AsyncMock(return_value={"ok": True})
    fallback = _engine(fallback_register)
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    monkeypatch.setattr(mod, "RESERVED_FALLBACK", fallback)
    pool = _pool(mod, engines, [])
    monkeypatch.setattr(pool, "_record_payload", lambda record: {"model_id": "m"})
    await pool.register(reserved.MODEL, MagicMock(deployment_generation="g"))
    fallback_register.assert_awaited_once_with({"model_id": "m"}, "g")
    dead.assert_not_awaited()


@pytest.mark.anyio
async def test_healthy_reserved_fleet_never_uses_modal_fallback(monkeypatch, mod):
    ok = AsyncMock(return_value={"ok": True})
    engines = tuple(_engine(ok) for _ in range(4))
    fallback = MagicMock()
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    monkeypatch.setattr(mod, "RESERVED_FALLBACK", fallback)
    pool = _pool(mod, engines, [1, 2, 3, 4])
    monkeypatch.setattr(pool, "_record_payload", lambda record: {"model_id": "m"})
    await pool.register(reserved.MODEL, MagicMock(deployment_generation="g"))
    await pool._engine_for(reserved.MODEL)
    assert ok.await_count == 4
    fallback.assert_not_called()


@pytest.mark.anyio
async def test_unregister_evicts_from_live_fallback_even_when_fleet_is_down(monkeypatch, mod):
    import flash.serving.app.provider_selection as selection

    evicted = []

    async def fake_unregister_active(engines, *args):
        evicted.append((engines, args))

    monkeypatch.setattr(selection, "unregister_active", fake_unregister_active)
    dead = AsyncMock(side_effect=RuntimeError("unreachable"))
    engines = tuple(_engine(dead) for _ in range(4))
    fallback = MagicMock()
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    monkeypatch.setattr(mod, "RESERVED_FALLBACK", fallback)
    pool = _pool(mod, engines, [])
    await pool.unregister(reserved.MODEL, "org", "flash-x/step-1", "g")
    assert evicted == [([fallback.return_value], ("org", "flash-x/step-1", "g"))]
