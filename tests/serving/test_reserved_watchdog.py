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


def workers(count=4):
    return [
        {"slot": i, "instance_id": f"inst-{i}", "ip": f"8.8.8.{i}", "revision": "a" * 40, "tls": {}}
        for i in range(1, count + 1)
    ]


LANE_HEALTH = {"ok": True, "engine_dead": False, "canary_lane": True}
LEGACY_HEALTH = {"ok": True, "engine_dead": False}


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
    configured = [1, 2, 3, 4]
    assert wd.routable_slots({4: bad}, (1, 2, 3, 4), configured) == [1, 2, 3]
    states = dict.fromkeys(range(1, 5), bad)
    assert wd.routable_slots(states, (1, 2, 3, 4), configured) == [1, 2, 3, 4]


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
            return LANE_HEALTH
        await asyncio.sleep(10)
        return None

    transport = MagicMock(call=call)
    assert await wd.probe(transport, reserved.MODEL) == (False, "canary timeout")
    assert await wd.probe_status(transport, reserved.MODEL) == (wd.FAILED, "canary timeout")


@pytest.mark.anyio
async def test_run_once_quarantines_and_reboots_only_the_hung_worker(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)

    def factory(ip, tls):
        hung = ip.endswith(".4")

        async def call(path, body=None):
            if path == "/health":
                return LANE_HEALTH
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
    store = MemoryStore({4: wd.SlotState(healthy=False).dump(), wd.CONFIGURED_KEY: [1, 2, 3, 4]})
    now = [0.0]
    view = wd.HealthView(store, range(1, 5), clock=lambda: now[0])
    assert await view.usable() == [1, 2, 3]
    store.data.clear()
    assert await view.usable() == [1, 2, 3]  # cached
    store.get = AsyncMock(side_effect=RuntimeError("dict unavailable"))
    now[0] += wd.ROUTER_CACHE_SECONDS
    assert await view.usable() == [1, 2, 3]


def _hung_factory(hung_suffixes, health=LANE_HEALTH):
    def factory(ip, tls):
        hung = ip.endswith(tuple(hung_suffixes))

        async def call(path, body=None):
            if path == "/health":
                return health
            if hung:
                await asyncio.sleep(10)
            return {"request_id": body["generation_id"]}

        return MagicMock(call=call, aclose=AsyncMock())

    return factory


async def _run_until_reboot_due(factory, store, api, *, extra_seconds=0, count=4):
    now = [0.0]
    reports = []
    seconds = wd.RESTART_AFTER_UNHEALTHY_SECONDS + extra_seconds
    for _ in range(int(seconds / wd.WATCHDOG_PERIOD_SECONDS) + 2):
        reports.append(
            await wd.run_once(
                workers(count),
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
async def test_lone_worker_that_stays_dead_is_rebooted(monkeypatch):
    # A singleton fleet has no peer to rule out a shared cause; never deferring forever.
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    store, api = MemoryStore(), AsyncMock()
    await _run_until_reboot_due(_hung_factory([".1"]), store, api, count=1)
    api.call.assert_awaited_once_with(
        "POST", "instance-operations/restart", {"instance_ids": ["inst-1"]}
    )


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
    store = MemoryStore({4: wd.SlotState(healthy=False).dump(), wd.CONFIGURED_KEY: [1, 2, 3, 4]})
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
    store.data[wd.CONFIGURED_KEY] = list(range(1, len(engines) + 1))
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


DAY = 86400


class LedgerStore(MemoryStore):
    async def put_if_absent(self, key, value):
        if key in self.data:
            return False
        self.data[key] = value
        return True

    async def pop(self, key):
        return self.data.pop(key, None)


def reservation(mode, hold_until):
    return {"campaign": "c", "mode": mode, "hold_until": hold_until, "slots": []}


def test_serving_hold_renews_before_it_can_lapse():
    state, event = wd.renew_hold(reservation("serving", 1000 + 2 * DAY), 1000)
    assert state["hold_until"] == 1000 + wd.HOLD_EXTENSION_SECONDS
    assert state["hold_renewed_at"] == 1000
    assert event.startswith("renewed")
    # Far enough ahead: untouched, no log noise.
    far = reservation("serving", 1000 + 20 * DAY)
    assert wd.renew_hold(far, 1000) == (far, None)


def test_non_serving_hold_alerts_within_a_day_and_closed_is_ignored():
    state = reservation("reserve_only", 1000 + 3600)
    renewed, event = wd.renew_hold(state, 1000)
    assert renewed is state
    assert event.startswith("alert")
    assert "1.0h" in event
    assert wd.renew_hold(reservation("closed", 0), 1000) == (reservation("closed", 0), None)
    assert wd.renew_hold(None, 1000) == (None, None)


def test_renew_reservation_takes_the_scheduler_tick_lock():
    ledger = LedgerStore({"state": reservation("serving", 1000 + DAY)})
    event = asyncio.run(wd.renew_reservation(ledger, clock=lambda: 1000))
    assert event.startswith("renewed")
    assert ledger.data["state"]["hold_until"] == 1000 + wd.HOLD_EXTENSION_SECONDS
    assert "tick_lock" not in ledger.data


def test_renew_reservation_defers_while_scheduler_ticks(capsys):
    held = {"token": "scheduler", "at": 999}
    ledger = LedgerStore({"state": reservation("serving", 1000 + 3600), "tick_lock": held})
    assert asyncio.run(wd.renew_reservation(ledger, clock=lambda: 1000)) is None
    assert ledger.data["state"]["hold_until"] == 1000 + 3600
    assert ledger.data["tick_lock"] == held
    # Inside the alert window the deferred renewal still alerts.
    assert "alert: hold expires in 1.0h" in capsys.readouterr().out
    ledger.data["state"]["hold_until"] = 1000 + 20 * DAY
    asyncio.run(wd.renew_reservation(ledger, clock=lambda: 1000))
    assert "alert" not in capsys.readouterr().out


def test_renew_reservation_replaces_a_stale_tick_lock(capsys):
    stale = {"token": "dead-tick", "at": 0}
    ledger = LedgerStore({"state": reservation("serving", 1000 + DAY), "tick_lock": stale})
    now = wd.TICK_LOCK_STALE_SECONDS + 1
    event = asyncio.run(wd.renew_reservation(ledger, clock=lambda: now))
    assert event.startswith("renewed")
    assert ledger.data["state"]["hold_until"] == now + wd.HOLD_EXTENSION_SECONDS
    # The stale lock is released too, which also unblocks the scheduler.
    assert "tick_lock" not in ledger.data
    assert "stale reservation tick_lock" in capsys.readouterr().out


class VanishingLockLedger(LedgerStore):
    """put_if_absent fails, then the first read finds no lock: the scheduler released it and
    a new tick took it again before the watchdog could act."""

    async def get(self, key):
        if key == "tick_lock" and not getattr(self, "raced", False):
            self.raced = True
            return None
        return await super().get(key)


def test_renew_reservation_retries_atomically_when_the_lock_vanishes():
    live = {"token": "scheduler", "at": 1000}
    ledger = VanishingLockLedger({"state": reservation("serving", 1000 + DAY), "tick_lock": live})
    assert asyncio.run(wd.renew_reservation(ledger, clock=lambda: 1000)) is None
    # The live scheduler lock is never overwritten, and the state is left to the scheduler.
    assert ledger.data["tick_lock"] == live
    assert ledger.data["state"]["hold_until"] == 1000 + DAY


@pytest.mark.anyio
async def test_laneless_canary_timeout_is_busy_not_dead(monkeypatch):
    # 2026-11-19: workers without the canary lane were rebooted while saturated with healthy
    # generations, killing every in-flight request. A busy worker stays routable and is not
    # rebooted within the busy allowance.
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    factory = _hung_factory([".3"], health=LEGACY_HEALTH)
    store, api = MemoryStore(), AsyncMock()
    reports = await _run_until_reboot_due(factory, store, api)
    api.call.assert_not_awaited()
    assert all(r[3]["healthy"] and not r[3]["restart"] for r in reports)
    assert reports[-1][3]["error"] == "busy: canary timeout"
    assert wd.SlotState.load(store.data[3]).failures == 0


@pytest.mark.anyio
async def test_laneless_worker_that_never_completes_a_canary_is_still_rebooted(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    factory = _hung_factory([".4"], health=LEGACY_HEALTH)
    store, api = MemoryStore(), AsyncMock()
    reports = await _run_until_reboot_due(
        factory, store, api, extra_seconds=wd.BUSY_HUNG_AFTER_SECONDS
    )
    api.call.assert_awaited_once_with(
        "POST", "instance-operations/restart", {"instance_ids": ["inst-4"]}
    )
    assert not reports[-1][4]["healthy"]
    assert all(reports[-1][s]["healthy"] for s in (1, 2, 3))


def test_busy_resets_on_success_and_hard_failure_counts_immediately():
    state, _ = wd.advance(wd.SlotState(), False, 0, "canary timeout", busy=True)
    assert state.busy_since == 0
    assert state.failures == 0
    state, _ = wd.advance(state, True, 120)
    assert state.busy_since is None
    # A busy spell restarts from the next timeout, not the first one ever seen.
    state, _ = wd.advance(state, False, 240, "canary timeout", busy=True)
    state, _ = wd.advance(state, False, 240 + wd.BUSY_HUNG_AFTER_SECONDS - 1, "x", busy=True)
    assert state.failures == 0
    state, _ = wd.advance(state, False, 240 + wd.BUSY_HUNG_AFTER_SECONDS, "x", busy=True)
    assert state.failures == 1
    assert state.last_error.startswith("no canary completed")
    # Health failures are never excused as busy.
    state, _ = wd.advance(wd.SlotState(), False, 0, "health timeout")
    assert state.failures == 1


@pytest.mark.anyio
async def test_probe_health_failure_is_failed_even_without_lane():
    async def call(path, body=None):
        raise ConnectionError("refused")

    status, error = await wd.probe_status(MagicMock(call=call), reserved.MODEL)
    assert status == wd.FAILED
    assert "refused" in error


@pytest.mark.anyio
async def test_run_once_handles_eight_workers_and_publishes_configured_slots(monkeypatch):
    monkeypatch.setattr(wd, "CANARY_TIMEOUT_SECONDS", 0.05)
    store, api = MemoryStore(), AsyncMock()
    reports = await _run_until_reboot_due(_hung_factory([".7"]), store, api, count=8)
    assert set(reports[-1]) == set(range(1, 9))
    assert store.data[wd.CONFIGURED_KEY] == list(range(1, 9))
    api.call.assert_awaited_once_with(
        "POST", "instance-operations/restart", {"instance_ids": ["inst-7"]}
    )


@pytest.mark.anyio
async def test_health_view_routes_only_configured_slots():
    view = wd.HealthView(MemoryStore({wd.CONFIGURED_KEY: [1, 2, 3, 4]}), range(1, 9))
    assert await view.usable() == [1, 2, 3, 4]
    # Every configured slot quarantined: fail open to the configured slots only.
    bad = wd.SlotState(healthy=False).dump()
    store = MemoryStore({wd.CONFIGURED_KEY: [1, 2], 1: bad, 2: bad})
    assert await wd.HealthView(store, range(1, 9)).usable() == [1, 2]


@pytest.mark.anyio
async def test_concurrent_first_requests_wait_for_the_configured_slots():
    # Requests that arrive while the first refresh is in flight must not see all 8 slots.
    class SlowStore(MemoryStore):
        async def get(self, key):
            await asyncio.sleep(0.01)
            return await super().get(key)

    view = wd.HealthView(SlowStore({wd.CONFIGURED_KEY: [1, 2, 3, 4]}), range(1, 9))
    views = await asyncio.gather(*(view.usable() for _ in range(5)))
    assert all(v == [1, 2, 3, 4] for v in views)


@pytest.mark.anyio
async def test_health_view_before_first_publication_uses_probed_slots():
    # Deployed with eight declared slots while the watchdog has only probed four.
    ok = wd.SlotState().dump()
    store = MemoryStore(dict.fromkeys(range(1, 5), ok))
    assert await wd.HealthView(store, range(1, 9)).usable() == [1, 2, 3, 4]
    # Nothing probed yet at all: every declared slot.
    assert await wd.HealthView(MemoryStore(), range(1, 9)).usable() == list(range(1, 9))


@pytest.mark.anyio
async def test_routing_rotates_over_eight_slots(monkeypatch, mod):
    engines = tuple(MagicMock() for _ in range(8))
    monkeypatch.setattr(mod, "RESERVED_ENGINES", engines)
    pool = object.__new__(mod._ModalEnginePool)
    store = MemoryStore(
        {wd.CONFIGURED_KEY: list(range(1, 9)), 6: wd.SlotState(healthy=False).dump()}
    )
    pool._reserved_health = wd.HealthView(store, range(1, 9))
    for _ in range(14):
        await pool._engine_for(reserved.MODEL)
    assert engines[5].call_count == 0
    assert all(e.call_count == 2 for i, e in enumerate(engines) if i != 5)
