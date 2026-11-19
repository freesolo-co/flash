"""Liveness, quarantine and automatic recovery for the reserved 9B Base H100 workers.

A worker can answer ``/health`` while its admitted generations never finish (observed
2026-11-14: one worker held stuck generation ids, so every retry was refused and every
broadcast adapter registration waited for the full request timeout). Health therefore
also requires a one-token base-model generation within a short deadline.

The watchdog quarantines a worker after consecutive failed probes; the router stops
routing and registering to quarantined workers. A worker that stays unhealthy is
rebooted through the Lambda API (same instance, IP, certificate and token; Docker
restarts the worker), then readmitted after it passes a probe again. At least one
worker is always kept routable: if every worker is quarantined the router fails open.
"""

import asyncio
import itertools
import time
import uuid
from dataclasses import asdict, dataclass

HEALTH_DICT_NAME = "flash-reserved-h100-health-20261112"
WATCHDOG_PERIOD_SECONDS = 120
HEALTH_TIMEOUT_SECONDS = 15
CANARY_TIMEOUT_SECONDS = 90
QUARANTINE_AFTER_FAILURES = 2
RESTART_AFTER_UNHEALTHY_SECONDS = 10 * 60
# Booting the VM and reloading the model takes several minutes; do not reboot it again
# until a restarted worker has had time to come back.
RESTART_COOLDOWN_SECONDS = 30 * 60
ROUTER_CACHE_SECONDS = 15
# Upper bound on how long a routed request can wait for a health-state refresh.
ROUTER_REFRESH_TIMEOUT_SECONDS = 1.0
# The reservation scheduler terminates all four VMs once ``hold_until`` passes (observed
# 2026-11-19: the week-long hold lapsed and production 9B Base lost every worker). While the
# fleet is serving, the watchdog keeps the hold rolling well ahead of now.
HOLD_RENEW_BELOW_SECONDS = 7 * 86400
HOLD_EXTENSION_SECONDS = 30 * 86400
HOLD_ALERT_WITHIN_SECONDS = 86400
# The scheduler's tick runs at most 180 s and never expires its own lock. A lock older than
# this was left by a killed tick; it would block renewal (and the scheduler) forever.
TICK_LOCK_STALE_SECONDS = 15 * 60


@dataclass
class SlotState:
    healthy: bool = True
    failures: int = 0
    unhealthy_since: float | None = None
    restarted_at: float | None = None
    last_error: str = ""
    checked_at: float | None = None

    @classmethod
    def load(cls, raw):
        if not isinstance(raw, dict):
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def dump(self):
        return asdict(self)


def advance(state, ok, now, error=""):
    """Return (new_state, restart) after one probe. Pure: no I/O."""
    state = SlotState(**state.dump())
    state.checked_at = now
    if ok:
        return SlotState(restarted_at=state.restarted_at, checked_at=now), False
    state.failures += 1
    state.last_error = error[:200]
    if state.unhealthy_since is None:
        state.unhealthy_since = now
    if state.failures >= QUARANTINE_AFTER_FAILURES:
        state.healthy = False
    restart = (
        not state.healthy
        and now - state.unhealthy_since >= RESTART_AFTER_UNHEALTHY_SECONDS
        and (state.restarted_at is None or now - state.restarted_at >= RESTART_COOLDOWN_SECONDS)
    )
    if restart:
        state.restarted_at = now
    return state, restart


def routable_slots(states, slots):
    """Slots the router may use; never empty (fail open when every slot is quarantined)."""
    usable = [s for s in slots if SlotState.load(states.get(s)).healthy]
    return usable or list(slots)


def canary_request(model):
    """One-token base-model generation; the worker validates both dicts itself.

    Plain dicts keep the watchdog image free of the serving schema dependencies.
    """
    generation_id = f"reserved-watchdog-{uuid.uuid4().hex}"
    return {
        "payload": {
            "model_id": model,
            "generation_id": generation_id,
            "prompt": "1",
            "max_tokens": 1,
            "temperature": 0.0,
        },
        "record": {
            "model_id": model,
            "repo_id": model,
            "base_model": model,
            "serve_base_model": True,
            "thinking": False,
            "status": "ready",
        },
        "expected_checkpoint": None,
        "generation_id": generation_id,
        # Admitted through the worker's reserved canary lane, not the request queue, so a
        # worker busy with long healthy generations is not mistaken for a hung one.
        "canary": True,
    }


async def probe(transport, model):
    """(ok, error). Talks to the worker directly, bypassing its possibly saturated proxy."""
    try:
        async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
            health = await transport.call("/health")
        if health.get("ok") is not True or health.get("engine_dead"):
            return False, "health not ok"
        async with asyncio.timeout(CANARY_TIMEOUT_SECONDS):
            await transport.call("/generate", canary_request(model))
        return True, ""
    except TimeoutError:
        return False, "timeout"
    except Exception as exc:  # any failure counts against liveness
        return False, f"{type(exc).__name__}: {exc}"[:200]


async def run_once(workers, store, lambda_api, model, *, transport_factory, clock=time.time):
    """Probe every worker, persist slot states and reboot persistently dead workers."""

    async def check(worker):
        transport = transport_factory(worker["ip"], worker["tls"])
        try:
            return worker, await probe(transport, model)
        finally:
            await transport.aclose()

    results = await asyncio.gather(*(check(w) for w in workers))
    # Reboot only a worker whose failure is its own: another worker must have passed this
    # cycle (so the probe path itself works), and at most one reboot per cycle, so a shared
    # cause such as fleet-wide saturation or a watchdog network fault never reboots them all.
    any_passed = any(ok for _, (ok, _) in results)
    rebooted = False
    report = {}
    for worker, (ok, error) in results:
        slot = worker["slot"]
        previous = SlotState.load(await store.get(slot))
        state, restart = advance(previous, ok, clock(), error)
        if restart and (rebooted or not any_passed):
            restart = False
            state.restarted_at = previous.restarted_at  # still due next cycle
            state.last_error = f"reboot deferred: {state.last_error}"[:200]
        if restart:
            rebooted = True
            try:
                await lambda_api.call(
                    "POST",
                    "instance-operations/restart",
                    {"instance_ids": [worker["instance_id"]]},
                )
                state.last_error = f"rebooted after: {state.last_error}"[:200]
            except Exception as exc:  # retried after the cooldown
                state.last_error = f"reboot failed: {type(exc).__name__}"[:200]
        await store.put(slot, state.dump())
        report[slot] = {"healthy": state.healthy, "restart": restart, "error": state.last_error}
        print(f"reserved H100 slot {slot}: ok={ok} {report[slot]}", flush=True)
    return report


def renew_hold(state, now):
    """Return (state, event) for the reservation ledger. Pure: no I/O.

    Only a reservation in ``serving`` mode is renewed: that is the explicit hand-over of the
    four VMs to production serving. ``closed``/``adopted`` reservations never terminate
    anything, so they need neither renewal nor an alert.
    """
    if not isinstance(state, dict) or state.get("mode") in {None, "closed", "adopted"}:
        return state, None
    hold_until = state.get("hold_until")
    if not isinstance(hold_until, (int, float)):
        return state, "alert: reserved H100 reservation has no hold_until"
    remaining = hold_until - now
    if state["mode"] == "serving" and remaining < HOLD_RENEW_BELOW_SECONDS:
        state = {**state, "hold_until": now + HOLD_EXTENSION_SECONDS, "hold_renewed_at": now}
        return state, (
            f"renewed reserved H100 hold: {remaining / 3600:.1f}h left -> "
            f"{HOLD_EXTENSION_SECONDS / 86400:.0f}d"
        )
    if remaining <= HOLD_ALERT_WITHIN_SECONDS:
        return state, (
            f"alert: reserved H100 hold expires in {remaining / 3600:.1f}h "
            f"(mode={state['mode']}); the reservation will terminate all four workers"
        )
    return state, None


async def renew_reservation(ledger, *, clock=time.time):
    """Extend the reservation's ``hold_until`` under the reservation scheduler's own tick lock.

    The scheduler rewrites its whole state each tick, so the lock keeps a renewal from being
    overwritten by a concurrent tick; a busy lock just defers renewal to the next cycle.
    """
    token = uuid.uuid4().hex
    lock = {"token": token, "at": clock()}
    acquired = await ledger.put_if_absent("tick_lock", lock)
    if not acquired:
        held = await ledger.get("tick_lock")
        held_at = (held or {}).get("at")
        stale = isinstance(held_at, (int, float)) and clock() - held_at > TICK_LOCK_STALE_SECONDS
        # Modal Dicts have no compare-and-swap: drop only the stale lock we saw (re-read first),
        # then contend with put_if_absent like everyone else, so a lock taken meanwhile wins.
        if stale and ((await ledger.get("tick_lock")) or {}).get("token") == held.get("token"):
            print(f"reserved H100 alert: dropping stale reservation tick_lock {held!r}", flush=True)
            await ledger.pop("tick_lock")
        if held is None or stale:
            acquired = await ledger.put_if_absent("tick_lock", lock)
    if not acquired:
        state = await ledger.get("state")
        hold_until = (state or {}).get("hold_until")
        if (
            (state or {}).get("mode") not in {None, "closed", "adopted"}
            and isinstance(hold_until, (int, float))
            and hold_until - clock() <= HOLD_ALERT_WITHIN_SECONDS
        ):
            print(
                f"reserved H100 alert: hold expires in {(hold_until - clock()) / 3600:.1f}h "
                "and renewal is waiting on the reservation tick lock",
                flush=True,
            )
        return None
    try:
        state = await ledger.get("state")
        renewed, event = renew_hold(state, clock())
        if renewed is not state:
            await ledger.put("state", renewed)
        if event:
            print(f"reserved H100 {event}", flush=True)
        return event
    finally:
        if ((await ledger.get("tick_lock")) or {}).get("token") == token:
            await ledger.pop("tick_lock")


async def each_slot(engines, model, slots, label, call):
    """Run ``call(engine)`` on each slot with a bound; succeed if any slot succeeds.

    Workers load adapters on first use from the forwarded record, so a worker that
    missed a registration still serves the adapter after it recovers.
    """
    from .reserved import REGISTER_TIMEOUT_SECONDS

    async def bounded(slot):
        async with asyncio.timeout(REGISTER_TIMEOUT_SECONDS):
            return await call(engines[slot - 1](base_model=model))

    slots = tuple(slots)
    results = await asyncio.gather(*(bounded(s) for s in slots), return_exceptions=True)
    failures = [(s, r) for s, r in zip(slots, results, strict=True) if isinstance(r, BaseException)]
    for slot, exc in failures:
        print(f"reserved H100 slot {slot} {label} failed: {exc!r}", flush=True)
    if len(failures) == len(slots):
        raise failures[0][1]


def register_watchdog(app, image, module_name):
    """Schedule the watchdog on the serving app (main only, alongside the reserved proxies)."""
    import sys

    import modal

    from . import reserved

    async def reserved_h100_watchdog():
        """Quarantine hung reserved workers and reboot ones that stay dead."""
        import json
        import os

        from .api import LambdaAPI
        from .transport import WorkerTransport

        workers = json.loads(os.environ["FLASH_RESERVED_H100_WORKERS"])
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            report = await run_once(
                workers,
                ModalDictStore(),
                api,
                reserved.MODEL,
                transport_factory=WorkerTransport,
            )
        finally:
            await api.aclose()
        try:
            await renew_reservation(ModalDictStore(reserved.RESERVATION_SECRET_NAME))
        except Exception as exc:  # never let renewal break probing; retried next cycle
            print(f"reserved H100 alert: hold renewal failed: {exc!r}", flush=True)
        return report

    # Modal resolves scheduled functions by their module-level identity.
    module = sys.modules[module_name]
    reserved_h100_watchdog.__module__ = module.__name__
    reserved_h100_watchdog.__qualname__ = "reserved_h100_watchdog"
    module.reserved_h100_watchdog = reserved_h100_watchdog
    function = app.function(
        image=image,
        secrets=[
            modal.Secret.from_name(reserved.SECRET_NAME),
            modal.Secret.from_name(reserved.RESERVATION_SECRET_NAME),
        ],
        schedule=modal.Period(seconds=WATCHDOG_PERIOD_SECONDS),
        cpu=0.25,
        memory=512,
        max_containers=1,
        nonpreemptible=True,
        timeout=5 * 60,
    )(reserved_h100_watchdog)
    module.reserved_h100_watchdog = function
    return function


class ModalDictStore:
    """Async get/put over the shared slot-state Modal Dict."""

    def __init__(self, name=HEALTH_DICT_NAME):
        import modal

        self._dict = modal.Dict.from_name(name, create_if_missing=True)

    async def get(self, slot):
        return await self._dict.get.aio(slot)

    async def put(self, slot, value):
        await self._dict.put.aio(slot, value)

    async def put_if_absent(self, key, value):
        return await self._dict.put.aio(key, value, skip_if_exists=True)

    async def pop(self, key):
        return await self._dict.pop.aio(key, None)


class HealthView:
    """Router-side cached read of slot states; never blocks routing on the store."""

    def __init__(self, store, slots, *, clock=time.monotonic):
        self._store = store
        self._slots = tuple(slots)
        self._clock = clock
        self._expires = 0.0
        self._usable = list(self._slots)
        self._cycle = itertools.cycle(self._slots)

    @classmethod
    def for_engines(cls, engines):
        """None when this environment has no reserved workers."""
        return cls(ModalDictStore(), range(1, len(engines) + 1)) if engines else None

    async def next_slot(self):
        """Round-robin over routable slots."""
        usable = set(await self.usable())
        while True:
            slot = next(self._cycle)
            if slot in usable:
                return slot

    async def usable(self):
        if self._clock() >= self._expires:
            self._expires = self._clock() + ROUTER_CACHE_SECONDS
            try:
                # Inference waits on this refresh, so bound it; a slow store keeps the last view.
                async with asyncio.timeout(ROUTER_REFRESH_TIMEOUT_SECONDS):
                    values = await asyncio.gather(*(self._store.get(s) for s in self._slots))
                self._usable = routable_slots(
                    dict(zip(self._slots, values, strict=True)), self._slots
                )
            except Exception as exc:  # keep the last known view
                print(f"reserved H100 health read failed: {exc!r}", flush=True)
        return self._usable
