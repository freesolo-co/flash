"""Liveness, quarantine, recovery and routing for the reserved H100 workers.

A worker can answer ``/health`` while its admitted generations never finish (observed
2026-11-14), so health also requires a one-token base-model generation of the worker's own
slot model. Workers admit that canary through a dedicated lane (``/health`` reports
``canary_lane``), so a worker saturated by long, healthy generations still answers it and a
canary timeout means the engine is hung. On a worker without the lane the canary queues
behind generations, so a timeout while ``/health`` is fine only means *busy* until no canary
has completed for ``BUSY_HUNG_AFTER_SECONDS``.

The watchdog quarantines a worker after consecutive failed probes; the router stops routing
and registering to quarantined workers. A worker that stays unhealthy is rebooted through
the Lambda API (same instance, IP, certificate and token; Docker restarts the worker), then
readmitted after it passes a probe again. Routing picks among the healthy slots of the
requested model and fails open within that model's slots if all are quarantined.

Adapter affinity: each checkpoint has a stable set of AFFINITY_SLOTS home slots (rendezvous
hashing of the checkpoint key against the slot numbers). Its registration and requests go only
to the healthy home slots, so each engine keeps a small set of adapters loaded instead of
swapping every deployed adapter through its 16 GPU/CPU LoRA slots. If no home slot is healthy,
the next healthy slots in the same hash order serve it (they load the adapter on first use).
A quarantined slot only moves its own checkpoints. Base-model requests have no key and spread
round-robin over every healthy slot.

The watchdog also publishes the configured slots (the workers secret); the router never
routes to a declared slot outside them.
"""

import asyncio
import hashlib
import itertools
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass

HEALTH_DICT_NAME = "flash-reserved-h100-health-20261127"
WATCHDOG_PERIOD_SECONDS = 120
HEALTH_TIMEOUT_SECONDS = 15
CANARY_TIMEOUT_SECONDS = 90
BUSY_HUNG_AFTER_SECONDS = 20 * 60
CONFIGURED_KEY = "configured_slots"  # sorted slots the workers secret configures
QUARANTINE_AFTER_FAILURES = 2
RESTART_AFTER_UNHEALTHY_SECONDS = 10 * 60
# Booting the VM and reloading the model takes several minutes.
RESTART_COOLDOWN_SECONDS = 30 * 60
ROUTER_CACHE_SECONDS = 15
ROUTER_REFRESH_TIMEOUT_SECONDS = 1.0
# Home slots per checkpoint. 0 disables affinity (every request round-robins over all slots).
AFFINITY_SLOTS = int(os.environ.get("FLASH_RESERVED_AFFINITY_SLOTS", "3"))
# Checkpoints whose rotation position the router remembers (least recently used dropped first).
ROTATION_KEYS = 4096


def affinity_key(record):
    """A checkpoint adapter's affinity key; None for base-model serving (no adapter)."""
    if record is None or getattr(record, "serve_base_model", False):
        return None
    org_id, model_id = getattr(record, "org_id", None), getattr(record, "model_id", None)
    return f"{org_id}/{model_id}" if org_id and model_id else None


def affinity_order(key, slots):
    """``slots`` in rendezvous-hash order for ``key``: stable per key, and removing one slot only
    moves the keys that ranked it first."""
    return sorted(
        slots,
        key=lambda slot: hashlib.sha256(f"{key}\0{slot}".encode()).digest(),
        reverse=True,
    )


def home_slots(key, slots, usable, count=None):
    """The healthy home slots for ``key``: the first ``count`` of its hash order that are usable,
    else the next usable slots in that order. Never empty while ``usable`` is not."""
    count = AFFINITY_SLOTS if count is None else count
    order = affinity_order(key, slots)
    usable = set(usable)
    home = [slot for slot in order[:count] if slot in usable]
    return home or [slot for slot in order if slot in usable][:count]


@dataclass
class SlotState:
    healthy: bool = True
    failures: int = 0
    unhealthy_since: float | None = None
    restarted_at: float | None = None
    last_error: str = ""
    checked_at: float | None = None
    busy_since: float | None = None  # first consecutive busy probe

    @classmethod
    def load(cls, raw):
        if not isinstance(raw, dict):
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def dump(self):
        return asdict(self)


def advance(state, ok, now, error="", *, busy=False):
    """Return (new_state, restart) after one probe. Pure: no I/O."""
    state = SlotState(**state.dump())
    state.checked_at = now
    if ok:
        return SlotState(restarted_at=state.restarted_at, checked_at=now), False
    if busy:
        if state.busy_since is None:
            state.busy_since = now
        if now - state.busy_since < BUSY_HUNG_AFTER_SECONDS:
            state.last_error = f"busy: {error}"[:200]
            return state, False
        error = f"no canary completed for {now - state.busy_since:.0f}s: {error}"
    else:
        state.busy_since = None
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


def routable_slots(states, slots, configured=None):
    """Slots the router may use among ``slots``; never empty (fail open).

    ``configured`` is the watchdog's published slot list. Without it (before the first
    watchdog cycle), only already probed slots count, unless none has been probed.
    """
    slots = list(slots)
    if isinstance(configured, list) and configured:
        slots = [s for s in slots if s in configured] or slots
    else:
        slots = [s for s in slots if states.get(s) is not None] or slots
    usable = [s for s in slots if SlotState.load(states.get(s)).healthy]
    return usable or slots


def canary_request(model):
    """One-token base-model generation; plain dicts keep the watchdog image schema-free."""
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
        # Admitted through the worker's canary lane, not the request queue.
        "canary": True,
    }


OK, BUSY, FAILED = "ok", "busy", "failed"


async def probe_status(transport, model):
    """(OK | BUSY | FAILED, error), talking to the worker directly, bypassing its proxy."""
    try:
        async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
            health = await transport.call("/health")
        if health.get("ok") is not True or health.get("engine_dead"):
            return FAILED, "health not ok"
        if health.get("base_model") != model:
            return FAILED, f"serves {health.get('base_model')!r}, slot serves {model}"
    except TimeoutError:
        return FAILED, "health timeout"
    except Exception as exc:  # any failure counts against liveness
        return FAILED, f"{type(exc).__name__}: {exc}"[:200]
    try:
        async with asyncio.timeout(CANARY_TIMEOUT_SECONDS):
            await transport.call("/generate", canary_request(model))
        return OK, ""
    except TimeoutError:
        return (FAILED if health.get("canary_lane") is True else BUSY), "canary timeout"
    except Exception as exc:
        return FAILED, f"{type(exc).__name__}: {exc}"[:200]


async def run_once(workers, store, lambda_api, model_for, *, transport_factory, clock=time.time):
    """Probe every worker with its slot's model, persist states, reboot persistent failures."""

    async def check(worker):
        try:
            transport = transport_factory(worker["ip"], worker["tls"])
        except Exception as exc:  # a malformed entry must not abort the whole cycle
            return worker, (FAILED, f"transport: {type(exc).__name__}"[:200])
        try:
            return worker, await probe_status(transport, model_for(worker["slot"]))
        finally:
            await transport.aclose()

    results = await asyncio.gather(*(check(w) for w in workers))
    # Reboot only a worker whose failure is its own: another worker passed this cycle, and
    # at most one reboot per cycle, so a shared cause never reboots them all.
    any_passed = any(status == OK for _, (status, _) in results)
    rebooted = False
    report = {}
    for worker, (status, error) in results:
        slot = worker["slot"]
        previous = SlotState.load(await store.get(slot))
        state, restart = advance(previous, status == OK, clock(), error, busy=status == BUSY)
        if restart and (rebooted or (not any_passed and len(results) > 1)):
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
        print(f"reserved H100 slot {slot}: {status} {report[slot]}", flush=True)
    # Published after the probes, so a newly configured slot is routed only once probed.
    await store.put(CONFIGURED_KEY, sorted(w["slot"] for w in workers))
    return report


async def each_slot(engines, model, slots, label, call):
    """Run ``call(engine)`` on each slot with a bound; succeed if any slot succeeds.

    Workers load adapters on first use from the forwarded record, so a worker that missed a
    registration still serves the adapter after it recovers.
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
        import os

        from .api import LambdaAPI
        from .transport import WorkerTransport

        workers = reserved.parse_workers(os.environ[reserved.WORKERS_ENV])
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            return await run_once(
                workers,
                ModalDictStore(),
                api,
                reserved.model_for,
                transport_factory=WorkerTransport,
            )
        finally:
            await api.aclose()

    # Modal resolves scheduled functions by their module-level identity.
    module = sys.modules[module_name]
    reserved_h100_watchdog.__module__ = module.__name__
    reserved_h100_watchdog.__qualname__ = "reserved_h100_watchdog"
    module.reserved_h100_watchdog = reserved_h100_watchdog
    function = app.function(
        image=image,
        # The workers secret holds LAMBDA_API_KEY too; it only reboots existing instances.
        secrets=[modal.Secret.from_name(reserved.SECRET_NAME)],
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


def configured_slots() -> list[int]:
    """Slots the watchdog last saw in the workers secret; every declared slot if unknown."""
    import modal

    from . import reserved

    declared = list(reserved.SLOT_MODELS)
    try:
        configured = modal.Dict.from_name(HEALTH_DICT_NAME).get(CONFIGURED_KEY)
    except Exception:  # a diagnostic must still warm what it can
        return declared
    if isinstance(configured, list) and configured and set(configured) <= set(declared):
        return sorted(configured)
    return declared


def warm_names(model, environment, engine_name):
    """Engine classes ``start_all`` warms for ``model``: its configured reserved slots, if any."""
    from . import reserved

    if not reserved.enabled(model, environment):
        return [engine_name]
    return [
        engine_name + f"_ReservedH100{slot}"
        for slot in configured_slots()
        if reserved.model_for(slot) == model
    ]


class HealthView:
    """Router-side cached read of slot states; never blocks routing on the store."""

    def __init__(self, store, slot_models, *, clock=time.monotonic):
        self._store = store
        self._slot_models = dict(slot_models)  # slot -> model
        self._slots = tuple(self._slot_models)
        self._clock = clock
        self._expires = 0.0
        self._states = {}
        self._configured = None
        self._cycles = {}
        self._rotation = OrderedDict()  # affinity key -> next position among its home slots
        self._loaded = False
        self._refresh = None

    @classmethod
    def for_engines(cls, engines):
        """None when this environment has no reserved workers."""
        from . import reserved

        return cls(ModalDictStore(), reserved.SLOT_MODELS) if engines else None

    def slots_for(self, model):
        return tuple(s for s in self._slots if self._slot_models[s] == model)

    async def home(self, model, record=None):
        """Slots that should hold ``record``'s checkpoint adapter; every usable slot for no
        record, a base-model record, or with affinity off."""
        usable = await self.usable(model)
        key = affinity_key(record)
        if key is None or AFFINITY_SLOTS <= 0:
            return usable
        return home_slots(key, self.slots_for(model), usable)

    async def next_slot(self, model, record=None):
        """Round-robin over the routable slots serving ``model``; for a checkpoint ``record``,
        round-robin over that checkpoint's healthy home slots only."""
        usable = set(await self.usable(model))
        key = affinity_key(record)
        if key is not None and AFFINITY_SLOTS > 0:
            home = home_slots(key, self.slots_for(model), usable)
            # Per-checkpoint position: a counter shared across checkpoints pins interleaved
            # checkpoints (e.g. three in a repeating order) to one home slot each.
            position = self._rotation.pop(key, 0)
            self._rotation[key] = position + 1
            if len(self._rotation) > ROTATION_KEYS:
                self._rotation.popitem(last=False)
            return home[position % len(home)]
        cycle = self._cycles.setdefault(model, itertools.cycle(self.slots_for(model)))
        while True:
            slot = next(cycle)
            if slot in usable:
                return slot

    async def usable(self, model):
        slots = self.slots_for(model)
        if not slots:
            raise ValueError(f"No reserved slot serves {model}")
        if self._clock() >= self._expires:
            self._expires = self._clock() + ROUTER_CACHE_SECONDS
            self._refresh = asyncio.ensure_future(self._reload())
            await asyncio.shield(self._refresh)
        elif not self._loaded and self._refresh is not None and not self._refresh.done():
            # Before the first view, concurrent callers share the in-flight read rather than
            # routing to every declared slot, some of which may not be configured.
            await asyncio.shield(self._refresh)
        if not self._loaded:
            return list(slots)
        return routable_slots(self._states, slots, self._configured)

    async def _reload(self):
        try:
            # Inference waits on this refresh, so bound it; a slow store keeps the last view.
            async with asyncio.timeout(ROUTER_REFRESH_TIMEOUT_SECONDS):
                configured, *values = await asyncio.gather(
                    self._store.get(CONFIGURED_KEY),
                    *(self._store.get(s) for s in self._slots),
                )
            self._states = dict(zip(self._slots, values, strict=True))
            self._configured = configured
            self._loaded = True
        except Exception as exc:  # keep the last known view
            print(f"reserved H100 health read failed: {exc!r}", flush=True)
