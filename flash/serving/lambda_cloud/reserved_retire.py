"""Release the reserved 9B Base H100 fleet once reserved serving is switched off.

Registered on the main serving app only while ``reserved.ENABLED`` is False. That deployment
never routes to reserved workers, but requests the previous release already sent them can
still be running, and worker ``/health`` cannot show when they finish. So the first cycle of
each deployment only records when it started, and nothing is released until ``GRACE_SECONDS``
later, past the router request deadline. After that, every cycle:

1. stops expansion acquisition (``target_slots = []``) and unpublishes the expansion workers;
2. expires the reservation scheduler's hold under its own tick lock, so the scheduler's next
   tick terminates its instances (``-1`` through ``-4``) and closes the reservation;
3. terminates every other reserved worker, one id per request: the workers-secret slots, the
   expansion records and any instance with a reservation slot name. A scheduler-owned instance
   is terminated here only if it is still running ``SCHEDULER_FALLBACK_SECONDS`` after its hold
   was expired (the scheduler is not running);
4. once none of them is live, clears the slot health and expansion records, so a later
   re-enable starts clean.

Every cycle re-lists instances, so an ambiguous launch that appears late is still released.
Ephemeral ``flash-main-*`` lease-pool workers are never touched.
"""

import time

from flash.serving.app.timeouts import ROUTER_TIMEOUT_SECONDS

from . import reserved, reserved_expansion, reserved_watchdog

RETIRE_KEY = "retire"  # in the health Dict
# Requests the previous release routed to a reserved worker finish within the router deadline.
GRACE_SECONDS = ROUTER_TIMEOUT_SECONDS + 30 * 60
# The scheduler ticks every minute; past this, its instances are terminated here instead.
SCHEDULER_FALLBACK_SECONDS = 15 * 60
PERIOD_SECONDS = 5 * 60
CAMPAIGN = reserved.RESERVATION_SECRET_NAME
SCHEDULER_SLOTS = range(1, 5)
GONE = {"terminated", "terminating"}


def _campaign(instance, slots):
    tags = {t.get("key"): t.get("value") for t in instance.get("tags") or []}
    return tags.get("reservation-campaign") == CAMPAIGN and instance.get("name") in {
        reserved_expansion.instance_name(s) for s in slots
    }


def scheduler_owned(instance):
    """The reservation scheduler's own ``owned()`` test: its tick terminates these."""
    return _campaign(instance, SCHEDULER_SLOTS)


def slot_named(instance):
    return _campaign(instance, range(1, reserved.SLOT_COUNT + 1))


def release_hold(state, now):
    """Return (state, event) with the reservation's hold expiring now. Pure: no I/O."""
    if not isinstance(state, dict) or state.get("mode") in {None, "closed", "adopted"}:
        return state, None
    hold_until = state.get("hold_until")
    if isinstance(hold_until, (int, float)) and hold_until <= now:
        return state, None  # already expired: the scheduler is terminating its instances
    released = {**state, "hold_until": now, "hold_released_at": now}
    return released, f"released reserved H100 hold (was {hold_until!r}, mode={state['mode']})"


async def release_reservation(ledger, *, clock=time.time):
    """Expire the hold under the scheduler's tick lock; False while the lock is busy."""
    token = await reserved_watchdog.acquire_tick_lock(ledger, clock=clock)
    if token is None:
        return False
    try:
        state = await ledger.get("state")
        released, event = release_hold(state, clock())
        if released is not state:
            await ledger.put("state", released)
        if event:
            print(f"reserved H100 retire: {event}", flush=True)
        return True
    finally:
        await reserved_watchdog.release_tick_lock(ledger, token)


async def run_once(store, expansion, ledger, api, secret_workers, *, revision, clock=time.time):
    """One cycle; returns a printable report."""
    now = clock()
    state = await store.get(RETIRE_KEY)
    if not isinstance(state, dict) or state.get("revision") != revision:
        # Every deployment restarts the drain clock; that only ever delays a release.
        state = {"revision": revision, "started_at": now}
        await store.put(RETIRE_KEY, state)
    release_at = state["started_at"] + GRACE_SECONDS
    if now < release_at:
        return {"phase": "draining", "release_in_seconds": round(release_at - now)}

    if await expansion.get(reserved_expansion.TARGET_KEY) != []:
        await expansion.put(reserved_expansion.TARGET_KEY, [])
    published = await expansion.get(reserved_expansion.WORKERS_KEY) or []
    if published:
        await expansion.put(reserved_expansion.WORKERS_KEY, [])
    slots = range(1, reserved.SLOT_COUNT + 1)
    records = [await expansion.get(reserved_expansion.slot_key(s)) for s in slots]

    if "hold_released_at" not in state and await release_reservation(ledger, clock=clock):
        state["hold_released_at"] = now
        await store.put(RETIRE_KEY, state)
    hold_released_at = state.get("hold_released_at")
    fallback = hold_released_at is not None and now - hold_released_at >= SCHEDULER_FALLBACK_SECONDS

    # Configured workers by instance id, with the address routing used (None: not known yet).
    expected = {}
    for worker in [*secret_workers, *published, *(r for r in records if r)]:
        if worker.get("instance_id"):
            expected.setdefault(worker["instance_id"], worker.get("ip"))
    ours = [x for x in await api.call("GET", "instances") if x["id"] in expected or slot_named(x)]
    report = {"phase": "releasing", "terminated": [], "waiting_for_scheduler": [], "errors": {}}
    for x in ours:
        if x.get("status") in GONE:
            continue
        if expected.get(x["id"]) not in {None, x.get("ip")}:
            report["errors"][x["id"]] = "address differs from the configured worker; review"
        elif scheduler_owned(x) and not fallback:
            report["waiting_for_scheduler"].append(x["id"])
        else:
            try:
                await api.call("POST", "instance-operations/terminate", {"instance_ids": [x["id"]]})
                report["terminated"].append(x["id"])
            except Exception as exc:  # retried next cycle
                report["errors"][x["id"]] = type(exc).__name__
    for instance_id, error in report["errors"].items():
        print(f"reserved H100 alert: retire {instance_id}: {error}", flush=True)
    if hold_released_at is None or any(x.get("status") != "terminated" for x in ours):
        return report

    if "released_at" not in state:
        for s in slots:
            await store.pop(s)
            await expansion.pop(reserved_expansion.slot_key(s))
        await store.pop(reserved_watchdog.CONFIGURED_KEY)
        state["released_at"] = now
        await store.put(RETIRE_KEY, state)
    return {"phase": "released", "released_at": state["released_at"]}


def register_retire(app, image, secrets, module_name):
    """Schedule the release on the serving app (main only, while reserved serving is off)."""
    import sys

    import modal

    async def reserved_h100_retire():
        import os

        from .api import LambdaAPI
        from .reserved_watchdog import ModalDictStore

        try:
            workers = reserved.parse_workers(os.environ["FLASH_RESERVED_H100_WORKERS"])
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            print(f"reserved H100 alert: retire ignoring the workers secret: {exc!r}", flush=True)
            workers = []
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            report = await run_once(
                ModalDictStore(),
                ModalDictStore(reserved_expansion.EXPANSION_DICT_NAME),
                ModalDictStore(reserved.RESERVATION_SECRET_NAME),
                api,
                workers,
                revision=os.environ["FREESOLO_DEPLOYMENT_SHA"],
            )
        finally:
            await api.aclose()
        print(f"reserved H100 retire: {report}", flush=True)
        return report

    module = sys.modules[module_name]
    reserved_h100_retire.__module__ = module.__name__
    reserved_h100_retire.__qualname__ = "reserved_h100_retire"
    module.reserved_h100_retire = reserved_h100_retire
    module.reserved_h100_retire = app.function(
        image=image,
        secrets=[
            *secrets,
            modal.Secret.from_name(reserved.SECRET_NAME),
            modal.Secret.from_name(reserved.RESERVATION_SECRET_NAME),
        ],
        schedule=modal.Period(seconds=PERIOD_SECONDS),
        cpu=0.25,
        memory=512,
        max_containers=1,
        nonpreemptible=True,
        timeout=5 * 60,
    )(reserved_h100_retire)
    return module.reserved_h100_retire
