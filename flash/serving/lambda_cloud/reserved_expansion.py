"""Acquire, qualify and publish reserved 9B Base H100s for slots the workers secret lacks.

Every period: launch at most one worker for the lowest empty target slot when Lambda has
single-H100 capacity (otherwise retry next period), and publish each active worker that
passes the identity check and a canary. The watchdog and slot proxies read the published
list beside the secret. Nothing here terminates an instance or touches secret slots.
"""

import asyncio
import json
import time

from . import reserved

EXPANSION_DICT_NAME = "flash-reserved-h100-expansion-20261121"
TARGET_KEY = "target_slots"  # set to [] to stop acquiring
WORKERS_KEY = "workers"  # same entry shape as the workers secret
REJECTED_KEY = "rejected_placements"
DEFAULT_TARGET_SLOTS = (9, 10, 11, 12, 13, 14, 15)
PERIOD_SECONDS = 120
TAGS = [
    {"key": "reservation-campaign", "value": "flash-h100-reservation-20261112"},
    {"key": "purpose", "value": "flash-9b-base-serving"},
]
# An ambiguous launch is never resubmitted before this; its instance may still appear.
AMBIGUOUS_LAUNCH_SECONDS = 30 * 60
REJECTION_COOLDOWN_SECONDS = 10 * 60
BOOT_ALERT_SECONDS = 90 * 60


def slot_key(slot):
    return f"slot-{slot}"


def instance_name(slot):
    return f"flash-h100-reservation-20261112-{slot}"


def merge_workers(secret_workers, published):
    """Secret workers plus published ones for other slots; secret only if any is malformed."""
    taken = {w["slot"] for w in secret_workers}
    try:
        extra = [w for w in published or [] if w["slot"] not in taken]
        if not all(w["tls"]["cert"] and w["tls"]["token"] for w in extra):
            raise ValueError("published worker without TLS pin and token")
        return reserved.parse_workers(json.dumps([*secret_workers, *extra]))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"reserved H100 alert: ignoring published expansion workers: {exc!r}", flush=True)
        return secret_workers


async def published_workers():
    import modal

    try:
        d = modal.Dict.from_name(EXPANSION_DICT_NAME, create_if_missing=True)
        return await d.get.aio(WORKERS_KEY) or []
    except Exception as exc:
        print(f"reserved H100 expansion read failed: {exc!r}", flush=True)
        return []


async def _qualify(slot, record, transport_factory, now):
    """The worker's published entry if it passes identity and canary, else None."""
    from .reserved_watchdog import OK, probe_status

    try:
        transport = transport_factory(record["ip"], record["tls"])
        try:
            async with asyncio.timeout(20):
                health = await transport.call("/health")
            if reserved.identity_ok(health, record["revision"]):
                status, error = await probe_status(transport, reserved.MODEL)
            else:
                status, error = "failed", "identity not ready"
        finally:
            await transport.aclose()
    except Exception as exc:  # booting workers refuse connections
        status, error = "failed", type(exc).__name__
    if status == OK:
        record.update(state="serving", last_error="")
        return {"slot": slot, **{k: record[k] for k in ("instance_id", "ip", "revision", "tls")}}
    record.update(state="qualifying", last_error=error)
    if now - record["submitted_at"] > BOOT_ALERT_SECONDS:
        print(f"reserved H100 alert: expansion slot {slot} not qualified: {error}", flush=True)
    return None


async def _launch(store, api, slot, revision, credentials, now):
    from .api import LambdaAPIError
    from .bootstrap import identity, user_data
    from .config import NoCapacityError, select_region
    from .resources import cache_filesystem, firewall

    rejected = {k: t for k, t in (await store.get(REJECTED_KEY) or {}).items() if t > now}
    try:
        kind, region = select_region(
            await api.call("GET", "instance-types"),
            reserved.MODEL,
            excluded={tuple(k.split("/")) for k in rejected},
        )
    except NoCapacityError:
        return "no H100 capacity; retrying next cycle"
    tls = await asyncio.to_thread(identity)
    # The private key goes only into cloud-init; routing needs the pin and token.
    record = {
        "state": "submitted",
        "submitted_at": now,
        "instance_id": None,
        "revision": revision,
        "tls": {"cert": tls["cert"], "token": tls["token"]},
    }
    body = {
        "region_name": region,
        "instance_type_name": kind,
        "ssh_key_names": ["freesolo-serving"],
        "firewall_rulesets": [{"id": await firewall(api, "main", region)}],
        "file_system_mounts": [
            {
                "file_system_id": await cache_filesystem(api, "main", region),
                "mount_point": "/data/hosting-cache",
            }
        ],
        "name": instance_name(slot),
        "image": {"family": "gpu-base-24-04"},
        "user_data": user_data(reserved.MODEL, revision, credentials, tls),
        "tags": TAGS,
    }
    await store.put(slot_key(slot), record)  # durable before the launch request
    try:
        ids = (await api.call("POST", "instance-operations/launch", body))["instance_ids"]
    except Exception as exc:
        if isinstance(exc, LambdaAPIError) and 400 <= exc.status < 500 and exc.status != 408:
            # Explicit rejection: nothing was created; try other placements first.
            await store.pop(slot_key(slot))
            rejected[f"{kind}/{region}"] = now + REJECTION_COOLDOWN_SECONDS
            await store.put(REJECTED_KEY, rejected)
            return f"{kind} {region} rejected ({exc.code})"
        record["last_error"] = type(exc).__name__
        await store.put(slot_key(slot), record)
        return "launch outcome unknown; reconciling by name"
    record.update(instance_id=ids[0], state="booting")
    await store.put(slot_key(slot), record)
    return f"launched slot {slot}: {kind} {region}"


async def run_once(
    store, api, secret_slots, *, revision, credentials, transport_factory, clock=time.time
):
    """One cycle; returns a printable report."""
    now = clock()
    targets = await store.get(TARGET_KEY)
    targets = [
        s
        for s in (DEFAULT_TARGET_SLOTS if targets is None else targets)
        if 1 <= s <= reserved.SLOT_COUNT and s not in secret_slots
    ]
    records = {s: await store.get(slot_key(s)) for s in targets}
    published = await store.get(WORKERS_KEY) or []
    serving = {w["slot"]: w["instance_id"] for w in published}
    live = {}
    for x in await api.call("GET", "instances"):
        if x["status"] not in {"terminated", "terminating"}:
            live.setdefault(x["name"], []).append(x)

    report, gone, qualifying = {}, set(), []
    for slot, record in records.items():
        matches = live.get(instance_name(slot), [])
        inst = matches[0] if len(matches) == 1 else None
        if len(matches) > 1 or (record is None and inst):
            report[slot] = "unrecorded or duplicate instance with this name; review"
        elif record is None:
            report[slot] = "empty"
        elif inst is None or record["instance_id"] not in {None, inst["id"]}:
            if record["instance_id"] is None and now - record["submitted_at"] < (
                AMBIGUOUS_LAUNCH_SECONDS
            ):
                report[slot] = "launch outcome pending"
            else:
                gone.add(slot)
                report[slot] = "instance gone; will reacquire"
        else:
            record.update(instance_id=inst["id"], ip=inst["ip"], status=inst["status"])
            if serving.get(slot) == inst["id"]:
                record["state"] = "serving"
            elif inst["status"] == "active" and inst["ip"]:
                qualifying.append(slot)
            else:
                record["state"] = "booting"

    # Probed concurrently so every pending worker fits one cycle.
    qualified = await asyncio.gather(
        *(_qualify(s, records[s], transport_factory, now) for s in qualifying)
    )
    # A target slot stays published only while it matches its record. Unrouting is written
    # before any record is forgotten.
    workers = [
        w
        for w in published
        if w["slot"] not in records
        or (
            w["slot"] not in gone
            and (records[w["slot"]] or {}).get("instance_id") == w["instance_id"]
        )
    ]
    workers += [w for w in qualified if w]
    if workers != published:
        await store.put(WORKERS_KEY, workers)
    for slot, record in records.items():
        if slot in gone:
            await store.pop(slot_key(slot))
        elif record is not None and slot not in report:
            await store.put(slot_key(slot), record)
            report[slot] = record["state"]

    empty = [s for s in targets if report.get(s) == "empty"]
    if empty:
        report["launch"] = await _launch(store, api, empty[0], revision, credentials, now)
    return report


def register_expansion(app, image, secrets, module_name):
    """Schedule the loop on the serving app (main only, beside the watchdog)."""
    import sys

    import modal

    from .reserved_watchdog import ModalDictStore

    async def reserved_h100_expansion():
        import os

        from .api import LambdaAPI
        from .transport import WorkerTransport

        raw = os.environ["FLASH_RESERVED_H100_WORKERS"]
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            report = await run_once(
                ModalDictStore(EXPANSION_DICT_NAME),
                api,
                {w["slot"] for w in reserved.parse_workers(raw)},
                revision=os.environ["FREESOLO_DEPLOYMENT_SHA"],
                credentials=os.environ,
                transport_factory=WorkerTransport,
            )
        finally:
            await api.aclose()
        print(f"reserved H100 expansion: {report}", flush=True)
        return report

    module = sys.modules[module_name]
    reserved_h100_expansion.__module__ = module.__name__
    reserved_h100_expansion.__qualname__ = "reserved_h100_expansion"
    module.reserved_h100_expansion = reserved_h100_expansion
    module.reserved_h100_expansion = app.function(
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
    )(reserved_h100_expansion)
    return module.reserved_h100_expansion
