"""Unattended acquisition and qualification of extra reserved 9B Base H100 slots.

Slots in the workers secret (``reserved.SECRET_NAME``) are operator-managed. This loop owns
the *target* slots that the secret does not configure (default 9..12): every period it asks
Lambda for an approved single-H100 shape, launches at most one worker per cycle (cloud-init
bootstrap identical to the dynamic Lambda path), and once a launched worker proves its
model, revision and hardware identity and completes a canary generation, publishes it under
``WORKERS_KEY`` in its Modal Dict. The watchdog and the slot proxies read that list next to
the secret, so the router starts routing to the slot after the watchdog's next cycle.

When Lambda has no H100 capacity the loop simply retries next period, indefinitely, so
capacity is picked up without anyone watching. The loop never terminates or reboots an
instance (the watchdog may reboot a published worker, as for every reserved slot); shrinking
the fleet is the operator procedure in ``docs/reserved-h100-serving.md``.

Every launch is at-most-once: the slot record is durable before the launch request, an
ambiguous launch outcome is never resubmitted until no instance with the slot's name has
appeared for ``AMBIGUOUS_LAUNCH_SECONDS``, and an unrecorded instance with a slot's name
is reported instead of adopted.
"""

import asyncio
import json
import time

from . import reserved

EXPANSION_DICT_NAME = "flash-reserved-h100-expansion-20261121"
# Operator-adjustable without a deploy: the slots this loop should fill (``[]`` stops it).
TARGET_KEY = "target_slots"
# Qualified workers, same entry shape as the workers secret (slot, instance_id, ip, revision,
# tls {cert, token}).
WORKERS_KEY = "workers"
DEFAULT_TARGET_SLOTS = (9, 10, 11, 12)
PERIOD_SECONDS = 120
# Same approved shapes and naming as the operator-launched reserved slots.
ALLOWED_TYPES = ("gpu_1x_h100_pcie", "gpu_1x_h100_sxm5")
NAME_PREFIX = "flash-h100-reservation-20261112-"
TAGS = (
    {"key": "reservation-campaign", "value": "flash-h100-reservation-20261112"},
    {"key": "purpose", "value": "flash-9b-base-serving"},
)
AMBIGUOUS_LAUNCH_SECONDS = 30 * 60
# Booting, pulling the runtime image and loading the model normally takes well under this.
BOOT_ALERT_SECONDS = 90 * 60
HEALTH_TIMEOUT_SECONDS = 20


def slot_key(slot):
    return f"slot-{slot}"


def instance_name(slot):
    return f"{NAME_PREFIX}{slot}"


def target_slots(raw):
    """Validated target slots; the default when unset."""
    if raw is None:
        return list(DEFAULT_TARGET_SLOTS)
    if not isinstance(raw, list) or any(
        type(s) is not int or not 1 <= s <= reserved.SLOT_COUNT for s in raw
    ):
        raise ValueError(f"expansion target slots must be integers 1..{reserved.SLOT_COUNT}")
    return sorted(set(raw))


def choices(catalog):
    """(hourly_cents, type, region) with capacity now, cheapest first; US regions only."""
    found = []
    for kind in ALLOWED_TYPES:
        offer = catalog.get(kind) or {}
        spec = offer.get("instance_type") or {}
        if spec.get("specs", {}).get("gpus") != 1 or "H100" not in spec.get("gpu_description", ""):
            continue
        found.extend(
            (spec.get("price_cents_per_hour", 0), kind, region["name"])
            for region in offer.get("regions_with_capacity_available") or []
            if region.get("name", "").startswith("us-")
        )
    return sorted(found)


def merge_workers(secret_workers, published):
    """Secret workers plus well-formed published workers for slots the secret does not hold.

    A malformed published entry is dropped, never allowed to break the secret's workers.
    """
    taken = {w["slot"] for w in secret_workers}
    ids = {w["instance_id"] for w in secret_workers}
    extra = []
    for w in published or []:
        if not (
            isinstance(w, dict)
            and type(w.get("slot")) is int
            and 1 <= w["slot"] <= reserved.SLOT_COUNT
            and w["slot"] not in taken
            and isinstance(w.get("instance_id"), str)
            and w["instance_id"] not in ids
            and all(isinstance(w.get(k), str) for k in ("ip", "revision"))
            and isinstance(w.get("tls"), dict)
        ):
            continue
        taken.add(w["slot"])
        ids.add(w["instance_id"])
        extra.append(w)
    return reserved.parse_workers(json.dumps([*secret_workers, *extra]))


async def identity_ok(transport, revision):
    async with asyncio.timeout(HEALTH_TIMEOUT_SECONDS):
        health = await transport.call("/health")
    return (
        health.get("ok") is True
        and not health.get("engine_dead")
        and health.get("base_model") == reserved.MODEL
        and health.get("deployment_sha") == revision
        and health.get("configured_gpu") == "H100"
    )


async def run_once(
    store,
    api,
    secret_slots,
    *,
    revision,
    credentials,
    transport_factory,
    identity,
    user_data,
    firewall,
    filesystem,
    clock=time.time,
):
    """One acquisition/qualification cycle. Returns a printable report."""
    from .api import LambdaAPIError
    from .reserved_watchdog import OK, probe_status

    now = clock()
    targets = [s for s in target_slots(await store.get(TARGET_KEY)) if s not in secret_slots]
    published = list(await store.get(WORKERS_KEY) or [])
    published_slots = {w["slot"] for w in published}
    instances = await api.call("GET", "instances")
    live = {}
    for x in instances:
        if x.get("status") not in {"terminated", "terminating"}:
            live.setdefault(x.get("name"), []).append(x)
    report, launchable, changed = {}, [], False

    for slot in targets:
        name = instance_name(slot)
        record = await store.get(slot_key(slot))
        matches = live.get(name, [])
        if len(matches) > 1:
            report[slot] = "duplicate instances with this name; review"
            continue
        inst = matches[0] if matches else None
        if record is None:
            if inst is not None:
                report[slot] = f"unrecorded instance {inst.get('id')} uses this name; review"
            else:
                launchable.append(slot)
                report[slot] = "empty"
            continue
        if inst is None or (record.get("instance_id") and inst.get("id") != record["instance_id"]):
            if record.get("instance_id") is None and now - record["submitted_at"] < (
                AMBIGUOUS_LAUNCH_SECONDS
            ):
                report[slot] = "launch outcome pending"
                continue
            # Never created, or gone: forget it (and stop routing it) and acquire again.
            await store.pop(slot_key(slot))
            if slot in published_slots:
                published = [w for w in published if w["slot"] != slot]
                published_slots.discard(slot)
                changed = True
            report[slot] = "instance gone; will reacquire"
            continue
        spec = inst.get("instance_type") or {}
        record.update(
            instance_id=inst["id"],
            ip=inst.get("ip"),
            status=inst.get("status"),
            instance_type=spec.get("name"),
            region=(inst.get("region") or {}).get("name"),
            hourly_usd=spec.get("price_cents_per_hour", 0) / 100,
        )
        if slot in published_slots:
            record["state"] = "serving"
        elif inst.get("status") == "active" and inst.get("ip"):
            transport = transport_factory(inst["ip"], record["tls"])
            try:
                if await identity_ok(transport, record["revision"]):
                    status, error = await probe_status(transport, reserved.MODEL)
                else:
                    status, error = "failed", "identity not ready"
            except Exception as exc:  # booting workers refuse connections
                status, error = "failed", f"{type(exc).__name__}"[:120]
            finally:
                await transport.aclose()
            if status == OK:
                published.append(
                    {
                        "slot": slot,
                        "instance_id": inst["id"],
                        "ip": inst["ip"],
                        "revision": record["revision"],
                        "tls": record["tls"],
                    }
                )
                published_slots.add(slot)
                changed = True
                record.update(state="serving", qualified_at=now, last_error="")
            else:
                record.update(state="qualifying", last_error=error)
                if now - record["submitted_at"] > BOOT_ALERT_SECONDS:
                    print(
                        f"reserved H100 alert: expansion slot {slot} not qualified after "
                        f"{(now - record['submitted_at']) / 60:.0f} min: {error}",
                        flush=True,
                    )
        else:
            record["state"] = "booting"
        await store.put(slot_key(slot), record)
        report[slot] = record["state"]

    if changed:
        await store.put(WORKERS_KEY, published)

    # At most one launch per cycle, lowest empty slot first.
    if launchable:
        slot = launchable[0]
        catalog = await api.call("GET", "instance-types")
        available = choices(catalog)
        if not available:
            report["capacity"] = "no single-H100 capacity; retrying next cycle"
        else:
            price, kind, region = available[0]
            firewall_id = await firewall(api, "main", region)
            filesystem_id = await filesystem(api, "main", region)
            tls = await asyncio.to_thread(identity)
            bootstrap = user_data(reserved.MODEL, revision, credentials, tls)
            record = {
                "state": "submitted",
                "submitted_at": now,
                "instance_id": None,
                "instance_type": kind,
                "region": region,
                "quoted_hourly_usd": price / 100,
                "revision": revision,
                # The private key goes only into cloud-init; routing needs the pin and token.
                "tls": {"cert": tls["cert"], "token": tls["token"]},
            }
            await store.put(slot_key(slot), record)  # durable before the launch request
            try:
                result = await api.call(
                    "POST",
                    "instance-operations/launch",
                    {
                        "region_name": region,
                        "instance_type_name": kind,
                        "quantity": 1,
                        "ssh_key_names": ["freesolo-serving"],
                        "firewall_rulesets": [{"id": firewall_id}],
                        "file_system_mounts": [
                            {"file_system_id": filesystem_id, "mount_point": "/data/hosting-cache"}
                        ],
                        "name": instance_name(slot),
                        "image": {"family": "gpu-base-24-04"},
                        "user_data": bootstrap,
                        "tags": [dict(t) for t in TAGS],
                    },
                )
            except LambdaAPIError as exc:
                if 400 <= exc.status < 500 and exc.status != 408:
                    await store.pop(slot_key(slot))  # explicit rejection: nothing was created
                    report["capacity"] = f"launch rejected ({exc.code}); retrying next cycle"
                else:
                    record["last_error"] = str(exc)[:120]
                    await store.put(slot_key(slot), record)
                    report["capacity"] = "launch outcome unknown; reconciling by name"
                return report
            except Exception as exc:
                record["last_error"] = type(exc).__name__
                await store.put(slot_key(slot), record)
                report["capacity"] = "launch outcome unknown; reconciling by name"
                return report
            ids = result.get("instance_ids", [])
            if len(ids) == 1:
                record.update(instance_id=ids[0], state="booting", accepted_at=now)
                await store.put(slot_key(slot), record)
            report["launched"] = {"slot": slot, "type": kind, "region": region, "usd": price / 100}
            print(f"reserved H100 expansion launched slot {slot}: {kind} {region}", flush=True)
    return report


async def published_workers():
    """Qualified expansion workers; empty if the Dict cannot be read."""
    import modal

    try:
        d = modal.Dict.from_name(EXPANSION_DICT_NAME, create_if_missing=True)
        return list(await d.get.aio(WORKERS_KEY) or [])
    except Exception as exc:
        print(f"reserved H100 expansion read failed: {exc!r}", flush=True)
        return []


def register_expansion(app, image, secrets, module_name):
    """Schedule the expansion loop on the serving app (main only, beside the watchdog)."""
    import sys

    import modal

    from .reserved_watchdog import ModalDictStore

    async def reserved_h100_expansion():
        """Acquire, bootstrap and qualify reserved H100s for unconfigured target slots."""
        import os

        from .api import LambdaAPI
        from .bootstrap import identity, user_data
        from .resources import cache_filesystem, firewall
        from .transport import WorkerTransport

        secret_slots = {
            w["slot"] for w in reserved.parse_workers(os.environ["FLASH_RESERVED_H100_WORKERS"])
        }
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            report = await run_once(
                ModalDictStore(EXPANSION_DICT_NAME),
                api,
                secret_slots,
                revision=os.environ["FREESOLO_DEPLOYMENT_SHA"],
                credentials=os.environ,
                transport_factory=WorkerTransport,
                identity=identity,
                user_data=user_data,
                firewall=firewall,
                filesystem=cache_filesystem,
            )
        finally:
            await api.aclose()
        print(f"reserved H100 expansion: {report}", flush=True)
        return report

    module = sys.modules[module_name]
    reserved_h100_expansion.__module__ = module.__name__
    reserved_h100_expansion.__qualname__ = "reserved_h100_expansion"
    module.reserved_h100_expansion = reserved_h100_expansion
    function = app.function(
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
    module.reserved_h100_expansion = function
    return function
