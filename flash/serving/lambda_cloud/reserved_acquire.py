"""Acquire and qualify reserved 9B Base H100s for the slots the workers secret lacks.

Scheduled on the main serving app. Every period, for each target slot that the workers secret
does not hold:

- reconcile its record with the Lambda instance named after the slot;
- qualify an active instance (identity: model, deploy revision, H100, then a one-token canary)
  and publish its secret-shaped entry under ``WORKERS_KEY`` in ``DICT_NAME``;
- launch at most one new worker per cycle, for the lowest empty slot, when Lambda reports
  single-H100 US capacity. Without capacity it simply retries next period, indefinitely.

Publishing does not route traffic. The operator copies the published entries into the workers
secret and redeploys (docs/reserved-h100-serving.md), which is what switches reserved serving
on, so 9B Base keeps its on-demand pool until every acquired worker has qualified.

Nothing here terminates an instance or touches the secret's slots. Set ``TARGET_KEY`` to ``[]``
in the Dict to stop acquiring.
"""

import asyncio
import time

from . import reserved

DICT_NAME = "flash-reserved-h100-acquire-20261127"
TARGET_KEY = "target_slots"  # absent: every declared slot; [] stops acquiring
WORKERS_KEY = "workers"  # qualified entries, same shape as the workers secret
REJECTED_KEY = "rejected_placements"
PERIOD_SECONDS = 120
CAMPAIGN = "flash-h100-reservation-20261127"
TAGS = [
    {"key": "reservation-campaign", "value": CAMPAIGN},
    {"key": "purpose", "value": "flash-9b-base-serving"},
]
# An ambiguous launch is never resubmitted before this; its instance may still appear.
AMBIGUOUS_LAUNCH_SECONDS = 30 * 60
REJECTION_COOLDOWN_SECONDS = 10 * 60
BOOT_ALERT_SECONDS = 90 * 60


def slot_key(slot):
    return f"slot-{slot}"


def instance_name(slot):
    """Never ends in ``-a<digits>``, so the training provider's sweep never matches it."""
    return f"{CAMPAIGN}-{slot}"


async def _qualify(slot, record, transport_factory, now):
    """The worker's secret entry if it passes identity and canary, else None."""
    from .reserved_watchdog import OK, probe_status

    try:
        transport = transport_factory(record["ip"], record["tls"])
        try:
            async with asyncio.timeout(20):
                health = await transport.call("/health")
            if reserved.identity_ok(health, record["revision"], reserved.MODEL):
                status, error = await probe_status(transport, reserved.MODEL)
            else:
                status, error = "failed", "identity not ready"
        finally:
            await transport.aclose()
    except Exception as exc:  # booting workers refuse connections
        status, error = "failed", type(exc).__name__
    if status == OK:
        record.update(state="serving", last_error="")
        return {
            "slot": slot,
            "model": reserved.MODEL,
            **{k: record[k] for k in ("instance_id", "ip", "revision", "tls")},
        }
    record.update(state="qualifying", last_error=error)
    if now - record["submitted_at"] > BOOT_ALERT_SECONDS:
        print(f"reserved H100 alert: slot {slot} not qualified: {error}", flush=True)
    return None


async def _launch(store, api, slot, revision, credentials, now):
    from .api import LambdaAPIError
    from .bootstrap import identity, user_data
    from .config import SSH_KEY_NAME, NoCapacityError, select_region
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
    if "h100" not in kind.lower():
        return f"{kind} is not an H100; retrying next cycle"
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
        "ssh_key_names": [SSH_KEY_NAME],
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
        for s in (list(reserved.SLOT_MODELS) if targets is None else targets)
        if s in reserved.SLOT_MODELS and s not in secret_slots
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
            record.update(instance_id=inst["id"], ip=inst.get("ip"), status=inst["status"])
            if serving.get(slot) == inst["id"]:
                record["state"] = "serving"
            elif inst["status"] == "active" and inst.get("ip"):
                qualifying.append(slot)
            else:
                record["state"] = "booting"

    # Probed concurrently so every pending worker fits one cycle.
    qualified = await asyncio.gather(
        *(_qualify(s, records[s], transport_factory, now) for s in qualifying)
    )
    # A target slot stays published only while it matches its record. Unpublishing is written
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
    workers.sort(key=lambda w: w["slot"])
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
    report["published"] = [w["slot"] for w in workers]
    return report


class ModalDictStore:
    def __init__(self, name=DICT_NAME):
        import modal

        self._dict = modal.Dict.from_name(name, create_if_missing=True)

    async def get(self, key):
        return await self._dict.get.aio(key)

    async def put(self, key, value):
        await self._dict.put.aio(key, value)

    async def pop(self, key):
        await self._dict.pop.aio(key, None)


def register_acquire(app, image, secrets, environment, module_name):
    """Schedule acquisition on the main serving app (Modal's unset environment is main).

    Registered from the environment alone, so the deploy and every remote import agree. The
    deploy workflow requires ``LAMBDA_API_KEY``; remote imports never read the value.
    """
    import os
    import sys

    import modal

    if (environment or "main") != "main":
        return None
    key = modal.Secret.from_dict({"LAMBDA_API_KEY": os.environ.get("LAMBDA_API_KEY", "")})
    workers = [modal.Secret.from_name(reserved.SECRET_NAME)] if reserved.active(environment) else []

    async def reserved_h100_acquire():
        from .api import LambdaAPI
        from .transport import WorkerTransport

        raw = os.environ.get(reserved.WORKERS_ENV)
        if not os.environ.get("LAMBDA_API_KEY"):
            raise RuntimeError("reserved H100 acquire needs LAMBDA_API_KEY in the deploy")
        api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
        try:
            report = await run_once(
                ModalDictStore(),
                api,
                {w["slot"] for w in reserved.parse_workers(raw)} if raw else set(),
                revision=os.environ["FREESOLO_DEPLOYMENT_SHA"],
                credentials=os.environ,
                transport_factory=WorkerTransport,
            )
        finally:
            await api.aclose()
        print(f"reserved H100 acquire: {report}", flush=True)
        return report

    # Modal resolves scheduled functions by their module-level identity.
    module = sys.modules[module_name]
    reserved_h100_acquire.__module__ = module.__name__
    reserved_h100_acquire.__qualname__ = "reserved_h100_acquire"
    module.reserved_h100_acquire = reserved_h100_acquire
    module.reserved_h100_acquire = app.function(
        image=image,
        secrets=[*secrets, key, *workers],
        schedule=modal.Period(seconds=PERIOD_SECONDS),
        cpu=0.5,  # openssl keygen per launch; distinct from the watchdog
        memory=512,
        max_containers=1,
        nonpreemptible=True,
        timeout=5 * 60,
    )(reserved_h100_acquire)
    return module.reserved_h100_acquire
