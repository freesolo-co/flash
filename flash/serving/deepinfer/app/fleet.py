"""Serialized replica lifecycle: warm replacements, fence admissions, then retire.

Each entry reserves its capacity before provisioning. A failed control call keeps
the reservation; neither telemetry loss nor a timeout is permission to kill work.
"""

import uuid
from contextlib import contextmanager

from flash.serving.deepinfer.app.autoscaling import required_containers
from flash.serving.deepinfer.contract.config import (
    MAIN_MAX_CONTAINERS,
    POOL_SAMPLE_MAX_AGE,
    SCALEDOWN_SECONDS,
)


def publish(store, state, health):
    endpoints = [
        item["url"]
        for item in state["replicas"]
        if item["phase"] == "active"
        and item["name"] in health
        and not health[item["name"]]["draining"]
    ]
    if (
        not endpoints
        and any(item["phase"] == "legacy" for item in state["replicas"])
        and store.get("routes", None) is None
    ):
        return  # The old router continues serving throughout first-fleet warmup.
    store.put(
        "routes",
        {"endpoints": endpoints},
    )


def save(store, state):
    store.put("fleet", state)


@contextmanager
def controller_lock(store):
    # Atomic across old/new CPU controller deployments, not just their individual
    # max_containers settings. Never expire a lock while its owner may still mutate
    # Modal resources. A hard-killed owner requires verified operator recovery.
    owner = uuid.uuid4().hex
    acquired = store.put("controller_lock", owner, skip_if_exists=True)
    try:
        yield acquired
    finally:
        if acquired:
            store.pop("controller_lock")


def reconcile_fleet(store, ops, *, now):
    """One tick, called only by the single serialized fleet-controller Function."""
    desired = store.get("desired", None)
    if desired is None:
        return {"status": "unconfigured"}
    state = store.get("fleet", {"replicas": [], "quiet_since": None})
    if not 0 <= now - state.get("at", 0) <= POOL_SAMPLE_MAX_AGE:
        state["quiet_since"] = None
    entries = state["replicas"]
    health = {}
    for item in list(entries):
        if item["phase"] == "legacy":
            continue  # Never infer permission to terminate pre-fence legacy code.
        if item["phase"] in {"stopping", "legacy_stopping"}:
            ops.stop(item)  # Idempotent after a controller interruption.
            if ops.stopped(item):
                entries.remove(item)
                save(store, state)
            continue
        if not item.get("url"):
            item["url"] = ops.launch(item)
            save(store, state)
        sample = ops.health(item)
        if sample is not None:
            health[item["name"]] = sample
            if item["phase"] == "warming" and not sample["draining"]:
                item["phase"] = "active"
                save(store, state)

    # Publish a warmed replacement before fencing old versions. Every stale-route
    # request then gets an explicit not-started rejection, safe to retry elsewhere.
    publish(store, state, health)
    for replacement in entries:
        if replacement["phase"] != "active" or replacement["name"] not in health:
            continue
        for item in entries:
            if item["name"] == replacement.get("replaces") and item["phase"] == "active":
                item["phase"] = "draining"
    state["at"] = now
    save(store, state)
    publish(store, state, health)

    for item in entries:
        if item["phase"] != "draining" or not item.get("url"):
            continue
        # A synchronous gate acknowledgment covers tokenization and full stream
        # cleanup. Missing telemetry and unsupported legacy endpoints stay alive.
        drained = ops.drain(item)
        if drained is not None and drained["active_http_requests"] == 0:
            item["phase"] = "stopping"
            save(store, state)
            ops.stop(item)

    active = [item for item in entries if item["phase"] == "active"]
    complete = all(item["name"] in health for item in active)
    load = sum(
        health[item["name"]]["active_http_requests"] for item in active if item["name"] in health
    )
    wanted = required_containers(load)
    # Scale-down uses the same admission fence as a rollout. Modal never receives
    # a scale-down instruction while that replica still has accepted requests.
    transitioning = any(item["version"] != desired["version"] for item in entries)
    if complete and len(active) > wanted and not transitioning:
        if state["quiet_since"] is None:
            state["quiet_since"] = now
        if now - state["quiet_since"] >= SCALEDOWN_SECONDS:
            victim = min(active, key=lambda item: health[item["name"]]["active_http_requests"])
            victim["phase"] = "draining"
            state["quiet_since"] = None
    else:
        state["quiet_since"] = None
    save(store, state)
    publish(store, state, health)

    old = transitioning
    limit = MAIN_MAX_CONTAINERS + int(old)
    reserved = sum(item.get("slots", 1) for item in entries)
    provisioned = sum(item["phase"] in {"active", "warming"} for item in entries)
    paired = {item.get("replaces") for item in entries}
    predecessor = next(
        (
            item
            for item in entries
            if item["version"] != desired["version"]
            and item["phase"] == "active"
            and item["name"] not in paired
        ),
        None,
    )
    if reserved < limit and (provisioned < wanted or predecessor):
        entry = {
            "name": "freesolo-deepseek-replica-" + uuid.uuid4().hex[:16],
            "version": desired["version"],
            "image_id": desired["image_id"],
            "phase": "warming",
            "created": now,
            "slots": 1,
        }
        if predecessor:
            entry["replaces"] = predecessor["name"]
        entries.append(entry)
        save(store, state)  # Reserve BEFORE making an external provisioning call.
        entry["url"] = ops.launch(entry)
        save(store, state)
    return {
        "status": "transitioning" if old else "active",
        "reserved": sum(item.get("slots", 1) for item in entries),
        "ready": len(store.get("routes", {"endpoints": []})["endpoints"]),
    }
