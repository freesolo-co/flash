"""Keep a fixed request reserve across the main pool, with delayed scale-down."""

from __future__ import annotations

import json
import logging
import math
import time
from urllib.request import urlopen

from flash.serving.deepinfer.contract.config import (
    MAIN_MAX_CONTAINERS,
    MAIN_MIN_CONTAINERS,
    POOL_SAMPLE_MAX_AGE,
    POOL_SAMPLE_SECONDS,
    POOL_SPARE_STREAMS,
    POOL_STREAM_CAPACITY,
    PORT,
    SCALEDOWN_SECONDS,
)

logger = logging.getLogger(__name__)


def required_containers(active: int, maximum: int = MAIN_MAX_CONTAINERS) -> int:
    """Add a replica at the reserve boundary: 36 -> 2, 136 -> 3."""
    return min(
        maximum, max(MAIN_MIN_CONTAINERS, (active + POOL_SPARE_STREAMS) // POOL_STREAM_CAPACITY + 1)
    )


def pool_load(container_ids, samples, now):
    """Return known demand and whether every ready replica has a fresh report."""
    active = 0
    complete = bool(container_ids)
    for container_id in container_ids:
        sample = samples.get(container_id)
        if not isinstance(sample, dict):
            complete = False
            continue
        timestamp, count = sample.get("at"), sample.get("active")
        if (
            not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not 0 <= now - timestamp <= POOL_SAMPLE_MAX_AGE
            or type(count) is not int
            or count < 0
        ):
            complete = False
            continue
        active += count
    return active, complete


def next_floor(current, active, complete, state, now, maximum):
    """Missing telemetry can increase capacity, but can never reduce it."""
    desired = required_containers(active, maximum)
    below_since = state.get("below_since")
    if desired > current:
        return desired, None
    if not complete or desired >= current:
        return current, None
    if (
        not isinstance(below_since, (int, float))
        or not 0 <= now - below_since <= SCALEDOWN_SECONDS * 2
    ):
        below_since = now
    if now - below_since >= SCALEDOWN_SECONDS:
        return desired, None
    return current, below_since


def reconcile(server, store, *, now=None):
    """One serialized CPU-controller tick; never change caps, buffers or routing."""
    settings = server.autoscaler.settings()
    maximum = min(settings.max_containers or 0, MAIN_MAX_CONTAINERS)
    if maximum == 0:
        return None  # Respect an operator disabling the pool.
    container_ids = {item.container_id for item in server._experimental_list_containers()}
    samples = dict(store.items())
    now = time.time() if now is None else now
    active, complete = pool_load(container_ids, samples, now)
    state = samples.get("controller", {})
    # A controller interruption or floor change breaks the continuous quiet period.
    if (
        not 0 <= now - state.get("at", 0) <= POOL_SAMPLE_MAX_AGE
        or state.get("floor") != settings.min_containers
    ):
        state = {}
    floor, below_since = next_floor(settings.min_containers, active, complete, state, now, maximum)
    if floor != settings.min_containers:
        server.autoscaler.update(min_containers=floor)
        logger.warning(
            "DeepSeek pool floor %s -> %s (active=%s, complete=%s)",
            settings.min_containers,
            floor,
            active,
            complete,
        )
    store.put("controller", {"at": now, "floor": floor, "below_since": below_since})
    # Finished containers are not demand. Remove their old reports without touching live ones.
    for key, value in samples.items():
        if (
            key != "controller"
            and key not in container_ids
            and now - value.get("at", now) > POOL_SAMPLE_MAX_AGE
        ):
            store.pop(key, None)
    return {"active": active, "complete": complete, "floor": floor}


def report_load(process, store, container_id):
    """Publish counts off the inference event loop; a failed report expires safely."""
    while process.poll() is None:
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                health = json.load(response)
            if health["status"] == "ok":
                store.put(
                    container_id, {"at": time.time(), "active": health["active_http_requests"]}
                )
        except Exception:
            logger.debug("DeepSeek load heartbeat unavailable", exc_info=True)
        time.sleep(POOL_SAMPLE_SECONDS)
