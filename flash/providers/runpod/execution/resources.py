"""Reconcile the RunPod account's shared resources: cache volumes and idle endpoints.

Two kinds of housekeeping a deploy has to do before it can launch. The weight-cache half finds or
grows the network volume the workers read base weights from, per datacenter. The sweep half frees
endpoint quota by retiring flash endpoints idle past their grace, tracking when each was first seen
idle so a brief lull does not delete a live one.

Split out of `flash.providers.runpod.execution.jobs` to keep that module under the file-size limit.
"""

from __future__ import annotations

import threading
import time

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import (
    CREATE_ALLOWANCE_S,
    deadline_kwargs,
    remaining_seconds,
)
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.serverless.naming import run_target_of

# Growing an existing cache volume is best-effort reconciliation, so it gets a short fixed ceiling
# rather than a share of the run deadline. An unreachable account would otherwise burn retries and
# backoff against the same clock the launch needs, leaving less than the create allowance and
# failing a deploy the healthy owning account could have served. Under a deadline the effective
# budget is this OR whatever is left above the create allowance, whichever is smaller.
WEIGHT_CACHE_GROW_BUDGET_S = 20.0


def weight_cache_grow_headroom_s() -> float:
    """Reconciliation headroom a whole ``deploy_train_endpoint`` call can need, in seconds.

    Reserve one grow per account because retries on an already-reconciled account skip it. The
    deployer separately protects this headroom from creates and backoffs.
    """
    from flash.providers.runpod.client import auth as rp_keys

    return WEIGHT_CACHE_GROW_BUDGET_S * max(1, rp_keys.key_count())


logger = get_logger(__name__)

# DCs that do NOT support network volumes — creating one there 500s the whole deploy.
# SDK exposes no capability flag; stale list degrades gracefully (falls back to cold cross-region run).
_VOLUME_INCAPABLE_DATACENTERS = frozenset({"US-MO-1"})


def weight_cache_datacenters() -> list:
    """Every volume-capable RunPod DC (DataCenter.all() minus _VOLUME_INCAPABLE_DATACENTERS)."""
    from runpod_flash.core.resources.datacenter import DataCenter

    return [dc for dc in DataCenter.all() if dc.value not in _VOLUME_INCAPABLE_DATACENTERS]


def weight_cache_volume_name(base: str, dc) -> str:
    """Physical volume name for ``base`` in datacenter ``dc``.

    DC MUST be in the name: the SDK keys resource tracking on name alone (no datacenter), so same-named
    volumes across DCs collide and the 2nd deploy crashes (unimplemented undeploy).
    """
    return f"{base}-{dc.value.lower()}"


def weight_cache_volumes(spec) -> list:
    """One NetworkVolume per storage datacenter; empty when the cache is off."""
    base = getattr(spec.gpu, "network_volume", None) if spec is not None else None
    if not base:
        return []
    dcs = weight_cache_datacenters()
    if not dcs:
        return []
    from runpod_flash import NetworkVolume

    from flash.core.spec_persistence import volume_gb
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    # The shared cache is platform-managed, so its size comes from the managed constant rather than
    # whatever the spec happens to carry: a stale/round-tripped spec can still hold a pre-bump size,
    # and honoring that would create (or attach) an undersized volume. Custom volumes are the
    # caller's to size, so those keep the spec value.
    size = volume_gb(getattr(spec.gpu, "network_volume_gb", WEIGHT_CACHE_VOLUME_GB))
    if str(base) == WEIGHT_CACHE_VOLUME_NAME:
        size = max(size, WEIGHT_CACHE_VOLUME_GB)
    return [
        NetworkVolume(name=weight_cache_volume_name(str(base), dc), size=size, datacenter=dc)
        for dc in dcs
    ]


def grow_weight_cache_volumes(
    spec, key: str, deadline_at: float | None = None, wanted: dict[str, int] | None = None
) -> None:
    """Raise this account's already-provisioned shared-cache volumes to the managed size.

    Existing volumes require REST growth because RunPod sizes only on create. Scope to `key` and
    optional `wanted`; remain best-effort and yield the create allowance under a deadline.
    """
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    try:
        if wanted is None:
            base = getattr(spec.gpu, "network_volume", None) if spec is not None else None
            if str(base or "") != WEIGHT_CACHE_VOLUME_NAME:
                return  # cache off, or a custom volume the caller owns the sizing of
            wanted = {
                weight_cache_volume_name(str(base), dc): WEIGHT_CACHE_VOLUME_GB
                for dc in weight_cache_datacenters()
            }
        if not wanted:
            return
        budget = WEIGHT_CACHE_GROW_BUDGET_S
        if deadline_at is not None:
            budget = min(budget, remaining_seconds(deadline_at) - CREATE_ALLOWANCE_S)
        if budget <= 0:
            logger.warning("weight cache: no room to grow volume(s) before launch; attaching as-is")
            return
        grown = runpod_api.grow_network_volumes_for_key(
            key, wanted, deadline_at=time.time() + budget
        )
    except Exception as exc:
        logger.warning("weight cache: could not grow volume(s) (%s); attaching as-is", exc)
        return
    for name, size in sorted(grown.items()):
        logger.info("weight cache: grew %s to %d GB (was under the managed size)", name, size)


def weight_cache_endpoint_kwargs(spec) -> dict:
    """Endpoint kwargs that attach the weight-cache fleet, or ``{}`` (best-effort; cache off = no volumes)."""
    try:
        vols = weight_cache_volumes(spec)
        if not vols:
            return {}
        return {"volume": vols, "datacenter": weight_cache_datacenters()}
    except Exception as exc:
        logger.warning("weight cache disabled for this run (%s); deploying with no volume", exc)
        return {}


# {endpoint_id: (first_observed_idle_ts, owning_key_fingerprint)} — grace timer per endpoint.
# Serialized by _idle_since_lock; two threads (periodic reaper + deploy-time quota sweep) can race.
_idle_since: dict[str, tuple[float, str]] = {}
_idle_since_lock = threading.Lock()


def canonical_endpoint_name(name: str) -> str:
    """Strip the SDK's ``live-`` prefix; the SDK registers ``live-flash-...`` but we track ``flash-...``."""
    return (name or "").removeprefix("live-")


def _is_flash_endpoint(name: str) -> bool:
    """True for a flash training endpoint (in either bare or live- registered form)."""
    return canonical_endpoint_name(name).startswith("flash-")


def _sweep_idle_flash_endpoints(
    protected: set[str],
    min_idle_s: float = 0.0,
    reap_warm: bool = True,
    known: set[str] | None = None,
    deadline_at: float | None = None,
) -> int:
    """Delete idle, ORPHANED flash training endpoints and return the count deleted.

    ``protected`` and ``known`` may hold either an exact endpoint name or a run-scoped target. An
    attempt endpoint is named ``<target>-a<n>``, so a target entry covers every attempt of that run
    rather than only the one attempt whose name happened to be persisted, while an exact name still
    protects a single endpoint (the deploy-time quota sweep and the warm-preload endpoints, which
    carry no attempt at all).

    The scope flags protect other planes and live runs; `min_idle_s` requires persistent idleness.
    List accounts independently so one bad key cannot block healthy cleanup.
    """
    deleted = 0
    try:
        by_fp, failed_fps = runpod_api.list_endpoints_by_key(
            **deadline_kwargs(runpod_api.list_endpoints_by_key, deadline_at)
        )
    except Exception:
        logger.warning(
            "idle-sweep: could not list any RunPod pool account; skipping sweep", exc_info=True
        )
        return 0
    if failed_fps:
        logger.warning(
            "idle-sweep: %d of %d RunPod pool account(s) failed to list this cycle; reaping the %d "
            "that responded and retrying the rest next sweep",
            len(failed_fps),
            len(by_fp) + len(failed_fps),
            len(by_fp),
        )
    responded_fps = set(by_fp)
    now = time.time()
    still_idle: set[str] = set()
    with _idle_since_lock:
        for fp, endpoints in by_fp.items():
            for ep in endpoints:
                ep_name = ep.get("name") or ""
                eid = ep.get("id")
                if not (eid and _is_flash_endpoint(ep_name)):
                    continue
                # an endpoint answers to either identity it carries: its own name, and -- when it
                # is an attempt endpoint named ``<target>-a<n>`` -- the run target covering every
                # attempt of that run. a warm endpoint carries no attempt, so only its own name.
                identities = {canonical_endpoint_name(ep_name)}
                target = run_target_of(ep_name)
                if target is not None:
                    identities.add(target)
                if identities & protected:
                    continue
                if known is not None and not (identities & known):
                    continue
                try:
                    health = (
                        runpod_api.endpoint_health_for_fingerprint(
                            eid,
                            fp,
                            **deadline_kwargs(
                                runpod_api.endpoint_health_for_fingerprint, deadline_at
                            ),
                        )
                        or {}
                    )
                    workers = health.get("workers")
                    jobs_info = health.get("jobs")
                    if (
                        not isinstance(workers, dict)
                        or not workers
                        or not isinstance(jobs_info, dict)
                    ):
                        continue
                    busy_workers = (workers.get("running") or 0) + (
                        workers.get("initializing") or 0
                    )
                    if not reap_warm:
                        busy_workers += (workers.get("ready") or 0) + (workers.get("idle") or 0)
                    in_flight = (jobs_info.get("inQueue") or 0) + (jobs_info.get("inProgress") or 0)
                    if busy_workers != 0 or in_flight != 0:
                        _idle_since.pop(eid, None)
                        continue
                    still_idle.add(eid)
                    first_idle, _owner = _idle_since.setdefault(eid, (now, fp))
                    if now - first_idle < min_idle_s:
                        continue
                    if runpod_api.delete_endpoint_for_fingerprint(eid, fp):
                        deleted += 1
                        _idle_since.pop(eid, None)
                        logger.info("idle-sweep: deleted idle endpoint %s (%s)", ep_name, eid)
                except Exception:
                    logger.debug(
                        "idle-sweep: error processing endpoint %s (%s)", ep_name, eid, exc_info=True
                    )
                    continue
        # Prune stale timers only for accounts that responded this cycle (authoritative view).
        # Timers owned by failed accounts are kept — resetting them would restart the grace on each flake.
        prunable = {
            eid for eid, (_ts, owner_fp) in _idle_since.items() if owner_fp in responded_fps
        }
        for stale in prunable - still_idle:
            _idle_since.pop(stale, None)
    return deleted
