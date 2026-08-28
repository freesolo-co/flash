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
from collections.abc import Callable
from dataclasses import dataclass

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import (
    CREATE_ALLOWANCE_S,
    deadline_kwargs,
    remaining_seconds,
)
from flash.providers.runpod.client import api as runpod_api

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


@dataclass(frozen=True)
class IdleEndpointSweepIssue:
    """One selected endpoint whose ownership or cleanup evidence stayed unresolved."""

    owner_fingerprint: str | None
    endpoint_name: str
    observed_endpoint_id: str
    reason: str


@dataclass(frozen=True)
class IdleEndpointSweepResult:
    """Typed outcome of one idle endpoint sweep without stronger absence claims."""

    deleted_ids: tuple[str, ...] = ()
    unresolved: tuple[IdleEndpointSweepIssue, ...] = ()
    failed_owner_fingerprints: tuple[str, ...] = ()
    inventory_unavailable: bool = False
    halted: bool = False

    def __post_init__(self) -> None:
        for field, evidence in (
            ("deleted_ids", self.deleted_ids),
            ("unresolved", self.unresolved),
            ("failed_owner_fingerprints", self.failed_owner_fingerprints),
        ):
            if len(evidence) != len(set(evidence)):
                raise ValueError(f"idle endpoint sweep {field} must be unique")

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_ids)

    @property
    def unresolved_count(self) -> int:
        """Every reason this sweep is not a clean one, endpoint-level and sweep-level alike.

        ``halted`` counts because a halt leaves selected inventory unvisited: without it a caller
        reading only the counts would take an interrupted sweep for a finished one.
        """
        return (
            len(self.unresolved)
            + len(self.failed_owner_fingerprints)
            + int(self.inventory_unavailable)
            + int(self.halted)
        )


def _selected_identity(value: object) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    return value


def _observed_identity(value: object) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= 120 else f"{rendered[:117]}..."


def canonical_endpoint_name(name: str) -> str:
    """Strip the SDK's ``live-`` prefix; the SDK registers ``live-flash-...`` but we track ``flash-...``."""
    return (name or "").removeprefix("live-")


def _is_flash_endpoint(name: str) -> bool:
    """True for a flash training endpoint (in either bare or live- registered form)."""
    return canonical_endpoint_name(name).startswith("flash-")


def _selected_owner_map(
    by_fingerprint: dict,
    protected: set[str],
    known: set[str] | None,
) -> dict[str, set[str | None]]:
    owners_by_endpoint: dict[str, set[str | None]] = {}
    for fingerprint, endpoints in by_fingerprint.items():
        owner_fingerprint = _selected_identity(fingerprint)
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            endpoint_name = _selected_identity(endpoint.get("name"))
            endpoint_id = _selected_identity(endpoint.get("id"))
            if (
                endpoint_name is None
                or endpoint_id is None
                or not _is_flash_endpoint(endpoint_name)
            ):
                continue
            canonical_name = canonical_endpoint_name(endpoint_name)
            if canonical_name in protected or (known is not None and canonical_name not in known):
                continue
            owners_by_endpoint.setdefault(endpoint_id, set()).add(owner_fingerprint)
    return owners_by_endpoint


def _sweep_issue(
    owner_fingerprint: str | None,
    endpoint_name: str,
    endpoint_id: str,
    reason: str,
) -> IdleEndpointSweepIssue:
    return IdleEndpointSweepIssue(owner_fingerprint, endpoint_name, endpoint_id, reason)


def _health_counter(group: dict, field: str) -> int | None:
    value = group.get(field)
    if type(value) is not int or value < 0:
        return None
    return value


def _cleanup_health_counts(health: object) -> tuple[dict[str, int], dict[str, int]] | None:
    if not isinstance(health, dict):
        return None
    workers = health.get("workers")
    jobs = health.get("jobs")
    if not isinstance(workers, dict) or not isinstance(jobs, dict):
        return None
    worker_counts = {
        field: _health_counter(workers, field)
        for field in ("running", "initializing", "ready", "idle")
    }
    job_counts = {field: _health_counter(jobs, field) for field in ("inQueue", "inProgress")}
    if any(value is None for value in (*worker_counts.values(), *job_counts.values())):
        return None
    return (
        {field: value for field, value in worker_counts.items() if value is not None},
        {field: value for field, value in job_counts.items() if value is not None},
    )


@dataclass(frozen=True)
class _EndpointReapOutcome:
    """What one endpoint's visit established, so the caller reads names instead of tuple slots."""

    deleted: bool = False
    observed_idle: bool = False
    issue: IdleEndpointSweepIssue | None = None
    halted: bool = False


def _reap_selected_endpoint(
    endpoint_id: str,
    owner_fingerprint: str,
    endpoint_name: str,
    displayed_name: str,
    *,
    now: float,
    min_idle_s: float,
    reap_warm: bool,
    deadline_at: float | None,
    should_stop: Callable[[], bool] | None,
) -> _EndpointReapOutcome:
    observed_idle = False
    try:
        health = runpod_api.endpoint_health_for_fingerprint(
            endpoint_id,
            owner_fingerprint,
            **deadline_kwargs(runpod_api.endpoint_health_for_fingerprint, deadline_at),
        )
        counts = _cleanup_health_counts(health)
        if counts is None:
            return _EndpointReapOutcome(
                issue=_sweep_issue(
                    owner_fingerprint,
                    endpoint_name,
                    endpoint_id,
                    "health evidence unavailable",
                )
            )
        workers, jobs_info = counts
        busy_workers = workers["running"] + workers["initializing"]
        if not reap_warm:
            busy_workers += workers["ready"] + workers["idle"]
        in_flight = jobs_info["inQueue"] + jobs_info["inProgress"]
        if busy_workers != 0 or in_flight != 0:
            _idle_since.pop(endpoint_id, None)
            return _EndpointReapOutcome()
        observed_idle = True
        first_idle, _owner = _idle_since.setdefault(endpoint_id, (now, owner_fingerprint))
        if now - first_idle < min_idle_s:
            return _EndpointReapOutcome(observed_idle=True)
        if should_stop is not None and should_stop():
            # the health lookup above is a blocking round-trip, so shutdown most often lands
            # DURING it. re-check at the destructive boundary itself: the loop-head check alone
            # would let a stop signal raised mid-request still delete the endpoint after it.
            # this endpoint has no unresolved evidence of its own -- it was simply never visited
            # to completion -- so it reports the sweep-level halt rather than an endpoint issue.
            return _EndpointReapOutcome(observed_idle=True, halted=True)
        if not runpod_api.delete_endpoint_for_fingerprint(
            endpoint_id,
            owner_fingerprint,
            **({} if should_stop is None else {"should_stop": should_stop}),
        ):
            return _EndpointReapOutcome(
                observed_idle=True,
                issue=_sweep_issue(
                    owner_fingerprint,
                    endpoint_name,
                    endpoint_id,
                    "delete was not confirmed",
                ),
            )
        _idle_since.pop(endpoint_id, None)
        logger.info("idle-sweep: deleted idle endpoint %s (%s)", displayed_name, endpoint_id)
        return _EndpointReapOutcome(deleted=True, observed_idle=True)
    except Exception:
        logger.debug(
            "idle-sweep: error processing endpoint %s (%s)",
            displayed_name,
            endpoint_id,
            exc_info=True,
        )
        return _EndpointReapOutcome(
            observed_idle=observed_idle,
            issue=_sweep_issue(
                owner_fingerprint,
                endpoint_name,
                endpoint_id,
                "provider operation failed",
            ),
        )


def _sweep_idle_flash_endpoints(
    protected: set[str],
    min_idle_s: float = 0.0,
    reap_warm: bool = True,
    known: set[str] | None = None,
    deadline_at: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> IdleEndpointSweepResult:
    """Delete idle, orphaned flash training endpoints and report unresolved selections.

    The scope flags protect other planes and live runs; `min_idle_s` requires persistent idleness.
    List accounts independently so one bad key cannot block healthy cleanup.

    ``should_stop`` is checked between endpoint deletions. The sweep runs in a worker thread that
    ``task.cancel()`` cannot interrupt, so without it a large in-flight sweep keeps deleting after
    the lifespan was told to stop.
    """
    deleted: list[str] = []
    unresolved: list[IdleEndpointSweepIssue] = []
    try:
        by_fp, failed_fps = runpod_api.list_endpoints_by_key(
            **deadline_kwargs(runpod_api.list_endpoints_by_key, deadline_at)
        )
    except Exception:
        logger.warning(
            "idle-sweep: could not list any RunPod pool account; skipping sweep", exc_info=True
        )
        return IdleEndpointSweepResult(inventory_unavailable=True)
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
    owners_by_endpoint = _selected_owner_map(by_fp, protected, known)
    ambiguous_reported: set[str] = set()
    processed_endpoints: set[str] = set()
    halted = False
    with _idle_since_lock:
        for fp, endpoints in by_fp.items():
            if halted:
                break
            owner_fingerprint = _selected_identity(fp)
            for ep in endpoints:
                if should_stop is not None and should_stop():
                    logger.info(
                        "idle-sweep: stop requested; halting after %d deletion(s)", len(deleted)
                    )
                    halted = True
                    break
                if not isinstance(ep, dict):
                    continue
                ep_name = _selected_identity(ep.get("name"))
                if ep_name is None or not _is_flash_endpoint(ep_name):
                    continue
                raw_endpoint_id = ep.get("id")
                canon = canonical_endpoint_name(ep_name)
                if canon in protected:
                    continue
                if known is not None and canon not in known:
                    continue
                endpoint_id = _selected_identity(raw_endpoint_id)
                if endpoint_id is None or owner_fingerprint is None:
                    unresolved.append(
                        IdleEndpointSweepIssue(
                            owner_fingerprint=_observed_identity(fp),
                            endpoint_name=canon,
                            observed_endpoint_id=_observed_identity(raw_endpoint_id),
                            reason="invalid selected endpoint identity",
                        )
                    )
                    continue
                if len(owners_by_endpoint.get(endpoint_id, ())) != 1:
                    if endpoint_id not in ambiguous_reported:
                        unresolved.append(
                            IdleEndpointSweepIssue(
                                owner_fingerprint=owner_fingerprint,
                                endpoint_name=canon,
                                observed_endpoint_id=endpoint_id,
                                reason="endpoint identity appeared under multiple owners",
                            )
                        )
                        ambiguous_reported.add(endpoint_id)
                    continue
                if endpoint_id in processed_endpoints:
                    continue
                processed_endpoints.add(endpoint_id)
                outcome = _reap_selected_endpoint(
                    endpoint_id,
                    owner_fingerprint,
                    canon,
                    ep_name,
                    now=now,
                    min_idle_s=min_idle_s,
                    reap_warm=reap_warm,
                    deadline_at=deadline_at,
                    should_stop=should_stop,
                )
                if outcome.observed_idle:
                    still_idle.add(endpoint_id)
                if outcome.deleted:
                    deleted.append(endpoint_id)
                if outcome.issue is not None:
                    unresolved.append(outcome.issue)
                if outcome.halted:
                    # the stop landed inside this endpoint's blocking health lookup, so the rest
                    # of the selected inventory is just as unvisited as a loop-head halt leaves it.
                    halted = True
                    break
        # prune stale timers only for accounts that responded this cycle.
        # timers owned by failed accounts are kept so a flake cannot restart their grace.
        # a halted sweep prunes nothing: it never reached the rest of the inventory, so their
        # absence from `still_idle` means "not visited", not "no longer idle" -- pruning there
        # would discard accumulated grace and restart the idle window on the next boot.
        prunable = (
            set()
            if halted
            else {eid for eid, (_ts, owner_fp) in _idle_since.items() if owner_fp in responded_fps}
        )
        for stale in prunable - still_idle:
            _idle_since.pop(stale, None)
    return IdleEndpointSweepResult(
        deleted_ids=tuple(dict.fromkeys(deleted)),
        unresolved=tuple(dict.fromkeys(unresolved)),
        failed_owner_fingerprints=tuple(
            dict.fromkeys(_selected_identity(fp) or _observed_identity(fp) for fp in failed_fps)
        ),
        halted=halted,
    )
