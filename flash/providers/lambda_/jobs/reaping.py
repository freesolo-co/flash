"""Lambda instance reaping: run-scoped teardown, remaining-instance proof, and orphan sweeps.

These read ``lambda_api`` attributes at call time so ``monkeypatch.setattr(lambda_api, ...)`` in
tests still takes effect, matching the rest of the jobs package.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.poll import preload_box_reap_due
from flash.providers._lifecycle.net.destructive import DestructiveOperationOutcome
from flash.providers.core.capabilities import CleanupOutcome, CleanupResult
from flash.providers.lambda_.client import api as lambda_api
from flash.providers.lambda_.jobs.builders import label_matches_run, run_label_prefix

logger = get_logger(__name__)


def run_labeled_instance_ids(instances: list[dict], run_id: str) -> list[str]:
    """Deduplicated ids of the instances carrying ``run_id``'s label, skipping malformed rows."""
    prefix = run_label_prefix(run_id)
    ids = [
        raw_id
        for instance in instances
        if isinstance((raw_id := instance.get("id")), str)
        and raw_id.strip()
        and label_matches_run(str(instance.get("name") or ""), prefix)
    ]
    return list(dict.fromkeys(ids))


def terminate_run_instances(run_id: str) -> list[str]:
    """Terminate every instance belonging to one run. Best-effort, never raises."""
    if not run_id:
        return []
    try:
        instances = lambda_api.list_instances()
    except Exception:
        return []
    ids = run_labeled_instance_ids(instances, run_id)
    return lambda_api.terminate_instances(ids) if ids else []


def run_instances_remaining(run_id: str) -> list[str]:
    """Return exact run-labeled instances only after complete enumeration and exact lookup.

    Any malformed row, incomplete listing, or lookup failure raises so handleless recovery cannot
    mistake an unknown fleet state for confirmed cleanup.
    """
    if not run_id:
        return []
    instances = lambda_api.list_instances(strict=True)
    prefix = run_label_prefix(run_id)
    remaining: list[str] = []
    for instance in instances:
        if not label_matches_run(str(instance.get("name") or ""), prefix):
            continue
        raw_id = instance.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise lambda_api.LambdaApiError(
                "lambda instance carries the exact run label but has no usable id"
            )
        instance_id = raw_id.strip()
        if lambda_api.get_instance(instance_id, strict=True) is not None:
            remaining.append(instance_id)
    return remaining


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
    known_labels: set[str] | Callable[[], set[str]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CleanupResult:
    """Terminate flash-prefixed instances not owned by a live run.

    ``should_stop`` is checked between terminations, and is forwarded into the inventory listing so
    a shutdown during its retries is not waited out: cancelling the caller cannot interrupt this
    worker thread, so a long sweep would otherwise keep destroying past the lifespan's shutdown.
    """
    # this listing is a single unpaginated request, so a stop during its retries surfaces as a
    # raise rather than a short list: unlike vast, it cannot hand back a partial inventory. observe
    # the stop anyway so a shutdown is reported as halted instead of as an ordinary retryable
    # failure, and so a stop that lands between a complete listing and the sweep body still stops.
    listing_halted = False

    def _observe_listing_stop() -> bool:
        nonlocal listing_halted
        listing_halted = listing_halted or (should_stop is not None and should_stop())
        return listing_halted

    try:
        instances = lambda_api.list_instances(
            **({} if should_stop is None else {"should_stop": _observe_listing_stop}),
        )
    except Exception as exc:
        logger.warning("lambda orphan sweep skipped: %s", exc)
        return CleanupResult(CleanupOutcome.RETRYABLE, halted=listing_halted)
    if listing_halted:
        logger.info("lambda orphan sweep: stop requested during inventory; listing is incomplete")
        return CleanupResult(CleanupOutcome.RETRYABLE, halted=True)
    try:
        labels = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        # never fall through to an empty set because that would reap every live run's instance.
        logger.warning("lambda orphan sweep skipped; could not resolve run sets: %s", exc)
        return CleanupResult(CleanupOutcome.RETRYABLE)
    active = {run_label_prefix(a) for a in (labels or set())}
    known_prefixes = (
        None if known_labels is None else {run_label_prefix(a) for a in (known or set())}
    )

    def _matches(prefixes: set[str]) -> bool:
        return any(label_matches_run(name, p) for p in prefixes)

    now = time.time()
    orphans: list[str] = []
    unresolved: list[str] = []
    for inst in instances:
        name = str(inst.get("name") or "")
        if not name.startswith("flash-"):
            continue
        if name.startswith("flash-preload-"):
            if not preload_box_reap_due(name, now):
                continue
            logger.warning(
                "reaping orphaned lambda preload box %s (outlived its wall deadline + grace; "
                "driver lost)",
                name,
            )
        elif _matches(active) or (known_prefixes is not None and not _matches(known_prefixes)):
            continue
        raw_id = inst.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            orphans.append(raw_id)
        else:
            unresolved.append(name)
    orphans = list(dict.fromkeys(orphans))
    if not orphans:
        outcome = CleanupOutcome.UNCONFIRMED if unresolved else CleanupOutcome.ABSENT
        return CleanupResult(outcome, unresolved_ids=tuple(unresolved) or None)
    deleted: list[str] = []
    halted = False
    if should_stop is None:
        deleted.extend(lambda_api.terminate_instances(orphans))
        unresolved.extend(iid for iid in orphans if iid not in set(deleted))
    else:
        for position, iid in enumerate(orphans):
            if should_stop():
                logger.info(
                    "lambda orphan sweep: stop requested; halting after %d termination(s)", position
                )
                halted = True
                break
            delete_outcome = lambda_api._terminate_instance_outcome(iid, should_stop=should_stop)
            if delete_outcome is DestructiveOperationOutcome.DELETED:
                deleted.append(iid)
            elif delete_outcome is DestructiveOperationOutcome.HALTED:
                halted = True
                break
            else:
                unresolved.append(iid)
    for iid in deleted:
        logger.warning("terminated orphaned lambda instance %s", iid)
    if halted:
        outcome = CleanupOutcome.UNCONFIRMED if deleted else CleanupOutcome.RETRYABLE
    elif not unresolved:
        outcome = CleanupOutcome.DELETED
    elif deleted:
        outcome = CleanupOutcome.UNCONFIRMED
    else:
        outcome = CleanupOutcome.RETRYABLE
    return CleanupResult(
        outcome,
        confirmed_deleted_ids=tuple(deleted),
        unresolved_ids=tuple(unresolved) or None,
        halted=halted,
    )
