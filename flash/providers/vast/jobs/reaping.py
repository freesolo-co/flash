"""Destroy Vast rentals: per-run teardown, absence confirmation, and the orphan sweep.

Split from the run lifecycle in ``__init__`` because reaping answers a different question: not
"is this run progressing" but "is anything still billing that no run owns". It reads only the
instance listing and the label helpers, so it stays a leaf -- the lifecycle imports it, never
the reverse.
"""

from __future__ import annotations

from collections.abc import Callable

from flash._internal.logging import get_logger
from flash.providers.core.capabilities import CleanupOutcome, CleanupResult
from flash.providers.vast.client import api as vast_api
from flash.providers.vast.jobs.builders import label_matches_run, run_label_prefix

logger = get_logger(__name__)


def _best_effort_destroy(instance_id, *, context: str) -> bool:
    """Best-effort destroy for non-raising teardown paths.

    Warn and return False when billing may continue; ``VastProvider.destroy`` escalates separately.
    Pass ``instance_id`` through because conversion here could raise inside ``finally`` cleanup.
    """
    ok = vast_api.destroy_instance(instance_id)
    if not ok:
        logger.warning(
            "vast teardown unconfirmed for instance %s (%s): success:false / breakdown — instance may "
            "still be billing; sweep_orphans is the backstop",
            instance_id,
            context,
        )
    return ok


def _coerce_instance_id(raw) -> int | None:
    """Return one strict positive Vast instance identity, else ``None`` for cleanup skipping."""
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


def cancel(remote: dict) -> None:
    """Cross-process cancel: destroy the persisted instance (stops billing)."""
    instance_id = remote.get("instance_id")
    if instance_id:
        _best_effort_destroy(instance_id, context="cancel")


def destroy_run_instances(run_id: str) -> list[int]:
    """Destroy every instance belonging to ONE run (labels start with its run prefix).

    Cancel/GC path: unlike ``sweep_orphans`` this never looks at other runs, so it is safe to call
    while they are in flight. Best-effort: never raises.
    """
    destroyed: list[int] = []
    if not run_id:
        return destroyed
    try:
        instances = vast_api.list_instances()
    except Exception:
        return destroyed
    prefix = run_label_prefix(run_id)
    selected: list[int] = []
    for inst in instances:
        iid = _coerce_instance_id(inst.get("id"))  # skip a non-intable id, don't abort the loop
        label = str(inst.get("label") or "")
        # Match on the label boundary (not a raw prefix) so ``flash-100`` can't also destroy ``flash-1000``.
        if iid and label_matches_run(label, prefix):
            selected.append(iid)
    destroyed.extend(iid for iid in dict.fromkeys(selected) if vast_api.destroy_instance(iid))
    return destroyed


def run_instances_remaining(run_id: str) -> list[int]:
    """Return instance ids still carrying the run label.

    Empty means confirmed clear. Strict listing raises on incomplete enumeration so handle-less
    recovery never launches over a possibly live worker still writing HF artifacts.
    """
    if not run_id:
        return []
    # strict: any incomplete enumeration raises -> caller treats as "could not confirm clear" (defers).
    instances = vast_api.list_instances(strict=True)
    prefix = run_label_prefix(run_id)
    remaining: list[int] = []
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label_matches_run(label, prefix):
            continue
        iid = _coerce_instance_id(inst.get("id"))
        if iid is None:
            # A row with THIS run's label but a non-numeric id is possibly-live yet un-targetable.
            # Skipping it (as the lenient destroy_run_instances does) would report a FALSE clear and
            # resubmit over a live box -> RAISE so the caller defers.
            raise vast_api.VastApiError(
                f"vast instance for run {run_id!r} carries the run label but an unparseable id "
                f"({inst.get('id')!r}); cannot confirm the run is clear"
            )
        remaining.append(iid)
    return remaining


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
    known_labels: set[str] | Callable[[], set[str]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CleanupResult:
    """Destroy unclaimed Flash-labeled instances and preserve partial evidence.

    ``should_stop`` is checked between destroys: cancelling the caller cannot interrupt this
    worker thread, so a long sweep would otherwise keep destroying past the lifespan's shutdown.
    """
    try:
        instances = vast_api.list_instances()
    except Exception as exc:
        logger.warning("vast orphan sweep skipped: %s", exc)
        return CleanupResult(CleanupOutcome.RETRYABLE)
    try:
        labels = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        # resolving a protection set failed, so skip rather than treat live instances as orphans.
        logger.warning("vast orphan sweep skipped; could not resolve run sets: %s", exc)
        return CleanupResult(CleanupOutcome.RETRYABLE)
    active = {run_label_prefix(a) for a in (labels or set())}
    known_prefixes = (
        None if known_labels is None else {run_label_prefix(a) for a in (known or set())}
    )

    def _matches(prefixes: set[str], label: str) -> bool:
        return any(label_matches_run(label, p) for p in prefixes)

    selected: list[int] = []
    unresolved: list[str] = []
    # keep each selected id's label: it is what maps a reaped box back to its run, so the
    # billing-cleanup audit trail needs it at destroy time, after this selection loop has ended.
    labels_by_id: dict[int, str] = {}
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label.startswith("flash-"):
            continue
        if _matches(active, label):
            continue
        if known_prefixes is not None and not _matches(known_prefixes, label):
            continue
        iid = _coerce_instance_id(inst.get("id"))
        if iid is None:
            unresolved.append(label)
        else:
            selected.append(iid)
            labels_by_id.setdefault(iid, label)
    selected = list(dict.fromkeys(selected))
    if not selected:
        outcome = CleanupOutcome.UNCONFIRMED if unresolved else CleanupOutcome.ABSENT
        return CleanupResult(outcome, unresolved_ids=tuple(unresolved) or None)
    destroyed: list[str] = []
    for position, iid in enumerate(selected):
        if should_stop is not None and should_stop():
            # halting leaves the rest selected but untouched. reporting them unresolved is what
            # keeps the outcome out of DELETED, so no caller reads a halted sweep as a clean one.
            logger.info(
                "vast orphan sweep: stop requested; halting after %d destroy attempt(s)", position
            )
            unresolved.extend(str(remaining) for remaining in selected[position:])
            break
        try:
            deleted = vast_api.destroy_instance(iid)
        except Exception:
            deleted = False
        if deleted:
            destroyed.append(str(iid))
            logger.warning(
                "destroyed orphaned vast instance %s (label %s)", iid, labels_by_id.get(iid, "?")
            )
        else:
            unresolved.append(str(iid))
    if not unresolved:
        outcome = CleanupOutcome.DELETED
    elif destroyed:
        outcome = CleanupOutcome.UNCONFIRMED
    else:
        outcome = CleanupOutcome.RETRYABLE
    return CleanupResult(
        outcome,
        confirmed_deleted_ids=tuple(destroyed),
        unresolved_ids=tuple(unresolved) or None,
    )
