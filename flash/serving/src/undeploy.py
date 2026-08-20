"""Durable half of an adapter undeploy: compare-and-swap every matched row to "disabled".

Split out of router.py's ``remove_adapter``. This is the phase that talks to persistence and
decides which rows converged; the caller owns routing removal, gpu eviction and the response
shape, so this function touches neither the router nor the engine pool.
"""

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, status

from flash.serving.src.persistence import PersistenceRecordError
from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import _get_stored, _replace_stored_cas

# Rows that lose the compare-and-swap are re-read and retried. Bounded so a row losing every race
# reports as stuck rather than spinning inside the request.
_CAS_ATTEMPTS = 3


async def disable_matched(
    matches: list[AdapterRecord],
    *,
    get_authoritative: Callable[[str], Awaitable[AdapterRecord | None]],
) -> tuple[list[str], list[str], list[str], list[tuple[AdapterRecord, AdapterRecord | None]]]:
    """CAS every matched row to "disabled", returning what converged and what did not.

    Returns ``(disabled_aliases, disabled_revisions, stuck_ready, pending_teardown)``. The cascade
    spans multiple rows with no cross-row transaction, so a sibling can stay stuck-ready. The
    caller still tears down every converged row (even on the conflict path) rather than deferring:
    a reload only rehydrates "ready" rows, so a row left disabled-but-registered would never be
    re-enumerated by a retry and its gpu lora registration would leak permanently.
    """
    disabled_aliases: list[str] = []
    disabled_revisions: list[str] = []
    stuck_ready: list[str] = []
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None]] = []

    for candidate in matches:
        current: AdapterRecord | None = candidate
        converged = False
        for _ in range(_CAS_ATTEMPTS):
            if current.updated_at is None:
                current = await get_authoritative(candidate.adapter_id)
                if current is None or current.status != "ready":
                    converged = True
                    break
                if current.updated_at is None:
                    break

            committed = await _replace_stored_cas(
                current.model_copy(update={"status": "disabled"}),
                expected_updated_at=current.updated_at,
            )
            if committed is not None:
                current = committed
                converged = True
                break
            try:
                current = await _get_stored(candidate.adapter_id)
            except PersistenceRecordError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "adapter storage is unavailable",
                ) from exc
            if current is None or current.status != "ready":
                converged = True
                break

        if not converged:
            stuck_ready.append(candidate.adapter_id)
            continue
        pending_teardown.append((candidate, current))
        if candidate.is_alias:
            disabled_aliases.append(candidate.adapter_id)
        else:
            disabled_revisions.append(candidate.adapter_id)

    return disabled_aliases, disabled_revisions, stuck_ready, pending_teardown


def apply_teardown(
    router: AdapterRouter,
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None]],
) -> list[tuple[AdapterRecord, str | None]]:
    """Remove every durably disabled row from routing, returning the rows still to evict.

    Routing is updated synchronously; gpu eviction is left to the caller to defer until after
    either the success or the conflict response, so a scaled-to-zero engine's cold start cannot
    make undeploy callers time out.
    """
    cleanup_records: list[tuple[AdapterRecord, str | None]] = []
    for candidate, current in pending_teardown:
        if current is None:
            router.remove(candidate.adapter_id)
        else:
            router.upsert(current)
        cleanup_record = current or candidate
        cleanup_records.append((cleanup_record, cleanup_record.deployment_generation))
    return cleanup_records


async def get_authoritative(adapter_id: str) -> AdapterRecord | None:
    """Read the persisted row, mapping a storage failure to 503 rather than a 500."""
    try:
        return await _get_stored(adapter_id)
    except PersistenceRecordError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "adapter storage is unavailable",
        ) from exc


async def resolve_undeploy_target(
    router: AdapterRouter, adapter_id: str
) -> tuple[AdapterRecord | None, str, list[AdapterRecord]]:
    """Resolve what a delete targets: ``(base_model_record, run_id, matched_rows)``.

    A base-model serve is its own single target and is returned as the first element with no
    matches. Otherwise every alias and revision sharing the run id is a match, because undeploying
    a run tears down the whole cascade rather than one row.
    """
    record = router.get(adapter_id)
    if record is None:
        record = await get_authoritative(adapter_id)
    if record is not None and record.status == "ready" and record.serve_base_model:
        return record, adapter_id, []

    if record is not None and (record.is_alias or record.is_revision) and record.run_id is not None:
        run_id = record.run_id
    elif record is None and "@" not in adapter_id:
        # a bare run id with no cached row: the alias may exist only in persistence, so the match
        # scan below still gets a chance rather than 404-ing on the cache alone.
        run_id = adapter_id
    else:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")

    matches = [
        candidate
        for candidate in router.ready_records()
        if (candidate.is_alias and candidate.adapter_id == run_id)
        or (candidate.is_revision and candidate.run_id == run_id)
    ]
    if not matches:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
    return None, run_id, matches
