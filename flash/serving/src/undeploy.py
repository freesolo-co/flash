"""Durable half of an adapter undeploy: compare-and-swap every matched row to "disabled".

Split out of router.py's ``remove_adapter``. This is the phase that talks to persistence and
decides which rows converged; the caller owns routing removal, gpu eviction and the response
shape, so this function touches neither the router nor the engine pool.
"""

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, status

from flash.serving.src.persistence import PersistenceRecordError
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
