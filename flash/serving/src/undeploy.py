"""Durable half of an adapter undeploy: compare-and-swap every matched row to "disabled".

Split out of router.py's ``remove_adapter``. This is the phase that talks to persistence and
decides which rows converged; the caller owns routing removal and gpu eviction, so this function
touches neither the router nor the engine pool. Failure responses live with the cascade result so
they cannot escape before its accumulated teardown is attached.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from flash.serving.src.persistence import PersistenceRecordError
from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import _get_stored, _list_run_stored, _replace_stored_cas

# Rows that lose the compare-and-swap are re-read and retried. Bounded so a row losing every race
# reports as stuck rather than spinning inside the request.
_CAS_ATTEMPTS = 3


# a named result keeps partial teardown and failure state together without widening a tuple.
@dataclass
class DisableResult:
    """Durable cascade state retained until routing and gpu teardown are scheduled."""

    disabled_aliases: list[str]
    disabled_revisions: list[str]
    stuck_ready: list[str]
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None]]
    storage_unavailable: bool = False

    def failure_response(
        self, run_id: str, background_tasks: BackgroundTasks
    ) -> JSONResponse | None:
        """Return a failure response only after the caller has applied pending teardown."""
        if self.storage_unavailable:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "adapter storage is unavailable"},
                background=background_tasks,
            )
        if not self.stuck_ready:
            return None
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=undeploy_conflict_detail(
                run_id,
                self.disabled_aliases,
                self.disabled_revisions,
                self.stuck_ready,
            ),
            background=background_tasks,
        )


async def _cas_row_to_disabled(
    candidate: AdapterRecord,
    *,
    get_authoritative: Callable[[str], Awaitable[AdapterRecord | None]],
) -> AdapterRecord | None:
    """CAS one row to "disabled", returning the terminal row it was last observed as.

    Storage outages propagate rather than being handled per call site: every read and write here
    fails the whole cascade the same way, so the caller owns that single decision. Disabled loading
    rows are also written so their timestamp advances and every in-flight promotion loses its stale
    cas authority. A vanished row and a non-ready row observed after a lost cas have converged;
    a row still ready after all attempts is the only conflict result.
    """

    current: AdapterRecord | None = candidate
    for _ in range(_CAS_ATTEMPTS):
        if current.updated_at is None:
            current = await get_authoritative(candidate.adapter_id)
            if current is None or current.updated_at is None:
                return current

        committed = await _replace_stored_cas(
            current.model_copy(update={"status": "disabled"}),
            expected_updated_at=current.updated_at,
        )
        if committed is not None:
            return committed
        current = await get_authoritative(candidate.adapter_id)
        if current is None or current.status != "ready":
            return current
    return current


async def disable_matched(
    matches: list[AdapterRecord],
    *,
    get_authoritative: Callable[[str], Awaitable[AdapterRecord | None]],
) -> DisableResult:
    """CAS every matched row to "disabled", returning what converged and what did not.

    The named result keeps teardown and failure state together without growing the former
    positional tuple. The cascade spans multiple rows with no cross-row transaction, so a sibling
    can stay stuck-ready. The
    caller still tears down every converged row (even on the conflict path) rather than deferring:
    a reload only rehydrates "ready" rows, so a row left disabled-but-registered would never be
    re-enumerated by a retry and its gpu lora registration would leak permanently.
    """
    disabled_aliases: list[str] = []
    disabled_revisions: list[str] = []
    stuck_ready: list[str] = []
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None]] = []
    storage_unavailable = False

    for candidate in matches:
        try:
            current = await _cas_row_to_disabled(candidate, get_authoritative=get_authoritative)
        except HTTPException as exc:
            if exc.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
                raise
            storage_unavailable = True
            break
        if current is not None and current.status == "ready":
            stuck_ready.append(candidate.adapter_id)
            continue
        pending_teardown.append((candidate, current))
        if candidate.is_alias:
            disabled_aliases.append(candidate.adapter_id)
        else:
            disabled_revisions.append(candidate.adapter_id)

    return DisableResult(
        disabled_aliases=disabled_aliases,
        disabled_revisions=disabled_revisions,
        stuck_ready=stuck_ready,
        pending_teardown=pending_teardown,
        storage_unavailable=storage_unavailable,
    )


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
        expected_generation = cleanup_record.deployment_generation
        # a loading revision has no persisted deployment generation, but register loaded its original
        # updated_at. carry that exact generation after fencing so cleanup can evict only that load.
        if expected_generation is None and candidate.is_revision and candidate.status == "disabled":
            expected_generation = candidate.updated_at
        cleanup_records.append((cleanup_record, expected_generation))
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


async def list_authoritative_run(run_id: str) -> list[AdapterRecord]:
    """Read every persisted lifecycle row for a run, including disabled loading revisions."""
    try:
        return await _list_run_stored(run_id)
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

    ready = [
        candidate
        for candidate in router.ready_records()
        if (candidate.is_alias and candidate.adapter_id == run_id)
        or (candidate.is_revision and candidate.run_id == run_id)
    ]
    persisted = await list_authoritative_run(run_id)
    # a disabled revision with no deployment generation is the durable row an in-flight load still
    # holds cas authority for. settled disabled rows already carry their former loaded generation.
    loading = [
        candidate
        for candidate in persisted
        if candidate.is_revision
        and candidate.run_id == run_id
        and candidate.status == "disabled"
        and candidate.deployment_generation is None
    ]
    ready_ids = {candidate.adapter_id for candidate in ready}
    matches = ready + [candidate for candidate in loading if candidate.adapter_id not in ready_ids]
    if not matches:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
    return None, run_id, matches


def undeploy_body(
    adapter_id: str,
    run_id: str,
    base_model: str,
    disabled_aliases: list[str],
    disabled_revisions: list[str],
) -> dict[str, Any]:
    return {
        "ok": True,
        "removed": adapter_id,
        "base_model": base_model,
        "run_id": run_id,
        "disabled_aliases": sorted(disabled_aliases),
        "disabled_revisions": sorted(disabled_revisions),
    }


def undeploy_conflict_detail(
    run_id: str,
    disabled_aliases: list[str],
    disabled_revisions: list[str],
    stuck_ready: list[str],
) -> dict[str, Any]:
    """409 body for a partially converged cascade.

    Same ``detail`` shape as an HTTPException, but returned as a response so the deferred gpu
    eviction still runs after the conflict is reported.
    """
    return {
        "detail": {
            "error": "adapter changed concurrently",
            "run_id": run_id,
            "disabled_aliases": sorted(disabled_aliases),
            "disabled_revisions": sorted(disabled_revisions),
            "stuck": sorted(stuck_ready),
        }
    }
