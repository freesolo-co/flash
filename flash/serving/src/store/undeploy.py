"""durable exact-checkpoint disablement with lifecycle generation fencing."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from flash.schema import parse_checkpoint_ref
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.access import _get_stored, _list_run_stored, _replace_stored_cas
from flash.serving.src.store.persistence import PersistenceRecordError

_CAS_ATTEMPTS = 3


@dataclass
class DisableResult:
    """durable disablement retained until routing and gpu teardown are scheduled."""

    disabled_checkpoints: list[str]
    stuck_ready: list[str]
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None, str | None]]
    storage_unavailable: bool = False

    def failure_response(
        self, checkpoint_id: str, background_tasks: BackgroundTasks
    ) -> JSONResponse | None:
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
            content={
                "detail": {
                    "error": "checkpoint changed concurrently",
                    "checkpoint_id": checkpoint_id,
                    "disabled_checkpoints": sorted(self.disabled_checkpoints),
                    "stuck": sorted(self.stuck_ready),
                }
            },
            background=background_tasks,
        )


async def _cas_row_to_disabled(
    candidate: AdapterRecord,
    *,
    get_authoritative: Callable[[str], Awaitable[AdapterRecord | None]],
) -> tuple[AdapterRecord | None, str | None]:
    current: AdapterRecord | None = candidate
    expected_generation = candidate.deployment_generation
    for _ in range(_CAS_ATTEMPTS):
        if current.updated_at is None:
            current = await get_authoritative(candidate.adapter_id)
            if current is None or current.updated_at is None:
                return current, expected_generation
        expected_generation = current.deployment_generation or expected_generation
        committed = await _replace_stored_cas(
            current.model_copy(update={"status": "disabled", "deployment_generation": None}),
            expected_updated_at=current.updated_at,
        )
        if committed is not None:
            return committed, expected_generation
        current = await get_authoritative(candidate.adapter_id)
        if current is None or (
            current.status == "disabled" and current.deployment_generation is None
        ):
            return current, expected_generation
    return current, expected_generation


async def disable_matched(
    matches: list[AdapterRecord],
    *,
    get_authoritative: Callable[[str], Awaitable[AdapterRecord | None]],
) -> DisableResult:
    disabled_checkpoints: list[str] = []
    stuck_ready: list[str] = []
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None, str | None]] = []
    storage_unavailable = False
    for candidate in matches:
        if candidate.status == "disabled" and candidate.deployment_generation is None:
            continue
        try:
            current, expected_generation = await _cas_row_to_disabled(
                candidate, get_authoritative=get_authoritative
            )
        except HTTPException as exc:
            if exc.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
                raise
            storage_unavailable = True
            break
        if current is not None and (
            current.status == "ready" or current.deployment_generation is not None
        ):
            stuck_ready.append(candidate.adapter_id)
            continue
        pending_teardown.append((candidate, current, expected_generation))
        disabled_checkpoints.append(candidate.adapter_id)
    return DisableResult(
        disabled_checkpoints=disabled_checkpoints,
        stuck_ready=stuck_ready,
        pending_teardown=pending_teardown,
        storage_unavailable=storage_unavailable,
    )


def apply_teardown(
    router: AdapterRouter,
    pending_teardown: list[tuple[AdapterRecord, AdapterRecord | None, str | None]],
) -> list[tuple[AdapterRecord, str | None]]:
    cleanup_records: list[tuple[AdapterRecord, str | None]] = []
    for candidate, current, expected_generation in pending_teardown:
        if current is None:
            router.remove(candidate.adapter_id, org_id=candidate.org_id)
        else:
            router.upsert(current)
        cleanup_records.append((current or candidate, expected_generation))
    return cleanup_records


async def get_authoritative(org_id: str, adapter_id: str) -> AdapterRecord | None:
    try:
        return await _get_stored(org_id, adapter_id)
    except PersistenceRecordError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "adapter storage is unavailable",
        ) from exc


async def list_authoritative_run(org_id: str, run_id: str) -> list[AdapterRecord]:
    """internal administrative listing for exact checkpoint cleanup."""

    try:
        return await _list_run_stored(org_id, run_id)
    except PersistenceRecordError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "adapter storage is unavailable",
        ) from exc


async def resolve_undeploy_target(
    router: AdapterRouter, org_id: str, checkpoint_id: str
) -> tuple[AdapterRecord | None, AdapterRecord | None]:
    """resolve one public exact-checkpoint undeploy target."""

    if parse_checkpoint_ref(checkpoint_id) is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Unknown adapter id: {checkpoint_id}",
        )
    record = router.get(checkpoint_id, org_id=org_id)
    if record is None:
        record = await get_authoritative(org_id, checkpoint_id)
    if record is not None and record.status == "ready" and record.serve_base_model:
        return record, record
    if record is None:
        return None, None
    if not record.is_checkpoint:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {checkpoint_id}")
    return None, record


async def resolve_run_cleanup_targets(org_id: str, run_id: str) -> list[AdapterRecord]:
    """enumerate exact tenant run checkpoints for internal administrative cleanup."""

    return [
        record for record in await list_authoritative_run(org_id, run_id) if record.is_checkpoint
    ]


def undeploy_body(
    checkpoint_id: str,
    run_id: str,
    base_model: str,
    disabled_checkpoints: list[str],
    *,
    gpu_cleanup: str | None = None,
) -> dict[str, Any]:
    body = {
        "ok": True,
        "removed": checkpoint_id,
        "checkpoint_id": checkpoint_id,
        "base_model": base_model,
        "run_id": run_id,
        "disabled_checkpoints": sorted(disabled_checkpoints),
    }
    if gpu_cleanup is not None:
        body["gpu_cleanup"] = gpu_cleanup
    return body
