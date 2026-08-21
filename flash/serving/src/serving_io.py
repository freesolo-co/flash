"""Request shaping and adapter persistence helpers for the serving app.

Split out of build_serving_app: every function here is a pure transform over its arguments or a
thin call into persistence, so none of them touch the app's closure state (the engine pool, the
router, the reload bookkeeping). Keeping them at module scope makes them directly testable and
keeps the app factory readable.
"""

# Do NOT add `from __future__ import annotations`: _parse_generate is annotated with the same
# pydantic body model the FastAPI handlers use, and the future import turns those annotations into
# unresolvable strings -> silent 422.

import asyncio
import re
from typing import Any

import orjson
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from flash.serving.src.http_headers import _checkpoint_headers
from flash.serving.src.model_config import (
    base_models,
    image_limit_for,
    is_supported_base_model,
    supports_image_input,
)
from flash.serving.src.multimodal import MultimodalRequestError, validate_multimodal_request
from flash.serving.src.persistence import PersistenceRecordError
from flash.serving.src.schemas import AdapterRecord, GenerateRequest


def _parse_generate(data: dict[str, Any]) -> GenerateRequest:
    # Untyped dict body -> surface a bad shape as 422, not 500.
    try:
        return GenerateRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _assert_supported_base_model(base_model: str) -> None:
    if is_supported_base_model(base_model):
        return
    allowed = ", ".join(base_models())
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        f"Unsupported base model: {base_model}. Supported base models: {allowed}",
    )


def _active_checkpoint_ref(record: AdapterRecord) -> str:
    checkpoint = (record.checkpoint or "").strip()
    if checkpoint:
        return checkpoint
    subfolder = (record.subfolder or "").strip().strip("/")
    if not subfolder:
        return ""
    match = re.search(r"(?:^|/)checkpoints/(step-\d+)(?:/|$)", subfolder)
    if match:
        return f"{record.adapter_id}/{match.group(1)}"
    return record.adapter_id


def require_attested_revision(result: dict[str, Any], target: AdapterRecord) -> None:
    """Refuse a generation the engine did not attest to the resolved immutable adapter.

    Reads without consuming, so the caller still owns when to strip the field off the response
    body. The check lives here rather than in each route because it has to run before usage is
    metered: billing a caller for a generation we then reject with a 502 charges them for nothing.
    """

    if target.is_revision and result.get("lora_request_adapter") != target.adapter_id:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The serving engine did not attest the resolved immutable adapter.",
        )


def _revision_provenance(target: AdapterRecord, active_checkpoint: Any) -> dict[str, str] | None:
    # immutable-revision provenance the deploy handshake verifies and external clients read to
    # confirm which revision answered. only concrete revisions carry it; base-model serving and
    # records missing a revision id, hub sha, or checkpoint do not.
    if not target.is_revision:
        return None
    adapter_revision = (target.adapter_id or "").strip()
    hf_revision = (target.hf_revision or "").strip()
    checkpoint = str(active_checkpoint or "").strip()
    if not adapter_revision or not hf_revision or not checkpoint:
        return None
    return {
        "adapter_revision": adapter_revision,
        "checkpoint": checkpoint,
        "hf_revision": hf_revision,
    }


def _provenance_headers(
    provenance: dict[str, str] | None, active_checkpoint: Any
) -> dict[str, str]:
    # full revision provenance headers for a revision, else the checkpoint-only header for
    # base-model and unresolved records (unchanged behaviour).
    if provenance is None:
        return _checkpoint_headers(active_checkpoint)
    return {
        "X-Freesolo-Adapter-Revision": provenance["adapter_revision"],
        "X-Freesolo-Checkpoint": provenance["checkpoint"],
        "X-Freesolo-HF-Revision": provenance["hf_revision"],
    }


def _inference_json_response(result: dict[str, Any], target: AdapterRecord) -> JSONResponse:
    # attach revision provenance while keeping engine-process attribution internal to metering.
    active_checkpoint = result.get("checkpoint")
    provenance = _revision_provenance(target, active_checkpoint)
    internal_fields = {
        "cached_tokens_reported",
        "engine_replica_id",
        "lora_request_adapter",
    }
    public_result = {key: value for key, value in result.items() if key not in internal_fields}
    body = {**public_result, "freesolo": provenance} if provenance is not None else public_result
    return JSONResponse(body, headers=_provenance_headers(provenance, active_checkpoint))


def _expected_checkpoint(request: Request) -> str | None:
    value = request.headers.get("X-Freesolo-Expected-Checkpoint")
    return value.strip() if value is not None else None


async def _get_stored(adapter_id: str) -> AdapterRecord | None:
    from flash.serving.src.persistence import PersistenceRecordError, get_adapter
    from flash.serving.src.settings import get_settings

    try:
        return await asyncio.to_thread(get_adapter, adapter_id, get_settings())
    except PersistenceRecordError:
        raise
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "adapter storage is unavailable"
        ) from exc


async def _insert_stored(record: AdapterRecord) -> AdapterRecord:
    from flash.serving.src.persistence import insert_adapter
    from flash.serving.src.settings import get_settings

    try:
        return await asyncio.to_thread(insert_adapter, record, get_settings())
    except Exception as exc:
        from flash.serving.src.persistence import PersistenceConflict, PersistenceReferenceError

        if isinstance(exc, PersistenceConflict):
            raise
        if isinstance(exc, PersistenceReferenceError):
            # a dangling reference is permanent, not an outage: report it as the caller's
            # unprocessable request so the client fails fast on the real cause instead of
            # retrying an unregistrable adapter against a 503.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "adapter storage is unavailable"
        ) from exc


async def _insert_or_read(record: AdapterRecord) -> tuple[AdapterRecord, bool]:
    from flash.serving.src.persistence import PersistenceConflict

    try:
        return await _insert_stored(record), True
    except PersistenceConflict as exc:
        winner = await _get_stored(record.adapter_id)
        if winner is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "adapter registration conflict could not be confirmed",
            ) from exc
        return winner, False


async def _replace_stored_cas(
    record: AdapterRecord, *, expected_updated_at: str
) -> AdapterRecord | None:
    from flash.serving.src.persistence import replace_adapter_cas
    from flash.serving.src.settings import get_settings

    try:
        return await asyncio.to_thread(
            replace_adapter_cas,
            record,
            expected_updated_at=expected_updated_at,
            settings=get_settings(),
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "adapter storage is unavailable"
        ) from exc


def _validate_alias(alias: AdapterRecord, revision: AdapterRecord) -> None:
    if alias.org_id != revision.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown adapter id")
    if (
        not alias.is_alias
        or alias.adapter_id != revision.run_id
        or alias.run_id != revision.run_id
        or alias.base_model != revision.base_model
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias namespace is occupied")


async def _validate_alias_target(
    alias: AdapterRecord, *, allow_missing: str | None = None
) -> AdapterRecord | None:
    alias_of = alias.alias_of
    if alias_of is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias target is invalid")
    try:
        target = await _get_stored(alias_of)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias target is invalid") from exc
    if target is None:
        if alias.status == "disabled" and alias_of == allow_missing:
            return None
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias target is invalid")
    if (
        not target.is_revision
        or target.org_id != alias.org_id
        or target.base_model != alias.base_model
        or target.run_id != alias.run_id
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias target is invalid")
    return target


async def _prepare_generate_request(payload: Any, target: AdapterRecord) -> None:
    messages = getattr(payload, "messages", None)
    # validate whenever any message uses list-form content so unsupported blocks
    # (e.g. video) are rejected before dispatch; plain string-content requests still bypass.
    if not isinstance(messages, list) or not any(
        isinstance(message, dict) and isinstance(message.get("content"), list)
        for message in messages
    ):
        return
    try:
        await asyncio.to_thread(
            validate_multimodal_request,
            messages,
            supports_images=supports_image_input(target.base_model),
            image_limit=image_limit_for(target.base_model),
        )
    except MultimodalRequestError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _sse(data: dict[str, Any] | str) -> bytes:
    encoded = data.encode("utf-8") if isinstance(data, str) else orjson.dumps(data)
    return b"data: " + encoded + b"\n\n"
