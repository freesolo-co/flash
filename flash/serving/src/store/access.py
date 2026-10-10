"""Canonical asynchronous access to durable adapter records."""

import asyncio

from fastapi import HTTPException, status

from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.persistence import PersistenceRecordError


async def _list_run_stored(org_id: str, run_id: str) -> list[AdapterRecord]:
    from flash.serving.src.store.persistence import list_run_adapters
    from flash.serving.src.store.settings import get_settings

    try:
        return await asyncio.to_thread(list_run_adapters, org_id, run_id, get_settings())
    except PersistenceRecordError:
        raise
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "adapter storage is unavailable"
        ) from exc


async def _get_stored(org_id: str, adapter_id: str) -> AdapterRecord | None:
    from flash.serving.src.store.persistence import PersistenceRecordError, get_adapter
    from flash.serving.src.store.settings import get_settings

    try:
        return await asyncio.to_thread(get_adapter, org_id, adapter_id, get_settings())
    except PersistenceRecordError:
        raise
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "adapter storage is unavailable"
        ) from exc


async def _insert_stored(record: AdapterRecord) -> AdapterRecord:
    from flash.serving.src.store.persistence import insert_adapter
    from flash.serving.src.store.settings import get_settings

    try:
        return await asyncio.to_thread(insert_adapter, record, get_settings())
    except Exception as exc:
        from flash.serving.src.store.persistence import (
            PersistenceConflict,
            PersistenceReferenceError,
        )

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
    from flash.serving.src.store.persistence import PersistenceConflict

    try:
        return await _insert_stored(record), True
    except PersistenceConflict as exc:
        if record.org_id is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "checkpoint registration requires org_id",
            ) from exc
        winner = await _get_stored(record.org_id, record.adapter_id)
        if winner is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "adapter registration conflict could not be confirmed",
            ) from exc
        return winner, False


async def _replace_stored_cas(
    record: AdapterRecord, *, expected_updated_at: str
) -> AdapterRecord | None:
    from flash.serving.src.store.persistence import replace_adapter_cas
    from flash.serving.src.store.settings import get_settings

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
