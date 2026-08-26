"""durable write-once checkpoint registration."""

from fastapi import HTTPException, status

from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.access import _get_stored, _insert_or_read
from flash.serving.src.store.persistence import PersistenceRecordError


def _assert_matches_existing(existing: AdapterRecord, checkpoint: AdapterRecord) -> None:
    """allow exact retries and reject every immutable identity mismatch."""

    if existing.org_id != checkpoint.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown adapter id")
    if (
        not existing.is_checkpoint
        or existing.immutable_fingerprint() != checkpoint.immutable_fingerprint()
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "immutable checkpoint already exists")


async def persist_checkpoint(checkpoint: AdapterRecord) -> AdapterRecord:
    """insert one checkpoint binding or return the identical existing record."""

    try:
        if checkpoint.org_id is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "checkpoint requires org_id")
        existing = await _get_stored(checkpoint.org_id, checkpoint.adapter_id)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "checkpoint namespace is occupied") from exc
    if existing is not None:
        _assert_matches_existing(existing, checkpoint)
        return existing

    try:
        stored, _ = await _insert_or_read(checkpoint)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "checkpoint namespace is occupied") from exc
    _assert_matches_existing(stored, checkpoint)
    return stored
