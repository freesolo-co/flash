"""Durable half of an immutable adapter registration.

Split out of router.py's ``add_adapter``. Reserves the run-alias namespace and persists the
revision; the caller owns routing upserts and the deferred gpu registration. Persistence is
reached through serving_io, and the adapter router is passed in rather than captured.
"""

from typing import Any

from fastapi import HTTPException, status

from flash.serving.src.persistence import PersistenceRecordError
from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord, internal_adapter_payload
from flash.serving.src.serving_io import (
    _get_stored,
    _insert_or_read,
    _replace_stored_cas,
    _validate_alias,
    _validate_alias_target,
)


def _assert_matches_existing(existing: AdapterRecord, revision: AdapterRecord) -> None:
    """A repeat registration is only idempotent against an identical immutable revision."""
    if existing.org_id != revision.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown adapter id")
    if (
        not existing.is_revision
        or existing.immutable_fingerprint() != revision.immutable_fingerprint()
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, "immutable adapter revision already exists")


async def _reserve_alias(
    router: AdapterRouter, revision: AdapterRecord, run_id: str
) -> tuple[AdapterRecord, bool]:
    """Claim the run-id alias namespace for this revision, inserting it disabled if absent."""
    in_memory_namespace = router.get(run_id)
    if in_memory_namespace is not None and in_memory_namespace.serve_base_model:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias namespace is occupied")
    try:
        alias = await _get_stored(run_id)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias namespace is occupied") from exc
    if alias is not None:
        return alias, False

    proposed_alias = revision.model_copy(
        update={
            "adapter_id": run_id,
            "checkpoint": None,
            "status": "disabled",
            "metadata": {
                "record_type": "alias",
                "run_id": run_id,
                "alias_of": revision.adapter_id,
            },
            "created_at": None,
            "updated_at": None,
        }
    )
    try:
        return await _insert_or_read(proposed_alias)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias namespace is occupied") from exc


async def _recover_or_validate_alias(
    alias: AdapterRecord,
    revision: AdapterRecord,
    *,
    allow_missing: str | None,
) -> AdapterRecord:
    alias_of = alias.alias_of
    if alias.status == "disabled" and alias_of != revision.adapter_id:
        assert alias_of is not None
        try:
            old_target = await _get_stored(alias_of)
        except PersistenceRecordError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "run alias target is invalid") from exc
        if old_target is None:
            if alias.updated_at is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "run alias has no CAS authority")
            replacement = alias.model_copy(
                update={
                    "metadata": {
                        "record_type": "alias",
                        "run_id": revision.run_id,
                        "alias_of": revision.adapter_id,
                    }
                }
            )
            committed = await _replace_stored_cas(
                replacement,
                expected_updated_at=alias.updated_at,
            )
            if committed is None:
                raise HTTPException(status.HTTP_409_CONFLICT, "run alias changed concurrently")
            alias = committed
    await _validate_alias_target(alias, allow_missing=allow_missing)
    return alias


async def persist_revision(
    router: AdapterRouter, revision: AdapterRecord
) -> tuple[AdapterRecord, AdapterRecord]:
    """Persist ``revision`` and its run alias, returning ``(alias, stored_revision)``.

    Idempotent: re-registering an identical revision returns the stored row rather than
    conflicting, which is what lets the control plane retry a registration safely.
    """
    try:
        existing = await _get_stored(revision.adapter_id)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "adapter namespace is occupied") from exc
    if existing is not None:
        _assert_matches_existing(existing, revision)
        stored = existing
    else:
        stored = revision

    run_id = revision.run_id
    assert run_id is not None
    alias, alias_inserted = await _reserve_alias(router, revision, run_id)
    _validate_alias(alias, revision)
    if not alias_inserted:
        alias = await _recover_or_validate_alias(
            alias,
            revision,
            allow_missing=revision.adapter_id if existing is None else None,
        )

    if existing is None:
        stored, _ = await _insert_or_read(revision)
        _assert_matches_existing(stored, revision)

    return alias, stored


async def activate_revision(
    router: AdapterRouter,
    revision_id: str,
    expected_adapter_revision: str | None,
) -> dict[str, Any]:
    """Point a run alias at ``revision_id``, compare-and-swapping against the caller's expectation.

    ``expected_adapter_revision`` is the revision the caller believes is live; a disabled alias
    reads as ``None``. A mismatch means someone else activated in between, which is a conflict
    rather than a silent overwrite.
    """
    try:
        revision = await _get_stored(revision_id)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "invalid revision record") from exc
    if revision is None or not revision.is_revision:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown adapter id")
    if revision.status != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, "adapter revision is not ready")
    run_id = revision.run_id
    assert run_id is not None
    try:
        alias = await _get_stored(run_id)
    except PersistenceRecordError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "invalid run alias") from exc
    if alias is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias is missing")
    _validate_alias(alias, revision)
    await _validate_alias_target(alias)
    previous = None if alias.status == "disabled" else alias.alias_of
    if expected_adapter_revision != previous:
        raise HTTPException(status.HTTP_409_CONFLICT, "stale adapter revision expectation")
    if alias.updated_at is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias has no CAS authority")

    replacement = alias.model_copy(
        update={
            "status": "ready",
            "metadata": {
                "record_type": "alias",
                "run_id": run_id,
                "alias_of": revision.adapter_id,
            },
        }
    )
    committed = await _replace_stored_cas(replacement, expected_updated_at=alias.updated_at)
    if committed is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "run alias changed concurrently")
    router.upsert(committed, revive=True)
    router.upsert(revision, revive=True)
    return {
        "adapter_id": run_id,
        "target_adapter_revision": revision.adapter_id,
        "previous_adapter_revision": previous,
        "checkpoint": revision.checkpoint,
        "updated_at": internal_adapter_payload(committed)["updated_at"],
    }
