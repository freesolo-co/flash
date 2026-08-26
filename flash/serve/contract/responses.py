"""pure interpretation of hosted serving responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.serve.contract.errors import ServingError

if TYPE_CHECKING:
    import httpx


def serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    message = f"serving backend error for {url}"
    if status is not None:
        message += f" (HTTP {status})"
    if detail:
        message += f": {detail}"
    if status is not None and status < 500:
        message += (
            " - the serving backend rejected the request; check FREESOLO_INTERNAL_KEY "
            "and the request payload"
        )
    else:
        message += " - the serving backend is unavailable or has no engine for this base model"
    headers = getattr(resp, "headers", {}) if resp is not None else {}
    return ServingError(
        message,
        status_code=status,
        retry_after=headers.get("Retry-After"),
    )


def matches_revision_identity(record: dict, expected: dict) -> bool:
    """compare every immutable checkpoint binding fact after an ambiguous registration."""

    scalar_fields = (
        "adapter_id",
        "repo_id",
        "repo_type",
        "subfolder",
        "base_model",
        "checkpoint",
        "thinking",
    )
    if any(record.get(field) != expected.get(field) for field in scalar_fields):
        return False
    if (record.get("org_id") or None) != (expected.get("org_id") or None):
        return False
    if (record.get("structured_outputs") or None) != (expected.get("structured_outputs") or None):
        return False
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected["metadata"]
    return all(
        metadata.get(field) == expected_metadata.get(field)
        for field in (
            "record_type",
            "run_id",
            "checkpoint_step",
            "artifact_revision",
            "artifact_digest",
        )
    )
