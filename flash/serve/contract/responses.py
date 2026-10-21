"""pure interpretation of hosted serving responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.serve.contract.errors import ServingError
from flash.serve.contract.provenance import immutable_binding_fingerprint

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
    """compare the canonical immutable checkpoint fingerprint after ambiguous registration."""

    try:
        return record.get("artifact_fingerprint") == expected.get(
            "artifact_fingerprint"
        ) and immutable_binding_fingerprint(record) == immutable_binding_fingerprint(expected)
    except (TypeError, ValueError):
        return False
