"""Pure interpretation of serving responses: no i/o, no client, no module state.

These four helpers all answer "what does this response actually say?" and nothing else. They are
split out of `deploy.py` so that module stays under the file-size gate, and because a function
that only reads a dict is far easier to review here than buried among the request plumbing.

Anything that performs i/o, holds a client, or reads a tunable constant stays in `deploy.py`:
tests monkeypatch those on the deploy module by name, and moving one here would silently make
the patch inert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flash.serve.contract.errors import ServingError

if TYPE_CHECKING:
    # httpx is not installed by the base `pip install freesolo-flash` (every runtime dep lives in an
    # extra), so importing it at module scope would crash the base install. it is only needed for an
    # annotation here, and `from __future__ import annotations` keeps those unevaluated.
    import httpx


def serving_status_error(url: str, exc: httpx.HTTPStatusError) -> ServingError:
    """Build a ServingError from an upstream HTTP failure with a tailored hint."""
    resp = exc.response
    status = resp.status_code if resp is not None else None
    detail = ((resp.text if resp is not None else "") or "").strip()[:500]
    msg = f"serving backend error for {url}"
    if status is not None:
        msg += f" (HTTP {status})"
    if detail:
        msg += f": {detail}"
    if status is not None and status < 500:
        msg += (
            " — the serving backend rejected the request (4xx); check FREESOLO_INTERNAL_KEY "
            "and the request payload (this is a client/auth error, not a serving outage)"
        )
    else:
        msg += (
            " — the serving backend is unavailable or has no engine for this base model; "
            "an operator must check the freesolo serving deployment"
        )
    headers = getattr(resp, "headers", {}) if resp is not None else {}
    retry_after = headers.get("Retry-After")
    return ServingError(msg, status_code=status, retry_after=retry_after)


def matches_revision_identity(
    record: dict, expected: dict, *, require_provenance: bool = True
) -> bool:
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
    if not require_provenance:
        # backends without revision_provenance do not echo provenance metadata; the immutable
        # adapter_id already pins the artifact, so this cross-check is best effort here.
        return True
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    expected_metadata = expected["metadata"]
    return all(
        metadata.get(field) == expected_metadata.get(field)
        for field in ("record_type", "run_id", "checkpoint_step", "hf_revision")
    )


def active_alias_target(record: dict | None) -> str | None:
    if not isinstance(record, dict) or record.get("status") == "disabled":
        return None
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("alias_of"), str):
        return metadata["alias_of"]
    return None


def validate_activation_response(
    response: object,
    *,
    run_id: str,
    revision: str,
    checkpoint: str,
    expected_adapter_revision: str | None,
) -> dict:
    if not isinstance(response, dict):
        raise ServingError("serving returned an invalid alias activation response")
    if response.get("adapter_id") != run_id or response.get("target_adapter_revision") != revision:
        raise ServingError("serving returned mismatched committed alias activation provenance")
    if response.get("previous_adapter_revision") != expected_adapter_revision:
        raise ServingError("serving returned mismatched previous alias revision")
    if response.get("checkpoint") != checkpoint:
        raise ServingError("serving returned mismatched committed alias checkpoint")
    if not isinstance(response.get("updated_at"), str) or not response["updated_at"].strip():
        raise ServingError("serving returned committed alias activation without updated_at")
    return response
