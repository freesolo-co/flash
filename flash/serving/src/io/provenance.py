"""Immutable adapter provenance validation and response headers."""

from typing import Any

from fastapi import HTTPException, status

from flash.serving.src.http.headers import _checkpoint_headers
from flash.serving.src.io.schemas import AdapterRecord


def _active_checkpoint_ref(record: AdapterRecord) -> str:
    return record.adapter_id if record.is_checkpoint else ""


def require_attested_checkpoint(result: dict[str, Any], target: AdapterRecord) -> None:
    """Refuse a generation the engine did not attest to the resolved immutable adapter.

    Reads without consuming, so the caller still owns when to strip the field off the response
    body. The check lives here rather than in each route because it has to run before usage is
    metered: billing a caller for a generation we then reject with a 502 charges them for nothing.
    """

    if not target.is_checkpoint:
        return
    if result.get("lora_request_adapter") != target.adapter_id:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The serving engine did not attest the resolved immutable adapter.",
        )
    if result.get("checkpoint") != target.adapter_id:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The serving engine did not attest the resolved immutable checkpoint.",
        )


def _checkpoint_provenance(target: AdapterRecord, active_checkpoint: Any) -> dict[str, str] | None:
    if not target.is_checkpoint:
        return None
    checkpoint = str(active_checkpoint or "").strip()
    if checkpoint != target.adapter_id:
        return None
    return {"checkpoint_id": checkpoint}


def _provenance_headers(
    provenance: dict[str, str] | None, _active_checkpoint: Any
) -> dict[str, str]:
    checkpoint = (provenance or {}).get("checkpoint_id", "")
    return _checkpoint_headers(checkpoint)
