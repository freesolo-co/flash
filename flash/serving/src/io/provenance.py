"""Immutable adapter provenance validation and response headers."""

import re
from typing import Any

from fastapi import HTTPException, status

from flash.serving.src.http.headers import _checkpoint_headers
from flash.serving.src.io.schemas import AdapterRecord


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
