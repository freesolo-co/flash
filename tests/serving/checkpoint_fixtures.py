from __future__ import annotations

import hashlib
from typing import Any

from flash.schema import format_checkpoint_ref
from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.io.schemas import (
    AdapterRecord,
    ImmutableCheckpointRegistration,
    internal_adapter_payload,
)


def checkpoint_record(
    run_id: str,
    base_model: str,
    *,
    org_id: str = "org-1",
    checkpoint_step: int | None = None,
    status: str = "ready",
    thinking: bool = False,
    structured_outputs: dict[str, Any] | None = None,
    repo_id: str | None = None,
    repo_type: str = "dataset",
    subfolder: str | None = None,
    url: str | None = None,
    updated_at: str | None = "2026-07-14T00:00:01+00:00",
    deployment_generation: str | None = None,
    lora_rank: int = 16,
    **overrides: Any,
) -> AdapterRecord:
    checkpoint_id = format_checkpoint_ref(run_id, checkpoint_step)
    artifact_revision = hashlib.sha1(run_id.encode()).hexdigest()
    artifact_digest = hashlib.sha256(f"{run_id}:artifact".encode()).hexdigest()
    values: dict[str, Any] = {
        "adapter_id": checkpoint_id,
        "repo_id": repo_id or f"org/{run_id}",
        "base_model": base_model,
        "subfolder": subfolder,
        "repo_type": repo_type,
        "org_id": org_id,
        "url": url,
        "checkpoint": checkpoint_id,
        "private": True,
        "thinking": thinking,
        "structured_outputs": structured_outputs,
        "status": status,
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": updated_at,
        "deployment_generation": deployment_generation,
        "run_id": run_id,
        "checkpoint_step": checkpoint_step,
        "artifact_revision": artifact_revision,
        "artifact_digest": artifact_digest,
        "lora_rank": lora_rank,
    }
    values.update(overrides)
    values["artifact_fingerprint"] = immutable_binding_fingerprint(values)
    return AdapterRecord.model_validate(values)


def checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return internal_adapter_payload(checkpoint_record(*args, **kwargs))


def checkpoint_registration_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    record = checkpoint_record(*args, **kwargs)
    forwarded = internal_adapter_payload(record)
    registration = {name: forwarded[name] for name in ImmutableCheckpointRegistration.model_fields}
    return ImmutableCheckpointRegistration.model_validate(registration).model_dump(mode="json")
