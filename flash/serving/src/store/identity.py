"""tenant-scoped hosted checkpoint identity helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from flash.schema import format_checkpoint_ref, parse_checkpoint_ref

CheckpointKey = tuple[str, str]

_BINDING_FIELDS = (
    "org_id",
    "run_id",
    "checkpoint_step",
    "adapter_id",
    "repo_id",
    "repo_type",
    "artifact_revision",
    "artifact_digest",
    "subfolder",
    "base_model",
    "lora_rank",
    "thinking",
    "structured_outputs",
)


def checkpoint_key(org_id: str, checkpoint_id: str) -> CheckpointKey:
    """return one validated tenant-scoped checkpoint key."""

    normalized_org = org_id.strip() if isinstance(org_id, str) else ""
    if not normalized_org:
        raise ValueError("org_id is required for checkpoint identity")
    if parse_checkpoint_ref(checkpoint_id) is None:
        raise ValueError("invalid permanent checkpoint identity")
    return normalized_org, checkpoint_id


def record_key(record: Any) -> CheckpointKey:
    """return the internal key for one checkpoint record."""

    return checkpoint_key(record.org_id, record.adapter_id)


def engine_adapter_name(org_id: str, checkpoint_id: str) -> str:
    """return an opaque tenant-scoped vllm adapter name."""

    org, checkpoint = checkpoint_key(org_id, checkpoint_id)
    digest = hashlib.sha256(f"{org}\0{checkpoint}".encode()).hexdigest()
    return f"fsckpt-{digest}"


def immutable_binding_projection(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    """project one canonical immutable checkpoint binding from a record or flat payload."""

    def get(name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    run_id = get("run_id")
    step = get("checkpoint_step")
    checkpoint_id = get("adapter_id") or get("checkpoint_id") or get("checkpoint")
    if isinstance(run_id, str) and (step is None or isinstance(step, int)):
        canonical = format_checkpoint_ref(run_id, step)
        if checkpoint_id != canonical:
            raise ValueError("checkpoint binding has inconsistent permanent identity")
    projection = {field: get(field) for field in _BINDING_FIELDS}
    projection["adapter_id"] = checkpoint_id
    projection["repo_type"] = projection["repo_type"] or "model"
    projection["structured_outputs"] = projection["structured_outputs"] or None
    return projection


def immutable_binding_fingerprint(value: Mapping[str, Any] | Any) -> str:
    """hash the canonical immutable binding independently of artifact content."""

    encoded = json.dumps(
        immutable_binding_projection(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
