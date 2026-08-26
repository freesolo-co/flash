"""tenant-scoped hosted checkpoint identity helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from flash.schema import parse_checkpoint_ref
from flash.serve.contract.provenance import (
    immutable_binding_fingerprint as immutable_binding_fingerprint,
)
from flash.serve.contract.provenance import (
    immutable_binding_projection as immutable_binding_projection,
)

CheckpointKey = tuple[str, str]


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
