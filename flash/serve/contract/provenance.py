"""typed permanent checkpoint provenance shared by serving boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flash.schema import format_checkpoint_ref, parse_checkpoint_ref

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


def immutable_binding_projection(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    """project one canonical immutable checkpoint binding from a record or flat payload."""

    def get(name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    org_id = get("org_id")
    run_id = get("run_id")
    step = get("checkpoint_step")
    checkpoint_id = get("adapter_id")
    if not isinstance(org_id, str) or not org_id.strip():
        raise ValueError("checkpoint binding requires org_id")
    if not isinstance(run_id, str):
        raise ValueError("checkpoint binding requires run_id")
    if isinstance(step, bool) or (step is not None and not isinstance(step, int)):
        raise ValueError("checkpoint binding has invalid checkpoint_step")
    if checkpoint_id != format_checkpoint_ref(run_id, step):
        raise ValueError("checkpoint binding has inconsistent permanent identity")
    projection = {field: get(field) for field in _BINDING_FIELDS}
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


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    """one public run-backed model identity."""

    checkpoint_id: str

    def __post_init__(self) -> None:
        if parse_checkpoint_ref(self.checkpoint_id) is None:
            raise ValueError("checkpoint provenance must be a permanent checkpoint identity")

    def validate(self, actual: CheckpointProvenance) -> None:
        if actual.checkpoint_id != self.checkpoint_id:
            raise ValueError("serving backend returned mismatched checkpoint provenance")

    def freesolo_body(self) -> dict[str, str]:
        return {"checkpoint_id": self.checkpoint_id}

    def freesolo_headers(self) -> dict[str, str]:
        return {"X-Freesolo-Checkpoint": self.checkpoint_id}


def decode_freesolo_body(payload: object) -> CheckpointProvenance:
    return _decode_field(
        _mapping(payload, "serving backend returned malformed checkpoint provenance")
    )


def decode_flash_body(payload: object) -> CheckpointProvenance:
    return _decode_field(
        _mapping(payload, "serving backend returned malformed packaged provenance")
    )


def decode_freesolo_headers(headers: Mapping[str, str]) -> CheckpointProvenance:
    normalized = {key.lower(): value for key, value in headers.items()}
    return _decode_header(normalized, "x-freesolo-checkpoint")


def decode_flash_headers(headers: Mapping[str, str]) -> CheckpointProvenance:
    normalized = {key.lower(): value for key, value in headers.items()}
    return _decode_header(normalized, "x-flash-checkpoint-id")


def validate_body_provenance(
    payload: dict[str, Any], expected: CheckpointProvenance
) -> dict[str, Any]:
    native = payload.get("freesolo")
    packaged = payload.get("flash_provenance")
    if native is None and packaged is None:
        raise ValueError("serving backend omitted checkpoint provenance")
    if native is not None:
        expected.validate(decode_freesolo_body(native))
    if packaged is not None:
        expected.validate(decode_flash_body(packaged))
    return {**payload, "freesolo": expected.freesolo_body()}


def validate_header_provenance(headers: Mapping[str, str], expected: CheckpointProvenance) -> None:
    normalized = {key.lower(): value for key, value in headers.items()}
    found = False
    if "x-freesolo-checkpoint" in normalized:
        expected.validate(decode_freesolo_headers(headers))
        found = True
    if "x-flash-checkpoint-id" in normalized:
        expected.validate(decode_flash_headers(headers))
        found = True
    if not found:
        raise ValueError("serving backend omitted checkpoint provenance")


def _mapping(value: object, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(detail)
    return value


def _decode_field(data: Mapping[str, Any]) -> CheckpointProvenance:
    value = data.get("checkpoint_id")
    if not isinstance(value, str) or not value:
        raise ValueError("serving backend omitted checkpoint provenance")
    return CheckpointProvenance(value)


def _decode_header(headers: Mapping[str, str], key: str) -> CheckpointProvenance:
    value = headers.get(key)
    if not value:
        raise ValueError("serving backend omitted checkpoint provenance")
    return CheckpointProvenance(value)
