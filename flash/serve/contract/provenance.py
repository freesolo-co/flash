"""typed permanent checkpoint provenance shared by serving boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flash.schema import parse_checkpoint_ref


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
    if "x-freesolo-checkpoint" in normalized:
        expected.validate(decode_freesolo_headers(headers))
        return
    if "x-flash-checkpoint-id" in normalized:
        expected.validate(decode_flash_headers(headers))
        return
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
