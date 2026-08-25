"""typed immutable adapter provenance shared by serving boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flash.schema import format_checkpoint_ref, parse_adapter_revision


@dataclass(frozen=True, slots=True)
class ImmutableProvenance:
    """one immutable adapter identity rendered across active wire contracts."""

    adapter_revision: str
    checkpoint: str
    hf_revision: str

    @classmethod
    def from_adapter_revision(cls, adapter_revision: str) -> ImmutableProvenance:
        parsed = parse_adapter_revision(adapter_revision)
        if parsed is None:
            raise ValueError("managed chat target is not an immutable adapter revision")
        run_id, step, hf_revision = parsed
        return cls(
            adapter_revision=adapter_revision,
            checkpoint=format_checkpoint_ref(run_id, step),
            hf_revision=hf_revision,
        )

    def validate(self, actual: ImmutableProvenance) -> None:
        for field in ("adapter_revision", "checkpoint", "hf_revision"):
            if getattr(actual, field) != getattr(self, field):
                raise ValueError(
                    f"serving backend returned mismatched immutable provenance field {field}"
                )

    def freesolo_body(self) -> dict[str, str]:
        return {
            "adapter_revision": self.adapter_revision,
            "checkpoint": self.checkpoint,
            "hf_revision": self.hf_revision,
        }

    def freesolo_headers(self) -> dict[str, str]:
        return {
            "X-Freesolo-Adapter-Revision": self.adapter_revision,
            "X-Freesolo-Checkpoint": self.checkpoint,
            "X-Freesolo-HF-Revision": self.hf_revision,
        }


def decode_freesolo_body(payload: object) -> ImmutableProvenance:
    data = _mapping(payload, "serving backend returned malformed immutable provenance")
    return _decode_fields(data, source_key="hf_revision")


def decode_flash_body(payload: object) -> ImmutableProvenance:
    data = _mapping(payload, "serving backend returned malformed packaged provenance")
    return _decode_fields(data, source_key="source_revision")


def decode_freesolo_headers(headers: Mapping[str, str]) -> ImmutableProvenance:
    normalized = {key.lower(): value for key, value in headers.items()}
    return _decode_header_fields(
        normalized,
        adapter_key="x-freesolo-adapter-revision",
        checkpoint_key="x-freesolo-checkpoint",
        source_key="x-freesolo-hf-revision",
    )


def decode_flash_headers(headers: Mapping[str, str]) -> ImmutableProvenance:
    normalized = {key.lower(): value for key, value in headers.items()}
    return _decode_header_fields(
        normalized,
        adapter_key="x-flash-adapter-revision",
        checkpoint_key="x-flash-checkpoint",
        source_key="x-flash-source-revision",
    )


def validate_body_provenance(
    payload: dict[str, Any], expected: ImmutableProvenance
) -> dict[str, Any]:
    native = payload.get("freesolo")
    packaged = payload.get("flash_provenance")
    if native is None and packaged is None:
        raise ValueError("serving backend omitted immutable provenance")
    if native is not None:
        expected.validate(decode_freesolo_body(native))
    if packaged is not None:
        actual = decode_flash_body(packaged)
        if actual.adapter_revision != expected.adapter_revision:
            raise ValueError(
                "serving backend returned mismatched immutable provenance field adapter_revision"
            )
        if actual.checkpoint != expected.checkpoint:
            raise ValueError(
                "serving backend returned mismatched immutable provenance field checkpoint"
            )
        if actual.hf_revision != expected.hf_revision:
            raise ValueError("serving backend returned mismatched immutable source revision")
    return {**payload, "freesolo": expected.freesolo_body()}


def validate_header_provenance(headers: Mapping[str, str], expected: ImmutableProvenance) -> None:
    normalized = {key.lower(): value for key, value in headers.items()}
    has_freesolo = any(
        key in normalized
        for key in (
            "x-freesolo-adapter-revision",
            "x-freesolo-checkpoint",
            "x-freesolo-hf-revision",
        )
    )
    has_flash = any(key.startswith("x-flash-") for key in normalized)
    if not has_freesolo and not has_flash:
        raise ValueError("serving backend omitted adapter_revision provenance")
    if has_freesolo:
        expected.validate(decode_freesolo_headers(headers))
    if has_flash:
        actual = decode_flash_headers(headers)
        for field in ("adapter_revision", "checkpoint", "hf_revision"):
            if getattr(actual, field) != getattr(expected, field):
                raise ValueError(f"serving backend returned mismatched {field} provenance")


def _mapping(value: object, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(detail)
    return value


def _decode_fields(data: Mapping[str, Any], *, source_key: str) -> ImmutableProvenance:
    values: dict[str, str] = {}
    for field, key in (
        ("adapter_revision", "adapter_revision"),
        ("checkpoint", "checkpoint"),
        ("hf_revision", source_key),
    ):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            label = "source_revision" if key == "source_revision" else field
            raise ValueError(f"serving backend omitted immutable provenance field {label}")
        values[field] = value
    return ImmutableProvenance(**values)


def _decode_header_fields(
    headers: Mapping[str, str],
    *,
    adapter_key: str,
    checkpoint_key: str,
    source_key: str,
) -> ImmutableProvenance:
    values: dict[str, str] = {}
    for field, key in (
        ("adapter_revision", adapter_key),
        ("checkpoint", checkpoint_key),
        ("hf_revision", source_key),
    ):
        value = headers.get(key)
        if not value:
            raise ValueError(f"serving backend omitted {field} provenance")
        values[field] = value
    return ImmutableProvenance(**values)
