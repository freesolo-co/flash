"""Internal queue protocol for cancellable hosted streaming."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

PROTOCOL_VERSION = 1
DATA_PARTITION = "data"
CONTROL_PARTITION = "control"
HEARTBEAT_INTERVAL_SECONDS = 1.0
LEASE_SECONDS = 4.0
CONTROL_POLL_SECONDS = 0.25
DATA_GET_SECONDS = 0.25
TERMINAL_DRAIN_SECONDS = 1.0
CLEANUP_SECONDS = 1.0
ENGINE_CLEANUP_STEPS = 3
ENGINE_CLEANUP_WAITS_PER_STEP = 2
CALL_RESULT_MARGIN_SECONDS = 1.0
CALL_RESULT_SECONDS = (
    ENGINE_CLEANUP_STEPS * ENGINE_CLEANUP_WAITS_PER_STEP * CLEANUP_SECONDS
    + CALL_RESULT_MARGIN_SECONDS
)
PARTITION_TTL_SECONDS = 60

_FORBIDDEN_CANONICAL_FIELDS = frozenset(
    {
        "accesstoken",
        "apikey",
        "apisecret",
        "authorization",
        "authtoken",
        "bearer",
        "bearertoken",
        "clientsecret",
        "credentials",
        "freesolointernalkey",
        "hftoken",
        "huggingfacehubtoken",
        "huggingfacetoken",
        "hfcredentials",
        "internalkey",
        "modalcredentials",
        "modaltoken",
        "modaltokenid",
        "modaltokensecret",
        "password",
        "providerapikey",
        "providercredentials",
        "providerkey",
        "providerpassword",
        "providersecret",
        "providertoken",
        "rawauth",
        "refreshtoken",
        "secret",
        "supabaseaccesstoken",
        "supabaseanonkey",
        "supabasecredentials",
        "supabasekey",
        "supabaseservicerolekey",
        "supabasetoken",
    }
)
_DATA_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "generation_id",
        "invocation_nonce",
        "function_call_id",
        "engine_replica_id",
        "sequence",
        "terminal",
        "event",
        "error_code",
    }
)
_CONTROL_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "generation_id",
        "invocation_nonce",
        "function_call_id",
        "sequence",
        "lease_deadline_unix",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "generation_id",
        "invocation_nonce",
        "function_call_id",
        "engine_replica_id",
        "final_sequence",
        "terminal_kind",
        "event_count",
    }
)


class ChannelErrorCode(StrEnum):
    CANCELLED = "cancelled"
    CHANNEL_FAULT = "channel_fault"
    DISPATCH_DEADLINE = "dispatch_deadline"
    ENGINE_ERROR = "engine_error"
    LEASE_EXPIRED = "lease_expired"
    PROTOCOL_ERROR = "protocol_error"


class StreamChannelError(RuntimeError):
    """Fail-closed internal transport error with a stable non-sensitive code."""

    def __init__(self, code: ChannelErrorCode, detail: str) -> None:
        self.code = code
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ControlEnvelope:
    kind: Literal["active", "cancel"]
    generation_id: str
    invocation_nonce: str
    function_call_id: str | None
    sequence: int
    lease_deadline_unix: float
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = {
            "protocol_version": self.protocol_version,
            "kind": self.kind,
            "generation_id": self.generation_id,
            "invocation_nonce": self.invocation_nonce,
            "function_call_id": self.function_call_id,
            "sequence": self.sequence,
            "lease_deadline_unix": self.lease_deadline_unix,
        }
        _assert_no_credentials(value)
        return value


@dataclass(frozen=True, slots=True)
class DataEnvelope:
    kind: Literal["event", "error"]
    generation_id: str
    invocation_nonce: str
    function_call_id: str
    engine_replica_id: str
    sequence: int
    terminal: bool
    event: dict[str, Any] | None = None
    error_code: ChannelErrorCode | None = None
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = {
            "protocol_version": self.protocol_version,
            "kind": self.kind,
            "generation_id": self.generation_id,
            "invocation_nonce": self.invocation_nonce,
            "function_call_id": self.function_call_id,
            "engine_replica_id": self.engine_replica_id,
            "sequence": self.sequence,
            "terminal": self.terminal,
            "event": self.event,
            "error_code": self.error_code.value if self.error_code is not None else None,
        }
        _assert_no_credentials(value)
        return value


@dataclass(frozen=True, slots=True)
class TerminalManifest:
    generation_id: str
    invocation_nonce: str
    function_call_id: str
    engine_replica_id: str
    final_sequence: int
    terminal_kind: Literal["event", "error"]
    event_count: int
    protocol_version: int = PROTOCOL_VERSION
    kind: Literal["terminal_manifest"] = "terminal_manifest"

    def to_dict(self) -> dict[str, Any]:
        value = {
            "protocol_version": self.protocol_version,
            "kind": self.kind,
            "generation_id": self.generation_id,
            "invocation_nonce": self.invocation_nonce,
            "function_call_id": self.function_call_id,
            "engine_replica_id": self.engine_replica_id,
            "final_sequence": self.final_sequence,
            "terminal_kind": self.terminal_kind,
            "event_count": self.event_count,
        }
        _assert_no_credentials(value)
        return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, f"{label} must be a mapping")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    fields = frozenset(value)
    if fields != expected:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, f"invalid {label} fields")


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, f"invalid {label}")
    return value


def _sequence(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid sequence")
    return value


def _version(value: Any) -> int:
    if type(value) is not int or value != PROTOCOL_VERSION:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "unsupported protocol version")
    return PROTOCOL_VERSION


def _canonical_field_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _is_credential_field(value: str) -> bool:
    canonical = _canonical_field_name(value)
    credential_suffixes = (
        "accesstoken",
        "apikey",
        "apisecret",
        "authorization",
        "bearertoken",
        "hftoken",
        "huggingfacetoken",
        "modaltoken",
        "providerpassword",
        "providersecret",
        "providertoken",
        "refreshtoken",
        "supabasetoken",
    )
    return (
        canonical in _FORBIDDEN_CANONICAL_FIELDS
        or canonical.endswith(credential_suffixes)
        or canonical.startswith("bearertoken")
    )


def _assert_no_credentials(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and _is_credential_field(key):
                raise StreamChannelError(
                    ChannelErrorCode.PROTOCOL_ERROR,
                    "credential field is forbidden in stream channel",
                )
            _assert_no_credentials(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_credentials(nested)


def validate_control(value: Any) -> ControlEnvelope:
    raw = _mapping(value, label="control envelope")
    _exact_fields(raw, _CONTROL_FIELDS, label="control envelope")
    _assert_no_credentials(raw)
    _version(raw["protocol_version"])
    kind = raw["kind"]
    if kind not in {"active", "cancel"}:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid control kind")
    call_id = raw["function_call_id"]
    if call_id is not None:
        call_id = _string(call_id, label="function call id")
    lease = raw["lease_deadline_unix"]
    if (
        not isinstance(lease, (int, float))
        or isinstance(lease, bool)
        or not math.isfinite(lease)
        or lease <= 0
    ):
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid lease deadline")
    return ControlEnvelope(
        kind=kind,
        generation_id=_string(raw["generation_id"], label="generation id"),
        invocation_nonce=_string(raw["invocation_nonce"], label="invocation nonce"),
        function_call_id=call_id,
        sequence=_sequence(raw["sequence"]),
        lease_deadline_unix=float(lease),
    )


def validate_data(value: Any) -> DataEnvelope:
    raw = _mapping(value, label="data envelope")
    _exact_fields(raw, _DATA_FIELDS, label="data envelope")
    _assert_no_credentials(raw)
    _version(raw["protocol_version"])
    kind = raw["kind"]
    if kind not in {"event", "error"}:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid data kind")
    terminal = raw["terminal"]
    if not isinstance(terminal, bool):
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid terminal marker")
    event = raw["event"]
    error_code = raw["error_code"]
    if kind == "event":
        if not isinstance(event, dict) or error_code is not None:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid event envelope")
        if terminal != (event.get("type") == "final"):
            raise StreamChannelError(
                ChannelErrorCode.PROTOCOL_ERROR, "invalid event terminal state"
            )
        parsed_error = None
    else:
        if event is not None or not terminal or not isinstance(error_code, str):
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid error envelope")
        try:
            parsed_error = ChannelErrorCode(error_code)
        except ValueError as exc:
            raise StreamChannelError(
                ChannelErrorCode.PROTOCOL_ERROR, "invalid channel error code"
            ) from exc
    return DataEnvelope(
        kind=kind,
        generation_id=_string(raw["generation_id"], label="generation id"),
        invocation_nonce=_string(raw["invocation_nonce"], label="invocation nonce"),
        function_call_id=_string(raw["function_call_id"], label="function call id"),
        engine_replica_id=_string(raw["engine_replica_id"], label="engine replica id"),
        sequence=_sequence(raw["sequence"]),
        terminal=terminal,
        event=event,
        error_code=parsed_error,
    )


def validate_manifest(value: Any) -> TerminalManifest:
    raw = _mapping(value, label="terminal manifest")
    _exact_fields(raw, _MANIFEST_FIELDS, label="terminal manifest")
    _assert_no_credentials(raw)
    _version(raw["protocol_version"])
    if raw["kind"] != "terminal_manifest":
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid manifest kind")
    terminal_kind = raw["terminal_kind"]
    if terminal_kind not in {"event", "error"}:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid manifest terminal kind")
    event_count = raw["event_count"]
    if not isinstance(event_count, int) or isinstance(event_count, bool) or event_count < 1:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid manifest event count")
    final_sequence = _sequence(raw["final_sequence"])
    if event_count != final_sequence + 1:
        raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "manifest count mismatch")
    return TerminalManifest(
        generation_id=_string(raw["generation_id"], label="generation id"),
        invocation_nonce=_string(raw["invocation_nonce"], label="invocation nonce"),
        function_call_id=_string(raw["function_call_id"], label="function call id"),
        engine_replica_id=_string(raw["engine_replica_id"], label="engine replica id"),
        final_sequence=final_sequence,
        terminal_kind=terminal_kind,
        event_count=event_count,
    )


@dataclass(slots=True)
class DataSequenceValidator:
    generation_id: str
    invocation_nonce: str
    function_call_id: str
    next_sequence: int = 0
    engine_replica_id: str | None = None
    terminal: DataEnvelope | None = None

    def accept(self, value: Any) -> DataEnvelope:
        envelope = validate_data(value)
        if self.terminal is not None:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "data after terminal")
        if envelope.generation_id != self.generation_id:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "wrong generation id")
        if envelope.invocation_nonce != self.invocation_nonce:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "wrong invocation nonce")
        if envelope.function_call_id != self.function_call_id:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "wrong function call id")
        if envelope.sequence != self.next_sequence:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "non-monotonic data sequence")
        if self.engine_replica_id is None:
            self.engine_replica_id = envelope.engine_replica_id
        elif envelope.engine_replica_id != self.engine_replica_id:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "wrong engine replica id")
        self.next_sequence += 1
        if envelope.terminal:
            self.terminal = envelope
        return envelope

    def reconcile(self, value: Any) -> TerminalManifest:
        manifest = validate_manifest(value)
        terminal = self.terminal
        if terminal is None:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "manifest before terminal")
        expected = (
            self.generation_id,
            self.invocation_nonce,
            self.function_call_id,
            self.engine_replica_id,
            terminal.sequence,
            terminal.kind,
            self.next_sequence,
        )
        actual = (
            manifest.generation_id,
            manifest.invocation_nonce,
            manifest.function_call_id,
            manifest.engine_replica_id,
            manifest.final_sequence,
            manifest.terminal_kind,
            manifest.event_count,
        )
        if actual != expected:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "terminal manifest mismatch")
        return manifest


@dataclass(slots=True)
class ControlSequenceValidator:
    generation_id: str
    invocation_nonce: str
    function_call_id: str
    next_sequence: int = 0
    latest: ControlEnvelope | None = None

    def accept(self, value: Any) -> ControlEnvelope:
        envelope = validate_control(value)
        if envelope.generation_id != self.generation_id:
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "wrong control generation id")
        if envelope.invocation_nonce != self.invocation_nonce:
            raise StreamChannelError(
                ChannelErrorCode.PROTOCOL_ERROR, "wrong control invocation nonce"
            )
        if envelope.sequence != self.next_sequence:
            raise StreamChannelError(
                ChannelErrorCode.PROTOCOL_ERROR, "non-monotonic control sequence"
            )
        if envelope.sequence == 0:
            if envelope.kind != "active" or envelope.function_call_id is not None:
                raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "invalid initial control")
        elif envelope.function_call_id != self.function_call_id:
            raise StreamChannelError(
                ChannelErrorCode.PROTOCOL_ERROR, "wrong control function call id"
            )
        if self.latest is not None and self.latest.kind == "cancel":
            raise StreamChannelError(ChannelErrorCode.PROTOCOL_ERROR, "control after cancel")
        self.next_sequence += 1
        self.latest = envelope
        return envelope
