"""strict training-attempt lifecycle records and immutable artifact identities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1
ATTEMPT_STATES = frozenset(
    {"reserved", "provisioning", "active", "result_pending", "settling", "settled"}
)
PROGRESS_KINDS = frozenset(
    {"attempt_started", "phase_changed", "progressed", "checkpoint_failed", "checkpoint_saved"}
)
RESULT_OUTCOMES = frozenset({"succeeded", "failed", "deadline", "cancelled"})
FAILURE_CLASSES = frozenset(
    {"oom", "checkpoint", "worker", "provider_preempted", "artifact_transport", "deadline"}
)


def _whole(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _text(value: object, label: str, *, limit: int = 4096, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{label} must not be empty")
    if len(value) > limit:
        raise ValueError(f"{label} exceeds {limit} characters")
    return value


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def digest_record(value: dict) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _digest(value: object, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    text = _text(value, label, limit=64)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return text


def attempt_prefix(phase: str, run_id: str, attempt_id: int, fence: int) -> str:
    phase = _text(phase, "phase", limit=32)
    run_id = _text(run_id, "run_id", limit=128)
    if any(part in {".", ".."} or "/" in part for part in (phase, run_id)):
        raise ValueError("attempt artifact identity contains an unsafe path component")
    return f"{phase}/{run_id}/attempts/{_whole(attempt_id, 'attempt_id')}-{_whole(fence, 'fence')}"


def progress_path(record: ProgressRecord) -> str:
    payload = record.to_dict()
    digest = digest_record(payload)
    return f"{attempt_prefix(record.phase_namespace, record.run_id, record.attempt_id, record.fence)}/progress/{record.sequence:020d}-{digest}.json"


def result_path(manifest: ResultManifest) -> str:
    payload = manifest.to_dict()
    digest = digest_record(payload)
    return f"{attempt_prefix(manifest.phase_namespace, manifest.run_id, manifest.attempt_id, manifest.fence)}/result/{digest}.json"


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: int
    fence: int
    state: str
    reserved_at: float
    grant_deadline_at: float
    work_deadline_at: float
    result_deadline_at: float
    run_deadline_at: float
    provider: str | None = None
    provider_contract: dict | None = None
    resource: dict | None = None
    allocation: dict | None = None
    progress_receipt: dict | None = None
    result_receipt: dict | None = None
    cleanup: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _whole(self.schema_version, "schema_version", minimum=1)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported attempt schema version")
        _whole(self.attempt_id, "attempt_id")
        _whole(self.fence, "fence", minimum=1)
        if self.state not in ATTEMPT_STATES:
            raise ValueError("invalid attempt state")
        reserved = _finite(self.reserved_at, "reserved_at", positive=True)
        grant = _finite(self.grant_deadline_at, "grant_deadline_at", positive=True)
        work = _finite(self.work_deadline_at, "work_deadline_at", positive=True)
        result = _finite(self.result_deadline_at, "result_deadline_at", positive=True)
        run = _finite(self.run_deadline_at, "run_deadline_at", positive=True)
        if not reserved <= grant <= work <= run <= result:
            raise ValueError("attempt deadlines are not monotonic")
        if self.provider is not None:
            _text(self.provider, "provider", limit=32)
        for label, value in (
            ("provider_contract", self.provider_contract),
            ("resource", self.resource),
            ("allocation", self.allocation),
            ("progress_receipt", self.progress_receipt),
            ("result_receipt", self.result_receipt),
            ("cleanup", self.cleanup),
        ):
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"{label} must be an object")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> AttemptRecord:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid attempt record schema")
        return cls(**value)


@dataclass(frozen=True)
class ProgressRecord:
    run_id: str
    phase_namespace: str
    attempt_id: int
    fence: int
    sequence: int
    previous_digest: str | None
    occurred_at: float
    kind: str
    phase: str
    training_entered: bool
    completed_steps: int
    metrics: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    checkpoint: dict = field(default_factory=dict)
    gpu_observation: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported progress schema version")
        _text(self.run_id, "run_id", limit=128)
        _text(self.phase_namespace, "phase_namespace", limit=32)
        _whole(self.attempt_id, "attempt_id")
        _whole(self.fence, "fence", minimum=1)
        _whole(self.sequence, "sequence", minimum=1)
        _digest(self.previous_digest, "previous_digest", optional=True)
        _finite(self.occurred_at, "occurred_at", positive=True)
        if self.kind not in PROGRESS_KINDS:
            raise ValueError("invalid progress kind")
        _text(self.phase, "phase", limit=128)
        if type(self.training_entered) is not bool:
            raise ValueError("training_entered must be boolean")
        _whole(self.completed_steps, "completed_steps")
        for label, value in (
            ("metrics", self.metrics),
            ("timing", self.timing),
            ("checkpoint", self.checkpoint),
            ("gpu_observation", self.gpu_observation),
            ("diagnostics", self.diagnostics),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{label} must be an object")
        if not isinstance(self.samples, list):
            raise ValueError("samples must be an array")
        canonical_bytes(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ProgressRecord:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid progress record schema")
        return cls(**value)

    def follows(self, previous: ProgressRecord | None) -> bool:
        if previous is None:
            return self.sequence == 1 and self.previous_digest is None
        return (
            self.run_id == previous.run_id
            and self.attempt_id == previous.attempt_id
            and self.fence == previous.fence
            and self.sequence == previous.sequence + 1
            and self.previous_digest == digest_record(previous.to_dict())
            and self.occurred_at >= previous.occurred_at
            and self.completed_steps >= previous.completed_steps
            and (not previous.training_entered or self.training_entered)
        )


@dataclass(frozen=True)
class ResultManifest:
    run_id: str
    phase_namespace: str
    attempt_id: int
    fence: int
    outcome: str
    failure_class: str | None
    started_at: float
    finished_at: float
    training_entered: bool
    completed_steps: int
    metrics: dict
    checkpoint: dict
    artifacts: dict
    source_attestation: dict | None
    diagnostics: dict
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported result schema version")
        _text(self.run_id, "run_id", limit=128)
        _text(self.phase_namespace, "phase_namespace", limit=32)
        _whole(self.attempt_id, "attempt_id")
        _whole(self.fence, "fence", minimum=1)
        if self.outcome not in RESULT_OUTCOMES:
            raise ValueError("invalid result outcome")
        if self.outcome in {"succeeded", "cancelled"}:
            if self.failure_class is not None:
                raise ValueError(f"{self.outcome} result cannot carry a failure class")
        elif self.outcome == "deadline":
            if self.failure_class != "deadline":
                raise ValueError("deadline result requires the deadline failure class")
        elif self.failure_class not in FAILURE_CLASSES:
            raise ValueError("failed result requires a closed failure class")
        started = _finite(self.started_at, "started_at", positive=True)
        finished = _finite(self.finished_at, "finished_at", positive=True)
        if finished < started:
            raise ValueError("result finished_at precedes started_at")
        if type(self.training_entered) is not bool:
            raise ValueError("training_entered must be boolean")
        _whole(self.completed_steps, "completed_steps")
        for label, value in (
            ("metrics", self.metrics),
            ("checkpoint", self.checkpoint),
            ("artifacts", self.artifacts),
            ("diagnostics", self.diagnostics),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{label} must be an object")
        if not isinstance(self.source_attestation, dict) or not self.source_attestation:
            raise ValueError("result requires source attestation")
        if self.outcome == "succeeded" and not self.metrics:
            raise ValueError("successful result requires final metrics")
        if self.outcome == "succeeded" and not self.artifacts:
            raise ValueError("successful result requires final artifacts")
        canonical_bytes(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ResultManifest:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid result manifest schema")
        return cls(**value)


def receipt(path: str, revision: str, digest: str) -> dict[str, str]:
    return {
        "path": _text(path, "path", limit=512),
        "revision": _text(revision, "revision", limit=128),
        "digest": _digest(digest, "digest"),
    }


def record_identity_matches(record: dict, attempt: AttemptRecord) -> bool:
    try:
        return (
            _whole(record.get("attempt_id"), "attempt_id") == attempt.attempt_id
            and _whole(record.get("fence"), "fence", minimum=1) == attempt.fence
        )
    except ValueError:
        return False


def bounded_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)[:200]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [bounded_json(item, depth=depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {
            str(key)[:120]: bounded_json(item, depth=depth + 1)
            for key, item in list(value.items())[:64]
        }
    return str(value)[:500]
