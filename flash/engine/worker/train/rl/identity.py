"""Fail-closed completion identity validation for GRPO reward bridges."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_IDENTITY_KEYS = frozenset({"optimizer_step", "sample_index", "rollout_ordinal", "validate"})


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GRPO rollout identity {name} must be an integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"GRPO rollout identity {name} must be a boolean")
    return value


@dataclass(frozen=True, order=True)
class RolloutIdentity:
    optimizer_step: int
    sample_index: int
    rollout_ordinal: int
    validate: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "optimizer_step": self.optimizer_step,
            "sample_index": self.sample_index,
            "rollout_ordinal": self.rollout_ordinal,
            "validate": self.validate,
        }


@dataclass(frozen=True)
class _SealedEvidence:
    """A sealed step's identity evidence, summarized rather than enumerated.

    Both sides are held as a count and a digest instead of the identities themselves. The exact
    tuple is what `seal` compares -- equality of the registered and observed SETS is checked there,
    while both are still in memory -- so keeping the members afterwards would preserve the detail of
    a comparison that has already succeeded, and nothing downstream reads an individual identity.

    That detail is not free: one dict per completion per step, at the documented envelope of 512
    completions per optimizer step, measures 43 KB of json per step. A 10,000-step run therefore
    writes ~434 MB into `train_meta.json`, and `verl_config.py` deepcopies the payload before
    serializing it, so the peak is ~867 MB -- reached during `finalize()`, after paid training has
    already succeeded. Summarizing bounds the evidence at a constant size per step.

    `registered_sha256` and `observed_sha256` are computed independently over the two sets rather
    than one being copied from the other, so they still corroborate each other: a digest that
    matched only because it was assigned from its counterpart would prove nothing.
    """

    registered_count: int
    registered_sha256: str
    observed_count: int
    observed_sha256: str


def _identity_digest(identities: Iterable[RolloutIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in sorted(identities):
        digest.update(
            (
                f"{identity.optimizer_step}:{identity.sample_index}:"
                f"{identity.rollout_ordinal}:{int(identity.validate)}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def parse_rollout_identity(value: Any) -> RolloutIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("GRPO reward request is missing its rollout identity")
    if set(value) != _IDENTITY_KEYS:
        raise ValueError("GRPO rollout identity has an invalid field set")
    identity = RolloutIdentity(
        optimizer_step=_strict_int(value["optimizer_step"], "optimizer_step"),
        sample_index=_strict_int(value["sample_index"], "sample_index"),
        rollout_ordinal=_strict_int(value["rollout_ordinal"], "rollout_ordinal"),
        validate=_strict_bool(value["validate"], "validate"),
    )
    if identity.optimizer_step <= 0:
        raise ValueError("GRPO rollout identity optimizer_step must be positive")
    if identity.sample_index < 0:
        raise ValueError("GRPO rollout identity sample_index must be nonnegative")
    if identity.rollout_ordinal < 0:
        raise ValueError("GRPO rollout identity rollout_ordinal must be nonnegative")
    return identity


class RolloutIdentityLedger:
    """Register expected steps, record exact members, and seal equal identity sets."""

    def __init__(self, prompts_per_step: int, group_size: int) -> None:
        self._prompts_per_step = int(prompts_per_step)
        self._group_size = int(group_size)
        self._expected_count = self._prompts_per_step * self._group_size
        self._lock = threading.Lock()
        self._registered: dict[int, frozenset[RolloutIdentity]] = {}
        self._observed: dict[int, set[RolloutIdentity]] = {}
        self._sealed_steps: set[int] = set()
        # retain detailed identities once and a bounded independent summary of observations.
        self._sealed_evidence: dict[int, _SealedEvidence] = {}
        self._finalized = False

    def _reject_finalized(self) -> None:
        if self._finalized:
            raise ValueError("GRPO rollout identity ledger is already finalized")

    def _ensure_open(self) -> None:
        with self._lock:
            self._reject_finalized()

    def _evidence_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "steps": [
                {
                    "optimizer_step": step,
                    "registered": {
                        "count": sealed.registered_count,
                        "sha256": sealed.registered_sha256,
                    },
                    "observed": {
                        "count": sealed.observed_count,
                        "sha256": sealed.observed_sha256,
                    },
                }
                for step, sealed in sorted(self._sealed_evidence.items())
            ],
            "validation": [],
        }

    def _parse_training_identity(self, value: Any) -> RolloutIdentity:
        identity = parse_rollout_identity(value)
        if identity.validate:
            raise ValueError("GRPO training rejects rollout identities with validate=true")
        if identity.rollout_ordinal >= self._group_size:
            raise ValueError(
                f"GRPO rollout ordinal {identity.rollout_ordinal} is outside "
                f"[0, {self._group_size})"
            )
        return identity

    def _validate_registered_shape(self, identities: frozenset[RolloutIdentity]) -> None:
        if len(identities) != self._expected_count:
            raise ValueError(
                f"GRPO identity registration has {len(identities)} unique identities; "
                f"expected exactly {self._expected_count}"
            )
        by_prompt: dict[int, set[int]] = {}
        for identity in identities:
            by_prompt.setdefault(identity.sample_index, set()).add(identity.rollout_ordinal)
        if len(by_prompt) != self._prompts_per_step:
            raise ValueError(
                f"GRPO identity registration has {len(by_prompt)} prompts; "
                f"expected exactly {self._prompts_per_step}"
            )
        expected_ordinals = set(range(self._group_size))
        malformed = sorted(
            index for index, ordinals in by_prompt.items() if ordinals != expected_ordinals
        )
        if malformed:
            raise ValueError(
                "GRPO identity registration has missing or out-of-range rollout ordinals for "
                f"prompt indexes {malformed}"
            )

    def register(self, values: Iterable[Any]) -> int:
        self._ensure_open()
        parsed = tuple(self._parse_training_identity(value) for value in values)
        if len(parsed) != self._expected_count:
            raise ValueError(
                f"GRPO identity registration has {len(parsed)} identities; "
                f"expected exactly {self._expected_count}"
            )
        identities = frozenset(parsed)
        if len(identities) != len(parsed):
            raise ValueError("GRPO identity registration contains duplicate identities")
        steps = {identity.optimizer_step for identity in identities}
        if len(steps) != 1:
            raise ValueError("GRPO identity registration must contain exactly one optimizer step")
        self._validate_registered_shape(identities)
        step = next(iter(steps))
        with self._lock:
            self._reject_finalized()
            if step in self._sealed_steps:
                raise ValueError(f"GRPO identity registration targets sealed step {step}")
            existing = self._registered.get(step)
            if existing is not None:
                disposition = "duplicate" if existing == identities else "conflicting"
                raise ValueError(f"{disposition} GRPO identity registration for step {step}")
            self._registered[step] = identities
            self._observed[step] = set()
        return step

    def validate_for_index(self, value: Any, expected_index: int) -> RolloutIdentity:
        identity = self._parse_training_identity(value)
        if identity.sample_index != int(expected_index):
            raise ValueError(
                "GRPO rollout identity sample_index does not match the reward example index"
            )
        return identity

    def require_registered(self, value: Any, expected_index: int) -> RolloutIdentity:
        self._ensure_open()
        identity = self.validate_for_index(value, expected_index)
        step = identity.optimizer_step
        with self._lock:
            self._reject_finalized()
            if step in self._sealed_steps:
                raise ValueError(f"late GRPO rollout identity for sealed step {step}")
            expected = self._registered.get(step)
            if expected is None:
                raise ValueError(
                    f"GRPO reward arrived before identity registration for step {step}"
                )
            if identity not in expected:
                raise ValueError(
                    f"GRPO rollout identity is not registered for step {step}: {identity.to_dict()}"
                )
        return identity

    def record(self, value: Any, expected_index: int) -> RolloutIdentity:
        identity = self.require_registered(value, expected_index)
        with self._lock:
            self._reject_finalized()
            step = identity.optimizer_step
            observed = self._observed.get(step)
            if observed is None:
                if step in self._sealed_steps:
                    raise ValueError(f"late GRPO rollout identity for sealed step {step}")
                raise ValueError(
                    f"GRPO reward arrived before identity registration for step {step}"
                )
            if identity in observed:
                raise ValueError(f"duplicate GRPO rollout identity {identity.to_dict()}")
            observed.add(identity)
        return identity

    def seal(self, optimizer_step: int) -> None:
        self._ensure_open()
        step = _strict_int(optimizer_step, "optimizer_step")
        with self._lock:
            self._reject_finalized()
            expected = self._registered.get(step)
            if expected is None:
                if step in self._sealed_steps:
                    raise ValueError(f"GRPO step {step} identities are already sealed")
                raise ValueError(f"GRPO step {step} has no registered rollout identity set")
            observed = self._observed[step]
            if observed != expected:
                missing = len(expected - observed)
                unexpected = len(observed - expected)
                raise ValueError(
                    f"GRPO step {step} observed identity set does not equal registration: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            # digested from `expected` and `observed` separately. the two sets were just proven
            # equal above, so the digests necessarily agree -- but they agree because each was
            # computed from its own side, which is what makes their agreement evidence.
            self._sealed_evidence[step] = _SealedEvidence(
                registered_count=len(expected),
                registered_sha256=_identity_digest(expected),
                observed_count=len(observed),
                observed_sha256=_identity_digest(observed),
            )
            self._sealed_steps.add(step)
            del self._registered[step]
            del self._observed[step]

    def evidence(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return self._evidence_unlocked()

    def finalize(self, expected_steps: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            self._reject_finalized()
            expected: set[int] = set()
            for value in expected_steps:
                step = _strict_int(value, "optimizer_step")
                if step <= 0:
                    raise ValueError("GRPO expected optimizer steps must be positive")
                expected.add(step)
            if self._registered or self._observed:
                raise ValueError(
                    "GRPO rollout identity registrations remain unsealed for steps "
                    f"{sorted(self._registered)}"
                )
            if self._sealed_steps != expected:
                missing = sorted(expected - self._sealed_steps)
                extra = sorted(self._sealed_steps - expected)
                raise ValueError(
                    "GRPO sealed rollout identity steps do not equal the expected steps: "
                    f"missing={missing}, extra={extra}"
                )
            self._finalized = True
            return self._evidence_unlocked()
