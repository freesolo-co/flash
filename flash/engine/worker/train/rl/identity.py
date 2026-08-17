"""Fail-closed completion identity validation for GRPO reward bridges."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_IDENTITY_KEYS = frozenset({"optimizer_step", "sample_index", "rollout_ordinal"})


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GRPO rollout identity {name} must be an integer")
    return value


@dataclass(frozen=True, order=True)
class RolloutIdentity:
    optimizer_step: int
    sample_index: int
    rollout_ordinal: int

    def to_dict(self) -> dict[str, int]:
        return {
            "optimizer_step": self.optimizer_step,
            "sample_index": self.sample_index,
            "rollout_ordinal": self.rollout_ordinal,
        }


def parse_rollout_identity(value: Any) -> RolloutIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("GRPO reward request is missing its rollout identity")
    if set(value) != _IDENTITY_KEYS:
        raise ValueError("GRPO rollout identity has an invalid field set")
    identity = RolloutIdentity(
        optimizer_step=_strict_int(value["optimizer_step"], "optimizer_step"),
        sample_index=_strict_int(value["sample_index"], "sample_index"),
        rollout_ordinal=_strict_int(value["rollout_ordinal"], "rollout_ordinal"),
    )
    if identity.optimizer_step <= 0:
        raise ValueError("GRPO rollout identity optimizer_step must be positive")
    if identity.sample_index < 0:
        raise ValueError("GRPO rollout identity sample_index must be nonnegative")
    if identity.rollout_ordinal < 0:
        raise ValueError("GRPO rollout identity rollout_ordinal must be nonnegative")
    return identity


class RolloutIdentityLedger:
    """Validate each returned completion and seal exact optimizer-step identity sets."""

    def __init__(self, prompts_per_step: int, group_size: int) -> None:
        self._prompts_per_step = int(prompts_per_step)
        self._group_size = int(group_size)
        self._expected_count = self._prompts_per_step * self._group_size
        self._lock = threading.Lock()
        self._active_step: int | None = None
        self._active: set[RolloutIdentity] = set()
        self._sealed_steps: set[int] = set()

    def validate_for_index(self, value: Any, expected_index: int) -> RolloutIdentity:
        identity = parse_rollout_identity(value)
        if identity.sample_index != int(expected_index):
            raise ValueError(
                "GRPO rollout identity sample_index does not match the reward example index"
            )
        if identity.rollout_ordinal >= self._group_size:
            raise ValueError(
                f"GRPO rollout ordinal {identity.rollout_ordinal} is outside "
                f"[0, {self._group_size})"
            )
        return identity

    def record(self, value: Any, expected_index: int) -> RolloutIdentity:
        identity = self.validate_for_index(value, expected_index)
        with self._lock:
            if identity.optimizer_step in self._sealed_steps:
                raise ValueError(
                    f"late GRPO rollout identity for sealed step {identity.optimizer_step}"
                )
            if self._active_step is None:
                self._active_step = identity.optimizer_step
            elif identity.optimizer_step != self._active_step:
                raise ValueError(
                    f"cross-step GRPO rollout identity {identity.optimizer_step} arrived while "
                    f"step {self._active_step} is active"
                )
            if identity in self._active:
                raise ValueError(f"duplicate GRPO rollout identity {identity.to_dict()}")
            if len(self._active) >= self._expected_count:
                raise ValueError(
                    f"GRPO step {identity.optimizer_step} returned more than "
                    f"{self._expected_count} completions"
                )
            self._active.add(identity)
        return identity

    def seal(self, optimizer_step: int) -> None:
        step = _strict_int(optimizer_step, "optimizer_step")
        with self._lock:
            if self._active_step != step:
                raise ValueError(
                    f"GRPO step {step} cannot seal identities for active step {self._active_step}"
                )
            if len(self._active) != self._expected_count:
                raise ValueError(
                    f"GRPO step {step} returned {len(self._active)} completion identities; "
                    f"expected exactly {self._expected_count}"
                )
            by_prompt: dict[int, set[int]] = {}
            for identity in self._active:
                by_prompt.setdefault(identity.sample_index, set()).add(identity.rollout_ordinal)
            if len(by_prompt) != self._prompts_per_step:
                raise ValueError(
                    f"GRPO step {step} returned identities for {len(by_prompt)} prompts; "
                    f"expected exactly {self._prompts_per_step}"
                )
            expected_ordinals = set(range(self._group_size))
            malformed = sorted(
                index for index, ordinals in by_prompt.items() if ordinals != expected_ordinals
            )
            if malformed:
                raise ValueError(
                    f"GRPO step {step} has missing or out-of-range rollout ordinals for prompt "
                    f"indexes {malformed}"
                )
            self._sealed_steps.add(step)
            self._active_step = None
            self._active.clear()

    def assert_idle(self) -> None:
        with self._lock:
            if self._active_step is not None or self._active:
                raise ValueError(
                    f"GRPO rollout identities remain unsealed for step {self._active_step}"
                )
