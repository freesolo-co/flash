"""single in-memory ledger for confirmed runpod create mutations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MutationKind = Literal["inference_secret", "artifact_secret", "volume", "template", "pod"]


@dataclass(frozen=True, slots=True)
class ConfirmedMutation:
    """one sanitized provider resource identity confirmed by a create response."""

    kind: MutationKind
    resource_id: str


@dataclass(slots=True)
class MutationLedger:
    """ordered confirmed creates for one bounded provision attempt."""

    _attempted: list[MutationKind] = field(default_factory=list)
    _confirmed: list[ConfirmedMutation] = field(default_factory=list)

    def begin(self, kind: MutationKind) -> None:
        if kind in self._attempted:
            raise RuntimeError("runpod mutation kind was attempted more than once")
        self._attempted.append(kind)

    def confirm(self, kind: MutationKind, resource_id: str) -> None:
        if kind not in self._attempted:
            raise RuntimeError("runpod mutation was confirmed before it began")
        if any(entry.kind == kind for entry in self._confirmed):
            raise RuntimeError("runpod mutation kind was confirmed more than once")
        self._confirmed.append(ConfirmedMutation(kind, resource_id))

    @property
    def has_attempted_creations(self) -> bool:
        return bool(self._attempted)

    def confirmed_id(self, kind: MutationKind) -> str | None:
        for entry in self._confirmed:
            if entry.kind == kind:
                return entry.resource_id
        return None

    def reversed(self) -> tuple[ConfirmedMutation, ...]:
        return tuple(reversed(self._confirmed))
