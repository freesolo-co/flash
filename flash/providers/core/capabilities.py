"""Typed provider capabilities and authoritative cleanup outcomes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

ResourceId = str
RunLabels = set[str] | Callable[[], set[str]] | None


class CleanupContractError(Exception):
    """Base error for cleanup results that fail their runtime contract."""


class CleanupContractTypeError(CleanupContractError, TypeError):
    """A cleanup result field has the wrong runtime type."""


class CleanupContractValueError(CleanupContractError, ValueError):
    """Cleanup evidence is blank, duplicate, or contradictory."""


class CleanupOutcome(StrEnum):
    """The strongest conclusion an authoritative cleanup operation established."""

    DELETED = "deleted"
    ABSENT = "absent"
    PRESENT = "present"
    UNCONFIRMED = "unconfirmed"
    RETRYABLE = "retryable"
    UNSUPPORTED = "unsupported"


def _normalize_evidence(field: str, values: object) -> tuple[ResourceId, ...]:
    if not isinstance(values, (list, tuple)):
        raise CleanupContractTypeError(f"{field} must be a list or tuple of resource id strings")
    normalized: list[str] = []
    for value in values:
        if type(value) is not str:
            raise CleanupContractTypeError(f"{field} entries must be strings")
        if not value or value != value.strip():
            raise CleanupContractValueError(f"{field} entries must be canonical nonblank strings")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise CleanupContractValueError(f"{field} cannot contain duplicate resource ids")
    return tuple(normalized)


def _normalize_optional_evidence(field: str, values: object) -> tuple[ResourceId, ...] | None:
    if values is None:
        return None
    return _normalize_evidence(field, values)


@dataclass(frozen=True)
class CleanupResult:
    """Immutable evidence from one provider cleanup or absence check."""

    outcome: CleanupOutcome
    confirmed_deleted_ids: tuple[ResourceId, ...] = ()
    surviving_ids: tuple[ResourceId, ...] | None = None
    unresolved_ids: tuple[ResourceId, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.outcome) is not CleanupOutcome or not any(
            self.outcome is member for member in CleanupOutcome
        ):
            raise CleanupContractTypeError(
                "cleanup outcome must be an actual CleanupOutcome member"
            )

        deleted = _normalize_evidence("confirmed_deleted_ids", self.confirmed_deleted_ids)
        surviving = _normalize_optional_evidence("surviving_ids", self.surviving_ids)
        unresolved = _normalize_optional_evidence("unresolved_ids", self.unresolved_ids)
        object.__setattr__(self, "confirmed_deleted_ids", deleted)
        object.__setattr__(self, "surviving_ids", surviving)
        object.__setattr__(self, "unresolved_ids", unresolved)

        evidence = {
            "deleted": set(deleted),
            "surviving": set(surviving or ()),
            "unresolved": set(unresolved or ()),
        }
        for left, right in (
            ("deleted", "surviving"),
            ("deleted", "unresolved"),
            ("surviving", "unresolved"),
        ):
            if evidence[left] & evidence[right]:
                raise CleanupContractValueError(
                    f"cleanup evidence cannot mark one resource {left} and {right}"
                )

        has_deleted = bool(deleted)
        has_surviving = bool(surviving)
        has_unresolved = bool(unresolved)
        if self.outcome in {CleanupOutcome.DELETED, CleanupOutcome.ABSENT}:
            if has_surviving or has_unresolved:
                raise CleanupContractValueError(
                    "confirmed cleanup outcomes cannot carry outstanding resource ids"
                )
            if self.outcome is CleanupOutcome.ABSENT and has_deleted:
                raise CleanupContractValueError(
                    "absent cleanup outcomes cannot carry deleted resource ids"
                )
        elif self.outcome is CleanupOutcome.PRESENT:
            if not has_surviving or has_deleted or has_unresolved:
                raise CleanupContractValueError(
                    "present cleanup outcomes require only surviving resource ids"
                )
        elif self.outcome is CleanupOutcome.UNCONFIRMED:
            if not has_unresolved or has_surviving:
                raise CleanupContractValueError(
                    "unconfirmed cleanup outcomes require unresolved resource ids"
                )
        elif self.outcome in {CleanupOutcome.RETRYABLE, CleanupOutcome.UNSUPPORTED}:
            if has_deleted or has_surviving:
                raise CleanupContractValueError(
                    "inconclusive cleanup outcomes cannot carry settled resource ids"
                )

    @property
    def confirmed(self) -> bool:
        """Whether deletion or owner-authenticated absence was confirmed."""
        self.__post_init__()
        return self.outcome in {CleanupOutcome.DELETED, CleanupOutcome.ABSENT}

    @property
    def deleted_count(self) -> int:
        self.__post_init__()
        return len(self.confirmed_deleted_ids)


def _validated_callback_result(
    result: object,
    *,
    unresolved_ids: tuple[ResourceId, ...] = (),
) -> CleanupResult:
    if type(result) is not CleanupResult:
        return CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=unresolved_ids or None)
    try:
        result.__post_init__()
    except (CleanupContractError, AttributeError):
        return CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=unresolved_ids or None)
    if result.outcome is CleanupOutcome.UNSUPPORTED:
        return CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=unresolved_ids or None)
    return result


def is_cleanup_confirmed(result: object) -> bool:
    """Accept confirmation only from a validated cleanup result instance."""
    if type(result) is not CleanupResult:
        return False
    try:
        return result.confirmed
    except (CleanupContractError, AttributeError):
        return False


ShouldStop = Callable[[], bool]
ConfirmRunAbsent = Callable[[str], CleanupResult]
SweepOrphans = Callable[[RunLabels, RunLabels, ShouldStop | None], CleanupResult]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Required immutable declaration of provider-specific behavior."""

    supports_weight_cache: bool
    live_capacity: bool
    confirm_run_absent: ConfirmRunAbsent | None
    sweep_orphans: SweepOrphans | None


def confirm_run_absent(capabilities: ProviderCapabilities, run_id: str) -> CleanupResult:
    """Invoke authoritative run absence confirmation when the provider supports it."""
    callback = capabilities.confirm_run_absent
    if callback is None:
        return CleanupResult(CleanupOutcome.UNSUPPORTED, unresolved_ids=(run_id,))
    if not callable(callback):
        return CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=(run_id,))
    try:
        result = callback(run_id)
    except Exception:
        return CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=(run_id,))
    return _validated_callback_result(result, unresolved_ids=(run_id,))


def sweep_orphans(
    capabilities: ProviderCapabilities,
    active_labels: RunLabels = None,
    known_labels: RunLabels = None,
    should_stop: ShouldStop | None = None,
) -> CleanupResult:
    """Invoke authoritative orphan cleanup when the provider supports it.

    ``should_stop`` is checked by the provider between teardowns. The sweep runs in a worker
    thread that ``task.cancel()`` cannot interrupt, so without it a large in-flight sweep keeps
    destroying provider resources after the lifespan was told to stop. A sweep that halts early
    reports the teardowns it did confirm as ``UNCONFIRMED``, never as a completed clean sweep.
    """
    callback = capabilities.sweep_orphans
    if callback is None:
        return CleanupResult(CleanupOutcome.UNSUPPORTED)
    if not callable(callback):
        return CleanupResult(CleanupOutcome.RETRYABLE)
    try:
        result = callback(active_labels, known_labels, should_stop)
    except Exception:
        return CleanupResult(CleanupOutcome.RETRYABLE)
    return _validated_callback_result(result)
