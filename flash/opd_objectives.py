"""dependency-free opd objective identifiers and config normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_OBJECTIVE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
OPD_OBJECTIVE_IDS = ("c0", "c05", "c06", "c07", "c08", "c09", "c10", "c11", "c12", "c13", "c14")

# objective ids that opd c14 cannot be combined with. c14 owns the on-policy rollout
# (k=2 sampled continuations per prompt) plus its teacher-input budget, cache, and metering.
# c07 and c12 own a greedy-sidecar rollout. c08 modifies the existing primary rollout and also
# owns a greedy-sidecar rollout. c13 owns an independent teacher-scoring and metering policy over
# top-k student prefixes. this dependency-free set gives submission validators and the runtime
# worker one source of truth; a worker test checks it against the corresponding registry requirements.
OPD_C14_CONFLICTING_OBJECTIVE_IDS = ("c07", "c08", "c12", "c13")
# c14 can compose with objectives that own no independent rollout or teacher-scoring policy.
OPD_C14_COMPATIBLE_OBJECTIVE_IDS = tuple(
    objective_id
    for objective_id in OPD_OBJECTIVE_IDS
    if objective_id != "c14" and objective_id not in OPD_C14_CONFLICTING_OBJECTIVE_IDS
)


def opd_c14_conflicting_objective_ids(objective_ids: Iterable[str]) -> tuple[str, ...]:
    """objectives with independent rollout or teacher-scoring policies that conflict with c14.

    preserves input order and de-duplicates. returns empty when c14 composes cleanly with
    c0/c05/c06/c09/c10/c11 or when c14 is absent; callers gate on c14 themselves.
    """
    seen: set[str] = set()
    conflicts: list[str] = []
    for objective_id in objective_ids:
        if objective_id in OPD_C14_CONFLICTING_OBJECTIVE_IDS and objective_id not in seen:
            conflicts.append(objective_id)
        seen.add(objective_id)
    return tuple(conflicts)


def opd_c14_conflict_message(conflicts: Iterable[str]) -> str:
    """shared error text for a rejected c14 policy conflict.

    names the conflicting ids and compatible objectives without claiming c14 must run alone, so the
    worker and control plane surface the same guidance.
    """
    conflicting = ", ".join(conflicts)
    compatible = ", ".join(OPD_C14_COMPATIBLE_OBJECTIVE_IDS)
    return (
        "opd c14 listwise distillation cannot be combined with objective(s) that own an independent "
        f"rollout or teacher-scoring policy: {conflicting}. c14 owns the on-policy rollout and "
        f"teacher-input policy. c14 may still be combined with {compatible}."
    )


def validate_opd_objective_id(objective_id: str) -> str:
    """validate one canonical objective identifier."""
    if not isinstance(objective_id, str) or not _OBJECTIVE_ID_RE.fullmatch(objective_id):
        raise ValueError(f"objective id must match [a-z][a-z0-9_.-]*, got {objective_id!r}")
    return objective_id


def resolve_opd_objective_ids(objective_ids: Iterable[str]) -> tuple[str, ...]:
    """resolve ids against the closed production identifier set."""
    requested = tuple(objective_ids)
    seen: set[str] = set()
    duplicate: list[str] = []
    for objective_id in requested:
        validate_opd_objective_id(objective_id)
        if objective_id in seen and objective_id not in duplicate:
            duplicate.append(objective_id)
        seen.add(objective_id)
    if duplicate:
        raise ValueError(f"duplicate opd objective id(s): {', '.join(duplicate)}")
    unknown = [objective_id for objective_id in requested if objective_id not in OPD_OBJECTIVE_IDS]
    if unknown:
        allowed = ", ".join(OPD_OBJECTIVE_IDS) or "none"
        raise ValueError(f"unknown opd objective id(s): {', '.join(unknown)}; allowed: {allowed}")
    return requested


def deserialize_opd_objective_ids(value: Any) -> tuple[str, ...]:
    """deserialize the typed id sequence without applying version-specific policy."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("train.opd_objective_ids must be a list of strings")
    if not all(isinstance(objective_id, str) for objective_id in value):
        raise ValueError("train.opd_objective_ids entries must be strings")
    return tuple(value)


def normalize_opd_objective_ids(value: Any, *, algorithm: str) -> tuple[str, ...]:
    """validate the typed, opd-only objective id list at submission boundaries."""
    objective_ids = deserialize_opd_objective_ids(value)
    if not all(objective_ids):
        raise ValueError("train.opd_objective_ids entries must be non-empty strings")
    if objective_ids and algorithm != "opd":
        raise ValueError('train.opd_objective_ids is only valid when algorithm = "opd"')
    return resolve_opd_objective_ids(objective_ids)
