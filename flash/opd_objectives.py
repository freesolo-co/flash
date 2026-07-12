"""dependency-free opd objective identifiers and config normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_OBJECTIVE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
OPD_OBJECTIVE_IDS = ("c0", "c05", "c06", "c07", "c08", "c09", "c10", "c11", "c12", "c13", "c14")

# rollout-owning objective ids that opd c14 cannot be combined with. c14 owns the on-policy rollout
# (K=2 sampled continuations per prompt) plus the exact teacher-input budget/cache/metering. an
# objective that drives its own greedy sidecar (c07, c08, c12), samples an extra primary rollout
# (c08), or scores its own top-k student prefixes (c13) would run a SECOND rollout/teacher policy
# under c14. this id set lives here, dependency-free, so the submission schema/spec validators and
# the runtime worker share ONE source of truth and cannot drift; a worker test asserts it equals the
# set derived from the registry requirements (greedy_sidecar | sampled_primary | candidate_topk).
OPD_C14_CONFLICTING_OBJECTIVE_IDS = ("c07", "c08", "c12", "c13")
# the objectives c14 CAN compose with (only add a local forward / no extra rollout): everything that
# is not c14 itself and not rollout-owning -> c0, c05, c06, c09, c10, c11.
OPD_C14_COMPATIBLE_OBJECTIVE_IDS = tuple(
    objective_id
    for objective_id in OPD_OBJECTIVE_IDS
    if objective_id != "c14" and objective_id not in OPD_C14_CONFLICTING_OBJECTIVE_IDS
)


def opd_c14_conflicting_objective_ids(objective_ids: Iterable[str]) -> tuple[str, ...]:
    """rollout-owning objective ids in ``objective_ids`` that c14 cannot be combined with, preserving
    input order and de-duplicated. empty when c14 composes cleanly (c0/c05/c06/c09/c10/c11) or when
    c14 is absent — callers gate on c14 being present themselves."""
    seen: set[str] = set()
    conflicts: list[str] = []
    for objective_id in objective_ids:
        if objective_id in OPD_C14_CONFLICTING_OBJECTIVE_IDS and objective_id not in seen:
            conflicts.append(objective_id)
        seen.add(objective_id)
    return tuple(conflicts)


def opd_c14_conflict_message(conflicts: Iterable[str]) -> str:
    """shared error text for a rejected c14 + rollout-owning combination. names the conflicting ids
    and the compatible objectives; never claims c14 must run alone (it composes with the compatible
    set), so the worker and control plane surface the same accurate guidance."""
    conflicting = ", ".join(conflicts)
    compatible = ", ".join(OPD_C14_COMPATIBLE_OBJECTIVE_IDS)
    return (
        f"opd c14 listwise distillation cannot be combined with rollout-owning objective(s) "
        f"{conflicting}: c14 owns the on-policy rollout and teacher-input policy. c14 may still be "
        f"combined with {compatible}."
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
