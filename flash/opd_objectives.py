"""dependency-free opd objective identifiers and config normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_OBJECTIVE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
OPD_OBJECTIVE_IDS = ("c0",)


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


def normalize_opd_objective_ids(value: Any, *, algorithm: str) -> tuple[str, ...]:
    """validate the typed, opd-only objective id list at serialization boundaries."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("train.opd_objective_ids must be a list of strings")
    if not all(isinstance(objective_id, str) and objective_id for objective_id in value):
        raise ValueError("train.opd_objective_ids entries must be non-empty strings")
    objective_ids = tuple(value)
    if objective_ids and algorithm != "opd":
        raise ValueError('train.opd_objective_ids is only valid when algorithm = "opd"')
    return resolve_opd_objective_ids(objective_ids)
