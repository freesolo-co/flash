"""Checkpoint deployment inventory and readiness, stored atomically in run status."""

from __future__ import annotations

import math

from flash.schema import parse_checkpoint_ref

DEPLOYMENT_STATES = frozenset({"deploying", "ready", "undeploying", "undeployed", "failed"})
BUSY_STATES = frozenset({"deploying"})


def require_checkpoint(run_id: str, checkpoint_id: str) -> int | None:
    parsed = parse_checkpoint_ref(checkpoint_id) if isinstance(checkpoint_id, str) else None
    if parsed is None or parsed[0] != run_id:
        raise ValueError("deployment requires a canonical checkpoint belonging to the run")
    return parsed[1]


def generation(record: dict) -> int:
    value = record.get("generation", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid checkpoint deployment generation")
    return value


def validate_deployments(run_id: str, records: dict) -> dict:
    if not isinstance(records, dict):
        raise ValueError("deployments must be keyed by checkpoint_id")
    for checkpoint_id, record in records.items():
        step = require_checkpoint(run_id, checkpoint_id)
        if not isinstance(record, dict) or record.get("checkpoint_id") != checkpoint_id:
            raise ValueError("deployment key and checkpoint_id must match")
        if isinstance(record.get("checkpoint_step"), bool) or record.get("checkpoint_step") != step:
            raise ValueError("deployment checkpoint_step does not match checkpoint_id")
        if record.get("state") not in DEPLOYMENT_STATES:
            raise ValueError("invalid checkpoint deployment state")
        generation(record)
    return records


def is_ready(run_id: str, record: dict) -> bool:
    """Only a completed, verified record grants serving authority."""
    checkpoint_id = record.get("checkpoint_id")
    parsed = parse_checkpoint_ref(checkpoint_id) if isinstance(checkpoint_id, str) else None
    verified_at = record.get("verified_at")
    serving_generation = record.get("deployment_generation")
    return bool(
        parsed is not None
        and parsed[0] == run_id
        and not isinstance(record.get("checkpoint_step"), bool)
        and parsed[1] == record.get("checkpoint_step")
        and record.get("state") == "ready"
        and isinstance(verified_at, (int, float))
        and not isinstance(verified_at, bool)
        and math.isfinite(verified_at)
        and verified_at > 0
        and isinstance(serving_generation, str)
        and bool(serving_generation.strip())
    )


def owns_attempt(current: dict, candidate: dict) -> bool:
    return (
        current.get("state") in BUSY_STATES
        and generation(current) == generation(candidate)
        and current.get("requested_at") == candidate.get("requested_at")
    )
