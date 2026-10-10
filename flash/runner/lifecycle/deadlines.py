"""Run wall-clock deadline calculations."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from flash.core.spec import JobSpec
from flash.providers._lifecycle.net.deadline import require_finite_positive
from flash.runner.lifecycle import state
from flash.runner.lifecycle import status as status_ops
from flash.runner.lifecycle.state import RunStatus

MIN_PROVIDER_WALL_SECONDS = 60


def _require_valid_deadline(value: object) -> float:
    """Return a finite positive unix deadline or fail closed.

    The control plane's message names the consequence the provider-side one cannot: reaching here
    with an unusable deadline means no further provisioning may happen for this run.
    """
    return require_finite_positive(
        value, "run wall deadline is invalid; no further provisioning is allowed"
    )


def _canonical_run_deadline(raw: dict) -> tuple[RunStatus, float]:
    status = status_ops._runstatus_from_json(raw)
    # max_wall_seconds is platform-managed and stripped from the public status.spec, so source the
    # run-global wall budget from the internal worker spec (the same value submit recorded).
    spec = state._internal_spec_from_status(status)
    created_at = _require_valid_deadline(status.created_at)
    max_wall_seconds = _require_valid_deadline(spec.gpu.max_wall_seconds)
    return status, _require_valid_deadline(created_at + max_wall_seconds)


def _checked_stored_run_deadline(stored: object, canonical: float) -> float:
    deadline = _require_valid_deadline(stored)
    if not math.isclose(deadline, canonical, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "persisted run wall deadline does not match canonical submission deadline; "
            "no further provisioning is allowed"
        )
    return deadline


def _load_run_deadline_at(run_id: str) -> float:
    """Return the persisted canonical submission-to-terminal deadline."""
    raw = status_ops._load_status_json(run_id)
    _status, canonical = _canonical_run_deadline(raw)
    if state._RUN_DEADLINE_AT_KEY not in raw:
        raise RuntimeError(
            "persisted run wall deadline is missing; no further provisioning is allowed"
        )
    return _checked_stored_run_deadline(raw[state._RUN_DEADLINE_AT_KEY], canonical)


def _remaining_run_wall_seconds(run_id: str, *, now: float | None = None) -> float:
    """Return non-negative wall allowance remaining on the run-global deadline."""
    current = time.time() if now is None else now
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or not math.isfinite(current)
        or current <= 0
    ):
        raise ValueError("current clock is invalid")
    return max(0.0, _load_run_deadline_at(run_id) - float(current))


def _worker_deadline_at(run_id: str, spec: JobSpec, *, now: float | None = None) -> float:
    """Return the absolute deadline the worker may enforce for this launch.

    Unarmed, the work budget from launch is the authority. So an unarmed profile also carries the
    provisioning allowance -- otherwise docker+gpu waits, the image pull, pip install and the code
    fetch all come out of the work budget, and a slow-to-boot box self-terminates before it can emit
    the first heartbeat that would have armed the plane's own clock.
    """
    return _load_run_deadline_at(run_id)


def _spec_with_remaining_wall(
    spec: JobSpec,
    *,
    require_provider_minimum: bool,
    now: float | None = None,
) -> JobSpec:
    """Copy a spec with only the run-global wall allowance still available."""
    remaining = _remaining_run_wall_seconds(spec.run_id, now=now)
    if remaining <= 0:
        raise RuntimeError("run wall deadline exhausted; no further provisioning is allowed")
    if require_provider_minimum and remaining < MIN_PROVIDER_WALL_SECONDS:
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining; "
            "no further provisioning is allowed"
        )
    allowance = max(1, int(remaining))
    return replace(spec, gpu=replace(spec.gpu, max_wall_seconds=allowance))
