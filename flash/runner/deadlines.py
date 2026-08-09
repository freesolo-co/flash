"""Run wall-clock deadline calculations."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import flash.runner as runner
from flash.core.spec import JobSpec
from flash.runner import RunStatus


def _require_valid_deadline(value: object) -> float:
    """Return a finite positive unix deadline or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0:
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    return deadline


def _profile_wall_armed_at(raw: dict) -> float | None:
    """Return when this profile's work budget started, or None if it has not started yet.

    Absent means no worker has spoken for this profile yet, which is the normal state while it
    queues. A stored value is validated like any other deadline input: a corrupt one fails closed
    rather than silently reverting to the submission basis and shortening the budget."""
    if runner._PROFILE_WALL_ARMED_AT_KEY not in raw:
        return None
    return runner._require_valid_deadline(raw[runner._PROFILE_WALL_ARMED_AT_KEY])


def _canonical_run_deadline(raw: dict) -> tuple[RunStatus, float]:
    status = runner._runstatus_from_json(raw)
    # max_wall_seconds is platform-managed and stripped from the public status.spec, so source the
    # run-global wall budget from the internal worker spec (the same value submit recorded).
    spec = runner._internal_spec_from_status(status)
    created_at = runner._require_valid_deadline(status.created_at)
    max_wall_seconds = runner._require_valid_deadline(spec.gpu.max_wall_seconds)
    if spec.workload_profile_kind:
        # a profile's wall budget bounds its work. before a worker speaks, the run is still waiting
        # on capacity, so it holds the queue allowance on top of its untouched work budget; once one
        # speaks, the work budget runs from that moment and the remaining queue allowance is
        # dropped. the basis is recomputed from persisted state (never from the wall clock), so this
        # stays a pure function of the record and _checked_stored_run_deadline still validates it.
        armed_at = runner._profile_wall_armed_at(raw)
        basis = (
            created_at + runner._WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS
            if armed_at is None
            else armed_at
        )
        return status, runner._require_valid_deadline(basis + max_wall_seconds)
    return status, runner._require_valid_deadline(created_at + max_wall_seconds)


def _checked_stored_run_deadline(stored: object, canonical: float) -> float:
    deadline = runner._require_valid_deadline(stored)
    if not math.isclose(deadline, canonical, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "persisted run wall deadline does not match canonical submission deadline; "
            "no further provisioning is allowed"
        )
    return deadline


def _load_run_deadline_at(run_id: str) -> float:
    """Return the persisted canonical submission-to-terminal deadline."""
    raw = runner._load_status_json(run_id)
    _status, canonical = runner._canonical_run_deadline(raw)
    if runner._RUN_DEADLINE_AT_KEY not in raw:
        raise RuntimeError(
            "persisted run wall deadline is missing; no further provisioning is allowed"
        )
    return runner._checked_stored_run_deadline(raw[runner._RUN_DEADLINE_AT_KEY], canonical)


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
    return max(0.0, runner._load_run_deadline_at(run_id) - float(current))


def _worker_deadline_at(run_id: str, spec: JobSpec, *, now: float | None = None) -> float:
    """Return the absolute deadline the worker may enforce for this launch.

    Unarmed, the work budget from launch is the authority. So an unarmed profile also carries the
    provisioning allowance -- otherwise docker+gpu waits, the image pull, pip install and the code
    fetch all come out of the work budget, and a slow-to-boot box self-terminates before it can emit
    the first heartbeat that would have armed the plane's own clock.
    """
    stored = runner._load_run_deadline_at(run_id)
    if not spec.workload_profile_kind:
        return stored
    current = time.time() if now is None else now
    work_budget_at = float(current) + float(runner._WORKLOAD_PROFILE_WALL_SECONDS)
    if runner._profile_wall_armed_at(runner._load_status_json(run_id)) is None:
        return work_budget_at + float(runner._WORKLOAD_PROFILE_PROVISION_ALLOWANCE_SECONDS)
    return min(stored, work_budget_at)


def _spec_with_remaining_wall(
    spec: JobSpec,
    *,
    require_provider_minimum: bool,
    now: float | None = None,
) -> JobSpec:
    """Copy a spec with only the run-global wall allowance still available."""
    remaining = runner._remaining_run_wall_seconds(spec.run_id, now=now)
    # exhaustion is judged on the real remaining allowance, before any profile substitution below.
    # a profile's grant replaces `remaining` outright, so deferring this check past that assignment
    # would make it unreachable for profiles and let a run provision after its own deadline had
    # passed -- and the first heartbeat would then arm a fresh work window from that moment,
    # turning the bounded queue allowance into an unbounded one.
    if remaining <= 0:
        raise RuntimeError("run wall deadline exhausted; no further provisioning is allowed")
    if spec.workload_profile_kind:
        # grant an unarmed profile its full work budget, not remaining submission allowance. the latter
        # still contains queue time and later truncates work after a long capacity wait; the run-global
        # deadline remains enforced by ``_worker_deadline_at``.
        remaining = float(runner._WORKLOAD_PROFILE_WALL_SECONDS)
    if require_provider_minimum and remaining < runner.MIN_PROVIDER_WALL_SECONDS:
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining; "
            "no further provisioning is allowed"
        )
    allowance = max(1, int(remaining))
    return replace(spec, gpu=replace(spec.gpu, max_wall_seconds=allowance))
