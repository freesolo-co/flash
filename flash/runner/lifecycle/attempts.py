"""Run attempt reservation and retry verification."""

from __future__ import annotations

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.core.spec import JobSpec
from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.runner.lifecycle import state
from flash.runner.lifecycle import status as status_ops
from flash.teacher.retry_contract import require_opd_retry_contract_version


def _infer_next_attempt(raw: dict) -> int:
    if state._NEXT_ATTEMPT_KEY not in raw:
        raise RuntimeError("stored next attempt identity is missing")
    stored = raw[state._NEXT_ATTEMPT_KEY]
    if _attempt_int(stored) is None:
        raise RuntimeError("stored next attempt identity is invalid")
    return stored


def _heartbeat_attempt_is_current(hb: object, raw: dict) -> bool:
    """True when a heartbeat carries the attempt identity this run most recently reserved.

    This is the plane-side counterpart of ``_heartbeat_matches_attempt``, which runs provider-side
    where the launch timestamp is in hand; here the equivalent identity
    is the reserved attempt, which the worker stamps on every heartbeat and ``_save_status`` already
    persists as ``next_attempt`` (the NEXT id to hand out, so the live attempt is one below it --
    same arithmetic as ``_latest_reserved_attempt``, computed from the caller's already-loaded
    record because this runs inside the status guard and must not re-read it).
    """
    if not isinstance(hb, dict):
        return False
    try:
        next_attempt = _attempt_int(_infer_next_attempt(raw))
    except RuntimeError:
        return False
    if next_attempt is None:
        return False
    # `_reserve_attempt` runs before the provider launch (lifecycle.py), so a live worker's
    # heartbeat always sits one below the stored counter. zero means nothing has been reserved yet;
    # accept attempt 0 there rather than rejecting, because the launch path writes the counter and
    # the worker's first heartbeat can be read back in either order, and refusing to arm would hand
    # the run a budget measured from a moment before it started working.
    expected = next_attempt - 1 if next_attempt > 0 else 0
    return _attempt_int(hb.get("attempt")) == expected


def _verified_opd_retry_state(run_id: str) -> tuple[int, str | None, int | None]:
    """Verify one locked opd retry snapshot: its attempt, resume revision, and checkpoint width.

    The width is the rank count the pinned checkpoint's fsdp shards were written at, or ``None``
    when nothing is pinned (no mutation) or the shards named no single width. A pinned retry has to
    be allocated at exactly that count -- see ``verify_opd_replacement_safe``.
    """
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        # hf_repo is platform-managed and stripped from the public status.spec; the opd replacement
        # locates its resume checkpoint by hf_repo, so source the complete internal worker spec.
        spec = state._internal_spec_from_status(status)
        if spec.algorithm != "opd":
            raise RuntimeError("opd retry verification requires an opd run")
        try:
            contract_version = require_opd_retry_contract_version(
                raw.get(state._OPD_RETRY_CONTRACT_KEY)
            )
        except ValueError as exc:
            raise RuntimeError(
                "opd retry contract is missing or invalid; replacement is blocked"
            ) from exc
        next_attempt = _infer_next_attempt(raw)
        hf_repo = spec.train.hf_repo
        # phase is the hf-prefix component the worker uploads under ({phase}/{run_id}/...), so it
        # locates both the markers and any full-state resume checkpoint the replacement can continue
        # from.
        phase = spec.phase
        seed = spec.seed
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    verified = verify_opd_replacement_safe(
        hf_repo=hf_repo,
        run_id=run_id,
        seed=seed,
        next_attempt=next_attempt,
        contract_version=contract_version,
        phase=phase,
    )
    resume_revision, checkpoint_world_size = verified if verified is not None else (None, None)
    return next_attempt, resume_revision, checkpoint_world_size


def _verified_opd_next_attempt(run_id: str) -> int:
    """Return just the verified next attempt, discarding the resume revision."""
    return _verified_opd_retry_state(run_id)[0]


def _reserve_attempt(
    run_id: str,
    *,
    minimum_attempt: int = 0,
    expected_next_attempt: int | None = None,
) -> int:
    """Durably consume one run-global attempt identity before provider creation."""
    minimum = _attempt_int(minimum_attempt)
    if minimum is None:
        raise RuntimeError("minimum attempt identity is invalid")
    expected = None
    if expected_next_attempt is not None:
        expected = _attempt_int(expected_next_attempt)
        if expected is None:
            raise RuntimeError("expected next attempt identity is invalid")
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        current = _infer_next_attempt(raw)
        if expected is not None and current != expected:
            raise RuntimeError("stored next attempt identity changed after retry verification")
        spec = JobSpec.from_dict(status.spec)
        if spec.algorithm == "opd":
            try:
                require_opd_retry_contract_version(raw.get(state._OPD_RETRY_CONTRACT_KEY))
            except ValueError as exc:
                raise RuntimeError(
                    "opd retry contract is missing or invalid; replacement is blocked"
                ) from exc
            if expected is None:
                raise RuntimeError("opd attempt reservation requires verified retry evidence")
            if minimum > expected:
                raise RuntimeError("minimum opd attempt exceeds the verified retry snapshot")
            attempt = expected
        else:
            attempt = max(current, minimum)
        if attempt >= MAX_ATTEMPT_ID:
            raise RuntimeError("run attempt identity is exhausted")
        state._save_status_unlocked(status, _next_attempt=attempt + 1)
        return attempt


def _latest_reserved_attempt(run_id: str) -> int | None:
    """Return the newest durably reserved attempt, or none before any reservation."""
    try:
        raw = status_ops._load_status_json(run_id)
        next_attempt = _infer_next_attempt(raw)
    except Exception:
        return None
    return next_attempt - 1 if next_attempt > 0 else None
