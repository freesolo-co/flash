"""Run attempt reservation and retry verification."""

from __future__ import annotations

import time

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.adapters.fused_experts import lora_target_parameters
from flash.core.spec import JobSpec
from flash.engine.support.verl_policy import _resolve_fsdp_generation
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


def _verified_opd_retry_state(run_id: str) -> tuple[int, str | None, int | None]:
    """Verify one locked opd retry snapshot: its attempt, resume revision, and checkpoint width.

    The width is the validated stamped rank count, or ``None`` when nothing is pinned because no
    mutation occurred. A pinned retry has to be allocated at exactly that count, as enforced by
    ``verify_opd_replacement_safe``.
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
        fsdp_generation = _resolve_fsdp_generation("opd", lora_target_parameters(spec.model))
    from flash.providers.artifacts.hf import verify_opd_replacement_safe

    verified = verify_opd_replacement_safe(
        hf_repo=hf_repo,
        run_id=run_id,
        seed=seed,
        next_attempt=next_attempt,
        contract_version=contract_version,
        phase=phase,
        expected_fsdp_generation=fsdp_generation,
    )
    resume_revision, checkpoint_world_size = verified if verified is not None else (None, None)
    return next_attempt, resume_revision, checkpoint_world_size


def _verified_opd_next_attempt(run_id: str) -> int:
    """Return just the verified next attempt, discarding the resume revision."""
    return _verified_opd_retry_state(run_id)[0]


def _reserve_attempt_record(
    run_id: str,
    *,
    minimum_attempt: int = 0,
    expected_next_attempt: int | None = None,
):
    """Durably reserve an attempt and monotonic fence before provider creation."""
    from flash.runner.lifecycle.deadlines import _derive_attempt_deadlines
    from flash.runner.lifecycle.protocol import AttemptRecord

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
        if status.state in state.TERMINAL_STATES:
            raise RuntimeError("cannot reserve an attempt for a terminal run")
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
            attempt_id = expected
        else:
            attempt_id = max(current, minimum)
        if attempt_id >= MAX_ATTEMPT_ID:
            raise RuntimeError("run attempt identity is exhausted")
        fence = raw.get(state._NEXT_FENCE_KEY)
        if isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
            raise RuntimeError("stored next fence identity is missing or invalid")
        reserved_at = time.time()
        grant, work, result, run = _derive_attempt_deadlines(
            raw,
            reserved_at=reserved_at,
        )
        record = AttemptRecord(
            attempt_id=attempt_id,
            fence=fence,
            state="reserved",
            reserved_at=reserved_at,
            grant_deadline_at=grant,
            work_deadline_at=work,
            result_deadline_at=result,
            run_deadline_at=run,
        )
        status.attempt = record.to_dict()
        status.progress = None
        status.resource = None
        status.result = None
        state._save_status_unlocked(
            status,
            _next_attempt=attempt_id + 1,
            _next_fence=fence + 1,
        )
        return record


def _reserve_attempt(
    run_id: str,
    *,
    minimum_attempt: int = 0,
    expected_next_attempt: int | None = None,
) -> int:
    """Compatibility-free internal convenience returning the reserved numeric attempt."""
    return _reserve_attempt_record(
        run_id,
        minimum_attempt=minimum_attempt,
        expected_next_attempt=expected_next_attempt,
    ).attempt_id


def _latest_reserved_attempt(run_id: str) -> int | None:
    """Return the newest durably reserved attempt, or none before any reservation."""
    try:
        raw = status_ops._load_status_json(run_id)
        next_attempt = _infer_next_attempt(raw)
    except Exception:
        return None
    return next_attempt - 1 if next_attempt > 0 else None
