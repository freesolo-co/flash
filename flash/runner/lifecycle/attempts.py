"""Run attempt reservation and durable launch ownership."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.adapters.fused_experts import lora_target_parameters
from flash.core.spec import JobSpec
from flash.engine.support.verl_policy import _resolve_fsdp_generation
from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.runner.lifecycle import claim_lock, state
from flash.runner.lifecycle import status as status_ops
from flash.teacher.retry_contract import require_opd_retry_contract_version

_CLAIM_KEYS = frozenset(
    {"attempt", "retry_snapshot", "token", "resume_revision", "resume_world_size"}
)


@dataclass(frozen=True)
class LaunchReservationResult:
    claim: AttemptLaunchClaim | None
    retry_plan: object | None = None
    active: bool = False


@dataclass(frozen=True, init=False)
class AttemptLaunchClaim:
    attempt: int
    _retry_snapshot: dict
    token: str
    resume_revision: str | None
    resume_world_size: int | None

    def __init__(
        self,
        attempt: int,
        retry_snapshot: dict,
        token: str,
        resume_revision: str | None = None,
        resume_world_size: int | None = None,
    ) -> None:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("launch claim attempt is invalid")
        if not isinstance(retry_snapshot, dict):
            raise ValueError("launch claim retry snapshot is invalid")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("launch claim token is invalid")
        if resume_revision is not None and (
            not isinstance(resume_revision, str) or not resume_revision.strip()
        ):
            raise ValueError("launch claim resume revision is invalid")
        if resume_world_size is not None and (
            isinstance(resume_world_size, bool)
            or not isinstance(resume_world_size, int)
            or resume_world_size < 1
        ):
            raise ValueError("launch claim resume world size is invalid")
        if (resume_revision is None) != (resume_world_size is None):
            raise ValueError("launch claim resume revision and world size must be paired")
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "_retry_snapshot", copy.deepcopy(retry_snapshot))
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "resume_revision", resume_revision)
        object.__setattr__(self, "resume_world_size", resume_world_size)

    @property
    def retry_snapshot(self) -> dict:
        return copy.deepcopy(self._retry_snapshot)

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "retry_snapshot": self.retry_snapshot,
            "token": self.token,
            "resume_revision": self.resume_revision,
            "resume_world_size": self.resume_world_size,
        }

    @classmethod
    def from_dict(cls, raw: object) -> AttemptLaunchClaim:
        if not isinstance(raw, dict) or set(raw) != _CLAIM_KEYS:
            raise ValueError("persisted launch claim has an invalid shape")
        return cls(
            raw["attempt"],
            raw["retry_snapshot"],
            raw["token"],
            raw["resume_revision"],
            raw["resume_world_size"],
        )


def claim_is_live(run_id: str, claim: AttemptLaunchClaim) -> bool:
    """Return whether any process still holds the run's os-shared launch lease."""
    if claim_lock.owned_locally(run_id, claim.token):
        return True
    fd = claim_lock.try_acquire(run_id)
    if fd is None:
        return True
    claim_lock.close(fd)
    return False


def release_launch_claim_token(run_id: str, token: str) -> None:
    """Release a locally held os-shared launch lease by its durable token."""
    claim_lock.release(run_id, token)


def release_launch_claim(run_id: str, claim: AttemptLaunchClaim) -> None:
    """Release this process's os-shared launch lease after durable consumption."""
    release_launch_claim_token(run_id, claim.token)


def active_launch_claim_from_raw(raw: dict) -> AttemptLaunchClaim | None:
    value = raw.get(state._ACTIVE_LAUNCH_CLAIM_KEY)
    if value is None:
        return None
    try:
        return AttemptLaunchClaim.from_dict(value)
    except ValueError as exc:
        raise RuntimeError("persisted launch claim is invalid") from exc


def _heartbeat_attempt_is_current(hb: object, raw: dict) -> bool:
    """Return whether a heartbeat names the newest reserved attempt."""
    if not isinstance(hb, dict):
        return False
    try:
        next_attempt = status_ops.decode_next_attempt(raw)
    except RuntimeError:
        return False
    expected = next_attempt - 1 if next_attempt > 0 else 0
    return _attempt_int(hb.get("attempt")) == expected


def _verified_opd_retry_state(run_id: str) -> tuple[int, str | None, int | None]:
    """Verify one opd retry snapshot and its optional resumable checkpoint."""
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
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
        next_attempt = status_ops.decode_next_attempt(raw)
        hf_repo = spec.train.hf_repo
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


def _validate_opd_evidence(
    spec: JobSpec,
    raw: dict,
    attempt: int,
    resume_revision: str | None,
    resume_world_size: int | None,
) -> None:
    if spec.algorithm != "opd":
        if resume_revision is not None or resume_world_size is not None:
            raise RuntimeError("non-opd launch cannot carry opd resume evidence")
        return
    try:
        require_opd_retry_contract_version(raw.get(state._OPD_RETRY_CONTRACT_KEY))
    except ValueError as exc:
        raise RuntimeError(
            "opd retry contract is missing or invalid; replacement is blocked"
        ) from exc
    if (resume_revision is None) != (resume_world_size is None):
        raise RuntimeError("opd resume revision and world size must be paired")
    if attempt > 0 and resume_revision is not None and resume_world_size is None:
        raise RuntimeError("opd resume checkpoint width is missing")


def _validate_attempt_reservation_from_raw(spec: JobSpec, raw: dict, attempt: int) -> None:
    """Validate retry authorization for one lock-held reservation."""
    from flash.runner.supervise.retry_decision import require_retry_authorization_from_raw

    require_retry_authorization_from_raw(spec, raw, attempt)


def _claim_matches_raw(raw: dict, claim: AttemptLaunchClaim) -> bool:
    try:
        persisted = active_launch_claim_from_raw(raw)
    except RuntimeError:
        return False
    return persisted == claim


def reserve_attempt_launch(
    run_id: str,
    *,
    expected_remote: dict | None = None,
    expected_state: str | None = None,
    expected_next_attempt: int | None = None,
    expected_retry_snapshot: dict | None = None,
    transition_state: str | None = None,
    resume_revision: str | None = None,
    resume_world_size: int | None = None,
    recover_handleless: bool = False,
    provider_clear_confirmed: bool = False,
    expected_stale_claim: AttemptLaunchClaim | None = None,
) -> LaunchReservationResult:
    """Atomically authorize, reserve, and own one exact provider launch."""
    if expected_next_attempt is not None and (
        isinstance(expected_next_attempt, bool)
        or not isinstance(expected_next_attempt, int)
        or expected_next_attempt < 0
    ):
        raise RuntimeError("expected next attempt identity is invalid")
    if recover_handleless and not provider_clear_confirmed:
        return LaunchReservationResult(None)
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        RetryState,
        transition_failure,
    )

    report_status = None
    claim_fd = None
    try:
        with state._status_guard(run_id):
            raw = status_ops._load_status_json(run_id)
            status = status_ops._runstatus_from_json(raw)
            if status.state in state.TERMINAL_STATES:
                return LaunchReservationResult(None)
            if expected_state is not None and status.state != expected_state:
                return LaunchReservationResult(None)
            if status.remote != expected_remote:
                return LaunchReservationResult(None)
            current = status_ops.decode_next_attempt(raw)
            if expected_next_attempt is not None and current != expected_next_attempt:
                return LaunchReservationResult(None)
            snapshot = raw.get(state._RETRY_STATE_KEY)
            if not isinstance(snapshot, dict):
                raise RuntimeError("persisted retry state is missing or invalid")
            if expected_retry_snapshot is not None and snapshot != expected_retry_snapshot:
                return LaunchReservationResult(None)
            spec = state._internal_spec_from_status(status)
            existing = active_launch_claim_from_raw(raw)
            if existing is not None:
                existing_state = RetryState.from_snapshot(spec, existing.retry_snapshot)
                snapshot = existing_state.to_snapshot()
                if not recover_handleless or expected_stale_claim != existing:
                    return LaunchReservationResult(None, active=True)
                claim_fd = claim_lock.try_acquire(run_id)
                if claim_fd is None:
                    return LaunchReservationResult(None, active=True)
                if current - 1 != existing.attempt:
                    raise RuntimeError("stale launch claim no longer names the newest attempt")
                _validate_attempt_reservation_from_raw(spec, raw, existing.attempt)
                _validate_opd_evidence(
                    spec,
                    raw,
                    existing.attempt,
                    existing.resume_revision,
                    existing.resume_world_size,
                )
                claim = AttemptLaunchClaim(
                    existing.attempt,
                    snapshot,
                    uuid.uuid4().hex,
                    existing.resume_revision,
                    existing.resume_world_size,
                )
            else:
                if expected_stale_claim is not None:
                    return LaunchReservationResult(None, active=True)
                retry_state = RetryState.from_snapshot(spec, snapshot)
                snapshot = retry_state.to_snapshot()
                if recover_handleless and current:
                    plan = retry_state.persisted_plan(current - 1)
                    if plan is None:
                        retry_state, plan = transition_failure(
                            retry_state,
                            FailureObservation.create(
                                "poll_error",
                                chosen=None,
                                candidates=None,
                                managed_cache_mounted=False,
                            ),
                            attempt=current - 1,
                        )
                        snapshot = retry_state.to_snapshot()
                    if not plan.retry:
                        state._save_status_unlocked(status, _retry_state=snapshot)
                        return LaunchReservationResult(None, retry_plan=plan)
                reservation_raw = {**raw, state._RETRY_STATE_KEY: snapshot}
                _validate_attempt_reservation_from_raw(spec, reservation_raw, current)
                _validate_opd_evidence(spec, raw, current, resume_revision, resume_world_size)
                if current >= MAX_ATTEMPT_ID:
                    raise RuntimeError("run attempt identity is exhausted")
                claim_fd = claim_lock.try_acquire(run_id)
                if claim_fd is None:
                    return LaunchReservationResult(None, active=True)
                claim = AttemptLaunchClaim(
                    current,
                    snapshot,
                    uuid.uuid4().hex,
                    resume_revision,
                    resume_world_size,
                )
                current += 1
            if expected_remote is not None:
                status.remote = None
            if transition_state is not None:
                status.state = transition_state
            status.updated_at = time.time()
            state._save_status_unlocked(
                status,
                _next_attempt=current,
                _retry_state=snapshot,
                _active_launch_claim=claim.to_dict(),
            )
            report_status = status
        claim_lock.register(run_id, claim.token, claim_fd)
        claim_fd = None
        try:
            if report_status is not None:
                from flash.runner.lifecycle.reporting import _report_status

                _report_status(report_status)
            return LaunchReservationResult(claim)
        except Exception:
            claim_lock.release(run_id, claim.token)
            raise
    finally:
        if claim_fd is not None:
            claim_lock.close(claim_fd)


def reserve_handleless_recovery_launch(
    run_id: str,
    *,
    expected_state: str,
    provider_clear_confirmed: bool,
    expected_stale_claim: AttemptLaunchClaim | None,
    resume_revision: str | None = None,
    resume_world_size: int | None = None,
) -> LaunchReservationResult:
    """Reserve one provider-clear recovery through the shared ownership primitive."""
    return reserve_attempt_launch(
        run_id,
        expected_state=expected_state,
        transition_state="provisioning",
        resume_revision=resume_revision,
        resume_world_size=resume_world_size,
        recover_handleless=True,
        provider_clear_confirmed=provider_clear_confirmed,
        expected_stale_claim=expected_stale_claim,
    )


def reserve_verified_attempt_launch(
    run_id: str,
    *,
    expected_remote: dict | None = None,
    expected_state: str | None = None,
    expected_next_attempt: int | None = None,
    expected_retry_snapshot: dict | None = None,
    transition_state: str | None = None,
) -> AttemptLaunchClaim | None:
    """Reserve one launch after verifying any required opd resume evidence."""
    status = status_ops.get_status(run_id)
    spec = state._internal_spec_from_status(status)
    if spec.algorithm == "opd":
        verified_next, revision, world_size = _verified_opd_retry_state(run_id)
        if expected_next_attempt is not None and verified_next != expected_next_attempt:
            return None
        expected_next = verified_next
    else:
        expected_next, revision, world_size = expected_next_attempt, None, None
    return reserve_attempt_launch(
        run_id,
        expected_remote=expected_remote,
        expected_state=expected_state,
        expected_next_attempt=expected_next,
        expected_retry_snapshot=expected_retry_snapshot,
        transition_state=transition_state,
        resume_revision=revision,
        resume_world_size=world_size,
    ).claim


def require_attempt_launch_current(run_id: str, spec: JobSpec, claim: AttemptLaunchClaim) -> None:
    """Fence provider launch to the exact durable ownership token."""
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        if raw.get("state") in state.TERMINAL_STATES:
            raise RuntimeError("run became terminal before provider launch")
        if raw.get("remote") is not None:
            raise RuntimeError("run already has a durable provider handle")
        if not _claim_matches_raw(raw, claim):
            raise RuntimeError("provider launch claim changed before launch")
        if status_ops.decode_next_attempt(raw) - 1 != claim.attempt:
            raise RuntimeError("provider launch claim is no longer the newest attempt")
        _validate_attempt_reservation_from_raw(spec, raw, claim.attempt)
        from flash.runner.supervise.retry_decision import RetryState

        retry_state = RetryState.from_snapshot(spec, claim.retry_snapshot)
        if retry_state.persisted_plan(claim.attempt) is not None:
            raise RuntimeError("reserved attempt already has an immutable retry decision")


def attempt_claim_is_current(run_id: str, claim: AttemptLaunchClaim) -> bool:
    """Return whether the run is still owned by this claim or its persisted handle."""
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        if raw.get("state") in state.TERMINAL_STATES:
            return False
        active = raw.get(state._ACTIVE_LAUNCH_CLAIM_KEY)
        if active is not None:
            return _claim_matches_raw(raw, claim)
        remote = raw.get("remote")
        if isinstance(remote, dict):
            return bool(
                remote.get("attempt") == claim.attempt
                and remote.get("launch_claim_token") == claim.token
            )
        if status_ops.decode_next_attempt(raw) - 1 != claim.attempt:
            return False
        from flash.runner.supervise.retry_decision import RetryState

        retry_state = RetryState.from_snapshot(
            spec=state._internal_spec_from_status(status_ops._runstatus_from_json(raw)),
            raw=raw.get(state._RETRY_STATE_KEY),
        )
        decision = retry_state.last_decision
        return decision is not None and decision.attempt == claim.attempt


def persist_claimed_remote(
    run_id: str,
    claim: AttemptLaunchClaim,
    persisted_remote: dict,
) -> bool:
    """Persist a provider handle only for the exact active launch token and consume it."""
    report_status = None
    try:
        with state._status_guard(run_id):
            raw = status_ops._load_status_json(run_id)
            status = status_ops._runstatus_from_json(raw)
            if status.state in state.TERMINAL_STATES or status.remote is not None:
                return False
            if not _claim_matches_raw(raw, claim):
                return False
            if status_ops.decode_next_attempt(raw) - 1 != claim.attempt:
                return False
            remote = dict(persisted_remote)
            remote["launch_claim_token"] = claim.token
            status.state = "running"
            status.remote = remote
            status.updated_at = time.time()
            state._save_status_unlocked(status, _active_launch_claim=None)
            report_status = status
        if report_status is not None:
            from flash.runner.lifecycle.reporting import _report_status

            _report_status(report_status)
        return True
    finally:
        release_launch_claim(run_id, claim)


def latest_reserved_attempt(run_id: str) -> int | None:
    """Return the newest durably reserved attempt, or none before reservation."""
    try:
        next_attempt = status_ops.decode_next_attempt(status_ops._load_status_json(run_id))
    except Exception:
        return None
    return next_attempt - 1 if next_attempt > 0 else None
