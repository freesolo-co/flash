"""Run attempt reservation and durable launch ownership."""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.adapters.fused_experts import lora_target_parameters
from flash.core.spec import JobSpec
from flash.engine.support.verl_policy import _resolve_fsdp_generation
from flash.runner.lifecycle import claim_lock, state
from flash.runner.lifecycle import status as status_ops
from flash.teacher.retry_contract import require_opd_retry_contract_version

_CLAIM_KEYS = frozenset({"attempt", "token", "resume_revision", "resume_world_size"})


@dataclass(frozen=True)
class LaunchReservationResult:
    claim: AttemptLaunchClaim | None
    retry_plan: object | None = None
    active: bool = False


@dataclass(frozen=True)
class AttemptLaunchClaim:
    attempt: int
    token: str
    resume_revision: str | None = None
    resume_world_size: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise ValueError("launch claim attempt is invalid")
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("launch claim token is invalid")
        if self.resume_revision is not None and (
            not isinstance(self.resume_revision, str) or not self.resume_revision.strip()
        ):
            raise ValueError("launch claim resume revision is invalid")
        if self.resume_world_size is not None and (
            isinstance(self.resume_world_size, bool)
            or not isinstance(self.resume_world_size, int)
            or self.resume_world_size < 1
        ):
            raise ValueError("launch claim resume world size is invalid")
        if (self.resume_revision is None) != (self.resume_world_size is None):
            raise ValueError("launch claim resume revision and world size must be paired")

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
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


def release_launch_claim(run_id: str, claim: AttemptLaunchClaim) -> None:
    """Release this process's os-shared launch lease after durable consumption."""
    claim_lock.release(run_id, claim.token)


def active_launch_claim_from_raw(raw: dict) -> AttemptLaunchClaim | None:
    value = raw.get(state._ACTIVE_LAUNCH_CLAIM_KEY)
    if value is None:
        return None
    try:
        return AttemptLaunchClaim.from_dict(value)
    except ValueError as exc:
        raise RuntimeError("persisted launch claim is invalid") from exc


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


def _owns_attempt(
    raw: dict,
    *,
    attempt: int,
    token: str,
    expected_remote: dict | None,
) -> bool:
    """Match exact ownership through either the active claim or durable handle."""
    if expected_remote is not None:
        return bool(
            raw.get("remote") == expected_remote
            and expected_remote.get("attempt") == attempt
            and expected_remote.get("launch_claim_token") == token
            and raw.get(state._ACTIVE_LAUNCH_CLAIM_KEY) is None
        )
    if raw.get("remote") is not None:
        return False
    try:
        persisted = active_launch_claim_from_raw(raw)
    except RuntimeError:
        return False
    return bool(persisted and persisted.attempt == attempt and persisted.token == token)


def _claim_matches_raw(raw: dict, claim: AttemptLaunchClaim) -> bool:
    return _owns_attempt(
        raw,
        attempt=claim.attempt,
        token=claim.token,
        expected_remote=None,
    )


def _clear_exact_persisted_claim_unlocked(run_id: str, claim: AttemptLaunchClaim) -> None:
    """Remove one exact claim a failed reservation write may still have persisted.

    The caller already holds `_status_guard`, which is a plain non-reentrant lock, so this must not
    re-enter it.
    """
    raw = status_ops._load_status_json(run_id)
    if not _claim_matches_raw(raw, claim):
        return
    state._save_status_unlocked(
        status_ops._runstatus_from_json(raw),
        _active_launch_claim=None,
    )


def _persist_reservation_unlocked(
    run_id: str,
    status,
    *,
    claim: AttemptLaunchClaim,
    next_attempt: int,
    snapshot,
    clear_remote: bool,
    clear_teardown_marker: bool,
    transition_state: str | None,
) -> None:
    """Write one reservation, rolling the claim back if the write raises after landing.

    The caller holds `_status_guard`, which is not reentrant, so every write here is unlocked.
    """
    if clear_remote:
        status.remote = None
        # a confirmed-teardown reservation consumes `cleanup_confirmed_remote`, so the marker has
        # to go with it. leaving it makes a crash before the replacement handle look like
        # confirmed-handle recovery: attach reattaches the old attempt, its reservation loses to
        # the advanced counter, and the run sits in `provisioning` with no handleless pass
        # scheduled. accounting keeps its own `realized_cost_remote`.
        if clear_teardown_marker:
            status.cleanup_confirmed_remote = None
    if transition_state is not None:
        status.state = transition_state
    status.updated_at = time.time()
    try:
        state._save_status_unlocked(
            status,
            _next_attempt=next_attempt,
            _retry_state=snapshot,
            _active_launch_claim=claim.to_dict(),
        )
    except Exception:
        # `os.replace` runs before the directory `fsync`, so a raise here can still leave the claim
        # persisted while the caller never receives it. attach then sees a plain `False` against the
        # old remote and schedules no reconciler, so the run is left handleless behind an unlocked
        # stale claim. clear only this exact claim.
        with contextlib.suppress(Exception):
            _clear_exact_persisted_claim_unlocked(run_id, claim)
        raise


def _reserve_attempt_launch(
    run_id: str,
    *,
    expected_remote: dict | None = None,
    expected_state: str | None = None,
    expected_next_attempt: int | None = None,
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
        require_retry_authorization_from_raw,
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
            remote_matches = status.remote == expected_remote
            confirmed_matches = (
                expected_remote is not None
                and status.remote is None
                and status.cleanup_confirmed_remote == expected_remote
            )
            if not remote_matches and not confirmed_matches:
                return LaunchReservationResult(None)
            current = status_ops.decode_next_attempt(raw)
            if expected_next_attempt is not None and current != expected_next_attempt:
                return LaunchReservationResult(None)
            snapshot = raw.get(state._RETRY_STATE_KEY)
            if not isinstance(snapshot, dict):
                raise RuntimeError("persisted retry state is missing or invalid")
            spec = state._internal_spec_from_status(status)
            existing = active_launch_claim_from_raw(raw)
            if existing is not None:
                if not recover_handleless or expected_stale_claim != existing:
                    return LaunchReservationResult(None, active=True)
                claim_fd = claim_lock.try_acquire(run_id)
                if claim_fd is None:
                    return LaunchReservationResult(None, active=True)
                if current - 1 != existing.attempt:
                    raise RuntimeError("stale launch claim no longer names the newest attempt")
                require_retry_authorization_from_raw(spec, raw, existing.attempt)
                # the claim pinned its resume evidence before its worker launched, and that worker
                # may have mutated the checkpoint before it was lost. take the caller's freshly
                # verified evidence, which covers every attempt the counter names.
                _validate_opd_evidence(
                    spec,
                    raw,
                    existing.attempt,
                    resume_revision,
                    resume_world_size,
                )
                claim = AttemptLaunchClaim(
                    existing.attempt,
                    uuid.uuid4().hex,
                    resume_revision,
                    resume_world_size,
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
                            FailureObservation("poll_error"),
                            attempt=current - 1,
                        )
                        snapshot = retry_state.to_snapshot()
                    if not plan.retry:
                        state._save_status_unlocked(status, _retry_state=snapshot)
                        return LaunchReservationResult(None, retry_plan=plan)
                reservation_raw = {**raw, state._RETRY_STATE_KEY: snapshot}
                require_retry_authorization_from_raw(spec, reservation_raw, current)
                _validate_opd_evidence(spec, raw, current, resume_revision, resume_world_size)
                if current >= MAX_ATTEMPT_ID:
                    raise RuntimeError("run attempt identity is exhausted")
                claim_fd = claim_lock.try_acquire(run_id)
                if claim_fd is None:
                    return LaunchReservationResult(None, active=True)
                claim = AttemptLaunchClaim(
                    current,
                    uuid.uuid4().hex,
                    resume_revision,
                    resume_world_size,
                )
                current += 1
            _persist_reservation_unlocked(
                run_id,
                status,
                claim=claim,
                next_attempt=current,
                snapshot=snapshot,
                clear_remote=expected_remote is not None,
                clear_teardown_marker=confirmed_matches,
                transition_state=transition_state,
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
    expected_next_attempt: int | None = None,
    resume_revision: str | None = None,
    resume_world_size: int | None = None,
) -> LaunchReservationResult:
    """Reserve one provider-clear recovery through the shared ownership primitive.

    `expected_next_attempt` is the counter the caller's resume evidence was verified against. opd
    verification reaches hugging face outside the status lock, so a concurrent observer can reserve
    and settle an attempt while it runs. Passing the verified counter rejects evidence that predates
    the intervening attempt's mutation marker instead of resuming from a checkpoint that excludes it.
    """
    return _reserve_attempt_launch(
        run_id,
        expected_state=expected_state,
        expected_next_attempt=expected_next_attempt,
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
    return _reserve_attempt_launch(
        run_id,
        expected_remote=expected_remote,
        expected_state=expected_state,
        expected_next_attempt=expected_next,
        transition_state=transition_state,
        resume_revision=revision,
        resume_world_size=world_size,
    ).claim


def require_attempt_launch_current(run_id: str, spec: JobSpec, claim: AttemptLaunchClaim):
    """Fence provider launch and return its current retry state."""
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
        from flash.runner.supervise.retry_decision import require_retry_authorization_from_raw

        retry_state = require_retry_authorization_from_raw(spec, raw, claim.attempt)
        if retry_state.persisted_plan(claim.attempt) is not None:
            raise RuntimeError("reserved attempt already has an immutable retry decision")
        return retry_state


def attempt_is_this_callers_to_fail(run_id: str, claim: AttemptLaunchClaim) -> bool:
    """Return whether this claim's caller still owes the run a terminal outcome.

    Exact ownership is the common case, but it is not the only one. A caller that already settled
    its attempt -- persisted a terminal retry decision, then confirmed teardown -- holds neither the
    active claim nor the remote any more, yet the run is still nonterminal and nobody else will fail
    it. That settled shape must stay this caller's to finish, or an unwound attempt leaves the run
    in `provisioning` with no owner and a later handleless pass relaunches decided work.

    Releasing ownership is not itself the duty. Three settled shapes look identical from the outside
    -- no remote, no active claim -- and only one of them is this caller's to fail:

    - a persisted `retry=False` decision for this attempt is terminal, and nobody else will write it;
    - a persisted `retry=True` decision is authorized replacement work, which handleless recovery
      relaunches. Failing it here destroys a retry the policy already granted;
    - no decision at all means the attempt never reached one. That is a success that settled its
      remote, or a pre-decision unwind, and adoption owns it.

    A newer attempt is the further boundary: once the counter has moved past this claim, ownership
    belongs to whoever reserved it, and this caller must not write a terminal state over them.
    """
    from flash.runner.supervise.retry_decision import RetryState

    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        if raw.get("state") in state.TERMINAL_STATES:
            return False
        remote = raw.get("remote")
        if _owns_attempt(
            raw,
            attempt=claim.attempt,
            token=claim.token,
            expected_remote=remote if isinstance(remote, dict) else None,
        ):
            return True
        # settled-and-unwound: nothing is owned because this caller released it. only a terminal
        # decision for this exact attempt, with no newer attempt reserved, is still this caller's.
        if remote is not None or raw.get(state._ACTIVE_LAUNCH_CLAIM_KEY) is not None:
            return False
        try:
            if status_ops.decode_next_attempt(raw) - 1 != claim.attempt:
                return False
            spec = state._internal_spec_from_status(status_ops._runstatus_from_json(raw))
            retry_state = RetryState.from_snapshot(spec, raw[state._RETRY_STATE_KEY])
        except (KeyError, ValueError, RuntimeError):
            return False
        plan = retry_state.persisted_plan(claim.attempt)
        return plan is not None and plan.retry is False


def decide_attempt_failure(
    run_id: str,
    *,
    claim_token: str,
    expected_remote: dict | None,
    observation,
    attempt: int,
):
    """Atomically persist one immutable failure decision for the exact attempt owner."""
    from flash.runner.supervise.retry_decision import (
        FailureObservation,
        require_retry_authorization_from_raw,
        transition_failure,
    )

    if not isinstance(observation, FailureObservation):
        raise TypeError("failure observation is invalid")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("retry decision attempt is invalid")
    persisted = False
    released = False
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        if (
            raw.get("state") in state.TERMINAL_STATES
            or status_ops.decode_next_attempt(raw) - 1 != attempt
            or not _owns_attempt(
                raw,
                attempt=attempt,
                token=claim_token,
                expected_remote=expected_remote,
            )
        ):
            # ownership is gone, so this caller can never persist a decision for the attempt and
            # drops its claim reference regardless. release the lease here or nothing downstream
            # can: a retained flock keeps claim_is_live true and blocks handleless recovery.
            released = True
            plan = None
        else:
            status = status_ops._runstatus_from_json(raw)
            spec = state._internal_spec_from_status(status)
            retry_state = require_retry_authorization_from_raw(spec, raw, attempt)
            plan = retry_state.persisted_plan(attempt)
            if plan is None:
                retry_state, plan = transition_failure(retry_state, observation, attempt=attempt)
                state._save_status_unlocked(
                    status,
                    _retry_state=retry_state.to_snapshot(),
                    _active_launch_claim=None,
                )
                persisted = True
    if persisted or released:
        claim_lock.release(run_id, claim_token)
    return plan


def consume_active_launch_claim(run_id: str, claim: AttemptLaunchClaim) -> bool:
    """Clear one exact unconsumed claim, then release its os-shared launch lease."""
    try:
        with state._status_guard(run_id):
            raw = status_ops._load_status_json(run_id)
            if not _claim_matches_raw(raw, claim):
                return False
            state._save_status_unlocked(
                status_ops._runstatus_from_json(raw),
                _active_launch_claim=None,
            )
        return True
    finally:
        release_launch_claim(run_id, claim)


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
            # a fresh live handle supersedes the torn-down handle retained only for delayed
            # reconciliation; leaving it would price this attempt against the previous resource.
            status.realized_cost_remote = None
            # first-start evidence is stamped once, so a later attempt never rewrites it.
            if status.lifecycle_started_attempt is None:
                status.lifecycle_started_attempt = claim.attempt
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
