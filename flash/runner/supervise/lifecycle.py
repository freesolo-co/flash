"""run-execution machinery for submission, supervision, and cleanup."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

from flash.core.spec import JobSpec, gpu_count_of, require_matching_seed

# Floor so a streak of broken/busy GPUs doesn't kill a run that left retries enabled.
# max_retries==0 (single-shot) is always respected; floor only applies when retries are on.
INFRA_RETRY_FLOOR = 5
INFRA_RETRY_FAILURES = frozenset({"no_capacity", "poll_error", "job_preempted"})
RETRY_FAILURES = INFRA_RETRY_FAILURES | {"oom"}
_STAGED_ENVIRONMENT_RETRY_S = 5.0


def _run_job_background(spec: JobSpec, runtime_secrets: dict[str, str] | None = None) -> None:
    """run a supervised job without leaking a daemon-thread traceback."""
    import logging

    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import _update, get_status

    try:
        _run_job(spec, runtime_secrets=runtime_secrets) if runtime_secrets else _run_job(spec)
    except Exception as exc:
        detail = f"{type(exc).__name__}: background run failed"
        with contextlib.suppress(Exception):
            if get_status(spec.run_id).state not in TERMINAL_STATES:
                _update(spec.run_id, "failed", error=detail)
        logging.getLogger(__name__).warning(
            "background run %s ended in error: %s", spec.run_id, detail
        )


@dataclass
class _RetryBudget:
    infra_retries: int
    oom_retries: int
    cache_fallbacks: int
    infra_used: int = 0
    oom_used: int = 0
    cache_used: int = 0

    @property
    def max_attempts(self) -> int:
        return 1 + self.infra_retries + self.oom_retries + self.cache_fallbacks

    def infra_exhausted(self, *, cache_fallback_available: bool) -> bool:
        return self.infra_used >= self.infra_retries and not cache_fallback_available

    def can_retry(self, failure: str | None, *, cache_drop: bool) -> bool:
        if failure not in RETRY_FAILURES:
            return False
        if cache_drop:
            return self.cache_used < self.cache_fallbacks
        if failure == "oom":
            return self.oom_used < self.oom_retries
        return self.infra_used < self.infra_retries

    def record_retry(self, failure: str | None, *, cache_drop: bool) -> None:
        if cache_drop:
            self.cache_used += 1
            return
        if failure == "oom":
            self.oom_used += 1
        elif failure in INFRA_RETRY_FAILURES:
            self.infra_used += 1


def _run_job(spec: JobSpec, runtime_secrets: dict[str, str] | None = None) -> None:
    from flash.content.multimodal import preflight_validate_image_opd

    preflight_validate_image_opd(spec)

    from flash.runner.lifecycle.state import RUNS_DIR, TERMINAL_STATES
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.lifecycle import _run_job_inner
    from flash.runner.supervise.recovery import _gc_run_endpoints

    # Cancel can land before this thread starts; don't overwrite a terminal state with provisioning.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        while True:
            try:
                _run_job_inner(spec, log_path, runtime_secrets=runtime_secrets)
                break
            except Exception as exc:
                from flash.envs.loading.staged import StagedEnvironmentTransientError
                from flash.runner.lifecycle.deadlines import _load_run_deadline_at

                if not isinstance(exc, StagedEnvironmentTransientError):
                    raise
                if get_status(spec.run_id).state in TERMINAL_STATES:
                    return
                remaining = _load_run_deadline_at(spec.run_id) - time.time()
                if remaining <= 0:
                    _update(
                        spec.run_id,
                        "failed",
                        error="RuntimeError: run wall deadline exhausted during environment staging",
                    )
                    raise RuntimeError(
                        "run wall deadline exhausted during environment staging"
                    ) from exc
                with open(log_path, "a") as log:
                    print(
                        "environment staging is temporarily unavailable; deferring before retry",
                        file=log,
                        flush=True,
                    )
                time.sleep(min(_STAGED_ENVIRONMENT_RETRY_S, remaining))
    finally:
        # gc registered endpoints because undeleted endpoints count against the account-wide worker quota.
        # skip when the run is still non-terminal: another live supervisor then owns the durable handle,
        # and reaping here would tear down its still-active provider resources.
        if get_status(spec.run_id).state in TERMINAL_STATES:
            _gc_run_endpoints(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str, gpu_count: int = 0) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class and card count.

    The allocator may satisfy the run with a multi-card combination of a smaller class than the spec
    named, so the count it CHOSE has to land on the spec too: the worker sizes its rank count from
    gpu.count, and the provider payload rents gpu.count cards. Letting those diverge would either
    strand rented cards or launch more ranks than were rented.
    """
    count = gpu_count if gpu_count >= 1 else gpu_count_of(spec)
    if spec.gpu.type == gpu_type and gpu_count_of(spec) == count:
        return spec
    d = spec.to_internal_dict()
    acceptable = spec.gpu.acceptable_types
    gpu = {**d["gpu"], "type": gpu_type, "count": count}
    if acceptable:
        gpu["type_fallbacks"] = tuple(name for name in acceptable if name != gpu_type)
    else:
        gpu.pop("type_fallbacks", None)
    d["gpu"] = gpu
    # the auto marker records that gpu.count was omitted and survives shape resolution.
    return JobSpec.from_dict(d)


def _drop_weight_cache(spec: JobSpec) -> JobSpec:
    """Spec with the SHARED weight-cache volume removed for an unrestricted cross-region retry.

    Only drops the platform-managed shared cache (WEIGHT_CACHE_VOLUME_NAME); a custom per-org
    network_volume is the user's own choice and is preserved across retries.
    """
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    if getattr(spec.gpu, "network_volume", None) != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    d = spec.to_internal_dict()
    d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)


def _submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    attempt_start: int = 0,
) -> dict:
    """Run one seed with bounded auto-retry on infra-shaped failures.

    Retries resume from the latest HF checkpoint on a fresh host. Genuine worker errors fail fast.
    ``attempt_start`` offsets persisted identities without expanding this invocation's retry budget.
    """
    seed = require_matching_seed(spec, seed)
    from flash.runner.supervise.seed_submission import submit_seed_supervised

    return submit_seed_supervised(
        spec,
        seed,
        log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        attempt_start=attempt_start,
    )


def _terminal_failure_detail(exc: BaseException) -> str:
    """Render a run's persisted `error` from the exception that ended it.

    `RunStatus.error` is shown to the submitter, so an arbitrary exception's text is NOT safe to
    put there: it can carry internal storage paths, provider payloads, and upstream bodies. The
    default therefore keeps only the type. The exception is the managed-teacher configuration gate,
    whose messages are authored for exactly this audience -- redacting those made a missing
    plane-side credential indistinguishable from a bad spec, since both surfaced as a bare
    `RuntimeError: run failed`.
    """
    from flash.server.domain.teacher.broker import TeacherBrokerConfigurationError

    if isinstance(exc, TeacherBrokerConfigurationError):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: run failed"


def _run_job_inner(
    spec: JobSpec,
    log_path: str,
    runtime_secrets: dict[str, str] | None = None,
) -> None:
    from flash.runner.accounting.artifacts import stage_environment_package
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import _update, get_status
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _run_training

    try:
        # dev replaced the explicit code upload with managed source snapshots, so staging only has
        # to pin the environment package before the provider is allocated. the staged package rides
        # into the persisted snapshot at the per-attempt persist in `_submit_seed_supervised`, which
        # already runs after this with the fully planned spec -- persisting a second time here would
        # hash a half-planned spec no later integrity check can reproduce.
        deadline_at = _load_run_deadline_at(spec.run_id)
        spec = stage_environment_package(spec, deadline_at=deadline_at)
        with open(log_path, "a") as log:
            _run_training(
                spec,
                log,
                prior_cost=0.0,
                runtime_secrets=runtime_secrets,
            )
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        from flash.envs.loading.staged import StagedEnvironmentTransientError

        if isinstance(exc, StagedEnvironmentTransientError):
            raise
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=_terminal_failure_detail(exc))
        raise


def _run_training(
    spec: JobSpec,
    log,
    *,
    prior_cost: float,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    attempt_start: int = 0,
) -> None:
    """Train the run's single adapter under supervision; finalize the run.

    Shared by a fresh submit and post-restart recovery (the worker resumes from its last HF
    checkpoint on a fresh allocation). ``prior_cost`` carries spend already booked before a
    recovery so the total isn't under-reported. ``attempt_start`` preserves globally monotonic
    worker identities while each invocation keeps its own bounded retry budget."""
    from flash.runner.accounting.costs import _status_estimated_charge
    from flash.runner.lifecycle.state import TERMINAL_STATES, artifacts_dir
    from flash.runner.lifecycle.status import (
        _persist_metrics,
        _update,
        get_status,
        source_snapshot_from_status,
        validate_terminal_source_metrics,
    )
    from flash.runner.supervise.errors import _RunCancelled
    from flash.runner.supervise.lifecycle import _submit_seed_supervised

    if spec.algorithm == "opd":
        from flash.server.domain.teacher.broker import preflight_validate_managed_teacher

        preflight_validate_managed_teacher(spec)
    status = get_status(spec.run_id)
    if source_snapshot is None:
        source_snapshot = source_snapshot_from_status(status, required=True)
    # defense in depth against the recovery toctou (see attach_run): a run can be flipped into any
    # terminal state — not just `cancelled` — by a concurrent thread/process between the resume
    # decision and here. Bail before _update + the supervised submit so we never submit PAID GPU
    # work for an already-terminal run. _RunCancelled is the terminal signal; callers swallow it.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        raise _RunCancelled(f"run {spec.run_id} is already terminal; not submitting")
    # The pre-check above closes most of the window, but a concurrent flip can still land between
    # it and this transition. _update is a compare-and-set: it returns False when the run is already
    # terminal and leaves the state untouched. Gate the PAID supervised submit on that result so a
    # run cancelled in this last instant is never resumed onto a GPU.
    if not _update(spec.run_id, "running"):
        raise _RunCancelled(f"run {spec.run_id} went terminal before submit; not submitting")
    print(
        f"starting phase={spec.phase} model={spec.model} gpu={spec.gpu.type}",
        file=log,
        flush=True,
    )
    metrics = _submit_seed_supervised(
        spec,
        spec.seed,
        log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        attempt_start=attempt_start,
    )
    metrics, verified_attempt = validate_terminal_source_metrics(get_status(spec.run_id), metrics)
    # measured wall x $/hr is recorded in metrics.json for analytics, but is not what we charge.
    measured_cost = prior_cost + _persist_metrics(spec, metrics)
    # full planned work is charged at the submit-time quote, never measured wall. a worker that
    # finishes fewer optimizer steps pays the matching estimated-work fraction; legacy records
    # without a completed-step metric preserve the quote exactly.
    charge_usd = _status_estimated_charge(get_status(spec.run_id), spec, fallback=measured_cost)
    # A cancel can land while this thread writes metrics — after the supervised late-cancel check.
    # Re-read before the terminal "done" so a late worker success doesn't resurrect a cancelled run.
    with contextlib.suppress(FileNotFoundError):
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
    # Gate side effects on the CAS succeeding — a concurrent cancel rejects the `done` write.
    applied = _update(
        spec.run_id,
        "done",
        cost_usd=charge_usd,
        artifacts_dir=artifacts_dir(spec),
        source_verified_attempt=verified_attempt,
    )
    print(
        f"done: train_wall={metrics.get('wall_seconds')} measured={measured_cost:.4f} "
        f"charge_usd={charge_usd:.4f}",
        file=log,
        flush=True,
    )
    if applied:
        _charge_completed_run_by_id(spec.run_id, log)
        _register_checkpoints_best_effort(spec, log)


# re-exported at the bottom rather than imported at the top: `recovery` resolves the patched
# helpers back through this module, so a top import would be circular. `supervise.deploy` and the
# run-management tests both address these as attributes of THIS module, which is why they stay on
# it after the move.
from flash.runner.supervise.recovery import (  # noqa: E402,F401
    _RUNPOD_STATUS_PROBE_TIMEOUT_S,
    _adopt_completed_attempt,
    _apply_charge_with_state,
    _attempt_result_metrics,
    _candidate_usable_vram_gb,
    _canonical_provider_handle,
    _charge_completed_run_by_id,
    _oom_escalated,
    _projected_retry_class,
    _register_checkpoints_best_effort,
    _select_candidate,
    _shape_key,
    _strict_teardown_handle,
    _worker_provably_gone,
)
