"""Run-execution machinery: the submit -> supervised training job -> GC flow.

Sibling helpers are imported function-locally to avoid the flash.runner.__init__ import cycle
and to keep monkeypatches reachable (``monkeypatch.setattr(runner, ...)`` vs a static copy).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

from flash.core.spec import JobSpec, gpu_count_of, require_matching_seed
from flash.providers._lifecycle.deadline import deadline_kwargs

# Floor so a streak of broken/busy GPUs doesn't kill a run that left retries enabled.
# max_retries==0 (single-shot) is always respected; floor only applies when retries are on.
INFRA_RETRY_FLOOR = 5
INFRA_RETRY_FAILURES = frozenset({"stalled", "no_capacity", "poll_error", "job_preempted"})
RETRY_FAILURES = INFRA_RETRY_FAILURES | {"oom"}
_STAGED_ENVIRONMENT_RETRY_S = 5.0


class _SelectedQuoteUnaffordable(RuntimeError):
    """The selected live candidate costs more than the owning organization can afford."""


def _recheck_selected_quote_affordability(status, selected_quote: float, log) -> None:
    """Recheck only a live quote that increases the amount accepted before allocation."""
    accepted_quote = float(getattr(status, "estimated_cost_usd", 0.0) or 0.0)
    if selected_quote <= accepted_quote:
        return
    context = getattr(status, "billing_context", None)
    if not isinstance(context, dict):
        return
    org_id = str(context.get("org_id") or "").strip()
    if not org_id:
        return

    from flash.server.platform.internal_client import internal_key

    key = internal_key()
    if not key:
        return
    try:
        from flash.server.billing.charges import precheck_training_run

        precheck_training_run(internal_key=key, org_id=org_id, estimate_usd=selected_quote)
    except Exception as exc:
        from flash.server.billing.charges import BillingError

        if isinstance(exc, BillingError) and exc.status_code == 402:
            raise _SelectedQuoteUnaffordable(
                "selected live GPU quote exceeds the organization's available training balance"
            ) from exc
        print(
            f"budget recheck skipped for selected quote (billing service error: {type(exc).__name__})",
            file=log,
            flush=True,
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

    # Lazy import: dry-run / unit tests never construct a Flash endpoint.
    from flash.providers._lifecycle.worker import upload_code
    from flash.runner import (
        RUNS_DIR,
        TERMINAL_STATES,
        _gc_run_endpoints,
        _run_job_inner,
        _update,
        get_status,
    )

    # Cancel can land before this thread starts; don't overwrite a terminal state with provisioning.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        while True:
            try:
                _run_job_inner(spec, log_path, upload_code, runtime_secrets=runtime_secrets)
                break
            except Exception as exc:
                from flash.envs.staged import StagedEnvironmentTransientError
                from flash.runner import _load_run_deadline_at

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
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

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
    code_prefix: str | None = None,
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
        code_prefix=code_prefix,
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
    from flash.server.domain.teacher_broker import TeacherBrokerConfigurationError

    if isinstance(exc, TeacherBrokerConfigurationError):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: run failed"


def _run_job_inner(
    spec: JobSpec,
    log_path: str,
    upload_code,
    runtime_secrets: dict[str, str] | None = None,
) -> None:
    from flash.runner import (
        _load_run_deadline_at,
        _persist_effective_worker_spec,
        _run_training,
        _RunCancelled,
        _update,
        flash_code_prefix,
        get_status,
        stage_environment_package,
    )

    try:
        code_prefix = flash_code_prefix()
        deadline_at = _load_run_deadline_at(spec.run_id)
        upload_code(
            spec.train.hf_repo,
            code_prefix=code_prefix,
            **deadline_kwargs(upload_code, deadline_at),
        )
        spec = stage_environment_package(spec, deadline_at=deadline_at)
        if not _persist_effective_worker_spec(spec):
            raise _RunCancelled(f"run {spec.run_id} went terminal before environment staging")
        with open(log_path, "a") as log:
            _run_training(
                spec,
                log,
                prior_cost=0.0,
                runtime_secrets=runtime_secrets,
                code_prefix=code_prefix,
            )
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        from flash.envs.staged import StagedEnvironmentTransientError

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
    code_prefix: str | None = None,
    attempt_start: int = 0,
) -> None:
    """Train the run's single adapter under supervision; finalize the run.

    Shared by a fresh submit and post-restart recovery (the worker resumes from its last HF
    checkpoint on a fresh allocation). ``prior_cost`` carries spend already booked before a
    recovery so the total isn't under-reported. ``attempt_start`` preserves globally monotonic
    worker identities while each invocation keeps its own bounded retry budget."""
    from flash.runner import (
        TERMINAL_STATES,
        _persist_metrics,
        _RunCancelled,
        _status_estimated_charge,
        _submit_seed_supervised,
        _update,
        artifacts_dir,
        get_status,
    )

    # Defense in depth against the recovery TOCTOU (see attach_run): a run can be flipped into ANY
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
        code_prefix=code_prefix,
        attempt_start=attempt_start,
    )
    # measured wall x $/hr is recorded in metrics.json for analytics, but is NOT what we charge.
    measured_cost = prior_cost + _persist_metrics(spec, metrics)
    # The customer is charged the submit-time QUOTE, not measured wall. Legacy runs without a
    # persisted quote are re-priced from the spec, falling back only for old/unpriceable records.
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
    _RECOVERY_MARKER_GRACE_S,
    _RECOVERY_METRICS_POLL_S,
    _RUNPOD_STATUS_PROBE_TIMEOUT_S,
    _adopt_completed_attempt,
    _apply_charge_with_state,
    _await_runpod_completed_metrics,
    _candidate_usable_vram_gb,
    _canonical_provider_handle,
    _charge_completed_run_by_id,
    _completed_attempt_metrics,
    _CompletedAttemptPending,
    _oom_escalated,
    _projected_retry_class,
    _register_checkpoints_best_effort,
    _runpod_completed_metrics,
    _select_candidate,
    _shape_key,
    _strict_teardown_handle,
    _worker_provably_gone,
)
