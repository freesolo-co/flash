"""Supervised attempt submission and retry phases.

Split out of ``flash.runner.supervise.lifecycle`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace

from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import JobSpec
from flash.runner.lifecycle.attempts import AttemptLaunchClaim
from flash.runner.supervise import lifecycle as _lifecycle
from flash.runner.supervise.retry_decision import (
    FailureObservation,
    _candidate_usable_vram_gb,
    _drop_weight_cache,
    _managed_cache_mounted,
)
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV


@dataclass
class _SubmitContext:
    spec: JobSpec
    log: object
    runtime_secrets: dict[str, str] | None
    source_snapshot: dict
    reserved_claim: AttemptLaunchClaim | None = None
    last_handle: dict = field(default_factory=dict)
    # tracks complete attempt-suffixed handles that registry-less gc cannot reconstruct by name.
    seen_endpoints: dict[str, dict] = field(default_factory=dict)
    last_detail: str | None = None
    current_claim: AttemptLaunchClaim | None = None

    def gc_seen_endpoints(self) -> None:
        # only runpod handles carry an endpoint_id, so this set is empty on a plane without it.
        if not self.seen_endpoints:
            return
        from flash.runner.accounting.reconciliation import _record_cleanup_remote

        for remote in self.seen_endpoints.values():
            try:
                if not _lifecycle._strict_teardown_handle(remote, self.spec.run_id):
                    _record_cleanup_remote(self.spec.run_id, remote)
            except Exception:
                _record_cleanup_remote(self.spec.run_id, remote)

    def cancel(self):
        """Reap this run's tracked endpoints before unwinding on cancel."""
        from flash.runner.supervise.errors import _RunCancelled

        # a handle whose `running` write loses the terminal-stickiness race never lands in
        # status.remote, so only seen_endpoints (rN walk endpoints _gc_run_endpoints can't name)
        # can free it.
        self.gc_seen_endpoints()
        return _RunCancelled(f"run {self.spec.run_id} was cancelled")

    def raise_if_cancelled(self) -> None:
        from flash.runner.lifecycle.status import get_status

        try:
            if get_status(self.spec.run_id).state == "cancelled":
                raise self.cancel()
        except FileNotFoundError:
            pass

    def return_completed_runpod_metrics(self, metrics: dict) -> dict:
        self.raise_if_cancelled()
        _settle_terminal_remote(self)
        self.gc_seen_endpoints()
        if self.last_handle.get("allocated_gpu"):
            metrics.setdefault("allocated_gpu", self.last_handle["allocated_gpu"])
        if self.last_handle.get("provider"):
            metrics.setdefault("allocated_provider", self.last_handle["provider"])
        if self.last_handle.get("allocated_gpu_count"):
            metrics.setdefault("allocated_gpu_count", int(self.last_handle["allocated_gpu_count"]))
        return metrics


def _build_context(
    spec: JobSpec,
    log,
    runtime_secrets: dict[str, str] | None,
    source_snapshot: dict | None,
    reserved_claim: AttemptLaunchClaim | None,
) -> _SubmitContext:
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status
    from flash.snapshot.archive import parse_descriptor

    source_snapshot = parse_descriptor(
        source_snapshot or source_snapshot_from_status(get_status(spec.run_id), required=True)
    ).to_dict()
    if reserved_claim is not None and not isinstance(reserved_claim, AttemptLaunchClaim):
        raise RuntimeError("reserved launch claim is invalid")
    return _SubmitContext(
        spec=spec,
        log=log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        reserved_claim=reserved_claim,
    )


def _require_opd_configuration(ctx: _SubmitContext) -> None:
    if ctx.spec.algorithm != "opd":
        return
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.server.domain.teacher.broker import require_teacher_broker_configuration

    # configuration and absolute policy fail before allocation can create a paid worker.
    require_teacher_broker_configuration(
        ctx.spec,
        deadline_at=_load_run_deadline_at(ctx.spec.run_id),
    )


def _cleanup_previous_attempt(ctx: _SubmitContext, attempt: int) -> dict | None:
    if not ctx.last_handle:
        return None
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.runner.accounting.reconciliation import (
        _compare_and_clear_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    completed_metrics = _lifecycle._await_runpod_completed_metrics(
        ctx.last_handle,
        _load_run_deadline_at(ctx.spec.run_id),
        check_cancelled=ctx.raise_if_cancelled,
    )
    if completed_metrics is not None:
        return completed_metrics
    resource_deleted = False
    teardown_error: Exception | None = None
    try:
        resource_deleted = _lifecycle._strict_teardown_handle(
            JobHandle.from_dict(ctx.last_handle), ctx.spec.run_id
        )
    except Exception as exc:
        teardown_error = exc
    resource_kind = "endpoint" if ctx.last_handle.get("endpoint_id") else "instance"
    resource_id = ctx.last_handle.get("endpoint_id") or ctx.last_handle.get("instance_id")
    worker_gone = teardown_error is None or _lifecycle._worker_provably_gone(
        ctx.spec.run_id, ctx.last_handle
    )
    if (
        worker_gone
        and ctx.last_handle.get("provider") == "runpod"
        and not resource_deleted
        and not _record_cleanup_remote(ctx.spec.run_id, ctx.last_handle)
    ):
        raise RuntimeError(
            f"run {ctx.spec.run_id}: terminal worker's leaked endpoint cleanup target could not be persisted"
        )
    if worker_gone:
        if not _compare_and_clear_remote(ctx.spec.run_id, ctx.last_handle):
            raise RuntimeError(
                f"run {ctx.spec.run_id}: previous attempt's persisted remote changed before clear; "
                "aborting replacement to avoid double-provisioning"
            )
        message = (
            "terminated"
            if resource_deleted
            else "teardown unconfirmed but worker terminal; proceeding, leaked resource persisted for cleanup"
        )
        print(
            f"retry {attempt}: {ctx.last_handle.get('provider')} {resource_kind} "
            f"{resource_id} {message}",
            file=ctx.log,
            flush=True,
        )
        ctx.last_handle.clear()
        if resource_kind == "endpoint":
            ctx.seen_endpoints.pop(resource_id, None)
        return None
    with contextlib.suppress(Exception):
        get_provider(ctx.last_handle["provider"]).gc(ctx.spec)
    ctx.gc_seen_endpoints()
    print(
        f"retry {attempt}: {ctx.last_handle.get('provider')} {resource_kind} {resource_id} "
        f"teardown unconfirmed ({type(teardown_error).__name__}); keeping the handle so the "
        "possibly-billing resource stays reachable for cleanup",
        file=ctx.log,
        flush=True,
    )
    raise RuntimeError(
        f"run {ctx.spec.run_id}: previous attempt's {ctx.last_handle.get('provider')} {resource_kind} "
        f"{resource_id} teardown could not be confirmed; failing to avoid "
        "double-provisioning a second worker over a possibly-live resource"
    )


def _mark_attempt_boundary(ctx: _SubmitContext, attempt: int) -> None:
    """Separate a replacement attempt from earlier append-only log output."""
    if attempt <= 0:
        return
    with contextlib.suppress(Exception):
        print(
            f"---- attempt {attempt} starts here; everything above it is from earlier attempts ----",
            file=ctx.log,
            flush=True,
        )


def _prepare_attempt(ctx: _SubmitContext):
    from flash.runner.lifecycle.attempts import reserve_verified_attempt_launch
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall

    ctx.raise_if_cancelled()
    if ctx.last_handle:
        completed_metrics = _cleanup_previous_attempt(
            ctx, int(ctx.last_handle.get("attempt", -1)) + 1
        )
        if completed_metrics is not None:
            return None, completed_metrics
    try:
        attempt_spec = _spec_with_remaining_wall(ctx.spec, require_provider_minimum=True)
    except RuntimeError:
        ctx.gc_seen_endpoints()
        raise
    claim = ctx.reserved_claim
    ctx.reserved_claim = None
    if claim is None:
        claim = reserve_verified_attempt_launch(ctx.spec.run_id)
        if claim is None:
            from flash.runner.supervise.errors import _LaunchOwnershipLost

            raise _LaunchOwnershipLost("attempt launch reservation lost ownership")
    from flash.runner.lifecycle.attempts import require_attempt_launch_current

    # own the claim before any call that can raise: an exception between reservation and this
    # assignment would leave the caller's cleanup with no claim to consume, stranding the lease.
    ctx.current_claim = claim
    retry_state = require_attempt_launch_current(ctx.spec.run_id, ctx.spec, claim)
    _mark_attempt_boundary(ctx, claim.attempt)
    attempt_runtime_secrets = dict(ctx.runtime_secrets or {})
    attempt_runtime_secrets.pop(OPD_RESUME_REVISION_ENV, None)
    if claim.resume_revision is not None:
        attempt_runtime_secrets[OPD_RESUME_REVISION_ENV] = claim.resume_revision
    return (claim, attempt_spec, attempt_runtime_secrets, retry_state), None


def _allocate_attempt(ctx: _SubmitContext, prepared):
    claim, attempt_spec, _, _ = prepared
    from flash.providers.core.allocator import allocate
    from flash.providers.core.base import CapacityUnavailableError, PollResult, UnsupportedGpuError
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import get_status

    with contextlib.suppress(FileNotFoundError):
        if get_status(ctx.spec.run_id).state == "cancelled":
            raise ctx.cancel()
    from flash.cost.spec import sft_ranking_overrides

    try:
        allocation = allocate(
            attempt_spec.model,
            attempt_spec.algorithm,
            train=attempt_spec.train,
            overrides=sft_ranking_overrides(attempt_spec),
            thinking=attempt_spec.thinking,
            disk_gb=float(getattr(attempt_spec.gpu, "disk_gb", 0.0) or 0.0),
            max_wall_seconds=max(
                float(getattr(attempt_spec.gpu, "max_wall_seconds", 0.0) or 0.0),
                max(0.0, _load_run_deadline_at(ctx.spec.run_id) - _lifecycle.time.time()),
            ),
            provider=getattr(attempt_spec.gpu, "provider", ""),
            providers=getattr(attempt_spec.gpu, "providers", ()),
            gpu_type=getattr(attempt_spec.gpu, "type", ""),
            gpu_type_fallbacks=getattr(attempt_spec.gpu, "type_fallbacks", ()),
            model_revision=attempt_spec.model_revision,
            max_gpu_count=attempt_spec.authored_gpu_count,
        )
    except UnsupportedGpuError:
        raise  # config-shaped: no gpu anywhere can run this job
    except CapacityUnavailableError as exc:
        return None, PollResult(False, failure="no_capacity", detail=str(exc))
    except Exception as exc:
        return None, PollResult(
            False,
            failure="poll_error",
            detail=f"allocation failed ({type(exc).__name__})",
        )
    return _pinned_to_resume_width(allocation, claim.resume_world_size), None


def _pinned_to_resume_width(allocation, resume_world_size: int | None):
    """Keep only candidates matching a pinned opd checkpoint's executed width."""
    if not resume_world_size:
        return allocation
    loadable = []
    for candidate in allocation.candidates:
        executed_gpu_count = getattr(candidate, "executed_gpu_count", None)
        width = (
            executed_gpu_count
            if type(executed_gpu_count) is int and executed_gpu_count > 0
            else int(getattr(candidate, "gpu_count", 1))
        )
        if width == resume_world_size:
            loadable.append(candidate)
    loadable = tuple(loadable)
    if loadable == allocation.candidates:
        return allocation
    if not loadable:
        return replace(allocation, candidates=())
    best = loadable[0]
    return replace(
        allocation,
        candidates=loadable,
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        gpu_count=best.gpu_count,
    )


def _build_candidate_plan(ctx: _SubmitContext, prepared, allocation):
    claim, attempt_spec, _, retry_state = prepared
    from flash.providers.core.allocator import allocation_summary
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall
    from flash.runner.supervise.lifecycle import _spec_with_gpu

    if claim.resume_world_size and not allocation.candidates:
        width = claim.resume_world_size
        ctx.last_detail = (
            f"no candidate executing at checkpoint world size {width} is available, and this "
            "retry must preserve that executed rank width"
        )
        return None

    candidates, chosen = retry_state.select_candidate(allocation.candidates)
    if chosen is None:
        floor = retry_state.usable_vram_floor
        ctx.last_detail = f"no candidate has more than {floor:g} GB usable vram"
        return None

    cache_fallback_available = (
        not retry_state.drop_weight_cache
        and retry_state.cache_retries > 0
        and _managed_cache_mounted(attempt_spec, chosen)
    )
    on_last_gpu = retry_state.on_last_gpu(
        chosen,
        candidates,
        cache_fallback_available=cache_fallback_available,
    )
    print(allocation_summary(allocation), file=ctx.log, flush=True)
    if (chosen.provider, chosen.gpu, getattr(chosen, "gpu_count", 1)) != (
        allocation.provider,
        allocation.gpu,
        getattr(allocation, "gpu_count", 1),
    ):
        print(
            f"retry {claim.attempt}: selected strictly larger {chosen.gpu} "
            f"@ {chosen.provider} ${chosen.hourly_usd:.2f}/hr",
            file=ctx.log,
            flush=True,
        )
    effective_spec = _spec_with_gpu(ctx.spec, chosen.gpu, getattr(chosen, "gpu_count", 1))
    if retry_state.drop_weight_cache:
        effective_spec = _drop_weight_cache(effective_spec)
    try:
        run_spec = _spec_with_remaining_wall(effective_spec, require_provider_minimum=True)
    except RuntimeError:
        ctx.gc_seen_endpoints()
        raise
    return candidates, chosen, on_last_gpu, effective_spec, run_spec


def _retry_delay(ctx: _SubmitContext, infra_retry_ordinal: int | None) -> float:
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    if infra_retry_ordinal is None:
        return 0
    remaining = _load_run_deadline_at(ctx.spec.run_id) - _lifecycle.time.time()
    return min(10 * infra_retry_ordinal, remaining) if remaining > 0 else 0


def _handle_callback(ctx: _SubmitContext, prepared, candidate_plan):
    """Bind provider handle persistence to one immutable reservation and candidate."""
    claim = prepared[0]
    _, chosen, on_last_gpu, _, _ = candidate_plan
    from flash.runner.accounting.reconciliation import (
        _expected_remote_matches,
        _preserve_cleanup_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.attempts import persist_claimed_remote
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.errors import _TerminalHandleRace

    def on_handle(handle: dict) -> None:
        canonical = _lifecycle._canonical_provider_handle(handle)
        remote = canonical.to_dict()
        if canonical.provider != chosen.provider:
            raise RuntimeError("provider handle identity does not match the selected provider")
        if remote["attempt"] != claim.attempt:
            raise RuntimeError("provider handle attempt does not match the reserved attempt")
        if remote.get("endpoint_id"):
            ctx.seen_endpoints[remote["endpoint_id"]] = dict(remote)
        persisted = {
            **remote,
            "launch_claim_token": claim.token,
            "allocated_gpu": chosen.gpu,
            "allocated_gpu_count": int(getattr(chosen, "gpu_count", 1) or 1),
            "allocated_usable_vram_gb": _candidate_usable_vram_gb(chosen),
            "on_last_gpu": on_last_gpu,
        }
        try:
            claimed = persist_claimed_remote(ctx.spec.run_id, claim, persisted)
        except Exception:
            # a raise does not prove nothing landed: `_save_status_unlocked` runs `os.replace`
            # before its directory `fsync`, so the remote can already be durable and visible. read
            # what actually landed instead of assuming the write failed.
            landed = False
            with contextlib.suppress(Exception):
                landed = _expected_remote_matches(get_status(ctx.spec.run_id).remote, persisted)
            if landed:
                # the write won. adopt it exactly as the success path would, so `_handle_failure`
                # reports the real expected remote and keeps ownership of the attempt.
                ctx.last_handle.clear()
                ctx.last_handle.update(persisted)
                ctx.current_claim = None
                raise
            # nothing durable names the resource, and it exists the moment the provider returned it.
            # `_submit_provider` turns this into a retryable `poll_error`, so unless the handle is
            # torn down or recorded here it is unreachable: the retry provisions a second worker
            # against the same run artifacts while this one keeps running.
            #
            # record the identity WITHOUT writing `status.remote`. the run is still nonterminal and
            # the launch claim is still on disk, so setting a remote here would leave nobody able to
            # own the run: this caller loses the ownership check, attach cannot own a remote while an
            # active claim exists, and handleless recovery refuses a set remote.
            deleted = False
            with contextlib.suppress(Exception):
                deleted = _lifecycle._strict_teardown_handle(remote, ctx.spec.run_id)
            if not deleted:
                with contextlib.suppress(Exception):
                    _record_cleanup_remote(ctx.spec.run_id, persisted)
            raise
        if claimed:
            ctx.last_handle.clear()
            ctx.last_handle.update(persisted)
            ctx.current_claim = None
            return
        deleted = False
        with contextlib.suppress(Exception):
            deleted = _lifecycle._strict_teardown_handle(remote, ctx.spec.run_id)
        if deleted:
            ctx.last_handle.clear()
        else:
            _preserve_cleanup_remote(ctx.spec.run_id, persisted)
        raise _TerminalHandleRace(
            f"run {ctx.spec.run_id} became terminal while its provider handle was being persisted"
        )

    return on_handle


def _submit_provider(ctx: _SubmitContext, prepared, candidate_plan):
    claim, _, attempt_runtime_secrets, _ = prepared
    _, chosen, on_last_gpu, _, run_spec = candidate_plan
    from flash.providers.core.base import (
        PollResult,
        RunExhaustedProviderPoolError,
        UnreconciledCreateError,
    )
    from flash.providers.core.registry import get_provider
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _worker_deadline_at
    from flash.runner.supervise.errors import _TerminalHandleRace
    from flash.server.domain.teacher.broker import teacher_attempt_transport

    provider = get_provider(chosen.provider)
    try:
        with teacher_attempt_transport(
            run_spec,
            attempt=claim.attempt,
            deadline_at=_load_run_deadline_at(ctx.spec.run_id),
        ) as teacher_secrets:
            attempt_runtime_secrets.update(teacher_secrets)
            submit_kwargs = {
                "log": ctx.log,
                "on_handle": _handle_callback(ctx, prepared, candidate_plan),
                "attempt": claim.attempt,
                "on_last_gpu": on_last_gpu,
                "source_snapshot": ctx.source_snapshot,
                # bounded, not the raw run deadline: while a profile is unarmed the
                # persisted one still carries the queue allowance, and the bootstrap
                # enforces whatever absolute deadline it is handed regardless of
                # max_wall_seconds. see _worker_deadline_at.
                "_deadline_at": _worker_deadline_at(ctx.spec.run_id, run_spec),
            }
            if attempt_runtime_secrets:
                submit_kwargs["runtime_secrets"] = attempt_runtime_secrets
            return provider.submit_attempt(run_spec, **submit_kwargs), False
    except _TerminalHandleRace:
        raise
    except Exception as exc:
        if isinstance(exc, UnreconciledCreateError):
            return (
                PollResult(
                    False,
                    failure="job_failed",
                    detail=f"provider create could not be reconciled ({type(exc).__name__})",
                ),
                False,
            )
        if isinstance(exc, RunExhaustedProviderPoolError):
            # the one submit-side message worth reading, and the only one safe to read: Flash
            # authors this string, so unlike a provider response body it cannot quote a request
            # that carried a credential. without this the class name alone would make "this run
            # burned the whole pool" indistinguishable from "the market is dry", which is exactly
            # the distinction the error exists to draw. still sanitized and bounded, because the
            # gpu class it interpolates comes from the spec.
            return (
                PollResult(
                    False,
                    failure="no_capacity",
                    detail=sanitize_diagnostic(str(exc), limit=1000),
                ),
                True,
            )
        # every other provider exception keeps its class name only. the text can quote a request
        # body, and this detail is persisted into the run record.
        return (
            PollResult(
                False,
                failure="poll_error",
                detail=f"provider submit failed ({type(exc).__name__})",
            ),
            True,
        )


def _submit_candidate(ctx: _SubmitContext, prepared, candidate_plan):
    claim = prepared[0]
    effective_spec = candidate_plan[3]
    from flash.runner.lifecycle.attempts import require_attempt_launch_current
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled

    try:
        require_attempt_launch_current(ctx.spec.run_id, ctx.spec, claim)
    except RuntimeError as exc:
        raise _RunCancelled(
            f"run {ctx.spec.run_id} attempt {claim.attempt} lost provider launch authorization"
        ) from exc
    # the accepted customer quote was frozen during preparation and is exactly the amount shown
    # by `flash train --cost`. allocation persists only the effective worker spec; live provider
    # rates and topology must never rewrite estimated_cost_usd after submission.
    if not _persist_effective_worker_spec(effective_spec):
        raise ctx.cancel()
    if get_status(ctx.spec.run_id).state in TERMINAL_STATES:
        raise ctx.cancel()
    return _submit_provider(ctx, prepared, candidate_plan)


def _run_attempt(ctx: _SubmitContext, prepared):
    from flash.providers.core.base import PollResult

    allocation, result = _allocate_attempt(ctx, prepared)
    if allocation is None:
        return result, None, (), True
    ctx.raise_if_cancelled()
    candidate_plan = _build_candidate_plan(ctx, prepared, allocation)
    if candidate_plan is None:
        result = PollResult(
            False, failure="no_capacity", detail=ctx.last_detail or "no eligible candidate"
        )
        return result, None, allocation.candidates, True
    result, candidate_not_started = _submit_candidate(ctx, prepared, candidate_plan)
    candidates, chosen, _, _, _ = candidate_plan
    return result, chosen, candidates, candidate_not_started


def _settle_terminal_remote(ctx: _SubmitContext) -> None:
    """track and clear an attempt only after exact teardown is confirmed."""
    if not ctx.last_handle:
        return
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _compare_and_remove_cleanup_remote,
        _record_cleanup_remote,
    )

    remote = dict(ctx.last_handle)
    cleanup_recorded = False
    with contextlib.suppress(Exception):
        cleanup_recorded = _record_cleanup_remote(ctx.spec.run_id, remote)
    try:
        deleted = _lifecycle._strict_teardown_handle(JobHandle.from_dict(remote), ctx.spec.run_id)
    except Exception:
        return
    if not deleted:
        return
    cleanup_cleared = False
    if cleanup_recorded:
        with contextlib.suppress(Exception):
            cleanup_cleared = _compare_and_remove_cleanup_remote(ctx.spec.run_id, remote)
    if not cleanup_cleared:
        with contextlib.suppress(Exception):
            _compare_and_confirm_remote_teardown(ctx.spec.run_id, remote)
    endpoint_id = remote.get("endpoint_id")
    if isinstance(endpoint_id, str):
        ctx.seen_endpoints.pop(endpoint_id, None)


def _return_success_metrics(ctx: _SubmitContext, outcome) -> dict:
    # a late worker success must not resurrect a cancelled run.
    ctx.raise_if_cancelled()
    _settle_terminal_remote(ctx)
    result, chosen, _, _ = outcome
    metrics = result.metrics
    if chosen is not None and isinstance(metrics, dict):
        metrics.setdefault("allocated_gpu", chosen.gpu)
        # the provider that actually billed this run, so cost attribution prices the
        # class on its substrate rather than assuming runpod's table.
        metrics.setdefault("allocated_provider", chosen.provider)
        # and how many cards of it. `hourly_rate(gpu_type)` is per card, so a fallback that
        # prices the wall once records a 2x/4x run at half or a quarter of its real spend.
        # the spec's own gpu.count cannot stand in: it is a ceiling, and allocation
        # routinely picks fewer (see training.md, "a ceiling, not an exact count").
        metrics.setdefault("allocated_gpu_count", int(getattr(chosen, "gpu_count", 1)))
    return metrics


def _handle_failure(ctx: _SubmitContext, prepared, outcome):
    claim, attempt_spec, _, _ = prepared
    result, chosen, candidates, candidate_not_started = outcome
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    ctx.raise_if_cancelled()
    completed_metrics = _lifecycle._await_runpod_completed_metrics(
        ctx.last_handle,
        _load_run_deadline_at(ctx.spec.run_id),
        check_cancelled=ctx.raise_if_cancelled,
    )
    if completed_metrics is not None:
        return ctx.return_completed_runpod_metrics(completed_metrics), False, 0.0, False

    ctx.last_detail = f"{result.failure}: {result.detail}"
    from flash.runner.lifecycle.attempts import decide_attempt_failure

    plan = decide_attempt_failure(
        ctx.spec.run_id,
        claim_token=claim.token,
        expected_remote=dict(ctx.last_handle) if ctx.last_handle else None,
        observation=FailureObservation(
            result.failure,
            chosen=chosen,
            candidates=candidates,
            managed_cache_mounted=_managed_cache_mounted(
                attempt_spec,
                chosen,
            ),
        ),
        attempt=claim.attempt,
    )
    ctx.current_claim = None
    if plan is None:
        return None, False, 0.0, True
    retry_delay = (
        _retry_delay(ctx, plan.infra_retry_ordinal) if plan.retry and candidate_not_started else 0.0
    )
    print(
        f"attempt={claim.attempt} failed ({result.failure}); {plan.action}"
        f"\n--- failure detail ---\n{(result.detail or '')[:2000]}\n---",
        file=ctx.log,
        flush=True,
    )
    return None, plan.retry, retry_delay, False


def run_attempts_supervised(
    spec: JobSpec,
    log,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> dict:
    """Run one run's attempts with bounded auto-retry on infra-shaped failures."""
    ctx = None
    try:
        if spec.algorithm == "opd":
            from flash.server.domain.teacher.broker import preflight_validate_managed_teacher

            # policy and plane configuration are spec-level gates and must fail before durable run state
            # or source identity is consulted. the deadline-dependent gate still runs after context load.
            preflight_validate_managed_teacher(spec)
        ctx = _build_context(
            spec,
            log,
            runtime_secrets,
            source_snapshot,
            reserved_claim,
        )
        _require_opd_configuration(ctx)
        while True:
            prepared, completed_metrics = _prepare_attempt(ctx)
            if completed_metrics is not None:
                return ctx.return_completed_runpod_metrics(completed_metrics)
            outcome = _run_attempt(ctx, prepared)
            if outcome[0].ok:
                metrics = _return_success_metrics(ctx, outcome)
                ctx.gc_seen_endpoints()
                return metrics
            metrics, retry, retry_delay, lost_ownership = _handle_failure(ctx, prepared, outcome)
            if metrics is not None:
                return metrics
            if lost_ownership:
                from flash.runner.supervise.errors import _LaunchOwnershipLost

                raise _LaunchOwnershipLost(f"run {spec.run_id} lost retry decision ownership")
            if not retry:
                break
            if retry_delay:
                _lifecycle.time.sleep(retry_delay)
        _settle_terminal_remote(ctx)
        ctx.gc_seen_endpoints()
        raise RuntimeError(f"run {spec.run_id} failed after retries: {ctx.last_detail}")
    finally:
        claim = reserved_claim if ctx is None else ctx.current_claim or ctx.reserved_claim
        if claim is not None:
            from flash.runner.lifecycle.attempts import consume_active_launch_claim

            with contextlib.suppress(Exception):
                consume_active_launch_claim(spec.run_id, claim)
            if ctx is not None:
                ctx.current_claim = None
                ctx.reserved_claim = None
