"""Supervised seed submission and retry phases.

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
    ObservedDecisionState,
    ObservedRetryDecision,
    RetryState,
    _candidate_usable_vram_gb,
    _drop_weight_cache,
    _managed_cache_mounted,
    decide_failure_atomically,
)
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV


@dataclass
class _SubmitContext:
    spec: JobSpec
    seed: int
    log: object
    runtime_secrets: dict[str, str] | None
    source_snapshot: dict
    reserved_claim: AttemptLaunchClaim | None = None
    last_handle: dict = field(default_factory=dict)
    current_gpu: dict = field(default_factory=dict)
    current_usable_vram_gb: float = 0.0
    # persisted into the run handle so attach_run recovery polls with the same stall tuning.
    current_on_last_gpu: bool = False
    current_attempt: int = 0
    # tracks complete rN-suffixed retry handles that registry-less gc cannot reconstruct by name.
    seen_endpoints: dict[str, dict] = field(default_factory=dict)
    submission_lock: object | None = None
    last_detail: str | None = None
    current_claim: AttemptLaunchClaim | None = None

    def on_handle(self, handle: dict) -> None:
        from flash.runner.accounting.reconciliation import _preserve_cleanup_remote
        from flash.runner.supervise.errors import _TerminalHandleRace

        try:
            selected_provider = self.current_gpu.get("provider")
            if not isinstance(selected_provider, str) or not selected_provider:
                raise RuntimeError("selected provider identity is unavailable")
            canonical = _lifecycle._canonical_provider_handle(handle)
            canonical_handle = canonical.to_dict()
            if canonical.provider != selected_provider:
                raise RuntimeError("provider handle identity does not match the selected provider")
            if canonical_handle["attempt"] != self.current_attempt:
                raise RuntimeError("provider handle attempt does not match the reserved attempt")
            claim = self.current_claim
            if claim is None or claim.attempt != self.current_attempt:
                raise RuntimeError("provider handle has no active launch claim")
            if canonical_handle.get("endpoint_id"):
                self.seen_endpoints[canonical_handle["endpoint_id"]] = dict(canonical_handle)
            persisted_handle = {
                **canonical_handle,
                "launch_claim_token": claim.token,
                "seed": int(self.seed),
                "allocated_gpu": self.current_gpu.get("name"),
                # carried beside the gpu name for the same reason and by the same route: the
                # canonical provider handle drops unknown keys, so a recovering process can only
                # learn the shape from what was persisted here. without the count a run adopted
                # after a control-plane restart prices its wall as a single card.
                "allocated_gpu_count": self.current_gpu.get("count"),
                "allocated_usable_vram_gb": self.current_usable_vram_gb,
                "on_last_gpu": bool(self.current_on_last_gpu),
            }
            from flash.runner.lifecycle.attempts import persist_claimed_remote

            if persist_claimed_remote(self.spec.run_id, claim, persisted_handle):
                self.last_handle.clear()
                self.last_handle.update(persisted_handle)
                self.current_claim = None
                return
            resource_deleted = False
            with contextlib.suppress(Exception):
                resource_deleted = _lifecycle._strict_teardown_handle(
                    canonical_handle, self.spec.run_id
                )
            if resource_deleted:
                self.last_handle.clear()
            else:
                _preserve_cleanup_remote(self.spec.run_id, persisted_handle)
            raise _TerminalHandleRace(
                f"run {self.spec.run_id} became terminal while its provider handle was being persisted"
            )
        finally:
            lock = self.submission_lock
            self.submission_lock = None
            if lock is not None:
                lock.release()

    def gc_seen_endpoints(self) -> None:
        # only runpod handles carry an endpoint_id, so this set is empty on a plane without it.
        if not self.seen_endpoints:
            return
        from flash.providers.core.base import JobHandle
        from flash.providers.core.registry import get_provider

        rp = get_provider("runpod")
        for remote in self.seen_endpoints.values():
            with contextlib.suppress(Exception):
                rp.destroy(JobHandle.from_dict(remote))

    def cancel(self):
        """Reap this seed's tracked endpoints before unwinding on cancel."""
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
        self.gc_seen_endpoints()
        if self.current_gpu.get("name"):
            metrics.setdefault("allocated_gpu", self.current_gpu["name"])
        if self.current_gpu.get("provider"):
            metrics.setdefault("allocated_provider", self.current_gpu["provider"])
        # the runpod serverless route returns here rather than through the `res.ok` stamp below, so
        # the card count has to be recorded on both or a sharded serverless run is still priced as
        # one card. same source either way: the candidate allocation actually chose.
        if self.current_gpu.get("count"):
            metrics.setdefault("allocated_gpu_count", int(self.current_gpu["count"]))
        return metrics


@dataclass(frozen=True)
class _PreparedAttempt:
    claim: AttemptLaunchClaim
    attempt_spec: JobSpec
    runtime_secrets: dict[str, str]
    retry_state: RetryState

    @property
    def attempt(self) -> int:
        return self.claim.attempt

    @property
    def retry_snapshot(self) -> dict:
        return self.claim.retry_snapshot

    # the rank count a pinned opd resume checkpoint was written at, or None when this attempt
    # resumes from nothing. allocation is pinned to it: the worker refuses a pinned checkpoint
    # whose fsdp width differs from the attempt's, so re-ranking onto another shape would strand
    # the run on the only checkpoint it is authorized to continue from.
    resume_world_size: int | None = None


@dataclass(frozen=True)
class _PreparationOutcome:
    prepared: _PreparedAttempt | None = None
    completed_metrics: dict | None = None


@dataclass(frozen=True)
class _CandidatePlan:
    allocation: object
    candidates: tuple
    chosen: object
    on_last_gpu: bool
    effective_spec: JobSpec
    run_spec: JobSpec


@dataclass(frozen=True)
class _AttemptOutcome:
    result: object | None = None
    chosen: object | None = None
    candidates: tuple = ()
    run_spec: JobSpec | None = None
    candidate_not_started: bool = False
    failure_decision: ObservedRetryDecision = field(default_factory=ObservedRetryDecision.pending)


@dataclass(frozen=True)
class _FailureDecision:
    metrics: dict | None
    retry: bool
    retry_delay: float = 0.0
    lost_ownership: bool = False


def _build_context(
    spec: JobSpec,
    seed: int,
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
        seed=seed,
        log=log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        reserved_claim=reserved_claim,
        current_attempt=reserved_claim.attempt if reserved_claim else 0,
    )


def _require_opd_configuration(ctx: _SubmitContext) -> None:
    if ctx.spec.algorithm != "opd":
        return
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.server.domain.teacher.broker import require_teacher_broker_configuration

    # configuration and absolute policy fail before allocation can create a paid worker.
    require_teacher_broker_configuration(ctx.spec)
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
            f"seed {ctx.seed}: terminal worker's leaked endpoint cleanup target could not be persisted"
        )
    if worker_gone:
        if not _compare_and_clear_remote(ctx.spec.run_id, ctx.last_handle):
            raise RuntimeError(
                f"seed {ctx.seed}: previous attempt's persisted remote changed before clear; "
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
        f"seed {ctx.seed}: previous attempt's {ctx.last_handle.get('provider')} {resource_kind} "
        f"{resource_id} teardown could not be confirmed; failing to avoid "
        "double-provisioning a second worker over a possibly-live resource"
    )


def _mark_attempt_boundary(ctx: _SubmitContext, attempt: int) -> None:
    """Write the line that says everything above belongs to a previous attempt.

    The run log is one append-only file for the whole run, so a retry's output lands directly after
    the dead attempt's traceback with nothing in between. `flash runs log` tails that file, so while
    a replacement worker is booting the tail still ends in the OOM stack that caused the retry --
    an operator checking a run that is currently fine reads a failure. The status line already
    carries `attempt=` and `(prev attempt)`, but those come from the heartbeat and describe the run,
    not the bytes; nothing marked the bytes themselves.

    Written for attempt > 0 only. Attempt 0 has nothing above it to disown, and a header on every
    single-attempt run would be noise on the common path.

    The line names only where attempt N BEGINS. Saying "everything above is attempt N-1" is false
    from the second retry on: above attempt 2 sit attempts 0 and 1, and the log is also written by
    the poller and the billing retry path, so no single attempt owns the preceding bytes anyway. A
    boundary the reader can trust is worth more than an attribution that is right once.
    """
    if attempt <= 0:
        return
    with contextlib.suppress(Exception):
        print(
            f"---- attempt {attempt} starts here; everything above it is from earlier attempts ----",
            file=ctx.log,
            flush=True,
        )


def _prepare_attempt(ctx: _SubmitContext) -> _PreparationOutcome:
    from flash.runner.lifecycle.attempts import reserve_verified_attempt_launch
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall

    ctx.raise_if_cancelled()
    if ctx.last_handle:
        completed_metrics = _cleanup_previous_attempt(ctx, ctx.current_attempt + 1)
        if completed_metrics is not None:
            return _PreparationOutcome(completed_metrics=completed_metrics)
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
    retry_state = RetryState.from_snapshot(ctx.spec, claim.retry_snapshot)
    ctx.current_claim = claim
    ctx.current_attempt = claim.attempt
    _mark_attempt_boundary(ctx, claim.attempt)
    attempt_runtime_secrets = dict(ctx.runtime_secrets or {})
    attempt_runtime_secrets.pop(OPD_RESUME_REVISION_ENV, None)
    if claim.resume_revision is not None:
        attempt_runtime_secrets[OPD_RESUME_REVISION_ENV] = claim.resume_revision
    return _PreparationOutcome(
        prepared=_PreparedAttempt(
            claim,
            attempt_spec,
            attempt_runtime_secrets,
            retry_state,
            resume_world_size=claim.resume_world_size,
        )
    )


def _allocate_attempt(ctx: _SubmitContext, prepared: _PreparedAttempt):
    from flash.providers.core.allocator import allocate
    from flash.providers.core.base import CapacityUnavailableError, PollResult, UnsupportedGpuError
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.status import get_status

    # a cancel can land after _run_training's pre-submit check but while
    # allocation/pricing runs, when no handle exists yet for cancel_run() to
    # delete. re-read state right before paid provisioning so a cancelled run
    # never launches a worker (the later checks only stop the final-state
    # overwrite, after the gpu has already run and billed).
    with contextlib.suppress(FileNotFoundError):
        if get_status(ctx.spec.run_id).state == "cancelled":
            raise ctx.cancel()
    from flash.cost.spec import sft_ranking_overrides

    try:
        allocation = allocate(
            prepared.attempt_spec.model,
            prepared.attempt_spec.algorithm,
            train=prepared.attempt_spec.train,
            # profile-derived knobs (executed batch, row count, measured length): ranking must price
            # the work that will run, not the authored request. see `sft_ranking_overrides`.
            overrides=sft_ranking_overrides(prepared.attempt_spec),
            thinking=prepared.attempt_spec.thinking,
            # the run's requested disk, so the vast capacity check searches at the same effective
            # floor submit provisions with — else a high-disk run is advertised vast capacity that
            # only exists at 60 gb and then can't rent.
            disk_gb=float(getattr(prepared.attempt_spec.gpu, "disk_gb", 0.0) or 0.0),
            # the remaining run-global wall cap, so retries cannot reset the duration budget.
            # searched at the LAUNCH DEADLINE, not the bare work grant: vast's rent-duration floor
            # widens the offer search to the deadline, and an unarmed profile carries a queue
            # allowance on top of its grant. Searching on the grant alone advertised classes whose
            # offers expire before submit's own floor, so the rent then found nothing at the wider
            # window and the run failed on capacity that was never really there.
            max_wall_seconds=max(
                float(getattr(prepared.attempt_spec.gpu, "max_wall_seconds", 0.0) or 0.0),
                max(0.0, _load_run_deadline_at(ctx.spec.run_id) - _lifecycle.time.time()),
            ),
            provider=getattr(prepared.attempt_spec.gpu, "provider", ""),
            providers=getattr(prepared.attempt_spec.gpu, "providers", ()),
            # `attempt_spec` is rebuilt from `ctx.spec` every attempt, so this is the authored pin
            # rather than the class a previous attempt allocated -- which is what lets every retry
            # re-search the whole acceptable set instead of narrowing to whichever class was tried
            # first. `_spec_with_gpu` stamps the chosen class onto the submitted spec, downstream of
            # here; the fallbacks ride along on it so a recovered run keeps its failover.
            gpu_type=getattr(prepared.attempt_spec.gpu, "type", ""),
            gpu_type_fallbacks=getattr(prepared.attempt_spec.gpu, "type_fallbacks", ()),
            model_revision=prepared.attempt_spec.model_revision,
            # an authored gpu.count is a hard ceiling. the digest-stable integer 1 on an unpinned
            # spec is only a placeholder, so the marker must reach allocation as none or auto-sizing
            # silently collapses back to one card after the preparation round trips.
            max_gpu_count=prepared.attempt_spec.authored_gpu_count,
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
    return _pinned_to_resume_width(allocation, prepared.resume_world_size), None


def _pinned_to_resume_width(allocation, resume_world_size: int | None):
    """Drop shapes a pinned OPD resume checkpoint cannot be loaded on.

    The worker fails closed when a pinned resume's fsdp shards were written at a width other than
    the one the attempt runs at: it refuses to train rather than restart from step 0, because the
    control-plane gate authorized this replacement only to continue from exactly that checkpoint.
    Allocation, though, re-ranks every retry from scratch -- capacity and live pricing move, and
    rentable shapes can carry more cards than the ranks that join the run. Comparing the checkpoint
    width with the rented count can reject a loadable shape or admit one the worker cannot load.

    Narrowing the candidates is what closes that gap, and it is done here rather than by passing a
    count into ``allocate()``: ``max_gpu_count`` is a ceiling, not an exact width, so a ceiling of 2
    still admits a 1-card shape. Allocator-stamped candidates carry their executed rank count while
    unstamped candidates fall back to their rented count. The filter leaves ranking and shapes intact.

    An empty result is left empty deliberately: the caller reports no-capacity and retries, which is
    the truthful outcome when the only loadable shape is unavailable. Restarting from step 0 instead
    would repeat already-billed teacher work and optimizer steps outside what the gate approved.
    """
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


def _build_candidate_plan(
    ctx: _SubmitContext, prepared: _PreparedAttempt, allocation
) -> _CandidatePlan | None:
    from flash.providers.core.allocator import allocation_summary
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall
    from flash.runner.supervise.lifecycle import _spec_with_gpu

    if prepared.resume_world_size and not allocation.candidates:
        width = prepared.resume_world_size
        ctx.last_detail = (
            f"no candidate executing at checkpoint world size {width} is available, and this "
            "retry must preserve that executed rank width"
        )
        return None

    candidates, chosen = prepared.retry_state.select_candidate(allocation.candidates)
    if chosen is None:
        floor = prepared.retry_state.usable_vram_floor
        ctx.last_detail = f"no candidate has more than {floor:g} GB usable vram"
        return None

    cache_fallback_available = (
        not prepared.retry_state.drop_weight_cache
        and prepared.retry_state.cache_retries > 0
        and _managed_cache_mounted(prepared.attempt_spec, prepared.retry_state, chosen)
    )
    on_last_gpu = prepared.retry_state.on_last_gpu(
        chosen,
        candidates,
        cache_fallback_available=cache_fallback_available,
    )
    ctx.current_on_last_gpu = on_last_gpu
    print(allocation_summary(allocation), file=ctx.log, flush=True)
    if (chosen.provider, chosen.gpu, getattr(chosen, "gpu_count", 1)) != (
        allocation.provider,
        allocation.gpu,
        getattr(allocation, "gpu_count", 1),
    ):
        print(
            f"retry {prepared.attempt}: selected strictly larger {chosen.gpu} "
            f"@ {chosen.provider} ${chosen.hourly_usd:.2f}/hr",
            file=ctx.log,
            flush=True,
        )
    effective_spec = _spec_with_gpu(ctx.spec, chosen.gpu, getattr(chosen, "gpu_count", 1))
    if prepared.retry_state.drop_weight_cache:
        effective_spec = _drop_weight_cache(effective_spec)
    try:
        run_spec = _spec_with_remaining_wall(effective_spec, require_provider_minimum=True)
    except RuntimeError:
        ctx.gc_seen_endpoints()
        raise
    ctx.current_gpu.update(
        name=chosen.gpu,
        provider=chosen.provider,
        count=int(getattr(chosen, "gpu_count", 1) or 1),
    )
    ctx.current_usable_vram_gb = _candidate_usable_vram_gb(chosen)
    ctx.current_attempt = prepared.attempt
    return _CandidatePlan(allocation, candidates, chosen, on_last_gpu, effective_spec, run_spec)


def _retry_delay(ctx: _SubmitContext, infra_retry_ordinal: int | None) -> float:
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    if infra_retry_ordinal is None:
        return 0
    remaining = _load_run_deadline_at(ctx.spec.run_id) - _lifecycle.time.time()
    return min(10 * infra_retry_ordinal, remaining) if remaining > 0 else 0


def _submit_provider(
    ctx: _SubmitContext,
    prepared: _PreparedAttempt,
    plan: _CandidatePlan,
):
    from flash.providers.core.base import (
        PollResult,
        RunExhaustedProviderPoolError,
        UnreconciledCreateError,
    )
    from flash.providers.core.registry import get_provider
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at, _worker_deadline_at
    from flash.runner.supervise.errors import _TerminalHandleRace
    from flash.server.domain.teacher.broker import teacher_attempt_transport

    provider = get_provider(plan.chosen.provider)
    try:
        with teacher_attempt_transport(
            plan.run_spec,
            attempt=prepared.attempt,
            deadline_at=_load_run_deadline_at(ctx.spec.run_id),
        ) as teacher_secrets:
            prepared.runtime_secrets.update(teacher_secrets)
            submit_kwargs = {
                "log": ctx.log,
                "on_handle": ctx.on_handle,
                "attempt": prepared.attempt,
                "on_last_gpu": plan.on_last_gpu,
                "source_snapshot": ctx.source_snapshot,
                # bounded, not the raw run deadline: while a profile is unarmed the
                # persisted one still carries the queue allowance, and the bootstrap
                # enforces whatever absolute deadline it is handed regardless of
                # max_wall_seconds. see _worker_deadline_at.
                "_deadline_at": _worker_deadline_at(ctx.spec.run_id, plan.run_spec),
            }
            if prepared.runtime_secrets:
                submit_kwargs["runtime_secrets"] = prepared.runtime_secrets
            return provider.submit_run(plan.run_spec, ctx.seed, **submit_kwargs), False
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


def _submit_candidate(
    ctx: _SubmitContext,
    prepared: _PreparedAttempt,
    plan: _CandidatePlan,
):
    from flash.runner.lifecycle.attempts import require_attempt_launch_current
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled
    from flash.server.platform.locks import _deploy_lock

    candidate_not_started = False
    ctx.submission_lock = _deploy_lock(ctx.spec.run_id)
    ctx.submission_lock.acquire()
    try:
        try:
            require_attempt_launch_current(ctx.spec.run_id, ctx.spec, prepared.claim)
        except RuntimeError as exc:
            raise _RunCancelled(
                f"run {ctx.spec.run_id} attempt {prepared.attempt} lost provider launch authorization"
            ) from exc
        # the accepted customer quote was frozen during preparation and is exactly the amount shown
        # by `flash train --cost`. allocation persists only the effective worker spec; live provider
        # rates and topology must never rewrite estimated_cost_usd after submission.
        if not _persist_effective_worker_spec(plan.effective_spec):
            raise ctx.cancel()
        if get_status(ctx.spec.run_id).state in TERMINAL_STATES:
            raise ctx.cancel()
        result, candidate_not_started = _submit_provider(ctx, prepared, plan)
    finally:
        lock = ctx.submission_lock
        ctx.submission_lock = None
        if lock is not None:
            lock.release()
    return result, candidate_not_started


def _run_attempt(ctx: _SubmitContext, prepared: _PreparedAttempt) -> _AttemptOutcome:
    from flash.providers.core.base import PollResult

    allocation, result = _allocate_attempt(ctx, prepared)
    if allocation is None:
        return _AttemptOutcome(result=result, candidate_not_started=True)
    ctx.raise_if_cancelled()
    plan = _build_candidate_plan(ctx, prepared, allocation)
    if plan is None:
        result = PollResult(
            False, failure="no_capacity", detail=ctx.last_detail or "no eligible candidate"
        )
        decision = decide_failure_atomically(
            ctx.spec.run_id,
            ctx.spec,
            claim_token=prepared.claim.token,
            expected_remote=dict(ctx.last_handle) if ctx.last_handle else None,
            expected_retry_snapshot=prepared.retry_snapshot,
            observation=FailureObservation.create(
                result.failure,
                chosen=None,
                candidates=allocation.candidates,
                managed_cache_mounted=False,
            ),
            attempt=prepared.attempt,
        )
        return _AttemptOutcome(
            result=result,
            candidates=allocation.candidates,
            candidate_not_started=True,
            failure_decision=decision,
        )
    result, candidate_not_started = _submit_candidate(ctx, prepared, plan)
    return _AttemptOutcome(
        result=result,
        chosen=plan.chosen,
        candidates=plan.candidates,
        run_spec=plan.run_spec,
        candidate_not_started=candidate_not_started,
    )


def _return_success_metrics(ctx: _SubmitContext, outcome: _AttemptOutcome) -> dict:
    # a late worker success must not resurrect a cancelled run.
    ctx.raise_if_cancelled()
    ctx.gc_seen_endpoints()
    metrics = outcome.result.metrics
    if outcome.chosen is not None and isinstance(metrics, dict):
        metrics.setdefault("allocated_gpu", outcome.chosen.gpu)
        # the provider that actually billed this run, so cost attribution prices the
        # class on its substrate rather than assuming runpod's table.
        metrics.setdefault("allocated_provider", outcome.chosen.provider)
        # and how many cards of it. `hourly_rate(gpu_type)` is per card, so a fallback that
        # prices the wall once records a 2x/4x run at half or a quarter of its real spend.
        # the spec's own gpu.count cannot stand in: it is a ceiling, and allocation
        # routinely picks fewer (see training.md, "a ceiling, not an exact count").
        metrics.setdefault("allocated_gpu_count", int(getattr(outcome.chosen, "gpu_count", 1)))
    return metrics


def _handle_failure(
    ctx: _SubmitContext,
    prepared: _PreparedAttempt,
    outcome: _AttemptOutcome,
) -> _FailureDecision:
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    ctx.raise_if_cancelled()
    completed_metrics = _lifecycle._await_runpod_completed_metrics(
        ctx.last_handle,
        _load_run_deadline_at(ctx.spec.run_id),
        check_cancelled=ctx.raise_if_cancelled,
    )
    if completed_metrics is not None:
        return _FailureDecision(ctx.return_completed_runpod_metrics(completed_metrics), False)

    result = outcome.result
    ctx.last_detail = f"{result.failure}: {result.detail}"
    observed = outcome.failure_decision
    if observed.state is ObservedDecisionState.PENDING:
        observed = decide_failure_atomically(
            ctx.spec.run_id,
            ctx.spec,
            claim_token=prepared.claim.token,
            expected_remote=dict(ctx.last_handle) if ctx.last_handle else None,
            expected_retry_snapshot=prepared.retry_snapshot,
            observation=FailureObservation.create(
                result.failure,
                chosen=outcome.chosen,
                candidates=outcome.candidates,
                managed_cache_mounted=_managed_cache_mounted(
                    prepared.attempt_spec,
                    prepared.retry_state,
                    outcome.chosen,
                ),
            ),
            attempt=prepared.attempt,
        )
    from flash.runner.lifecycle.attempts import release_launch_claim

    release_launch_claim(ctx.spec.run_id, prepared.claim)
    ctx.current_claim = None
    if observed.state is ObservedDecisionState.OWNERSHIP_LOST:
        return _FailureDecision(None, False, lost_ownership=True)
    if observed.state is not ObservedDecisionState.PERSISTED or observed.decision is None:
        raise AssertionError("unhandled retry decision state")
    plan = observed.decision.plan
    retry_delay = (
        _retry_delay(ctx, plan.infra_retry_ordinal)
        if plan.retry and outcome.candidate_not_started
        else 0.0
    )
    print(
        f"seed={ctx.seed} attempt={prepared.attempt} failed ({result.failure}); {plan.action}"
        f"\n--- failure detail ---\n{(result.detail or '')[:2000]}\n---",
        file=ctx.log,
        flush=True,
    )
    return _FailureDecision(None, plan.retry, retry_delay)


def submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    reserved_claim: AttemptLaunchClaim | None = None,
) -> dict:
    """Run one seed with bounded auto-retry on infra-shaped failures."""
    if spec.algorithm == "opd":
        from flash.server.domain.teacher.broker import preflight_validate_managed_teacher

        # policy and plane configuration are spec-level gates and must fail before durable run state
        # or source identity is consulted. the deadline-dependent gate still runs after context load.
        preflight_validate_managed_teacher(spec)
    ctx = _build_context(
        spec,
        seed,
        log,
        runtime_secrets,
        source_snapshot,
        reserved_claim,
    )
    _require_opd_configuration(ctx)
    while True:
        preparation = _prepare_attempt(ctx)
        if preparation.completed_metrics is not None:
            return ctx.return_completed_runpod_metrics(preparation.completed_metrics)
        prepared = preparation.prepared
        outcome = _run_attempt(ctx, prepared)
        if outcome.result.ok:
            return _return_success_metrics(ctx, outcome)
        decision = _handle_failure(ctx, prepared, outcome)
        if decision.metrics is not None:
            return decision.metrics
        if decision.lost_ownership:
            from flash.runner.supervise.errors import _LaunchOwnershipLost

            raise _LaunchOwnershipLost(f"seed {seed} lost retry decision ownership")
        if not decision.retry:
            break
        if decision.retry_delay:
            _lifecycle.time.sleep(decision.retry_delay)
    ctx.gc_seen_endpoints()
    raise RuntimeError(f"seed {seed} failed after retries: {ctx.last_detail}")
