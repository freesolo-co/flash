"""Supervised attempt execution and retry phases.

Split out of ``flash.runner.supervise.lifecycle`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, replace

from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import JobSpec
from flash.runner.supervise import lifecycle as _lifecycle
from flash.runner.supervise.retry_decision import (
    _EXHAUSTED_CAPACITY_ACTION,
    _capacity_exhausted,
    _capacity_refusal_key,
    _retry_target,
)
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV


@dataclass
class _SubmitContext:
    spec: JobSpec
    log: object
    runtime_secrets: dict[str, str] | None
    source_snapshot: dict
    attempt_start: int
    infra_budget: int
    retry_budget: object
    started_with_shared_cache: bool
    last_handle: dict = field(default_factory=dict)
    current_gpu: dict = field(default_factory=dict)
    # persisted into the run handle so attach recovery keeps the same queue-capacity window.
    current_on_last_gpu: bool = False
    current_attempt: int = 0
    current_fence: int = 0
    # tracks complete attempt handles that registry-less gc cannot reconstruct by name.
    seen_endpoints: dict[str, dict] = field(default_factory=dict)
    submission_lock: object | None = None
    # grow only when an attempt actually provisioned a class and lost it to infra.
    failed_providers: set[str] = field(default_factory=set)
    tried_classes: set[tuple[str, str, int]] = field(default_factory=set)
    # how many times each shape has refused capacity, so a second refusal of the SAME one is a
    # repeat rather than a first look. counted per shape, not collected as a set of names: over
    # several classes a set says "everything has refused" while one of them has been asked once.
    # only capacity failures land here. provider loss says nothing about current rentability.
    capacity_refusals: dict[tuple[str, str, int], int] = field(default_factory=dict)
    oom_vram_floor: float = 0.0
    last_detail: str | None = None
    # sticky: once dropped stays dropped so all remaining attempts run on the unrestricted all-dc pool.
    drop_weight_cache: bool = False

    def on_handle(self, handle: dict) -> None:
        from flash.runner.accounting.reconciliation import _preserve_cleanup_remote
        from flash.runner.lifecycle.status import record_attempt_handle
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
            if canonical_handle["fence"] != self.current_fence:
                raise RuntimeError("provider handle fence does not match the reserved attempt")
            self.last_handle.clear()
            self.last_handle.update(canonical_handle)
            if canonical_handle.get("endpoint_id"):
                self.seen_endpoints[canonical_handle["endpoint_id"]] = dict(canonical_handle)
            persisted_handle = {
                **canonical_handle,
                "allocated_gpu": self.current_gpu.get("name"),
                # carried beside the gpu name for the same reason and by the same route: the
                # canonical provider handle drops unknown keys, so a recovering process can only
                # learn the shape from what was persisted here. without the count a run adopted
                # after a control-plane restart prices its wall as a single card.
                "allocated_gpu_count": self.current_gpu.get("count"),
                "on_last_gpu": bool(self.current_on_last_gpu),
            }
            if record_attempt_handle(
                self.spec.run_id,
                persisted_handle,
                attempt_id=self.current_attempt,
                fence=self.current_fence,
            ):
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
            self.release_submission_lock()

    def release_submission_lock(self) -> None:
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
        """Reap this run's tracked attempt endpoints before unwinding on cancel."""
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
    invocation_ordinal: int
    attempt: int
    fence: int
    attempt_spec: JobSpec
    runtime_secrets: dict[str, str]
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
    stop: bool = False


@dataclass(frozen=True)
class _FailureDecision:
    metrics: dict | None
    retry: bool


def _build_context(
    spec: JobSpec,
    log,
    runtime_secrets: dict[str, str] | None,
    source_snapshot: dict | None,
    attempt_start: int,
) -> _SubmitContext:
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status
    from flash.snapshot.archive import parse_descriptor

    source_snapshot = parse_descriptor(
        source_snapshot or source_snapshot_from_status(get_status(spec.run_id), required=True)
    ).to_dict()
    attempt_start = max(0, int(attempt_start))
    retry_budget, retry_placement = _lifecycle._load_durable_retry_state(spec)
    infra_budget = retry_budget.infra_retries
    started_with_shared_cache = (
        getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    )
    # environment transport is already pinned and staged by the controller before allocation.
    return _SubmitContext(
        spec=spec,
        log=log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        attempt_start=attempt_start,
        infra_budget=infra_budget,
        retry_budget=retry_budget,
        started_with_shared_cache=started_with_shared_cache,
        failed_providers=set(retry_placement.failed_providers),
        tried_classes=set(retry_placement.tried_classes),
        oom_vram_floor=retry_placement.oom_vram_floor,
        drop_weight_cache=retry_budget.cache_used > 0,
        current_attempt=attempt_start,
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
    with contextlib.suppress(Exception):
        metrics = _lifecycle._attempt_result_metrics(ctx.spec.run_id, ctx.last_handle)
        if metrics is not None:
            return metrics
    from flash.providers.core.base import JobHandle
    from flash.providers.core.registry import get_provider
    from flash.runner.accounting.reconciliation import (
        _compare_and_clear_remote,
        _record_cleanup_remote,
    )

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
            f"seed {ctx.spec.seed}: terminal worker's leaked endpoint cleanup target could not be persisted"
        )
    if worker_gone:
        if not _compare_and_clear_remote(ctx.spec.run_id, ctx.last_handle):
            raise RuntimeError(
                f"seed {ctx.spec.seed}: previous attempt's persisted remote changed before clear; "
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
        f"seed {ctx.spec.seed}: previous attempt's {ctx.last_handle.get('provider')} {resource_kind} "
        f"{resource_id} teardown could not be confirmed; failing to avoid "
        "double-provisioning a second worker over a possibly-live resource"
    )


def _mark_attempt_boundary(ctx: _SubmitContext, attempt: int) -> None:
    """Write the line that says everything above belongs to a previous attempt.

    The run log is one append-only file for the whole run, so a retry's output lands directly after
    the dead attempt's traceback with nothing in between. `flash runs log` tails that file, so while
    a replacement worker is booting the tail still ends in the OOM stack that caused the retry --
    an operator checking a run that is currently fine reads a failure. The status line already
    carries `attempt=` and `(prev attempt)`, but those come from lifecycle state and describe the run,
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


def _prepare_attempt(ctx: _SubmitContext, invocation_ordinal: int) -> _PreparationOutcome:
    from flash.runner.lifecycle.attempts import _reserve_attempt_record, _verified_opd_retry_state
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.errors import _RunCancelled
    from flash.server.platform.locks import _deploy_lock

    attempt = ctx.attempt_start + invocation_ordinal
    ctx.raise_if_cancelled()
    if invocation_ordinal > 0:
        completed_metrics = _cleanup_previous_attempt(ctx, attempt)
        if completed_metrics is not None:
            return _PreparationOutcome(completed_metrics=completed_metrics)
    try:
        attempt_spec = _spec_with_remaining_wall(ctx.spec, require_provider_minimum=True)
    except RuntimeError:
        ctx.gc_seen_endpoints()
        raise
    if ctx.spec.algorithm == "opd":
        expected_next_attempt, opd_resume_revision, resume_world_size = _verified_opd_retry_state(
            ctx.spec.run_id
        )
    else:
        expected_next_attempt, opd_resume_revision, resume_world_size = None, None, None
    ctx.submission_lock = _deploy_lock(ctx.spec.run_id)
    ctx.submission_lock.acquire()
    try:
        latest = get_status(ctx.spec.run_id)
        if latest.state in TERMINAL_STATES:
            raise ctx.cancel()
        if latest.remote:
            raise _RunCancelled(
                f"run {ctx.spec.run_id} already has a durable provider handle; not resubmitting"
            )
        attempt_record = _reserve_attempt_record(
            ctx.spec.run_id,
            minimum_attempt=ctx.attempt_start if invocation_ordinal == 0 else 0,
            expected_next_attempt=expected_next_attempt,
        )
        attempt = attempt_record.attempt_id
        ctx.current_attempt = attempt
        _mark_attempt_boundary(ctx, attempt)
        attempt_runtime_secrets = dict(ctx.runtime_secrets or {})
        attempt_runtime_secrets.pop(OPD_RESUME_REVISION_ENV, None)
        if opd_resume_revision is not None:
            attempt_runtime_secrets[OPD_RESUME_REVISION_ENV] = opd_resume_revision
        return _PreparationOutcome(
            prepared=_PreparedAttempt(
                invocation_ordinal,
                attempt,
                attempt_record.fence,
                attempt_spec,
                attempt_runtime_secrets,
                # only a pinned resume constrains the shape; without one the retry re-ranks freely.
                resume_world_size=resume_world_size if opd_resume_revision is not None else None,
            )
        )
    except BaseException:
        ctx.release_submission_lock()
        raise


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
    from flash.providers.core.registry import get_provider
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall
    from flash.runner.supervise.lifecycle import _spec_with_gpu

    candidates = tuple(_lifecycle._oom_escalated(allocation.candidates, ctx.oom_vram_floor))
    if not candidates:
        # an exhausted list has two causes and they need different words. attributing the pinned-
        # width one to OOM would send an operator to raise VRAM when the run is not out of memory
        # at all: no fitting candidate executes at the resume checkpoint's world size.
        if prepared.resume_world_size and not allocation.candidates:
            width = prepared.resume_world_size
            ctx.last_detail = (
                f"no candidate executing at checkpoint world size {width} is available, and this "
                "retry must preserve that executed rank width"
            )
            print(
                f"seed={ctx.spec.seed} no candidate executes at pinned OPD checkpoint world size "
                f"{width}; not retrying",
                file=ctx.log,
                flush=True,
            )
            return None
        ctx.last_detail = f"oom: exceeded the largest available GPU ({ctx.oom_vram_floor:g} GB)"
        print(
            f"seed={ctx.spec.seed} OOM on the largest GPU class ({ctx.oom_vram_floor:g} GB); not retrying",
            file=ctx.log,
            flush=True,
        )
        return None
    chosen = _lifecycle._select_candidate(candidates, ctx.failed_providers, ctx.tried_classes)
    untried = [c for c in candidates if _lifecycle._shape_key(c) not in ctx.tried_classes]
    cache_fallback_available = (
        ctx.retry_budget.cache_used < ctx.retry_budget.cache_fallbacks
        and ctx.started_with_shared_cache
        and not ctx.drop_weight_cache
        and chosen is not None
        and getattr(get_provider(chosen.provider), "supports_weight_cache", False)
    )
    on_last_gpu = len(untried) <= 1 or ctx.retry_budget.infra_exhausted(
        cache_fallback_available=cache_fallback_available
    )
    ctx.current_on_last_gpu = on_last_gpu
    print(allocation_summary(allocation), file=ctx.log, flush=True)
    if (chosen.provider, chosen.gpu) != (allocation.provider, allocation.gpu):
        print(
            f"retry {prepared.attempt}: walking past the cheapest class to {chosen.gpu} "
            f"@ {chosen.provider} ${chosen.hourly_usd:.2f}/hr",
            file=ctx.log,
            flush=True,
        )
    elif prepared.attempt and not untried:
        # every fitting class has been tried, so the picker re-selects the one that just
        # failed -- correct (never strand a run with no candidates), but silent: a
        # no_capacity retry then spends another full LAST_GPU_CAPACITY_GRACE_S waiting on
        # the same unavailable class. say so, because the operator's fix is to unpin
        # gpu.type rather than to keep waiting.
        print(
            f"retry {prepared.attempt}: no untried class left; re-selecting {chosen.gpu} "
            f"@ {chosen.provider}"
            + (" (gpu.type is pinned)" if getattr(ctx.spec.gpu, "type", None) else ""),
            file=ctx.log,
            flush=True,
        )
    effective_spec = _spec_with_gpu(ctx.spec, chosen.gpu, getattr(chosen, "gpu_count", 1))
    if ctx.drop_weight_cache:
        effective_spec = _lifecycle._drop_weight_cache(effective_spec)
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
    ctx.current_attempt = prepared.attempt
    ctx.current_fence = prepared.fence
    return _CandidatePlan(allocation, candidates, chosen, on_last_gpu, effective_spec, run_spec)


def _retry_delay(ctx: _SubmitContext, invocation_ordinal: int) -> float:
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at

    if invocation_ordinal >= ctx.infra_budget:
        return 0
    remaining = _load_run_deadline_at(ctx.spec.run_id) - _lifecycle.time.time()
    return min(10 * (invocation_ordinal + 1), remaining) if remaining > 0 else 0


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
                "fence": prepared.fence,
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
            return provider.submit_attempt(plan.run_spec, **submit_kwargs), False
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
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.lifecycle.status import get_status
    from flash.runner.lifecycle.submit import _persist_effective_worker_spec
    from flash.runner.supervise.errors import _RunCancelled

    retry_delay = 0.0
    candidate_not_started = False
    try:
        latest = get_status(ctx.spec.run_id)
        if latest.state in TERMINAL_STATES:
            raise ctx.cancel()
        if latest.remote:
            raise _RunCancelled(
                f"run {ctx.spec.run_id} already has a durable provider handle; not resubmitting"
            )
        # the accepted customer quote was frozen during preparation and is exactly the amount shown
        # by `flash train --cost`. allocation persists only the effective worker spec; live provider
        # rates and topology must never rewrite estimated_cost_usd after submission.
        if not _persist_effective_worker_spec(plan.effective_spec):
            raise ctx.cancel()
        if get_status(ctx.spec.run_id).state in TERMINAL_STATES:
            raise ctx.cancel()
        result, candidate_not_started = _submit_provider(ctx, prepared, plan)
        if candidate_not_started:
            retry_delay = _retry_delay(ctx, prepared.invocation_ordinal)
    finally:
        ctx.release_submission_lock()
    if retry_delay:
        _lifecycle.time.sleep(retry_delay)  # let the transient clear
    return result


def _run_attempt(ctx: _SubmitContext, prepared: _PreparedAttempt) -> _AttemptOutcome:
    try:
        allocation, result = _allocate_attempt(ctx, prepared)
        if allocation is None:
            return _AttemptOutcome(result=result)
        ctx.raise_if_cancelled()
        plan = _build_candidate_plan(ctx, prepared, allocation)
        if plan is None:
            return _AttemptOutcome(stop=True)
        result = _submit_candidate(ctx, prepared, plan)
        return _AttemptOutcome(
            result=result,
            chosen=plan.chosen,
            candidates=plan.candidates,
            run_spec=plan.run_spec,
        )
    finally:
        ctx.release_submission_lock()


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


def _return_success_metrics(ctx: _SubmitContext, outcome: _AttemptOutcome) -> dict:
    # a late worker success must not resurrect a cancelled run.
    ctx.raise_if_cancelled()
    _settle_terminal_remote(ctx)
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
    from flash.providers.core.registry import get_provider
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    # cancel wins over any retry-shaped failure.
    ctx.raise_if_cancelled()
    if ctx.last_handle:
        try:
            completed_metrics = _lifecycle._attempt_result_metrics(ctx.spec.run_id, ctx.last_handle)
        except Exception:
            completed_metrics = None
        if completed_metrics is not None:
            from flash.providers.core.base import PollResult

            recovered = replace(
                outcome,
                result=PollResult(True, metrics=completed_metrics),
            )
            return _FailureDecision(_return_success_metrics(ctx, recovered), False)
    result = outcome.result
    ctx.last_detail = f"{result.failure}: {result.detail}"
    if outcome.chosen is not None and result.failure in ("job_preempted", "oom"):
        # these outcomes happen after the class admitted the run, so an older no-capacity refusal no
        # longer describes the current market. poll_error stays ambiguous because submit and lookup
        # failures can happen before any capacity was granted.
        ctx.capacity_refusals.pop(_lifecycle._shape_key(outcome.chosen), None)
    oom_shaped = result.failure == "oom"
    if oom_shaped and outcome.chosen is not None:
        # same measure the filter compares against, see _candidate_usable_vram_gb
        ctx.oom_vram_floor = max(
            ctx.oom_vram_floor,
            _lifecycle._candidate_usable_vram_gb(outcome.chosen),
        )
    run_had_cache = bool(
        outcome.chosen is not None
        and getattr(get_provider(outcome.chosen.provider), "supports_weight_cache", False)
        and getattr(outcome.run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    )
    first_cache_drop = (
        run_had_cache
        and not ctx.drop_weight_cache
        and result.failure in ("no_capacity", "poll_error")
    )
    oom_mode = ctx.oom_vram_floor > 0
    will_retry = ctx.retry_budget.can_retry(
        result.failure,
        cache_drop=first_cache_drop,
    )
    capacity_exhausted = will_retry and _capacity_exhausted(
        ctx, outcome, first_cache_drop=first_cache_drop
    )
    if capacity_exhausted:
        will_retry = False
    # after the check, which asks whether this shape had refused BEFORE this attempt.
    #
    # in-memory and per-process on purpose: _build_context starts this empty every time, so a run
    # resumed after a control-plane restart gets a fresh pair of looks at the market rather than
    # inheriting a verdict from minutes ago. capacity is exactly the thing that changes in between.
    if result.failure == "no_capacity":
        refused_key = _capacity_refusal_key(ctx, outcome)
        if refused_key is not None:
            ctx.capacity_refusals[refused_key] = ctx.capacity_refusals.get(refused_key, 0) + 1
    durable_retry_state = None
    if result.failure in _lifecycle.RETRY_FAILURES:
        from flash.runner.lifecycle.retry_policy import consume_retry

        failed_providers = frozenset()
        tried_classes = frozenset()
        if not first_cache_drop and outcome.chosen is not None:
            tried_classes = frozenset({_lifecycle._shape_key(outcome.chosen)})
            if not oom_shaped:
                failed_providers = frozenset({outcome.chosen.provider})
        policy = consume_retry(
            ctx.spec.run_id,
            ctx.spec,
            expected_attempt=(prepared.attempt, prepared.fence),
            failure=result.failure,
            cache_drop=first_cache_drop,
            allow_retry=will_retry,
            failed_providers=failed_providers,
            tried_classes=tried_classes,
            oom_vram_floor=ctx.oom_vram_floor,
        )
        if policy is None:
            will_retry = False
        else:
            durable_retry_state = _lifecycle._retry_state_from_policy(ctx.spec, policy)
    retry_target = _retry_target(
        ctx,
        outcome,
        will_retry=will_retry,
        first_cache_drop=first_cache_drop,
    )
    action = (
        f"retrying on a larger GPU (> {ctx.oom_vram_floor:g} GB)"
        if (will_retry and oom_mode)
        else retry_target
        if will_retry
        else _EXHAUSTED_CAPACITY_ACTION
        if capacity_exhausted
        else "not retrying"
    )
    print(
        f"seed={ctx.spec.seed} attempt={prepared.attempt} failed ({result.failure}); {action}"
        f"\n--- failure detail ---\n{(result.detail or '')[:2000]}\n---",
        file=ctx.log,
        flush=True,
    )
    if not will_retry:
        return _FailureDecision(None, False)
    ctx.retry_budget, retry_placement = durable_retry_state
    ctx.failed_providers = set(retry_placement.failed_providers)
    ctx.tried_classes = set(retry_placement.tried_classes)
    ctx.oom_vram_floor = retry_placement.oom_vram_floor
    if first_cache_drop:
        ctx.drop_weight_cache = True
        # dropping the cache WIDENS the search: the weight-cache volume pins the run to the region
        # holding those weights, so every refusal so far answered "any capacity for this class in
        # that one region?" -- a narrower question than the cacheless retry is about to ask. Keeping
        # the tally would let one region's shortage plus a single blip in the unrestricted pool
        # reach two and stop the run, having heard the wider market refuse only once. Clear it so
        # the widened search gets its own pair of looks.
        ctx.capacity_refusals.clear()
    return _FailureDecision(None, True)


def run_attempts_supervised(
    spec: JobSpec,
    log,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    attempt_start: int = 0,
) -> dict:
    """Run one run through bounded attempts on infra-shaped failures."""
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
        attempt_start,
    )
    _require_opd_configuration(ctx)
    for invocation_ordinal in range(ctx.retry_budget.max_attempts):
        preparation = _prepare_attempt(ctx, invocation_ordinal)
        if preparation.completed_metrics is not None:
            return ctx.return_completed_runpod_metrics(preparation.completed_metrics)
        prepared = preparation.prepared
        outcome = _run_attempt(ctx, prepared)
        if outcome.stop:
            break
        if outcome.result.ok:
            return _return_success_metrics(ctx, outcome)
        decision = _handle_failure(ctx, prepared, outcome)
        if decision.metrics is not None:
            return decision.metrics
        if not decision.retry:
            break
    _settle_terminal_remote(ctx)
    ctx.gc_seen_endpoints()
    raise RuntimeError(
        f"run {spec.run_id} seed {spec.seed} failed after retries: {ctx.last_detail}"
    )
