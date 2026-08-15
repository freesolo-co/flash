"""RunPod endpoint deployment and job polling.

Split out of ``flash.providers.runpod.jobs`` to keep that module under the file-size limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flash.providers._lifecycle.deadline import (
    CREATE_ALLOWANCE_S,
    deadline_kwargs,
    remaining_seconds,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.poll import (
    PollErrorTracker,
    _attempt_int,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers.base import PollResult
from flash.providers.runpod import jobs as _jobs
from flash.providers.runpod.resources import WEIGHT_CACHE_GROW_BUDGET_S
from flash.providers.runpod.serverless import (
    WORKER_SYSTEM_DEPS,
    _train_body,
    resolve_worker_deps,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Job statuses that PROVE RunPod took the job out of the queue and therefore granted it a worker.
# Kept as an allowlist because the complement (`!= "IN_QUEUE"`) also matches None and any status
# this code does not recognise, which would let one flaky reading stand in as proof of a grant.
_GRANT_PROVING_STATUSES = frozenset(
    {"IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
)


@dataclass
class _DeployContext:
    """Deadline and cache-reconciliation state shared by deploy attempts."""

    deadline_at: float | None
    spec: Any
    cache_volumes: dict[str, int] | None
    reconciled: set[str]

    def reconciles_a_managed_cache(self) -> bool:
        """Whether a grow on this call can actually spend budget.

        Mirrors ``grow_weight_cache_volumes``'s own early return: a run attaching no managed cache
        reconciles nothing, so reserving for it would shorten the deadline for a create that was
        never going to grow anything.
        """
        if self.cache_volumes is not None:
            return bool(self.cache_volumes)
        from flash.runner import WEIGHT_CACHE_VOLUME_NAME

        base = getattr(self.spec.gpu, "network_volume", None) if self.spec is not None else None
        return str(base or "") == WEIGHT_CACHE_VOLUME_NAME

    def create_deadline(self, key_count: int) -> float | None:
        """``deadline_at`` less the grow budget still owed to unreconciled accounts.

        Creates, sweeps, and backoffs cannot spend this reserve. Any attempt reaching create has
        reconciled; cache-free runs reserve nothing. See ``require_launchable`` for admission.
        """
        if self.deadline_at is None or not self.reconciles_a_managed_cache():
            return self.deadline_at
        owed = max(0, key_count - len(self.reconciled))
        return self.deadline_at - WEIGHT_CACHE_GROW_BUDGET_S * owed

    def attempt_deadline(self, key_count: int, active_key: str | None) -> float | None:
        """``deadline_at`` less the ONE grow slice this attempt's reconciliation can need.

        Admission funds only this attempt's grow. Once its account reconciles, do not charge the
        slice again; before selection, reserve conservatively.
        """
        if self.deadline_at is None or not self.reconciles_a_managed_cache():
            return self.deadline_at
        if active_key is not None and active_key in self.reconciled:
            return self.deadline_at
        if key_count <= len(self.reconciled):
            return self.deadline_at
        return self.deadline_at - WEIGHT_CACHE_GROW_BUDGET_S

    def require_launchable(self, key_count: int, active_key: str | None) -> None:
        """Fail closed unless this attempt can still reconcile itself and then create.

        Raw-deadline admission can yield zero growth and mount a stale volume. Reserving one slice
        prevents the later "Disk quota exceeded" failure.
        """
        if self.deadline_at is not None:
            require_create_allowance(self.attempt_deadline(key_count, active_key))


def _prepare_quota_retry(
    context: _DeployContext,
    name: str,
    quota_attempt: int,
    quota_max_retries: int,
    key_count: int,
    active_key: str | None,
) -> None:
    """Sweep safe idle endpoints and back off before a quota retry."""
    # under quota pressure, reap only scaled-to-zero orphans on this account.
    # `reap_warm=False` protects live runs' between-job workers; reserve failover growth
    # from this sweep and backoff.
    context.require_launchable(key_count, active_key)
    quota_deadline = context.create_deadline(key_count)
    swept = _jobs._sweep_idle_flash_endpoints(
        protected={_jobs.canonical_endpoint_name(name)},
        min_idle_s=0.0,
        reap_warm=False,
        **deadline_kwargs(_jobs._sweep_idle_flash_endpoints, quota_deadline),
    )
    wait_s = 30 * quota_attempt
    if context.deadline_at is not None:
        wait_s = min(wait_s, remaining_seconds(quota_deadline))
    _jobs.logger.warning(
        "RunPod worker quota hit (attempt %d/%d): swept %d idle flash-* endpoint(s); "
        "retrying in %ds",
        quota_attempt + 1,
        quota_max_retries,
        swept,
        wait_s,
    )
    if wait_s > 0:
        _jobs.time.sleep(wait_s)


def _deploy_with_failover(
    context: _DeployContext,
    name: str,
    deploy_once: Callable[[], tuple[object, str]],
    rp_keys: Any,
) -> tuple[object, str]:
    """Retry quota failures and walk configured accounts without losing cleanup guards."""
    quota_max_retries = 3
    resource = None
    owning_fingerprint = None
    # bound by count, not advance_key() return value: advance_key() always wraps so can't signal exhaustion.
    failovers_left = max(0, rp_keys.key_count() - 1)
    while resource is None:
        deploy_failover_exc: Exception | None = None
        for quota_attempt in range(quota_max_retries):
            if quota_attempt > 0:
                _prepare_quota_retry(
                    context,
                    name,
                    quota_attempt,
                    quota_max_retries,
                    rp_keys.key_count(),
                    rp_keys.active_key(),
                )
            try:
                resource, owning_fingerprint = deploy_once()
                break  # success
            except Exception as exc:
                if _jobs._is_balance_error(exc):
                    # a broke account can't be helped by sweeping idle endpoints: fail over now
                    deploy_failover_exc = exc
                    break
                if not _jobs._is_workers_quota_error(exc):
                    raise
                deploy_failover_exc = exc
        if resource is not None:
            break
        if failovers_left > 0:
            with _jobs.FLASH_SDK_LOCK:
                rp_keys.advance_key()
            failovers_left -= 1
            reason = (
                "has insufficient balance"
                if deploy_failover_exc is not None and _jobs._is_balance_error(deploy_failover_exc)
                else "worker quota exhausted after sweeping"
            )
            _jobs.logger.warning(
                "RunPod account %s; failing over to the next RUNPOD_API_KEY account (%d configured)",
                reason,
                rp_keys.key_count(),
            )
            continue
        raise deploy_failover_exc or RuntimeError(
            "deploy_train_endpoint: deploy failover exhausted"
        )
    return resource, owning_fingerprint


def deploy_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
    endpoint_kwargs: dict | Callable[[], dict] | None = None,
    deadline_at: float | None = None,
    cache_volumes: dict[str, int] | None = None,
) -> tuple[str, str, str]:
    """Deploy a uniquely-named worker endpoint and return its id, name, and owning fingerprint.

    Rebuild callable `endpoint_kwargs` per account so failover cannot reuse another account's
    volume id. `cache_volumes` supplies managed sizes when `spec` is absent.
    """
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.resource_manager import ResourceManager

    from flash.core.spec import gpu_count_of
    from flash.providers.runpod import auth as rp_keys
    from flash.providers.runpod.auth import ensure_auth

    _jobs._patch_runpod_backoff()
    friendly = _jobs.canonical_gpu(friendly_gpu)
    name = _jobs.endpoint_name(friendly, name_suffix)
    image = _jobs.worker_image_for_gpu(friendly, allow_default=True)
    context = _DeployContext(deadline_at, spec, cache_volumes, set())

    def _deploy_once() -> tuple[object, str]:
        """Create under one serialized account selection and return its owning fingerprint."""
        context.require_launchable(rp_keys.key_count(), rp_keys.active_key())
        with _jobs.FLASH_SDK_LOCK:
            owning_key = ensure_auth()
            # re-check the selected key under the lock: another deploy may advance the global key
            # after admission. an unreconciled real account must retain its grow slice or fail
            # closed.
            context.require_launchable(rp_keys.key_count(), owning_key)
            owning_fingerprint = _jobs.runpod_api.key_fingerprint(owning_key)
            _jobs.isolate_flash_state(name_suffix)
            kwargs = {
                "name": name,
                "gpu": _jobs.flash_gpu(friendly),
                # one worker occupies gpu.count cards of this class; count == 1 is the historical path.
                "gpu_count": gpu_count_of(spec),
                "min_cuda_version": _jobs.min_cuda_for(friendly),
                "execution_timeout_ms": execution_timeout_ms or _jobs.DEFAULT_EXECUTION_TIMEOUT_MS,
                "workers": (0, 1),
            }
            if image:
                kwargs["image"] = image
            else:
                kwargs["dependencies"] = resolve_worker_deps()
                kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
            # reconcile the selected account before attach because the SDK returns existing volumes
            # at their old size. do this once per account and preserve the later create allowance.
            if owning_key not in context.reconciled:
                context.reconciled.add(owning_key)
                # spends against the raw deadline: this account's slice of the reserve is released
                # by the line above, so the grow draws its own budget and still yields the create
                # allowance. the slices of the accounts not yet reconciled stay held back.
                _jobs.grow_weight_cache_volumes(spec, owning_key, deadline_at, wanted=cache_volumes)
            # re-invoke factory per account (avoids reusing a volume id stamped for the prior account).
            override = endpoint_kwargs() if callable(endpoint_kwargs) else endpoint_kwargs
            kwargs.update(
                override if override is not None else _jobs.weight_cache_endpoint_kwargs(spec)
            )
            ep = Endpoint(**kwargs)
            ep._qb_target = _train_body
            config = ep._build_resource_config()
            _jobs.apply_disk_gb(config, disk_gb)
            rm = ResourceManager()
            if deadline_at is None:
                resource = _jobs.asyncio.run(rm.get_or_deploy_resource(config))
            else:
                # this attempt's own grow has run by now (the reconciled check above), so the
                # re-check is judged without its slice: charging it again would re-deduct a cost
                # already paid and reject a launchable create within one slice of the deadline.
                context.require_launchable(rp_keys.key_count(), owning_key)
                create_deadline = context.create_deadline(rp_keys.key_count())
                # create may spend only down to the failover grow reserve. yield the proven create
                # allowance so a large reserve cannot produce a zero timeout.
                remaining = max(CREATE_ALLOWANCE_S, remaining_seconds(create_deadline))
                resource = _jobs.asyncio.run(
                    _jobs.asyncio.wait_for(
                        rm.get_or_deploy_resource(config),
                        timeout=remaining,
                    )
                )
            return resource, owning_fingerprint

    resource, owning_fingerprint = _deploy_with_failover(context, name, _deploy_once, rp_keys)
    endpoint_id = getattr(resource, "id", None)
    if not endpoint_id:
        raise RuntimeError(f"deploy_train_endpoint: no endpoint id on resource {resource!r}")
    if owning_fingerprint is None:
        raise RuntimeError("deploy_train_endpoint: owning RunPod key is unavailable")
    return endpoint_id, name, owning_fingerprint


@dataclass(frozen=True)
class _PollContext:
    """Values that remain fixed throughout one job poll."""

    handle: Any
    say: Callable[[str], None]
    interval_s: float
    heartbeat_reader: Any
    failure_detail_reader: Any
    stall_after_s: float
    setup_grace_s: float
    unhealthy_grace_s: float
    throttled_grace_s: float
    queue_grace_s: float
    absolute_deadline: float | None
    current_attempt: int
    launch_ts: float
    next_gpu_note: str
    poll_errors: PollErrorTracker


@dataclass
class _PollState:
    """Mutable observations threaded through the polling loop."""

    last_status: Any
    last_hb_key: Any
    last_hb_ts: float
    last_hb_attempt: int
    last_progress: float
    seen_training_hb: bool
    last_health_probe: float
    unhealthy_timer: Any
    throttled_timer: Any
    queued_timer: Any
    worker_coming_up_at: float | None
    # latched once RunPod has ever granted a worker. `worker_coming_up_at` is a TTL'd sighting and
    # so goes false again on any health gap; this never does. The queued-wait stall exemption keys
    # on it so a flapping health read cannot keep rearming the cold-start budget.
    ever_saw_worker: bool


def _wall_deadline_result(context: _PollContext) -> PollResult | None:
    if context.absolute_deadline is not None and _jobs.time.time() >= context.absolute_deadline:
        return PollResult(False, failure="stalled", detail="run wall deadline exceeded")
    return None


def _read_job_status(context: _PollContext) -> tuple[dict | None, PollResult | None]:
    """Read one provider status; neither value means a transient error should be retried."""
    try:
        status = _jobs.runpod_api.job_status(
            context.handle.endpoint_id,
            context.handle.job_id,
            key_fingerprint=context.handle.key_fingerprint,
            **deadline_kwargs(_jobs.runpod_api.job_status, context.absolute_deadline),
        )
        context.poll_errors.reset()
    except _jobs.runpod_api.RunpodApiError as exc:
        if context.poll_errors.record(exc, deadline_at=context.absolute_deadline):
            return None, PollResult(False, failure="poll_error", detail=str(exc))
        return None, None
    return status, _wall_deadline_result(context)


def _classify_terminal_status(
    context: _PollContext,
    state: _PollState,
    provider_status: dict,
    status: Any,
) -> PollResult | None:
    """Return a terminal result, or None when polling must continue."""
    if status in _jobs.TERMINAL_OK:
        # read heartbeats already committed at the wall deadline (deadline_at=None) so a job that
        # completes right at the boundary still surfaces its final per-step metrics for `flash runs log -f`
        forced_reader = (
            (lambda: context.heartbeat_reader(force=True, deadline_at=None))
            if context.heartbeat_reader is not None
            else None
        )
        state.last_hb_key, _ = surface_heartbeat(forced_reader, state.last_hb_key, context.say)
        try:
            return PollResult(True, metrics=_jobs.decode_output(provider_status.get("output")))
        except RuntimeError as exc:
            state.last_hb_key, retriable, oom = _jobs.surfaced_worker_flags(
                context.heartbeat_reader,
                state.last_hb_key,
                context.say,
                context.current_attempt,
                launch_ts=context.launch_ts,
            )
            detail = _jobs._append_failure_artifacts(str(exc), context.failure_detail_reader)
            return PollResult(
                False,
                failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
                detail=detail,
            )
    if status not in _jobs.TERMINAL_FAIL:
        return None
    # this detail reaches the user-readable run log, so every part of it is sanitized before its
    # tail is selected: a control-plane secret echoed by the worker would otherwise be printed
    # verbatim, and slicing first could cut a credential at the boundary so its surviving part no
    # longer value-matches (the instance providers sanitize each part of theirs the same way).
    detail = _jobs._safe_failure_text(provider_status.get("error") or "", 1500)
    output = provider_status.get("output")
    if isinstance(output, dict) and output.get("stdout"):
        detail += "\n--- worker stdout tail ---\n" + _jobs._safe_failure_text(output["stdout"])
    elif not detail:
        detail = _jobs._safe_failure_text(output, 1500)
    if status in _jobs.PLATFORM_TERMINATIONS:
        return PollResult(False, failure="job_preempted", detail=f"[{status}] {detail}")
    state.last_hb_key, retriable, oom = _jobs.surfaced_worker_flags(
        context.heartbeat_reader,
        state.last_hb_key,
        context.say,
        context.current_attempt,
        launch_ts=context.launch_ts,
    )
    detail = _jobs._append_failure_artifacts(detail, context.failure_detail_reader)
    return PollResult(
        False,
        failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
        detail=f"[{status}] {detail}",
    )


def _worker_is_coming_up(at: float | None, now: float) -> bool:
    """Whether a worker sighting at ``at`` is recent enough to suppress the capacity timer."""
    return at is not None and (now - at) <= _jobs.WORKER_COMING_UP_TTL_S


def _probe_worker_coming_up_at(context: _PollContext, now: float) -> float | None:
    """One health read answering only "has runpod granted a worker yet".

    Do not update continuous unhealthy/throttled timers here. A failed read returns None and
    leaves existing evidence unchanged.
    """
    try:
        health = _jobs.runpod_api.endpoint_health_for_fingerprint(
            context.handle.endpoint_id,
            context.handle.key_fingerprint,
            **deadline_kwargs(
                _jobs.runpod_api.endpoint_health_for_fingerprint,
                context.absolute_deadline,
            ),
        )
    except Exception:
        return None
    workers = health.get("workers") or {}
    usable = workers.get("running") or workers.get("ready") or workers.get("idle")
    return now if (usable or workers.get("initializing")) else None


def _classify_queue_state(
    context: _PollContext,
    state: _PollState,
    status: Any,
    now: float,
) -> PollResult | None:
    """Return a sustained queue-health failure, or None while the wait remains viable."""
    coming_up = _worker_is_coming_up(state.worker_coming_up_at, now)
    would_expire = (
        status == "IN_QUEUE"
        and not coming_up
        and state.queued_timer.since is not None
        and now - state.queued_timer.since > context.queue_grace_s
    )
    if would_expire:
        # the last worker reading may be 90s stale, so probe once before abandoning a GPU RunPod
        # may already have granted. the existing TTL prevents repeated boundary probes.
        state.last_health_probe = now
        coming_up = _worker_is_coming_up(_probe_worker_coming_up_at(context, now), now)
        if coming_up:
            state.worker_coming_up_at = now
            state.ever_saw_worker = True
    if state.queued_timer.expired(
        status == "IN_QUEUE" and not coming_up, now, context.queue_grace_s
    ):
        return PollResult(
            False,
            failure="no_capacity",
            detail=f"never scheduled: job stuck IN_QUEUE for "
            f"{int(now - state.queued_timer.since)}s "
            f"(no RunPod capacity for the pinned GPU class); {context.next_gpu_note}",
        )
    if status != "IN_QUEUE":
        # the in-queue grace timers measure continuous throttle/unhealthy while queued (like
        # queued_timer, driven every iteration above); reset them whenever the job leaves the
        # queue so a re-queue after an IN_PROGRESS spell doesn't fire on a stale arm time.
        state.unhealthy_timer.since = None
        state.throttled_timer.since = None
        return None
    if now - state.last_health_probe <= 90:
        return None
    state.last_health_probe = now
    try:
        health = _jobs.runpod_api.endpoint_health_for_fingerprint(
            context.handle.endpoint_id,
            context.handle.key_fingerprint,
            **deadline_kwargs(
                _jobs.runpod_api.endpoint_health_for_fingerprint,
                context.absolute_deadline,
            ),
        )
        workers = health.get("workers") or {}
        usable = workers.get("running") or workers.get("ready") or workers.get("idle")
        recovering = workers.get("initializing")
        # runpod granted the gpu the moment a worker is initializing or usable, so from
        # here on the wait is startup, not capacity starvation. stamped rather than a bare
        # flag: a probe that starts failing must not leave a stale true suppressing the
        # capacity timer forever, so it expires after WORKER_COMING_UP_TTL_S.
        state.worker_coming_up_at = now if (usable or recovering) else None
        if usable or recovering or workers.get("unhealthy"):
            # an unhealthy worker is an ALLOCATED box whose image failed to start, so it proves the
            # grant just as much as a healthy one -- `preload_runpod._has_worker` counts it for the
            # same reason. it deliberately does not feed `worker_coming_up_at` above: that
            # suppresses the capacity timer for a worker that is coming up, which an unhealthy one
            # is not. this only records that capacity was never the problem, so health flickering
            # between unhealthy and empty cannot make a broken box look like starvation.
            state.ever_saw_worker = True
        if any(workers.get(k) for k in ("throttled", "unhealthy", "initializing")) or not usable:
            message = f"queued; workers: {workers}"
            if not usable and not recovering and state.queued_timer.since is not None:
                elapsed_s = max(0, int(now - state.queued_timer.since))
                message += f"; waited {elapsed_s}s of {int(context.queue_grace_s)}s capacity grace"
            context.say(message)
        if state.unhealthy_timer.expired(
            workers.get("unhealthy") and not usable and not recovering,
            now,
            context.unhealthy_grace_s,
        ):
            return PollResult(
                False,
                failure="stalled",
                detail=f"worker stuck unhealthy for "
                f"{int(now - state.unhealthy_timer.since)}s while IN_QUEUE (likely a failed "
                f"image pull); retrying on a fresh endpoint",
            )
        if state.throttled_timer.expired(
            workers.get("throttled") and not usable and not recovering,
            now,
            context.throttled_grace_s,
        ):
            return PollResult(
                False,
                failure="no_capacity",
                detail=f"never scheduled: worker stuck THROTTLED for "
                f"{int(now - state.throttled_timer.since)}s while IN_QUEUE (no RunPod "
                f"capacity for the pinned GPU class); {context.next_gpu_note}",
            )
    except Exception:
        pass
    return None


def _update_heartbeat(context: _PollContext, state: _PollState) -> None:
    new_key, stage = surface_heartbeat(context.heartbeat_reader, state.last_hb_key, context.say)
    if new_key == state.last_hb_key:
        return
    state.last_hb_key = new_key
    hb_attempt = _attempt_int(new_key[3])
    if hb_attempt != context.current_attempt:
        # non-current heartbeat: ignore so stale progress never tightens the stall window, and so a
        # previous attempt's worker -- which ran on an allocation this attempt no longer holds --
        # never stands as proof that THIS attempt was granted one.
        return
    # a heartbeat for THIS attempt was written by a worker that ran, so it proves a grant on its own.
    # that matters on the reattach path: recovery starts with `ever_saw_worker` false and, if the job
    # was already requeued before attaching, never gets to observe the earlier IN_PROGRESS. with
    # health also unreadable the job would look never-granted, stay exempt from the stall check, and
    # finally be reported `no_capacity` despite having demonstrably run.
    #
    # deliberately ABOVE the `stage is None` return: that return drops liveness pings, which are most
    # of what setup publishes (`sft_model_load`, `*_data_loading`, `*_configuring`). A ping proves a
    # worker ran just as well as a staged heartbeat -- it must not advance the stall clock, which is
    # why it still returns below, but it is evidence of a grant, and only the grant question is
    # settled here.
    state.ever_saw_worker = True
    if stage is None:
        return
    hb_ts = new_key[2]
    hb_step = new_key[1]
    is_training_hb = is_training_heartbeat(stage, hb_step)
    if hb_attempt > state.last_hb_attempt:
        # fresh attempt: reset ts baseline and re-derive seen_training_hb so cold-start grace rearms.
        state.last_hb_attempt = hb_attempt
        state.last_hb_ts = hb_ts or 0.0
        state.last_progress = _jobs.time.time()
        state.seen_training_hb = is_training_hb
    elif hb_attempt == state.last_hb_attempt and (hb_ts is None or hb_ts > state.last_hb_ts):
        # gate progress on ts advancing: a stale late upload must not buy a fresh stall window.
        if hb_ts is not None:
            state.last_hb_ts = hb_ts
        state.last_progress = _jobs.time.time()
        if is_training_hb:
            state.seen_training_hb = True


def _classify_stall(context: _PollContext, state: _PollState, status: Any) -> PollResult | None:
    in_setup = context.heartbeat_reader is not None and not state.seen_training_hb
    stall_limit = context.setup_grace_s if in_setup else context.stall_after_s
    # a job still IN_QUEUE with no worker granted is not stalled: nothing is running that could
    # make progress, and _classify_queue_state above already owns this wait on the capacity grace
    # (returning `no_capacity`, the failure the supervisor routes on). The two limits are
    # independent, so a capacity grace scaled past the stall limit -- a 4-card shape waits 3600s
    # against a 3000s setup grace -- would otherwise be cut short HERE by a timer measuring the
    # absence of a worker that was never granted, mislabelled "stalled", and re-requesting the same
    # class exactly as before. Defer to the capacity timer for this state instead of racing it.
    #
    # Only for the no-worker case: once health shows one coming up, the queue timer is suppressed
    # and the setup grace legitimately governs the cold start, unscaled, as it always did.
    now = _jobs.time.time()
    if (
        status == "IN_QUEUE"
        and not state.ever_saw_worker
        and not _worker_is_coming_up(state.worker_coming_up_at, now)
    ):
        # The wait is exempt, so the clock this function measures has to be exempt with it: it
        # anchors on the last status CHANGE, which for a job queued from the start is the moment it
        # entered the queue. Leaving it there would bank the whole queued wait against the cold
        # start, so a worker granted late -- after 3000s of queueing, but inside a 4-card's 3600s
        # capacity grace -- would be declared stalled on its very first poll, having been given no
        # time to boot at all. Roll the anchor forward while the exemption holds so the grant
        # starts the cold-start budget from zero, exactly as an immediate grant does.
        #
        # Strictly PRE-grant, hence `ever_saw_worker` gating the branch itself rather than just the
        # re-anchor. `worker_coming_up_at` is a TTL'd sighting that goes false again on any health
        # gap, so exempting on it alone means a worker granted and then lost -- health reporting it
        # once, then empty -- keeps skipping the stall check forever. The queue timer (rearmed by
        # the same gap) then runs to the scaled capacity grace and reports `no_capacity` for a GPU
        # that WAS granted, which is both the wrong limit and the wrong label: it can trip the
        # supervisor's weight-cache drop on a run that never had a capacity problem. Past the first
        # grant every observation belongs to the setup timer.
        state.last_progress = now
        return None
    if now - state.last_progress <= stall_limit:
        return None
    phase = "setup (pre-training)" if in_setup else "training"
    return PollResult(
        False,
        failure="stalled",
        detail=f"no worker progress for {int(now - state.last_progress)}s "
        f"during {phase} (job status {status}, limit {int(stall_limit)}s)",
    )


def _sleep_until_next_poll(context: _PollContext) -> None:
    delay = context.interval_s
    if context.absolute_deadline is not None:
        remaining = remaining_seconds(context.absolute_deadline)
        if remaining <= 0:
            return
        delay = min(delay, remaining)
    if delay > 0:
        _jobs.time.sleep(delay)


def poll_job(
    handle,
    log=None,
    interval_s: float = 10.0,
    heartbeat_reader=None,
    failure_detail_reader=None,
    stall_after_s: float = 1200.0,
    setup_grace_s: float = 3000.0,
    unhealthy_grace_s: float = 240.0,
    throttled_grace_s: float = 300.0,
    queue_grace_s: float = 300.0,
    deadline_at: float | None = None,
    current_attempt: int | None = None,
    on_last_gpu: bool = False,
) -> PollResult:
    """Poll a queue job to completion; resilient to transient API errors.

    Use setup grace before the first heartbeat, then stall grace; fail fast on sustained throttled,
    unhealthy, or over-queued states. `on_last_gpu` states only whether escalation remains.
    """
    if not handle.job_id:
        raise ValueError("endpoint-only RunPod handles cannot be polled")
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    launch_ts = handle.started_ts
    if not _jobs.math.isfinite(launch_ts) or launch_ts <= 0:
        raise ValueError("persisted RunPod launch timestamp is invalid")
    attempt_id = handle.attempt if current_attempt is None else _attempt_int(current_attempt)
    if attempt_id is None or attempt_id != handle.attempt:
        raise ValueError("RunPod poll attempt identity does not match the persisted handle")
    say = make_say(log)
    context = _PollContext(
        handle=handle,
        say=say,
        interval_s=interval_s,
        heartbeat_reader=heartbeat_reader,
        failure_detail_reader=failure_detail_reader,
        stall_after_s=stall_after_s,
        setup_grace_s=setup_grace_s,
        unhealthy_grace_s=unhealthy_grace_s,
        throttled_grace_s=throttled_grace_s,
        queue_grace_s=queue_grace_s,
        absolute_deadline=absolute_deadline,
        current_attempt=attempt_id,
        launch_ts=launch_ts,
        next_gpu_note=_jobs.capacity_escalation_note(on_last_gpu),
        poll_errors=PollErrorTracker(say, interval_s),
    )
    state = _PollState(
        last_status=None,
        last_hb_key=None,
        last_hb_ts=0.0,
        # -1 sentinel < any real attempt; gates out prior-attempt leftover heartbeats
        last_hb_attempt=-1,
        last_progress=_jobs.time.time(),
        seen_training_hb=False,
        last_health_probe=0.0,
        unhealthy_timer=_jobs.GraceTimer(),
        throttled_timer=_jobs.GraceTimer(),
        queued_timer=_jobs.GraceTimer(),
        # a runpod job stays IN_QUEUE for the whole worker cold start, image pull included, so the
        # queue timer alone cannot tell "runpod never gave us the gpu" from "the gpu arrived and we
        # are still pulling a multi-GB image". the unhealthy/throttled timers already refuse to fire
        # while a worker is initializing or usable; carry that observation forward so the capacity
        # timer gets it too, and a heavy image can never self-report as no_capacity.
        worker_coming_up_at=None,
        ever_saw_worker=False,
    )
    while True:
        terminal = _wall_deadline_result(context)
        if terminal is not None:
            return terminal
        provider_status, terminal = _read_job_status(context)
        if terminal is not None:
            return terminal
        if provider_status is None:
            continue
        status = provider_status.get("status")
        if status in _GRANT_PROVING_STATUSES:
            # leaving the queue proves RunPod granted a worker, whatever health says. the two
            # health-derived latch sites go quiet when the health endpoint is unreachable (both
            # swallow their errors), and RunPod can requeue a job that already ran -- which would
            # otherwise leave the requeued job looking never-granted, exempt from the setup-stall
            # check forever.
            #
            # an allowlist, never `!= "IN_QUEUE"`: that shape also matches None and any unrecognized
            # string, so a single flaky job_status response would permanently "prove" a grant, drop
            # the queued-wait exemption, and let a genuine capacity wait die as `stalled` -- losing
            # the weight-cache fallback that only `no_capacity` triggers. `preload_runpod.py` hit
            # this first and its comment warns against exactly this pattern.
            state.ever_saw_worker = True
        if status != state.last_status:
            say(f"job {handle.job_id}: {status}")
            state.last_status = status
            state.last_progress = _jobs.time.time()
        terminal = _classify_terminal_status(context, state, provider_status, status)
        if terminal is not None:
            return terminal
        now = _jobs.time.time()
        terminal = _classify_queue_state(context, state, status, now)
        if terminal is not None:
            return terminal
        _update_heartbeat(context, state)
        terminal = _classify_stall(context, state, status)
        if terminal is not None:
            return terminal
        _sleep_until_next_poll(context)
