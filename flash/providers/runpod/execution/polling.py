"""RunPod queue-job polling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flash.providers._lifecycle.instances.poll import (
    PollErrorTracker,
    _attempt_int,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
    remaining_seconds,
    require_deadline_at,
)
from flash.providers.core.base import PollResult
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.execution.jobs import (
    PLATFORM_TERMINATIONS,
    TERMINAL_FAIL,
    TERMINAL_OK,
    WORKER_COMING_UP_TTL_S,
    GraceTimer,
    _append_failure_artifacts,
    _safe_failure_text,
    decode_output,
    surfaced_worker_flags,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# statuses that prove RunPod granted a worker rather than merely queueing the job.
_GRANT_PROVING_STATUSES = frozenset(
    {"IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
)


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
    if context.absolute_deadline is not None and time.time() >= context.absolute_deadline:
        return PollResult(False, failure="stalled", detail="run wall deadline exceeded")
    return None


def _read_job_status(context: _PollContext) -> tuple[dict | None, PollResult | None]:
    """Read one provider status; neither value means a transient error should be retried."""
    try:
        status = runpod_api.job_status(
            context.handle.endpoint_id,
            context.handle.job_id,
            key_fingerprint=context.handle.key_fingerprint,
            **deadline_kwargs(runpod_api.job_status, context.absolute_deadline),
        )
        context.poll_errors.reset()
    except runpod_api.RunpodApiError as exc:
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
    if status in TERMINAL_OK:
        # read heartbeats already committed at the wall deadline (deadline_at=None) so a job that
        # completes right at the boundary still surfaces its final per-step metrics for `flash runs log -f`
        forced_reader = (
            (lambda: context.heartbeat_reader(force=True, deadline_at=None))
            if context.heartbeat_reader is not None
            else None
        )
        state.last_hb_key, _ = surface_heartbeat(forced_reader, state.last_hb_key, context.say)
        try:
            return PollResult(True, metrics=decode_output(provider_status.get("output")))
        except RuntimeError as exc:
            output = provider_status.get("output")
            output_retriable = isinstance(output, dict) and output.get("_flash_retriable") is True
            state.last_hb_key, retriable, oom = surfaced_worker_flags(
                context.heartbeat_reader,
                state.last_hb_key,
                context.say,
                context.current_attempt,
                launch_ts=context.launch_ts,
            )
            detail = _append_failure_artifacts(str(exc), context.failure_detail_reader)
            return PollResult(
                False,
                failure=(
                    "oom"
                    if oom
                    else ("job_preempted" if retriable or output_retriable else "job_failed")
                ),
                detail=detail,
            )
    if status not in TERMINAL_FAIL:
        return None
    # this detail reaches the user-readable run log, so every part of it is sanitized before its
    # tail is selected: a control-plane secret echoed by the worker would otherwise be printed
    # verbatim, and slicing first could cut a credential at the boundary so its surviving part no
    # longer value-matches (the instance providers sanitize each part of theirs the same way).
    detail = _safe_failure_text(provider_status.get("error") or "", 1500)
    output = provider_status.get("output")
    if isinstance(output, dict) and output.get("stdout"):
        detail += "\n--- worker stdout tail ---\n" + _safe_failure_text(output["stdout"])
    elif not detail:
        detail = _safe_failure_text(output, 1500)
    if status in PLATFORM_TERMINATIONS:
        return PollResult(False, failure="job_preempted", detail=f"[{status}] {detail}")
    state.last_hb_key, retriable, oom = surfaced_worker_flags(
        context.heartbeat_reader,
        state.last_hb_key,
        context.say,
        context.current_attempt,
        launch_ts=context.launch_ts,
    )
    detail = _append_failure_artifacts(detail, context.failure_detail_reader)
    return PollResult(
        False,
        failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
        detail=f"[{status}] {detail}",
    )


def _worker_is_coming_up(at: float | None, now: float) -> bool:
    """Whether a worker sighting at ``at`` is recent enough to suppress the capacity timer."""
    return at is not None and (now - at) <= WORKER_COMING_UP_TTL_S


def _probe_worker_coming_up_at(context: _PollContext, now: float) -> float | None:
    """One health read answering only "has runpod granted a worker yet".

    Do not update continuous unhealthy/throttled timers here. A failed read returns None and
    leaves existing evidence unchanged.
    """
    try:
        health = runpod_api.endpoint_health_for_fingerprint(
            context.handle.endpoint_id,
            context.handle.key_fingerprint,
            **deadline_kwargs(
                runpod_api.endpoint_health_for_fingerprint,
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
            "(no RunPod capacity for the pinned GPU class)",
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
        health = runpod_api.endpoint_health_for_fingerprint(
            context.handle.endpoint_id,
            context.handle.key_fingerprint,
            **deadline_kwargs(
                runpod_api.endpoint_health_for_fingerprint,
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
            if (
                not usable
                and not recovering
                and not state.ever_saw_worker
                and state.queued_timer.since is not None
            ):
                elapsed_s = max(0, int(now - state.queued_timer.since))
                budget = (
                    f"{context.queue_grace_s:g}s"
                    if math.isfinite(context.queue_grace_s)
                    else "unbounded"
                )
                message += f"; waited {elapsed_s}s of {budget} capacity grace"
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
                f"{int(now - state.throttled_timer.since)}s while IN_QUEUE "
                "(no RunPod capacity for the pinned GPU class)",
            )
    except Exception:
        pass
    return None


def _progress_ts(context: _PollContext, hb_key: tuple) -> float:
    """The progress this heartbeat proves: its own ts, clamped to [launch, now].

    The companion ``fresh`` flag is deliberately dropped. Its attempt half is already decided by the
    caller, which returns early on any heartbeat that is not this attempt's, and its ts half is
    already expressed in the clamp: a heartbeat stamped before launch cannot describe work this
    attempt did, so it is worth exactly the launch anchor and no more. What remains is the
    unusable-ts case, where the helper falls back to now; that matches the previous behaviour of
    treating an untimestamped heartbeat as present-tense progress.
    """
    ts, _fresh = heartbeat_progress_ts(hb_key, context.launch_ts, context.current_attempt)
    return ts


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
    # Credit the heartbeat's OWN timestamp, clamped to [launch, now], never the time we happened to
    # read it. A heartbeat written long before a reattach and only seen now describes progress that
    # already happened; paying it out at read time hands a wedged worker a full fresh stall window
    # on every reattach. `_progress_ts` returns that clamped value, deliberately distinct from the
    # raw `hb_ts` above, which stays the uploaded-ts bookkeeping the advance checks below compare.
    # A heartbeat with no usable ts still clamps to now, preserving the existing behaviour that an
    # untimestamped heartbeat counts as progress. It is computed only where it is used: this loop's
    # clock is observable through `time.time`, so an unused read would be a visible side effect.
    #
    # heartbeat ordering is independent of other progress sources. a queue exemption or status
    # transition may already have advanced `last_progress` beyond this heartbeat's timestamp, so
    # preserve the newer stall anchor while updating heartbeat-specific bookkeeping.
    if hb_attempt > state.last_hb_attempt:
        # fresh attempt: reset ts baseline and re-derive seen_training_hb so cold-start grace rearms.
        state.last_hb_attempt = hb_attempt
        state.last_hb_ts = hb_ts or 0.0
        state.last_progress = max(state.last_progress, _progress_ts(context, new_key))
        state.seen_training_hb = is_training_hb
    elif hb_attempt == state.last_hb_attempt and (hb_ts is None or hb_ts > state.last_hb_ts):
        # gate progress on ts advancing: a stale late upload must not buy a fresh stall window.
        if hb_ts is not None:
            state.last_hb_ts = hb_ts
        state.last_progress = max(state.last_progress, _progress_ts(context, new_key))
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
    now = time.time()
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
        time.sleep(delay)


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
) -> PollResult:
    """Poll a queue job to completion; resilient to transient API errors.

    Use setup grace before the first heartbeat, then stall grace; fail fast on sustained throttled,
    unhealthy, or over-queued states.
    """
    if not handle.job_id:
        raise ValueError("endpoint-only RunPod handles cannot be polled")
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    launch_ts = handle.started_ts
    if not math.isfinite(launch_ts) or launch_ts <= 0:
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
        poll_errors=PollErrorTracker(say, interval_s),
    )
    state = _PollState(
        last_status=None,
        last_hb_key=None,
        last_hb_ts=0.0,
        # -1 sentinel < any real attempt; gates out prior-attempt leftover heartbeats
        last_hb_attempt=-1,
        # Seed from the persisted LAUNCH, not this poll's start. A reattach has been billing since
        # launch, so a worker already wedged before we attached must not be handed a fresh setup
        # window by the mere act of looking at it -- that is what let a restart loop keep a dead
        # worker alive to the absolute deadline. For a job queued from the start this value is
        # overwritten anyway by the queued exemption in `_classify_stall`, which re-anchors on every
        # pre-grant poll; it is the ALREADY-PLACED job at poll start where the difference is real.
        last_progress=launch_ts,
        seen_training_hb=False,
        last_health_probe=0.0,
        unhealthy_timer=GraceTimer(),
        throttled_timer=GraceTimer(),
        queued_timer=GraceTimer(),
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
            # A genuine TRANSITION is progress, so it moves the clock. The FIRST observation is not a
            # transition -- `last_status` simply starts unset -- and crediting it would throw away the
            # launch anchor on poll one, which is exactly how a delayed reattach used to win back a
            # whole setup window before any evidence of progress had been seen.
            if state.last_status is not None:
                state.last_progress = time.time()
            state.last_status = status
        terminal = _classify_terminal_status(context, state, provider_status, status)
        if terminal is not None:
            return terminal
        now = time.time()
        terminal = _classify_queue_state(context, state, status, now)
        if terminal is not None:
            return terminal
        _update_heartbeat(context, state)
        terminal = _classify_stall(context, state, status)
        if terminal is not None:
            return terminal
        _sleep_until_next_poll(context)
