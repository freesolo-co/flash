"""shared rent-a-box instance polling and terminal-artifact handling.

provider-specific status, markers, costs, and messages live in ``InstancePollAdapter``. every timeout,
dead-host, stall, or status-outage path performs a bounded artifact reread so hf visibility lag cannot
turn completed work into a retry.

reference ``time`` as a module attribute so provider poll tests patch the shared fake clock.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.providers._lifecycle.deadline import (
    deadline_kwargs,
    remaining_seconds,
    require_deadline_at,
)
from flash.providers._lifecycle.poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    PollErrorTracker,
    _attempt_int,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers.artifacts.hf import worker_flagged_retriable
from flash.providers.base import PollResult

# a strict success marker can precede the separately uploaded metrics.json under hf read-after-write
# lag. re-read metrics before falling back to the infra-retryable poll_error so marker-authorized
# success is not hard-failed on a transient read gap.
_METRICS_AFTER_SUCCESS_RETRIES = 6
_METRICS_AFTER_SUCCESS_WAIT_S = 5.0

# any boundary exit can race terminal-artifact visibility under hf read-after-write lag. re-read
# terminal artifacts before classifying the exit so a completed seed is not retried against its work.
_TERMINAL_REREAD_RETRIES = 6
_TERMINAL_REREAD_WAIT_S = 5.0


def _read_with_retries(
    read,
    *,
    tries: int,
    wait_s: float,
    say,
    message: str,
    deadline_at: float | None = None,
):
    """Re-read an artifact with every wait capped by the absolute run deadline."""
    value = read()
    while value is None and tries > 0:
        delay = wait_s
        if deadline_at is not None:
            remaining = remaining_seconds(deadline_at)
            if remaining <= 0:
                break
            delay = min(delay, remaining)
        say(message)
        if delay > 0:
            time.sleep(delay)
        if deadline_at is not None and remaining_seconds(deadline_at) <= 0:
            break
        value = read()
        tries -= 1
    return value


@dataclass
class InstancePollAdapter:
    """The per-provider seams :func:`poll_instance_job` is parameterized by. A wrapper (poll_vast_job /
    poll_lambda_job) builds one — resolving its api module + readers at CALL time so test monkeypatches
    of ``vast_api.get_instance`` / ``_make_hf_file_reader`` / heartbeat readers all still bite — then
    hands it to the shared driver. Everything NOT here is identical across providers and lives in the
    driver, baselined on Vast."""

    instance_id: object
    run_id: str
    current_attempt: int
    launch_ts: float

    # Terminal-artifact readers (built from the provider's monkeypatchable ``_make_hf_file_reader``).
    done_reader: Callable[..., str | None]
    marker_reader: Callable[..., str | None]
    metrics_reader: Callable[..., str | None]

    # Instance status fetch + classification.
    fetch_instance: Callable[[], dict | None]  # resolves the api's get_instance at call time
    poll_error_exceptions: tuple  # transient fetch errors: count against the budget, keep polling
    status_field: str  # "actual_status" (vast) / "status" (lambda)
    running_status: str  # the "up" value: "running" (vast) / "active" (lambda)
    dead_states: frozenset  # states meaning "the box is gone / won't progress"
    missing_dead_threshold: (
        int  # consecutive missing reads that count as a disappearance (vast 4/lambda 3)
    )

    # Liveness + failure evidence.
    early_liveness_alive: Callable[
        [], bool
    ]  # non-empty container log (vast) / boot.log present (lambda)
    read_current_error: Callable[
        [], str | None
    ]  # THIS attempt's error_<phase>_attempt<N>.txt, force-read

    # Provider-specific stamping + human detail (KEEP per-provider — billing basis + wording differ).
    stamp_cost_and_notes: Callable[..., None]  # (metrics, *, end_ts, launch_ts) -> None
    failure_detail: Callable[[dict | None], str]  # (marker) -> best root-cause detail
    load_timeout_detail: Callable[[str, float], str]  # (status, elapsed_s) -> str
    first_liveness_detail: Callable[[float, float], str]  # (elapsed_s, first_liveness_s) -> str


def decode_terminal_marker(
    raw: str,
    *,
    run_id: str,
    attempt: int,
    launch_floor: float,
    deadline_at: float | None = None,
) -> dict:
    """Validate one attempt-scoped terminal marker against its durable identity."""
    marker = json.loads(raw)
    if not isinstance(marker, dict) or set(marker) != {
        "attempt",
        "error",
        "ok",
        "retriable",
        "run_id",
        "ts",
    }:
        raise ValueError("invalid terminal marker schema")
    marker_attempt = marker["attempt"]
    ts = marker["ts"]
    error = marker["error"]
    now = time.time()
    if (
        isinstance(launch_floor, bool)
        or not isinstance(launch_floor, (int, float))
        or not math.isfinite(launch_floor)
        or launch_floor <= 0
        or isinstance(now, bool)
        or not isinstance(now, (int, float))
        or not math.isfinite(now)
        or now <= 0
        or _attempt_int(marker_attempt) is None
        or marker_attempt != _attempt_int(attempt)
        or type(marker["run_id"]) is not str
        or marker["run_id"] != run_id
        or type(marker["ok"]) is not bool
        or type(marker["retriable"]) is not bool
        or type(error) is not str
        or len(error) > 4096
        or isinstance(ts, bool)
        or not isinstance(ts, (int, float))
        or not math.isfinite(ts)
        or ts < launch_floor
        or ts > now + 120.0
        or (deadline_at is not None and ts > deadline_at)
    ):
        raise ValueError("invalid terminal marker identity")
    return marker


def _decode_terminal_marker(
    raw: str,
    adapter: InstancePollAdapter,
) -> dict:
    return decode_terminal_marker(
        raw,
        run_id=adapter.run_id,
        attempt=adapter.current_attempt,
        launch_floor=adapter.launch_ts,
    )


@dataclass(frozen=True)
class _InstancePollContext:
    adapter: InstancePollAdapter
    say: Callable
    interval_s: float
    heartbeat_reader: Callable | None
    setup_grace_s: float
    stall_after_s: float
    first_liveness_s: float
    load_timeout_s: float
    absolute_deadline: float | None
    launch_ts: float
    done_reader: Callable
    marker_reader: Callable
    metrics_reader: Callable


@dataclass
class _InstancePollState:
    deferred_deadline_failure: PollResult | None
    poll_errors: PollErrorTracker
    start: float
    last_status: str | None
    last_hb_key: object | None
    last_progress: float
    became_running: bool
    running_since: float
    observed_running_since: float | None
    seen_training_hb: bool
    seen_fresh_hb: bool
    liveness_seen: bool
    liveness_absent_polls: int
    missing_streak: int


def _read_instance_artifact(
    ctx: _InstancePollContext,
    reader,
    *,
    force: bool,
    read_deadline_at: float | None,
):
    return reader(
        force=force,
        **deadline_kwargs(reader, read_deadline_at),
    )


def _finish_instance_ok(
    ctx: _InstancePollContext,
    end_ts_hint: float | str | None = None,
    *,
    read_deadline_at: float | None,
) -> PollResult:
    # the strict success marker authorizes completion, but metrics.json visibility can lag it on hf.
    # re-read before falling back to the poll_error retry used for a success without metrics.
    raw = _read_with_retries(
        lambda: _read_instance_artifact(
            ctx,
            ctx.metrics_reader,
            force=True,
            read_deadline_at=read_deadline_at,
        ),
        tries=_METRICS_AFTER_SUCCESS_RETRIES,
        wait_s=_METRICS_AFTER_SUCCESS_WAIT_S,
        say=ctx.say,
        message="DONE seen but metrics.json not visible yet; waiting for HF read-after-write",
        deadline_at=read_deadline_at,
    )
    if raw is None:
        # the strict success marker has authorized completion, so missing metrics after the inline
        # retries is a transient hf read gap, not a worker error. return the infra-retryable
        # poll_error so the run gets its bounded infra budget instead of a hard terminal failure.
        return PollResult(
            False, failure="poll_error", detail="DONE without metrics.json (transient HF read)"
        )
    try:
        metrics = json.loads(raw)
    except ValueError:
        # a present but unparseable metrics.json must not escape as a raw json decode error and
        # bypass the poll result path. treat this marker-authorized success like missing metrics:
        # infra-retryable poll_error, not job_failed.
        return PollResult(
            False,
            failure="poll_error",
            detail="DONE with unparseable metrics.json (transient HF read)",
        )
    # end_ts is the worker completion time: prefer the optional fresh done timestamp supplied by
    # the caller, otherwise use the strict ok marker timestamp. adopt only values in [launch, now].
    end_ts = time.time()
    if end_ts_hint is not None:
        with contextlib.suppress(TypeError, ValueError):
            ts = float(end_ts_hint)
            if ctx.launch_ts <= ts <= end_ts:
                end_ts = ts
    ctx.adapter.stamp_cost_and_notes(metrics, end_ts=end_ts, launch_ts=ctx.launch_ts)
    return PollResult(True, metrics=metrics)


def _instance_done_is_fresh(ctx: _InstancePollContext, content: str) -> bool:
    # done carries only a finite completion timestamp bound to this attempt and deadline.
    try:
        ts = float(content.strip())
    except (AttributeError, TypeError, ValueError):
        return False
    now = time.time()
    return bool(
        math.isfinite(ts)
        and math.isfinite(now)
        and ts > ctx.launch_ts - 120.0
        and ts <= now + 120.0
        and (ctx.absolute_deadline is None or ts <= ctx.absolute_deadline)
    )


def _finish_instance_from_ok_marker(
    ctx: _InstancePollContext,
    marker: dict,
    *,
    read_deadline_at: float | None,
) -> PollResult:
    # a strict ok marker authorizes completion even if done is stale or absent. prefer a fresh done
    # timestamp when available; otherwise use the marker's completion timestamp for the wall note.
    d = _read_instance_artifact(
        ctx,
        ctx.done_reader,
        force=True,
        read_deadline_at=read_deadline_at,
    )
    fresh = d is not None and _instance_done_is_fresh(ctx, d)
    marker_ts = marker["ts"]
    return _finish_instance_ok(
        ctx,
        d if fresh else marker_ts,
        read_deadline_at=read_deadline_at,
    )


def _fail_instance_from_marker(ctx: _InstancePollContext, marker: dict) -> PollResult:
    # a real worker error fails fast unless flagged retriable in the marker or the worker heartbeat.
    # gate the heartbeat flag to this attempt so a stale prior-attempt flag cannot trigger a retry.
    retriable = marker["retriable"] or worker_flagged_retriable(
        ctx.heartbeat_reader,
        launch_ts=ctx.launch_ts,
        current_attempt=ctx.adapter.current_attempt,
    )
    return PollResult(
        False,
        failure="job_preempted" if retriable else "job_failed",
        detail=ctx.adapter.failure_detail(marker),
    )


def _terminal_instance_artifact_result(
    ctx: _InstancePollContext,
    state: _InstancePollState,
    *,
    force: bool,
    read_deadline_at: float | None,
    defer_deadline_failure: bool,
) -> PollResult | None:
    # the strict attempt marker is the sole terminal authority. done may help timestamp a
    # marker-authorized success, but it cannot complete a run by itself.
    raw = _read_instance_artifact(
        ctx,
        ctx.marker_reader,
        force=force,
        read_deadline_at=read_deadline_at,
    )
    if raw is None:
        return None
    try:
        marker = _decode_terminal_marker(raw, ctx.adapter)
    except (TypeError, ValueError):
        return PollResult(
            False,
            failure="job_failed",
            detail="terminal marker is invalid or unverifiable",
        )
    if marker["ok"]:
        return _finish_instance_from_ok_marker(
            ctx,
            marker,
            read_deadline_at=read_deadline_at,
        )
    failure = _fail_instance_from_marker(ctx, marker)
    # the watchdog marker can race a successful final marker or lagging done upload. only the
    # deadline exit defers it; every other failure marker still terminates immediately.
    if defer_deadline_failure and marker["error"].startswith("run wall deadline exceeded"):
        state.deferred_deadline_failure = failure
        return None
    return failure


def _surface_instance_final_heartbeat(
    ctx: _InstancePollContext,
    state: _InstancePollState,
) -> None:
    # persist the latest metrics before any terminal give-up so `flash runs log -f` still shows the final
    # per-step metrics for runs that end via deadline / dead-host / poll-outage / stall, not only the
    # per-iteration terminal path below. deadline_at=None reads heartbeats already committed at the
    # boundary even though this runs past the compute deadline; capturing the returned key stops a
    # repeat (pre- vs post-reread) call from surfacing the same heartbeat twice.
    forced_reader = (
        (lambda: ctx.heartbeat_reader(force=True, deadline_at=None))
        if ctx.heartbeat_reader is not None
        else None
    )
    state.last_hb_key, _ = surface_heartbeat(forced_reader, state.last_hb_key, ctx.say)


def _instance_deadline_unless_terminal(
    ctx: _InstancePollContext,
    state: _InstancePollState,
) -> PollResult:
    # the run deadline stops worker compute, not bounded observation of artifacts already committed
    # at the boundary. share one fixed reread budget across terminal and metrics visibility lag.
    state.deferred_deadline_failure = None
    _surface_instance_final_heartbeat(ctx, state)
    read_deadline = time.time() + (_TERMINAL_REREAD_RETRIES * _TERMINAL_REREAD_WAIT_S)
    terminal = _read_with_retries(
        lambda: _terminal_instance_artifact_result(
            ctx,
            state,
            force=True,
            read_deadline_at=read_deadline,
            defer_deadline_failure=True,
        ),
        tries=_TERMINAL_REREAD_RETRIES,
        wait_s=_TERMINAL_REREAD_WAIT_S,
        say=ctx.say,
        message="deadline reached; waiting for HF to expose any terminal DONE/marker before stalled",
        deadline_at=read_deadline,
    )
    if terminal is not None:
        _surface_instance_final_heartbeat(ctx, state)
        return terminal
    if state.deferred_deadline_failure is not None:
        return state.deferred_deadline_failure
    return PollResult(False, failure="stalled", detail="client-side deadline exceeded")


def _instance_stalled_unless_terminal(
    ctx: _InstancePollContext,
    state: _InstancePollState,
    detail: str,
) -> PollResult:
    # A stall exit still checks for terminal artifacts — the worker may have finished right at the
    # boundary. Use the BOUNDED read (like the deadline / dead-host / poll-error paths) so a fresh
    # DONE/marker not yet visible under HF read-after-write lag isn't missed and mis-classified stalled
    # (which would fail a max_retries=0 run that actually completed, or rent a second box for the seed).
    _surface_instance_final_heartbeat(ctx, state)
    terminal = _read_with_retries(
        lambda: _terminal_instance_artifact_result(
            ctx,
            state,
            force=True,
            read_deadline_at=ctx.absolute_deadline,
            defer_deadline_failure=False,
        ),
        tries=_TERMINAL_REREAD_RETRIES,
        wait_s=_TERMINAL_REREAD_WAIT_S,
        say=ctx.say,
        message="stall boundary; waiting for HF to expose any terminal DONE/marker before stalled",
        deadline_at=ctx.absolute_deadline,
    )
    if terminal is not None:
        _surface_instance_final_heartbeat(ctx, state)
        return terminal
    return PollResult(False, failure="stalled", detail=detail)


def _new_instance_poll_state(ctx: _InstancePollContext) -> _InstancePollState:
    poll_errors = PollErrorTracker(ctx.say, ctx.interval_s)
    # Seed the load/stall clocks from LAUNCH, not this poll's start: a delayed reattach has been billing
    # since launch, so a still-loading box that already blew load_timeout_s fails over now.
    start = ctx.launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_running = False
    # Anchored to launch so a reattach already-running measures first-liveness from the original launch.
    running_since = start
    # Wall-clock THIS session first saw it running (set once) — "how long WE have watched it", so a
    # reattach doesn't fast-fail a box that just came up.
    observed_running_since = None
    seen_training_hb = False
    # Any FRESH heartbeat from this attempt proves the worker started -> clears the first-liveness
    # deadline (distinct from seen_training_hb, which gates the tighter training window).
    seen_fresh_hb = False
    # A positive early-liveness signal (non-empty container log / present boot.log) proves the bootstrap
    # is alive on a slow cold start (pip/code fetch can outlast first_liveness_s before the first
    # heartbeat), so we don't fast-fail a healthy box. Require silence across BOOT_LOG_ABSENT_POLLS so a
    # log-API blip can't burn a retry.
    liveness_seen = False
    liveness_absent_polls = 0
    missing_streak = 0
    return _InstancePollState(
        deferred_deadline_failure=None,
        poll_errors=poll_errors,
        start=start,
        last_status=last_status,
        last_hb_key=last_hb_key,
        last_progress=last_progress,
        became_running=became_running,
        running_since=running_since,
        observed_running_since=observed_running_since,
        seen_training_hb=seen_training_hb,
        seen_fresh_hb=seen_fresh_hb,
        liveness_seen=liveness_seen,
        liveness_absent_polls=liveness_absent_polls,
        missing_streak=missing_streak,
    )


def _fetch_instance_for_poll(
    ctx: _InstancePollContext,
    state: _InstancePollState,
) -> tuple[dict | None, PollResult | None, bool]:
    try:
        inst = ctx.adapter.fetch_instance()
        state.poll_errors.reset()
    except ctx.adapter.poll_error_exceptions as e:
        # A transient status-fetch failure (api error, or a malformed 200 body that surfaces as a
        # decode / incomplete-read error rather than the api's own error type): count it against the
        # budget and keep polling — a read blip must not look like a gone instance.
        if state.poll_errors.record(e, deadline_at=ctx.absolute_deadline):
            _surface_instance_final_heartbeat(ctx, state)
            # the status endpoint can fail while the worker finishes through hf. perform the same
            # bounded terminal-artifact reread used by deadline/dead-host paths so visibility lag
            # cannot relaunch completed work and double-bill it.
            terminal = _read_with_retries(
                lambda: _terminal_instance_artifact_result(
                    ctx,
                    state,
                    force=True,
                    read_deadline_at=ctx.absolute_deadline,
                    defer_deadline_failure=False,
                ),
                tries=_TERMINAL_REREAD_RETRIES,
                wait_s=_TERMINAL_REREAD_WAIT_S,
                say=ctx.say,
                message="status-poll outage; waiting for HF to expose any terminal DONE/marker before poll_error",
                deadline_at=ctx.absolute_deadline,
            )
            if terminal is not None:
                _surface_instance_final_heartbeat(ctx, state)
                return None, terminal, False
            return (
                None,
                PollResult(
                    False,
                    failure="poll_error",
                    detail="provider status polling failed repeatedly",
                ),
                False,
            )
        return None, None, True
    return inst, None, False


def _instance_running_stall_result(
    ctx: _InstancePollContext,
    state: _InstancePollState,
    status: str,
) -> PollResult | None:
    # after running without a heartbeat, consult early-liveness before declaring a stall: slow
    # installs and fetches can be healthy. latch positive bootstrap evidence; require silence for
    # BOOT_LOG_ABSENT_POLLS and respect the observed-running floor on reattach.
    if (
        not state.seen_fresh_hb
        and not state.liveness_seen
        and time.time() - state.running_since > ctx.first_liveness_s
        and state.observed_running_since is not None
        and time.time() - state.observed_running_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
    ):
        if not ctx.adapter.early_liveness_alive():
            # A lone absent read can be a transient log-API error -> require BOOT_LOG_ABSENT_POLLS.
            state.liveness_absent_polls += 1
            if state.liveness_absent_polls >= BOOT_LOG_ABSENT_POLLS:
                return _instance_stalled_unless_terminal(
                    ctx,
                    state,
                    ctx.adapter.first_liveness_detail(
                        time.time() - state.running_since, ctx.first_liveness_s
                    ),
                )
        else:
            # Bootstrap is producing output -> healthy slow cold start; setup/stall below backstop.
            state.liveness_seen = True
    limit = ctx.stall_after_s if state.seen_training_hb else ctx.setup_grace_s
    if time.time() - state.last_progress > limit:
        phase = "training" if state.seen_training_hb else "setup (pre-training)"
        return _instance_stalled_unless_terminal(
            ctx,
            state,
            f"no worker progress for {int(time.time() - state.last_progress)}s during "
            f"{phase} (instance status {status}, limit {int(limit)}s)",
        )
    return None


def _poll_instance_observation(
    ctx: _InstancePollContext,
    state: _InstancePollState,
    inst: dict | None,
) -> PollResult | None:
    # The instance-detail route can transiently answer as if the instance were absent for healthy
    # (and brand-new) boxes. One missing read means nothing — only a sustained streak is a real
    # disappearance.
    state.missing_streak = state.missing_streak + 1 if inst is None else 0

    status = (inst or {}).get(ctx.adapter.status_field) or (
        "missing" if inst is None else "unknown"
    )
    if status != state.last_status:
        ctx.say(f"instance {ctx.adapter.instance_id}: {status}")
        # Count a status TRANSITION as progress, but NOT the first observation (last_status starts
        # None, so the first read always "changes" — crediting it would hand a silent-since-launch
        # worker a fresh setup grace after every restart).
        if state.last_status is not None:
            state.last_progress = time.time()
            if status == ctx.adapter.running_status:
                state.running_since = time.time()  # genuine ->running: start the liveness clock
        state.last_status = status
    if status == ctx.adapter.running_status:
        state.became_running = True
        if state.observed_running_since is None:
            state.observed_running_since = time.time()

    # Per-iteration terminal check: the SAME detector every give-up path uses (force=False — the loop
    # paces its own reads). Folds the DONE and ok/err-marker checks into one call.
    terminal = _terminal_instance_artifact_result(
        ctx,
        state,
        force=False,
        read_deadline_at=ctx.absolute_deadline,
        defer_deadline_failure=False,
    )
    if terminal is not None:
        forced_reader = (
            (lambda: ctx.heartbeat_reader(force=True, deadline_at=None))
            if ctx.heartbeat_reader is not None
            else None
        )
        state.last_hb_key, _ = surface_heartbeat(forced_reader, state.last_hb_key, ctx.say)
        return terminal

    # ``unknown`` = "host has no recent heartbeat and won't progress" (host loss) -> dead for fast
    # failover. Gate on ``became_running`` because ``unknown`` is ALSO this driver's no-status
    # fallback during provisioning; only a box that WAS running then goes unknown is genuine loss.
    dead = (
        state.missing_streak >= ctx.adapter.missing_dead_threshold
        or status in ctx.adapter.dead_states
        or (state.became_running and status == "unknown")
    )
    if dead:
        _surface_instance_final_heartbeat(ctx, state)
        # The worker may have finished just before the box self-destroyed, with its DONE/marker not
        # yet visible on HF (read-after-write lag). Re-read terminal artifacts a bounded number of
        # times before concluding loss.
        terminal = _read_with_retries(
            lambda: _terminal_instance_artifact_result(
                ctx,
                state,
                force=True,
                read_deadline_at=ctx.absolute_deadline,
                defer_deadline_failure=False,
            ),
            tries=_TERMINAL_REREAD_RETRIES,
            wait_s=_TERMINAL_REREAD_WAIT_S,
            say=ctx.say,
            message="instance gone; waiting for HF to expose any terminal DONE/marker before failover",
            deadline_at=ctx.absolute_deadline,
        )
        if terminal is not None:
            _surface_instance_final_heartbeat(ctx, state)
            return terminal
        # Dead host, no ok-marker/DONE. Distinguish genuine host LOSS (retry on a fresh host) from a
        # worker that RAN and CRASHED early leaving error_{phase}_attempt<N>.txt (bad env/config/OOM):
        # that is DETERMINISTIC -> fail FAST. A crash the worker flagged retriable still retries.
        err = ctx.adapter.read_current_error()
        # error files are attempt-scoped, so a present file already belongs to this exact handle.
        worker_crashed = bool(err and err.strip()) and not worker_flagged_retriable(
            ctx.heartbeat_reader,
            launch_ts=ctx.launch_ts,
            current_attempt=ctx.adapter.current_attempt,
        )
        return PollResult(
            False,
            failure="job_failed" if worker_crashed else "job_preempted",
            detail=ctx.adapter.failure_detail(None),
        )

    new_key, stage = surface_heartbeat(ctx.heartbeat_reader, state.last_hb_key, ctx.say)
    if new_key != state.last_hb_key:
        state.last_hb_key = new_key
        # Credit the heartbeat's OWN ts (clamped to [launch, now]) so a pre-restart-stale heartbeat
        # buys no fresh window. ``fresh`` is False for a leftover prior-attempt heartbeat.
        hb_ts, fresh = heartbeat_progress_ts(new_key, ctx.launch_ts, ctx.adapter.current_attempt)
        if fresh:
            state.seen_fresh_hb = True
            # Advance the stall clock ONLY on a STAGED heartbeat (stage is not None): a bare liveness
            # ping must not let a wedged worker keep resetting the setup/training stall window.
            # MONOTONIC: never regress on an out-of-order upload.
            if stage is not None:
                state.last_progress = max(state.last_progress, hb_ts)
                # Tighten setup_grace -> stall only once training genuinely begins (the shared helper
                # keeps cold-start pings, incl. the silent step=0 first rollout, under setup grace).
                if is_training_heartbeat(stage, new_key[1]):
                    state.seen_training_hb = True

    # a fresh heartbeat proves startup even if the detail api still says loading/unknown, so it
    # disarms load timeout. order matters: this check follows heartbeat processing so a heartbeat
    # arriving exactly at the boundary wins on the same tick. the absolute deadline still caps spend.
    if (
        not state.became_running
        and not state.seen_fresh_hb
        and time.time() - state.start > ctx.load_timeout_s
    ):
        return PollResult(
            False,
            failure="stalled",
            detail=ctx.adapter.load_timeout_detail(status, time.time() - state.start),
        )

    if state.became_running:
        return _instance_running_stall_result(ctx, state, status)
    return None


def _sleep_before_instance_poll(ctx: _InstancePollContext) -> None:
    delay = ctx.interval_s
    if ctx.absolute_deadline is not None:
        remaining = remaining_seconds(ctx.absolute_deadline)
        if remaining <= 0:
            return
        delay = min(delay, remaining)
    if delay > 0:
        time.sleep(delay)


def _poll_instance_loop(ctx: _InstancePollContext, state: _InstancePollState) -> PollResult:
    while True:
        if ctx.absolute_deadline is not None and time.time() >= ctx.absolute_deadline:
            return _instance_deadline_unless_terminal(ctx, state)
        inst, result, retry = _fetch_instance_for_poll(ctx, state)
        if result is not None:
            return result
        if retry:
            continue
        if ctx.absolute_deadline is not None and time.time() >= ctx.absolute_deadline:
            return _instance_deadline_unless_terminal(ctx, state)
        result = _poll_instance_observation(ctx, state, inst)
        if result is not None:
            return result
        _sleep_before_instance_poll(ctx)


def poll_instance_job(
    adapter: InstancePollAdapter,
    *,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    setup_grace_s: float,
    stall_after_s: float,
    first_liveness_s: float,
    load_timeout_s: float,
    deadline_at: float | None = None,
) -> PollResult:
    """poll instance status and hf artifacts to a terminal result.

    completed requires a strict ok marker and metrics. worker markers fail fast; dead hosts, stalls,
    status outages, and unreadable signalled-success metrics remain infrastructure-retryable.
    """
    say = make_say(log)
    launch_ts = adapter.launch_ts
    if (
        isinstance(launch_ts, bool)
        or not isinstance(launch_ts, (int, float))
        or not math.isfinite(launch_ts)
        or launch_ts <= 0
    ):
        raise ValueError("persisted instance launch timestamp is invalid")
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    done_reader = adapter.done_reader
    marker_reader = adapter.marker_reader
    metrics_reader = adapter.metrics_reader
    ctx = _InstancePollContext(
        adapter=adapter,
        say=say,
        interval_s=interval_s,
        heartbeat_reader=heartbeat_reader,
        setup_grace_s=setup_grace_s,
        stall_after_s=stall_after_s,
        first_liveness_s=first_liveness_s,
        load_timeout_s=load_timeout_s,
        absolute_deadline=absolute_deadline,
        launch_ts=launch_ts,
        done_reader=done_reader,
        marker_reader=marker_reader,
        metrics_reader=metrics_reader,
    )
    state = _new_instance_poll_state(ctx)
    return _poll_instance_loop(ctx, state)
