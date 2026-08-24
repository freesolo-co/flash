"""shared rent-a-box instance polling and terminal-artifact handling.

provider-specific status, markers, costs, and messages live in ``InstancePollAdapter``. every timeout,
dead-host, stall, or status-outage path performs a bounded artifact reread so hf visibility lag cannot
turn completed work into a retry.

reference ``time`` as a module attribute so provider poll tests patch the shared fake clock.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.providers._lifecycle.instances.poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    PollErrorTracker,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers._lifecycle.instances.terminal_artifacts import (
    INVALID_MARKER_DETAIL,
    AttemptIdentity,
    ProbeBudget,
    TerminalKind,
    read_within,
    resolve_terminal_artifacts,
)
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
    remaining_seconds,
    require_deadline_at,
)
from flash.providers.artifacts.hf import worker_flagged_retriable
from flash.providers.core.base import PollResult

# a strict success marker can precede the separately uploaded metrics.json under hf read-after-write
# lag. re-read metrics before falling back to the infra-retryable poll_error so marker-authorized
# success is not hard-failed on a transient read gap.
_METRICS_AFTER_SUCCESS_RETRIES = 6
_METRICS_AFTER_SUCCESS_WAIT_S = 5.0

# any boundary exit can race terminal-artifact visibility under hf read-after-write lag. re-read
# terminal artifacts before classifying the exit so a completed seed is not retried against its work.
# distinguishes "caller passed no read deadline" from an explicit ``None`` (unbounded read). the
# closures this replaced bound ``absolute_deadline`` as a default at definition time; a method cannot,
# so the sentinel restores that exact behavior.
_UNSET_DEADLINE = object()

_TERMINAL_REREAD_RETRIES = 6
_TERMINAL_REREAD_WAIT_S = 5.0


def _reread_budget(cutoff_at: float | None) -> ProbeBudget:
    """The give-up-path reread window, read from the module constants tests patch."""
    return ProbeBudget(
        tries=_TERMINAL_REREAD_RETRIES,
        wait_s=_TERMINAL_REREAD_WAIT_S,
        cutoff_at=cutoff_at,
    )


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


class _TerminalArtifacts:
    """Reads HF terminal artifacts (marker, done, metrics) and classifies them into a ``PollResult``.

    Groups the reads that every give-up path shares. ``last_hb_key`` and ``deferred_deadline_failure``
    live here because the give-up paths mutate them across calls, exactly as the closures they replace
    did through ``nonlocal``.
    """

    def __init__(self, adapter, *, say, heartbeat_reader, launch_ts, absolute_deadline):
        self._adapter = adapter
        self._say = say
        self._heartbeat_reader = heartbeat_reader
        self._launch_ts = launch_ts
        self._absolute_deadline = absolute_deadline
        self.last_hb_key = None
        self.deferred_deadline_failure: PollResult | None = None

    def _read_artifact(self, reader, *, force: bool, read_deadline_at: float | None):
        return reader(
            force=force,
            **deadline_kwargs(reader, read_deadline_at),
        )

    def _done_is_fresh(self, content: str) -> bool:
        # done carries only a finite completion timestamp bound to this attempt and deadline.
        try:
            ts = float(content.strip())
        except (AttributeError, TypeError, ValueError):
            return False
        now = time.time()
        return bool(
            math.isfinite(ts)
            and math.isfinite(now)
            and ts > self._launch_ts - 120.0
            and ts <= now + 120.0
            and (self._absolute_deadline is None or ts <= self._absolute_deadline)
        )

    def _succeed(
        self, marker: dict, metrics: dict, *, read_deadline_at: float | None
    ) -> PollResult:
        """Stamp a marker-authorized success with the best completion timestamp available.

        A strict ok marker authorizes completion even if DONE is stale or absent, so DONE only
        refines the wall note. Adopt a timestamp only inside [launch, now].
        """
        d = self._read_artifact(
            self._adapter.done_reader,
            force=True,
            read_deadline_at=read_deadline_at,
        )
        hint = d if (d is not None and self._done_is_fresh(d)) else marker["ts"]
        end_ts = time.time()
        with contextlib.suppress(TypeError, ValueError):
            ts = float(hint)
            if self._launch_ts <= ts <= end_ts:
                end_ts = ts
        self._adapter.stamp_cost_and_notes(metrics, end_ts=end_ts, launch_ts=self._launch_ts)
        return PollResult(True, metrics=metrics)

    def _fail_from_marker(self, marker: dict) -> PollResult:
        # a real worker error fails fast unless flagged retriable in the marker or the worker heartbeat.
        # gate the heartbeat flag to this attempt so a stale prior-attempt flag cannot trigger a retry.
        retriable = marker["retriable"] or worker_flagged_retriable(
            self._heartbeat_reader,
            launch_ts=self._launch_ts,
            current_attempt=self._adapter.current_attempt,
        )
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=self._adapter.failure_detail(marker),
        )

    def result(
        self,
        force: bool = True,
        *,
        read_deadline_at: float | None = _UNSET_DEADLINE,
        defer_deadline_failure: bool = False,
    ) -> PollResult | None:
        # the strict attempt marker is the sole terminal authority. done may help timestamp a
        # marker-authorized success, but it cannot complete a run by itself.
        if read_deadline_at is _UNSET_DEADLINE:
            read_deadline_at = self._absolute_deadline
        resolution = resolve_terminal_artifacts(
            AttemptIdentity(
                run_id=self._adapter.run_id,
                attempt=self._adapter.current_attempt,
                launch_floor=self._adapter.launch_ts,
            ),
            read_marker=lambda: self._read_artifact(
                self._adapter.marker_reader,
                force=force,
                read_deadline_at=read_deadline_at,
            ),
            read_metrics=lambda: self._read_artifact(
                self._adapter.metrics_reader,
                force=True,
                read_deadline_at=read_deadline_at,
            ),
            budget=ProbeBudget(
                tries=_METRICS_AFTER_SUCCESS_RETRIES,
                wait_s=_METRICS_AFTER_SUCCESS_WAIT_S,
                cutoff_at=read_deadline_at,
            ),
            say=self._say,
            metrics_message=(
                "DONE seen but metrics.json not visible yet; waiting for HF read-after-write"
            ),
        )
        if resolution.kind is TerminalKind.ABSENT:
            return None
        if resolution.kind is TerminalKind.UNVERIFIABLE:
            return PollResult(False, failure="job_failed", detail=INVALID_MARKER_DETAIL)
        if resolution.kind is TerminalKind.SUCCESS:
            return self._succeed(
                resolution.marker,
                resolution.metrics,
                read_deadline_at=read_deadline_at,
            )
        if resolution.kind is TerminalKind.PENDING:
            # the strict success marker has authorized completion, so unreadable metrics are a
            # transient hf read gap, not a worker error. return the infra-retryable poll_error so the
            # run gets its bounded infra budget instead of a hard terminal failure.
            return PollResult(
                False,
                failure="poll_error",
                detail=(
                    "DONE with unparseable metrics.json (transient HF read)"
                    if resolution.metrics_unparseable
                    else "DONE without metrics.json (transient HF read)"
                ),
            )
        marker = resolution.marker
        failure = self._fail_from_marker(marker)
        # the watchdog marker can race a successful final marker or lagging done upload. only the
        # deadline exit defers it; every other failure marker still terminates immediately.
        if defer_deadline_failure and marker["error"].startswith("run wall deadline exceeded"):
            self.deferred_deadline_failure = failure
            return None
        return failure

    def surface_final_heartbeat(self) -> None:
        # persist the latest metrics before any terminal give-up so `flash runs log -f` still shows the final
        # per-step metrics for runs that end via deadline / dead-host / poll-outage / stall, not only the
        # per-iteration terminal path below. deadline_at=None reads heartbeats already committed at the
        # boundary even though this runs past the compute deadline; capturing the returned key stops a
        # repeat (pre- vs post-reread) call from surfacing the same heartbeat twice.
        self.last_hb_key, _ = surface_heartbeat(self._forced_reader(), self.last_hb_key, self._say)

    def _forced_reader(self):
        if self._heartbeat_reader is None:
            return None
        return lambda: self._heartbeat_reader(force=True, deadline_at=None)

    def reread_before_giving_up(self, message: str) -> PollResult | None:
        """Bounded terminal reread shared by the dead-host, poll-outage, and stall give-up paths."""
        terminal = read_within(
            self.result,
            _reread_budget(self._absolute_deadline),
            say=self._say,
            message=message,
        )
        if terminal is not None:
            self.surface_final_heartbeat()
        return terminal

    def deadline_unless_terminal(self) -> PollResult:
        # the run deadline stops worker compute, not bounded observation of artifacts already committed
        # at the boundary. share one fixed reread budget across terminal and metrics visibility lag.
        self.deferred_deadline_failure = None
        self.surface_final_heartbeat()
        read_deadline = time.time() + (_TERMINAL_REREAD_RETRIES * _TERMINAL_REREAD_WAIT_S)
        terminal = read_within(
            lambda: self.result(
                read_deadline_at=read_deadline,
                defer_deadline_failure=True,
            ),
            _reread_budget(read_deadline),
            say=self._say,
            message="deadline reached; waiting for HF to expose any terminal DONE/marker before stalled",
        )
        if terminal is not None:
            self.surface_final_heartbeat()
            return terminal
        if self.deferred_deadline_failure is not None:
            return self.deferred_deadline_failure
        return PollResult(False, failure="stalled", detail="client-side deadline exceeded")

    def stalled_unless_terminal(self, detail: str) -> PollResult:
        # A stall exit still checks for terminal artifacts — the worker may have finished right at the
        # boundary. Use the BOUNDED read (like the deadline / dead-host / poll-error paths) so a fresh
        # DONE/marker not yet visible under HF read-after-write lag isn't missed and mis-classified stalled
        # (which would fail a max_retries=0 run that actually completed, or rent a second box for the seed).
        self.surface_final_heartbeat()
        terminal = self.reread_before_giving_up(
            "stall boundary; waiting for HF to expose any terminal DONE/marker before stalled"
        )
        if terminal is not None:
            return terminal
        return PollResult(False, failure="stalled", detail=detail)


@dataclass(frozen=True)
class _PollContext:
    adapter: InstancePollAdapter
    say: Callable[[str], None]
    interval_s: float
    setup_grace_s: float
    stall_after_s: float
    first_liveness_s: float
    load_timeout_s: float
    absolute_deadline: float | None
    heartbeat_reader: Callable | None
    artifacts: _TerminalArtifacts
    terminal_artifact_result: Callable[..., PollResult | None]
    deadline_unless_terminal: Callable[[], PollResult]
    stalled_unless_terminal: Callable[[str], PollResult]
    surface_final_heartbeat: Callable[[], None]
    reread_before_giving_up: Callable[[str], PollResult | None]


@dataclass
class _PollState:
    last_status: str | None
    last_progress: float
    became_running: bool
    running_since: float
    observed_running_since: float | None
    seen_training_hb: bool
    seen_fresh_hb: bool
    liveness_seen: bool
    liveness_absent_polls: int
    missing_streak: int


def _build_poll_context(
    adapter: InstancePollAdapter,
    *,
    log,
    interval_s: float,
    heartbeat_reader,
    setup_grace_s: float,
    stall_after_s: float,
    first_liveness_s: float,
    load_timeout_s: float,
    deadline_at: float | None,
) -> _PollContext:
    """validate fixed inputs and bind the terminal-artifact operations for one poll call."""
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
    artifacts = _TerminalArtifacts(
        adapter,
        say=say,
        heartbeat_reader=heartbeat_reader,
        launch_ts=launch_ts,
        absolute_deadline=absolute_deadline,
    )
    return _PollContext(
        adapter=adapter,
        say=say,
        interval_s=interval_s,
        setup_grace_s=setup_grace_s,
        stall_after_s=stall_after_s,
        first_liveness_s=first_liveness_s,
        load_timeout_s=load_timeout_s,
        absolute_deadline=absolute_deadline,
        heartbeat_reader=heartbeat_reader,
        artifacts=artifacts,
        terminal_artifact_result=artifacts.result,
        deadline_unless_terminal=artifacts.deadline_unless_terminal,
        stalled_unless_terminal=artifacts.stalled_unless_terminal,
        surface_final_heartbeat=artifacts.surface_final_heartbeat,
        reread_before_giving_up=artifacts.reread_before_giving_up,
    )


def _initial_poll_state(ctx: _PollContext) -> _PollState:
    """seed mutable clocks and liveness evidence from the persisted launch."""
    # Seed the load/stall clocks from LAUNCH, not this poll's start: a delayed reattach has been billing
    # since launch, so a still-loading box that already blew load_timeout_s fails over now.
    start = ctx.adapter.launch_ts
    # Anchored to launch so a reattach already-running measures first-liveness from the original launch.
    running_since = start
    # Wall-clock THIS session first saw it running (set once) — "how long WE have watched it", so a
    # reattach doesn't fast-fail a box that just came up.
    observed_running_since = None
    # Any FRESH heartbeat from this attempt proves the worker started -> clears the first-liveness
    # deadline (distinct from seen_training_hb, which gates the tighter training window).
    seen_fresh_hb = False
    # A positive early-liveness signal (non-empty container log / present boot.log) proves the bootstrap
    # is alive on a slow cold start (pip/code fetch can outlast first_liveness_s before the first
    # heartbeat), so we don't fast-fail a healthy box. Require silence across BOOT_LOG_ABSENT_POLLS so a
    # log-API blip can't burn a retry.
    liveness_seen = False
    liveness_absent_polls = 0
    return _PollState(
        last_status=None,
        last_progress=start,
        became_running=False,
        running_since=running_since,
        observed_running_since=observed_running_since,
        seen_training_hb=False,
        seen_fresh_hb=seen_fresh_hb,
        liveness_seen=liveness_seen,
        liveness_absent_polls=liveness_absent_polls,
        missing_streak=0,
    )


def _classify_deadline(ctx: _PollContext, _state: _PollState) -> PollResult | None:
    """decide whether the absolute run deadline ends polling on this boundary."""
    if ctx.absolute_deadline is not None and time.time() >= ctx.absolute_deadline:
        return ctx.deadline_unless_terminal()
    return None


def _classify_poll_error(
    ctx: _PollContext,
    _state: _PollState,
    poll_errors: PollErrorTracker,
    error: Exception,
) -> PollResult | None:
    """decide whether repeated provider read failures exhaust the poll budget."""
    # A transient status-fetch failure (api error, or a malformed 200 body that surfaces as a
    # decode / incomplete-read error rather than the api's own error type): count it against the
    # budget and keep polling — a read blip must not look like a gone instance.
    if not poll_errors.record(error, deadline_at=ctx.absolute_deadline):
        return None
    ctx.surface_final_heartbeat()
    # the status endpoint can fail while the worker finishes through hf. perform the same
    # bounded terminal-artifact reread used by deadline/dead-host paths so visibility lag
    # cannot relaunch completed work and double-bill it.
    terminal = ctx.reread_before_giving_up(
        "status-poll outage; waiting for HF to expose any terminal DONE/marker before poll_error"
    )
    if terminal is not None:
        return terminal
    return PollResult(
        False,
        failure="poll_error",
        detail="provider status polling failed repeatedly",
    )


def _update_instance_state(
    ctx: _PollContext,
    state: _PollState,
    inst: dict | None,
) -> str:
    """record one provider status observation and return its normalized state."""
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
    return status


def _classify_terminal_artifacts(
    ctx: _PollContext,
    _state: _PollState,
) -> PollResult | None:
    """decide whether this tick exposes an authoritative worker terminal marker."""
    # Per-iteration terminal check: the SAME detector every give-up path uses (force=False — the loop
    # paces its own reads). Folds the DONE and ok/err-marker checks into one call.
    terminal = ctx.terminal_artifact_result(force=False)
    if terminal is not None:
        ctx.surface_final_heartbeat()
    return terminal


def _classify_dead_host(
    ctx: _PollContext,
    state: _PollState,
    status: str,
) -> PollResult | None:
    """decide whether host-loss evidence requires failover or a deterministic failure."""
    # ``unknown`` = "host has no recent heartbeat and won't progress" (host loss) -> dead for fast
    # failover. Gate on ``became_running`` because ``unknown`` is ALSO this driver's no-status
    # fallback during provisioning; only a box that WAS running then goes unknown is genuine loss.
    dead = (
        state.missing_streak >= ctx.adapter.missing_dead_threshold
        or status in ctx.adapter.dead_states
        or (state.became_running and status == "unknown")
    )
    if not dead:
        return None
    ctx.surface_final_heartbeat()
    # The worker may have finished just before the box self-destroyed, with its DONE/marker not
    # yet visible on HF (read-after-write lag). Re-read terminal artifacts a bounded number of
    # times before concluding loss.
    terminal = ctx.reread_before_giving_up(
        "instance gone; waiting for HF to expose any terminal DONE/marker before failover"
    )
    if terminal is not None:
        return terminal
    # Dead host, no ok-marker/DONE. Distinguish genuine host LOSS (retry on a fresh host) from a
    # worker that RAN and CRASHED early leaving error_{phase}_attempt<N>.txt (bad env/config/OOM):
    # that is DETERMINISTIC -> fail FAST. A crash the worker flagged retriable still retries.
    err = ctx.adapter.read_current_error()
    # error files are attempt-scoped, so a present file already belongs to this exact handle.
    worker_crashed = bool(err and err.strip()) and not worker_flagged_retriable(
        ctx.heartbeat_reader,
        launch_ts=ctx.adapter.launch_ts,
        current_attempt=ctx.adapter.current_attempt,
    )
    return PollResult(
        False,
        failure="job_failed" if worker_crashed else "job_preempted",
        detail=ctx.adapter.failure_detail(None),
    )


def _process_heartbeat(ctx: _PollContext, state: _PollState) -> None:
    """surface one heartbeat and advance only the clocks its evidence supports."""
    new_key, stage = surface_heartbeat(
        ctx.heartbeat_reader,
        ctx.artifacts.last_hb_key,
        ctx.say,
    )
    if new_key != ctx.artifacts.last_hb_key:
        ctx.artifacts.last_hb_key = new_key
        # Credit the heartbeat's OWN ts (clamped to [launch, now]) so a pre-restart-stale heartbeat
        # buys no fresh window. ``fresh`` is False for a leftover prior-attempt heartbeat.
        hb_ts, fresh = heartbeat_progress_ts(
            new_key,
            ctx.adapter.launch_ts,
            ctx.adapter.current_attempt,
        )
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


def _classify_load_timeout(
    ctx: _PollContext,
    state: _PollState,
    status: str,
) -> PollResult | None:
    """decide whether a box that never started has exhausted its load window."""
    # a fresh heartbeat proves startup even if the detail api still says loading/unknown, so it
    # disarms load timeout. order matters: this check follows heartbeat processing so a heartbeat
    # arriving exactly at the boundary wins on the same tick. the absolute deadline still caps spend.
    if (
        not state.became_running
        and not state.seen_fresh_hb
        and time.time() - ctx.adapter.launch_ts > ctx.load_timeout_s
    ):
        return PollResult(
            False,
            failure="stalled",
            detail=ctx.adapter.load_timeout_detail(
                status,
                time.time() - ctx.adapter.launch_ts,
            ),
        )
    return None


def _classify_running_stall(
    ctx: _PollContext,
    state: _PollState,
    status: str,
) -> PollResult | None:
    """decide whether a running worker lacks bootstrap or staged progress evidence."""
    if not state.became_running:
        return None
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
                return ctx.stalled_unless_terminal(
                    ctx.adapter.first_liveness_detail(
                        time.time() - state.running_since,
                        ctx.first_liveness_s,
                    )
                )
        else:
            # Bootstrap is producing output -> healthy slow cold start; setup/stall below backstop.
            state.liveness_seen = True
    limit = ctx.stall_after_s if state.seen_training_hb else ctx.setup_grace_s
    if time.time() - state.last_progress > limit:
        phase = "training" if state.seen_training_hb else "setup (pre-training)"
        return ctx.stalled_unless_terminal(
            f"no worker progress for {int(time.time() - state.last_progress)}s during "
            f"{phase} (instance status {status}, limit {int(limit)}s)"
        )
    return None


def _sleep_until_next_poll(ctx: _PollContext) -> None:
    """sleep for one poll interval without crossing the absolute deadline."""
    delay = ctx.interval_s
    if ctx.absolute_deadline is not None:
        remaining = remaining_seconds(ctx.absolute_deadline)
        if remaining <= 0:
            return
        delay = min(delay, remaining)
    if delay > 0:
        time.sleep(delay)


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
    ctx = _build_poll_context(
        adapter,
        log=log,
        interval_s=interval_s,
        heartbeat_reader=heartbeat_reader,
        setup_grace_s=setup_grace_s,
        stall_after_s=stall_after_s,
        first_liveness_s=first_liveness_s,
        load_timeout_s=load_timeout_s,
        deadline_at=deadline_at,
    )
    poll_errors = PollErrorTracker(ctx.say, interval_s)
    state = _initial_poll_state(ctx)

    while True:
        result = _classify_deadline(ctx, state)
        if result is not None:
            return result
        try:
            inst = adapter.fetch_instance()
            poll_errors.reset()
        except adapter.poll_error_exceptions as error:
            result = _classify_poll_error(ctx, state, poll_errors, error)
            if result is not None:
                return result
            continue
        result = _classify_deadline(ctx, state)
        if result is not None:
            return result
        status = _update_instance_state(ctx, state, inst)
        result = _classify_terminal_artifacts(ctx, state)
        if result is not None:
            return result
        result = _classify_dead_host(ctx, state, status)
        if result is not None:
            return result
        _process_heartbeat(ctx, state)
        result = _classify_load_timeout(ctx, state, status)
        if result is not None:
            return result
        result = _classify_running_stall(ctx, state, status)
        if result is not None:
            return result
        _sleep_until_next_poll(ctx)
