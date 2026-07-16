"""Shared rent-a-box instance poll driver.

RunPod aside, the two "rent a whole box, watch its HF artifacts" providers (Vast and Lambda) ran two
~80%-duplicate poll loops. This module is the ONE driver they now share: ``poll_instance_job`` plus the
terminal-artifact detector family (``finish_ok`` / ``done_is_fresh`` / ``finish_from_ok_marker`` /
``fail_from_marker`` / ``terminal_artifact_result``) and the bounded ``_read_with_retries`` re-reader.

Everything provider-specific is factored into a small per-call :class:`InstancePollAdapter` (the marker
filename, the status field/vocabulary, the instance fetcher and the exceptions its transient failures
raise, the early-liveness probe, the cost/notes stamping, and the failure-detail + stall-message
builders). The kernel here is baselined on VAST, whose loop is strictly the more robust of the two:
every give-up path (deadline / dead host / stalled / status-poll outage) does a BOUNDED terminal-artifact
re-read before concluding loss, so a seed that finished right at the boundary — with its DONE/marker not
yet visible under HF read-after-write lag — is recognised instead of mis-retried against its own work.

``time`` is referenced as a module attribute (``time.time`` / ``time.sleep``) so a test that patches the
shared ``time`` module (or a provider jobs module's ``time``, which is the same singleton object) reaches
this driver too — the fake clocks/sleeps the provider poll tests install still bite.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.providers._deadline import deadline_kwargs, remaining_seconds, require_deadline_at
from flash.providers._hf_artifacts import worker_flagged_retriable
from flash.providers._poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    PollErrorTracker,
    _attempt_int,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
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


def _decode_terminal_marker(
    raw: str,
    adapter: InstancePollAdapter,
) -> dict:
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
    attempt = marker["attempt"]
    ts = marker["ts"]
    error = marker["error"]
    now = time.time()
    launch_floor = adapter.launch_ts
    if (
        isinstance(launch_floor, bool)
        or not isinstance(launch_floor, (int, float))
        or not math.isfinite(launch_floor)
        or launch_floor <= 0
        or isinstance(now, bool)
        or not isinstance(now, (int, float))
        or not math.isfinite(now)
        or now <= 0
        or _attempt_int(attempt) is None
        or attempt != _attempt_int(adapter.current_attempt)
        or type(marker["run_id"]) is not str
        or marker["run_id"] != adapter.run_id
        or type(marker["ok"]) is not bool
        or type(marker["retriable"]) is not bool
        or type(error) is not str
        or len(error) > 4096
        or isinstance(ts, bool)
        or not isinstance(ts, (int, float))
        or not math.isfinite(ts)
        or ts < launch_floor
        or ts > now + 120.0
    ):
        raise ValueError("invalid terminal marker identity")
    return marker


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
    """Poll instance status + HF artifacts to a terminal state (the shared kernel behind poll_vast_job /
    poll_lambda_job).

    COMPLETED     strict ok marker on HF -> metrics.json (cost stamped by the adapter).
    job_failed    attempt marker with ok=false (a real worker error; fails fast unless flagged retriable).
    job_preempted instance died without DONE/marker (host loss) -> infra-shaped, retried.
    stalled       never left loading within ``load_timeout_s``; OR running but emitted NO liveness within
                  ``first_liveness_s``; OR heartbeat frozen past the setup/stall window; OR deadline passed.
    poll_error    status endpoint down past budget, OR DONE without a readable/parseable metrics.json —
                  infra-retryable (bounded by infra_retries), never a fast-fail on a signalled success.
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
    deferred_deadline_failure: PollResult | None = None

    def read_artifact(reader, *, force: bool, read_deadline_at: float | None):
        return reader(
            force=force,
            **deadline_kwargs(reader, read_deadline_at),
        )

    def finish_ok(
        end_ts_hint: float | str | None = None,
        *,
        read_deadline_at: float | None = absolute_deadline,
    ) -> PollResult:
        # the strict success marker authorizes completion, but metrics.json visibility can lag it on hf.
        # re-read before falling back to the poll_error retry used for a success without metrics.
        raw = _read_with_retries(
            lambda: read_artifact(
                metrics_reader,
                force=True,
                read_deadline_at=read_deadline_at,
            ),
            tries=_METRICS_AFTER_SUCCESS_RETRIES,
            wait_s=_METRICS_AFTER_SUCCESS_WAIT_S,
            say=say,
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
                if launch_ts <= ts <= end_ts:
                    end_ts = ts
        adapter.stamp_cost_and_notes(metrics, end_ts=end_ts, launch_ts=launch_ts)
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        # done carries only a finite completion timestamp bound to this attempt and deadline.
        try:
            ts = float(content.strip())
        except (AttributeError, TypeError, ValueError):
            return False
        now = time.time()
        return bool(
            math.isfinite(ts)
            and math.isfinite(now)
            and ts > launch_ts - 120.0
            and ts <= now + 120.0
            and (absolute_deadline is None or ts <= absolute_deadline)
        )

    def finish_from_ok_marker(
        marker: dict,
        *,
        read_deadline_at: float | None = absolute_deadline,
    ) -> PollResult:
        # a strict ok marker authorizes completion even if done is stale or absent. prefer a fresh done
        # timestamp when available; otherwise use the marker's completion timestamp for the wall note.
        d = read_artifact(
            done_reader,
            force=True,
            read_deadline_at=read_deadline_at,
        )
        fresh = d is not None and done_is_fresh(d)
        marker_ts = marker["ts"]
        return finish_ok(d if fresh else marker_ts, read_deadline_at=read_deadline_at)

    def fail_from_marker(marker: dict) -> PollResult:
        # a real worker error fails fast unless flagged retriable in the marker or the worker heartbeat.
        # gate the heartbeat flag to this attempt so a stale prior-attempt flag cannot trigger a retry.
        retriable = marker["retriable"] or worker_flagged_retriable(
            heartbeat_reader, launch_ts=launch_ts, current_attempt=adapter.current_attempt
        )
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=adapter.failure_detail(marker),
        )

    def terminal_artifact_result(
        force: bool = True,
        *,
        read_deadline_at: float | None = absolute_deadline,
        defer_deadline_failure: bool = False,
    ) -> PollResult | None:
        nonlocal deferred_deadline_failure
        # the strict attempt marker is the sole terminal authority. done may help timestamp a
        # marker-authorized success, but it cannot complete a run by itself.
        raw = read_artifact(
            marker_reader,
            force=force,
            read_deadline_at=read_deadline_at,
        )
        if raw is None:
            return None
        try:
            marker = _decode_terminal_marker(raw, adapter)
        except (TypeError, ValueError):
            return PollResult(
                False,
                failure="job_failed",
                detail="terminal marker is invalid or unverifiable",
            )
        if marker["ok"]:
            return finish_from_ok_marker(marker, read_deadline_at=read_deadline_at)
        failure = fail_from_marker(marker)
        # the watchdog marker can race a successful final marker or lagging done upload. only the
        # deadline exit defers it; every other failure marker still terminates immediately.
        if defer_deadline_failure and marker["error"].startswith("run wall deadline exceeded"):
            deferred_deadline_failure = failure
            return None
        return failure

    def deadline_unless_terminal() -> PollResult:
        nonlocal deferred_deadline_failure
        # the run deadline stops worker compute, not bounded observation of artifacts already committed
        # at the boundary. share one fixed reread budget across terminal and metrics visibility lag.
        deferred_deadline_failure = None
        read_deadline = time.time() + (
            _TERMINAL_REREAD_RETRIES * _TERMINAL_REREAD_WAIT_S
        )
        terminal = _read_with_retries(
            lambda: terminal_artifact_result(
                read_deadline_at=read_deadline,
                defer_deadline_failure=True,
            ),
            tries=_TERMINAL_REREAD_RETRIES,
            wait_s=_TERMINAL_REREAD_WAIT_S,
            say=say,
            message="deadline reached; waiting for HF to expose any terminal DONE/marker before stalled",
            deadline_at=read_deadline,
        )
        if terminal is not None:
            return terminal
        if deferred_deadline_failure is not None:
            return deferred_deadline_failure
        return PollResult(False, failure="stalled", detail="client-side deadline exceeded")

    def stalled_unless_terminal(detail: str) -> PollResult:
        # A stall exit still checks for terminal artifacts — the worker may have finished right at the
        # boundary. Use the BOUNDED read (like the deadline / dead-host / poll-error paths) so a fresh
        # DONE/marker not yet visible under HF read-after-write lag isn't missed and mis-classified stalled
        # (which would fail a max_retries=0 run that actually completed, or rent a second box for the seed).
        terminal = _read_with_retries(
            terminal_artifact_result,
            tries=_TERMINAL_REREAD_RETRIES,
            wait_s=_TERMINAL_REREAD_WAIT_S,
            say=say,
            message="stall boundary; waiting for HF to expose any terminal DONE/marker before stalled",
            deadline_at=absolute_deadline,
        )
        return (
            terminal
            if terminal is not None
            else PollResult(False, failure="stalled", detail=detail)
        )

    poll_errors = PollErrorTracker(say, interval_s)
    # Seed the load/stall clocks from LAUNCH, not this poll's start: a delayed reattach has been billing
    # since launch, so a still-loading box that already blew load_timeout_s fails over now.
    start = launch_ts
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
    while True:
        if absolute_deadline is not None and time.time() >= absolute_deadline:
            return deadline_unless_terminal()
        try:
            inst = adapter.fetch_instance()
            poll_errors.reset()
        except adapter.poll_error_exceptions as e:
            # A transient status-fetch failure (api error, or a malformed 200 body that surfaces as a
            # decode / incomplete-read error rather than the api's own error type): count it against the
            # budget and keep polling — a read blip must not look like a gone instance.
            if poll_errors.record(e, deadline_at=absolute_deadline):
                # The status endpoint is down, but the worker may have COMPLETED during the outage and
                # written its terminal DONE/marker to HF (a different endpoint). Do the BOUNDED terminal
                # read (same as the deadline / dead-host paths) before giving up: a prolonged outage can
                # end right as the worker finishes, so a single read can miss the just-written artifact
                # under HF read-after-write lag. Else poll_error tears the box down and the retry relaunches
                # a second worker for an attempt that already finished (duplicate work + double-bill).
                terminal = _read_with_retries(
                    terminal_artifact_result,
                    tries=_TERMINAL_REREAD_RETRIES,
                    wait_s=_TERMINAL_REREAD_WAIT_S,
                    say=say,
                    message="status-poll outage; waiting for HF to expose any terminal DONE/marker before poll_error",
                    deadline_at=absolute_deadline,
                )
                if terminal is not None:
                    return terminal
                return PollResult(
                    False,
                    failure="poll_error",
                    detail="provider status polling failed repeatedly",
                )
            continue
        if absolute_deadline is not None and time.time() >= absolute_deadline:
            return deadline_unless_terminal()
        # The instance-detail route can transiently answer as if the instance were absent for healthy
        # (and brand-new) boxes. One missing read means nothing — only a sustained streak is a real
        # disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get(adapter.status_field) or (
            "missing" if inst is None else "unknown"
        )
        if status != last_status:
            say(f"instance {adapter.instance_id}: {status}")
            # Count a status TRANSITION as progress, but NOT the first observation (last_status starts
            # None, so the first read always "changes" — crediting it would hand a silent-since-launch
            # worker a fresh setup grace after every restart).
            if last_status is not None:
                last_progress = time.time()
                if status == adapter.running_status:
                    running_since = time.time()  # genuine ->running: start the liveness clock
            last_status = status
        if status == adapter.running_status:
            became_running = True
            if observed_running_since is None:
                observed_running_since = time.time()

        # Per-iteration terminal check: the SAME detector every give-up path uses (force=False — the loop
        # paces its own reads). Folds the DONE and ok/err-marker checks into one call.
        terminal = terminal_artifact_result(force=False)
        if terminal is not None:
            return terminal

        # ``unknown`` = "host has no recent heartbeat and won't progress" (host loss) -> dead for fast
        # failover. Gate on ``became_running`` because ``unknown`` is ALSO this driver's no-status
        # fallback during provisioning; only a box that WAS running then goes unknown is genuine loss.
        dead = (
            missing_streak >= adapter.missing_dead_threshold
            or status in adapter.dead_states
            or (became_running and status == "unknown")
        )
        if dead:
            # The worker may have finished just before the box self-destroyed, with its DONE/marker not
            # yet visible on HF (read-after-write lag). Re-read terminal artifacts a bounded number of
            # times before concluding loss.
            terminal = _read_with_retries(
                terminal_artifact_result,
                tries=_TERMINAL_REREAD_RETRIES,
                wait_s=_TERMINAL_REREAD_WAIT_S,
                say=say,
                message="instance gone; waiting for HF to expose any terminal DONE/marker before failover",
                deadline_at=absolute_deadline,
            )
            if terminal is not None:
                return terminal
            # Dead host, no ok-marker/DONE. Distinguish genuine host LOSS (retry on a fresh host) from a
            # worker that RAN and CRASHED early leaving error_{phase}_attempt<N>.txt (bad env/config/OOM):
            # that is DETERMINISTIC -> fail FAST. A crash the worker flagged retriable still retries.
            err = adapter.read_current_error()
            # error files are attempt-scoped, so a present file already belongs to this exact handle.
            worker_crashed = (
                bool(err and err.strip())
                and not worker_flagged_retriable(
                    heartbeat_reader,
                    launch_ts=launch_ts,
                    current_attempt=adapter.current_attempt,
                )
            )
            return PollResult(
                False,
                failure="job_failed" if worker_crashed else "job_preempted",
                detail=adapter.failure_detail(None),
            )

        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            # Credit the heartbeat's OWN ts (clamped to [launch, now]) so a pre-restart-stale heartbeat
            # buys no fresh window. ``fresh`` is False for a leftover prior-attempt heartbeat.
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts, adapter.current_attempt)
            if fresh:
                seen_fresh_hb = True
                # Advance the stall clock ONLY on a STAGED heartbeat (stage is not None): a bare liveness
                # ping must not let a wedged worker keep resetting the setup/training stall window.
                # MONOTONIC: never regress on an out-of-order upload.
                if stage is not None:
                    last_progress = max(last_progress, hb_ts)
                    # Tighten setup_grace -> stall only once training genuinely begins (the shared helper
                    # keeps cold-start pings, incl. the silent step=0 first rollout, under setup grace).
                    if is_training_heartbeat(stage, new_key[1]):
                        seen_training_hb = True

        # Load timeout: a box that never left loading/unknown within load_timeout_s never started. But a
        # fresh heartbeat PROVES the worker booted even while the detail API lags in loading/unknown
        # (never flipping to 'running'), so it disarms this — else a healthy, heartbeating box is torn
        # down on a lagging status feed. the absolute deadline remains the spend backstop once this is
        # disarmed. order matters: this runs after the heartbeat read
        # above, so THIS tick's first fresh heartbeat (which sets seen_fresh_hb) disarms the timeout on the
        # very iteration it arrives — a heartbeat landing exactly at the timeout mark must not be raced by
        # a stale seen_fresh_hb from the prior tick.
        if not became_running and not seen_fresh_hb and time.time() - start > load_timeout_s:
            return PollResult(
                False,
                failure="stalled",
                detail=adapter.load_timeout_detail(status, time.time() - start),
            )

        if became_running:
            # Fast-failover: a box that reached 'running' but emitted NO heartbeat past first_liveness_s
            # might be wedged -> 'stalled'. But a healthy slow cold start (pip install / code fetch) also
            # has no heartbeat yet, so before failing over consult the early-liveness probe: a positive
            # signal means the bootstrap is alive (latch, let setup_grace_s govern); only a box SILENT
            # across BOOT_LOG_ABSENT_POLLS is the wedged host. The observed-running floor keeps a reattach
            # from fast-failing a box that just came up.
            if (
                not seen_fresh_hb
                and not liveness_seen
                and time.time() - running_since > first_liveness_s
                and observed_running_since is not None
                and time.time() - observed_running_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                if not adapter.early_liveness_alive():
                    # A lone absent read can be a transient log-API error -> require BOOT_LOG_ABSENT_POLLS.
                    liveness_absent_polls += 1
                    if liveness_absent_polls >= BOOT_LOG_ABSENT_POLLS:
                        return stalled_unless_terminal(
                            adapter.first_liveness_detail(
                                time.time() - running_since, first_liveness_s
                            )
                        )
                else:
                    # Bootstrap is producing output -> healthy slow cold start; setup/stall below backstop.
                    liveness_seen = True
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
                phase = "training" if seen_training_hb else "setup (pre-training)"
                return stalled_unless_terminal(
                    f"no worker progress for {int(time.time() - last_progress)}s during "
                    f"{phase} (instance status {status}, limit {int(limit)}s)"
                )
        delay = interval_s
        if absolute_deadline is not None:
            remaining = remaining_seconds(absolute_deadline)
            if remaining <= 0:
                continue
            delay = min(delay, remaining)
        if delay > 0:
            time.sleep(delay)
