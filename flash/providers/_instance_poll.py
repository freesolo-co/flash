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
import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.providers._hf_artifacts import (
    heartbeat_is_stale_prior_attempt,
    worker_flagged_retriable,
)
from flash.providers._poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    PollErrorTracker,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers.base import PollResult

# A fresh DONE can precede the separately-uploaded metrics.json (HF read-after-write lag). Re-read
# metrics a few times before falling back to the infra-retryable poll_error, so a DONE-signalled
# success isn't hard-failed on a transient read gap.
_METRICS_AFTER_DONE_RETRIES = 6
_METRICS_AFTER_DONE_WAIT_S = 5.0

# A successful box self-destroys / vanishes the instant it finishes — often before HF exposes its DONE /
# attempt marker. Re-read terminal artifacts a few times before concluding host loss, so a
# finished-then-gone seed isn't mis-retried against its own artifacts.
_TERMINAL_AFTER_DEAD_RETRIES = 6
_TERMINAL_AFTER_DEAD_WAIT_S = 5.0


def _read_with_retries(read, *, tries: int, wait_s: float, say, message: str):
    """Call ``read()`` (a zero-arg reader), re-reading up to ``tries`` times ``wait_s`` apart while it
    returns None — for HF read-after-write lag (metrics.json after DONE; terminal artifacts after a box
    self-destructs). ``time.sleep`` is mocked in tests, so this adds no test wall-time."""
    value = read()
    while value is None and tries > 0:
        say(message)
        time.sleep(wait_s)
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

    provider: str
    instance_id: object
    current_attempt: int
    launch_ts: float  # now-floored; 0.0 launch means "unknown" -> falls back to now (elapsed math + cost)
    dating_launch: float  # TRUE launch (0.0 == unknown, NOT now-floored) for heartbeat attempt-dating

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
    missing_dead_threshold: int  # consecutive missing reads that count as a disappearance (vast 4/lambda 3)

    # Liveness + failure evidence.
    early_liveness_alive: Callable[[], bool]  # non-empty container log (vast) / boot.log present (lambda)
    read_current_error: Callable[[], str | None]  # THIS attempt's error_<phase>_attempt<N>.txt, force-read

    # Provider-specific stamping + human detail (KEEP per-provider — billing basis + wording differ).
    stamp_cost_and_notes: Callable[..., None]  # (metrics, *, end_ts, launch_ts) -> None
    failure_detail: Callable[[dict | None], str]  # (marker) -> best root-cause detail
    load_timeout_detail: Callable[[str, float], str]  # (status, elapsed_s) -> str
    first_liveness_detail: Callable[[float, float], str]  # (elapsed_s, first_liveness_s) -> str


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
    deadline_s: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (the shared kernel behind poll_vast_job /
    poll_lambda_job).

    COMPLETED     fresh DONE sentinel on HF -> metrics.json (cost stamped by the adapter).
    job_failed    attempt marker with ok=false (a real worker error; fails fast unless flagged retriable).
    job_preempted instance died without DONE/marker (host loss) -> infra-shaped, retried.
    stalled       never left loading within ``load_timeout_s``; OR running but emitted NO liveness within
                  ``first_liveness_s``; OR heartbeat frozen past the setup/stall window; OR deadline passed.
    poll_error    status endpoint down past budget, OR DONE without a readable/parseable metrics.json —
                  infra-retryable (bounded by infra_retries), never a fast-fail on a signalled success.
    """
    say = make_say(log)
    launch_ts = adapter.launch_ts
    dating_launch = adapter.dating_launch
    done_reader = adapter.done_reader
    marker_reader = adapter.marker_reader
    metrics_reader = adapter.metrics_reader

    def finish_ok(end_ts_hint: float | str | None = None) -> PollResult:
        # metrics.json is written before DONE but HF read-after-write lags: re-read a few times before
        # falling back to the poll_error retry on a DONE-without-metrics.
        raw = _read_with_retries(
            lambda: metrics_reader(force=True),
            tries=_METRICS_AFTER_DONE_RETRIES,
            wait_s=_METRICS_AFTER_DONE_WAIT_S,
            say=say,
            message="DONE seen but metrics.json not visible yet; waiting for HF read-after-write",
        )
        if raw is None:
            # DONE means the worker SIGNALLED SUCCESS; an unreadable metrics.json after the in-line
            # retries is a transient HF read-after-write gap, not a worker error. Don't fast-fail a
            # successful run as job_failed — return the infra-retryable poll_error so it gets its
            # bounded infra budget (never a forever-spin) instead of a hard terminal failure.
            return PollResult(
                False, failure="poll_error", detail="DONE without metrics.json (transient HF read)"
            )
        try:
            metrics = json.loads(raw)
        except ValueError:
            # A present-but-unparseable metrics.json (truncated read-after-write / corrupt) must NOT
            # escape the poll loop as a raw JSONDecodeError and abort the run past the teardown finally.
            # Same transient read-after-write gap on a DONE-signalled success: classify it exactly like
            # the DONE-without-metrics case above -- infra-retryable poll_error, not job_failed.
            return PollResult(
                False, failure="poll_error", detail="DONE with unparseable metrics.json (transient HF read)"
            )
        # end_ts is the worker's completion time — the DONE sentinel's payload (a str) or an ok marker's
        # ts (a float) — else now; adopt only if in [launch, now]. float() tolerates surrounding
        # whitespace, so no pre-strip is needed.
        end_ts = time.time()
        if end_ts_hint is not None:
            with contextlib.suppress(TypeError, ValueError):
                ts = float(end_ts_hint)
                if launch_ts <= ts <= end_ts:
                    end_ts = ts
        adapter.stamp_cost_and_notes(metrics, end_ts=end_ts, launch_ts=launch_ts)
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        # DONE carries the worker's time.time(); 120 s of clock-skew grace. Anything older predates this
        # attempt (leftover from a prior attempt's resume).
        try:
            return float(content.strip()) > launch_ts - 120.0
        except ValueError:
            return False

    def finish_from_ok_marker(marker: dict | None = None) -> PollResult:
        # An ok marker means the worker finished (metrics.json was written first), even if DONE is stale.
        # Pass DONE only when fresh; else use the marker's own completion ts for the wall note.
        d = done_reader(force=True)
        fresh = d is not None and done_is_fresh(d)
        marker_ts = marker.get("ts") if isinstance(marker, dict) else None
        return finish_ok(d if fresh else marker_ts)

    def fail_from_marker(marker: dict | None) -> PollResult:
        # A real worker error fails fast UNLESS flagged retriable (in the marker, or the worker's
        # heartbeat for a RetriableInfraError). Gate the heartbeat flag to THIS attempt so a stale
        # retriable=True from a prior attempt can't turn a fast-fail into a GPU-burning retry loop.
        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(
            heartbeat_reader, launch_ts=dating_launch, current_attempt=adapter.current_attempt
        )
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=adapter.failure_detail(marker),
        )

    def terminal_artifact_result(force: bool = True) -> PollResult | None:
        # The worker's terminal HF artifacts (DONE / attempt marker) -> a terminal PollResult when it
        # definitively finished or errored, else None. The SINGLE terminal detector shared by the poll
        # loop and every give-up path. ``force`` bypasses the read cache: True for the one-shot give-up
        # reads; False for the per-iteration loop poll, which already paces its own reads.
        d = done_reader(force=force)
        if d is not None and done_is_fresh(d):
            return finish_ok(d)
        raw = marker_reader(force=force)
        if raw:
            with contextlib.suppress(ValueError):
                m = json.loads(raw)
                if m.get("ok"):
                    return finish_from_ok_marker(m)
                return fail_from_marker(m)
        return None

    def stalled_unless_terminal(detail: str) -> PollResult:
        # A stall exit still checks for terminal artifacts — the worker may have finished right at the
        # boundary. Use the BOUNDED read (like the deadline / dead-host / poll-error paths) so a fresh
        # DONE/marker not yet visible under HF read-after-write lag isn't missed and mis-classified stalled
        # (which would fail a max_retries=0 run that actually completed, or rent a second box for the seed).
        terminal = _read_with_retries(
            terminal_artifact_result,
            tries=_TERMINAL_AFTER_DEAD_RETRIES,
            wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
            say=say,
            message="stall boundary; waiting for HF to expose any terminal DONE/marker before stalled",
        )
        return terminal if terminal is not None else PollResult(False, failure="stalled", detail=detail)

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
        if deadline_s is not None and time.time() - start > deadline_s:
            # A recovered run can blow a launch-anchored deadline on the first reattach tick (the outage
            # lasted past max_wall+grace). The worker may also have just finished / self-destructed AT the
            # wall cap with its DONE/marker not yet visible on HF (read-after-write lag), so do the BOUNDED
            # terminal-artifact wait (not a single read) before declaring stalled — else the retry re-rents
            # a GPU for the same seed while the completed attempt's artifacts surface moments later.
            terminal = _read_with_retries(
                terminal_artifact_result,
                tries=_TERMINAL_AFTER_DEAD_RETRIES,
                wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                say=say,
                message="deadline reached; waiting for HF to expose any terminal DONE/marker before giving up",
            )
            if terminal is not None:
                return terminal
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            inst = adapter.fetch_instance()
            poll_errors.reset()
        except adapter.poll_error_exceptions as e:
            # A transient status-fetch failure (api error, or a malformed 200 body that surfaces as a
            # decode / incomplete-read error rather than the api's own error type): count it against the
            # budget and keep polling — a read blip must not look like a gone instance.
            if poll_errors.record(e):
                # The status endpoint is down, but the worker may have COMPLETED during the outage and
                # written its terminal DONE/marker to HF (a different endpoint). Do the BOUNDED terminal
                # read (same as the deadline / dead-host paths) before giving up: a prolonged outage can
                # end right as the worker finishes, so a single read can miss the just-written artifact
                # under HF read-after-write lag. Else poll_error tears the box down and the retry relaunches
                # a second worker for an attempt that already finished (duplicate work + double-bill).
                terminal = _read_with_retries(
                    terminal_artifact_result,
                    tries=_TERMINAL_AFTER_DEAD_RETRIES,
                    wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                    say=say,
                    message="status-poll outage; waiting for HF to expose any terminal DONE/marker before poll_error",
                )
                if terminal is not None:
                    return terminal
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        # The instance-detail route can transiently answer as if the instance were absent for healthy
        # (and brand-new) boxes. One missing read means nothing — only a sustained streak is a real
        # disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get(adapter.status_field) or ("missing" if inst is None else "unknown")
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
                tries=_TERMINAL_AFTER_DEAD_RETRIES,
                wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                say=say,
                message="instance gone; waiting for HF to expose any terminal DONE/marker before failover",
            )
            if terminal is not None:
                return terminal
            # Dead host, no ok-marker/DONE. Distinguish genuine host LOSS (retry on a fresh host) from a
            # worker that RAN and CRASHED early leaving error_{phase}_attempt<N>.txt (bad env/config/OOM):
            # that is DETERMINISTIC -> fail FAST. A crash the worker flagged retriable still retries.
            err = adapter.read_current_error()
            # Error files are attempt-scoped but the heartbeat is run-scoped: gate the crash evidence on
            # heartbeat provenance (dated by dating_launch, the TRUE launch) so a stale prior-attempt
            # heartbeat can't flip a genuine host-loss retry into a fast-fail job_failed.
            crash_evidence_is_current = not heartbeat_is_stale_prior_attempt(
                heartbeat_reader, launch_ts=dating_launch, current_attempt=adapter.current_attempt
            )
            worker_crashed = (
                bool(err and err.strip())
                and crash_evidence_is_current
                and not worker_flagged_retriable(
                    heartbeat_reader, launch_ts=dating_launch, current_attempt=adapter.current_attempt
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
                seen_fresh_hb = True  # any fresh hb (incl. a bare liveness ping) disarms first-liveness
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
        # down on a lagging status feed. deadline_s (+ the finally destroy and periodic sweep) stays the
        # ultimate spend backstop once this is disarmed. ORDER MATTERS: this runs AFTER the heartbeat read
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
                            adapter.first_liveness_detail(time.time() - running_since, first_liveness_s)
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
        time.sleep(interval_s)
