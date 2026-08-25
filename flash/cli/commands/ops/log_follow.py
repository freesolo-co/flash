"""Surviving a blip while following a run that is already training.

`flash train` submits a run, then streams it. A 502 from the polling endpoints says nothing
about the run -- it keeps training and keeps billing on a GPU -- but an unretried one aborted
the follow, and the CLI exited nonzero as though the submit itself had failed. The next thing a
user does with a failed submit is submit again, and pay for two.

This module holds the decision of what counts as a blip and how long to tolerate one. The
follow loops live in the package module and call in here.
"""

from __future__ import annotations

import sys
import time

from flash.cli.ui import heartbeat as heartbeat_ui
from flash.cli.ui import render
from flash.cli.ui.tty import TtyStatusLine
from flash.client import ApiError, ClientError, RequestTimeoutError, ServiceUnreachableError


class FollowInterrupted(ClientError):
    """The follow stream broke, but the run is unaffected and still going.

    Carries `run_id` because every handler has to name the run: the caller's next move is to
    resume or cancel it, and it cannot do either from a bare transport error.
    """

    def __init__(self, run_id: str, message: str):
        super().__init__(message)
        self.run_id = run_id


# `flash train` streams a run it already created. A blip on the polling endpoints says nothing
# about the run, which keeps training and keeps billing on the GPU. Retry across one long enough
# to outlast a plane restart or a proxy 502 storm, and only then hand back to a caller that must
# name the run rather than report a failure.
_FOLLOW_RETRY_SECONDS = 300.0
_FOLLOW_RETRY_BACKOFF = (1.0, 2.0, 5.0, 10.0, 15.0)
# 5xx and timeouts are the plane or a proxy in front of it, never a verdict on the run. 4xx is a
# real answer about this request (a bad id, a revoked key) and must surface immediately.
_FOLLOW_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _follow_transient_reason(exc: BaseException) -> str | None:
    """Why `exc` is a transient stream failure, or None when it is a real answer about the run."""
    if isinstance(exc, RequestTimeoutError):
        return "the request timed out"
    if isinstance(exc, ApiError):
        if exc.status in _FOLLOW_TRANSIENT_STATUSES:
            return f"the service answered HTTP {exc.status}"
        return None
    if isinstance(exc, ServiceUnreachableError):
        # the transport never got a verdict from the plane: nobody answered.
        return "the service was unreachable"
    # a bare `ClientError` is the base class, and the client raises it for permanent conditions
    # too -- a proxy answering 200 text/html, a body of the wrong shape. Retrying those buries the
    # message that says how to fix them, so anything not proven transient surfaces now.
    return None


class _FollowRetry:
    """Retry budget shared by every polling call in one follow loop.

    One budget for the whole loop, not per call: a plane that is down fails `get_logs` and
    `get_run` alternately, and a per-call budget would reset on each and never give up.
    """

    def __init__(self, run_id: str, *, window: float | None = None):
        self._run_id = run_id
        # read at construction, not bound as a default, so the budget stays patchable by name.
        self._window = _FOLLOW_RETRY_SECONDS if window is None else window
        self._failing_since: float | None = None
        self._attempt = 0
        self.warned = False

    def reset(self) -> None:
        self._failing_since = None
        self._attempt = 0
        # per outage, not per process: a second outage after a recovery is news again.
        self.warned = False

    def note_failure(self, exc: BaseException) -> tuple[float, str]:
        """Record a failure and return (sleep_seconds, reason), or raise once the budget is out."""
        reason = _follow_transient_reason(exc)
        if reason is None:
            raise exc
        now = time.monotonic()
        if self._failing_since is None:
            self._failing_since = now
        elif now - self._failing_since >= self._window:
            # the reason already names the status; appending `exc` too would read
            # "answered HTTP 502: HTTP Error 502: Bad Gateway".
            raise FollowInterrupted(
                self._run_id,
                f"lost the log stream for {self._window:.0f}s ({reason})",
            ) from exc
        delay = _FOLLOW_RETRY_BACKOFF[min(self._attempt, len(_FOLLOW_RETRY_BACKOFF) - 1)]
        self._attempt += 1
        return delay, reason


def _warn_follow_retry(
    spinner: _LogFollowSpinner | None, run_id: str, reason: str, retry: _FollowRetry
) -> None:
    """Say once that the stream is retrying, so a stalled follow is not read as a stalled run.

    Once per outage, not once per attempt: a plane down for five minutes would otherwise bury the
    logs already printed under a wall of identical warnings.
    """
    if retry.warned:
        return
    retry.warned = True
    if spinner is not None:
        spinner.clear()
    message = f"lost the log stream for {run_id} ({reason}); retrying, the run is unaffected"
    print(
        render.warn(message) if render.styled() else f"warning: {message}",
        file=sys.stderr,
        flush=True,
    )


_SPINNER_FRAMES = "|/-\\"
_SPINNER_TICK_SECONDS = 0.1


class _LogFollowSpinner(TtyStatusLine):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id
        self._frame = 0

    def render(self, progress: str) -> None:
        if not self._enabled:
            return
        frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        message = f"{frame} following logs for {self._run_id} ({progress})"
        self._write(message)


def _sleep_with_spinner(interval: float, spinner: _LogFollowSpinner, progress: str) -> None:
    if interval <= 0:
        return
    if not spinner.enabled:
        time.sleep(interval)
        return
    ticks = max(1, int(interval / _SPINNER_TICK_SECONDS))
    sleep_for = interval / ticks
    for _ in range(ticks):
        spinner.render(progress)
        time.sleep(sleep_for)


_FOLLOW_METRIC_FIELDS = (
    ("reward", "reward"),
    ("reward_std", "reward_std"),
    ("grad_norm", "grad_norm"),
    ("kl", "kl"),
    ("entropy", "entropy"),
    ("frac_reward_zero_std", "frac_zero_std"),
    ("mean_completion_tokens", "comp_len"),
    ("truncation_rate", "trunc"),
    ("discarded_rollouts", "discarded"),
    ("max_completion_tokens", "max_comp_tokens"),
)


def _log_follow_metric_rows(status: dict | None, seen_steps: set) -> list[str]:
    """Return unseen heartbeat-backed RL metric rows, deduplicated by attempt and optimizer step."""
    heartbeat = (status or {}).get("last_heartbeat")
    if not isinstance(heartbeat, dict):
        return []
    # during a retry, status.remote.attempt can already point at the replacement worker while
    # last_heartbeat still belongs to the prior attempt; don't render that stale attempt's rows
    if not heartbeat_ui.heartbeat_is_current_attempt(status, heartbeat):
        return []
    metrics_last = heartbeat.get("metrics_last")
    if not isinstance(metrics_last, list):
        return []
    rows = []
    attempt = heartbeat.get("attempt")
    for metrics in metrics_last:
        if not isinstance(metrics, dict):
            continue
        step = metrics.get("step")
        if step is None:
            continue
        try:
            step_key = int(step)
        except (TypeError, ValueError):
            step_key = str(step)
        metric_key = (attempt, step_key)
        if metric_key in seen_steps:
            continue
        seen_steps.add(metric_key)
        parts = [f"step={step_key}"]
        for key, label in _FOLLOW_METRIC_FIELDS:
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                value = f"{value:.6g}"
            parts.append(f"{label}={value}")
        rows.append(" ".join(parts))
    return rows
