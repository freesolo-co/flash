"""Console upload cadence shared by instance bootstrap callers.

This module owns only the wedge state machine. The caller owns file reads, artifact uploads and
credential-safe error rendering, passed as callbacks so this leaf stays independent of the
bootstrap and remains importable when shipped next to it as a bare module on a rented box.
"""

from __future__ import annotations

_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
_CONSOLE_UPLOAD_POLL_S = 120.0
_CONSOLE_UPLOAD_QUIET_POLLS = 4
_CONSOLE_UPLOAD_CREDITS = 2


def _run_console_upload_loop(
    console: str, interval_s: float, stop_upload, *, progress, upload
) -> None:
    """Poll ``console`` and upload on schedule or after sustained loss of progress.

    Polling is free; committing is not. Heartbeats spend 4/hour and the steady console cadence uses
    the remaining 1/hour in this run's allocation. Repository-wide admission is outside this worker;
    this loop only bounds its own steady rate and two fixed emergency credits. The stall classifier
    kills a wedged run at 1200s/3000s, before an hourly snapshot can preserve its diagnostics.

    ``armed`` means real progress was observed and then stopped. Before the first committed
    heartbeat, an uncommitted heartbeat may arm the loop because it is the only proof setup reached
    the worker. After one commit, only committed heartbeats count because the provider's stall clock is
    anchored to what reached the artifact repo. Progress re-arms the latch; the per-run credit cap
    prevents a flapping run from turning the poll cadence into the commit cadence.

    Before any heartbeat commits, one additional setup snapshot lands before the fixed 3000-second
    teardown. It does not enter the steady rate: the first committed heartbeat moves the deadline to
    the normal interval. failed uploads change no upload state, so the next poll retries.
    """
    poll_s = min(_CONSOLE_UPLOAD_POLL_S, interval_s)
    due_s = min(_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S, interval_s)
    sent = cursor = eof = -1
    since, quiet, armed, spent, ever = 0.0, 0.0, False, 0, False
    while not stop_upload.wait(poll_s):
        since += poll_s
        cursor, eof, staged, beats = progress(console, max(cursor, 0))
        had_committed = ever
        ever = ever or bool(staged)
        if ever and not had_committed and sent >= 0:
            due_s = interval_s
        made_progress = staged if ever else beats
        armed = armed or bool(made_progress)
        quiet = 0.0 if made_progress else quiet + 1
        due = since >= due_s
        cap = armed and quiet >= _CONSOLE_UPLOAD_QUIET_POLLS
        wedged = cap and spent < _CONSOLE_UPLOAD_CREDITS and not due
        if eof == sent or not (due or wedged):
            continue
        if not upload():
            continue
        spent += 1 if cap and not due else 0
        if cap:
            armed, quiet = False, 0.0
        due_s = interval_s if ever else min(1800.0, interval_s)
        sent, since = eof, 0.0
