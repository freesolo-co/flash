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

    Polling is free; committing is not. Heartbeats already spend 4 of the shared repo's 5
    commits/hour, while the stall classifier kills a wedged run at 1200s/3000s, before an hourly
    snapshot. An upload therefore requires unseen bytes and either a due schedule or a bounded
    wedge credit.

    ``armed`` means real progress was observed and then stopped. Before the first committed
    heartbeat, a pending heartbeat may arm the loop because it is the only proof setup reached the
    worker. After one commit, only committed heartbeats count because the provider's stall clock is
    anchored to what reached the artifact repo. Progress re-arms the latch; the per-run credit cap
    prevents a flapping run from turning the poll cadence into the commit cadence.

    A setup that never reaches a heartbeat stays uncovered deliberately: holding the first-snapshot
    cadence until progress starts costs 5.25 commits/hour against a hard 5.0. Failed uploads advance
    neither the size marker nor the deadline, so the next poll retries before teardown.
    """
    poll_s = min(_CONSOLE_UPLOAD_POLL_S, interval_s)
    due_s = min(_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S, interval_s)
    sent, size, since, quiet, armed, spent, ever = -1, -1, 0.0, 0.0, False, 0, False
    while not stop_upload.wait(poll_s):
        since += poll_s
        size, staged, beats = progress(console, max(size, 0))
        ever = ever or bool(staged)
        made_progress = staged if ever else beats
        armed = armed or bool(made_progress)
        quiet = 0.0 if made_progress else quiet + 1
        due = since >= due_s
        wedged = armed and not due and spent < _CONSOLE_UPLOAD_CREDITS
        wedged = wedged and quiet >= _CONSOLE_UPLOAD_QUIET_POLLS
        if size == sent or not (due or wedged):
            continue
        uploaded = upload()
        spent += 1 if wedged and uploaded else 0
        armed = armed and not (wedged and uploaded)
        if uploaded:
            sent, since, due_s = size, 0.0, interval_s
