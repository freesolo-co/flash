"""console scanning and upload cadence shared by instance bootstrap callers.

this module owns console reads, scanning, and the wedge state machine. callers provide artifact
upload callbacks, keeping the module importable when shipped bare on a rented box.
"""

from __future__ import annotations

import os
import re

_CONSOLE_SCAN_BYTES = 1_048_576
_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
_CONSOLE_UPLOAD_POLL_S = 120.0
_CONSOLE_UPLOAD_QUIET_POLLS = 4
_CONSOLE_UPLOAD_CREDITS = 2


def _console_progress(console: str, offset: int) -> tuple[int, int, int, int]:
    """Return ``(scan_cursor, observed_eof, committed, any)`` at the captured tail.

    reads are capped. oversized lines skip in chunks; partial lines remain behind the cursor until
    newline so they are judged once, while observed eof still exposes them to snapshot uploads.
    """
    try:
        with open(console, "rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            at = min(max(offset, 0), end)
            handle.seek(max(at - 1, 0))
            start = not at or handle.read(1) == b"\n"
            hits = beats = 0
            while at < end:
                handle.seek(at)
                buf = handle.read(min(_CONSOLE_SCAN_BYTES, end - at))
                if not buf:
                    break
                if not start:
                    newline = buf.find(b"\n") + 1
                    if not newline:
                        at += len(buf)
                        continue
                    at += newline
                    buf, start = buf[newline:], True
                    if not buf:
                        continue
                cut = buf.rfind(b"\n") + 1
                if not cut:
                    if at + len(buf) < end:
                        at += len(buf)
                        start = False
                        continue
                    break
                heartbeats = re.findall(rb'(?m)^HEARTBEAT (?!.*"liveness":).*$', buf[:cut])
                hits += sum(
                    b'"pending":' not in line and b'"throttled":' not in line for line in heartbeats
                )
                beats += len(heartbeats)
                at += cut
    except OSError:
        return -1, -1, 0, 0
    return at, end, hits, beats


def _run_console_upload_loop(console: str, interval_s: float, stop_upload, *, upload) -> None:
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
    teardown. It does not enter the steady rate: the first committed heartbeat promotes the deadline
    to the normal interval, so a run that commits before 600 seconds never spends that write. the
    promotion is skipped while ``due`` holds, so it only ever moves a deadline that has not arrived.
    A 600-second deadline that already expired with nothing committed still spends its snapshot --
    the startup evidence it exists for -- and a failed upload on that same poll leaves the deadline
    due, so the next poll retries rather than waiting out the hour. Once promoted the deadline is the
    hourly interval, so the guard cannot re-fire early and needs no separate first-commit flag.
    """
    poll_s = min(_CONSOLE_UPLOAD_POLL_S, interval_s)
    due_s = min(_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S, interval_s)
    sent = cursor = eof = -1
    since, quiet, armed, spent, ever = 0.0, 0.0, False, 0, False
    while not stop_upload.wait(poll_s):
        since += poll_s
        cursor, eof, staged, beats = _console_progress(console, max(cursor, 0))
        ever = ever or bool(staged)
        due = since >= due_s
        if ever and not due:
            due_s = interval_s
        made_progress = staged if ever else beats
        armed = armed or bool(made_progress)
        quiet = 0.0 if made_progress else quiet + 1
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
