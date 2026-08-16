"""console scanning and upload cadence shared by instance bootstrap callers.

this module owns console reads, heartbeat scanning, and the wedge state machine. callers provide the
artifact upload callback, keeping the module importable when shipped bare on a rented box.
"""

from __future__ import annotations

import json
import os

_CONSOLE_PROGRESS_READ_LIMIT = 1_048_576
_CONSOLE_PROGRESS_LINE_LIMIT = 64_000
_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
_CONSOLE_UPLOAD_POLL_S = 120.0
_CONSOLE_UPLOAD_QUIET_POLLS = 4
_CONSOLE_UPLOAD_CREDITS = 2


def new_progress_state() -> dict:
    """Fresh cursor for :func:`console_progress`; one per loop, never shared between consoles."""
    return {"offset": 0, "partial": b"", "dropping": False}


def console_progress(path: str, state: dict) -> tuple[int, int, int]:
    """Return ``(observed_eof, committed, any)`` for console text appended since the last call.

    ``state`` carries the parse cursor across calls so each line is judged exactly once: reads are
    capped, a partial trailing line stays buffered until its newline arrives, and a line longer than
    the limit is dropped through to its newline rather than buffered without bound. Only TOP-LEVEL
    ``pending``/``throttled`` keys demote a heartbeat, so a sampled completion that happens to
    contain those words as nested data cannot fake a committed beat. Returns ``-1`` for eof when the
    console cannot be read, which the caller treats as "no new bytes" rather than progress.
    """
    committed = beats = 0
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            eof, offset = handle.tell(), state["offset"]
            if eof < offset:
                state.update(offset=0, partial=b"", dropping=False)
                offset = 0
            while offset < eof:
                handle.seek(offset)
                chunk = handle.read(min(_CONSOLE_PROGRESS_READ_LIMIT, eof - offset))
                if not chunk:
                    break
                offset += len(chunk)
                lines = (state["partial"] + chunk).split(b"\n")
                state["partial"] = lines.pop()
                if state["dropping"]:
                    if not lines:
                        state["partial"] = b""
                        state["offset"] = offset
                        continue
                    state["dropping"], lines = False, lines[1:]
                if len(state["partial"]) > _CONSOLE_PROGRESS_LINE_LIMIT:
                    state.update(partial=b"", dropping=True)
                for line in lines:
                    if len(line) > _CONSOLE_PROGRESS_LINE_LIMIT or not line.startswith(
                        b"HEARTBEAT "
                    ):
                        continue
                    try:
                        beat = json.loads(line[len(b"HEARTBEAT ") :])
                    except (TypeError, ValueError):
                        continue
                    if isinstance(beat, dict) and not beat.get("liveness"):
                        beats += 1
                        committed += not {"pending", "throttled"} & set(beat)
                state["offset"] = offset
    except OSError:
        return -1, 0, 0
    return eof, committed, beats


def run_console_upload_loop(console: str, interval_s: float, stop_upload, *, upload) -> None:
    """Poll ``console`` and upload on schedule or after sustained loss of progress.

    Polling is free; committing is not. Heartbeats spend 4/hour and the steady console cadence uses
    the remaining 1/hour in this run's allocation. Repository-wide admission is outside this worker;
    this loop only bounds its own steady rate and two fixed emergency credits. The stall classifier
    kills a wedged run at 1200s/3000s, before an hourly snapshot could preserve its diagnostics.

    ``armed`` means real progress was observed and then stopped. Before the first committed
    heartbeat, an uncommitted heartbeat may arm the loop because it is the only proof setup reached
    the worker. After one commit, only committed heartbeats count, because the provider's stall clock
    is anchored to what actually reached the artifact repo. Progress re-arms the latch; the per-run
    credit cap keeps a flapping run from turning the poll cadence into the commit cadence.

    Before any heartbeat commits, one fallback snapshot lands ahead of the fixed teardown. It does
    not enter the steady rate: the first committed heartbeat promotes the deadline to the normal
    interval. A failed upload advances neither the watermark nor the deadline, so the next poll
    retries it.
    """
    poll_s = min(_CONSOLE_UPLOAD_POLL_S, interval_s)
    due_s = min(_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S, interval_s)
    sent, eof = -1, -1
    since, quiet, armed, spent, ever = 0.0, 0.0, False, 0, False
    state = new_progress_state()
    while not stop_upload.wait(poll_s):
        since += poll_s
        eof, staged, beats = console_progress(console, state)
        # a first commit arriving before the fallback is due promotes straight to the steady
        # interval, so a healthy run never spends the 600s write at all.
        if staged and not ever and since < due_s:
            due_s = interval_s
        ever = ever or bool(staged)
        made_progress = staged if ever else beats
        armed = armed or bool(made_progress)
        quiet = 0.0 if made_progress else quiet + 1
        due = since >= due_s
        wedged = (
            armed
            and not due
            and spent < _CONSOLE_UPLOAD_CREDITS
            and quiet >= _CONSOLE_UPLOAD_QUIET_POLLS
        )
        if eof == sent or not (due or wedged):
            continue
        if not upload():
            continue
        spent += 1 if wedged else 0
        armed = armed and not wedged
        sent, since, due_s = eof, 0.0, interval_s
