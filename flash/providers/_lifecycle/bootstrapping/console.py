"""bounded console snapshot cadence shared by instance bootstrap callers."""

from __future__ import annotations

import os

_CONSOLE_UPLOAD_INTERVAL_S = 3600.0
_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S = 600.0
_CONSOLE_UPLOAD_POLL_S = 120.0


def _console_size(console: str) -> int:
    try:
        return os.path.getsize(console)
    except OSError:
        return -1


def _run_console_upload_loop(console: str, interval_s: float, stop_upload, *, upload) -> None:
    """upload one early snapshot and then at a fixed bounded cadence when bytes changed."""
    poll_s = min(_CONSOLE_UPLOAD_POLL_S, interval_s)
    due_s = min(_CONSOLE_UPLOAD_FIRST_SNAPSHOT_S, interval_s)
    since = 0.0
    sent_size = -1
    while not stop_upload.wait(poll_s):
        since += poll_s
        size = _console_size(console)
        if size < 0 or size == sent_size or since < due_s:
            continue
        if not upload():
            continue
        sent_size = size
        since = 0.0
        due_s = interval_s
