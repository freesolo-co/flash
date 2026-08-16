from __future__ import annotations

from flash.providers._lifecycle import bootstrap_console as console


def test_console_progress_classifies_complete_lines_and_tracks_observed_eof(tmp_path):
    path = tmp_path / "console.txt"
    path.write_bytes(
        b"prefix HEARTBEAT {}\n"
        b'HEARTBEAT {"pending": true}\n'
        b'HEARTBEAT {"throttled": true}\n'
        b'HEARTBEAT {"liveness": true}\n'
        b'HEARTBEAT {"stage": "rl_step"}\n'
        b'HEARTBEAT {"stage": "partial"}'
    )

    cursor, eof, committed, beats = console._console_progress(str(path), 0)
    assert (committed, beats) == (1, 3)
    assert cursor < eof == path.stat().st_size

    with path.open("ab") as handle:
        handle.write(b"\n")
    cursor2, eof2, committed2, beats2 = console._console_progress(str(path), cursor)
    assert (cursor2, eof2, committed2, beats2) == (path.stat().st_size, path.stat().st_size, 1, 1)

    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * (console._CONSOLE_SCAN_BYTES + 17) + b"\nHEARTBEAT {}\n")
    assert console._console_progress(str(oversized), 0)[2:] == (1, 1)


def _drive(monkeypatch, rows, outcomes=None):
    waits: list[float] = []
    uploads: list[int] = []
    state = {"index": 0}
    results = iter(outcomes or [])

    class Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return state["index"] >= len(rows)

    def progress(_path: str, _offset: int):
        row = rows[state["index"]]
        state["index"] += 1
        return row

    def upload() -> bool:
        uploads.append(state["index"])
        return next(results, True)

    monkeypatch.setattr(console, "_console_progress", progress)
    console._run_console_upload_loop("unused", 3600.0, Stop(), upload=upload)
    return waits, uploads


def test_console_upload_cadence_covers_setup_and_steady_state(monkeypatch):
    setup = [(n, n, 0, 0) for n in range(1, 21)]
    waits, uploads = _drive(monkeypatch, setup)
    assert set(waits) == {120.0}
    assert uploads == [5, 20]

    steady = [(n, n, int(n == 1), int(n == 1)) for n in range(1, 36)]
    _waits, uploads = _drive(monkeypatch, steady)
    assert uploads == [5, 35]


def test_console_upload_retries_failure_using_observed_eof(monkeypatch):
    rows = [(0, eof, 0, 0) for eof in range(1, 7)]
    _waits, uploads = _drive(monkeypatch, rows, outcomes=[False, True])
    assert uploads == [5, 6]


def test_console_quiet_trigger_rearms_twice_without_spending_scheduled_overlap(monkeypatch):
    rows = []
    for poll in range(1, 21):
        committed = int(poll in {1, 6, 11, 16})
        rows.append((poll, poll, committed, committed))
    _waits, uploads = _drive(monkeypatch, rows)
    assert uploads == [5, 10, 15]


def test_pending_heartbeat_arms_only_before_committed_progress(monkeypatch):
    rows = []
    for poll in range(1, 16):
        committed = int(poll in {1, 11})
        pending = int(poll == 6)
        rows.append((poll, poll, committed, committed + pending))
    _waits, uploads = _drive(monkeypatch, rows)
    assert uploads == [5, 15]
