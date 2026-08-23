from __future__ import annotations

from flash.providers._lifecycle.bootstrapping import console as console


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


def test_console_quiet_trigger_rearms_only_until_the_credit_cap(monkeypatch):
    """a flapping run gets exactly _CONSOLE_UPLOAD_CREDITS emergency writes, not one extra.

    committing at poll 1 promotes the deadline to the hourly interval, so every later write here is
    an emergency one and each spends a credit. before the promotion was unconditional the 600-second
    deadline was still pending at poll 5, which classified that write as scheduled and let a flapping
    run take 3 writes out of a 2-credit cap.
    """
    rows = []
    for poll in range(1, 21):
        committed = int(poll in {1, 6, 11, 16})
        rows.append((poll, poll, committed, committed))
    _waits, uploads = _drive(monkeypatch, rows)
    assert uploads == [5, 10]
    assert len(uploads) == console._CONSOLE_UPLOAD_CREDITS


def test_healthy_run_committing_before_600s_skips_the_setup_snapshot(monkeypatch):
    """the 600-second snapshot is startup evidence, so a run that already committed must not pay it.

    it costs a repository write per run on every healthy run, and the first-hour budget is 4
    heartbeat writes plus 1 console write. the fallback still fires when nothing has committed by
    then, which the setup half of the cadence test covers.
    """
    healthy = [(n, n, 1, 1) for n in range(1, 30)]
    _waits, uploads = _drive(monkeypatch, healthy)
    assert uploads == []

    # the same run one poll past the hourly deadline takes the steady write, and only that one.
    hourly = [(n, n, 1, 1) for n in range(1, 32)]
    _waits, uploads = _drive(monkeypatch, hourly)
    assert uploads == [30]


def test_first_commit_landing_on_the_600s_deadline_still_spends_that_snapshot(monkeypatch):
    """promotion is read after ``due``, so it only ever moves the NEXT deadline.

    this is the one poll where the ordering is observable: the 600-second deadline is already due
    and the first commit lands on that same poll. reading the promotion first would cancel a
    deadline that had already come due, losing the startup snapshot -- the only evidence covering
    everything before the first commit. every other script agrees under both orderings, so without
    this case the ordering is documented but unproven.
    """
    rows = [(n, n, 0, 0) for n in range(1, 5)]
    rows += [(n, n, 1, 1) for n in range(5, 12)]
    _waits, uploads = _drive(monkeypatch, rows)
    assert uploads == [5]


def test_a_failed_setup_snapshot_retries_on_the_next_poll_after_promotion(monkeypatch):
    """promoting the deadline is upload state, so a failed upload must roll it back.

    same poll as the test above -- the first commit lands while the 600-second deadline is already
    due -- but the upload fails. promotion that survives the failure moves the deadline to the hourly
    interval with ``since`` untouched, so the retry does not come at the next poll but an hour later,
    long after the 3000-second teardown would have killed the run. the startup snapshot exists to
    capture exactly that window, so losing it loses the evidence for runs that never reach step 1.
    """
    rows = [(n, n, 0, 0) for n in range(1, 5)]
    rows += [(n, n, 1, 1) for n in range(5, 32)]
    _waits, uploads = _drive(monkeypatch, rows, outcomes=[False])
    # _drive records attempts: poll 5 is the one that failed, poll 6 is the retry it must not lose.
    assert uploads == [5, 6]


def test_pending_heartbeat_arms_only_before_committed_progress(monkeypatch):
    rows = []
    for poll in range(1, 16):
        committed = int(poll in {1, 11})
        pending = int(poll == 6)
        rows.append((poll, poll, committed, committed + pending))
    _waits, uploads = _drive(monkeypatch, rows)
    assert uploads == [5, 15]
