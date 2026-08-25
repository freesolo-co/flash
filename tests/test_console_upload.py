from __future__ import annotations

from flash.providers._lifecycle.bootstrapping import console


def test_console_size_reports_bytes_and_missing_file(tmp_path) -> None:
    path = tmp_path / "console.txt"
    assert console._console_size(str(path)) == -1
    path.write_bytes(b"worker output\n")
    assert console._console_size(str(path)) == len(b"worker output\n")


def _drive(monkeypatch, sizes, outcomes=None):
    waits: list[float] = []
    uploads: list[int] = []
    state = {"index": 0}
    results = iter(outcomes or [])

    class Stop:
        def wait(self, seconds: float) -> bool:
            waits.append(seconds)
            return state["index"] >= len(sizes)

    def size(_path: str) -> int:
        value = sizes[state["index"]]
        state["index"] += 1
        return value

    def upload() -> bool:
        uploads.append(state["index"])
        return next(results, True)

    monkeypatch.setattr(console, "_console_size", size)
    console._run_console_upload_loop("unused", 3600.0, Stop(), upload=upload)
    return waits, uploads


def test_console_upload_uses_fixed_early_and_hourly_cadence(monkeypatch) -> None:
    waits, uploads = _drive(monkeypatch, list(range(1, 36)))
    assert set(waits) == {120.0}
    assert uploads == [5, 35]


def test_console_upload_retries_failed_due_snapshot(monkeypatch) -> None:
    _waits, uploads = _drive(monkeypatch, list(range(1, 7)), outcomes=[False, True])
    assert uploads == [5, 6]


def test_console_upload_skips_unchanged_bytes(monkeypatch) -> None:
    sizes = [1] * 34 + [2]
    _waits, uploads = _drive(monkeypatch, sizes)
    assert uploads == [5, 35]
