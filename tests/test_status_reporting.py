"""Delivery coordination across reporter cancellation and reopening."""

from threading import Event
from types import SimpleNamespace

import pytest

from flash.runner.lifecycle import reporting


@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("sequence", [0, None, True, -1, "7", 1.5, {"corrupt": True}])
def test_invalid_sequence_is_rejected_without_inventing_a_number(monkeypatch, wait, sequence):
    reporting._shutdown_status_reporter()
    reporting._open_status_reporter()
    monkeypatch.setattr(
        reporting,
        "_send_status_report",
        lambda _status: pytest.fail("invalid sequence must not be delivered"),
    )
    status = SimpleNamespace(run_id="run-invalid-sequence", report_sequence=sequence)
    with pytest.raises(ValueError, match="persisted positive report_sequence"):
        reporting._queue_status_report(status, wait=wait)
    assert reporting._wait_for_status_reports(0)


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("in_flight_succeeds", [False, True])
def test_cancelled_report_can_be_requeued_before_in_flight_delivery_finishes(
    monkeypatch, close, in_flight_succeeds
):
    reporting._shutdown_status_reporter()
    reporting._open_status_reporter()
    first_started = Event()
    release_first = Event()
    second_started = Event()
    release_second = Event()
    attempts = []

    def status(sequence):
        return SimpleNamespace(run_id="run-cancel-requeue", report_sequence=sequence)

    def send(snapshot):
        sequence = snapshot.report_sequence
        attempts.append(sequence)
        if sequence == 1:
            first_started.set()
            assert release_first.wait(5)
        if sequence == 2:
            second_started.set()
            assert release_second.wait(5)
            return in_flight_succeeds
        return True

    monkeypatch.setattr(reporting, "_send_status_report", send)
    try:
        reporting._report_status_async(status(1))
        assert first_started.wait(2)
        reporting._report_status_async(status(2))
        reporting._report_status_async(status(3))
        release_first.set()
        assert second_started.wait(2)

        # Cancel #3 while #2 is in flight. #1 has already been acknowledged.
        assert reporting._shutdown_status_reporter(0, close=close) is False
        if close:
            reporting._open_status_reporter()
        reporting._report_status(status(1))
        reporting._report_status_async(status(3))
        assert attempts == [1, 2]
        assert reporting._wait_for_status_reports(0) is False

        release_second.set()
        assert reporting._wait_for_status_reports(5)
        assert attempts == ([1, 2, 3] if in_flight_succeeds else [1, 2, 2, 3])
    finally:
        release_first.set()
        release_second.set()
        reporting._shutdown_status_reporter()
