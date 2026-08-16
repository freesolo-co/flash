"""Hermetic branch coverage for isolated structured-output validation.

The spawned-process failure paths only: a child that will not start, one that dies without
answering, and each status it can report. These drive `_Context` fakes rather than a real spawn, so
they cover branches the real-subprocess tests in `test_deployment_smoke.py` cannot reach.
"""

from __future__ import annotations

import pytest

import flash.server.domain.deployment_smoke as smoke
from flash.serve.deploy import ServingError


class _Connection:
    def __init__(self, *, outcome=None, eof: bool = False) -> None:
        self.sent = []
        self.closed = False
        self.outcome = outcome
        self.eof = eof

    def send(self, value) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True

    def poll(self, _timeout: float) -> bool:
        return self.outcome is not None or self.eof

    def recv(self):
        if self.eof:
            raise EOFError
        return self.outcome


class _Process:
    def __init__(self, *, start_error: Exception | None = None) -> None:
        self.start_error = start_error
        self.pid = None
        self.closed = False

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error

    def close(self) -> None:
        self.closed = True


class _Context:
    def __init__(self, receive: _Connection, process: _Process) -> None:
        self.receive = receive
        self.send = _Connection()
        self.process = process

    def Pipe(self, duplex: bool):
        assert duplex is False
        return self.receive, self.send

    def Process(self, **kwargs):
        assert kwargs["target"] is smoke.json_schema_validation_worker
        assert kwargs["name"] == smoke.JSON_SCHEMA_PROCESS_NAME
        assert kwargs["daemon"] is True
        return self.process


def _install_context(monkeypatch, *, outcome=None, eof=False, start_error=None):
    receive = _Connection(outcome=outcome, eof=eof)
    process = _Process(start_error=start_error)
    context = _Context(receive, process)
    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda method: context)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 10.0)
    return receive, context.send, process


def test_validate_json_schema_reports_process_start_failure(monkeypatch) -> None:
    """A child start failure must close both pipe ends and expose a safe serving error."""
    receive, send, process = _install_context(monkeypatch, start_error=RuntimeError("spawn denied"))

    with pytest.raises(ServingError, match="could not start isolated JSON schema validation"):
        smoke.validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)

    assert receive.closed is True
    assert send.closed is True
    assert process.closed is False


def test_validate_json_schema_maps_start_failure_after_deadline(monkeypatch) -> None:
    """A spawn failure that consumes the global budget must report the smoke timeout instead."""
    ticks = iter([10.0, 11.0])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(ticks))
    receive = _Connection()
    process = _Process(start_error=RuntimeError("late spawn"))
    context = _Context(receive, process)
    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda method: context)

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        smoke.validate_json_schema({}, {}, deadline=10.5, budget_s=0.5)

    assert receive.closed is True
    assert context.send.closed is True


def test_validate_json_schema_treats_child_eof_as_safe_failure(monkeypatch) -> None:
    """A child that exits without an outcome must fail closed rather than validating the instance."""
    receive, send, _process = _install_context(monkeypatch, eof=True)

    with pytest.raises(ServingError, match="wall-clock deadline"):
        smoke.validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)

    assert receive.closed is True
    assert send.closed is True


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (("ok", ""), None),
        (("schema", "bad schema"), "configured JSON schema is invalid"),
        (("validation", "bad value"), "structured smoke output violates"),
        (("error", "safe detail"), "JSON schema validation failed safely"),
        (("reference", "unresolvable"), "reference could not be resolved"),
    ],
)
def test_validate_json_schema_maps_child_statuses(monkeypatch, outcome, message) -> None:
    """Each child status must map to the public validation contract without a real subprocess."""
    _install_context(monkeypatch, outcome=outcome)

    if message is None:
        smoke.validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)
    else:
        with pytest.raises(ServingError, match=message):
            smoke.validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)
