"""Hermetic branch coverage for isolated structured-output validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import flash.server.routes.serving as serving
from flash.serve.contract.errors import ServingError


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
        assert kwargs["target"] is serving._json_schema_validation_worker
        assert kwargs["name"] == serving._JSON_SCHEMA_PROCESS_NAME
        assert kwargs["daemon"] is True
        return self.process


def test_bounded_regex_rejects_an_expired_global_deadline(monkeypatch) -> None:
    """An exhausted smoke budget must fail before regex evaluation can start."""
    monkeypatch.setattr(serving.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        serving.safe_regex,
        "fullmatch",
        lambda *args, **kwargs: pytest.fail("expired budgets must not evaluate regexes"),
    )

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving._bounded_regex_fullmatch("a", "a", deadline=10.0, budget_s=2.0)


def test_bounded_regex_maps_a_global_timeout_to_the_smoke_budget(monkeypatch) -> None:
    """A regex timeout constrained by the global deadline must retain the global error contract."""
    monkeypatch.setattr(serving.time, "monotonic", lambda: 10.0)

    def timeout(*args, **kwargs):
        raise TimeoutError("slow regex")

    monkeypatch.setattr(serving.safe_regex, "fullmatch", timeout)
    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving._bounded_regex_fullmatch("(a+)+$", "a!", deadline=10.01, budget_s=1.0)


def test_schema_error_sanitization_collapses_and_bounds_untrusted_text() -> None:
    """Schema diagnostics must be single-line and bounded before reaching users or logs."""
    exc = SimpleNamespace(message=("  unsafe\n\ttext  " * 100))
    message = serving._sanitized_schema_error(exc)

    assert "\n" not in message
    assert "\t" not in message
    assert len(message) == 500
    assert message.startswith("unsafe text unsafe text")


@pytest.mark.parametrize(
    ("instance", "schema", "status"),
    [
        ({"answer": 4}, {"type": "object", "required": ["answer"]}, "ok"),
        ({}, {"type": "object", "required": ["answer"]}, "validation"),
        ({}, {"type": 4}, "schema"),
        ({}, {"$ref": "https://schemas.invalid/missing.json"}, "reference"),
    ],
)
def test_json_schema_worker_reports_expected_outcomes(instance, schema, status) -> None:
    """The worker must classify normal schema outcomes and always close its connection."""
    connection = _Connection()

    serving._json_schema_validation_worker(connection, instance, schema)

    assert connection.closed is True
    assert len(connection.sent) == 1
    assert connection.sent[0][0] == status


def test_json_schema_worker_safely_classifies_unexpected_failures(monkeypatch) -> None:
    """Unexpected validator failures must become inert error tuples instead of escaping the worker."""
    connection = _Connection()

    def boom(_schema):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(serving, "validator_for", boom)
    serving._json_schema_validation_worker(connection, {}, {})

    assert connection.sent == [("error", "RuntimeError: validator exploded")]
    assert connection.closed is True


def _install_context(monkeypatch, *, outcome=None, eof=False, start_error=None):
    receive = _Connection(outcome=outcome, eof=eof)
    process = _Process(start_error=start_error)
    context = _Context(receive, process)
    monkeypatch.setattr(serving.multiprocessing, "get_context", lambda method: context)
    monkeypatch.setattr(serving.time, "monotonic", lambda: 10.0)
    return receive, context.send, process


def test_validate_json_schema_reports_process_start_failure(monkeypatch) -> None:
    """A child start failure must close both pipe ends and expose a safe serving error."""
    receive, send, process = _install_context(monkeypatch, start_error=RuntimeError("spawn denied"))

    with pytest.raises(ServingError, match="could not start isolated JSON schema validation"):
        serving._validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)

    assert receive.closed is True
    assert send.closed is True
    assert process.closed is False


def test_validate_json_schema_maps_start_failure_after_deadline(monkeypatch) -> None:
    """A spawn failure that consumes the global budget must report the smoke timeout instead."""
    ticks = iter([10.0, 11.0])
    monkeypatch.setattr(serving.time, "monotonic", lambda: next(ticks))
    receive = _Connection()
    process = _Process(start_error=RuntimeError("late spawn"))
    context = _Context(receive, process)
    monkeypatch.setattr(serving.multiprocessing, "get_context", lambda method: context)

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving._validate_json_schema({}, {}, deadline=10.5, budget_s=0.5)

    assert receive.closed is True
    assert context.send.closed is True


def test_validate_json_schema_treats_child_eof_as_safe_failure(monkeypatch) -> None:
    """A child that exits without an outcome must fail closed rather than validating the instance."""
    receive, send, _process = _install_context(monkeypatch, eof=True)

    with pytest.raises(ServingError, match="wall-clock deadline"):
        serving._validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)

    assert receive.closed is True
    assert send.closed is True


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (("ok", ""), None),
        (("schema", "bad schema"), "configured JSON schema is invalid"),
        (("validation", "bad value"), "structured smoke output violates"),
        (("error", "safe detail"), "JSON schema validation failed safely"),
    ],
)
def test_validate_json_schema_maps_child_statuses(monkeypatch, outcome, message) -> None:
    """Each child status must map to the public validation contract without a real subprocess."""
    _install_context(monkeypatch, outcome=outcome)

    if message is None:
        serving._validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)
    else:
        with pytest.raises(ServingError, match=message):
            serving._validate_json_schema({}, {}, deadline=20.0, budget_s=5.0)
