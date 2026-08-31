"""Process ownership coverage for deployment-smoke network calls."""

from __future__ import annotations

import ast
import inspect
import json
import multiprocessing
import os
import signal
import socket
import struct
import threading
import time
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from flash.serve.contract.errors import RetryableServingUnavailable, ServingError
from flash.server.platform import children, ipc
from flash.server.routes import serving_smoke

_CHECKPOINT_ID = "run-1/final"
_SECRET = "smoke-secret-that-must-not-enter-argv"


def _chat_kwargs() -> dict:
    return {
        "run_id": _CHECKPOINT_ID,
        "messages": [{"role": "user", "content": "what is 2+2?"}],
        "temperature": 0.0,
        "max_tokens": 32,
        "thinking": False,
        "expected_checkpoint": _CHECKPOINT_ID,
        "org_id": "org-1",
        "timeout_s": 5.0,
        "retry_unavailable": True,
        "stop": None,
    }


def _smoke_child_pids() -> set[int | None]:
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.name == serving_smoke._SMOKE_CHAT_PROCESS_NAME
    }


def _ignore_terminate(ready) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(1.0)


def _faulty_frame_worker(connection, fault: str) -> None:
    try:
        if fault == "oversized":
            os.write(
                connection.fileno(),
                struct.pack("!I", ipc._IPC_MAX_PAYLOAD_BYTES + 1),
            )
        elif fault == "malformed":
            payload = b"{"
            os.write(connection.fileno(), struct.pack("!I", len(payload)) + payload)
        else:
            os.write(connection.fileno(), struct.pack("!I", 8) + b"{")
            time.sleep(10.0)
    finally:
        connection.close()


class _FaultyFrameContext:
    def __init__(self, context, fault: str) -> None:
        self._context = context
        self._fault = fault

    def Pipe(self, *, duplex: bool):
        return self._context.Pipe(duplex=duplex)

    def Process(self, **kwargs):
        return self._context.Process(
            target=_faulty_frame_worker,
            args=(kwargs["args"][0], self._fault),
            name=kwargs["name"],
            daemon=kwargs["daemon"],
        )


def _named_child_pids(name: str) -> set[int | None]:
    return {child.pid for child in multiprocessing.active_children() if child.name == name}


def _process_cmdline(pid: int) -> bytes:
    return Path(f"/proc/{pid}/cmdline").read_bytes()


def _record_raw_ipc(monkeypatch) -> list[object]:
    outcomes: list[object] = []
    original_receive = serving_smoke._receive_framed_ipc

    def record(connection, *, deadline, description):
        outcome = original_receive(connection, deadline=deadline, description=description)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(serving_smoke, "_receive_framed_ipc", record)
    return outcomes


def _assert_ipc_redacted(outcomes: list[object]) -> None:
    assert outcomes
    assert _SECRET not in repr(outcomes)


def _framed_round_trip(value):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    try:
        ipc._send_framed_ipc(
            send,
            value,
            deadline=time.monotonic() + 1.0,
            description="test worker",
        )
    finally:
        send.close()
    try:
        return ipc._receive_framed_ipc(
            receive,
            deadline=time.monotonic() + 1.0,
            description="test worker",
        )
    finally:
        receive.close()


def test_framed_ipc_full_pipe_write_obeys_absolute_deadline() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        while True:
            try:
                os.write(write_fd, b"x" * 4096)
            except BlockingIOError:
                break
        started = time.monotonic()
        with pytest.raises(ipc._IpcDeadlineExceeded):
            ipc._write_all(
                write_fd,
                b"z",
                deadline=started + 0.05,
                description="test worker",
            )
        assert time.monotonic() - started < 0.5
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_framed_ipc_partial_write_rechecks_the_absolute_deadline(monkeypatch) -> None:
    clock = [10.0]
    writes = []

    monkeypatch.setattr(ipc.os, "get_blocking", lambda _fd: False)
    monkeypatch.setattr(ipc.select, "select", lambda *_args: ([], [99], []))
    monkeypatch.setattr(ipc.time, "monotonic", lambda: clock[0])

    def partial_write(_fd, value):
        writes.append(bytes(value))
        clock[0] = 11.0
        return 1

    monkeypatch.setattr(ipc.os, "write", partial_write)

    with pytest.raises(ipc._IpcDeadlineExceeded):
        ipc._write_all(
            99,
            b"abc",
            deadline=10.5,
            description="test worker",
        )

    assert writes == [b"abc"]


def test_framed_ipc_rejects_oversize_before_writing_header() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)

    class Connection:
        def fileno(self) -> int:
            return write_fd

    try:
        with pytest.raises(ServingError, match="oversized IPC frame"):
            ipc._send_framed_ipc(
                Connection(),
                "x" * ipc._IPC_MAX_PAYLOAD_BYTES,
                deadline=time.monotonic() + 1.0,
                description="test worker",
            )
        with pytest.raises(BlockingIOError):
            os.read(read_fd, 1)
    finally:
        os.close(write_fd)
        os.close(read_fd)


@contextmanager
def _http_server(
    *,
    status: int = 200,
    hold: threading.Event | None = None,
    response_body: bytes | None = None,
    echo_secret_error: bool = False,
):
    requests: list[dict] = []
    request_started = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(
                {
                    "path": self.path,
                    "key": self.headers.get("X-Freesolo-Internal-Key"),
                    "body": json.loads(self.rfile.read(length)),
                    "cmdlines": [
                        _process_cmdline(child.pid)
                        for child in multiprocessing.active_children()
                        if child.name == serving_smoke._SMOKE_CHAT_PROCESS_NAME
                        and child.pid is not None
                    ],
                }
            )
            request_started.set()
            if hold is not None:
                hold.wait(timeout=10.0)
            if response_body is not None:
                encoded = response_body
            elif status == 200:
                encoded = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "The answer is 4"}, "finish_reason": "stop"}
                        ],
                        "freesolo": {"checkpoint_id": _CHECKPOINT_ID},
                    }
                ).encode()
            else:
                encoded = json.dumps({"error": {"message": "request rejected"}}).encode()
            reason = f"rejected-{_SECRET}" if echo_secret_error else None
            self.send_response(status, message=reason)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            if echo_secret_error:
                self.send_header(f"X-{_SECRET}", _SECRET)
            if status == 200:
                self.send_header("X-Freesolo-Checkpoint", _CHECKPOINT_ID)
            self.end_headers()
            with suppress(BrokenPipeError):
                self.wfile.write(encoded)

        def log_message(self, _format: str, *_args) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], requests, request_started
    finally:
        if hold is not None:
            hold.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_isolated_smoke_chat_preserves_success_and_cleans_process(monkeypatch) -> None:
    children_before = _smoke_child_pids()
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)

    with _http_server() as (port, requests, _started):
        monkeypatch.setenv("FREESOLO_SERVING_URL", f"http://127.0.0.1:{port}")
        result = serving_smoke._isolated_smoke_chat(
            _chat_kwargs(), deadline=time.monotonic() + 5.0, budget_s=5.0
        )

    assert result["choices"][0]["message"]["content"] == "The answer is 4"
    assert result["_freesolo_headers"] == {"checkpoint_id": _CHECKPOINT_ID}
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["key"] == _SECRET
    assert requests[0]["body"]["model"] == _CHECKPOINT_ID
    assert requests[0]["cmdlines"]
    assert all(_SECRET.encode() not in cmdline for cmdline in requests[0]["cmdlines"])
    assert _smoke_child_pids() == children_before


def test_isolated_http_error_redacts_every_raw_ipc_field(monkeypatch) -> None:
    children_before = _smoke_child_pids()
    raw_outcomes = _record_raw_ipc(monkeypatch)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)
    response_body = json.dumps({"error": {"message": f"rejected-{_SECRET}"}}).encode()

    with _http_server(
        status=400,
        response_body=response_body,
        echo_secret_error=True,
    ) as (port, _requests, _started):
        monkeypatch.setenv("FREESOLO_SERVING_URL", f"http://127.0.0.1:{port}/{_SECRET}")
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            serving_smoke._isolated_smoke_chat(
                _chat_kwargs(), deadline=time.monotonic() + 5.0, budget_s=5.0
            )

    _assert_ipc_redacted(raw_outcomes)
    error = exc_info.value
    assert error.response.status_code == 400
    serialized_error = repr(
        (
            str(error),
            str(error.request.url),
            tuple(error.request.headers.multi_items()),
            error.request.content,
            tuple(error.response.headers.multi_items()),
            error.response.content,
        )
    ).encode()
    assert _SECRET.encode() not in serialized_error
    assert _smoke_child_pids() == children_before


def test_isolated_malformed_json_preserves_direct_exception_taxonomy(monkeypatch) -> None:
    children_before = _smoke_child_pids()
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)

    with _http_server(response_body=b'{"choices": [') as (port, _requests, _started):
        monkeypatch.setenv("FREESOLO_SERVING_URL", f"http://127.0.0.1:{port}")
        with pytest.raises(json.JSONDecodeError) as direct_info:
            serving_smoke._app.serve_chat(**_chat_kwargs())
        with pytest.raises(json.JSONDecodeError) as isolated_info:
            serving_smoke._isolated_smoke_chat(
                _chat_kwargs(), deadline=time.monotonic() + 5.0, budget_s=5.0
            )

    direct = direct_info.value
    isolated = isolated_info.value
    assert type(isolated) is type(direct)
    assert (isolated.msg, isolated.doc, isolated.pos, isolated.lineno, isolated.colno) == (
        direct.msg,
        direct.doc,
        direct.pos,
        direct.lineno,
        direct.colno,
    )
    assert _smoke_child_pids() == children_before


def test_delayed_http_timeout_kills_owned_smoke_process(monkeypatch) -> None:
    children_before = _smoke_child_pids()
    hold = threading.Event()
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)

    with _http_server(hold=hold) as (port, requests, request_started):
        monkeypatch.setenv("FREESOLO_SERVING_URL", f"http://127.0.0.1:{port}")
        started = time.monotonic()
        with pytest.raises(ServingError, match=r"deployment_smoke_timeout"):
            serving_smoke._isolated_smoke_chat(_chat_kwargs(), deadline=started + 3.0, budget_s=3.0)
        elapsed = time.monotonic() - started
        assert request_started.is_set()
        assert requests[0]["key"] == _SECRET
        assert requests[0]["cmdlines"]
        assert all(_SECRET.encode() not in cmdline for cmdline in requests[0]["cmdlines"])
        assert elapsed < 4.0
        assert _smoke_child_pids() == children_before


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("malformed", "malformed IPC frame"),
        ("oversized", "oversized IPC frame"),
        ("truncated", "truncated IPC frame"),
    ],
)
def test_spawned_smoke_worker_rejects_faulty_frames_and_cleans_process(
    monkeypatch, fault, message
) -> None:
    children_before = _named_child_pids(serving_smoke._SMOKE_CHAT_PROCESS_NAME)
    get_context = multiprocessing.get_context
    monkeypatch.setattr(
        serving_smoke.multiprocessing,
        "get_context",
        lambda method: _FaultyFrameContext(get_context(method), fault),
    )

    with pytest.raises(ServingError, match=message):
        serving_smoke._isolated_smoke_chat(
            _chat_kwargs(), deadline=time.monotonic() + 3.0, budget_s=3.0
        )

    assert _named_child_pids(serving_smoke._SMOKE_CHAT_PROCESS_NAME) == children_before


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("malformed", "malformed IPC frame"),
        ("oversized", "oversized IPC frame"),
        ("truncated", "truncated IPC frame"),
    ],
)
def test_spawned_schema_worker_rejects_faulty_frames_and_cleans_process(
    monkeypatch, fault, message
) -> None:
    children_before = _named_child_pids(serving_smoke._JSON_SCHEMA_PROCESS_NAME)
    get_context = multiprocessing.get_context
    monkeypatch.setattr(
        serving_smoke.multiprocessing,
        "get_context",
        lambda method: _FaultyFrameContext(get_context(method), fault),
    )

    with pytest.raises(ServingError, match=message):
        serving_smoke._validate_json_schema({}, {}, deadline=time.monotonic() + 3.0, budget_s=3.0)

    assert _named_child_pids(serving_smoke._JSON_SCHEMA_PROCESS_NAME) == children_before


def test_isolated_request_error_preserves_direct_semantics(monkeypatch) -> None:
    children_before = _smoke_child_pids()
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    monkeypatch.setenv("FREESOLO_SERVING_URL", f"http://127.0.0.1:{port}")

    with pytest.raises(httpx.RequestError) as direct_info:
        serving_smoke._app.serve_chat(**_chat_kwargs())
    with pytest.raises(httpx.RequestError) as isolated_info:
        serving_smoke._isolated_smoke_chat(
            _chat_kwargs(), deadline=time.monotonic() + 5.0, budget_s=5.0
        )

    assert isinstance(direct_info.value, httpx.RequestError)
    assert type(isolated_info.value) is type(direct_info.value)
    assert isolated_info.value.request.method == direct_info.value.request.method
    assert isolated_info.value.request.url == direct_info.value.request.url
    assert _smoke_child_pids() == children_before


def test_safe_outcomes_preserve_serving_and_request_error_taxonomy(monkeypatch) -> None:
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _SECRET)
    serving_error = ServingError(
        f"serving {_SECRET}", status_code=409, retry_after=f"after-{_SECRET}"
    )
    serving_outcome = serving_smoke._redact_ipc_value(
        serving_smoke._encode_smoke_chat_exception(serving_error)
    )
    with pytest.raises(ServingError) as serving_info:
        serving_smoke._decode_smoke_chat_outcome(serving_outcome)
    assert type(serving_info.value) is ServingError
    assert serving_info.value.status_code == 409
    assert _SECRET not in str(serving_info.value)
    assert _SECRET not in str(serving_info.value.retry_after)

    retryable = RetryableServingUnavailable(f"loading-{_SECRET}", 1.25)
    retryable_outcome = serving_smoke._redact_ipc_value(
        serving_smoke._encode_smoke_chat_exception(retryable)
    )
    with pytest.raises(RetryableServingUnavailable) as retryable_info:
        serving_smoke._decode_smoke_chat_outcome(retryable_outcome)
    assert retryable_info.value.retry_after_seconds == 1.25
    assert _SECRET not in retryable_info.value.code

    request = httpx.Request("GET", f"http://127.0.0.1/{_SECRET}")
    for error_class in (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
    ):
        request_error = error_class(f"timeout-{_SECRET}", request=request)
        request_outcome = _framed_round_trip(
            serving_smoke._redact_ipc_value(
                serving_smoke._encode_smoke_chat_exception(request_error)
            )
        )
        with pytest.raises(error_class) as request_info:
            serving_smoke._decode_smoke_chat_outcome(request_outcome)
        assert type(request_info.value) is error_class
        assert _SECRET not in str(request_info.value)
        assert _SECRET not in str(request_info.value.request.url)


def test_spoofed_request_error_name_degrades_to_generic_request_error() -> None:
    request = httpx.Request("GET", "https://example.invalid")
    spoofed_class = type("ReadTimeout", (httpx.RequestError,), {})
    outcome = serving_smoke._encode_smoke_chat_exception(
        spoofed_class("spoofed timeout", request=request)
    )

    with pytest.raises(httpx.RequestError) as exc_info:
        serving_smoke._decode_smoke_chat_outcome(outcome)

    assert type(exc_info.value) is httpx.RequestError


def test_unknown_exception_uses_deliberate_serving_error_fallback() -> None:
    outcome = serving_smoke._encode_smoke_chat_exception(ValueError("not safe to reconstruct"))
    with pytest.raises(ServingError, match="unsupported exception") as exc_info:
        serving_smoke._decode_smoke_chat_outcome(outcome)
    assert type(exc_info.value) is ServingError


def test_all_production_process_joins_have_finite_timeouts() -> None:
    """Every join on a smoke child is bounded, wherever the reap ladder lives.

    The ladder is owned by `flash.server.platform.children` so both smoke children share one
    implementation; this guard follows it there rather than pinning it to a module.
    """
    tree = ast.parse(inspect.getsource(children))
    joins = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "process"
        and node.func.attr == "join"
    ]
    assert joins
    for call in joins:
        timeout = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "timeout"), None
        )
        assert timeout is not None
        assert not (isinstance(timeout, ast.Constant) and timeout.value is None)


class _NeverExitsProcess:
    def __init__(self) -> None:
        self.pid = 123
        self.join_timeouts: list[float] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        return None

    def join(self, *, timeout: float) -> None:
        assert timeout is not None
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _NoResultConnection:
    def __init__(self, fd: int) -> None:
        self._fd = fd
        self.close_calls = 0

    def fileno(self) -> int:
        return self._fd

    def close(self) -> None:
        self.close_calls += 1
        os.close(self._fd)


class _NeverExitsContext:
    def __init__(self, process: _NeverExitsProcess) -> None:
        self.process = process
        receive_fd, send_fd = os.pipe()
        self.receive = _NoResultConnection(receive_fd)
        self.send = _NoResultConnection(send_fd)

    def Pipe(self, *, duplex: bool):
        assert duplex is False
        return self.receive, self.send

    def Process(self, *args, **kwargs):
        return self.process


class _ExplodingCloseConnection(_NoResultConnection):
    """A pipe end whose close raises, as a real closed or broken fd can."""

    def close(self) -> None:
        self.close_calls += 1
        os.close(self._fd)
        raise OSError("pipe close exploded")


def test_isolated_smoke_retains_live_process_ownership_without_closing(monkeypatch) -> None:
    process = _NeverExitsProcess()
    context = _NeverExitsContext(process)
    monkeypatch.setattr(serving_smoke.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(
        serving_smoke,
        "_receive_framed_ipc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(serving_smoke._IpcDeadlineExceeded()),
    )

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving_smoke._isolated_smoke_chat(
            _chat_kwargs(), deadline=time.monotonic() + 1.0, budget_s=1.0
        )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.close_calls == 0
    assert context.receive.close_calls == 1
    assert context.send.close_calls == 1
    assert process in children._LIVE_CHILDREN
    children._release(process)


def test_a_raising_pipe_close_cannot_drop_a_live_smoke_child(monkeypatch) -> None:
    """Closing the receive pipe must not be able to preempt the reap or the ownership record.

    The close sits in the same ``finally`` as the reap, so a raising close used to skip the reap
    entirely and leave a live child with no owner. Ownership now begins at ``spawn_owned``, so the
    child is registered before the pipe is ever touched and the lifespan boundary still finds it.
    """
    process = _NeverExitsProcess()
    context = _NeverExitsContext(process)
    context.receive = _ExplodingCloseConnection(context.receive.fileno())
    monkeypatch.setattr(serving_smoke.multiprocessing, "get_context", lambda _method: context)
    monkeypatch.setattr(
        serving_smoke,
        "_receive_framed_ipc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(serving_smoke._IpcDeadlineExceeded()),
    )

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        serving_smoke._isolated_smoke_chat(
            _chat_kwargs(), deadline=time.monotonic() + 1.0, budget_s=1.0
        )

    assert context.receive.close_calls == 1
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process in children._LIVE_CHILDREN
    children._release(process)


def test_bounded_process_reaper_kills_a_terminate_resistant_child() -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_ignore_terminate, args=(ready,), daemon=True)
    try:
        children.spawn_owned(process)
        assert ready.wait(timeout=5.0)
        assert children.reap_owned(process) is True
        assert process.is_alive() is False
        assert process.exitcode is not None
        assert process not in children._LIVE_CHILDREN
    finally:
        children._release(process)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        if not process.is_alive():
            process.close()
