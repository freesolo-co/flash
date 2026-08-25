from __future__ import annotations

import asyncio
from typing import Any

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES
from flash.serving.src.http.body_limit import RequestBodyLimitMiddleware
from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app


def _scope(
    headers: list[tuple[bytes, bytes]],
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }


def _response_start(sent: list[dict[str, Any]]) -> dict[str, Any]:
    return next(message for message in sent if message["type"] == "http.response.start")


def _assert_connection_close(message: dict[str, Any]) -> None:
    assert (b"connection", b"close") in message["headers"], "413 response must close the connection"


def test_hosted_app_rejects_declared_oversize_before_reading_body() -> None:
    app = build_serving_app(object(), AdapterRouter(), internal_key="synthetic-internal-key")
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("oversized body reached fastapi parsing")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = _scope(
        [
            (b"content-type", b"application/json"),
            (b"content-length", str(MAX_CHAT_REQUEST_BYTES + 1).encode("ascii")),
        ]
    )

    asyncio.run(app(scope, receive, send))

    assert receive_calls == 0
    response_start = _response_start(sent)
    assert response_start["status"] == 413
    _assert_connection_close(response_start)


def test_hosted_healthz_dispatches_without_consuming_chunked_body() -> None:
    app = build_serving_app(object(), AdapterRouter(), internal_key="synthetic-internal-key")
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("healthz body must not be consumed")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(
        app(
            _scope(
                [(b"transfer-encoding", b"chunked")],
                method="GET",
                path="/healthz",
            ),
            receive,
            send,
        )
    )

    assert receive_calls == 0
    assert _response_start(sent)["status"] == 200


def test_hosted_app_caps_chunked_body_without_reading_past_rejection(monkeypatch) -> None:
    from flash.serving.src.http import router as router_module

    monkeypatch.setattr(router_module, "MAX_CHAT_REQUEST_BYTES", 4)
    app = router_module.build_offline_serving_app(
        object(), AdapterRouter(), internal_key="synthetic-internal-key"
    )
    chunks = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5", "more_body": True},
        {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
    ]
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        message = chunks[receive_calls]
        receive_calls += 1
        return message

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(
        app(
            _scope([(b"content-type", b"application/json")]),
            receive,
            send,
        )
    )

    assert receive_calls == 2
    response_start = _response_start(sent)
    assert response_start["status"] == 413
    _assert_connection_close(response_start)


def test_middleware_replays_successful_body_unchanged_after_rejection() -> None:
    replayed: list[dict[str, Any]] = []

    async def downstream(scope: dict[str, Any], receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            replayed.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = RequestBodyLimitMiddleware(downstream, max_bytes=8)
    rejected_sent: list[dict[str, Any]] = []

    async def rejected_receive() -> dict[str, Any]:
        raise AssertionError("declared oversize body must not be read")

    async def rejected_send(message: dict[str, Any]) -> None:
        rejected_sent.append(message)

    asyncio.run(
        app(
            _scope([(b"content-length", b"9")]),
            rejected_receive,
            rejected_send,
        )
    )
    _assert_connection_close(_response_start(rejected_sent))

    chunks = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"45678", "more_body": False},
    ]
    pending = iter(chunks)
    successful_sent: list[dict[str, Any]] = []

    async def successful_receive() -> dict[str, Any]:
        return next(pending)

    async def successful_send(message: dict[str, Any]) -> None:
        successful_sent.append(message)

    asyncio.run(app(_scope([]), successful_receive, successful_send))

    assert replayed == chunks
    assert _response_start(successful_sent)["status"] == 204
