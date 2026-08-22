from __future__ import annotations

import asyncio
from typing import Any

from flash.serve.contract import MAX_CHAT_REQUEST_BYTES
from flash.serving.src.router import AdapterRouter, build_serving_app


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

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(MAX_CHAT_REQUEST_BYTES + 1).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }

    asyncio.run(app(scope, receive, send))

    assert receive_calls == 0
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


def test_hosted_app_caps_chunked_body_before_json_materialization(monkeypatch) -> None:
    from flash.serving.src import router as router_module

    monkeypatch.setattr(router_module, "MAX_CHAT_REQUEST_BYTES", 4)
    app = router_module.build_serving_app(
        object(), AdapterRouter(), internal_key="synthetic-internal-key"
    )
    chunks = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5", "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(chunks)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }

    asyncio.run(app(scope, receive, send))

    assert [message for message in sent if message["type"] == "http.response.start"] == [
        {"type": "http.response.start", "status": 413, "headers": sent[0]["headers"]}
    ]
