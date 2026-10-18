from __future__ import annotations

import pytest

from tests.serving.test_router import QWEN, FakePool, _router_for, _serve


@pytest.mark.parametrize(
    ("stream", "stream_options", "detail"),
    [
        (False, {"include_usage": True}, "stream_options requires stream=true"),
        (True, {"include_usage": "true"}, "stream_options accepts only boolean include_usage"),
        (True, "include_usage", "stream_options must be an object"),
        (
            True,
            {"include_usage": True, "extra": False},
            "stream_options accepts only boolean include_usage",
        ),
    ],
)
def test_hosted_chat_rejects_malformed_stream_options_before_generation(
    stream: bool,
    stream_options: object,
    detail: str,
) -> None:
    pool = FakePool()
    with _serve(pool, _router_for("qa", QWEN), internal_key="sekret") as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qa/final",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
                "stream_options": stream_options,
            },
        )

    assert (response.status_code, pool.generated) == (422, [])
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    "stream_options", [pytest.param(None, id="null"), pytest.param(..., id="absent")]
)
def test_hosted_chat_accepts_absent_or_null_stream_options(stream_options: object) -> None:
    pool = FakePool()
    payload = {
        "model": "qa/final",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if stream_options is not ...:
        payload["stream_options"] = stream_options

    with _serve(pool, _router_for("qa", QWEN), internal_key="sekret") as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert len(pool.generated) == 1
    assert pool.generated[0][0] == QWEN
