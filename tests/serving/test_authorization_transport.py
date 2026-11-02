"""Transient lookups retry within a deadline; denials and secrets never leak through."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from flash.serving.src.http import authorization_transport as transport


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError]
)
def test_transport_recovers_without_exposing_credentials(failure, monkeypatch, caplog):
    monkeypatch.setattr(transport.random, "uniform", lambda *_: 0)
    client = AsyncMock()
    client.post.side_effect = [failure("private-key private-url"), httpx.Response(200)]
    result = asyncio.run(
        transport.post_authorization(
            client, "private-url", api_key="private-key", model_id="m", scope={"X-Job": "job"}
        )
    )
    assert result.status_code == 200
    assert client.post.await_count == 2
    assert client.post.call_args_list[0] == client.post.call_args_list[1]
    assert failure.__name__ in caplog.text
    assert "private-key" not in caplog.text
    assert "private-url" not in caplog.text


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 429])
def test_nontransient_responses_are_not_retried(status):
    client = AsyncMock()
    client.post.return_value = httpx.Response(status)
    result = asyncio.run(
        transport.post_authorization(client, "url", api_key="key", model_id="m", scope={})
    )
    assert result.status_code == status
    client.post.assert_awaited_once()


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_status_exhaustion_stays_unavailable(status, monkeypatch):
    monkeypatch.setattr(transport.random, "uniform", lambda *_: 0)
    client = AsyncMock()
    client.post.return_value = httpx.Response(status)
    with pytest.raises(transport.TransientAuthorizationError) as error:
        asyncio.run(
            transport.post_authorization(client, "url", api_key="key", model_id="m", scope={})
        )
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "1"}
    assert client.post.await_count == transport._ATTEMPTS


def test_total_deadline_bounds_a_stuck_backend(monkeypatch):
    monkeypatch.setattr(transport, "AUTHORIZATION_BUDGET_SECONDS", 0.01)

    async def scenario():
        client = AsyncMock()

        async def stuck(*args, **kwargs):
            await asyncio.Event().wait()

        client.post.side_effect = stuck
        with pytest.raises(transport.TransientAuthorizationError):
            await asyncio.wait_for(
                transport.post_authorization(client, "url", api_key="key", model_id="m", scope={}),
                timeout=2,
            )
        client.post.assert_awaited_once()

    asyncio.run(scenario())


def test_cancellation_is_not_retried_or_converted_to_auth_failure():
    client = AsyncMock()
    client.post.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            transport.post_authorization(client, "url", api_key="key", model_id="m", scope={})
        )
    client.post.assert_awaited_once()


def test_programming_errors_fail_closed_without_retry(caplog):
    client = AsyncMock()
    client.post.side_effect = ValueError("private-key")
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            transport.post_authorization(client, "url", api_key="key", model_id="m", scope={})
        )
    assert not isinstance(error.value, transport.TransientAuthorizationError)
    client.post.assert_awaited_once()
    assert "private-key" not in caplog.text


def test_deadline_also_bounds_fast_failures_and_backoff(monkeypatch):
    monkeypatch.setattr(transport, "AUTHORIZATION_BUDGET_SECONDS", 0.01)
    client = AsyncMock()
    client.post.return_value = httpx.Response(502)

    async def scenario():
        with pytest.raises(transport.TransientAuthorizationError):
            await asyncio.wait_for(
                transport.post_authorization(client, "url", api_key="key", model_id="m", scope={}),
                timeout=2,
            )

    asyncio.run(scenario())
    client.post.assert_awaited_once()


def test_cancellation_during_backoff_stops_further_attempts():
    async def scenario():
        failed = asyncio.Event()
        client = AsyncMock()

        async def unavailable(*args, **kwargs):
            failed.set()
            return httpx.Response(502)

        client.post.side_effect = unavailable
        task = asyncio.create_task(
            transport.post_authorization(client, "url", api_key="key", model_id="m", scope={})
        )
        await failed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        client.post.assert_awaited_once()

    asyncio.run(scenario())
