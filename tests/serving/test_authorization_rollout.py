"""Real TCP restart coverage for the serving authorization boundary (no GPU or credentials)."""

import asyncio
import contextlib
import json

import httpx
import pytest

from flash.serving.src.http import authorization_transport as transport


async def _backend_restart(authorizer_transport, model_id):
    requests = []

    async def authorize(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            requests.append(json.loads(await reader.readexactly(length)))
            body = b'{"orgId":"test-org"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    old = await asyncio.start_server(authorize, "127.0.0.1", 0)
    port = old.sockets[0].getsockname()[1]
    old.close()
    await old.wait_closed()

    async def replace():
        # a replacement backend has not bound its listener yet; the old router is still live.
        await asyncio.sleep(4)
        return await asyncio.start_server(authorize, "127.0.0.1", port)

    replacement = asyncio.create_task(replace())
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await authorizer_transport.post_authorization(
                client,
                f"http://127.0.0.1:{port}/api/serving/authorize",
                api_key="test-only",
                model_id=model_id,
                scope={},
            )
        assert response.status_code == 200
        assert requests == [{"apiKey": "test-only", "modelId": model_id}]
    finally:
        if not replacement.done():
            replacement.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            server = await replacement
            server.close()
            await server.wait_closed()


@pytest.mark.wallclock
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-9B", "flash-1234567890-abcd1234/step-1"])
def test_authorization_survives_four_second_backend_restart(model_id):
    asyncio.run(_backend_restart(transport, model_id))
