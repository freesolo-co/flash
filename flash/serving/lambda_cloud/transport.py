"""Authenticated, certificate-pinned worker RPC with bounded streaming cleanup."""

import asyncio
import contextlib
import json
import ssl

import httpx

from flash.serving.app.timeouts import REQUEST_TIMEOUT_SECONDS
from flash.serving.src.stream_channel.protocol import HEARTBEAT_INTERVAL_SECONDS

from .config import PORT, SERVER_NAME


class WorkerError(RuntimeError):
    def __init__(self, message, status_code=502):
        self.status_code = status_code
        super().__init__(message)


def check_response(response):
    if response.status_code == 400:
        body = response.json()
        if body.get("kind") == "value_error":
            raise ValueError(body["message"])
    if response.status_code != 200:
        raise WorkerError(
            "Lambda worker request failed", 504 if response.status_code == 504 else 502
        )


class WorkerTransport:
    def __init__(self, ip, tls, *, transport=None):
        import ipaddress

        address = ipaddress.ip_address(ip)
        if not address.is_global:
            raise ValueError("Lambda worker must have a public IP")
        context = ssl.create_default_context(cadata=tls["cert"])
        self.client = httpx.AsyncClient(
            base_url=f"https://{address}:{PORT}",
            verify=context,
            headers={"Authorization": f"Bearer {tls['token']}"},
            follow_redirects=False,
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10),
            transport=transport,
        )

    def request(self, method, path, body=None):
        return self.client.build_request(
            method,
            path,
            json=body,
            extensions={"sni_hostname": SERVER_NAME},
        )

    async def call(self, path, body=None):
        lease_task = None
        if path == "/generate":
            await self.call("/lease", {"generation_id": body["generation_id"]})
            lease_task = asyncio.create_task(self._renew_lease(body["generation_id"]))
        try:
            response = await self.client.send(
                self.request("GET" if body is None else "POST", path, body)
            )
            check_response(response)
            result = response.json()
            if path == "/generate" and result.get("request_id") != body["generation_id"]:
                raise WorkerError("Lambda returned a mismatched generation id")
            return result
        except httpx.TimeoutException as exc:
            raise WorkerError("Lambda worker request timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise WorkerError("Lambda worker connection failed") from exc
        finally:
            if lease_task is not None:
                lease_task.cancel()
                await asyncio.gather(lease_task, return_exceptions=True)

    async def _renew_lease(self, generation_id):
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            with contextlib.suppress(WorkerError, TimeoutError):
                async with asyncio.timeout(HEARTBEAT_INTERVAL_SECONDS):
                    await self.call("/lease", {"generation_id": generation_id})

    async def stream(self, body):
        response = None
        terminal = False
        await self.call("/lease", {"generation_id": body["generation_id"]})
        lease_task = asyncio.create_task(self._renew_lease(body["generation_id"]))
        try:
            response = await self.client.send(self.request("POST", "/stream", body), stream=True)
            check_response(response)
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("lambda_error") == "value_error":
                    raise ValueError(event["message"])
                if event.get("lambda_error"):
                    raise WorkerError(
                        "Lambda generation failed",
                        504 if event["lambda_error"] == "timeout" else 502,
                    )
                if terminal:
                    raise WorkerError("Lambda returned data after completion")
                terminal = event.get("type") == "final"
                yield event
            if not terminal:
                raise WorkerError("Lambda stream ended without completion")
        except httpx.HTTPError as exc:
            raise WorkerError("Lambda stream connection failed") from exc
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)
            if response is not None:
                await response.aclose()

    async def abort(self, generation_id):
        await self.call("/abort", {"generation_id": generation_id})

    async def aclose(self):
        await self.client.aclose()
