"""Authenticated, certificate-pinned worker RPC with bounded streaming cleanup."""

import asyncio
import contextlib
import json
import re
import ssl

import httpx

from flash.serving.app.timeouts import REQUEST_TIMEOUT_SECONDS
from flash.serving.src.stream_channel.protocol import HEARTBEAT_INTERVAL_SECONDS, LEASE_SECONDS

from .config import PORT, SERVER_NAME

# The next renewal starts one heartbeat after the last success; splitting the rest of the lease
# in two leaves a full retry before it lapses when that first attempt stalls.
LEASE_RENEW_TIMEOUT_SECONDS = (LEASE_SECONDS - HEARTBEAT_INTERVAL_SECONDS) / 2


class WorkerError(RuntimeError):
    def __init__(self, message, status_code=502):
        self.status_code = status_code
        super().__init__(message)


class BatchUnsupported(RuntimeError):
    """The worker predates /generate_batch."""


_SAFE_KIND = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _failure_kind(response):
    """The worker's fixed error kind (e.g. ``timeout``), never free-form body text."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    kind = body.get("kind", body.get("error"))
    return kind if isinstance(kind, str) and _SAFE_KIND.fullmatch(kind) else None


def _failure(status_code, body):
    if status_code == 400 and isinstance(body, dict) and body.get("kind") == "value_error":
        return ValueError(body["message"])
    kind = body.get("kind", body.get("error")) if isinstance(body, dict) else None
    kind = kind if isinstance(kind, str) and _SAFE_KIND.fullmatch(kind) else None
    detail = f"HTTP {status_code}" + (f", kind={kind}" if kind else "")
    return WorkerError(
        f"Lambda worker request failed ({detail})", 504 if status_code == 504 else 502
    )


def check_response(response):
    if response.status_code == 400:
        body = response.json()
        if body.get("kind") == "value_error":
            raise ValueError(body["message"])
    if response.status_code != 200:
        kind = _failure_kind(response)
        detail = f"HTTP {response.status_code}" + (f", kind={kind}" if kind else "")
        raise WorkerError(
            f"Lambda worker request failed ({detail})",
            504 if response.status_code == 504 else 502,
        )


def batch_item_outcome(item, generation_id):
    """A /generate_batch item as the result, or the exception, a lone /generate gives."""
    if not isinstance(item, dict) or type(item.get("status")) is not int:
        return WorkerError("Lambda returned a malformed batch item")
    if item["status"] != 200:
        return _failure(item["status"], item)
    result = item.get("result")
    if not isinstance(result, dict) or result.get("request_id") != generation_id:
        return WorkerError("Lambda returned a mismatched generation id")
    return result


class WorkerTransport:
    def __init__(self, ip, tls, *, transport=None):
        import ipaddress

        address = ipaddress.ip_address(ip)
        if not address.is_global:
            raise ValueError("Lambda worker must have a public IP")
        context = ssl.create_default_context(cadata=tls["cert"])

        def client(timeout):
            return httpx.AsyncClient(
                base_url=f"https://{address}:{PORT}",
                verify=context,
                headers={"Authorization": f"Bearer {tls['token']}"},
                follow_redirects=False,
                timeout=timeout,
                transport=transport,
            )

        self.client = client(httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10))
        # Lease renewals get their own connection pool: a proxy saturated with long-lived
        # generations must never make a renewal wait for a connection until its lease lapses,
        # which makes the worker abort a healthy generation.
        self.lease_client = client(httpx.Timeout(LEASE_RENEW_TIMEOUT_SECONDS))

    def request(self, method, path, body=None):
        return self._client_for(path).build_request(
            method,
            path,
            json=body,
            extensions={"sni_hostname": SERVER_NAME},
        )

    async def call(self, path, body=None):
        lease_task = None
        if path == "/generate":
            lease = {"generation_id": body["generation_id"]}
            await self.call("/lease", lease)
            lease_task = asyncio.create_task(self._renew_lease(lease))
        try:
            response = await self._client_for(path).send(
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

    async def call_batch(self, items):
        """Run ``items`` (``/generate`` bodies) as one /generate_batch call.

        Returns one result or exception per item, each what ``call("/generate", item)`` would
        have returned or raised; raises ``BatchUnsupported`` on a worker without the route.
        """
        lease = {"generation_ids": [item["generation_id"] for item in items]}
        await self.call("/lease", lease)
        lease_task = asyncio.create_task(self._renew_lease(lease))
        try:
            response = await self.client.send(
                self.request("POST", "/generate_batch", {"items": items})
            )
            if response.status_code in (404, 405):
                raise BatchUnsupported("Lambda worker has no /generate_batch")
            check_response(response)
            results = response.json().get("results")
            if not isinstance(results, list) or len(results) != len(items):
                raise WorkerError("Lambda returned a malformed batch")
            return [
                batch_item_outcome(result, item["generation_id"])
                for result, item in zip(results, items, strict=True)
            ]
        except httpx.TimeoutException as exc:
            raise WorkerError("Lambda worker request timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise WorkerError("Lambda worker connection failed") from exc
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)

    def _client_for(self, path):
        return self.lease_client if path == "/lease" else self.client

    async def _renew_lease(self, lease):
        loop = asyncio.get_running_loop()
        started = loop.time()
        while True:
            # Attempts start a heartbeat apart, or at once when the previous one overran.
            await asyncio.sleep(max(0.0, started + HEARTBEAT_INTERVAL_SECONDS - loop.time()))
            started = loop.time()
            with contextlib.suppress(WorkerError, TimeoutError):
                async with asyncio.timeout(LEASE_RENEW_TIMEOUT_SECONDS):
                    await self.call("/lease", lease)

    async def stream(self, body):
        response = None
        terminal = False
        lease = {"generation_id": body["generation_id"]}
        await self.call("/lease", lease)
        lease_task = asyncio.create_task(self._renew_lease(lease))
        try:
            response = await self.client.send(self.request("POST", "/stream", body), stream=True)
            if response.status_code != 200:
                await response.aread()  # check_response parses the error body
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
        await self.lease_client.aclose()
