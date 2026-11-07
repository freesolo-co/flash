"""Bounded Lambda API calls. Ambiguous mutations are never automatically retried."""

import asyncio
import time

import httpx

from .config import API_URL


class LambdaAPIError(RuntimeError):
    def __init__(self, status, code):
        self.status = status
        self.code = code
        super().__init__(f"Lambda API failed ({status}, {code})")


class LambdaAPI:
    def __init__(self, key, *, transport=None):
        if not key:
            raise ValueError("LAMBDA_API_KEY is required")
        self.client = httpx.AsyncClient(
            base_url=API_URL + "/",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "User-Agent": "Flash-Lambda-Serving/1.0",
            },
            timeout=30,
            follow_redirects=False,
            transport=transport,
        )
        self._lock = asyncio.Lock()
        self._next_request = 0.0
        self._next_launch = 0.0

    async def call(self, method, path, body=None):
        async with self._lock:
            launch = path == "instance-operations/launch"
            deadline = max(self._next_request, self._next_launch if launch else 0)
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            self._next_request = time.monotonic() + 1.1
            if launch:
                self._next_launch = time.monotonic() + 12.1
            response = await self.client.request(method, path, json=body)
            if response.status_code >= 300:
                try:
                    code = response.json().get("error", {}).get("code", "unknown")
                except ValueError:
                    code = "non-json-response"
                # Never include request headers, cloud-init, or the provider's echoed body.
                raise LambdaAPIError(response.status_code, str(code)[:120])
            return response.json()["data"]

    async def aclose(self):
        await self.client.aclose()
