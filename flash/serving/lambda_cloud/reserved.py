"""Explicitly provisioned H100s: CPU proxies never launch or release their VMs."""

import asyncio
import json
import os
import time

from .proxy import LambdaEngineProxy
from .transport import WorkerTransport

MODEL = "Qwen/Qwen3.5-9B-Base"
SECRET_NAME = "flash-reserved-h100-workers-20261112"
# Serving declares one CPU proxy per slot up to this ceiling; the workers secret and
# ``reserved_expansion`` configure a subset. Unconfigured slots are never routed.
SLOT_COUNT = 12
# Holds LAMBDA_API_KEY for the reservation's own lifecycle; the watchdog uses it only to
# reboot an existing reserved instance, never to launch or terminate one.
RESERVATION_SECRET_NAME = "flash-h100-reservation-20261112"
# Bounded per-worker registration: a hung worker must not hold every adapter promotion.
REGISTER_TIMEOUT_SECONDS = 10 * 60
REBIND_CHECK_SECONDS = 15  # how often an expansion-slot proxy re-reads its worker


def enabled(model, environment):
    return model == MODEL and environment == "main"


def parse_workers(raw):
    """Validated worker entries from the secret: unique slots within 1..SLOT_COUNT."""
    workers = json.loads(raw)
    if not isinstance(workers, list) or not workers:
        raise ValueError("Reserved fleet must configure at least one slot")
    slots = [w.get("slot") for w in workers]
    if any(type(s) is not int or not 1 <= s <= SLOT_COUNT for s in slots):
        raise ValueError(f"Reserved fleet slots must be integers 1..{SLOT_COUNT}")
    if len(set(slots)) != len(slots):
        raise ValueError("Reserved fleet slots must be unique")
    if len({w["instance_id"] for w in workers}) != len(workers):
        raise ValueError("Reserved fleet instance IDs must be distinct")
    return sorted(workers, key=lambda w: w["slot"])


def worker_config(slot, raw):
    worker = next((w for w in parse_workers(raw) if w["slot"] == slot), None)
    if worker is None:
        raise ValueError(f"Reserved slot {slot} is not configured")
    return worker


def identity_ok(health, revision):
    return (
        health.get("ok") is True
        and health.get("base_model") == MODEL
        and health.get("deployment_sha") == revision
        and health.get("configured_gpu") == "H100"
    )


class ReservedLambdaEngineProxy(LambdaEngineProxy):
    reserved_slot = 0

    async def _load(self):
        if self.base_model != MODEL:
            raise ValueError("Reserved H100s only serve the 9B Base model")
        self._transport = None
        self._rebind_at = time.monotonic() + REBIND_CHECK_SECONDS
        await self._bind(await self._configured())

    async def _configured(self):
        from .reserved_expansion import merge_workers, published_workers

        workers = parse_workers(os.environ["FLASH_RESERVED_H100_WORKERS"])
        # Secret slots never change under a running proxy; expansion slots can be replaced.
        self._expansion = all(w["slot"] != self.reserved_slot for w in workers)
        if self._expansion:
            workers = merge_workers(workers, await published_workers())
        return worker_config(self.reserved_slot, json.dumps(workers))

    async def _bind(self, worker):
        transport = WorkerTransport(worker["ip"], worker["tls"])
        try:
            async with asyncio.timeout(30):
                health = await transport.call("/health")
            if not identity_ok(health, worker["revision"]):
                raise RuntimeError("Reserved H100 model, revision, or hardware identity mismatch")
        except BaseException:
            await transport.aclose()
            raise
        previous, self._transport = self._transport, transport
        self.engine, self._owner, self._health_cache = transport, worker["instance_id"], health
        if previous is not None:
            await previous.aclose()

    async def _rebind(self):
        """Follow an expansion slot's replaced worker; keep the binding if that fails."""
        if not self._expansion or time.monotonic() < self._rebind_at:
            return
        self._rebind_at = time.monotonic() + REBIND_CHECK_SECONDS
        try:
            worker = await self._configured()
            if worker["instance_id"] != self._owner:
                await self._bind(worker)
        except Exception as exc:
            print(f"reserved H100 slot {self.reserved_slot} rebind failed: {exc!r}", flush=True)

    async def _health(self):
        await self._rebind()
        return await super()._health()

    async def _register(self, *args, **kwargs):
        await self._rebind()
        return await super()._register(*args, **kwargs)

    async def _unregister(self, *args, **kwargs):
        await self._rebind()
        return await super()._unregister(*args, **kwargs)

    async def _generate(self, *args, **kwargs):
        await self._rebind()
        return await super()._generate(*args, **kwargs)

    async def _stream_generate(self, *args, **kwargs):
        await self._rebind()
        async for event in super()._stream_generate(*args, **kwargs):
            yield event

    async def _shutdown(self):
        transport = getattr(self, "_transport", None)
        if transport is not None:
            await transport.aclose()
