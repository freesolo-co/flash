"""Explicitly provisioned H100s: CPU proxies never launch or release their VMs."""

import asyncio
import json
import os

from .proxy import LambdaEngineProxy
from .transport import WorkerTransport

MODEL = "Qwen/Qwen3.5-9B-Base"
SECRET_NAME = "flash-reserved-h100-workers-20261112"
# Serving declares one CPU proxy per slot up to this ceiling; the workers secret configures any
# non-empty subset of them, and ``reserved_expansion`` publishes qualified workers for target
# slots the secret does not hold. Unconfigured slots are never routed (see reserved_watchdog).
SLOT_COUNT = 12
# Holds LAMBDA_API_KEY for the reservation's own lifecycle; the watchdog uses it only to
# reboot an existing reserved instance, never to launch or terminate one.
RESERVATION_SECRET_NAME = "flash-h100-reservation-20261112"
# Bounded per-worker registration: a hung worker must not hold every adapter promotion.
REGISTER_TIMEOUT_SECONDS = 10 * 60


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


async def configured_worker(slot):
    """The slot's worker from the secret, else from the published expansion workers."""
    from .reserved_expansion import merge_workers, published_workers

    raw = os.environ["FLASH_RESERVED_H100_WORKERS"]
    if any(w["slot"] == slot for w in parse_workers(raw)):
        return worker_config(slot, raw)
    workers = merge_workers(parse_workers(raw), await published_workers())
    return worker_config(slot, json.dumps(workers))


class ReservedLambdaEngineProxy(LambdaEngineProxy):
    reserved_slot = 0

    async def _load(self):
        if self.base_model != MODEL:
            raise ValueError("Reserved H100s only serve the 9B Base model")
        worker = await configured_worker(self.reserved_slot)
        self._owner = worker["instance_id"]
        self._transport = WorkerTransport(worker["ip"], worker["tls"])
        try:
            async with asyncio.timeout(30):
                health = await self._transport.call("/health")
            if not (
                health.get("ok") is True
                and health.get("base_model") == MODEL
                and health.get("deployment_sha") == worker["revision"]
                and health.get("configured_gpu") == "H100"
            ):
                raise RuntimeError("Reserved H100 model, revision, or hardware identity mismatch")
            self._health_cache = health
            self.engine = self._transport
        except BaseException:
            await self._transport.aclose()
            raise

    async def _shutdown(self):
        transport = getattr(self, "_transport", None)
        if transport is not None:
            await transport.aclose()
