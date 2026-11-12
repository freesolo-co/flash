"""Explicitly provisioned H100s: CPU proxies never launch or release their VMs."""

import asyncio
import json
import os

from .proxy import LambdaEngineProxy
from .transport import WorkerTransport

MODEL = "Qwen/Qwen3.5-9B-Base"
SECRET_NAME = "flash-reserved-h100-workers-20261112"
SLOT_COUNT = 4


def enabled(model, environment):
    return model == MODEL and environment == "main"


def worker_config(slot, raw):
    workers = json.loads(raw)
    if len(workers) != SLOT_COUNT or {w["slot"] for w in workers} != set(range(1, 5)):
        raise ValueError("Reserved fleet must contain four unique slots")
    if len({w["instance_id"] for w in workers}) != SLOT_COUNT:
        raise ValueError("Reserved fleet instance IDs must be distinct")
    return next(w for w in workers if w["slot"] == slot)


class ReservedLambdaEngineProxy(LambdaEngineProxy):
    reserved_slot = 0

    async def _load(self):
        if self.base_model != MODEL:
            raise ValueError("Reserved H100s only serve the 9B Base model")
        worker = worker_config(self.reserved_slot, os.environ["FLASH_RESERVED_H100_WORKERS"])
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
