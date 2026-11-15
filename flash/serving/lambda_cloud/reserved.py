"""Explicitly provisioned Lambda GPUs: CPU proxies never launch or release their VMs."""

import asyncio
import json
import os
from dataclasses import dataclass

from .proxy import LambdaEngineProxy
from .transport import WorkerTransport


@dataclass(frozen=True)
class ReservedFleet:
    gpu: str
    slots: int
    secret_name: str
    workers_env: str


# Each model's workers, identities and credentials live in its own main-environment Modal Secret.
FLEETS = {
    "Qwen/Qwen3.5-9B-Base": ReservedFleet(
        gpu="H100",
        slots=4,
        secret_name="flash-reserved-h100-workers-20261112",
        workers_env="FLASH_RESERVED_H100_WORKERS",
    ),
    "Qwen/Qwen3.5-0.8B-Base": ReservedFleet(
        gpu="H200",
        slots=1,
        secret_name="flash-reserved-h200-workers-20261114",
        workers_env="FLASH_RESERVED_H200_WORKERS",
    ),
}
MODELS = tuple(FLEETS)

# The watchdog-supervised four-H100 9B Base fleet; other reserved models route without it.
MODEL = "Qwen/Qwen3.5-9B-Base"
SECRET_NAME = FLEETS[MODEL].secret_name
SLOT_COUNT = FLEETS[MODEL].slots
# Holds LAMBDA_API_KEY for the reservation's own lifecycle; the watchdog uses it only to
# reboot an existing reserved instance, never to launch or terminate one.
RESERVATION_SECRET_NAME = "flash-h100-reservation-20261112"
# Bounded per-worker registration: a hung worker must not hold every adapter promotion.
REGISTER_TIMEOUT_SECONDS = 10 * 60


def fleet_for(model):
    try:
        return FLEETS[model]
    except KeyError:
        raise ValueError(f"{model} has no reserved serving fleet") from None


def enabled(model, environment):
    return model in FLEETS and environment == "main"


def engine_name_suffix(model, slot):
    """Stable per-slot Modal class suffix, e.g. ``_ReservedH1001``."""
    return f"_Reserved{fleet_for(model).gpu}{slot}"


def worker_config(slot, raw, model=MODEL):
    fleet = fleet_for(model)
    workers = json.loads(raw)
    if len(workers) != fleet.slots or {w["slot"] for w in workers} != set(
        range(1, fleet.slots + 1)
    ):
        raise ValueError(f"Reserved fleet must contain {fleet.slots} unique slots")
    if len({w["instance_id"] for w in workers}) != fleet.slots:
        raise ValueError("Reserved fleet instance IDs must be distinct")
    return next(w for w in workers if w["slot"] == slot)


class ReservedLambdaEngineProxy(LambdaEngineProxy):
    reserved_slot = 0

    async def _load(self):
        fleet = fleet_for(self.base_model)
        worker = worker_config(
            self.reserved_slot, os.environ[fleet.workers_env], model=self.base_model
        )
        self._owner = worker["instance_id"]
        self._transport = WorkerTransport(worker["ip"], worker["tls"])
        try:
            async with asyncio.timeout(30):
                health = await self._transport.call("/health")
            if not (
                health.get("ok") is True
                and health.get("base_model") == self.base_model
                and health.get("deployment_sha") == worker["revision"]
                and health.get("configured_gpu") == fleet.gpu
            ):
                raise RuntimeError(
                    f"Reserved {fleet.gpu} model, revision, or hardware identity mismatch"
                )
            self._health_cache = health
            self.engine = self._transport
        except BaseException:
            await self._transport.aclose()
            raise

    async def _shutdown(self):
        transport = getattr(self, "_transport", None)
        if transport is not None:
            await transport.aclose()
