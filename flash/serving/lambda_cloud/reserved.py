"""Explicitly provisioned H100s: CPU proxies never launch or release their VMs.

Eight Lambda H100 workers, one model per slot (``SLOT_MODELS``):

- slots 1-5 serve ``Qwen/Qwen3.8-27B`` (its official FP8 checkpoint, one H100 each);
- slots 6-8 serve ``Qwen/Qwen3.5-0.8B-Base`` (the spec-decoding drafter).

Qwen3.5-9B-Base has no reserved slots; it is served by the on-demand Lambda pool with its
Modal GPU fallback. In ``main`` the reserved models are served only by their slots: no
on-demand Lambda allocation and no Modal GPU fallback. Development keeps ordinary pools.

The workers secret lists exactly the configured workers. Every entry names its model, which
must be its slot's model, so a worker can never be routed requests for another model.

Nothing here launches, renews or terminates the instances: the required operator procedure
(bring-up, replacing a worker, retiring the fleet) is docs/reserved-h100-serving.md.
"""

import asyncio
import json

from .proxy import LambdaEngineProxy
from .transport import WorkerTransport

LARGE_MODEL = "Qwen/Qwen3.8-27B"
DRAFTER_MODEL = "Qwen/Qwen3.5-0.8B-Base"
SLOT_COUNT = 8
SLOT_MODELS = {slot: LARGE_MODEL if slot <= 5 else DRAFTER_MODEL for slot in range(1, 9)}
MODELS = tuple(dict.fromkeys(SLOT_MODELS.values()))
# Fresh names (2026-11-23): the 15-slot 9B Base fleet's secret and health Dict hold entries for
# a different slot layout and must never be read as this fleet's configuration.
SECRET_NAME = "flash-reserved-h100-workers-20261123"
WORKERS_ENV = "FLASH_RESERVED_H100_WORKERS"
REGISTER_TIMEOUT_SECONDS = 10 * 60  # a hung worker must not hold every adapter promotion


def model_for(slot):
    try:
        return SLOT_MODELS[slot]
    except KeyError:
        raise ValueError(f"Reserved slot {slot} is not declared") from None


def slots_for(model):
    return tuple(slot for slot, slot_model in SLOT_MODELS.items() if slot_model == model)


def enabled(model, environment):
    """Reserved serving runs in main only (Modal resolves an unset environment to main)."""
    return model in MODELS and (environment or "main") == "main"


def parse_workers(raw):
    """Validated secret entries: unique declared slots, each naming its slot's model."""
    workers = json.loads(raw)
    if not isinstance(workers, list) or not workers:
        raise ValueError("Reserved fleet must configure at least one slot")
    slots = [w.get("slot") for w in workers]
    if any(type(s) is not int or s not in SLOT_MODELS for s in slots):
        raise ValueError(f"Reserved fleet slots must be integers 1..{SLOT_COUNT}")
    if len(set(slots)) != len(slots):
        raise ValueError("Reserved fleet slots must be unique")
    if len({w["instance_id"] for w in workers}) != len(workers):
        raise ValueError("Reserved fleet instance IDs must be distinct")
    if any(w.get("model") != model_for(w["slot"]) for w in workers):
        raise ValueError("Every reserved worker must name its slot's model")
    return sorted(workers, key=lambda w: w["slot"])


def worker_config(slot, raw):
    worker = next((w for w in parse_workers(raw) if w["slot"] == slot), None)
    if worker is None:
        raise ValueError(f"Reserved slot {slot} is not configured")
    return worker


def identity_ok(health, revision, model):
    return (
        health.get("ok") is True
        and health.get("base_model") == model
        and health.get("deployment_sha") == revision
        and health.get("configured_gpu") == "H100"
    )


class ReservedLambdaEngineProxy(LambdaEngineProxy):
    """CPU proxy bound to one reserved slot's worker, read from the workers secret."""

    reserved_slot = 0

    async def _load(self):
        import os

        model = model_for(self.reserved_slot)
        if self.base_model != model:
            raise ValueError(f"Reserved slot {self.reserved_slot} serves {model}")
        worker = worker_config(self.reserved_slot, os.environ[WORKERS_ENV])
        transport = WorkerTransport(worker["ip"], worker["tls"])
        try:
            async with asyncio.timeout(30):
                health = await transport.call("/health")
            if not identity_ok(health, worker["revision"], model):
                raise RuntimeError("Reserved H100 model, revision, or hardware identity mismatch")
        except BaseException:
            await transport.aclose()
            raise
        self._transport = transport
        self.engine, self._owner, self._health_cache = transport, worker["instance_id"], health

    async def _shutdown(self):
        transport = getattr(self, "_transport", None)
        if transport is not None:
            await transport.aclose()
