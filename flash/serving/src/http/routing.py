"""Adapter -> base-model routing state: which base model each adapter belongs to, and the
engine pool protocol the app dispatches through.

Split out of router.py so the routing state machine can be exercised without constructing the
FastAPI app. Holds no app or request state -- pure bookkeeping over adapter records.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from flash.schema import parse_checkpoint_ref
from flash.serving.src.engine.model_config import (
    gpu_for,
    is_supported_base_model,
)
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.registry import AdapterRegistry


class EnginePool(Protocol):
    """One vLLM engine per base model (Modal container in prod, fake in tests)."""

    async def generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        # record is forwarded so the engine can lazy-load an adapter it hasn't seen.
        ...

    def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        # Streaming variant of generate(), yielding delta/final events.
        ...

    async def register(self, base_model: str, record: AdapterRecord) -> None: ...
    async def unregister(
        self,
        base_model: str,
        org_id: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        raise NotImplementedError


class AdapterRouter:
    """Tracks adapter -> base_model so a request naming an adapter routes to its engine."""

    def __init__(self, records: list[AdapterRecord] | None = None) -> None:
        self._registry = AdapterRegistry()
        if records:
            self._registry.hydrate(records)

    def hydrate(self, records: list[AdapterRecord]) -> None:
        self._registry.hydrate(records)

    def upsert(self, record: AdapterRecord, *, revive: bool = False) -> AdapterRecord:
        return self._registry.upsert(record, revive=revive)

    def get(self, adapter_id: str, *, org_id: str | None = None) -> AdapterRecord | None:
        return self._registry.get(org_id, adapter_id)

    def has(self, adapter_id: str, *, org_id: str | None = None) -> bool:
        return self._registry.has(org_id, adapter_id)

    def remove(self, adapter_id: str, *, org_id: str | None = None) -> AdapterRecord | None:
        return self._registry.remove(org_id, adapter_id)

    def resolve(
        self, adapter_id: str, *, org_id: str | None = None
    ) -> tuple[AdapterRecord, AdapterRecord] | None:
        lookup_org = org_id if parse_checkpoint_ref(adapter_id) is not None else None
        requested = self._registry.get(lookup_org, adapter_id)
        if requested is None or requested.status != "ready":
            return None
        if requested.serve_base_model:
            return requested, requested
        if requested.is_checkpoint and requested.org_id is not None:
            return requested, requested
        return None

    def base_models(self) -> list[str]:
        return sorted({target.base_model for _, target in self._resolved_ready()})

    def ready_adapters(self, *, org_id: str | None = None) -> list[AdapterRecord]:
        return [requested for requested, _ in self._resolved_ready(org_id=org_id)]

    def ready_records(self, *, org_id: str | None = None) -> list[AdapterRecord]:
        return self._registry.list_ready(org_id=org_id)

    def _resolved_ready(
        self, *, org_id: str | None = None
    ) -> list[tuple[AdapterRecord, AdapterRecord]]:
        resolved: list[tuple[AdapterRecord, AdapterRecord]] = []
        for record in self._registry.list_ready(org_id=org_id):
            pair = self.resolve(record.adapter_id, org_id=record.org_id)
            if pair is not None:
                resolved.append(pair)
        return resolved


def health_body(
    router: AdapterRouter,
    *,
    deployment_sha: str,
    deployment_id: str,
    capabilities: list[str],
) -> dict[str, Any]:
    """/healthz body: what this deployment is, and which base models it can serve."""
    models = router.base_models()
    supported_models = [m for m in models if is_supported_base_model(m)]
    unsupported_models = [m for m in models if not is_supported_base_model(m)]
    # report configured per-model gpu tiers rather than live container counts, which modal does
    # not expose here. ``gpus`` is the supported base-model engine count and remains stable when
    # demand-driven containers scale to zero.
    gpu_by_model = {m: gpu_for(m) for m in supported_models}
    body = {
        "ok": True,
        "deployment_sha": deployment_sha,
        "deployment_id": deployment_id,
        "capabilities": capabilities,
        "base_models": models,
        "gpus": len(supported_models),
        "gpu_by_model": gpu_by_model,
        "gpu_tiers": sorted(set(gpu_by_model.values())),
        "adapters": len(router.ready_adapters()),
    }
    if unsupported_models:
        body["unsupported_base_models"] = unsupported_models
    return body
