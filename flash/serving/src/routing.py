"""Adapter -> base-model routing state: which base model each adapter belongs to, and the
engine pool protocol the app dispatches through.

Split out of router.py so the routing state machine can be exercised without constructing the
FastAPI app. Holds no app or request state -- pure bookkeeping over adapter records.
"""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from flash.serving.src.model_config import (
    gpu_for,
    is_supported_base_model,
)
from flash.serving.src.registry import AdapterRegistry
from flash.serving.src.schemas import AdapterRecord


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
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        raise NotImplementedError


class AdapterRouter:
    """Tracks adapter -> base_model so a request naming an adapter routes to its engine."""

    def __init__(
        self,
        records: list[AdapterRecord] | None = None,
        *,
        require_base_qualification: bool = False,
    ) -> None:
        self._registry = AdapterRegistry()
        self._require_base_qualification = require_base_qualification
        if records:
            self._registry.hydrate(records)

    def hydrate(self, records: list[AdapterRecord]) -> None:
        self._registry.hydrate(records)

    def upsert(self, record: AdapterRecord, *, revive: bool = False) -> AdapterRecord:
        return self._registry.upsert(record, revive=revive)

    def get(self, adapter_id: str) -> AdapterRecord | None:
        return self._registry.get(adapter_id)

    def has(self, adapter_id: str) -> bool:
        return self._registry.has(adapter_id)

    def remove(self, adapter_id: str) -> AdapterRecord | None:
        return self._registry.remove(adapter_id)

    def _qualified_base(self, base_model: str) -> AdapterRecord | None:
        record = self._registry.get(base_model)
        if (
            record is None
            or record.status != "ready"
            or not record.serve_base_model
            or record.adapter_id != base_model
            or record.base_model != base_model
            or record.repo_id != base_model
            or record.org_id is not None
        ):
            return None
        return record

    def _resolve_adapter(self, adapter_id: str) -> tuple[AdapterRecord, AdapterRecord] | None:
        requested = self._registry.get(adapter_id)
        if requested is None or requested.status != "ready" or requested.serve_base_model:
            return None
        if requested.is_revision and requested.org_id is not None:
            return requested, requested
        if not requested.is_alias or requested.org_id is None or requested.alias_of is None:
            return None
        target = self._registry.get(requested.alias_of)
        if target is None or target.status != "ready" or not target.is_revision:
            return None
        if (
            target.org_id != requested.org_id
            or target.base_model != requested.base_model
            or target.run_id != requested.run_id
            or requested.adapter_id != requested.run_id
        ):
            return None
        return requested, target

    def resolve(self, adapter_id: str) -> tuple[AdapterRecord, AdapterRecord] | None:
        requested = self._registry.get(adapter_id)
        if requested is not None and requested.status == "ready" and requested.serve_base_model:
            qualified = self._qualified_base(adapter_id)
            return (qualified, qualified) if qualified is not None else None
        resolved = self._resolve_adapter(adapter_id)
        if resolved is None:
            return None
        if (
            self._require_base_qualification
            and self._qualified_base(resolved[1].base_model) is None
        ):
            return None
        return resolved

    def is_unqualified_adapter(self, adapter_id: str) -> bool:
        if not self._require_base_qualification:
            return False
        resolved = self._resolve_adapter(adapter_id)
        return resolved is not None and self._qualified_base(resolved[1].base_model) is None

    def base_models(self) -> list[str]:
        return sorted({target.base_model for _, target in self._resolved_ready()})

    def ready_adapters(self) -> list[AdapterRecord]:
        return [requested for requested, _ in self._resolved_ready()]

    def ready_records(self) -> list[AdapterRecord]:
        return self._registry.list_ready()

    def _resolved_ready(self) -> list[tuple[AdapterRecord, AdapterRecord]]:
        resolved: list[tuple[AdapterRecord, AdapterRecord]] = []
        for record in self._registry.list_ready():
            pair = self.resolve(record.adapter_id)
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
