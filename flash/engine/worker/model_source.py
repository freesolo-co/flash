"""resolve the concrete model source shared by trainers and rollout engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from flash.engine.worker._pkg import W as _w


@dataclass(frozen=True)
class ResolvedModelSource:
    canonical_model: str
    source: str
    setup_seconds: float
    interpolation: dict[str, Any] | None = None


def resolve_model_source(canonical_model: str) -> ResolvedModelSource:
    """return one concrete source and cache it for every consumer in this worker."""
    cached = getattr(_w, "RESOLVED_MODEL_SOURCE", None)
    if cached is not None:
        if cached.canonical_model != canonical_model:
            raise RuntimeError(
                "model source parity failure: worker already resolved "
                f"{cached.canonical_model!r}, requested {canonical_model!r}"
            )
        return cached

    started = time.time()
    interpolation = getattr(getattr(_w, "JOB_SPEC", None), "model_initialization", None)
    if interpolation is None:
        seconds = _w.prefetch_model(canonical_model)
        resolved = ResolvedModelSource(
            canonical_model=canonical_model,
            source=canonical_model,
            setup_seconds=seconds,
        )
    else:
        from flash.engine.worker.hf import _shared_weight_cache_dir
        from flash.engine.worker.model_interpolation import materialize_model_interpolation

        result = materialize_model_interpolation(
            interpolation,
            cache_dir=_shared_weight_cache_dir(),
        )
        seconds = round(time.time() - started, 1)
        resolved = ResolvedModelSource(
            canonical_model=canonical_model,
            source=result.source,
            setup_seconds=seconds,
            interpolation=result.manifest,
        )
        _w.heartbeat(
            "model_interpolation_materialized",
            model=canonical_model,
            model_source=result.source,
            output_fingerprint=result.output_fingerprint,
            download_seconds=seconds,
        )
    _w.RESOLVED_MODEL_SOURCE = resolved
    return resolved


def assert_model_source_parity(*sources: str) -> str:
    """fail closed unless every trainer and rollout source is exactly identical."""
    normalized = [str(source) for source in sources]
    if not normalized or any(not source for source in normalized):
        raise RuntimeError("model source parity failure: source is empty")
    if len(set(normalized)) != 1:
        raise RuntimeError(f"model source parity failure: {normalized}")
    return normalized[0]
