"""Benchmark view of the hosted serving catalog.

The capacity campaign measures each model **on the tier that actually serves it**, so this module
adds no routing of its own. It delegates to ``model_config``'s public surface and exists only to

* fix the set of models under measurement to the deployed catalog, and
* carry provenance (tier, checkpoint, context, engine concurrency) into the report manifest.

An earlier revision of this file pinned every model to one card and merged in an inert 27B
candidate. Both are gone: all three models are in ``SERVING_MODELS`` on this head, and rewriting
``gpu`` would make a measured number attributable to a card the model is not served on.

Nothing here mutates ``SERVING_MODELS``. A benchmark that edits the deployed contract can be
deployed by accident, and its diff reads as a tier change rather than a measurement.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from flash.serving.src.engine.model_config import (
    base_models,
    engine_overrides_for,
    gpu_for,
    serve_model_for,
)

# The models under measurement ARE the deployed catalog, in its declared order. Derived rather than
# hardcoded so a model added to hosted serving cannot be silently omitted from the envelope.
BENCH_MODELS: tuple[str, ...] = tuple(base_models())


def bench_base_models() -> list[str]:
    return list(BENCH_MODELS)


def _require_catalogued(base_model: str) -> None:
    if base_model not in BENCH_MODELS:
        allowed = ", ".join(BENCH_MODELS)
        raise ValueError(f"unsupported benchmark base model {base_model!r}; supported: {allowed}")


def bench_gpu_for(base_model: str) -> str:
    """The production tier for ``base_model`` — never a benchmark-chosen card."""
    _require_catalogued(base_model)
    return gpu_for(base_model)


def bench_engine_overrides_for(base_model: str) -> dict[str, Any]:
    """A defensive copy of the production vLLM overrides for ``base_model``.

    Deliberately NOT re-derived here. ``engine_overrides_for`` already resolves the pre-quantized
    checkpoint centrally, so re-resolving it in the benchmark would be a second implementation free
    to drift from the one production boots.

    ``max_loras``/``max_lora_rank`` come through untouched. vLLM PRE-ALLOCATES those buffers at
    engine init, so dropping them would free VRAM the deployed engine does not have and every
    KV-cache and concurrency number would overstate the tier it represents.
    """
    _require_catalogued(base_model)
    return deepcopy(engine_overrides_for(base_model))


def bench_catalog_summary() -> list[dict[str, Any]]:
    """Provenance rows recorded in the report manifest."""
    rows = []
    for model in bench_base_models():
        overrides = bench_engine_overrides_for(model)
        rows.append(
            {
                "base_model": model,
                "gpu": bench_gpu_for(model),
                "serve_model_id": overrides.get("serve_model_id") or serve_model_for(model),
                "max_model_len": overrides.get("max_model_len"),
                "max_num_seqs": overrides.get("max_num_seqs"),
                "quantization": overrides.get("quantization", "fp8"),
                "max_loras": overrides.get("max_loras"),
                "max_lora_rank": overrides.get("max_lora_rank"),
            }
        )
    return rows


__all__ = [
    "BENCH_MODELS",
    "bench_base_models",
    "bench_catalog_summary",
    "bench_engine_overrides_for",
    "bench_gpu_for",
]
