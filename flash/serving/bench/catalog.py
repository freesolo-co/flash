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
    immutable_serving_revisions,
    serve_model_for,
    tokenizer_model_for,
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
    """Provenance rows recorded in the report manifest.

    The IMMUTABLE pins travel with the row, not just the repository names. A repository name alone
    identifies a moving target: the 27B engine pins its weights to a commit in the ``-FP8`` repo and
    its tokenizer and processor to a DIFFERENT commit in the base repo, so once either repository
    advances, a published capacity number could no longer be tied to the exact weights or tokenizer
    that produced it. ``immutable_serving_revisions`` returns only the keys the engine actually
    pins, so a model without pins contributes an empty mapping rather than invented ones.
    """
    rows = []
    for model in bench_base_models():
        overrides = bench_engine_overrides_for(model)
        rows.append(
            {
                "base_model": model,
                "gpu": bench_gpu_for(model),
                "serve_model_id": overrides.get("serve_model_id") or serve_model_for(model),
                "tokenizer_model": tokenizer_model_for(model),
                "immutable_revisions": immutable_serving_revisions(model),
                "max_model_len": overrides.get("max_model_len"),
                "max_num_seqs": overrides.get("max_num_seqs"),
                "quantization": overrides.get("quantization", "fp8"),
                "max_loras": overrides.get("max_loras"),
                "max_lora_rank": overrides.get("max_lora_rank"),
                # The whole resolved mapping, not only the fields promoted above. Those are kept as
                # top-level keys because readers and tests index them by name, but a hand-picked
                # subset omits capacity-defining settings -- `gpu_memory_utilization`,
                # `enforce_eager`, `max_num_batched_tokens`, `pin_loras`, `reasoning_parser` -- and
                # the 35B curve is only interpretable against the exact engine shape that produced
                # it. Once production drifts, an artifact without this cannot say which shape it
                # measured, and the promoted subset would silently stay plausible.
                "engine_overrides": overrides,
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
