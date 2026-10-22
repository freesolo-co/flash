"""Shared model weight-cache sizing and assignment."""

from __future__ import annotations

from flash.core.catalog import ModelInfo
from flash.core.spec import JobSpec

WEIGHT_CACHE_VOLUME_NAME = "flash-weights"
WEIGHT_CACHE_VOLUME_GB = 250
_WEIGHT_CACHE_PEAK_FACTOR = 2.0


def _download_gb(info: ModelInfo) -> float:
    """Full bf16 checkpoint size in GB, from catalog geometry (2 bytes/param).

    Same rule as ``cost.facts.download_weight_gb``, which cannot be reused here: it resolves a model
    *id* and fail-closes on anything off-catalog, while cache sizing runs on a ``ModelInfo`` that
    may legitimately carry no ``params_b``.
    """
    return (info.params_b or 0.0) * 2.0


def _peak_gb(info: ModelInfo) -> float:
    """GB the volume must have free for this model to finish downloading."""
    return _WEIGHT_CACHE_PEAK_FACTOR * _download_gb(info)


def _fits_weight_cache(info: ModelInfo) -> bool:
    """Whether the model's peak download footprint fits the shared weight-cache volume.

    Sizes the model against an EMPTY volume. ``weight_cache_catalog_peak_gb`` answers that
    cumulative question; keep both in agreement when adding a large model.
    """
    if not info.params_b:
        return True
    return _peak_gb(info) <= WEIGHT_CACHE_VOLUME_GB


def weight_cache_catalog_peak_gb() -> float:
    """Peak GB the volume must hold to warm the WHOLE catalog onto one shared volume.

    That is strictly more than any single model's own peak, which is why sizing the volume per-model
    let the 35B fail with "Disk quota exceeded" on every datacenter at 200 GB. Derived from
    ``params_b``, so it slightly understates a repo that also ships tokenizer/config/ index files;
    the measured-bytes figure for today's catalog is ~5 GB higher.
    """
    from flash.core.catalog import MODELS

    cached = [info for info in MODELS.values() if _fits_weight_cache(info)]
    if not cached:
        return 0.0
    largest = max(cached, key=_download_gb)
    resident_others = sum(_download_gb(info) for info in cached if info is not largest)
    return resident_others + _peak_gb(largest)


def _assign_weight_cache_volume(spec: JobSpec, info: ModelInfo | None = None) -> JobSpec:
    """Attach the shared weight-cache volume, which every run is now eligible for.

    Only curated models are trainable, and their weights are public, so the shared cross-tenant
    mount holds nothing private. A pre-set non-shared volume is always honored as-is, and an
    oversized model still fails ``_fits_weight_cache``.
    """
    existing = getattr(spec.gpu, "network_volume", None)
    if existing and existing != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    attach = info is None or _fits_weight_cache(info)
    pinned = existing == WEIGHT_CACHE_VOLUME_NAME
    # an already-pinned spec is only correct if it also carries the current managed size. a stale or
    # internally-round-tripped spec can hold the shared name at a previous, smaller size. taking the
    # no-op return there would deploy an undersized volume for models this size now admits.
    sized = getattr(spec.gpu, "network_volume_gb", None) == WEIGHT_CACHE_VOLUME_GB
    if attach == pinned and (sized or not attach):
        return spec
    d = spec.to_internal_dict()
    if attach:
        d["gpu"] = {
            **d["gpu"],
            "network_volume": WEIGHT_CACHE_VOLUME_NAME,
            "network_volume_gb": WEIGHT_CACHE_VOLUME_GB,
        }
    else:
        d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)
