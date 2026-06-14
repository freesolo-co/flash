"""Per-GPU hourly rates: RunPod live pricing with a static fallback.

Cost projection (orchestrator, serve) and the ``gpu.type = "cheapest"`` policy both
need $/hr per GPU class. Rates move with the market, so we read them live from the
RunPod pricing API (the ``runpod`` SDK's GraphQL wrapper — the plain REST surface has
no GPU-types route and direct GraphQL is 403 for scoped keys) and cache them on disk;
any failure falls back to the static snapshot in ``flash.gpus.GPU_INFO``.

Rates are RunPod secure-cloud on-demand — representative for ranking and projection,
not an exact serverless invoice (the worker also records wall time; real cost comes
from RunPod billing).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from autoslm._logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_S = float(os.environ.get("AUTOSLM_PRICE_TTL_S", str(6 * 3600)))
_CACHE_PATH = Path.home() / ".autoslm" / "gpu_rates.json"
_MEM: dict = {"ts": 0.0, "rates": {}}


def _static_rates() -> dict[str, float]:
    from autoslm.flash.gpus import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items()}


def _pick_rate(detail: dict) -> float | None:
    """Best representative on-demand rate from a RunPod gpuTypes detail row."""
    for key in ("securePrice", "communityPrice"):
        v = detail.get(key)
        if v:
            return float(v)
    v = (detail.get("lowestPrice") or {}).get("uninterruptablePrice")
    return float(v) if v else None


def _fetch_live_rates() -> dict[str, float]:
    """One pricing call per managed GPU class (the list query carries no prices)."""
    import runpod

    from autoslm.flash.gpus import GPU_INFO, gpu_api_id

    if not runpod.api_key:
        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
    rates: dict[str, float] = {}
    for name in GPU_INFO:
        try:
            rate = _pick_rate(runpod.get_gpu(gpu_api_id(name)) or {})
        except Exception as exc:
            logger.debug("live rate fetch failed for %s: %s", name, exc)
            continue
        if rate:
            rates[name] = rate
    return rates


def live_rates(refresh: bool = False) -> dict[str, float]:
    """Friendly-name -> $/hr. Live (cached ``CACHE_TTL_S``) over the static snapshot.

    Offline-safe: AUTOSLM_SKIP_NET (or any fetch failure) returns the static rates.
    """
    static = _static_rates()
    if os.environ.get("AUTOSLM_SKIP_NET"):
        return static
    now = time.time()
    if not refresh:
        if _MEM["rates"] and now - _MEM["ts"] < CACHE_TTL_S:
            return {**static, **_MEM["rates"]}
        try:
            disk = json.loads(_CACHE_PATH.read_text())
            if now - float(disk.get("ts", 0)) < CACHE_TTL_S and disk.get("rates"):
                _MEM.update(ts=float(disk["ts"]), rates=dict(disk["rates"]))
                return {**static, **_MEM["rates"]}
        except Exception:
            # Corrupt/unreadable cache: ignore and fall through to a live fetch.
            pass
    try:
        fetched = _fetch_live_rates()
    except Exception as exc:
        logger.warning("live GPU pricing unavailable (%s); using static rates", exc)
        fetched = {}
    if fetched:
        _MEM.update(ts=now, rates=fetched)
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps({"ts": now, "rates": fetched}))
        except Exception:
            # Cache write is an optimization; a read-only/full FS shouldn't fail pricing.
            pass
    return {**static, **_MEM["rates"]}


def hourly_rate(gpu_name: str) -> float:
    """$/hr for one friendly GPU name (live if available, else static)."""
    from autoslm.flash.gpus import canonical_gpu

    name = canonical_gpu(gpu_name)
    return live_rates().get(name) or _static_rates()[name]
