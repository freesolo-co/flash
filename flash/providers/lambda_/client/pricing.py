"""use lambda's live instance-type rate with a lambda-specific static fallback.

never fall back to ``GpuClass.hourly_usd``; that is a runpod snapshot and prices differ by provider.
"""

from __future__ import annotations

from flash._internal.logging import get_logger

logger = get_logger(__name__)

# Lambda list prices (snapshot 2026-06-25, from /instance-types). Live rates override these.
_STATIC_RATES: dict[str, float] = {
    "A10": 1.29,
    "A100 SXM 40GB": 1.99,
    "H100": 3.29,
    "B200": 6.99,
}


def static_hourly_rate(name: str) -> float:
    from flash.providers.core.base import get_gpu_info

    return _STATIC_RATES.get(name) or get_gpu_info(name).hourly_usd


def hourly_rate(gpu_name: str, *, gpu_count: int = 1, deadline_at: float | None = None) -> float:
    """$/hr for a Lambda INSTANCE of this class (live ``/instance-types`` if available, else static).

    ``gpu_count`` > 1 prices the N-card instance type, whose live rate is for the whole box. The
    static fallback is a 1x list price, so it is scaled by the count to stay in the same units.
    """
    from flash.providers.core.base import canonical_gpu, get_gpu_info

    name = canonical_gpu(gpu_name)
    info = get_gpu_info(name)
    count = max(1, int(gpu_count))
    if info.lambda_name:
        try:
            from flash.providers.lambda_.client.api import (
                instance_type_price_usd_hr,
                list_instance_types,
            )
            from flash.providers.lambda_.client.gpus import instance_type_for

            # query lambda's actual multi-card sku because its suffix can differ from the 1x name.
            # skip the catalog on single-card quotes: instance_type_for already has the registry name,
            # and an unnecessary catalog outage would otherwise force a static fallback.
            catalog = list_instance_types(deadline_at=deadline_at) if count > 1 else None
            live = instance_type_price_usd_hr(
                instance_type_for(name, count, catalog), deadline_at=deadline_at
            )
            if live:
                return live
        except Exception as exc:
            logger.debug("live lambda pricing unavailable for %s (%s); using static", name, exc)
    return static_hourly_rate(name) * count
