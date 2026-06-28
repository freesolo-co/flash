"""Lambda Cloud $/hr: live ``/instance-types`` rate per class, static fallback.

NB: the static fallback is a Lambda-specific map, NOT ``GpuClass.hourly_usd`` — that field is the
RunPod secure-cloud snapshot, which differs from Lambda's list price (e.g. RTX A6000 is $0.49 on
RunPod but $1.09 on Lambda).
"""

from __future__ import annotations

from flash._logging import get_logger

logger = get_logger(__name__)

# Lambda list prices (snapshot 2026-06-25, from /instance-types). Live rates override these.
_STATIC_RATES: dict[str, float] = {
    "A10": 1.29,
    "RTX A6000": 1.09,
    "A100 SXM 40GB": 1.99,
    "H100": 3.29,
    "B200": 6.99,
}


def _static_rate(name: str) -> float:
    from flash.providers.base import GPU_INFO

    return _STATIC_RATES.get(name) or GPU_INFO[name].hourly_usd


def hourly_rate(gpu_name: str) -> float:
    """$/hr for one friendly GPU name on Lambda (live ``/instance-types`` if available, else static)."""
    from flash.providers.base import canonical_gpu, get_gpu_info

    name = canonical_gpu(gpu_name)
    info = get_gpu_info(name)
    if info.lambda_name:
        try:
            from flash.providers.lambdalabs.api import instance_type_price_usd_hr

            live = instance_type_price_usd_hr(info.lambda_name)
            if live:
                return live
        except Exception as exc:
            logger.debug("live lambda pricing unavailable for %s (%s); using static", name, exc)
    return _static_rate(name)
