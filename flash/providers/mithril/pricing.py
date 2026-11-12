"""Fixed Spot entry estimates and separate hold caps, in USD per GPU-hour.

These are user-selected thresholds, not current market prices or capacity guarantees.
GPU priority follows insertion order: H100, H200, B200.
"""

from dataclasses import dataclass
from decimal import Decimal

from flash.providers.core.base import canonical_gpu


@dataclass(frozen=True)
class SpotGpu:
    hourly_usd: Decimal
    keep_running_usd: Decimal
    counts: tuple[int, ...] = (1, 8)


SPOT_GPUS = {
    "H100": SpotGpu(Decimal("1.00"), Decimal("1.50")),
    "H200": SpotGpu(Decimal("1.00"), Decimal("2.00")),
    "B200": SpotGpu(Decimal("4.00"), Decimal("5.00")),
}


def hourly_rate(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].hourly_usd)


def accepts_price(gpu: str, value, *, currency: str, unit: str) -> bool:
    if currency.lower() != "usd" or unit.lower() != "gpu hour" or isinstance(value, bool):
        return False
    try:
        price = Decimal(str(value))
        return price.is_finite() and 0 < price < SPOT_GPUS[canonical_gpu(gpu)].hourly_usd
    except (ValueError, ArithmeticError):
        return False


def priority(gpu: str) -> int:
    return tuple(SPOT_GPUS).index(canonical_gpu(gpu))


def running_limit(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].keep_running_usd)
