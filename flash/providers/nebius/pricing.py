"""Fixed Spot rates, entry thresholds and separate hold caps, in USD per GPU-hour.

``hourly_usd`` is the fixed rate Flash estimates and charges. ``entry_below_usd`` admits a
rental when the live price is strictly below it, and ``keep_running_usd`` is the Nebius
maximum-price policy a running VM keeps. Each threshold sits $0.01 above its round figure so a
price of exactly $1.00 (or $2.00, $4.00, $5.00) still admits or keeps running.
These are user-selected thresholds, not current market prices or capacity guarantees.
GPU priority follows insertion order: H100, H200, B200.
"""

from dataclasses import dataclass
from decimal import Decimal

from flash.providers.core.base import canonical_gpu


@dataclass(frozen=True)
class SpotGpu:
    platform: str
    hourly_usd: Decimal
    entry_below_usd: Decimal
    keep_running_usd: Decimal
    cpu: int
    memory_gib: int
    counts: tuple[int, ...] = (1, 8)

    def preset(self, count: int) -> str:
        if type(count) is not int or count not in self.counts:
            raise ValueError("unsupported Nebius GPU count")
        return f"{count}gpu-{self.cpu * count}vcpu-{self.memory_gib * count}gb"


SPOT_GPUS = {
    "H100": SpotGpu("gpu-h100-sxm", Decimal("1.00"), Decimal("1.01"), Decimal("1.51"), 16, 200),
    "H200": SpotGpu("gpu-h200-sxm", Decimal("1.00"), Decimal("1.01"), Decimal("2.01"), 16, 200),
    "B200": SpotGpu("gpu-b200-sxm", Decimal("4.00"), Decimal("4.01"), Decimal("5.01"), 20, 224),
}


def hourly_rate(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].hourly_usd)


def accepts_price(gpu: str, value, *, currency: str, unit: str) -> bool:
    if currency.lower() != "usd" or unit.lower() != "gpu hour" or isinstance(value, bool):
        return False
    try:
        price = Decimal(str(value))
        return price.is_finite() and 0 < price < SPOT_GPUS[canonical_gpu(gpu)].entry_below_usd
    except (ValueError, ArithmeticError):
        return False


def priority(gpu: str) -> int:
    return tuple(SPOT_GPUS).index(canonical_gpu(gpu))


def running_limit(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].keep_running_usd)
