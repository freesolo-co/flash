"""Fixed Spot rates and one price cap per GPU class, in USD per GPU-hour.

These are user-selected thresholds, not current market prices or capacity guarantees.
Each class has a single cap: a rental starts only while the live price is below it, and the
Mithril bid's limit price keeps a running VM up to the same cap. ``hourly_usd`` is the fixed
rate Flash estimates and charges. GPU priority follows insertion order: H100, H200.
"""

from dataclasses import dataclass
from decimal import Decimal

from flash.providers.core.base import canonical_gpu


@dataclass(frozen=True)
class SpotGpu:
    hourly_usd: Decimal
    keep_running_usd: Decimal  # the one cap: start below it, keep running through it
    counts: tuple[int, ...] = (1, 8)


SPOT_GPUS = {
    "H100": SpotGpu(Decimal("2.00"), Decimal("2.01")),
    "H200": SpotGpu(Decimal("2.50"), Decimal("2.51")),
}


def hourly_rate(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].hourly_usd)


def accepts_price(gpu: str, value, *, currency: str, unit: str) -> bool:
    if currency.lower() != "usd" or unit.lower() != "gpu hour" or isinstance(value, bool):
        return False
    shape = SPOT_GPUS.get(canonical_gpu(gpu))
    if shape is None:
        return False  # not a Spot class
    try:
        price = Decimal(str(value))
        return price.is_finite() and 0 < price < shape.keep_running_usd
    except (ValueError, ArithmeticError):
        return False


def priority(gpu: str) -> int:
    return tuple(SPOT_GPUS).index(canonical_gpu(gpu))


def running_limit(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].keep_running_usd)


def quote_rate(gpu: str, count: int) -> float:
    """Per-GPU rate a Spot quote is priced at: the costliest class the run may land on.

    A run allocated on ``gpu`` may launch on any ``interchangeable`` class, so its quote is an
    upper bound; billing scales it to the class actually rented (``billed_rate_factor``).
    """
    return max(hourly_rate(name) for name in interchangeable(gpu, count) or (gpu,))


def billed_rate_factor(remote) -> float:
    """Charge share of a Spot quote for the class actually rented: rented rate / quoted rate.

    Only handles that record ``quoted_gpu`` (quotes priced by ``quote_rate``) are scaled; older
    handles were quoted at their own class rate and keep a factor of 1.
    """
    if not isinstance(remote, dict) or remote.get("provider") != "spot":
        return 1.0
    quoted, rented, count = remote.get("quoted_gpu"), remote.get("gpu"), remote.get("gpu_count", 1)
    try:
        quoted, rented = canonical_gpu(quoted), canonical_gpu(rented)
        if quoted not in SPOT_GPUS or rented not in SPOT_GPUS:
            return 1.0
        if type(count) is not int or count < 1:
            count = 1
        return min(1.0, hourly_rate(rented) / quote_rate(quoted, count))
    except Exception:
        return 1.0


def interchangeable(gpu: str, count: int) -> tuple[str, ...]:
    """Spot classes a run allocated on ``gpu`` x ``count`` may take instead, in priority order.

    Every catalog class offering ``count`` cards with at least the allocated class's VRAM can host
    the same plan, so a queued run waits for all of them and takes whichever comes up first.
    """
    from flash.providers.core.base import get_gpu_info

    floor = get_gpu_info(gpu).vram_gb
    return tuple(
        name
        for name, shape in SPOT_GPUS.items()
        if count in shape.counts and get_gpu_info(name).vram_gb >= floor
    )
