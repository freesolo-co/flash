"""One fixed Spot rate per GPU class, in USD per GPU-hour, which is also the price cap.

These are user-selected thresholds, not current market prices or capacity guarantees.
``hourly_usd`` is what Flash estimates and charges, and the cap: a rental starts only while the
live price is at or below it, and the Mithril bid's limit price keeps a running VM up to it.
GPU priority follows insertion order: H100, H200.
"""

from dataclasses import dataclass
from decimal import Decimal

from flash.providers.core.base import canonical_gpu


@dataclass(frozen=True)
class SpotGpu:
    # the charged rate, and by default the one cap: start at or below it, keep running up to it
    hourly_usd: Decimal
    counts: tuple[int, ...] = (1, 8)
    # only a paid smoke test lowers this below the rate (scripts/smoke_mithril_sft.py); production
    # classes leave it unset, so the cap is the rate
    keep_running_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if self.keep_running_usd is None:
            object.__setattr__(self, "keep_running_usd", self.hourly_usd)


SPOT_GPUS = {
    "H100": SpotGpu(Decimal("2.10")),
    "H200": SpotGpu(Decimal("2.60")),
}


# The tariff every Spot run accepted before rates rose to the current table. Runs submitted before
# submit-time rates were recorded (``RunStatus.submitted_spot_rates`` absent) settle on these, so a
# rate change never reprices a quote that was already accepted.
LEGACY_SPOT_RATES = {"H100": Decimal("2.00"), "H200": Decimal("2.50")}


def tariff() -> dict[str, float]:
    """Current per-GPU-hour rate of every Spot class, as recorded on a run at submission."""
    return {gpu: float(shape.hourly_usd) for gpu, shape in SPOT_GPUS.items()}


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
        return price.is_finite() and 0 < price <= shape.keep_running_usd
    except (ValueError, ArithmeticError):
        return False


MAX_SPOT_CARDS = max(count for shape in SPOT_GPUS.values() for count in shape.counts)


def shape_price(gpu: str, count: int) -> Decimal:
    """Fixed rental price of ``count`` cards of ``gpu`` per hour: what Spot allocation ranks on."""
    return SPOT_GPUS[canonical_gpu(gpu)].hourly_usd * count


def priority(gpu: str) -> int:
    return tuple(SPOT_GPUS).index(canonical_gpu(gpu))


def running_limit(gpu: str) -> float:
    return float(SPOT_GPUS[canonical_gpu(gpu)].keep_running_usd)


def quote_rate() -> float:
    """Per-GPU rate every Spot quote is priced at: the costliest Spot class.

    Any Spot run may end up on the costliest class (an H100 run may take an H200 that comes up
    first), so quotes are an upper bound; settlement bills each stretch at the class it actually
    ran on (``flash.runner.accounting.spot_usage``).
    """
    return max(float(shape.hourly_usd) for shape in SPOT_GPUS.values())


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


def spot_unsupported_message(model_id: str, algorithm: str, need: float) -> str:
    """Why a run no Spot shape can hold is refused: Spot rents only these classes and counts."""
    from flash.providers.core.base import get_gpu_info

    shapes = ", ".join(
        f"{count}x {name} ({count * get_gpu_info(name).vram_gb} GB)"
        for name, shape in SPOT_GPUS.items()
        for count in shape.counts
    )
    return (
        f"Spot is not supported for {model_id} with algorithm {algorithm!r}: it needs at least "
        f"{need} GB of GPU memory, which no Spot shape can hold (Spot rents only {shapes}). "
        "Lower max_context_tokens / max_completion_tokens / batch size, or pick a smaller model"
    )
