"""Join live auctions to actual GPU shapes; prices are USD per whole instance-hour."""

from dataclasses import dataclass
from decimal import Decimal

from flash.providers.core.base import GPU_INFO, CapacityLookupError
from flash.providers.mithril.client import MithrilApiError, call
from flash.providers.mithril.pricing import SPOT_GPUS, accepts_price
from flash.providers.mithril.region_health import cooling_down, refused_key

# Regions never bid in. af-south1-a lists capacity and accepts bids, but none of 112 Flash VMs
# allocated there booted (93 boot timeouts) while us-central2-a booted 35 of 39 (2026-11-11..13).
EXCLUDED_REGIONS = frozenset({"af-south1-a"})


@dataclass(frozen=True)
class Offer:
    gpu: str
    count: int
    instance_type: str
    hourly_usd: float  # per GPU, like Candidate.hourly_usd
    disk_gb: int
    region: str


def dollars(raw):
    if not isinstance(raw, str) or not raw.startswith("$"):
        raise ValueError("expected USD price")
    value = Decimal(raw[1:])
    if not value.is_finite() or value <= 0:
        raise ValueError("invalid USD price")
    return value


def affordable_offers(*, deadline_at=None) -> list[Offer]:
    """Qualifying offers across every region's auctions."""
    try:
        types = call("GET", "/instance-types", deadline_at=deadline_at)
        auctions = call("GET", "/spot/availability", deadline_at=deadline_at)
        if not isinstance(types, list) or not isinstance(auctions, list):
            raise ValueError("invalid catalog")
        catalog = {row["fid"]: row for row in types}
        offers = []
        for auction in auctions:
            if not isinstance(auction.get("region"), str) or not auction["region"]:
                continue
            if auction["region"] in EXCLUDED_REGIONS:
                continue
            if cooling_down(auction["region"]):
                continue  # its last filled bid never booted
            if cooling_down(refused_key(auction["region"], auction["instance_type"])):
                continue  # Mithril just refused a bid on this shape (an unlisted limit)
            shape = catalog[auction["instance_type"]]
            gpu = shape["gpu_type"]
            count = shape["num_gpus"]
            if (
                gpu not in SPOT_GPUS
                or type(count) is not int
                or count not in SPOT_GPUS[gpu].counts
                or not shape["gpu_socket"].upper().startswith("SXM")
                or shape["gpu_memory_gb"] < GPU_INFO[gpu].vram_gb
                or type(auction["capacity"]) is not int
                or auction["capacity"] <= 0
            ):
                continue
            price = dollars(auction["last_instance_price"]) / count
            if accepts_price(gpu, price, currency="usd", unit="GPU hour"):
                offers.append(
                    Offer(
                        gpu,
                        count,
                        shape["fid"],
                        float(price),
                        shape["local_storage_gb"],
                        auction["region"],
                    )
                )
        return offers
    except (MithrilApiError, ValueError, ArithmeticError, KeyError, TypeError, AttributeError):
        raise CapacityLookupError("Mithril spot prices unavailable; keeping job queued") from None
