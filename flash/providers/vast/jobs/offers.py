"""Decode and filter Vast market offers before construction."""

from __future__ import annotations

from flash._internal.logging import get_logger
from flash.providers._lifecycle.net.deadline import deadline_kwargs
from flash.providers.core._decoding import (
    MISSING_PROVIDER_FIELD,
    MalformedProviderFieldError,
    decode_finite_number,
    decode_positive_int,
)
from flash.providers.core.base import (
    GPU_INFO,
    UnsupportedGpuError,
    canonical_gpu,
    min_cuda_modern,
    vast_gpu_for_offer,
)
from flash.providers.vast.client import api as vast_api
from flash.providers.vast.jobs.builders import VastOffer

logger = get_logger(__name__)

RELIABILITY_FLOOR = 0.995
MIN_INET_MBPS = 200.0
_SEARCH_VRAM_SLACK = 0.92
MIN_DISK_GB = 60.0


def _exact_search_aliases(info) -> tuple[str, ...]:
    """Return Vast aliases safe for an exact-class search."""
    kept: list[str] = []
    for alias in info.vast_aliases:
        try:
            if canonical_gpu(alias) == info.name:
                kept.append(alias)
        except UnsupportedGpuError:
            pass
    return tuple(kept)


def _vast_number(row: dict, field: str) -> float:
    value = decode_finite_number(
        row.get(field, MISSING_PROVIDER_FIELD),
        provider="vast",
        field=field,
    )
    if value is MISSING_PROVIDER_FIELD or value is None:
        raise MalformedProviderFieldError("vast", field, "a finite number")
    return value


def _vast_positive_int(row: dict, field: str) -> int:
    value = decode_positive_int(
        row.get(field, MISSING_PROVIDER_FIELD),
        provider="vast",
        field=field,
    )
    if value is MISSING_PROVIDER_FIELD or value is None:
        raise MalformedProviderFieldError("vast", field, "a positive integer")
    return value


def _vast_nonempty_string(row: dict, field: str) -> str:
    value = row.get(field, MISSING_PROVIDER_FIELD)
    if not isinstance(value, str) or not value.strip():
        raise MalformedProviderFieldError("vast", field, "a nonempty string")
    return value


def _decode_vast_offer_row(row: object) -> dict:
    """Decode one Vast row before any offer or candidate is constructed."""
    if not isinstance(row, dict):
        raise MalformedProviderFieldError("vast", "offer", "an object")
    decoded = {
        "id": _vast_positive_int(row, "id"),
        "machine_id": _vast_positive_int(row, "machine_id"),
        "gpu_name": _vast_nonempty_string(row, "gpu_name"),
        "verification": _vast_nonempty_string(row, "verification"),
        "gpu_ram": _vast_number(row, "gpu_ram"),
        "num_gpus": _vast_positive_int(row, "num_gpus"),
        "dph_total": _vast_number(row, "dph_total"),
        "hosting_type": _vast_number(row, "hosting_type"),
        "reliability2": _vast_number(row, "reliability2"),
        "cuda_max_good": _vast_number(row, "cuda_max_good"),
        "disk_space": _vast_number(row, "disk_space"),
        "inet_down": _vast_number(row, "inet_down"),
    }
    hosting_type = decoded["hosting_type"]
    if not hosting_type.is_integer() or int(hosting_type) not in (0, 1):
        raise MalformedProviderFieldError("vast", "hosting_type", "0 or 1")
    decoded["hosting_type"] = int(hosting_type)
    bounds = {
        "gpu_ram": decoded["gpu_ram"] > 0,
        "dph_total": decoded["dph_total"] > 0,
        "cuda_max_good": decoded["cuda_max_good"] >= 0,
        "reliability2": 0 <= decoded["reliability2"] <= 1,
        "disk_space": decoded["disk_space"] >= 0,
        "inet_down": decoded["inet_down"] >= 0,
    }
    for field, valid in bounds.items():
        if not valid:
            raise MalformedProviderFieldError("vast", field, "a value within provider bounds")
    geolocation = row.get("geolocation", MISSING_PROVIDER_FIELD)
    if geolocation is MISSING_PROVIDER_FIELD or geolocation is None:
        decoded["geolocation"] = ""
    elif isinstance(geolocation, str):
        decoded["geolocation"] = geolocation
    else:
        raise MalformedProviderFieldError("vast", "geolocation", "a string or null")
    return decoded


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    limit: int = 256,
    max_wall_seconds: float = 0,
    gpu_type: str = "",
    num_gpus: int = 1,
    deadline_at: float | None = None,
) -> list[VastOffer]:
    """Return fitting verified-datacenter offers, cheapest first."""
    min_duration = (
        max(60.0, float(max_wall_seconds)) if max_wall_seconds and max_wall_seconds > 0 else 0
    )
    exact_info = GPU_INFO.get(gpu_type) if gpu_type else None
    if gpu_type and exact_info is None:
        raise ValueError(f"unknown exact Vast GPU class {gpu_type!r}")
    gpu_names = (
        (exact_info.vast_name, *_exact_search_aliases(exact_info))
        if exact_info is not None and exact_info.vast_name
        else ()
    )
    search_vram_gb = max(min_vram_gb, exact_info.vram_gb if exact_info is not None else 0)
    search_kwargs = {"gpu_names": gpu_names} if gpu_names else {}
    if exact_info is not None:
        search_kwargs["max_vram_mb"] = int(exact_info.vram_gb * 1024)
    cards = max(1, int(num_gpus))
    rows = vast_api.search_offers(
        int(search_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        min_duration_seconds=min_duration,
        limit=int(limit),
        num_gpus=cards,
        **search_kwargs,
        **deadline_kwargs(vast_api.search_offers, deadline_at),
    )
    out: list[VastOffer] = []
    decoded_rows = 0
    for raw_row in rows:
        try:
            row = _decode_vast_offer_row(raw_row)
        except MalformedProviderFieldError as exc:
            logger.warning("dropping malformed vast offer row: %s", exc)
            continue
        decoded_rows += 1
        gpu = vast_gpu_for_offer(row["gpu_name"], row["gpu_ram"])
        if gpu is None:
            continue
        info = GPU_INFO[gpu]
        if (
            row["hosting_type"] != 1
            or row["verification"] != "verified"
            or info.vram_gb < min_vram_gb
            or (gpu_type and gpu != gpu_type)
            or row["reliability2"] < RELIABILITY_FLOOR
            or row["disk_space"] < float(disk_gb)
            or row["inet_down"] < MIN_INET_MBPS
            or row["cuda_max_good"] < float(min_cuda_modern(gpu))
            or row["num_gpus"] != cards
            or row["machine_id"] in exclude_machine_ids
        ):
            continue
        out.append(
            VastOffer(
                offer_id=row["id"],
                machine_id=row["machine_id"],
                gpu=gpu,
                vram_gb=info.vram_gb,
                dph_total=row["dph_total"] / cards,
                cuda_max_good=row["cuda_max_good"],
                disk_space=row["disk_space"],
                reliability=row["reliability2"],
                inet_down=row["inet_down"],
                geolocation=row["geolocation"],
                gpu_count=cards,
            )
        )
    if rows and decoded_rows == 0:
        raise MalformedProviderFieldError("vast", "offers", "at least one well-formed row")
    return sorted(out, key=lambda offer: (offer.dph_total, offer.vram_gb))
