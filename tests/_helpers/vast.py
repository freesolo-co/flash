"""Shared VastOffer factory.

The single source of truth for the test offer shape, replacing the four hand-rolled
copies the audit flagged (test_vast_runner / test_provider_routing / test_providers_
symmetry each built their own). Adding a VastOffer field only changes the defaults here.
"""

from __future__ import annotations


def make_vast_offer(
    *,
    offer_id: int = 1,
    machine_id: int = 10,
    gpu: str = "RTX 3090",
    vram_gb: int = 24,
    dph_total: float = 0.25,
    cuda_max_good: float = 12.8,
    disk_space: float = 200.0,
    reliability: float = 0.99,
    inet_down: float = 5000.0,
    geolocation: str = "CZ",
):
    """A fully-vetted ``VastOffer`` with sensible defaults (override any field)."""
    from autoslm.providers.vast.jobs import VastOffer

    return VastOffer(
        offer_id=offer_id,
        machine_id=machine_id,
        gpu=gpu,
        vram_gb=vram_gb,
        dph_total=dph_total,
        cuda_max_good=cuda_max_good,
        disk_space=disk_space,
        reliability=reliability,
        inet_down=inet_down,
        geolocation=geolocation,
    )
