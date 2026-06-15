"""Vast's GPU classes + the offer->class mapping.

The class table is provider-agnostic and lives in ``providers/base.py``. This module
carves out Vast's rows (``gpu_classes()`` == every class with a ``vast_name``,
including the Vast-only classes L40S / RTX Pro 4000 / A100 SXM 40GB) and re-exports the
offer disambiguation (``vast_gpu_for_offer``) so the durable path can map a live offer
back to a canonical class.
"""

from __future__ import annotations

from autoslm.providers.base import GpuClass, vast_gpu_for_offer

__all__ = ["gpu_classes", "vast_gpu_for_offer"]


def gpu_classes() -> list[GpuClass]:
    """The GPU classes Vast can provision (those with a ``vast_name``)."""
    from autoslm.providers.base import GPU_INFO

    return [g for g in GPU_INFO.values() if g.vast_name]
