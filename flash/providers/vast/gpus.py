"""Vast's GPU classes + the offer->class mapping.

The class table is provider-agnostic and lives in ``providers/base.py``. This module
carves out Vast's rows (``gpu_classes()`` == every class with a ``vast_name`` — currently
RTX 4090 / RTX 5090 / A100 PCIe / A100 SXM 40GB / A100 SXM 80GB / H100). The
offer->class mapping (``vast_gpu_for_offer``) lives in ``providers/base.py`` and the job
path imports it from there directly.
"""

from __future__ import annotations

from flash.providers.base import GpuClass

__all__ = ["gpu_classes"]


def gpu_classes() -> list[GpuClass]:
    """The GPU classes Vast can provision (those with a ``vast_name``)."""
    from flash.providers.base import GPU_INFO

    return [g for g in GPU_INFO.values() if g.vast_name]
