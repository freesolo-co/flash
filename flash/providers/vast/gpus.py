"""Vast's GPU classes (every class with a ``vast_name``).

The class table is provider-agnostic (``providers/base.py``); the offer->class mapping
(``vast_gpu_for_offer``) also lives there and the job path imports it directly.
"""

from __future__ import annotations

from flash.providers.base import GpuClass

__all__ = ["gpu_classes"]


def gpu_classes() -> list[GpuClass]:
    """The GPU classes Vast can provision (those with a ``vast_name``)."""
    from flash.providers.base import GPU_INFO

    return [g for g in GPU_INFO.values() if g.vast_name]
