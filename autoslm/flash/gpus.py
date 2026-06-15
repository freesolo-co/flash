"""Compatibility shim: the GPU table moved to ``autoslm.providers.base`` (shared,
provider-agnostic) and the RunPod-specific bits to ``autoslm.providers.runpod.gpus``.

Re-exports every historical ``flash.gpus`` symbol so existing imports keep working.
"""

from __future__ import annotations

from autoslm.providers.base import (
    GPU_INFO,
    KNOWN,
    POLICY_NAMES,
    SUPPORTED,
    GpuClass,
    GpuInfo,
    UnsupportedGpuError,
    canonical_gpu,
    cheapest_gpu,
    get_gpu_info,
    gpu_short,
    is_validated,
    min_cuda_modern,
    providers_for,
    resolve_gpu_policy,
    unvalidated_allowed,
    vast_gpu_for_offer,
)
from autoslm.providers.runpod.gpus import flash_gpu, gpu_api_id

# Historical provider tuple lived on this module.
PROVIDERS = ("runpod", "vast")

__all__ = [
    "GPU_INFO",
    "KNOWN",
    "POLICY_NAMES",
    "PROVIDERS",
    "SUPPORTED",
    "GpuClass",
    "GpuInfo",
    "UnsupportedGpuError",
    "canonical_gpu",
    "cheapest_gpu",
    "flash_gpu",
    "get_gpu_info",
    "gpu_api_id",
    "gpu_short",
    "is_validated",
    "min_cuda_modern",
    "providers_for",
    "resolve_gpu_policy",
    "unvalidated_allowed",
    "vast_gpu_for_offer",
]
