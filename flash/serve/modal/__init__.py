"""reusable modal plumbing for vllm serving deployments.

importing this package must not require modal, vllm, torch, or transformers, so `modal deploy`
can discover an app locally with only the lightweight serve extra installed. the modal import
happens inside `base_image`, and the runtime import happens when a container starts.
"""

from __future__ import annotations

from .engine import RuntimeContainer, bind_module_class
from .image import APT_PACKAGES, CUDA_IMAGE, PYTHON_VERSION, base_image, cache_environment

__all__ = [
    "APT_PACKAGES",
    "CUDA_IMAGE",
    "PYTHON_VERSION",
    "RuntimeContainer",
    "base_image",
    "bind_module_class",
    "cache_environment",
]
