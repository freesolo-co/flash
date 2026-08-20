"""shared modal image and cache environment for gpu serving containers.

this module owns only the parts every vllm serving deployment needs: the cuda base image, the
build toolchain, and the cache environment that keeps cold starts cheap. callers append their own
package installation, secrets, volumes, and source mounts, because those carry deployment policy.
"""

from __future__ import annotations

from typing import Any

CUDA_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"
# vllm jits kernels at runtime, so the image needs a full toolchain rather than a slim runtime.
APT_PACKAGES = ("build-essential", "git", "ninja-build")


def cache_environment(cache_mount: str) -> dict[str, str]:
    """environment that points every cache vllm uses at one persistent mount.

    vllm's cache root defaults to ephemeral container storage, so leaving it unset makes every
    scale-from-zero recompile the model from scratch. the directory name is a hash over the engine
    config, code, and compiler, so a different version or config lands elsewhere rather than
    loading a stale graph.
    """
    if not isinstance(cache_mount, str) or not cache_mount.strip():
        raise ValueError("cache_mount must be a non-empty string")
    mount = cache_mount.rstrip("/")
    return {
        "HF_HOME": mount,
        "HF_HUB_CACHE": f"{mount}/hub",
        "VLLM_CACHE_ROOT": f"{mount}/vllm",
        # the xet download path intermittently fails engine init, so force the http downloader.
        "HF_HUB_DISABLE_XET": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # expandable segments reduce fragmentation from repeated adapter load and unload.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # keep deepgemm out of the fp8 fused-moe backend selection.
        "VLLM_MOE_USE_DEEP_GEMM": "0",
    }


def base_image(cache_mount: str) -> Any:
    """cuda base image with the build toolchain and cache environment applied.

    the caller installs its own packages on top, so this never pins vllm or any serving dependency.
    """
    import modal

    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
        .apt_install(*APT_PACKAGES)
        .env(cache_environment(cache_mount))
    )
