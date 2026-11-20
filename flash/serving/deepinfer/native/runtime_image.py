"""Rebuild the compiled DeepInfer runtime image from pinned upstream sources.

Modal images are workspace-local, so each workspace that serves DeepSeek builds
this once and pins the printed id as ``RUNTIME_IMAGE_ID``:

    MODAL_PROFILE=<workspace> python -m flash.serving.deepinfer.native.runtime_image

The recipe reproduces the original snapshot's recorded identity: upstream
``./build.sh`` at ``RUNTIME_REVISION`` (whose submodules pin DeepEP, DeepGEMM,
FlashMLA and DeepSelect), CUDA 13.2.0 on Ubuntu 24.04 (glibc 2.39), Python 3.12,
torch 2.13.0+cu132, a CPU-only cross-compile for B200 (``sm_100a``), the pinned
HTTP dependencies, and finally ``native/image.py``, which checks the source
revision and bakes in the hash-checked patches. It does not seed the compiler
kernel cache, so startup uses the exhaustive warmup until a manifest is qualified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

# The compile step runs before Flash source is in the image, so this module imports
# Flash configuration only on the client and passes the pinned values explicitly.
CUDA_BASE = "nvidia/cuda:13.2.0-devel-ubuntu24.04"
UPSTREAM = "https://github.com/deepseek-ai/DeepInfer.git"
VENV = "/opt/deepinfer-venv"
PIP = "pip==26.2.2"
APT_PACKAGES = (
    "git",
    "build-essential",
    "gcc-13",
    "g++-13",
    "libnuma-dev",
    "libibverbs-dev",
    "rdma-core",
    "cmake",
    "pciutils",
    "libdw-dev",
    "libelf-dev",
)
HTTP_DEPENDENCIES = ("fastapi==0.141.1", "uvicorn==0.52.0")
BUILD_ENV = {
    "DEEPINFER_ACCELERATOR": "nvidia",
    "DEEPINFER_CROSS_COMPILE": "1",
    "VIRTUAL_ENV": VENV,
    "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CUDA_HOME": "/usr/local/cuda",
    "TORCH_CUDA_ARCH_LIST": "10.0a",
    "CC": "/usr/bin/gcc-13",
    "CXX": "/usr/bin/g++-13",
    "MAX_JOBS": "32",
    "NVCC_THREADS": "8",
    # Two FlashMLA sparse-decode kernels spill under nvcc 13.2; the qualified image's
    # FlashMLA binary contains the same spilled frames, so it was built past this gate.
    "FLASH_MLA_SKIP_REG_SPILL_CHECK": "1",
}
# The vendored setup scripts probe `lspci`, which reports no NVIDIA device on a CPU
# builder. These are the exact build-only changes recorded for the original image.
PLATFORM_FILES = (
    "vendor/DeepSelect/tests/kernelkit/platform.py",
    "vendor/FlashMLA/tests/kernelkit/platform.py",
)
PLATFORM_OLD = (
    '        return Platform.ASCEND\n    output = subprocess.check_output(["lspci"], text=True)\n'
)
PLATFORM_NEW = (
    '        return Platform.ASCEND\n    if os.getenv("DEEPINFER_CROSS_COMPILE") == "1":\n'
    '        return Platform.CUDA\n    output = subprocess.check_output(["lspci"], text=True)\n'
)
FLASH_MLA_SETUP = "vendor/FlashMLA/setup.py"
ARCHS_OLD = 'CUDA_ARCHS = ["sm_100a", "sm_103a"]'
ARCHS_NEW = 'CUDA_ARCHS = ["sm_100a"]'
BUILD_APP = "freesolo-deepinfer-image-build"


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"Unrecognized pinned build source: {path}")
    path.write_text(source.replace(old, new))


def compile_runtime(root: str, revision: str, python: str) -> None:
    """Fetch, cross-compile and install DeepInfer with its vendored operators."""
    root = Path(root)
    subprocess.run(["git", "clone", UPSTREAM, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", revision], check=True)
    subprocess.run(["git", "-C", str(root), "submodule", "update", "--init"], check=True)
    for name in PLATFORM_FILES:
        _replace_once(root / name, PLATFORM_OLD, PLATFORM_NEW)
    _replace_once(root / FLASH_MLA_SETUP, ARCHS_OLD, ARCHS_NEW)
    subprocess.run(["/usr/local/bin/python", "-m", "venv", VENV], check=True)
    subprocess.run([python, "-m", "pip", "install", PIP], check=True)
    subprocess.run(["bash", "./build.sh"], cwd=root, check=True)
    subprocess.run([python, "-m", "pip", "install", *HTTP_DEPENDENCIES], check=True)


def runtime_image() -> modal.Image:
    from flash.serving.deepinfer.contract import config

    if f"{VENV}/bin/python" != config.RUNTIME_PYTHON:
        raise RuntimeError("DeepInfer runtime Python must live in the build virtualenv")
    pinned = {
        "root": config.RUNTIME_ROOT,
        "revision": config.RUNTIME_REVISION,
        "python": config.RUNTIME_PYTHON,
    }
    return (
        modal.Image.from_registry(CUDA_BASE, add_python="3.12")
        # The qualified image's apt history: toolchain, NUMA/RDMA headers for DeepEP and
        # elfutils for its JIT exception backtraces.
        .apt_install(*APT_PACKAGES)
        .env(BUILD_ENV)
        .run_function(compile_runtime, kwargs=pinned, cpu=32, memory=131072, timeout=6 * 3600)
        .add_local_python_source("flash", copy=True)
        # Verifies the source revision and bakes in the hash-checked runtime patches.
        .run_commands(
            f"{config.RUNTIME_PYTHON} -m flash.serving.deepinfer.native.image",
            env={"PYTHONPATH": f"/root:{config.RUNTIME_ROOT}"},
        )
    )


def main() -> None:
    with modal.enable_output():
        image = runtime_image().build(modal.App.lookup(BUILD_APP, create_if_missing=True))
    print(image.object_id)


if __name__ == "__main__":
    main()
