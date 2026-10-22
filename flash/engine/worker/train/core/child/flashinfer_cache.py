"""Load image-baked FlashInfer libraries without replaying Ninja dependency checks."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

MODULE = "gdn_prefill_sm90"


def _identity(env, torch):
    return {
        "format": 1,
        "flashinfer": env.flashinfer_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "sm": list(torch.cuda.get_device_capability(0)),
    }


def _root():
    value = os.environ.get("FLASHINFER_CACHE_DIR")
    return Path(value) / "aot" if value else None


def activate(env):
    """Use FlashInfer's AOT loader only for an exact library/runtime/architecture match."""
    root = _root()
    if root is None or not (root / "manifest.json").is_file():
        return False
    try:
        import torch

        if json.loads((root / "manifest.json").read_text()) != _identity(env, torch):
            return False
        library = root / MODULE / f"{MODULE}.so"
        if not library.is_file() or not library.stat().st_size:
            return False
        # Do not hide an image's separately installed, broader AOT package.
        if root != env.FLASHINFER_AOT_DIR and any(env.FLASHINFER_AOT_DIR.glob("*/*.so")):
            return False
        env.FLASHINFER_AOT_DIR = root
        print("[flash-kernel-cache] validated FlashInfer GDN AOT library", flush=True)
        return True
    except (OSError, ValueError, RuntimeError, AttributeError, ImportError):
        return False


def publish(env, torch):
    """Promote a successfully loaded bake artifact into FlashInfer's portable AOT layout."""
    root = _root()
    if root is None:
        raise RuntimeError("FlashInfer cache root is not configured")
    destination = root / MODULE / f"{MODULE}.so"
    source = env.FLASHINFER_AOT_DIR / MODULE / f"{MODULE}.so"
    if not source.is_file():
        source = env.FLASHINFER_JIT_DIR / MODULE / f"{MODULE}.so"
    if not source.is_file() or not source.stat().st_size:
        raise RuntimeError("FlashInfer bake did not produce its GDN library")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    (root / "manifest.json").write_text(json.dumps(_identity(env, torch), sort_keys=True))


def activate_current():
    """Select a validated cache after device initialization, before model kernels load."""
    root = _root()
    if root is None or not (root / "manifest.json").is_file():
        return False
    import importlib

    try:
        env = importlib.import_module("flashinfer.jit.env")
    except ImportError:
        return False
    return activate(env)
