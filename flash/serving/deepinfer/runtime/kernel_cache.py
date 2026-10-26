"""Qualify immutable compiler artifacts before shortening prefill warmup."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from flash.serving.deepinfer.contract import config

MANIFEST_PATH = "/opt/deepinfer-compiled-kernel-manifest.json"
MANIFEST_SHA256 = "7cbfab3b44c3237c7ba2a5693af3249404d3093c2bc207d19c768d6511040ff5"
NATIVE_SHA256 = "62d3f2996eab97dca43f2e1070f916cb6cbde9f14c552c850f6ea6eaf98f9992"
QUALIFIED_CONFIG = {
    "MODEL_ID": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "MODEL_REVISION": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "RUNTIME_REVISION": "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9",
    "GPU_COUNT": 4,
    "CONTEXT_PARALLEL_SIZE": 2,
    "RANK_COUNT": 2,
    "MAX_MODEL_LEN": 1048576,
    "SEQUENCES_PER_RANK": 64,
    "PREFILL_TOKENS": 16384,
    "PREFILL_ANY_INTERVAL": 8,
    "PREFILL_COALESCING_SECONDS": 0.42,
    "PREFILL_COALESCING_MIN_TOKENS": 8192,
    "GPU_PREFIX_CACHE_TOKENS": 2097152,
    "GPU_PREFIX_CACHE_ENTRIES": 64,
}
# The eight-turn scheduler was qualified with exhaustive startup, correctness
# checks and a matched four-B200 replay. Other overrides below change only the
# two module import paths; their bytes were checked against the qualified source
# after those exact substitutions. Every current file still requires an exact hash.
QUALIFIED_SOURCE_OVERRIDES = {
    "flash/serving/deepinfer/native/nvlink_compat.py": "4d8e7c0067c6a5c7f48b8c6f861943128dc79c02eba9caa69af3074e380b3ceb",
    "flash/serving/deepinfer/runtime/prefill_scheduler.py": "059ba9374ccd77b45348cc1887ca7b7dbf4a703e6c7b9d0260cef6d8056aa76c",
    "flash/serving/deepinfer/native/image.py": "770732bbde9a6a6f106cc3ba821ca38e16d6a791666f5bd8cab150bbd2186257",
    "flash/serving/deepinfer/runtime/worker.py": "fd654ac63b90e33d7c38a424ac7bf4b2a4489c9455b939543abe86067b4bab29",
}
# The immutable compiler seed predates these descriptive module names.
SOURCE_PATH_RENAMES = {
    "flash/serving/deepinfer/native/local_collectives.py": "flash/serving/deepinfer/native/nvlink_compat.py",
    "flash/serving/deepinfer/runtime/prefill_coalescing.py": "flash/serving/deepinfer/runtime/prefill_scheduler.py",
}
_SOURCE_PREFIX = "flash/serving/deepinfer/"
# Deployment/transport code does not specialize GPU kernels. Runtime and native
# code must match the seed, apart from this verifier and the warmup it selects.
_WARMUP_FILES = {"runtime/kernel_cache.py", "runtime/warmup.py"}
_CACHE_ROOTS = ("/root/.dj/cache/", "/root/.tilelang/cache/", "/root/.triton/cache/")


def _kernel_source(name: str) -> bool:
    relative = name.removeprefix(_SOURCE_PREFIX)
    return relative.startswith(("runtime/", "native/")) and relative not in _WARMUP_FILES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_compiler_cache(provenance: dict, *, filesystem_root: Path = Path("/")) -> bool:
    """Fail closed to exhaustive warmup on any missing, changed or unknown input.

    Compiler artifacts are only a startup optimization. A failed qualification
    never suppresses native initialization or prevents the ordinary full warmup.
    """
    try:
        return _verify(provenance, filesystem_root)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        PackageNotFoundError,
        subprocess.SubprocessError,
    ):
        return False


def _verify(provenance: dict, root: Path) -> bool:
    manifest = json.loads((root / MANIFEST_PATH.lstrip("/")).read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != MANIFEST_SHA256:
        return False
    if any(getattr(config, key) != value for key, value in QUALIFIED_CONFIG.items()):
        return False
    if set(provenance) != set(range(config.RANK_COUNT)):
        return False
    expected = {
        "native_sha256": NATIVE_SHA256,
        "revision": config.RUNTIME_REVISION,
        "gpu": "NVIDIA B200",
        "cuda": "13.2",
        "torch": manifest["libraries"]["torch"],
        "batch_invariant": False,
        "nextn": 5,
        "zero_overhead_scheduling": True,
    }
    if any(
        proof.get(key) != value for proof in provenance.values() for key, value in expected.items()
    ):
        return False
    if any(version(name) != value for name, value in manifest["libraries"].items()):
        return False
    compiler = root / "usr/local/cuda/bin/nvcc"
    if (
        subprocess.check_output([str(compiler), "--version"], text=True, timeout=5)
        != manifest["cuda"]
    ):
        return False
    sources = {
        SOURCE_PATH_RENAMES.get(name, name): digest
        for name, digest in manifest["runtime_sources"].items()
        if _kernel_source(name)
    }
    if not set(QUALIFIED_SOURCE_OVERRIDES) <= set(sources):
        return False
    sources.update(QUALIFIED_SOURCE_OVERRIDES)
    source_root = root / "root"
    current = {
        str(path.relative_to(source_root))
        for path in (source_root / _SOURCE_PREFIX).rglob("*.py")
        if _kernel_source(str(path.relative_to(source_root)))
    }
    if not sources or current != set(sources):
        return False
    if any(_sha256(source_root / name) != digest for name, digest in sources.items()):
        return False
    for name, record in manifest["files"].items():
        if not name.startswith(_CACHE_ROOTS) or ".." in Path(name).parts:
            return False
        path = root / name.lstrip("/")
        if (
            path.is_symlink()
            or path.stat().st_size != record["size"]
            or _sha256(path) != record["sha256"]
        ):
            return False
    return bool(manifest["files"])
