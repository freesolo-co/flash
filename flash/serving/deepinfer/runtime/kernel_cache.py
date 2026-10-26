"""Verify the prebuilt GPU kernel cache before selecting shorter startup warmup."""

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
# Expected values stay independent of live config so configuration drift is detected.
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
# Reviewed source changes relative to the immutable seed: the qualified scheduler,
# naming-only edits, and the single-host transport patch. Qualification history
# lives in app/README.md; every accepted source still requires its exact hash.
QUALIFIED_SOURCE_OVERRIDES = {
    "flash/serving/deepinfer/native/transport.py": "9000b752404f0bf1136988dabb398a9df4598110c61038b178964c36ad6fc868",
    "flash/serving/deepinfer/runtime/prefill_scheduler.py": "391ad5a235b5667e87a59cfbec3dc80117a044ab2640e4fd4934e7500fd4eb8d",
    "flash/serving/deepinfer/native/image.py": "6ee502e5c530ff27fbf76bf19b8717c3c3db9f48bc93c689e1de1639f9e09282",
    "flash/serving/deepinfer/runtime/worker.py": "51ee532c2a39cd468436f6352fc029f3dac78beda3c12da62705e77e349088c3",
}
# The immutable compiler seed predates these descriptive module names.
SOURCE_PATH_RENAMES = {
    "flash/serving/deepinfer/native/local_collectives.py": "flash/serving/deepinfer/native/transport.py",
    "flash/serving/deepinfer/runtime/prefill_coalescing.py": "flash/serving/deepinfer/runtime/prefill_scheduler.py",
}
_SOURCE_PREFIX = "flash/serving/deepinfer/"
# Deployment/transport code does not specialize GPU kernels. Runtime and native
# code must match the seed, apart from this verifier and the warmup it selects.
_WARMUP_FILES = {"runtime/kernel_cache.py", "runtime/warmup.py"}
_CACHE_ROOTS = ("/root/.dj/cache/", "/root/.tilelang/cache/", "/root/.triton/cache/")


def verified_compiler_cache(provenance: dict, *, filesystem_root: Path = Path("/")) -> bool:
    """Select shorter warmup only when every part of the qualified seed matches.

    Missing, modified or unreadable inputs return False; the caller then uses
    exhaustive warmup. This check never skips native runtime initialization.
    """
    try:
        manifest = _load_manifest(filesystem_root)
        return (
            _runtime_matches(provenance, manifest)
            and _toolchain_matches(manifest, filesystem_root)
            and _sources_match(manifest, filesystem_root)
            and _artifacts_match(manifest, filesystem_root)
        )
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


def _load_manifest(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST_PATH.lstrip("/")).read_text())
    canonical = json.dumps(manifest, sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Compiler cache manifest does not match the qualified seed")
    return manifest


def _runtime_matches(provenance: dict, manifest: dict) -> bool:
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
    for proof in provenance.values():
        if any(proof.get(key) != value for key, value in expected.items()):
            return False
    return True


def _toolchain_matches(manifest: dict, root: Path) -> bool:
    if any(version(name) != value for name, value in manifest["libraries"].items()):
        return False
    compiler = root / "usr/local/cuda/bin/nvcc"
    compiler_version = subprocess.check_output([str(compiler), "--version"], text=True, timeout=5)
    return compiler_version == manifest["cuda"]


def _sources_match(manifest: dict, root: Path) -> bool:
    expected = {
        SOURCE_PATH_RENAMES.get(name, name): digest
        for name, digest in manifest["runtime_sources"].items()
        if _kernel_source(name)
    }
    # Overrides can update known files, but cannot introduce an unmanifested module.
    if not set(QUALIFIED_SOURCE_OVERRIDES) <= set(expected):
        return False
    expected.update(QUALIFIED_SOURCE_OVERRIDES)
    source_root = root / "root"
    current = {
        str(path.relative_to(source_root))
        for path in (source_root / _SOURCE_PREFIX).rglob("*.py")
        if _kernel_source(str(path.relative_to(source_root)))
    }
    if not expected or current != set(expected):
        return False
    return all(_sha256(source_root / name) == digest for name, digest in expected.items())


def _artifacts_match(manifest: dict, root: Path) -> bool:
    artifacts = manifest["files"]
    if not artifacts:
        return False
    for name, record in artifacts.items():
        if not name.startswith(_CACHE_ROOTS) or ".." in Path(name).parts:
            return False
        path = root / name.lstrip("/")
        if (
            path.is_symlink()
            or path.stat().st_size != record["size"]
            or _sha256(path) != record["sha256"]
        ):
            return False
    return True


def _kernel_source(name: str) -> bool:
    relative = name.removeprefix(_SOURCE_PREFIX)
    return relative.startswith(("runtime/", "native/")) and relative not in _WARMUP_FILES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
