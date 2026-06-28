"""FlashInfer cubin/cache dir redirect (CPU-only, no GPU):

``_point_backends_at`` must ALSO point FLASHINFER_CUBIN_DIR / FLASHINFER_CACHE_DIR at DISTINCT subdirs
of the persistent kernel-cache tree, so a cold B200 worker serves FlashInfer's trtllm-gen MoE cubins
from the bake/network-volume instead of re-fetching them from NVIDIA Artifactory on every cold start.
"""

import os

from flash.engine.worker import kernel_warmup


def test_point_backends_sets_flashinfer_dirs_under_tmp(monkeypatch, tmp_path):
    # isolate os.environ so the assertions can't pass on leftover values
    for var in ("FLASHINFER_CUBIN_DIR", "FLASHINFER_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"):
        monkeypatch.delenv(var, raising=False)

    kernel_warmup._point_backends_at(str(tmp_path))

    cubin = os.environ["FLASHINFER_CUBIN_DIR"]
    cache = os.environ["FLASHINFER_CACHE_DIR"]
    workspace = os.environ["FLASHINFER_WORKSPACE_BASE"]

    # all live UNDER the cache tree we passed in
    assert cubin.startswith(str(tmp_path))
    assert cache.startswith(str(tmp_path))
    assert workspace.startswith(str(tmp_path))
    # ... and were created as real directories
    assert os.path.isdir(cubin)
    assert os.path.isdir(cache)
    # FLASHINFER_WORKSPACE_BASE is what flashinfer 0.6.x actually derives its cache from, so it must
    # equal the cache dir (else the JIT wrapper lands elsewhere and the warmed cache is missed).
    assert workspace == cache
    # DISTINCT from each other and from the triton/inductor trees (no collision)
    assert cubin != cache
    triton = os.environ["TRITON_CACHE_DIR"]
    inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    assert cubin not in (triton, inductor)
    assert cache not in (triton, inductor)


def test_point_backends_overrides_existing_flashinfer_env(monkeypatch, tmp_path):
    # a stale value from a previous run / image ENV must be overridden, not left in place — including
    # FLASHINFER_WORKSPACE_BASE, which flashinfer reads in preference to FLASHINFER_CACHE_DIR (a stale
    # base would silently defeat the redirect, the Codex Mr3Je class of bug).
    monkeypatch.setenv("FLASHINFER_CUBIN_DIR", "/old/flashinfer_cubin")
    monkeypatch.setenv("FLASHINFER_CACHE_DIR", "/old/flashinfer")
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/old/flashinfer")

    kernel_warmup._point_backends_at(str(tmp_path))

    assert os.environ["FLASHINFER_CUBIN_DIR"] == str(tmp_path / "flashinfer_cubin")
    assert os.environ["FLASHINFER_CACHE_DIR"] == str(tmp_path / "flashinfer")
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "flashinfer")
