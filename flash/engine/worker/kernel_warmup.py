"""Pre-compile the worker's hot kernels on a real GPU and persist a portable mega-cache.

Run this ON A GPU BUILDER (an image-build runner that actually has the target arch's GPU) to kill
the ~10-15 min first-use JIT that #194 reintroduced on a cold worker. It warms the kernels the
trainer hits early — FlashAttention fwd/bwd, the Liger fused cross-entropy, flash-linear-attention's
Gated-DeltaNet (Qwen3.5/3.6 hybrid), and a representative ``torch.compile`` — then calls
``torch.compiler.save_cache_artifacts()`` to write ONE portable mega-cache blob into the cache dir.
``flash.engine.worker._load_kernel_cache_if_present`` loads it back at worker boot
(``torch.compiler.load_cache_artifacts``); the Dockerfile bakes the produced ``build/kernel_cache/``
into the image when built with ``--build-arg BUILD_KERNEL_CACHE=true``.

Measured: cold compile ~124s -> warm load ~0.2s (537x).

This module is import-safe WITHOUT torch installed (it must ``py_compile`` on the CPU-only CI image
that builds the worker): every heavy import lives INSIDE a function. Everything is best-effort —
each warm step is independently guarded so a missing/uncompilable kernel never aborts the bake; we
save whatever did compile. CLI: ``python -m flash.engine.worker.kernel_warmup --arch <sm> --out <dir>``.
"""

from __future__ import annotations

import argparse
import os
import time

# Default bake dir. Mirrors the Dockerfile's /opt/flash/kernelcache; the saved mega-cache file lands
# directly under it so _load_kernel_cache_if_present finds it. Keep this name in lockstep with
# engine.worker._KERNEL_CACHE_DIR / _KERNEL_CACHE_FILE.
DEFAULT_CACHE_DIR = "/opt/flash/kernelcache"
MEGA_CACHE_FILENAME = "mega_cache.bin"


def _log(msg: str) -> None:
    """Single progress channel so the GPU builder's logs show each warm step."""
    print(f"[kernel-warmup] {msg}", flush=True)


def _point_backends_at(cache_dir: str) -> None:
    """Point Triton + TorchInductor at ``cache_dir`` so anything compiled below is content-addressed
    under the same tree the worker reads (matches the Dockerfile ENV)."""
    os.makedirs(os.path.join(cache_dir, "triton"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "inductor"), exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_dir, "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(cache_dir, "inductor"))


def _require_gpu():
    """Return the torch module if a CUDA GPU is live, else None (with a clear log).

    The warm steps are only meaningful on a real GPU of the target arch — kernels are
    content-addressed by arch + toolchain, so a CPU run would bake nothing usable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            _log("no CUDA device visible — kernel warmup must run on a GPU builder; nothing baked")
            return None
        cap = torch.cuda.get_device_capability(0)
        _log(f"GPU: {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]}), torch {torch.__version__}")
        return torch
    except Exception as e:
        _log(f"torch unavailable ({e}); cannot warm kernels")
        return None


def warm_flash_attn(torch) -> bool:
    """Compile FlashAttention fwd + bwd by running one tiny attention with grad."""
    try:
        from flash_attn import flash_attn_func

        q, k, v = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()  # exercise the bwd kernel too
        torch.cuda.synchronize()
        _log("flash-attn fwd/bwd compiled")
        return True
    except Exception as e:
        _log(f"flash-attn warm skipped: {e}")
        return False


def warm_liger_ce(torch) -> bool:
    """Compile the Liger fused linear cross-entropy (the big large-vocab win)."""
    try:
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        loss_fn = LigerCrossEntropyLoss()
        logits = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, 4096, (64,), device="cuda")
        loss_fn(logits, labels).backward()
        torch.cuda.synchronize()
        _log("liger fused cross-entropy compiled")
        return True
    except Exception as e:
        _log(f"liger CE warm skipped: {e}")
        return False


def warm_fla_gdn(torch) -> bool:
    """Compile flash-linear-attention's Gated-DeltaNet chunk kernels (Qwen3.5/3.6 hybrid path)."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        b, h, t, d = 1, 4, 256, 64
        q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(b, t, h, device="cuda", dtype=torch.float32)
        beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
        chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        _log("flash-linear-attention GDN compiled")
        return True
    except Exception as e:
        _log(f"fla GDN warm skipped: {e}")
        return False


def warm_torch_compile(torch) -> bool:
    """Trigger a representative ``torch.compile`` so TorchInductor populates its cache."""
    try:

        @torch.compile
        def _fused(a, b):
            return torch.nn.functional.gelu(a @ b)

        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        _fused(a, b)
        torch.cuda.synchronize()
        _log("torch.compile (TorchInductor) warmed")
        return True
    except Exception as e:
        _log(f"torch.compile warm skipped: {e}")
        return False


def save_mega_cache(torch, out_dir: str) -> bool:
    """Persist everything compiled this session into one portable blob via
    ``torch.compiler.save_cache_artifacts()`` so the worker can ``load_cache_artifacts`` it at boot.
    """
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if not artifacts:
            _log("save_cache_artifacts returned nothing — no compiled kernels to persist")
            return False
        # save_cache_artifacts returns (bytes, meta); persist the bytes payload.
        blob = artifacts[0] if isinstance(artifacts, tuple) else artifacts
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, MEGA_CACHE_FILENAME)
        with open(path, "wb") as f:
            f.write(blob)
        _log(f"mega-cache saved: {path} ({len(blob)} bytes)")
        return True
    except Exception as e:
        _log(f"save_cache_artifacts failed: {e}")
        return False


def warmup(out_dir: str = DEFAULT_CACHE_DIR, arch: str | None = None) -> int:
    """Run every warm step then persist the mega-cache. Returns a process exit code.

    Best-effort end to end: individual kernel failures are tolerated (we bake what compiled); only a
    total absence of GPU/torch or a failed save is a non-zero exit so the builder surfaces it.
    """
    t0 = time.time()
    _point_backends_at(out_dir)
    if arch:
        # let the caller pin the compile target for source builds that read it (e.g. flash-attn)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
        _log(f"target arch pinned: TORCH_CUDA_ARCH_LIST={arch}")
    torch = _require_gpu()
    if torch is None:
        return 1
    warmed = sum(
        [
            warm_flash_attn(torch),
            warm_liger_ce(torch),
            warm_fla_gdn(torch),
            warm_torch_compile(torch),
        ]
    )
    _log(f"{warmed}/4 kernel groups compiled in {time.time() - t0:.1f}s; saving mega-cache")
    saved = save_mega_cache(torch, out_dir)
    _log(f"done in {time.time() - t0:.1f}s (saved={saved})")
    return 0 if saved else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="pre-compile hot worker kernels and bake a mega-cache")
    ap.add_argument(
        "--arch",
        default=None,
        help="target TORCH_CUDA_ARCH_LIST (e.g. '9.0' for Hopper); default: probe the live GPU",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_CACHE_DIR,
        help=f"cache output dir (default: {DEFAULT_CACHE_DIR}); the bake produces <out>/{MEGA_CACHE_FILENAME}",
    )
    args = ap.parse_args()
    return warmup(out_dir=args.out, arch=args.arch)


if __name__ == "__main__":
    raise SystemExit(main())
