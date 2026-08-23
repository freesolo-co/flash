"""Pre-compile hot worker kernels on a GPU builder and persist a portable mega-cache.

Measured: cold compile ~124s -> warm load ~0.2s. Import-safe without torch (every heavy import is
inside a function). CLI: ``python -m flash.engine.worker.runtime.kernel_warmup --arch <sm> --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

DEFAULT_CACHE_DIR = "/opt/flash/kernelcache"
MEGA_CACHE_FILENAME = "mega_cache.bin"
MEGA_CACHE_META_FILENAME = "mega_cache.json"

# every env var that addresses the baked kernel cache, mapped to its subdirectory of the cache root.
# Dockerfile.worker sets these image-wide to the same layout; this module re-forces them so a stale
# image or a prior run cannot defeat the redirect. it is also the allowlist the verl child inherits
# (see sft_train._CHILD_ENV_EXACT) -- the child is the interpreter that actually trains, so a cache
# only the parent can see is no cache at all. one definition, so a var added here cannot be added
# without the child getting it too.
KERNEL_CACHE_ENV_SUBDIRS = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    # flashinfer downloads b200 attention cubins and builds a host wrapper on first use. redirect
    # both to distinct persistent-cache subdirs so cold workers do not depend on nvidia artifactory
    # or pay the first-step build repeatedly. it reads FLASHINFER_CUBIN_DIR for cubins and
    # FLASHINFER_WORKSPACE_BASE for the 0.6.x jit wrapper; FLASHINFER_CACHE_DIR is
    # forward-compatible only, and is set anyway so a future version cannot silently miss the cache.
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "FLASHINFER_CACHE_DIR": "flashinfer",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
}


def _log(msg: str) -> None:
    print(f"[kernel-warmup] {msg}", flush=True)


def _point_backends_at(cache_dir: str) -> None:
    for var, subdir in KERNEL_CACHE_ENV_SUBDIRS.items():
        path = os.path.join(cache_dir, subdir)
        os.makedirs(path, exist_ok=True)
        os.environ[var] = path


def _torch_sm(torch) -> str:
    cap = torch.cuda.get_device_capability(0)
    return f"sm{cap[0]}{cap[1]}"


def _current_cuda_sm(torch) -> str | None:
    """Guarded arch probe for boot-time cache load; returns None if unavailable."""
    try:
        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability(0)
        return f"sm{cap[0]}{cap[1]}"
    except Exception:
        return None


def _require_gpu():
    try:
        import torch

        if not torch.cuda.is_available():
            _log("no CUDA device visible — kernel warmup must run on a GPU builder; nothing baked")
            return None
        _log(
            f"GPU: {torch.cuda.get_device_name(0)} ({_torch_sm(torch)}), torch {torch.__version__}"
        )
        return torch
    except Exception as e:
        _log(f"torch unavailable ({e}); cannot warm kernels")
        return None


def warm_flash_attn(torch) -> bool:
    """Compile FlashAttention fwd + bwd with one tiny attention.

    FA2 only. The FA3 (``flash_attn_interface``) warm was removed with the wheel: it was installed
    into this interpreter while training runs in /opt/verl-venv, and nothing ever set
    ``attn_implementation``, so verl's "flash_attention_2" default meant FA3 never ran in a training
    forward on any arch. Warming a kernel the trainer cannot reach only bloated the mega-cache.
    """
    warmed = False
    try:
        from flash_attn import flash_attn_func

        q, k, v = (
            torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()
        torch.cuda.synchronize()
        _log("flash-attn (FA2) fwd/bwd compiled")
        warmed = True
    except Exception as e:
        _log(f"flash-attn (FA2) warm skipped: {e}")
    return warmed


def warm_fla_gdn(torch) -> bool:
    """Compile flash-linear-attention's Gated-DeltaNet chunk kernels (Qwen3.5/3.6 hybrid path)."""
    try:
        # sm100: tilelang's GDN bwd miscomputes grads — warm the Triton path the worker will run.
        try:
            from flash.engine.worker.perf import _force_fla_triton_gdn_on_sm100

            _force_fla_triton_gdn_on_sm100()
        except Exception as e:
            _log(f"sm100 fla tilelang opt-out skipped: {e}")
        # fla #640: GDN chunk_bwd needs tilelang on Hopper; Triton path miscomputes/raises there.
        try:
            from flash.engine.worker.perf import _ensure_fla_fastpath_on_hopper

            _ensure_fla_fastpath_on_hopper()
        except Exception as e:
            _log(f"hopper fla fast-path setup skipped: {e}")
        # fla #913: on Blackwell, warm the SAME restricted autotune space the worker boot will
        # run (grad-correct configs only) so the baked cache holds the config that actually runs.
        try:
            from flash.engine.worker.perf import _restrict_fla_gdn_autotune_on_blackwell

            _restrict_fla_gdn_autotune_on_blackwell()
        except Exception as e:
            _log(f"blackwell fla autotune restriction skipped: {e}")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        b, h, t, d = 1, 4, 256, 64
        q = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn(b, t, h, device="cuda", dtype=torch.float32, requires_grad=True)
        beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        torch.cuda.synchronize()
        _log("flash-linear-attention GDN fwd/bwd compiled")
        # also warm varlen path (cu_seqlens) used by SFT token-packing; compiles different chunk kernels.
        try:
            tv = 128 + 96
            qv = torch.randn(1, tv, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            kv = torch.randn(1, tv, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            vv = torch.randn(1, tv, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            gv = torch.randn(1, tv, h, device="cuda", dtype=torch.float32, requires_grad=True)
            betav = torch.rand(1, tv, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            cu = torch.tensor([0, 128, tv], device="cuda", dtype=torch.int32)
            outv = chunk_gated_delta_rule(
                qv, kv, vv, gv, betav, use_qk_l2norm_in_kernel=True, cu_seqlens=cu
            )
            if isinstance(outv, tuple):
                outv = outv[0]
            outv.sum().backward()
            torch.cuda.synchronize()
            _log("flash-linear-attention GDN varlen (cu_seqlens) fwd/bwd compiled")
        except Exception as e:
            _log(f"fla GDN varlen warm skipped: {e}")
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
    """Persist compiled kernels into a portable blob via ``torch.compiler.save_cache_artifacts()``."""
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if not artifacts:
            _log("save_cache_artifacts returned nothing — no compiled kernels to persist")
            return False
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


def save_cache_metadata(torch, out_dir: str, *, requested_arch: str | None, warmed: int) -> bool:
    try:
        meta = {
            "sm": _torch_sm(torch),
            "requested_arch": requested_arch,
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "device": torch.cuda.get_device_name(0),
            "warmed_groups": int(warmed),
            "created_at": int(time.time()),
        }
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, MEGA_CACHE_META_FILENAME)
        with open(path, "w") as f:
            json.dump(meta, f, sort_keys=True)
        _log(f"cache metadata saved: {path} ({meta['sm']})")
        return True
    except Exception as e:
        _log(f"cache metadata save failed: {e}")
        return False


def load_mega_cache() -> bool:
    """Load a baked mega-cache at worker boot; no-op (returns False) if absent or wrong arch."""
    cache_file = os.path.join(DEFAULT_CACHE_DIR, MEGA_CACHE_FILENAME)
    meta_file = os.path.join(DEFAULT_CACHE_DIR, MEGA_CACHE_META_FILENAME)

    def _reject(reason: str) -> bool:
        # Repoint triton/inductor off baked trees so JIT compiles fresh; wrong-arch entries would collide.
        print(f"[kernel-cache] {reason} -> first-run JIT fallback")
        scratch = os.path.join(tempfile.gettempdir(), "flash-kernelcache-jit")
        for sub, var in (("triton", "TRITON_CACHE_DIR"), ("inductor", "TORCHINDUCTOR_CACHE_DIR")):
            d = os.path.join(scratch, sub)
            os.makedirs(d, exist_ok=True)
            os.environ[var] = d
        return False

    if not os.path.isfile(cache_file):
        print(f"[kernel-cache] no baked cache at {cache_file} -> first-run JIT (expected default)")
        return False
    try:
        import torch

        current_sm = _current_cuda_sm(torch)
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except FileNotFoundError:
            return _reject("baked cache has no metadata")
        except Exception as e:
            return _reject(f"metadata unreadable ({e})")
        cached_sm = str(meta.get("sm") or "")
        if not current_sm:
            return _reject("worker GPU arch undetermined")
        if cached_sm != current_sm:
            return _reject(
                f"baked cache arch {cached_sm or 'unknown'} does not match worker arch {current_sm}"
            )
        with open(cache_file, "rb") as f:
            blob = f.read()
        torch.compiler.load_cache_artifacts(blob)
        print(
            f"[kernel-cache] loaded baked mega-cache for {cached_sm or 'unknown'} "
            f"({len(blob)} bytes) -> skipping first-run JIT"
        )
        return True
    except Exception as e:
        return _reject(f"load skipped ({e})")


def warmup(out_dir: str = DEFAULT_CACHE_DIR, arch: str | None = None) -> int:
    """Run every warm step then persist the mega-cache. Returns a process exit code."""
    t0 = time.time()
    _point_backends_at(out_dir)
    if arch:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
        _log(f"target arch pinned: TORCH_CUDA_ARCH_LIST={arch}")
    torch = _require_gpu()
    if torch is None:
        return 1
    if arch:
        # Fail if --arch doesn't match the live GPU: a mislabeled cache (e.g. sm90 image with sm89
        # metadata) causes every matching worker to reject it and cold-JIT.
        want_sm = "sm" + arch.replace(".", "")
        live_sm = _torch_sm(torch)
        if want_sm != live_sm:
            _log(
                f"ERROR: --arch {arch} ({want_sm}) does not match live GPU {live_sm}; refusing to "
                f"bake a mislabeled cache (a {want_sm} image would carry {live_sm} metadata). "
                "re-run on a matching GPU."
            )
            return 1
    warmed = sum(
        [
            warm_flash_attn(torch),
            warm_fla_gdn(torch),
            warm_torch_compile(torch),
        ]
    )
    _log(f"{warmed}/3 kernel groups compiled in {time.time() - t0:.1f}s; saving mega-cache")
    saved = save_mega_cache(torch, out_dir)
    meta_saved = save_cache_metadata(torch, out_dir, requested_arch=arch, warmed=warmed)
    _log(f"done in {time.time() - t0:.1f}s (saved={saved})")
    return 0 if saved and meta_saved else 1


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
