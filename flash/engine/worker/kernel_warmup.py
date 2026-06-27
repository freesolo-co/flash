"""Pre-compile hot worker kernels on a GPU builder and persist a portable mega-cache.

Measured: cold compile ~124s -> warm load ~0.2s. Import-safe without torch (every heavy import is
inside a function). CLI: ``python -m flash.engine.worker.kernel_warmup --arch <sm> --out <dir>``.
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


def _log(msg: str) -> None:
    print(f"[kernel-warmup] {msg}", flush=True)


def _point_backends_at(cache_dir: str) -> None:
    os.makedirs(os.path.join(cache_dir, "triton"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "inductor"), exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, "inductor")


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
        _log(f"GPU: {torch.cuda.get_device_name(0)} ({_torch_sm(torch)}), torch {torch.__version__}")
        return torch
    except Exception as e:
        _log(f"torch unavailable ({e}); cannot warm kernels")
        return None


def warm_flash_attn(torch) -> bool:
    """Compile FlashAttention fwd + bwd (FA2 everywhere, + FA3 on Hopper) with one tiny attention."""
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
    # FA3 is Hopper-only: launching it on another arch produces a "no kernel image" CUDA error that
    # POISONS the context (not cleared by Python exception) and breaks save_cache_artifacts.
    if _torch_sm(torch) == "sm90":
        try:
            import flash_attn_interface

            q, k, v = (
                torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                for _ in range(3)
            )
            out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            torch.cuda.synchronize()
            _log("flash-attn-3 (Hopper) fwd/bwd compiled")
            warmed = True
        except Exception as e:
            _log(f"flash-attn-3 warm skipped: {e}")
    else:
        _log("flash-attn-3 warm skipped (Hopper-only; not this arch)")
    return warmed


def warm_liger_ce(torch) -> bool:
    """Compile Liger cross-entropy kernels."""
    warmed = False
    try:
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        loss_fn = LigerCrossEntropyLoss()
        logits = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, 4096, (64,), device="cuda")
        loss_fn(logits, labels).backward()
        torch.cuda.synchronize()
        _log("liger fused cross-entropy compiled")
        warmed = True
    except Exception as e:
        _log(f"liger CE warm skipped: {e}")
    try:
        candidates = (
            ("liger_kernel.ops.fused_linear_cross_entropy", "LigerFusedLinearCrossEntropyLoss"),
            (
                "liger_kernel.transformers.fused_linear_cross_entropy",
                "LigerFusedLinearCrossEntropyLoss",
            ),
        )
        loss_cls = None
        for module_name, attr in candidates:
            try:
                mod = __import__(module_name, fromlist=[attr])
                loss_cls = getattr(mod, attr)
                break
            except Exception:
                continue
        if loss_cls is None:
            raise ImportError("no fused-linear Liger loss class found")
        # warm qwen3.5/3.6 production vocab width (~248k); triton/liger chunk specialization differs from toy 4096.
        vocab = 248_320
        hidden = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(vocab, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, vocab, (64,), device="cuda")
        loss_fn = loss_cls()
        # upstream signature: forward(lin_weight, _input, target) — weight first; call known-good form
        # first to avoid mismatched-shape CUDA illegal access that would poison the context.
        attempts = (
            lambda: loss_fn(weight, hidden, labels),
            lambda: loss_fn(weight, hidden, target=labels),
            lambda: loss_fn(hidden, weight, labels),
        )
        for call in attempts:
            try:
                out = call()
                if isinstance(out, tuple):
                    out = out[0]
                out.backward()
                torch.cuda.synchronize()
                _log("liger fused-linear loss compiled")
                warmed = True
                break
            except Exception:
                continue
        else:
            raise RuntimeError("fused-linear Liger calls were not accepted")
    except Exception as e:
        _log(f"liger fused-linear warm skipped: {e}")
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb

        rms = LigerRMSNorm(hidden_size=256).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        rms(x).sum().backward()
        b, h, t, d = 1, 4, 64, 64
        q = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
        liger_rotary_pos_emb(q, k, cos, sin)
        torch.cuda.synchronize()
        _log("liger model-layer kernels (rmsnorm/rope) compiled")
        warmed = True
    except Exception as e:
        _log(f"liger model-layer warm skipped: {e}")
    return warmed


def warm_fla_gdn(torch) -> bool:
    """Compile flash-linear-attention's Gated-DeltaNet chunk kernels (Qwen3.5/3.6 hybrid path)."""
    try:
        # fla #640: GDN chunk_bwd needs tilelang on Hopper; Triton path miscomputes/raises there.
        try:
            from flash.engine.worker.perf import _ensure_fla_fastpath_on_hopper

            _ensure_fla_fastpath_on_hopper()
        except Exception as e:
            _log(f"hopper fla fast-path setup skipped: {e}")
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


def warm_chalk_kernels() -> bool:
    """Compile default chalk self-test kernels when freesolo-chalk is installed."""
    warmed = False
    try:
        from chalk.transformers import install_fused_lora_delta, install_qwen35_rope

        warmed = bool(install_qwen35_rope()) or warmed
        warmed = bool(install_fused_lora_delta()) or warmed
        try:
            from chalk.transformers import install_fused_embedding

            warmed = bool(install_fused_embedding()) or warmed
        except Exception as e:
            _log(f"chalk fused-embedding warm skipped: {e}")
        _log(f"chalk default kernel installers ran (warmed={warmed})")
    except Exception as e:
        _log(f"chalk warm skipped: {e}")
    return warmed


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
            warm_liger_ce(torch),
            warm_fla_gdn(torch),
            warm_chalk_kernels(),
            warm_torch_compile(torch),
        ]
    )
    _log(f"{warmed}/5 kernel groups compiled in {time.time() - t0:.1f}s; saving mega-cache")
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
