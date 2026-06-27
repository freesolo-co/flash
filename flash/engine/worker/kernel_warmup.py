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
import json
import os
import time

# Default bake dir. Mirrors the Dockerfile's /opt/flash/kernelcache; the saved mega-cache file lands
# directly under it so _load_kernel_cache_if_present finds it. Keep this name in lockstep with
# engine.worker._KERNEL_CACHE_DIR / _KERNEL_CACHE_FILE.
DEFAULT_CACHE_DIR = "/opt/flash/kernelcache"
MEGA_CACHE_FILENAME = "mega_cache.bin"
MEGA_CACHE_META_FILENAME = "mega_cache.json"


def _log(msg: str) -> None:
    """Single progress channel so the GPU builder's logs show each warm step."""
    print(f"[kernel-warmup] {msg}", flush=True)


def _point_backends_at(cache_dir: str) -> None:
    """Point Triton + TorchInductor at ``cache_dir`` so anything compiled below is content-addressed
    under the same tree the worker reads (matches the Dockerfile ENV)."""
    os.makedirs(os.path.join(cache_dir, "triton"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "inductor"), exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_dir, "inductor")


def _torch_sm(torch) -> str:
    cap = torch.cuda.get_device_capability(0)
    return f"sm{cap[0]}{cap[1]}"


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
        out.sum().backward()  # exercise the bwd kernel too
        torch.cuda.synchronize()
        _log("flash-attn (FA2) fwd/bwd compiled")
        warmed = True
    except Exception as e:
        _log(f"flash-attn (FA2) warm skipped: {e}")
    # FA3 (flash_attn_interface) is HOPPER-ONLY: production selects attn_implementation="flash_attention_3"
    # only on sm90. Launching it on any other arch runs a Hopper kernel that has no image there ->
    # a "no kernel image" CUDA error that POISONS the context (a caught Python exception does NOT clear
    # it), which then breaks save_cache_artifacts so the bake produces no mega_cache.bin. Only warm FA3
    # on sm90; off-Hopper it is irrelevant anyway (the worker never selects it there).
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
        # representative catalog vocab width (qwen3.5/3.6 lm_head ~248k); triton/liger specialize the
        # fused-ce chunking to the vocab shape, so warm the production width, not a toy 4096.
        vocab = 248_320
        hidden = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(vocab, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, vocab, (64,), device="cuda")
        loss_fn = loss_cls()
        # upstream signature is forward(self, lin_weight, _input, target): weight first, then hidden.
        # call the known-good form first so we never launch a mismatched shape (which would trigger a
        # cuda illegal access and poison the context before a later attempt can run).
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
                break  # fall through to the model-layer warm below; don't exit the function
            except Exception:
                continue
        else:
            raise RuntimeError("fused-linear Liger calls were not accepted")
    except Exception as e:
        _log(f"liger fused-linear warm skipped: {e}")
    try:
        # model-layer liger kernels (rmsnorm + rope) that use_liger_kernel patches in besides the
        # loss; these still jit on the first real forward/backward if only the ce loss was warmed.
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


def warm_fla_gdn(
    torch, *, head_dim: int = 64, num_heads: int = 4, seq_len: int = 256, varlen: bool = True
) -> bool:
    """Compile flash-linear-attention's Gated-DeltaNet chunk kernels (Qwen3.5/3.6 hybrid path).

    ``head_dim`` is the GatedDeltaNet linear-attention head dim (the model's ``linear_key_head_dim``);
    it is the feature dim the fused ``l2norm`` kernel normalizes over. ``seq_len`` x ``num_heads`` is the
    ROW count that — together with head_dim — drives that kernel's Triton autotune key: at the pinned
    fla SHA ``l2norm_{fwd,bwd}_kernel`` is ``@fla_cache_autotune(..., key=["D", "NB"])`` where
    ``D = head_dim`` and ``NB = cdiv(B*seq*num_heads, 65536)`` — a COARSE bucket of the row count, NOT
    head-dim-only. So one warm covers only one NB bucket; the runtime caller (``prewarm_gdn_autotune``)
    drives ``seq_len`` across every NB bucket a run can visit so the live training backward gets a cache
    hit instead of benchmarking 25 configs on the memory-tight colocate step. The defaults match a
    representative Qwen3.5 head for the build-time bake. Pass ``varlen=False`` to skip the (cu_seqlens)
    packing warm — GRPO is unpacked, so the bucket sweep only needs the plain path."""
    try:
        # mirror the worker boot: on Hopper (sm90) the production path runs this first so fla's GDN
        # chunk_bwd uses the CORRECT tilelang backend (fla #640: the Triton path miscomputes/raises).
        # off-Hopper this is a no-op. bake the same backend the runtime will actually select.
        try:
            from flash.engine.worker.perf import _ensure_fla_fastpath_on_hopper

            _ensure_fla_fastpath_on_hopper()
        except Exception as e:
            _log(f"hopper fla fast-path setup skipped: {e}")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        b, h, t, d = 1, int(num_heads or 4), int(seq_len or 256), int(head_dim or 64)
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
        # Free the warm tensors so a multi-bucket sweep (prewarm_gdn_autotune drives this across NB
        # buckets) doesn't pile resident allocations ahead of the real model/engine load — the in-memory
        # autotune cache (what must persist) is independent of these tensors.
        del q, k, v, g, beta, out
        torch.cuda.empty_cache()
        _log(f"flash-linear-attention GDN fwd/bwd compiled (seq_len={t}, heads={h}, head_dim={d})")
        # also warm the varlen path SFT token-packing (#218) uses: BlockDiagonalCollator(emit_varlen=True)
        # feeds cu_seq_lens into the fla DeltaNet so the recurrence resets per packed example, which
        # compiles different chunk kernels than the equal-length call above. shape it like one packed
        # block (batch flattened to 1) with two unequal segments. Skipped on the GRPO NB-bucket sweep
        # (varlen=False): GRPO rollouts are unpacked, so only the plain path is hit at train time.
        if varlen:
            try:
                tv = 128 + 96  # two example lengths packed into one sequence
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
                del qv, kv, vv, gv, betav, cu, outv
                torch.cuda.empty_cache()
                _log("flash-linear-attention GDN varlen (cu_seqlens) fwd/bwd compiled")
            except Exception as e:
                _log(f"fla GDN varlen warm skipped: {e}")
        return True
    except Exception as e:
        _log(f"fla GDN warm skipped: {e}")
        return False


# fla's l2norm autotune key is ["D", "NB"] with NB = cdiv(rows, _L2NORM_NB_GRANULARITY) where rows is
# the FLATTENED token count (B*seq*num_key_heads). So the key buckets by row count, not head-dim-only:
# one warm shape covers only ONE NB bucket. Keep this in lockstep with the pinned fla SHA in
# Dockerfile.worker (fla/modules/l2norm.py: NB = triton.cdiv(T, 2048 * 32)).
_L2NORM_NB_GRANULARITY = 2048 * 32  # 65536
# Cap the bucket sweep so a pathological (huge group x context x heads) config can't warm dozens of
# shapes; the common GRPO range is NB 1-6. A run that legitimately exceeds this re-benchmarks only the
# uncovered tail buckets (logged), not every step.
_MAX_NB_BUCKETS = 6
# Above this head dim fla dispatches a NON-autotuned l2norm kernel (no benchmark -> no OOM to prevent),
# so there is nothing to warm. Mirrors fla/modules/l2norm.py's BLOCK_N<=512 fast-path gate.
_L2NORM_AUTOTUNE_MAX_D = 512


def _gdn_dims_from_config(model_id: str):
    """``(head_dim, num_key_heads)`` of the model's GatedDeltaNet block, read from its HF config WITHOUT
    loading weights — or ``None`` when the model has no linear-attention layer (a plain Llama / MiniCPM
    checkpoint has no l2norm kernel to warm). Mirrors ``packing.py``'s probe: the GDN dims live on
    ``linear_key_head_dim`` / ``linear_num_key_heads`` (under ``text_config`` for VL checkpoints)."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg = getattr(cfg, "text_config", None) or cfg
        d = getattr(cfg, "linear_key_head_dim", None)
        if not d:
            return None  # not a GatedDeltaNet hybrid -> no l2norm kernel to warm
        h = getattr(cfg, "linear_num_key_heads", None) or 4
        return int(d), int(h)
    except Exception as e:
        _log(f"GDN dim probe failed for {model_id} ({e}); skipping autotune prewarm")
        return None


def gdn_autotune_nb_buckets(num_key_heads: int, max_rows: int) -> list[int]:
    """The l2norm sequence lengths to warm so the autotune cache covers every NB bucket a GRPO run can
    visit. ``NB = cdiv(rows, 65536)`` and ``rows = B*seq*num_key_heads``; for each bucket ``nb`` we pick
    the SMALLEST seq whose row count lands in that bucket (cheapest warm that still trains the right
    ``[D, NB]`` key). Capped at ``_MAX_NB_BUCKETS``."""
    h = max(1, int(num_key_heads))
    nb_max = max(1, -(-int(max_rows) // _L2NORM_NB_GRANULARITY))  # ceil
    nb_max = min(nb_max, _MAX_NB_BUCKETS)
    seqs = []
    for nb in range(1, nb_max + 1):
        # smallest seq with rows just inside bucket nb: rows just above (nb-1)*granularity
        seq = max(256, (nb - 1) * _L2NORM_NB_GRANULARITY // h + 64)
        seqs.append(int(seq))
    return seqs


def prewarm_gdn_autotune(model_id: str, *, max_length: int = 0, group_size: int = 8) -> bool:
    """Run the GatedDeltaNet ``l2norm`` (and chunk) fwd/bwd while VRAM is free, across every NB autotune
    bucket the run can reach — so TRL's GRPO training backward gets a cache HIT instead of benchmarking
    25 Triton configs on the memory-tight colocate step.

    WHY: ``l2norm_{fwd,bwd}_kernel`` is ``@fla_cache_autotune(..., key=["D","NB"])``; the worker's pinned
    fla defaults ``FLA_CACHE_MODE`` to DISABLED, so a cold ``[D,NB]`` key falls through to Triton's
    benchmark (``do_bench`` over 25 configs). On a colocated GRPO step the vLLM rollout engine stays
    RESIDENT through the backward (sleep mode is off whenever the run fits resident — and the sleep/wake
    cycle stalls large-model GRPO, so we can't just force it), leaving ~0 headroom, and the benchmark's
    transient OOMs (prod run flash-1782588906: 9B GRPO on an 80 GB A100 PCIe died in ``l2norm_bwd``'s
    ``do_bench``, NOT in steady-state training). Benchmarking HERE — before the engine/optimizer load —
    has the whole card free, and the in-process autotune cache then serves every training step.

    NB depends on the ROW count (``B*seq*num_key_heads``), not just head_dim, so a single shape would
    cover only one bucket; we sweep ``seq`` across the buckets bounded by the run's
    ``group_size x max_length x num_key_heads`` rows. Best-effort: a non-GDN model, head_dim above the
    autotuned range, missing fla, or no CUDA is a silent no-op and NEVER blocks a paid run."""
    dims = _gdn_dims_from_config(model_id)
    if not dims:
        return False
    head_dim, num_key_heads = dims
    if head_dim > _L2NORM_AUTOTUNE_MAX_D:
        _log(f"head_dim={head_dim} uses fla's non-autotuned l2norm kernel; no prewarm needed")
        return False
    try:
        import torch
    except Exception:
        return False
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return False
    # Upper-bound the training l2norm rows: the per-device logp micro-batch is <= group_size, the seq is
    # <= the run's engine context (max_length). Size the bucket sweep to that worst case.
    max_rows = max(1, int(group_size)) * max(256, int(max_length or 1024)) * num_key_heads
    seqs = gdn_autotune_nb_buckets(num_key_heads, max_rows)
    _log(
        f"pre-warming GatedDeltaNet l2norm autotuner: head_dim={head_dim}, num_key_heads={num_key_heads}, "
        f"NB buckets via seq_lens={seqs} (free VRAM; avoids a train-time benchmark OOM on the colocate step)"
    )
    warmed = False
    for i, seq in enumerate(seqs):
        # varlen only on the first pass: GRPO rollouts are unpacked, so the plain path is what the
        # training backward hits — the bucket sweep just needs that one.
        warmed = warm_fla_gdn(
            torch, head_dim=head_dim, num_heads=num_key_heads, seq_len=seq, varlen=(i == 0)
        ) or warmed
    return warmed


def warm_chalk_kernels() -> bool:
    """Compile default chalk self-test kernels when freesolo-chalk is installed."""
    warmed = False
    try:
        from chalk.transformers import install_fused_lora_delta, install_qwen35_rope

        warmed = bool(install_qwen35_rope()) or warmed
        warmed = bool(install_fused_lora_delta()) or warmed
        # embedding gather is chalk's third default gap-filler (chalk_kernels._KERNELS); import it
        # separately so a name/version skew can't also drop the rope + lora-delta warms above.
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
    if arch:
        # --arch pins the compile target but the JIT/source builds key off the LIVE GPU, and the
        # saved metadata records the physical sm. a mismatch (e.g. the sm90 publish step mis-scheduled
        # onto an sm89 runner) would bake a cu128-sm90 image whose metadata says sm89 -> every H100
        # worker rejects it and cold-JITs. FAIL the bake rather than publish a mislabeled artifact.
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
