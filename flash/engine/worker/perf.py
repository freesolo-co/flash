"""Pure GPU/perf/optimizer probes for the fine-tuning worker.

These helpers take the model id / max length / capability as ARGUMENTS and read NONE of
the worker's run-scoped module globals (HF_REPO/RUN_ID/SEED/RUN_MODE/PHASE/JOB_SPEC/
ACTIVE_ENV/THINKING or the _HB_* heartbeat family), so they live here as a leaf module.
``flash.engine.worker`` re-exports them; this module must NOT import that package (no cycle).
Torch and other heavy deps are imported lazily inside the functions (CPU-importable).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time


def _attn_impl_for_capability(major: int) -> str | None:
    """Map a CUDA compute capability to the trainer ``attn_implementation``.

    Attention uses PyTorch SDPA (its flash/efficient backend is already selected automatically
    on Ampere/Ada/Hopper) — the HF Kernels-Hub FA path is disabled because the torch2.10-
    compatible ``kernels`` versions break transformers' import. So:
      sm120 (Blackwell consumer 5090/RTX Pro) -> "sdpa" (forced to the cuDNN backend at train
      time — its default SDPA can fall to the slow math kernel); all other archs -> None (let
      transformers pick SDPA, which already flash-backs on Ampere/Ada/Hopper). The big LoRA
      win comes from the Liger fused kernels, not the attention path. Pure function (no torch)
      so it's unit-testable on CPU.
    """
    if major == 12:  # Blackwell consumer: force cuDNN SDPA (avoid the math fallback)
        return "sdpa"
    return None


def _flash_attn_available() -> bool:
    """True when the ``flash_attn`` wheel is importable (our worker image builds it from source).

    Gates the packing default: TRL's ``packing_strategy='bfd'`` produces flattened/padding-free
    batches whose example boundaries are carried by ``position_ids`` and enforced ONLY by an
    attention impl that honors them (FlashAttention-2 varlen / flex_attention). Under plain SDPA,
    packed examples attend ACROSS boundaries (silent quality loss). find_spec only — no import side
    effects (and no CUDA init)."""
    try:
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def optimal_attn_impl() -> str | None:
    """Best ``attn_implementation`` for the live GPU (None = leave transformers' default)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as e:
        print("optimal_attn_impl probe failed:", e)
        return None
    impl = _attn_impl_for_capability(major)
    if impl:
        print(f"[attn] sm{major}{minor} -> attn_implementation={impl}")
    return impl



# Liger's fused linear cross-entropy is a MEMORY optimization (it never materializes the fp32
# [B,T,vocab] logits), not a fixed-batch speed win. PR #174 ledger: on a 1B model at fixed batch
# it is a measured NET LOSS on EVERY arch (min-of-3: A100 0.86x, H100 0.90x, RTX 3090 0.78x,
# RTX 4090 0.83x, RTX 5090 0.79x) — the per-step Triton overhead isn't repaid because the small
# model's logits don't dominate memory. Its value appears on LARGE models (lets a bigger batch
# fit / avoids OOM). So gate by estimated model size.
_LIGER_MIN_PARAMS = 3e9  # ~3B; 1B-class models measured net-negative -> Liger off below this


def _estimate_params(cfg) -> float:
    """Rough param count from a HF config: embeddings (+untied lm_head) + transformer blocks.
    For multimodal checkpoints (e.g. Qwen3.5-VL) the LM dims live under ``text_config`` — read it
    when the top-level dims are absent, else the gate underestimates and wrongly disables the
    memory path (GC/Liger) for the 4B/9B tiers."""
    tc = getattr(cfg, "text_config", None)
    src = cfg if getattr(cfg, "hidden_size", 0) else (tc or cfg)
    h = getattr(src, "hidden_size", 0) or 0
    v = getattr(src, "vocab_size", 0) or getattr(cfg, "vocab_size", 0) or 0
    n = getattr(src, "num_hidden_layers", 0) or 0
    tied = getattr(src, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))
    emb = v * h * (1 if tied else 2)
    blocks = n * 12 * h * h  # ~12 h^2 per transformer block (attn + MLP)
    return float(emb + blocks)


def _liger_default_for_model(model_id: str) -> bool:
    """Default Liger ON only for models large enough that fused-CE's memory win pays off
    (≥ _LIGER_MIN_PARAMS, ~3B). 1B-class models measured net-negative -> default OFF."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return _estimate_params(cfg) >= _LIGER_MIN_PARAMS
    except Exception as e:
        print("liger model-size probe failed (default off):", e)
        return False


def liger_on(default_on: bool) -> bool:
    """Whether to enable a Liger kernel path. ``default_on`` is the model-size decision (on only
    for models large enough that fused-CE's memory win pays off; 1B-class is a measured net loss).
    Even when on, require a CUDA GPU AND that ``liger_kernel`` is importable — the local
    ``flash[gpu]`` extra doesn't ship it, so blindly setting use_liger_kernel would crash a
    local GPU run. No GPU / absent -> off."""
    if not default_on:
        return False
    try:
        import importlib.util

        import torch

        return bool(
            torch.cuda.is_available() and importlib.util.find_spec("liger_kernel") is not None
        )
    except Exception:
        return False


def setup_perf_backends() -> None:
    """Universal, arch-agnostic throughput knobs — safe on every CUDA arch, no JIT/compile cost.

    - TF32 for fp32 matmuls/cuDNN (Ampere+): the residual fp32 ops in a bf16 LoRA run (some
      norms, the optimizer's fp32 master step, any fp32 GEMM) run on the TF32 tensor cores at
      ~no accuracy cost. No-op on pre-Ampere.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] TF32 matmul/cuDNN enabled")
    except Exception as e:
        print("setup_perf_backends skipped:", e)



def _remove_fla_from_disk() -> tuple[list[str], bool]:
    """Physically delete every importable ``fla`` package dir from the worker's REAL sys.path.

    Loops until ``find_spec('fla')`` is clean (removing one copy can expose another further down
    the path) and invalidates import caches so transformers' is_fla_available() probe sees it
    gone. ``pip uninstall`` alone is unreliable here — it targets one site-packages but the base
    image bakes ``fla`` into another dir on the path (and can report success while leaving the
    package dir). Returns ``(removed_dirs, still_importable)``. Used by the Hopper auto-drop.
    """
    import importlib
    import importlib.util
    import shutil

    removed: list[str] = []
    for _ in range(6):  # a few passes: removing one copy can reveal another earlier on the path
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
        # Resolve the package directory (submodule_search_locations for a package, else the file dir).
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if not locs and spec.origin:
            locs = [os.path.dirname(spec.origin)]
        progressed = False
        for loc in locs:
            if loc and os.path.isdir(loc) and os.path.basename(loc.rstrip("/")) == "fla":
                try:
                    shutil.rmtree(loc)
                    removed.append(loc)
                    progressed = True
                except Exception as e:
                    print(f"[fla] could not remove {loc}: {e}", flush=True)
        if not progressed:
            break
    importlib.invalidate_caches()
    return removed, importlib.util.find_spec("fla") is not None


# Long-context runs are memory-bound (activations + vLLM KV cache scale with sequence length), so
# they need the memory features even on a SMALL model — PR #174 measured a 1B model OOM on GRPO at
# 4096 ctx in speed mode, but it fits in memory mode. So "memory mode" = large model OR long ctx.
_LONG_CONTEXT_TOKENS = 2048


def _memory_mode(model_id: str, max_length: int = 0) -> bool:
    """Whether to default the memory-saving features (Liger, grad-checkpointing, vLLM sleep) ON:
    a large model (fused-CE memory win) OR a long context (activations/KV dominate). Small model +
    short context -> off (optimize for speed)."""
    if max_length and max_length >= _LONG_CONTEXT_TOKENS:
        return True
    return _liger_default_for_model(model_id)


def grad_checkpointing_on(model_id: str, max_length: int = 0) -> bool:
    """Gradient checkpointing recomputes the forward in backward (~25% slower) to save activation
    memory — a MEMORY feature, not speed. ON for large models / long context that need the
    headroom; OFF for small+short runs that fit without it (the speed win)."""
    return _memory_mode(model_id, max_length)


def fused_optim_name() -> str:
    """TRL/HF ``optim`` value: 8-bit paged AdamW (bitsandbytes int8 optimizer state paged to host
    RAM). It fits a smaller/cheaper GPU and is the better default across the catalog."""
    return "paged_adamw_8bit"


def _reset_peak_gpu() -> None:
    """Reset the CUDA peak-memory counter so a subsequent ``_peak_gpu_gb`` measures only the work
    that follows (e.g. the train loop, isolating the optimizer-state A/B from setup/model load)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_gpu_gb() -> float:
    """Peak torch-allocated CUDA memory (GB) since the last reset; 0.0 if no CUDA. Note: bnb paged
    8-bit optimizer state lives in unified/managed memory outside torch's caching allocator and is
    NOT counted here — so this OVERSTATES the 8-bit saving. _GpuPeakSampler measures the true
    device footprint (incl. bnb managed pages) for the honest A/B number."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return 0.0


class _GpuPeakSampler:
    """Background sampler of true device memory (GB) = (total - free) from cuda.mem_get_info, which
    DOES include bitsandbytes managed/paged optimizer pages while they're GPU-resident (torch's
    max_memory_allocated does not). This is the honest peak for the fp32-vs-8-bit optimizer A/B."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_used = 0
        self._stop = False
        self._thread = None

    def _run(self):
        import torch

        while not self._stop:
            try:
                free, total = torch.cuda.mem_get_info()
                self.peak_used = max(self.peak_used, total - free)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        try:
            import threading

            import torch

            if not torch.cuda.is_available():
                return self
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            pass
        return self

    def stop_gb(self) -> float:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        return round(self.peak_used / 1e9, 3)


def loraplus_optimizer_cls(optim_name: str):
    """Optimizer class for the LoRA+ ``create_optimizer`` override (returns ``(cls, extra_kwargs)``).

    The LoRA+ override has to *build* the optimizer itself (PEFT splits the LoRA A/B matrices into
    separate param groups with different LRs), so it cannot inherit TRL's ``optim=`` string — it has
    to choose a concrete class. Historically it always built a full-precision ``torch.optim.AdamW``,
    which silently discarded the catalog's ``paged_adamw_8bit`` setting whenever LoRA+ was on.

    PEFT's ``create_loraplus_optimizer`` accepts ANY ``optimizer_cls`` — including bitsandbytes 8-bit
    optimizers (it registers embedding overrides with bnb's ``GlobalOptimManager`` to keep them
    32-bit) — so LoRA+ and the 8-bit paged optimizer state coexist. An ``8bit`` ``optim`` value
    (the fleet default; ``fused_optim_name`` -> ``paged_adamw_8bit``) selects
    ``bnb.optim.PagedAdamW8bit``; a non-8-bit ``optim`` keeps fp32 AdamW. This simply mirrors the
    configured ``optim`` — there is no separate toggle: an on-GPU A/B (Qwen3.5-4B SFT, rank-128
    LoRA, same seed/data/init) measured the 8-bit paged state at -75% optimizer memory
    (1359 -> 346 MB) and -0.72 GB peak with NO convergence penalty (final loss 10.64 vs 11.16 from
    an identical start), so it's unconditionally the default wherever ``optim`` is 8-bit. Falls
    back to fp32 AdamW only if bitsandbytes is missing."""
    import torch as _torch

    # case-insensitive + str-safe: TRL normalizes optim to an OptimizerNames enum whose str() is
    # "OptimizerNames.PAGED_ADAMW_8BIT" (uppercase), so a bare `"8bit" in optim_name` would miss it.
    if "8bit" in str(optim_name or "").lower():
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit, {}
        except Exception as e:  # bnb missing / no CUDA build -> safe fp32 fallback
            print(f"[lora+] bitsandbytes 8-bit optimizer unavailable ({e}); using fp32 AdamW")
    return _torch.optim.AdamW, {}


def _sdpa_cudnn_ctx(attn_impl: str | None):
    """Context forcing the cuDNN SDPA backend (real Blackwell-consumer kernels) when we fell
    back to plain SDPA on sm120; a no-op context otherwise. Best-effort."""
    if attn_impl != "sdpa":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Priority-ordered: prefer the fast cuDNN/flash/efficient kernels, but ALWAYS include MATH
        # as the final fallback. Restricting to only [CUDNN, EFFICIENT] makes sm120 GRPO crash with
        # "RuntimeError: No available kernel" when neither has a kernel for the completion-batch
        # attention shape (MEASURED: Qwen3.5 GRPO on RTX 5090). MATH is universal, so the candidate
        # set is never empty; set_priority keeps cuDNN first whenever it CAN serve the shape (SFT
        # fast path unchanged), only falling through for the shapes cuDNN/efficient reject.
        return sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
    except Exception as e:
        print("[attn] cuDNN SDPA backend unavailable, using default SDPA:", e)
        return contextlib.nullcontext()



def gpu_diagnostics() -> dict:
    """Collect CUDA/driver diagnostics to pin down GPU init failures on rented nodes."""
    diag = {}
    try:
        import torch

        diag["torch"] = torch.__version__
        diag["torch_cuda"] = torch.version.cuda
        diag["cuda_available"] = torch.cuda.is_available()
        try:
            diag["device_count"] = torch.cuda.device_count()
            diag["device_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            diag["device_query_err"] = str(e)[:160]
    except Exception as e:
        diag["torch_import_err"] = str(e)[:160]
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        diag["nvidia_smi"] = (out.stdout or out.stderr).strip()[:200]
    except Exception as e:
        diag["nvidia_smi_err"] = str(e)[:160]
    return diag


def wait_for_gpu():
    """Rented nodes sometimes report 'CUDA device not ready' transiently at startup.
    Poll a trivial CUDA op until it succeeds before doing real work; raise if never ready."""
    import time as _t

    last = None
    for i in range(12):
        try:
            import torch

            if torch.cuda.is_available():
                # Force an actual kernel launch (alloc + add) to confirm the GPU is live.
                _ = torch.zeros(8, device="cuda") + 1
                torch.cuda.synchronize()
                print(f"GPU ready after {i} retries: {torch.cuda.get_device_name(0)}")
                return True
            last = "cuda not available"
        except Exception as e:
            last = str(e)[:160]
        print(f"GPU not ready (try {i + 1}/12): {last}; sleeping 10s")
        _t.sleep(10)
    raise RuntimeError(f"GPU never became ready after 12 tries: {last}")


def free_gpu(trainer=None):
    try:
        import gc

        import torch

        try:
            if trainer is not None and hasattr(trainer, "model"):
                trainer.model = None
        except Exception:
            # Best-effort VRAM release before gc; any failure here is non-fatal cleanup.
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print("free_gpu warn:", e)


def _metric_curve(trainer, key: str) -> list:
    """The logged values of `key` (e.g. 'loss' or 'reward') from the trainer's log history,
    rounded + capped. Lets metrics.json carry the convergence/reward curve for an A/B without
    relying on a checkpoint's trainer_state.json (only written on save_steps) or the console
    (only uploaded on failure). Never raises."""
    try:
        vals = [round(float(h[key]), 4) for h in trainer.state.log_history if key in h]
        return vals[:400]
    except Exception:
        return []


def _ensure_fla_fastpath_on_hopper() -> None:
    """Make flash-linear-attention's GatedDeltaNet fast path CORRECT + fast on Hopper (sm90)
    instead of dropping it.

    fla's gated chunk_bwd Triton kernel is miscomputed on Hopper with Triton>=3.4 and HARD-RAISES
    (fla #640). The worker historically DROPPED fla here and fell back to the pure-PyTorch delta
    rule — correct but slow + memory-heavy. The real fix is fla's **tilelang** backend, which is
    correct on Triton>=3.4. So on Hopper we ensure the working stack is present rather than
    removing fla:
      * the pinned ``tilelang==0.1.11`` (the correct GDN chunk_bwd backend) + the pinned
        ``apache-tvm-ffi==0.1.11`` (0.1.12 double-registers the TVM-FFI runtime -> ``import
        tilelang`` aborts; and tilelang's own ``apache-tvm-ffi~=0.1.0`` range would let 0.1.12
        back in, so the pin is force-installed last and its resolved version is verified), and
      * a COMPLETE ``fla`` (the PyPI ``flash-linear-attention`` wheel is a broken stub missing
        ``fla.modules``; reinstall from git if the resident copy is incomplete).
    Validated A/B (H100 SXM, Qwen3.5 hidden-2560 LoRA, controlled fla on/off): seq4096 435->105
    ms/step & 9.9->6.1 GB (4.2x / 1.6x); seq8192 7.1x; seq16384 3106->247 ms & 32->17 GB (12.6x /
    1.9x). Forward loss matches the torch delta to 1.8e-4 (correct). Runs on BOTH substrates
    (RunPod + Vast exec this module), after all installs and BEFORE any model import. Non-Hopper:
    no-op (fla's Triton kernel is correct there). Best-effort + FAIL-CLOSED: a failed install
    (pip rc!=0), a missing module, or the wrong resolved ``apache-tvm-ffi`` version all flip the
    gate off and DISABLE fla, leaving the (correct) pure-PyTorch delta rule in place — a worker
    never crashes on a dep hiccup, and it never silently runs fla's broken Hopper GDN kernel.
    """
    import importlib
    import importlib.util
    import subprocess

    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
            return  # not Hopper: fla's Triton kernel is correct here.
    except Exception:
        return

    def _have(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    def _ver(dist: str) -> str | None:
        """Installed version of a distribution (by metadata), or None if absent/unreadable.

        Distinct from _have (a find_spec import probe): the install can silently leave the WRONG
        version resolved (e.g. tilelang's ``apache-tvm-ffi~=0.1.0`` range happily keeps 0.1.12,
        which still find_spec-imports but aborts ``import tilelang``), so the gate must check the
        actual installed version, not just importability.
        """
        try:
            import importlib.metadata as _md

            return _md.version(dist)
        except Exception:
            return None

    def _pip(*args: str) -> bool:
        """Run pip install; return True only if pip exited 0. A failed install (network/build/
        resolver) must NOT be silently treated as success — the caller gates ``ok`` on this."""
        try:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *args], check=False
            ).returncode
        except Exception:
            return False
        return rc == 0

    # The exact apache-tvm-ffi pin the tilelang backend needs (0.1.12 double-registers the TVM-FFI
    # runtime -> `import tilelang` aborts). Kept as a constant so the install spec and the post-
    # install version gate below can't drift apart. Keep in lockstep with WORKER_DEPS / Dockerfile.
    TVM_FFI_PIN = "0.1.11"
    TILELANG_PIN = "0.1.11"  # pin the GDN backend too (same rationale as the fla SHA pin)

    try:
        # 1. tilelang backend (correct GDN chunk_bwd on Triton>=3.4) + the pinned tvm-ffi.
        #    Track whether each install actually succeeded — a failed pip (rc!=0) must flip the
        #    gate to the pure-PyTorch fallback rather than be ignored. (_have-only would also pass
        #    on a stale/partial copy from a previous boot.) tilelang pulls apache-tvm-ffi via a
        #    range that allows the broken 0.1.12, so force-reinstall the exact pin AFTER tilelang
        #    and verify the resolved version below.
        # Enforce the EXACT pin: (re)install when tilelang is absent OR a different version is
        # resident (a job or the base image may carry another tilelang; _have-only would treat that
        # as healthy and skip the install, leaving the wrong/uncertain GDN backend in place). Mirror
        # the apache-tvm-ffi handling: check the installed version via _ver and reinstall on mismatch.
        tilelang_ok = True
        if _ver("tilelang") != TILELANG_PIN:
            tilelang_ok = _pip(f"tilelang=={TILELANG_PIN}")
        # Force the exact tvm-ffi pin last (overriding whatever tilelang's range resolved). If this
        # install fails we DON'T trust the resident copy — tvm_ffi_ok gates `ok` below.
        tvm_ffi_ok = _pip(f"apache-tvm-ffi=={TVM_FFI_PIN}")
        # 2. a COMPLETE fla — the PyPI wheel ships a stub without `fla.modules`. Reinstall from git
        #    when the resident copy is missing the real package (or absent entirely).
        fla_ok = True
        if not (_have("fla") and _have("fla.modules")):
            _remove_fla_from_disk()  # clear any broken stub before the git install
            # Pinned to the same commit as WORKER_DEPS / Dockerfile.worker so a runtime reinstall is
            # reproducible (the moving default branch could pull a broken/incompatible fla).
            fla_ok = _pip(
                "--no-deps",
                "git+https://github.com/fla-org/flash-linear-attention.git"
                "@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
            )
        importlib.invalidate_caches()
        # Gate on BOTH (a) every install we ran exiting 0 — a failed pip (network/build/resolver)
        # must NOT be treated as healthy just because a stale/partial copy still find_spec-imports —
        # AND (b) the modules importing AND (c) the resolved apache-tvm-ffi being exactly the pin.
        # (c) matters because tilelang depends on `apache-tvm-ffi~=0.1.0`, so the resolver can keep
        # the broken 0.1.12 (which find_spec-imports fine but aborts `import tilelang`); checking the
        # version is the only reliable signal the pin actually landed.
        tvm_ffi_ver = _ver("apache-tvm-ffi")
        tilelang_ver = _ver("tilelang")
        installs_ok = tilelang_ok and tvm_ffi_ok and fla_ok
        ok = (
            installs_ok
            and _have("fla")
            and _have("fla.modules")
            and _have("tilelang")
            and tilelang_ver == TILELANG_PIN
            and tvm_ffi_ver == TVM_FFI_PIN
        )
        if not ok:
            # The healthy fla+tilelang stack could not be assembled, so fla's GDN chunk_bwd would
            # still hit the broken Triton>=3.4 path on Hopper (fla #640) and HARD-RAISE. A print
            # alone does NOT prevent that: transformers gates GDN on is_fla_available() (a
            # find_spec('fla') probe), so as long as fla stays importable it gets engaged. PHYSICALLY
            # remove fla so the probe sees it gone and transformers uses the correct pure-PyTorch
            # delta rule instead of crashing. _remove_fla_from_disk loops over the real sys.path +
            # invalidates caches, so find_spec('fla') is None afterwards (the gate flips off).
            _removed, _still = _remove_fla_from_disk()
            print(
                "[hopper] fla GDN fast path unavailable -> DISABLING fla "
                f"(installs_ok={installs_ok} [tilelang={tilelang_ok} tvm_ffi={tvm_ffi_ok} "
                f"fla={fla_ok}], tilelang_ver={tilelang_ver!r} (want {TILELANG_PIN}), "
                f"tvm_ffi_ver={tvm_ffi_ver!r} (want {TVM_FFI_PIN}); "
                f"removed {len(_removed)} copy(ies); still_importable={_still}); "
                "pure-PyTorch delta fallback",
                flush=True,
            )
        else:
            print(
                "[hopper] fla GDN fast path ENABLED (fla+tilelang "
                f"{tilelang_ver}/tvm-ffi {tvm_ffi_ver}, fla #640 fixed)",
                flush=True,
            )
    except Exception as e:  # never let a dep hiccup crash the worker — torch delta still runs
        print(
            f"[hopper] fla fast-path setup skipped ({type(e).__name__}: {e}); pure-PyTorch delta",
            flush=True,
        )
