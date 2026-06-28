"""GPU/backend setup the worker runs at boot or just before TRL builds the rollout engine.

Run-scoped state (``PHASE``/``JOB_SPEC``) is read through the worker package at CALL time so a
test's ``monkeypatch.setattr(worker, ...)`` reaches these readers."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import grpo_sleep_mode


def force_vllm_backend_for_sm120() -> str | None:
    """On RTX 5090 / consumer Blackwell (sm120), force a PTX-independent vLLM attention backend.

    vLLM's default rollout backend is flash-attn, whose PRE-BUILT PTX needs a newer driver JIT than
    many 5090 RunPod hosts have — when the JIT fails the colocated rollout silently produces NO
    completions (empty reward_history, ~1.4 s "done"; a whole 22-run sweep hit this on every 5090).
    FLASHINFER is vLLM's Blackwell-native backend (no flash-attn PTX dependency) and trains on a 5090
    (~116 s); TRITON_ATTN (also PTX-independent) is the fallback when flashinfer is unimportable. Both
    are REGISTERED decoder backends on the pinned vllm 0.19.1 — unlike TORCH_SDPA, which that version
    registers as ViT-only (empty decoder class path) so selecting it for the rollout engine would
    raise. This mirrors the trainer's cuDNN-SDPA forcing on sm120 (``_attn_impl_for_capability``).

    The pinned vLLM (0.19.1) dropped ``VLLM_ATTENTION_BACKEND`` from its env registry, so the backend
    is no longer selectable via ``os.environ`` (that assignment would be a silent no-op). It is now an
    ``LLM(...)`` constructor kwarg (``attention_backend``), so we inject it into TRL's colocate rollout
    engine via ``patch_trl_colocate_llm_kwargs`` (which runs BEFORE GRPOTrainer builds the engine);
    ``EngineArgs`` coerces the bare member name through ``AttentionConfig.validate_backend_before``. The
    GRPO no-op guard remains the backstop. Returns the backend pinned (None if not sm120). Fixed — no
    operator override."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
            return None
    except Exception as e:
        print("[rl] sm120 vLLM backend probe skipped:", e)
        return None
    # flashinfer is MANDATORY for this path but the Dockerfile imports it OPTIONALLY, so flashinfer-python
    # can install yet be ABI-broken against this torch and an UNCONDITIONAL FLASHINFER would ship fine and
    # only crash at the first sm120 engine init. Gate on the import and fall back to
    # TRITON_ATTN — a REGISTERED decoder backend (vllm 0.19.1 registry) that is PTX-independent
    # (Triton-compiled) and trains on the 5090 — so a broken flashinfer degrades gracefully instead of
    # hard-failing. NOT TORCH_SDPA: in vllm 0.19.1 that name is ViT-ONLY (empty decoder class path), so
    # selecting it for the rollout engine RAISES at backend validation. NOT vLLM's default flash-attn:
    # its prebuilt PTX is exactly the consumer-Blackwell unreliability this function exists to avoid.
    try:
        import flashinfer  # noqa: F401

        backend = "FLASHINFER"
    except Exception as e:
        backend = "TRITON_ATTN"
        print(
            f"[rl] sm120 (RTX 5090): flashinfer import failed ({e}) -> attention_backend=TRITON_ATTN "
            "(PTX-independent registered decoder backend; flashinfer wheel is ABI-broken against this torch)"
        )
    # Inject as a colocate LLM(...) kwarg (the env var is a no-op on vllm 0.19.1). Runs here, before
    # TRL builds the engine, so the wrapper is in place when GRPOTrainer constructs the rollout LLM.
    patch_trl_colocate_llm_kwargs(attention_backend=backend)
    if backend == "FLASHINFER":
        print(
            "[rl] sm120 (RTX 5090): pinned attention_backend=FLASHINFER on the colocate rollout engine "
            "(flash-attn PTX is unreliable on consumer Blackwell hosts -> empty-rollout failures)"
        )
    return backend


def finalize_alloc_conf_for_sleep() -> None:
    """Sync the CUDA allocator conf with the worker's RESOLVED vLLM sleep default (RL runs only).

    The launcher (providers/*/train.py build_worker_env) picks the sleep-SAFE non-expandable
    PYTORCH_ALLOC_CONF for RL before this process starts, but it can't know the GRPO sleep decision:
    for a small model the worker resolves sleep OFF (the speed default), so the non-expandable conf
    is safe but fragments a long colocate run. Here (we have the model config + GPU) we resolve the
    SAME deterministic sleep default (``grpo_sleep_mode``, exactly run_rl's gate) and, if sleep is
    OFF, switch to expandable_segments — which only crashes WITH sleep on, a case we've just ruled
    out. PYTORCH_ALLOC_CONF is read lazily at the first CUDA allocation, so this must run before any
    allocation (it does — called at boot)."""
    if _w.PHASE != "rl":
        return
    try:
        model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else ""
        # Resolve the sleep decision EXACTLY as run_rl does (grpo_sleep_mode: the size/context gate
        # PLUS the resident-fit check against the live card), so the alloc conf matches the sleep
        # mode the trainer will actually use.
        _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
        ctx = 0
        try:
            if _t and _t.max_length:
                ctx = int(_t.max_length)
        except Exception:
            ctx = 0
        card_gb = 0.0
        # fp8 KV (cc>=8.9) halves the resident KV bytes, so grpo_sleep_mode's resident-fit gate admits
        # a LONGER context as fitting. Derive it EXACTLY as run_rl does (device capability >= (8, 9))
        # and feed it to the SAME gate below -- else a long-context MoE that fits resident ONLY with
        # fp8 KV resolves sleep ON here (keeping the non-expandable conf) while run_rl runs sleep OFF,
        # diverging from this function's documented "exactly run_rl's gate" contract.
        fp8_kv = False
        try:
            import torch as _torch_card

            if _torch_card.cuda.is_available():
                # Decimal GB (/1e9) to match grpo_fits_resident's comparison target (see run_rl):
                # estimate_vram_gb is decimal, so the card size fed to the same sleep gate must be
                # decimal too or this alloc-conf decision would diverge from the trainer's.
                card_gb = _torch_card.cuda.get_device_properties(0).total_memory / 1e9
                fp8_kv = _torch_card.cuda.get_device_capability() >= (8, 9)
        except Exception:
            card_gb = 0.0
        # Resolve group_size EXACTLY as run_rl does (gcfg override, else the recipe default), not a
        # flat 8: if the recipe's rl.group_size differs from 8 the alloc-conf sleep decision here
        # would diverge from the trainer's, picking the wrong expandable/non-expandable conf.
        from flash.engine.recipe import RECIPE as _RECIPE
        from flash.engine.worker.grpo import grpo_overrides

        _gcfg = grpo_overrides()
        _group_size = int(_gcfg.get("group_size") or _RECIPE.rl.group_size)
        sleep_on = grpo_sleep_mode(
            model_id,
            max_length=ctx,
            group_size=_group_size,
            max_tokens=(_t.max_tokens if _t else None),
            lora_rank=int(_t.lora_rank) if _t and _t.lora_rank else 32,
            thinking=_w.THINKING,
            card_vram_gb=card_gb,
            fp8_kv=fp8_kv,
        )
        if not sleep_on:  # sleep resolves OFF -> expandable is safe + better
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)


def patch_trl_colocate_llm_kwargs(
    *,
    kv_cache_dtype: str | None = None,
    max_num_batched_tokens: int | None = None,
    enforce_eager: bool | None = None,
    attention_backend: str | None = None,
) -> bool:
    """Inject vLLM ``LLM(...)`` kwargs into TRL's colocated rollout engine that can't be expressed via
    ``GRPOConfig`` or the (pinned) vLLM env registry — must run BEFORE ``GRPOTrainer.__init__`` builds
    the engine.

    TRL 1.6 builds the colocate engine in ``trl/generation/vllm_generation.py``
    (``VLLMGeneration.__init__``) where ``max_num_batched_tokens`` is HARDCODED to 4096 and
    ``kv_cache_dtype`` is NEVER passed (KV stays bf16). ``GRPOConfig`` 1.6 exposes neither
    ``vllm_kv_cache_dtype`` nor ``vllm_max_num_batched_tokens`` (so run_rl's earlier ``_set_vllm_field``
    attempts are silent no-ops on this TRL). The pinned vLLM (0.19.1) ALSO dropped
    ``VLLM_ATTENTION_BACKEND`` and ``VLLM_TORCH_COMPILE_LEVEL`` from its env registry, so the rollout
    attention backend and eager mode are no longer settable via ``os.environ`` either — they are
    ``LLM(...)`` constructor kwargs (``attention_backend`` / ``enforce_eager``) now. The clean, surgical
    injection for ALL of these is to wrap the module-global ``LLM`` symbol that file imports
    (``from vllm import LLM``) so OUR kwargs are applied at construction; the real vLLM ``LLM`` still
    runs.

    * ``kv_cache_dtype="fp8"``: e4m3 fp8 KV — zero-config on Ada/Hopper/Blackwell + FlashInfer (no
      calibration/scales file; scales default to 1.0), ~halving the KV bytes/token so the same KV pool
      holds ~2x the concurrent rollouts (the A3B decode is concurrency/KV-bound). ~1-2 pt of decode
      degradation, acceptable for RL rollouts.
    * ``max_num_batched_tokens``: override TRL's profiler-driven 4096 floor (it under-allocates prefill
      on a big card); speeds prompt prefill (decode is gated by concurrency + KV, not this).
    * ``enforce_eager=True``: run the rollout engine in pure eager mode (no torch.compile, no CUDA-graph
      capture) — the SUPPORTED replacement for the removed ``VLLM_TORCH_COMPILE_LEVEL=0`` env. Dodges
      the vLLM 0.19.1 aot_compile (Ampere sm86) + Triton slot-mapping (graph-capture) crashes off B200.
    * ``attention_backend="FLASHINFER"|"TRITON_ATTN"``: pin a PTX-independent decoder attention backend
      (consumer-Blackwell sm120) — the SUPPORTED replacement for the removed ``VLLM_ATTENTION_BACKEND``
      env. ``EngineArgs`` coerces the bare member name through ``AttentionConfig.validate_backend_before``.

    Repeated calls COMPOSE: run_rl injects the attention backend, the KV/prefill knobs, and eager mode
    at three SEPARATE points, so each call merges its kwargs into one accumulated module-level override
    dict that the wrapper reads at construction time. Without this the first call's wrap-once guard
    would silently drop every later call's kwargs. A missing trl/vllm import is a safe no-op. Returns
    True iff the override is in effect (wrapper installed)."""
    new_overrides: dict = {}
    if kv_cache_dtype is not None:
        new_overrides["kv_cache_dtype"] = kv_cache_dtype
    if max_num_batched_tokens is not None:
        new_overrides["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if enforce_eager is not None:
        new_overrides["enforce_eager"] = bool(enforce_eager)
    if attention_backend is not None:
        new_overrides["attention_backend"] = attention_backend
    if not new_overrides:
        return False
    try:
        import trl.generation.vllm_generation as _vg
    except Exception as e:
        print("[rl] trl colocate-LLM patch skipped (no trl.generation.vllm_generation):", e)
        return False
    # Accumulate across calls into one module-level dict the wrapper reads at build time. Merge
    # in place (never rebind) so the wrapper's reference stays valid; later keys win.
    overrides = getattr(_vg, "_flash_llm_overrides", None)
    if overrides is None:
        overrides = {}
        _vg._flash_llm_overrides = overrides
    overrides.update(new_overrides)
    if getattr(_vg, "_flash_llm_kwargs_patched", False):
        print(f"[rl] colocate vLLM LLM kwargs extended (wrapper already installed): {new_overrides}")
        return True
    # TRL binds the module-global ``LLM`` only under ``if is_vllm_available():`` (it's the symbol the
    # colocate ``self.llm = LLM(...)`` references). If vLLM isn't importable here there is nothing to
    # wrap — a safe no-op (this is the worker's GRPO path, where vLLM is always present).
    _orig_LLM = getattr(_vg, "LLM", None)
    if _orig_LLM is None:
        print("[rl] trl colocate-LLM patch skipped (vllm not available -> no LLM symbol to wrap)")
        return False

    def _patched_LLM(*args, **kwargs):
        # Override TRL's hardcoded/absent values with ours (the kwargs TRL/env can't set). Read the
        # accumulated dict live so kwargs registered by later calls are applied too.
        kwargs.update(_vg._flash_llm_overrides)
        print(f"[rl] colocate vLLM LLM(...) kwargs override applied: {_vg._flash_llm_overrides}")
        return _orig_LLM(*args, **kwargs)

    _vg.LLM = _patched_LLM
    _vg._flash_llm_kwargs_patched = True
    print(f"[rl] patched trl.generation.vllm_generation.LLM for colocate rollout: {overrides}")
    return True
