"""GPU/backend setup the worker runs at boot or just before TRL builds the rollout engine.

Run-scoped state is read through the worker package at CALL time so monkeypatch reaches these readers."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w


def force_vllm_backend_for_sm120() -> str | None:
    """Force FLASHINFER on sm120 (RTX 5090): flash-attn PTX is unreliable on consumer Blackwell hosts,
    causing silent empty-rollout failures. Returns the backend set, or None if not sm120."""
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


def force_vit_sdpa_on_blackwell() -> bool:
    """Force the VISION-encoder (ViT) attention backend to TORCH_SDPA on Blackwell (sm100/sm120).

    Qwen3.6-35B-A3B is a VL model, so vLLM builds its vision tower even for a text-only GRPO rollout
    and runs a ViT attention during engine init/profiling. On Blackwell vLLM 0.19.1 routes the ViT to
    its CUTE flash-attn (``vit_flash_attn_wrapper`` -> ``flash_attn_varlen_func`` ->
    ``vllm.vllm_flash_attn.cute``), which is UNIMPORTABLE against every published ``nvidia-cutlass-dsl``:
    the vendored cute references ``cutlass.cute.core.ThrMma`` (present only <=4.5.x) while
    ``vit_attn_wrappers`` also imports ``cutlass._mlir_helpers`` (present only >=4.6.0) — the two symbols
    never coexist, so the first ViT attention call aborts with
    ``AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'`` (4.6.0) or
    ``ModuleNotFoundError: cutlass._mlir_helpers`` (<=4.5.2), crashing EVERY B200 GRPO rollout (a
    version pin cannot fix it — measured 2026-07-07). ``get_vit_attn_backend`` honors
    ``MultiModalConfig.mm_encoder_attn_backend`` UNCONDITIONALLY, and TORCH_SDPA is a supported ViT
    backend on cc>=8.0, so pinning it sidesteps the CUTE import entirely (the LM/decoder attention is
    unaffected — that path is FLASHINFER/flash-attn, chosen separately).

    * Injected as a colocate ``LLM(...)`` kwarg (``EngineArgs.mm_encoder_attn_backend``, str-typed),
      composing with the other overrides; must run BEFORE ``GRPOTrainer.__init__`` builds the engine.
    * No-op off Blackwell and on non-multimodal models (a text-only model simply never builds a ViT).
    Returns True iff the override was injected."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] not in (10, 12):
            return False
    except Exception as e:
        print("[rl] ViT-SDPA Blackwell probe skipped:", e)
        return False
    ok = patch_trl_colocate_llm_kwargs(mm_encoder_attn_backend="TORCH_SDPA")
    if ok:
        print(
            "[rl] Blackwell (sm100/sm120): mm_encoder_attn_backend=TORCH_SDPA on the colocate rollout "
            "engine (vLLM 0.19.1 ViT CUTE flash-attn is unimportable vs every nvidia-cutlass-dsl: "
            "cute.core.ThrMma <=4.5.x XOR cutlass._mlir_helpers >=4.6.0 -> crashes the B200 rollout)"
        )
    return ok


def finalize_alloc_conf_for_sleep() -> None:
    """Sync PYTORCH_ALLOC_CONF with the resolved GRPO sleep mode (TRL RL only).

    PYTORCH_ALLOC_CONF is read at first CUDA allocation — must run before any allocation."""
    if _w.PHASE != "rl":
        return
    # verl owns its own rollout engine and always builds a CuMemAllocator (it leaves
    # rollout.enable_sleep_mode defaulted True), which asserts on "expandable_segments:True"
    # (vllm/device_allocator/cumem.py:132, pytorch#147851). flash's sleep resolution describes the TRL
    # colocate engine and does not apply, so never let it upgrade the conf out from under verl.
    if os.environ.get("FLASH_RL_BACKEND", "trl").strip().lower() == "verl":
        print(
            "[alloc] verl backend -> keeping launcher's non-expandable conf (vllm CuMemAllocator)"
        )
        return
    try:
        from flash.engine.worker.grpo import resolve_grpo_sleep_mode

        sleep_on, _ctx, _card_gb, _fp8_kv = resolve_grpo_sleep_mode()
        if not sleep_on:  # expandable_segments crashes only when sleep is ON
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)


def _is_cudagraph_capture_failure(exc: BaseException) -> bool:
    markers = (
        "capture_model",
        "cuda graph",
        "cudagraph",
        "graph capture",
    )
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(marker in f"{type(current).__name__}: {current}".lower() for marker in markers):
            return True
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            location = f"{frame.f_code.co_filename}:{frame.f_code.co_name}".lower()
            if any(marker in location for marker in markers):
                return True
            traceback = traceback.tb_next
        current = current.__cause__ or current.__context__
    return False


def _release_failed_cudagraph_capture(exc: BaseException) -> None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        next_exc = current.__cause__ or current.__context__
        current.__traceback__ = None
        current = next_exc
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as cleanup_exc:
        print(f"[rl][warn] cuda graph fallback cleanup failed: {cleanup_exc}")


def patch_trl_colocate_llm_kwargs(
    *,
    kv_cache_dtype: str | None = None,
    max_num_batched_tokens: int | None = None,
    enforce_eager: bool | None = None,
    compilation_config: dict | None = None,
    attention_backend: str | None = None,
    mm_encoder_attn_backend: str | None = None,
    reasoning_parser: str | None = None,
    revision: str | None = None,
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
    * ``compilation_config``: select a narrower vLLM compilation profile, such as decode-only CUDA
      graphs with torch.compile/AOT disabled.
    * ``attention_backend="FLASHINFER"|"TRITON_ATTN"``: pin a PTX-independent decoder attention backend
      (consumer-Blackwell sm120) — the SUPPORTED replacement for the removed ``VLLM_ATTENTION_BACKEND``
      env. ``EngineArgs`` coerces the bare member name through ``AttentionConfig.validate_backend_before``.
    * ``mm_encoder_attn_backend="TORCH_SDPA"``: force the VISION-encoder (ViT) attention backend for a
      multimodal model (Qwen3.6-35B-A3B is VL) off vLLM's CUTE flash-attn path — see
      ``force_vit_sdpa_on_blackwell``. Maps to ``MultiModalConfig.mm_encoder_attn_backend``, which
      ``get_vit_attn_backend`` honors unconditionally.
    * ``reasoning_parser="deepseek_r1"``: gate the ``[train] structured_outputs`` grammar on the
      ``</think>`` boundary so a thinking-mode rollout reasons freely before its answer is constrained
      (``EngineArgs.reasoning_parser`` -> vLLM's V1 structured-output manager defers the bitmask until
      reasoning ends). Only meaningful alongside a structured-outputs constraint under ``thinking``;
      TRL/``GRPOConfig`` exposes no field for it, so it must ride this ``LLM(...)`` override.
    * ``revision``: pin the colocated rollout engine to the same student repository revision as the
      trainer. empty revisions are omitted rather than forwarded as ``revision=""``.

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
    if compilation_config is not None:
        new_overrides["compilation_config"] = dict(compilation_config)
    if attention_backend is not None:
        new_overrides["attention_backend"] = attention_backend
    if mm_encoder_attn_backend is not None:
        new_overrides["mm_encoder_attn_backend"] = mm_encoder_attn_backend
    if reasoning_parser is not None:
        new_overrides["reasoning_parser"] = reasoning_parser
    if revision:
        new_overrides["revision"] = revision
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
        print(
            f"[rl] colocate vLLM LLM kwargs extended (wrapper already installed): {new_overrides}"
        )
        return True
    # TRL binds the module-global ``LLM`` only under ``if is_vllm_available():`` (it's the symbol the
    # colocate ``self.llm = LLM(...)`` references). If vLLM isn't importable here there is nothing to
    # wrap — a safe no-op (this is the worker's GRPO path, where vLLM is always present).
    _orig_LLM = getattr(_vg, "LLM", None)
    if _orig_LLM is None:
        print("[rl] trl colocate-LLM patch skipped (vllm not available -> no LLM symbol to wrap)")
        return False

    def _patched_LLM(*args, **kwargs):
        # override trl's hardcoded/absent values with ours (the kwargs trl/env can't set). read the
        # accumulated dict live so kwargs registered by later calls are applied too.
        kwargs.update(_vg._flash_llm_overrides)
        print(f"[rl] colocate vLLM LLM(...) kwargs override applied: {_vg._flash_llm_overrides}")
        compilation = kwargs.get("compilation_config") or {}
        decode_graphs = (
            not kwargs.get("enforce_eager", False)
            and compilation.get("cudagraph_mode") == "FULL_DECODE_ONLY"
        )
        try:
            return _orig_LLM(*args, **kwargs)
        except Exception as capture_exc:
            if not decode_graphs or not _is_cudagraph_capture_failure(capture_exc):
                raise
            _release_failed_cudagraph_capture(capture_exc)
            eager_kwargs = dict(kwargs)
            eager_kwargs["enforce_eager"] = True
            eager_kwargs.pop("compilation_config", None)
            _vg._flash_llm_overrides["enforce_eager"] = True
            _vg._flash_llm_overrides.pop("compilation_config", None)
            print(
                f"[rl][warn] decode-only cuda graph capture failed ({capture_exc}); "
                "retrying colocate vLLM initialization once in eager mode"
            )
            return _orig_LLM(*args, **eager_kwargs)

    _vg.LLM = _patched_LLM
    _vg._flash_llm_kwargs_patched = True
    print(f"[rl] patched trl.generation.vllm_generation.LLM for colocate rollout: {overrides}")
    return True
