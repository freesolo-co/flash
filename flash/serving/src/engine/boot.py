"""Engine startup: tokenizer selection and vLLM ``AsyncEngineArgs`` construction.

Split out of ``_LoraEngineImpl._load``, which had grown to hold four unrelated phases at once.
These two are pure functions of ``(base_model, settings, overrides)``, so a sizing decision can be
asserted without a GPU: ``engine_args_for`` returns the kwargs rather than an engine.

Every override here is real-GPU-validated; the comments record what breaks without it, because the
failures (a profiling OOM, a rejected token budget, an FP8 MoE crash-loop) are only reproducible on
the card they were found on.
"""

from typing import Any

from flash.serving.src.engine.model_config import (
    engine_overrides_for,
    image_limit_for,
    immutable_serving_revisions,
    supports_image_input,
    tokenizer_model_for,
)
from flash.serving.src.engine.support import (
    _async_engine_arg_names,
    _require_reasoning_api_compatibility,
)


def load_tokenizer(
    base_model: str,
    settings: Any,
    cfg: Any,
) -> tuple[Any, Any]:
    """Return ``(processor, tokenizer)`` for ``base_model``.

    An image-capable model's tokenizer must come from its processor, not from ``AutoTokenizer``:
    the processor owns the image-token bookkeeping that the plain tokenizer has no knowledge of.
    """
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    tokenizer_model = tokenizer_model_for(base_model)
    revisions = immutable_serving_revisions(base_model)
    tokenizer_revision = revisions.get("tokenizer_revision")
    processor_revision = revisions.get("processor_revision")
    if supports_image_input(base_model):
        revision_kwargs = {"revision": processor_revision} if processor_revision else {}
        processor = AutoProcessor.from_pretrained(
            tokenizer_model,
            **revision_kwargs,
            token=settings.hf_api_key,
            trust_remote_code=cfg.TRUST_REMOTE_CODE,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("image-capable model processor has no tokenizer")
    else:
        revision_kwargs = {"revision": tokenizer_revision} if tokenizer_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            **revision_kwargs,
            token=settings.hf_api_key,
            trust_remote_code=cfg.TRUST_REMOTE_CODE,
        )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return processor, tokenizer


def _multimodal_args(
    base_model: str,
) -> dict[str, Any]:
    image_limit = image_limit_for(base_model)
    if image_limit is None:
        return {}
    return {
        "limit_mm_per_prompt": {"image": image_limit},
        "mm_processor_cache_gb": 0,
        "enable_tower_connector_lora": True,
    }


def _required_immutable_args(
    base_model: str, overrides: dict[str, Any], engine_arg_names: set[str]
) -> dict[str, Any]:
    required = {
        "revision": overrides.get("model_revision"),
        "tokenizer": overrides.get("tokenizer_model"),
        "tokenizer_revision": overrides.get("tokenizer_revision"),
    }
    missing = [name for name, value in required.items() if value and name not in engine_arg_names]
    if missing:
        raise RuntimeError(
            f"vLLM build cannot pin {base_model} immutable serving identity; missing engine args: "
            f"{', '.join(sorted(missing))}"
        )
    return {name: value for name, value in required.items() if value}


def _build_specific_args(
    base_model: str, overrides: dict[str, Any], engine_arg_names: set[str]
) -> dict[str, Any]:
    """Overrides this vllm build may or may not expose, warned about rather than dropped silently.

    A missing arg is not fatal on its own, but it changes what the engine will tolerate, so each
    branch prints what is likely to break instead of failing at first request with no context.
    """
    extra: dict[str, Any] = {}
    # max_num_batched_tokens: the 35B's attention block size is 2096 tokens, so vLLM rejects the
    # default 2048 token budget. Forward the validated 4096 override only on builds exposing it.
    if "max_num_batched_tokens" in overrides:
        if "max_num_batched_tokens" in engine_arg_names:
            extra["max_num_batched_tokens"] = int(overrides["max_num_batched_tokens"])
        else:
            print(
                f"serving: vLLM build has no max_num_batched_tokens arg; {base_model} "
                "may fail if its attention block size exceeds the default scheduler budget",
                flush=True,
            )
    # moe_backend: optional fused-MoE backend override. Current 35B validation PASSED
    # only with this unset/auto; keep the forwarding path for future canaries and emergency
    # overrides, but do not set it in the catalog unless real-GPU validation proves the value.
    if overrides.get("moe_backend"):
        if "moe_backend" in engine_arg_names:
            extra["moe_backend"] = str(overrides["moe_backend"])
        else:
            print(
                f"serving: vLLM build has no moe_backend arg; {base_model} FP8 MoE + LoRA "
                "may crash-loop (needs vLLM>=0.19, or set quantization=None for bf16 weights)",
                flush=True,
            )
    return extra


def engine_args_for(
    base_model: str,
    overrides: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """The ``AsyncEngineArgs`` kwargs for ``base_model``, as a plain dict.

    Returning kwargs rather than the engine args object is what makes the sizing decisions
    (quantization, lora caps, pinning, the multimodal limits) assertable off-GPU.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    # serve_model_id: load a PRE-QUANTIZED FP8 checkpoint instead of online-quantizing the bf16
    # base (avoids the bf16 load transient — see model_config). Adapters still key off base_model;
    # the tokenizer stays the base model's (canonical). vLLM auto-detects the checkpoint's FP8, so
    # a pre-quant base passes NO quantization (online quantization=fp8 only when absent).
    served_model = overrides.get("serve_model_id") or base_model
    quant_default = None if overrides.get("serve_model_id") else cfg.QUANTIZATION
    max_loras = int(overrides.get("max_loras", cfg.MAX_LORAS))

    extra: dict[str, Any] = {}
    # max_num_seqs is left at vLLM's (large) default unless a model overrides it. vLLM's startup
    # memory-PROFILING forward runs max_num_seqs sequences at once; for the 35B MoE (248k-vocab
    # logits + all-expert activations) the default (~256) spikes the profiling peak to nearly the
    # whole card and the subsequent LoRA-module creation OOMs — so the 35B caps it low.
    if "max_num_seqs" in overrides:
        extra["max_num_seqs"] = int(overrides["max_num_seqs"])
    extra.update(_multimodal_args(base_model))

    parser = overrides.get("reasoning_parser")
    reasoning_parser = str(parser) if parser else None
    _require_reasoning_api_compatibility(AsyncEngineArgs, AsyncLLMEngine.generate, reasoning_parser)
    if reasoning_parser is not None:
        extra["reasoning_parser"] = reasoning_parser

    # some engine args are newer than our floor or build-specific, so forward them only when this
    # vllm build exposes them.
    engine_arg_names = _async_engine_arg_names(AsyncEngineArgs)
    extra.update(_required_immutable_args(base_model, overrides, engine_arg_names))
    extra.update(_build_specific_args(base_model, overrides, engine_arg_names))

    return {
        "model": served_model,
        "trust_remote_code": cfg.TRUST_REMOTE_CODE,
        "dtype": cfg.DTYPE,
        # FP8 weights: ONLINE E4M3 quant of the bf16 base (default), or auto-detected from a
        # pre-quantized serve_model_id checkpoint (then quant_default is None). `overrides.get`
        # returns the override's value even when it is None, so a base can also opt out to bf16
        # explicitly (the documented 35B H200 fallback). See settings.QUANTIZATION.
        "quantization": overrides.get("quantization", quant_default),
        # FP8 KV cache (E4M3) for every base, including pre-quant checkpoints and explicit bf16
        # fallbacks: ~half the KV VRAM, LoRA/MoE-safe. See settings.KV_CACHE_DTYPE.
        # calculate_kv_scales stays off (default) on purpose.
        "kv_cache_dtype": overrides.get("kv_cache_dtype", cfg.KV_CACHE_DTYPE),
        "tensor_parallel_size": cfg.TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": overrides.get(
            "gpu_memory_utilization", cfg.GPU_MEMORY_UTILIZATION
        ),
        "max_model_len": overrides.get("max_model_len", cfg.MAX_MODEL_LEN),
        # CUDA graphs default ON (disabling them measured -31% throughput), but a per-model
        # override can disable them — the 35B MoE's CUDA-graph capture (specialize_lora over many
        # sizes) costs ~20+ GiB and tips the LoRA-module creation into OOM.
        "enforce_eager": bool(overrides.get("enforce_eager", False)),
        "enable_lora": True,
        "max_loras": max_loras,
        "max_lora_rank": overrides.get("max_lora_rank", cfg.MAX_LORA_RANK),
        "max_cpu_loras": cfg.MAX_CPU_LORAS,
        **extra,
        **cfg.vllm_engine_kwargs(),
    }


def load_engine_config(
    base_model: str,
    settings: Any,
    cfg: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Resolve tokenizer, overrides, and engine kwargs for one engine startup."""
    processor, tokenizer = load_tokenizer(base_model, settings, cfg)
    overrides = engine_overrides_for(base_model)
    kwargs = engine_args_for(base_model, overrides, cfg)
    return processor, tokenizer, overrides, kwargs


def pin_loras_default(overrides: dict[str, Any], cfg: Any) -> bool:
    """Whether adapters should be pinned (non-evictable) for this base model.

    Pinning makes an adapter non-evictable, so it only makes sense when the GPU hot pool covers the
    whole deployable CPU pool. Otherwise >max_loras registered adapters must remain unpinned so
    vLLM can LRU-swap them from max_cpu_loras on demand.
    """
    max_loras = int(overrides.get("max_loras", cfg.MAX_LORAS))
    return bool(overrides.get("pin_loras", max_loras >= cfg.MAX_CPU_LORAS))
