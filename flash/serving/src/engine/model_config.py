"""Per-base-model serving config for the autoslm catalog.

The serving app runs one vLLM GPU engine per base model, each on the Modal GPU class set by its
catalog ``gpu`` (see below). Adapters and routing key off the logical ``base_model``; the engine
loads its configured checkpoint at that checkpoint's own dtype. The active dense 9B uses Freesolo
compressed-tensors FP8 and the active dense 27B uses the official Qwen FP8 checkpoint. The active
35B-A3B MoE serves base bf16 weights because its fused-MoE LoRA path will not compile on FP8, as
detailed below.
"""

from __future__ import annotations

from typing import Any

from flash.serve.request.tool_calls import qualified_tool_parser
from flash.serving.src.engine.prequant_config import (
    fp8_serve_model_for as _prequant_serve_model_for,
)

# base_model -> serving config. Mirrors flash/flash/catalog.py (the flash submodule).
#
# `gpu` (optional): the Modal GPU class to run this cataloged base model's engine on. Routing/adapters
# key off the logical `base_model`; only the engine's GPU class changes. The serving app builds one
# `LoraEngine` Modal class per distinct GPU value and dispatches each base model to its tier. Catalog
# entries with no `gpu` fall back to ``DEFAULT_GPU``; uncataloged base models are rejected instead of
# silently using a small default tier. ``TENSOR_PARALLEL_SIZE`` stays 1 (one card per engine).
#
# `engine` (optional): per-model vLLM engine-arg overrides (LoRA buffer shape, scheduler/memory caps,
# language_model_only, …). The engine's LOADED checkpoint is NOT in here — it is resolved centrally by
# ``serve_model_for`` from ``src.prequant_config`` and injected into the overrides as ``serve_model_id``.
# active models resolve through prequant_config unless an exact validated override is present.
# ⚠ serve_model_id is pointed only at checkpoints VERIFIED to exist (a missing repo 404-crash-loops
# the engine — the reason this mechanism was removed once); the owned repos are VL-preserving FP8
# checkpoints published to the operator HF org.
#
# sizing rationale. preallocated lora buffers and loaded checkpoint weights dominate engine vram.
# all three active tiers serve on b200: the 9b uses 16 rank-128 slots, the 27b 16 rank-64 slots, and
# the 35b moe 6 rank-64 slots.
#   - Qwen3.6-35B-A3B (vision-language MoE; arch ``Qwen3_5MoeForConditionalGeneration``) -> B200
#     (178 GiB) with the base bf16 weights, 6 x 64 LoRA at 32k. bf16 (not FP8) is the one path giving
#     full-expert LoRA + CUDA graphs because the fused-MoE LoRA path won't compile on fp8e4nv. see the
#     detailed 35B block below.
#
# NOTE: every new tier/shape needs a one-time real-GPU cold-boot smoke test with the serving canary.
# Training GPU validation is separate; this file is only the serving vLLM engine matrix.
DEFAULT_GPU = "L4"

SERVING_MODELS: list[dict[str, Any]] = [
    {
        "base_model": "Qwen/Qwen3.5-9B",
        "image_input_limit": 4,
        "gpu": "B200",
        "engine": {
            # measured on B200 (2026-08-31): 132.87 GiB of KV, 8,713,056 tokens, 217x concurrency at
            # 32k with rank-128 x 16 LoRA. this tier previously ran on the L40S (48 GiB, Ada sm89),
            # the cheapest card that fit the LoRA buffer; L4 and 2xL4 OOMed in the real-GPU sweep.
            # keep CUDA graphs on because eager is about 10x slower for this hybrid GatedDeltaNet
            # model, and keep 0.90 for graph-capture headroom.
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 128,  # rank-128 / 16 hot LoRAs (cheap on the 9 GiB FP8 9B); 32k context.
            "max_model_len": 32768,
            # 32, not the 8 inherited from the L40S, which left most of the B200 idle. a real sweep
            # (2026-08-31, one boot per value) measured container throughput at each cap's OWN
            # capacity: 2165 t/s at 8, 3307 at 16, 4793 at 32 -- 2.2x -- with p50 TTFT flat
            # (0.054s -> 0.175s). 32 is the KNEE, not merely the largest tried: headroom above the
            # cap collapses from +16.9% at 16 to +2.2% at 32, so 64 would buy ~nothing. no memory
            # cost (kv pool moves 0.07% across the range, free vram flat at ~18.4 GiB). the one
            # cost is the first boot after the change: max_num_seqs is part of the vllm compile
            # cache key, so it recompiles once (169s -> 254s cache-warm on this tier).
            "max_num_seqs": 32,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    # 35B-A3B MoE: bf16 on a B200 (178 GiB) is the one serving path that gives a flash adapter its
    # full all-expert LoRA and CUDA graphs at speed. it gets rank 64 at 6 hot slots (6 x 64).
    # why bf16 and not the FP8 checkpoint used by every other tier:
    #   * FP8 on A100 materializes the FP8 experts back to bf16 in the fused-MoE LoRA path, leaving no
    #     room for CUDA-graph capture and forcing eager at about 4-10 tok/s.
    #   * FP8 on H200/B200 fails the fused-MoE LoRA kernel with "Unsupported lhs dtype fp8e4nv"; only
    #     the A100's Marlin kernel runs this MoE's full-expert LoRA.
    #   * bf16 sidesteps the FP8 kernel. on the 141 GiB H200 this tier used to run on, 8 x 64 LoRA
    #     plus 32k overflowed the card (only about 19k context fit) and 6 x 64 measured about 20x
    #     concurrency; the 178 GiB B200 measured 112x at the same 6 x 64 shape.
    #   * cold boot measured 488s on the B200 (67 gibibytes of weights plus compile, graph capture
    #     and warmup), so it needs the raised startup_timeout in modal_app. inference or adapter
    #     registration starts it.
    {
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "image_input_limit": 4,
        "gpu": "B200",
        "engine": {
            # Load the BASE bf16 weights, NOT the FP8 checkpoint — bf16 is what lets full-expert LoRA
            # + graphs coexist (see the block comment above). serve_model_id overrides the FP8 default
            # injected by engine_overrides_for; quantization=None keeps it bf16 (no online quant).
            "serve_model_id": "Qwen/Qwen3.6-35B-A3B",
            "quantization": None,
            # 0.90 (not 0.98) leaves headroom above the roughly 108 GiB model and LoRA load for KV
            # cache and graph capture.
            "gpu_memory_utilization": 0.90,
            "max_loras": 6,
            "max_lora_rank": 64,
            "pin_loras": False,
            # 32k context at 6 hot rank-64 LoRAs. measured on B200 (2026-08-30): 49.38 GiB of KV,
            # 5,177,120 tokens, 112x concurrency at 32k -- the 178 GiB card leaves far more room than
            # the 141 GiB H200 this tier previously ran on (which measured about 20x).
            "max_model_len": 32768,
            "max_num_batched_tokens": 4096,
            # CUDA graphs ON — the whole point. On bf16/B200 the graph capture fits (measured 0.38 GiB)
            # and is LoRA-specialized, so adapters serve under graphs too.
            "enforce_eager": False,
            # Startup memory-profiling runs max_num_seqs sequences, so this was held at 8 out of
            # concern that the 248k-vocab logits + all-expert MoE activations would spike the
            # profiling peak. Measured on B200 (2026-08-31) that concern does not bind here: booting
            # and loading at 32 showed no OOM or preemption, free vram moved 18.11 -> 17.77 GiB and
            # the kv pool 0.24% across 8/16/32. Container throughput at each cap's OWN capacity was
            # 721 t/s at 8, 1200 at 16, 3196 at 32 -- 4.4x, with the 16->32 step (+166.3%) the
            # largest in the sweep -- and p50 TTFT IMPROVES at 32 (0.553s -> 0.463s), because past
            # the cap requests queue instead of running. 32 is the ceiling regardless: the earlier
            # B200/B300 sweep saw a 4x regression and a hang at 64.
            "max_num_seqs": 32,
            "reasoning_parser": "qwen3",
            # NB: the vision encoder is now LOADED (no language_model_only) — flash adapters adapt the
            # full multimodal tree, so their vision-tower LoRA keys must have real modules to bind to.
            # this adds the vision encoder's weights on top of the already weight-bound 6 x 64 LoRA buffer.
            # the complete model and LoRA load is about 108 GiB, on the 178 GiB B200.
        },
    },
    # 27B dense on a B200 (178 GiB). the load is 44.25 GiB -- above the FP8 weight size alone, because
    # the vision tower and the non-quantized tensors stay bf16. measured on B200 (2026-08-30):
    # 112.66 GiB of KV cache, 3,691,072 tokens, 87x concurrency at 32k, after the 16 rank-64 LoRA
    # buffers and graph capture; the 80 GiB H100 this tier previously ran on measured 10.71x. every
    # repository here is pinned to an immutable revision so a served engine cannot silently follow an
    # upstream retag; note the model pin names a commit in the -FP8 repo while the tokenizer/processor
    # pins name one in the base repo, which are separate sha namespaces.
    {
        "base_model": "Qwen/Qwen3.8-27B",
        "image_input_limit": 4,
        "gpu": "B200",
        "engine": {
            "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "tokenizer_model": "Qwen/Qwen3.8-27B",
            "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            # 32, not the 8 inherited from the H100. measured on B200 (2026-08-31), container
            # throughput at each cap's OWN capacity: 721 t/s at 8, 1110 at 16, 1899 at 32 -- 2.6x.
            # the 16->32 step (+71.1%) is the largest in the sweep, so low headroom above a lower
            # cap does NOT mean the next cap is unhelpful: past the cap requests queue instead of
            # running. costs no memory: the kv pool moves 0.13% across the range and free vram
            # stays flat at ~18.2 GiB. changing this value invalidates the vllm compile cache
            # (max_num_seqs is part of its key), so the FIRST boot after this change pays a
            # one-time torch.compile -- 430s of the 948s cold boot measured at 32, against 202s
            # for the cache-warm boot at 16. both are far inside STARTUP_TIMEOUT_SECONDS (2700s).
            "max_num_seqs": 32,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
]

_BY_MODEL: dict[str, dict[str, Any]] = {m["base_model"]: m for m in SERVING_MODELS}


def base_models() -> list[str]:
    return [model["base_model"] for model in SERVING_MODELS]


def is_supported_base_model(base_model: str) -> bool:
    return base_model in _BY_MODEL


def _config_for(base_model: str) -> dict[str, Any]:
    cfg = _BY_MODEL.get(base_model)
    if cfg is None:
        allowed = ", ".join(base_models())
        raise ValueError(
            f"Unsupported base model {base_model!r}; add it to hosted serving only after a "
            f"real-GPU serving canary. Supported base models: {allowed}"
        )
    return cfg


def supports_image_input(base_model: str) -> bool:
    return image_limit_for(base_model) is not None


def image_limit_for(base_model: str) -> int | None:
    return _config_for(base_model)["image_input_limit"]


def gpu_for(base_model: str) -> str:
    """return the Modal GPU class for an active hosted engine."""
    return _config_for(base_model).get("gpu") or DEFAULT_GPU


def serve_model_for(base_model: str) -> str:
    """return the pre-quantized checkpoint an active hosted engine loads."""
    _config_for(base_model)
    return _prequant_serve_model_for(base_model)


def tokenizer_model_for(base_model: str) -> str:
    """return the logical tokenizer and processor repository for a hosted engine."""
    engine = _config_for(base_model).get("engine") or {}
    return str(engine.get("tokenizer_model") or base_model)


def immutable_serving_revisions(base_model: str) -> dict[str, str]:
    """return model/tokenizer/processor pins required by this hosted engine."""
    engine = _config_for(base_model).get("engine") or {}
    return {
        key: str(engine[key])
        for key in ("model_revision", "tokenizer_revision", "processor_revision")
        if engine.get(key)
    }


def engine_overrides_for(base_model: str) -> dict[str, Any]:
    """return vLLM overrides for an active hosted engine."""
    config = _config_for(base_model)
    overrides = dict(config.get("engine") or {})
    if "serve_model_id" not in overrides:
        overrides["serve_model_id"] = serve_model_for(base_model)
    return overrides


def tool_parser_for(base_model: str) -> str | None:
    """return the qualified flash-owned output parser for one exact hosted base."""

    _config_for(base_model)
    return qualified_tool_parser(base_model)


def reasoning_parser_for(base_model: str) -> str | None:
    """the model-scoped vLLM reasoning parser, or None when parsing is disabled."""
    parser = (_config_for(base_model).get("engine") or {}).get("reasoning_parser")
    return str(parser) if parser else None
