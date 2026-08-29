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
# the active 9b uses 16 rank-128 slots on l40s, the active 27b uses 16 rank-64 slots on h100, and the
# active 35b moe uses 6 rank-64 slots on h200.
#   - Qwen3.6-35B-A3B (vision-language MoE; arch ``Qwen3_5MoeForConditionalGeneration``) -> H200
#     (141 GiB) with the base bf16 weights, 6 x 64 LoRA at 32k. bf16 (not FP8) is the one path giving
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
        "gpu": "L40S",
        "engine": {
            # the L40S (48 GiB, Ada sm89) is the cheapest Modal card that fits rank-128 x 16 LoRA
            # at 32k; L4 and 2xL4 OOMed in the real-GPU sweep. keep CUDA graphs on because eager is
            # about 10x slower for this hybrid GatedDeltaNet model, and keep 0.90 for graph-capture headroom.
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 128,  # rank-128 / 16 hot LoRAs (cheap on the 9 GiB FP8 9B); 32k context.
            "max_model_len": 32768,
            # 16 decode slots, matching max_inputs below so Modal never packs a request this engine
            # cannot decode. this is a HYBRID GatedDeltaNet (24 of 32 layers linear), so each slot
            # costs a ~49 MiB recurrent+conv state that is CONSTANT in context length rather than the
            # ~1 MiB logits share a dense model would charge. 8 -> 16 costs ~0.38 GiB (0.9% of the
            # 43.2 GiB budget), taken out of the KV pool: ~25k tokens, about 62 requests of headroom
            # at this tier's measured 407-token p99 (1.2% of its window; 71,453 real requests). the
            # queueing it removes was measured twice on live traffic: per-request throughput is flat
            # at ~43.7 tok/s up to ~15 concurrent, then collapses to ~20.5 (2.13x) by 30-33.
            # NOT free beyond that state: max_num_seqs also raises vLLM's CUDA-graph capture ceiling
            # (min(seqs*2, 512, max_num_batched_tokens)), so the ladder goes 5 -> 7 rungs and, with
            # LoRA specialization doubling each rung, 10 -> 14 captures. small, but it is why this
            # needs a real-GPU cold boot to confirm rather than arithmetic alone.
            "max_num_seqs": 16,
            # logical REQUESTS Modal packs per container (see _engine_concurrency). equal to
            # max_num_seqs so an n=1 burst is fully decodable; n>1 still oversubscribes by design.
            "max_inputs": 16,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    # 35B-A3B MoE: bf16 on an H200 (141 GiB) is the one serving path that gives a flash adapter its
    # full all-expert LoRA and CUDA graphs at speed. it gets rank 64 at 6 hot slots (6 x 64).
    # why bf16/H200 and not the FP8 checkpoint used by every other tier:
    #   * FP8 on A100 materializes the FP8 experts back to bf16 in the fused-MoE LoRA path, leaving no
    #     room for CUDA-graph capture and forcing eager at about 4-10 tok/s.
    #   * FP8 on H200/B200 fails the fused-MoE LoRA kernel with "Unsupported lhs dtype fp8e4nv"; only
    #     the A100's Marlin kernel runs this MoE's full-expert LoRA.
    #   * bf16 on H200 sidesteps the FP8 kernel. the real-GPU canary found that 8 x 64 LoRA plus 32k
    #     overflows the 141 GiB card, with only about 19k context fitting. 6 x 64 plus 32k fits cleanly
    #     with a 679,701-token KV cache, about 20x concurrency at 32k.
    #   * cold boot is about 17 min (67 gibibytes of weights plus compile, graph capture, and warmup),
    #     so it needs the raised startup_timeout in modal_app. inference or adapter registration starts it.
    {
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "image_input_limit": 4,
        "gpu": "H200",
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
            # 32k context at 6 hot rank-64 LoRAs. the real-GPU canary produced a healthy 679,701-token
            # KV cache, about 20x concurrency at 32k; 8 hot LoRAs overflowed and only fit about 19k.
            "max_model_len": 32768,
            "max_num_batched_tokens": 4096,
            # CUDA graphs ON — the whole point. On bf16/H200 the graph capture fits (~0.2-0.8 GiB) and
            # is LoRA-specialized, so adapters serve under graphs too.
            "enforce_eager": False,
            # Startup memory-profiling runs max_num_seqs sequences; cap low so the 248k-vocab logits +
            # all-expert MoE activations don't spike the profiling peak. this is the tier the
            # documented profiling OOM actually came from, so it stays at 8 until a real-GPU boot
            # confirms the peak at a higher cap — the arithmetic says a slot costs only ~61 MiB
            # (0.38% of budget), but that arithmetic does not model the all-expert activation spike.
            # a slot here also costs far more KV than the 9B's: this tier's measured p99 total is
            # 8,712 tokens (26.6% of its 32k window, driven by completions hitting an 8k generation
            # cap) versus the 9B's 407, so at p99 its ~85 MiB of KV OUTWEIGHS the recurrent state.
            # size this tier against KV, not against the 9B's state-bound answer.
            "max_num_seqs": 8,
            # logical REQUESTS per container, held equal to max_num_seqs so Modal autoscales instead
            # of queueing inside an engine that decodes 8 (see _engine_concurrency). PREVENTIVE on
            # this tier, not corrective: per-replica telemetry shows offered load never reached 9+
            # here, so the surplus admission slots were never exercised. the measured degradation is
            # on the 9B/4B; this removes the same hazard where a container is most expensive.
            "max_inputs": 8,
            "reasoning_parser": "qwen3",
            # NB: the vision encoder is now LOADED (no language_model_only) — flash adapters adapt the
            # full multimodal tree, so their vision-tower LoRA keys must have real modules to bind to.
            # this adds the vision encoder's weights on top of the already weight-bound 6 x 64 LoRA buffer.
            # the complete model and LoRA load is about 108 GiB on the 141 GiB H200.
        },
    },
    # 27B dense on an H100 (80 GiB). the real-GPU canary measured the load at 44.25 GiB -- above the
    # FP8 weight size alone, because the vision tower and the non-quantized tensors stay bf16 -- which
    # still leaves 23.07 GiB of KV cache (350,981 tokens, 10.71x concurrency at 32k) after the 16
    # rank-64 LoRA buffers and a 0.35 GiB graph capture. every repository here is pinned to an
    # immutable revision so a served engine cannot silently follow an upstream retag; note the model
    # pin names a commit in the -FP8 repo while the tokenizer/processor pins name one in the base
    # repo, which are separate sha namespaces.
    {
        "base_model": "Qwen/Qwen3.8-27B",
        "image_input_limit": 4,
        "gpu": "H100",
        "engine": {
            "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "tokenizer_model": "Qwen/Qwen3.8-27B",
            "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            # logical REQUESTS per container, held equal to max_num_seqs so Modal autoscales rather
            # than queueing the surplus inside an engine that decodes 8 (see _engine_concurrency).
            # this tier was activated by #1333 and has never been load-measured, so it keeps the
            # conservative 1:1 admission until the capacity sweep gives it a measured value.
            "max_inputs": 8,
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


def reasoning_parser_for(base_model: str) -> str | None:
    """the model-scoped vLLM reasoning parser, or None when parsing is disabled."""
    parser = (_config_for(base_model).get("engine") or {}).get("reasoning_parser")
    return str(parser) if parser else None
