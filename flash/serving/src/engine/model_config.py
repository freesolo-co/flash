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
# every hosted model now serves on B200 (180 GiB, Blackwell sm100): the 9b keeps 16 rank-128
# slots, the 27b keeps 16 rank-64 slots, and the 35b moe keeps 6 rank-64 slots.
#   - Qwen3.6-35B-A3B (vision-language MoE; arch ``Qwen3_5MoeForConditionalGeneration``) -> B200
#     (180 GiB) with the base bf16 weights, 6 x 64 LoRA at 32k. bf16 (not FP8) is the one path giving
#     full-expert LoRA + CUDA graphs because the fused-MoE LoRA path won't compile on fp8e4nv. see the
#     detailed 35B block below.
#
# the shapes below were all measured on the SMALLER cards these tiers used before (L40S 48 GiB, H100
# 80 GiB, H200 141 GiB) and are carried across unchanged, because 180 GiB is a strict superset of each
# fit. that makes the card the only changed variable, so a canary failure is unambiguous. it does NOT
# make the shapes optimal for B200: re-tuning max_loras / max_num_seqs upward to use the extra ~40-130
# GiB is deliberate follow-up work, gated on the canary below.
#
# BLACKWELL PREREQUISITE: vllm 0.23.0 picks its GDN prefill kernel per-arch, and on SM10.x it also
# requires `_is_libs_cu13_install_intact()`. that check fails on a stock resolve and falls back to
# Triton SILENTLY (one `warning_once`), so an unrepaired B200 boots, serves correct output, and bills
# the Blackwell rate while running slower than the H200 it replaced. the serving image repairs the
# cuTe-DSL install for exactly this reason (see `modal_app.py`), and every model here is a Qwen3
# GDN-hybrid, so this applies to ALL of them. the canary must assert the RESOLVED backend is
# flashinfer, never infer it from a successful boot.
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
            # B200 (180 GiB, Blackwell sm100). the prior L40S (48 GiB, Ada sm89) was the cheapest card
            # that fits rank-128 x 16 LoRA at 32k; L4 and 2xL4 OOMed in the real-GPU sweep. B200 is a
            # strict superset of that fit, so the engine knobs below are carried over UNCHANGED and the
            # only variable is the card. keep CUDA graphs on because eager is about 10x slower for this
            # hybrid GatedDeltaNet model, and keep 0.90 for graph-capture headroom.
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 128,  # rank-128 / 16 hot LoRAs (cheap on the 9 GiB FP8 9B); 32k context.
            "max_model_len": 32768,
            # 16, not 8. modal admission is now sized 1:1 to this number (see _engine_concurrency),
            # so this IS the container's throughput. the cost of a slot on this hybrid model is its
            # per-sequence GatedDeltaNet recurrent state, which is allocated per slot and is CONSTANT
            # in context length (~49 MiB/seq across the 24 linear layers, roughly 50x the logits
            # buffer) -- so 8 extra slots is well under a GiB on a 180 GiB B200, against 48 GiB on
            # the L40S this tier used to run. verify on the tier's cold-boot canary: the figure is
            # derived from the published config shapes, not read from vLLM's allocator.
            "max_num_seqs": 16,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    # 35B-A3B MoE: bf16 on a B200 (180 GiB) is the one serving path that gives a flash adapter its
    # full all-expert LoRA and CUDA graphs at speed. it gets rank 64 at 6 hot slots (6 x 64).
    # why bf16 and not the FP8 checkpoint used by every other tier:
    #   * FP8 on A100 materializes the FP8 experts back to bf16 in the fused-MoE LoRA path, leaving no
    #     room for CUDA-graph capture and forcing eager at about 4-10 tok/s.
    #   * FP8 on H200/B200 fails the fused-MoE LoRA kernel with "Unsupported lhs dtype fp8e4nv"; only
    #     the A100's Marlin kernel runs this MoE's full-expert LoRA. the B200 move does NOT revisit
    #     this: the tier stays bf16 for exactly the same reason it did on the H200.
    #   * bf16 sidesteps the FP8 kernel. the H200 canary found that 8 x 64 LoRA plus 32k overflows the
    #     141 GiB card, with only about 19k context fitting; 6 x 64 plus 32k fit cleanly with a
    #     679,701-token KV cache, about 20x concurrency at 32k. the 180 GiB B200 is a strict superset
    #     of that fit, so the knobs are carried over unchanged pending its own canary.
    #   * cold boot is about 17 min (67 gibibytes of weights plus compile, graph capture, and warmup),
    #     so it needs the raised startup_timeout in modal_app. inference or adapter registration starts it.
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
            # cache and graph capture. unchanged from the H200: on the larger B200 it is strictly
            # more headroom, not less.
            "gpu_memory_utilization": 0.90,
            "max_loras": 6,
            "max_lora_rank": 64,
            "pin_loras": False,
            # 32k context at 6 hot rank-64 LoRAs. the real-GPU canary produced a healthy 679,701-token
            # KV cache, about 20x concurrency at 32k; 8 hot LoRAs overflowed and only fit about 19k.
            "max_model_len": 32768,
            "max_num_batched_tokens": 4096,
            # CUDA graphs ON — the whole point. On bf16 the graph capture fits (~0.2-0.8 GiB) and
            # is LoRA-specialized, so adapters serve under graphs too.
            "enforce_eager": False,
            # Startup memory-profiling runs max_num_seqs sequences; cap low so the 248k-vocab logits +
            # all-expert MoE activations don't spike the profiling peak.
            # HELD AT 8 while the 9B and 27B go to 16. this is the one tier whose profiling peak is a
            # documented OOM hazard (see boot.py), and at p99 it is KV-bound rather than state-bound
            # (~85 MiB/seq of KV at an 8,712-token p99, on top of ~61 MiB/seq of recurrent state), so
            # raising it is not the sub-GiB change it is on the 9B. it needs its own real-GPU canary
            # before it moves; do not raise it on the strength of the other two tiers.
            "max_num_seqs": 8,
            "reasoning_parser": "qwen3",
            # NB: the vision encoder is now LOADED (no language_model_only) — flash adapters adapt the
            # full multimodal tree, so their vision-tower LoRA keys must have real modules to bind to.
            # this adds the vision encoder's weights on top of the already weight-bound 6 x 64 LoRA buffer.
            # the complete model and LoRA load is about 108 GiB, measured on the 141 GiB H200 and
            # carried to the 180 GiB B200.
        },
    },
    # 27B dense, now on a B200 (180 GiB). the H100 (80 GiB) real-GPU canary measured the load at 44.25 GiB -- above the
    # FP8 weight size alone, because the vision tower and the non-quantized tensors stay bf16 -- which
    # still leaves 23.07 GiB of KV cache (350,981 tokens, 10.71x concurrency at 32k) after the 16
    # rank-64 LoRA buffers and a 0.35 GiB graph capture. every repository here is pinned to an
    # immutable revision so a served engine cannot silently follow an upstream retag; note the model
    # pin names a commit in the -FP8 repo while the tokenizer/processor pins name one in the base
    # repo, which are separate sha namespaces.
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
            # 16, matching the 9B: dense (no MoE activation spike) and the B200 more than doubles the
            # 80 GiB H100 this was canaried on, where 23.07 GiB of KV already gave 10.71x concurrency
            # at 32k. confirm on the B200 canary.
            "max_num_seqs": 16,
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
