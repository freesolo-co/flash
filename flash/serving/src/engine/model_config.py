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

from flash.content.image_descriptors import MAX_IMAGES_PER_CONVERSATION
from flash.serve.request.tool_calls import qualified_tool_parser
from flash.serving.deepinfer.contract.config import GPU as DEEPSEEK_GPU
from flash.serving.deepinfer.contract.config import MODEL_ID as DEEPSEEK_MODEL_ID
from flash.serving.deepinfer.contract.config import MODEL_REVISION as DEEPSEEK_MODEL_REVISION
from flash.serving.src.engine.prequant_config import (
    fp8_serve_model_for as _prequant_serve_model_for,
)

# base_model -> serving config. Mirrors flash/flash/catalog.py (the flash submodule).
#
# `gpu` (required): the Modal GPU class to run this cataloged base model's engine on. Routing/adapters
# key off the logical `base_model`; only the engine's GPU class changes. The serving app builds one
# `LoraEngine` Modal class per distinct GPU value and dispatches each base model to its tier.
# every entry declares its GPU; uncataloged base models are rejected instead of
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
# The dense small models use L4, the FP8 9B uses L40S, the FP8 27B uses A100 80GB, and
# the BF16 35B MoE uses H200. See the hosted GPU qualification table in app/README.md.
#   - Qwen3.6-35B-A3B (vision-language MoE; arch ``Qwen3_5MoeForConditionalGeneration``) -> H200
#     (141 GB) with the base bf16 weights, 6 x 64 LoRA at native 256k. bf16 (not FP8) is the one path giving
#     full-expert LoRA + CUDA graphs because the fused-MoE LoRA path won't compile on fp8e4nv. see the
#     detailed 35B block below.
#
# NOTE: every new tier/shape needs a one-time real-GPU cold-boot smoke test with the serving canary.
# Training GPU validation is separate; this file is only the serving vLLM engine matrix.

SERVING_MODELS: list[dict[str, Any]] = [
    {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "image_input_limit": None,
        "gpu": "L4",
        "engine": {
            "serve_model_id": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "tokenizer_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "processor_revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "quantization": None,
            "gpu_memory_utilization": 0.90,
            "max_loras": 4,
            "max_cpu_loras": 4,
            "max_lora_rank": 64,
            "max_model_len": 262144,
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    {
        "base_model": "Qwen/Qwen3.5-4B",
        "image_input_limit": None,
        "gpu": "L4",
        "engine": {
            "serve_model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "processor_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "quantization": None,
            "gpu_memory_utilization": 0.90,
            "max_loras": 4,
            "max_cpu_loras": 4,
            "max_lora_rank": 64,
            "max_model_len": 262144,
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    {
        "base_model": "Qwen/Qwen3.5-9B",
        "image_input_limit": MAX_IMAGES_PER_CONVERSATION,
        "gpu": "L40S",
        "engine": {
            "tokenizer_model": "Qwen/Qwen3.5-9B",
            "tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "processor_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            # Use the checkpoint's native 262,144-token context, including generated output.
            # The former B200's KV capacity measurements do not apply to this smaller GPU.
            # keep CUDA graphs on because eager is about 10x slower for this hybrid GatedDeltaNet
            # model, and keep 0.90 for graph-capture headroom.
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 128,  # rank-128 / 16 hot LoRAs (cheap on the 9 GiB FP8 9B); native 256k context.
            "max_model_len": 262144,
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    # 35B-A3B MoE: bf16 on an H200 (141 GB) is the one serving path that gives a flash adapter its
    # full all-expert LoRA and CUDA graphs at speed. it gets rank 64 at 6 hot slots (6 x 64).
    # why bf16 and not the FP8 checkpoint used by every other tier:
    #   * FP8 on A100 materializes the FP8 experts back to bf16 in the fused-MoE LoRA path, leaving no
    #     room for CUDA-graph capture and forcing eager at about 4-10 tok/s.
    #   * FP8 on H200/B200 fails the fused-MoE LoRA kernel with "Unsupported lhs dtype fp8e4nv"; only
    #     the A100's Marlin kernel runs this MoE's full-expert LoRA.
    #   * bf16 sidesteps the FP8 kernel. on the 141 GB H200 this tier used to run on, 8 x 64 LoRA
    #     plus 32k overflowed the card (only about 19k context fit); six slots preserve the
    #     original 32k context and eight-sequence admission on H200.
    #   * H200 qualification loaded 108.46 GiB for the model and configured LoRA buffers.
    #     Smaller 80/96 GB cards cannot fit that allocation; inference or adapter registration
    #     starts the engine within the raised startup_timeout in modal_app.
    {
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "image_input_limit": MAX_IMAGES_PER_CONVERSATION,
        "gpu": "H200",
        "engine": {
            "tokenizer_model": "Qwen/Qwen3.6-35B-A3B",
            "tokenizer_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
            "processor_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
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
            # Native 256k context with six hot rank-64 LoRAs; long requests share KV capacity.
            "max_model_len": 262144,
            "max_num_batched_tokens": 4096,
            # CUDA graphs remain enabled and LoRA-specialized on H200.
            "enforce_eager": False,
            # Startup memory-profiling runs max_num_seqs sequences; cap low so the 248k-vocab logits +
            # all-expert MoE activations don't spike the profiling peak.
            "max_num_seqs": 8,
            "reasoning_parser": "qwen3",
            # NB: the vision encoder is now LOADED (no language_model_only) — flash adapters adapt the
            # full multimodal tree, so their vision-tower LoRA keys must have real modules to bind to.
            # this adds the vision encoder's weights on top of the already weight-bound 6 x 64 LoRA buffer.
            # the complete model and LoRA load is about 108 GiB, on the 141 GB H200.
        },
    },
    # 27B dense on A100 80GB. Qualification loaded 45.53 GiB with 16 rank-64 LoRA
    # slots; L40S ran out of memory while allocating those buffers. The vision tower and
    # non-quantized tensors remain bf16. Every
    # repository here is pinned to an immutable revision so a served engine cannot silently follow an
    # upstream retag; note the model pin names a commit in the -FP8 repo while the tokenizer/processor
    # pins name one in the base repo, which are separate sha namespaces.
    {
        "base_model": "Qwen/Qwen3.8-27B",
        "image_input_limit": MAX_IMAGES_PER_CONVERSATION,
        "gpu": "A100-80GB",
        "engine": {
            "model_revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "tokenizer_model": "Qwen/Qwen3.8-27B",
            "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "processor_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 262144,
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
]

# Native base-only backend is registered independently of the vLLM/LoRA matrix.
# The router advertises it only when the authenticated backend connection is configured.
_BY_MODEL: dict[str, dict[str, Any]] = {m["base_model"]: m for m in SERVING_MODELS}
_BY_MODEL[DEEPSEEK_MODEL_ID] = {"gpu": DEEPSEEK_GPU, "image_input_limit": None}


def hosted_base_models() -> list[str]:
    from flash.serving.deepinfer.app.client import configured

    return base_models() + ([DEEPSEEK_MODEL_ID] if configured() else [])


def base_models() -> list[str]:
    return [model["base_model"] for model in SERVING_MODELS]


def is_supported_base_model(base_model: str) -> bool:
    return base_model in _BY_MODEL


def supports_lora(base_model: str) -> bool:
    """Only models in the vLLM engine matrix can register LoRA checkpoints."""
    return base_model in base_models()


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
    return _config_for(base_model)["gpu"]


def serve_model_for(base_model: str) -> str:
    """return the pre-quantized checkpoint an active hosted engine loads."""
    _config_for(base_model)
    if base_model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"}:
        return base_model
    if base_model == DEEPSEEK_MODEL_ID:
        return DEEPSEEK_MODEL_ID
    return _prequant_serve_model_for(base_model)


def tokenizer_model_for(base_model: str) -> str:
    """return the logical tokenizer and processor repository for a hosted engine."""
    engine = _config_for(base_model).get("engine") or {}
    return str(engine.get("tokenizer_model") or base_model)


def immutable_serving_revisions(base_model: str) -> dict[str, str]:
    """return model/tokenizer/processor pins required by this hosted engine."""
    if base_model == DEEPSEEK_MODEL_ID:
        return {
            "model_revision": DEEPSEEK_MODEL_REVISION,
            "tokenizer_revision": DEEPSEEK_MODEL_REVISION,
        }
    engine = _config_for(base_model).get("engine") or {}
    return {
        key: str(engine[key])
        for key in ("model_revision", "tokenizer_revision", "processor_revision")
        if engine.get(key)
    }


def engine_overrides_for(base_model: str) -> dict[str, Any]:
    """return vLLM overrides for an active hosted engine."""
    config = _config_for(base_model)
    if base_model == DEEPSEEK_MODEL_ID:
        raise ValueError("DeepInfer base model cannot be loaded by the vLLM LoRA engine")
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
