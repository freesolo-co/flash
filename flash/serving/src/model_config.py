"""Per-base-model serving config for the autoslm catalog.

The serving app runs one vLLM GPU engine per base model, each on the Modal GPU class set by its
catalog ``gpu`` (see below). Adapters and routing key off the logical ``base_model``; the engine
loads a PRE-QUANTIZED FP8 checkpoint for that model (see ``src.prequant_config``) at the checkpoint's
own dtype. Every DENSE catalog model serves a pre-quantized FP8 checkpoint (no online quantization, no
community-repo dependence); vLLM auto-detects the checkpoint's compressed-tensors quantization, so
the engine passes no online ``quantization``. The 35B-A3B MoE is the exception: it serves the BASE
bf16 weights (the fused-MoE LoRA path won't compile on FP8), see the 35B block below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from flash.serving.src.prequant_config import fp8_serve_model_for as _prequant_serve_model_for

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
# so every model loads a pre-quantized FP8 checkpoint: Freesolo-owned FP8_DYNAMIC for the dense
# models (including the 27B) and official Qwen FP8 for the 35B VL MoE.
# ⚠ serve_model_id is pointed only at checkpoints VERIFIED to exist (a missing repo 404-crash-loops
# the engine — the reason this mechanism was removed once); the owned repos are VL-preserving FP8
# checkpoints published to the operator HF org.
#
# Sizing rationale. Two things dominate per-engine VRAM:
#   1. The PRE-ALLOCATED LoRA buffers (max_loras x max_lora_rank, fixed at init, independent of how
#      many adapters load) — they can DWARF the (now-quantized) base weights. Both are linear levers.
#      the dense tiers keep max_loras=16 (the global default) and move the rank: rank-128 for
#      0.8B/2B/4B/9B, rank-64 for 27B. the 35B MoE is the exception; its fused-MoE LoRA buffer is far
#      larger per (lora, rank), so it runs rank-64 at 6 hot slots.
#   2. Quantized base weights (FP8 ~half bf16) + FP8 KV (~half). Loaded directly from the
#      pre-quantized serve_model_id checkpoint (no bf16 load transient).
# Net result across native-FP8 cards (compute capability >= 8.9) and A100's Marlin FP8 fallback:
#   - 0.8B / 2B -> L4 (24 GiB, Ada sm89, ~$0.80/hr — the cheapest vLLM-capable card on
#     Modal), owned pre-quantized checkpoints, 16 x 128 LoRA, at gpu_memory_utilization 0.98 (same as
#     the 4B on this card). The FP8 weights + LoRA buffer occupy only a fraction of the 24 GiB, so
#     the small tiers previously inherited the global 0.90 and left ~2 GiB idle; pinning 0.98 turns
#     that headroom into KV-cache blocks (more concurrent sequences / longer prefix-cache residency).
#     They CANNOT drop to a smaller/cheaper card: T4 is the only cheaper Modal GPU and is NOT an
#     option — sm75 (Turing) is below vLLM V1's compute-capability >= 8.0 floor, so vLLM >= 0.19 won't
#     initialize (no V1 attention backend for Turing); the next card up, A10G, costs MORE than the L4.
#   - 4B -> L4 with an owned PRE-QUANTIZED checkpoint, max_model_len=32768, max_num_seqs=8, and
#     16 x 128 LoRA. CUDA graphs remain enabled.
#   - 9B -> L40S (48 GiB, Ada sm89) at 32k context with CUDA graphs on and 16 x 128 LoRA.
#   - 27B -> H100 (80 GiB) at 32k context with CUDA graphs on and 16 x 64 LoRA.
#   - Qwen3.6-35B-A3B (vision-language MoE; arch ``Qwen3_5MoeForConditionalGeneration``) -> H200
#     (141 GiB) with the base bf16 weights, 6 x 64 LoRA at 32k. bf16 (not FP8) is the one path giving
#     full-expert LoRA + CUDA graphs because the fused-MoE LoRA path won't compile on fp8e4nv. see the
#     detailed 35B block below.
#
# NOTE: every new tier/shape needs a one-time real-GPU cold-boot smoke test with the serving canary.
# Training GPU validation is separate; this file is only the serving vLLM engine matrix.
DEFAULT_GPU = "L4"


@dataclass(frozen=True, slots=True)
class HostedTrafficPolicy:
    """Validated per-model limits for hosted request admission and container scaling."""

    min_containers: int
    max_containers: int
    buffer_containers: int
    queue_capacity: int
    retry_after_seconds: int
    max_num_seqs: int
    max_inputs: int
    target_inputs: int

    @classmethod
    def from_engine(cls, engine: Mapping[str, Any]) -> Self:
        max_num_seqs = engine.get("max_num_seqs")
        if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int):
            raise ValueError("hosted traffic policy requires an explicit positive max_num_seqs")
        if max_num_seqs <= 0:
            raise ValueError("hosted traffic policy requires an explicit positive max_num_seqs")
        return cls(
            min_containers=1,
            max_containers=2,
            buffer_containers=0,
            queue_capacity=2,
            retry_after_seconds=1,
            max_num_seqs=max_num_seqs,
            max_inputs=max_num_seqs,
            target_inputs=max(1, max_num_seqs * 3 // 4),
        )

    def __post_init__(self) -> None:
        values = (
            self.min_containers,
            self.max_containers,
            self.buffer_containers,
            self.queue_capacity,
            self.retry_after_seconds,
            self.max_num_seqs,
            self.max_inputs,
            self.target_inputs,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("hosted traffic policy values must be integers")
        if self.max_num_seqs <= 0:
            raise ValueError("hosted traffic max_num_seqs must be positive")
        if self.max_inputs != self.max_num_seqs:
            raise ValueError("hosted traffic max_inputs must equal max_num_seqs")
        if self.target_inputs != max(1, self.max_num_seqs * 3 // 4):
            raise ValueError("hosted traffic target_inputs must equal 75 percent of max_num_seqs")
        if self.min_containers != 1 or self.max_containers != 2:
            raise ValueError("hosted traffic container limits must be exactly one warm and two max")
        if self.buffer_containers != 0:
            raise ValueError("hosted traffic buffer_containers must be zero")
        if self.queue_capacity != 2:
            raise ValueError("hosted traffic queue_capacity must be exactly two")
        if self.retry_after_seconds != 1:
            raise ValueError("hosted traffic retry_after_seconds must be exactly one")


SERVING_MODELS: list[dict[str, Any]] = [
    # Small L4 tiers: owned pre-quantized FP8 checkpoints with rank-128 adapters, keeping 16 hot LoRAs
    # resident (the global default). The loaded checkpoint is resolved centrally (serve_model_for) —
    # the engine dict carries only the LoRA/tier shape. These tiny models have ample L4 headroom, so
    # the 16 x 128 buffer fits comfortably; they now pin gpu_memory_utilization=0.98 (matching the
    # 4B on the same L4) instead of inheriting the global 0.90, so the ~2 GiB the default left idle
    # becomes KV cache — more concurrent sequences / longer prefix-cache residency per engine.
    # They also cap max_num_seqs=64 to the container's real concurrency ceiling (modal_app.MAX_INPUTS
    # packs at most 64 in-flight requests per engine). Left at vLLM's ~1024 default the engine
    # over-reserved logits/activation + captured CUDA graphs for ~1000 sequences that can never
    # arrive; capping to 64 reclaims that reservation as KV cache (thoroughly-used memory) and
    # shortens cold-boot, with NO throughput loss — all 64 in-flight requests still decode at once.
    {
        "base_model": "Qwen/Qwen3.5-0.8B",
        "image_input_limit": 4,
        "gpu": "L4",
        "engine": {
            "gpu_memory_utilization": 0.98,
            "max_lora_rank": 128,
            "max_model_len": 32768,
            "max_num_seqs": 64,
            # CUDA graphs ON (explicit; was the vLLM default) — documents intent, de-risks a default change.
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    {
        "base_model": "Qwen/Qwen3.5-2B",
        "image_input_limit": 4,
        "gpu": "L4",
        "engine": {
            "gpu_memory_utilization": 0.98,
            "max_lora_rank": 128,
            "max_model_len": 32768,
            "max_num_seqs": 64,
            # CUDA graphs ON (explicit; was the vLLM default) — documents intent, de-risks a default change.
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    # the 4B and 9B use rank-128 LoRA buffers, the 27B rank-64 (16 hot each) and 32k context. adapters still key
    # off the logical base_model while the engine loads the configured pre-quantized checkpoint.
    {
        "base_model": "Qwen/Qwen3.5-4B",
        "image_input_limit": 4,
        "gpu": "L4",
        "engine": {
            "gpu_memory_utilization": 0.98,
            "max_loras": 16,
            "max_lora_rank": 128,  # rank-128 / 16 hot LoRAs (the 4 GiB FP8 4B has ample L4 headroom); 32k.
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
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
            "max_num_seqs": 8,
            "enforce_eager": False,
            "reasoning_parser": "qwen3",
        },
    },
    {
        "base_model": "Qwen/Qwen3.6-27B",
        "image_input_limit": 4,
        "gpu": "H100",
        "engine": {
            # 0.90 (was 0.98) makes room for CUDA-graph capture. this hybrid GatedDeltaNet measured
            # about 11 tok/s eager versus 80 tok/s with CUDA graphs on the H100.
            "gpu_memory_utilization": 0.90,
            "max_loras": 16,
            "max_lora_rank": 64,
            "max_model_len": 32768,
            "max_num_seqs": 8,
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
            # all-expert MoE activations don't spike the profiling peak.
            "max_num_seqs": 8,
            "reasoning_parser": "qwen3",
            # NB: the vision encoder is now LOADED (no language_model_only) — flash adapters adapt the
            # full multimodal tree, so their vision-tower LoRA keys must have real modules to bind to.
            # this adds the vision encoder's weights on top of the already weight-bound 6 x 64 LoRA buffer.
            # the complete model and LoRA load is about 108 GiB on the 141 GiB H200.
        },
    },
]

_BY_MODEL: dict[str, dict[str, Any]] = {m["base_model"]: m for m in SERVING_MODELS}
_HOSTED_TRAFFIC_POLICY_BY_MODEL: dict[str, HostedTrafficPolicy] = {
    model["base_model"]: HostedTrafficPolicy.from_engine(model.get("engine") or {})
    for model in SERVING_MODELS
}


def supports_image_input(base_model: str) -> bool:
    return image_limit_for(base_model) is not None


def image_limit_for(base_model: str) -> int | None:
    return _config_for(base_model)["image_input_limit"]


def base_models() -> list[str]:
    return [m["base_model"] for m in SERVING_MODELS]


def is_supported_base_model(base_model: str) -> bool:
    return base_model in _BY_MODEL


def hosted_traffic_policy_for(base_model: str) -> HostedTrafficPolicy:
    _config_for(base_model)
    return _HOSTED_TRAFFIC_POLICY_BY_MODEL[base_model]


def configured_warm_container_floor() -> int:
    return sum(hosted_traffic_policy_for(model).min_containers for model in base_models())


def configured_hard_gpu_ceiling() -> int:
    return sum(hosted_traffic_policy_for(model).max_containers for model in base_models())


def configured_router_async_capacity() -> int:
    """Finite router concurrency for every model's hard slots plus bounded waiters."""
    capacity = sum(
        policy.max_inputs * policy.max_containers + policy.queue_capacity
        for policy in _HOSTED_TRAFFIC_POLICY_BY_MODEL.values()
    )
    if capacity <= 0:
        raise ValueError("hosted router async capacity must be positive")
    return capacity


def _config_for(base_model: str) -> dict[str, Any]:
    cfg = _BY_MODEL.get(base_model)
    if cfg is None:
        allowed = ", ".join(base_models())
        raise ValueError(
            f"Unsupported base model {base_model!r}; add it to SERVING_MODELS after a "
            f"real-GPU serving canary. Supported base models: {allowed}"
        )
    return cfg


def gpu_for(base_model: str) -> str:
    """The Modal GPU class to run ``base_model``'s engine on (catalog ``gpu``, else ``DEFAULT_GPU``)."""
    return _config_for(base_model).get("gpu") or DEFAULT_GPU


def serve_model_for(base_model: str) -> str:
    """The pre-quantized FP8 checkpoint the engine LOADS for ``base_model``. Every catalog model
    resolves to a verified pre-quant checkpoint; adapters and routing still key off the logical
    ``base_model``, and vLLM auto-detects the checkpoint's compressed-tensors quantization (no online
    override)."""
    _config_for(base_model)  # reject uncataloged models before resolving a checkpoint
    return _prequant_serve_model_for(base_model)


def engine_overrides_for(base_model: str) -> dict[str, Any]:
    """Per-base-model vLLM engine-arg overrides, with the resolved pre-quantized FP8 ``serve_model_id``
    injected. So every model carries a ``serve_model_id`` (the FP8 checkpoint to load), plus any tier
    shape (the 35B MoE's lower max_loras, the L4 rank overrides, …)."""
    overrides = dict(_config_for(base_model).get("engine") or {})
    overrides.setdefault("serve_model_id", serve_model_for(base_model))
    return overrides


def reasoning_parser_for(base_model: str) -> str | None:
    """The model-scoped vLLM reasoning parser, or None when parsing is disabled."""
    parser = (_config_for(base_model).get("engine") or {}).get("reasoning_parser")
    return str(parser) if parser else None
