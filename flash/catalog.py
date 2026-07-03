"""Curated model catalog for one-consumer-GPU LoRA jobs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

ALGORITHMS = ("sft", "grpo", "opd")


def normalize_algorithm(value: str) -> str:
    """Canonical (lowercased, validated) algorithm name."""
    if not value:
        value = "grpo"
    elif not isinstance(value, str):
        # A truthy non-string (e.g. a JSON number/bool/array) would AttributeError on .lower(), which
        # escapes the callers' ValueError/ConfigError guards -> uncaught 500. Raise ValueError instead.
        raise ValueError(f"algorithm must be a string, got {type(value).__name__}")
    value = value.lower()
    if value not in ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {value}; known: {', '.join(ALGORITHMS)}")
    return value


DEFAULT_GPU = "RTX 5090"

# Over-estimating is memory-safe (larger VRAM estimate, smaller cap); fallback = largest catalog vocab.
_DEFAULT_VOCAB_SIZE = 248_320


@dataclass(frozen=True)
class ServingCapacity:
    gpu: str
    max_loras: int
    max_lora_rank: int
    max_model_len: int
    serve_model_id: str = ""
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    tensor_parallel_size: int = 0
    gpu_memory_utilization: float = 0.0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    params: str
    algos: tuple[str, ...]
    min_vram_gb: int
    # Total parameters in billions — the numeric model size the cost estimator + VRAM equations read
    # DIRECTLY (no parsing of the ``params`` display string). Drives the memory/size terms (VRAM, disk,
    # download), which always size the FULL checkpoint. REQUIRED: every ModelInfo must state it — a
    # curated catalog model sets its true count, and the open-model policy passes the HF/estimated count
    # (or 0.0 when the size is genuinely "unknown size"). ``test_every_catalog_entry_sets_params_b``
    # asserts every curated MODELS entry sets it > 0, so a new entry can never silently fall back to a
    # parsed string again.
    params_b: float
    quant: str = "bf16"
    recommended_gpu: str = DEFAULT_GPU
    # 0 => GRPO uses min_vram_gb like SFT; set when colocated vLLM rollout needs a bigger card.
    grpo_min_vram_gb: int = 0
    # 0 => SFT sizes from param-based estimate; set only when a model must not down-route to the cheapest card.
    sft_min_vram_gb: int = 0
    # vLLM sleep mode (offload the colocate rollout engine between GRPO steps) is NON-FUNCTIONAL for
    # this model: the wake/reload HANGS the rollout (a ~70 GB weight reallocation can't be placed in
    # the fragmented non-expandable allocator sleep forces -- live-confirmed on the 35B-A3B, every
    # attempt stalled). So this model is RESIDENT-ONLY: a config that doesn't fit resident must be
    # REJECTED (model_required_vram_gb sizes it on the resident peak) rather than routed to the hanging
    # sleep path. grpo_sleep_mode raises for it instead of ever returning True. Dense/small models that
    # sleep cleanly leave this False.
    sleep_unsupported: bool = False
    notes: str = ""
    # 0 = platform default (64 GB) suffices. Runner raises gpu.disk_gb to at least this.
    min_disk_gb: int = 0
    # Deployment capacity of the external freesolo multi-LoRA serving app. This is separate from
    # Flash's training GPU recommendation above; serving uses Modal/vLLM and sizes hot LoRA buffers
    # by max_loras x max_lora_rank at engine init.
    serving: ServingCapacity | None = None
    # "none" / "hybrid" (Qwen3-style) / "always" (can't disable) / "unknown" (open-model policy)
    thinking: str = "none"
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    # Parameters ACTIVE per token in billions — only meaningful for an MoE, where a token routes
    # through a small subset of experts. The cost estimator's per-token FLOPs/step-time term reads
    # this (a token exercises only the active params), while VRAM/disk/download keep using the total
    # ``params_b``. 0.0 (the dense default) means "same as params_b" — every token hits every param.
    active_params_b: float = 0.0
    # Transformer geometry (decoder layers x hidden width) — the SFT gradient-checkpointing-OFF gate
    # sizes the no-recompute activation peak from these (engine.vram.sft_gc_off_peak_gb). 0/0 (the
    # default) means "unknown": the worker falls back to reading the HF config at runtime, and the
    # GC-off gate stays conservative (keeps GC on) if neither is available. Curated for the MoE whose
    # SFT runs the gate — a live B200 SFT showed the runtime AutoConfig probe returning (0, 0) on the
    # multimodal-nested config, so the curated values are what actually engage the gate.
    num_layers: int = 0
    hidden_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        serving = data.get("serving")
        if serving is None:
            data.pop("serving", None)
        else:
            for key in (
                "serve_model_id",
                "max_num_seqs",
                "max_num_batched_tokens",
                "tensor_parallel_size",
                "gpu_memory_utilization",
            ):
                if serving.get(key) in ("", 0, 0.0, None):
                    serving.pop(key, None)
        return data


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# The pre-quantized FP8 checkpoint each base model's serving engine LOADS (mirrors serving's
# ``src.prequant_config``). Every dense model now serves a Freesolo-OWNED FP8_DYNAMIC checkpoint (no
# community-repo dependence); the 35B VL MoE serves the OFFICIAL Qwen FP8 (it preserves the full
# vision-language model). Informational for the catalog mirror — deploy gating reads only max_lora_rank.
SERVING_FP8_MODEL_REPOS: dict[str, str] = {
    "openbmb/MiniCPM5-1B": "Freesolo-Co/MiniCPM5-1B-FP8",
    "Qwen/Qwen3.5-0.8B": "Freesolo-Co/Qwen3.5-0.8B-FP8",
    "Qwen/Qwen3.5-2B": "Freesolo-Co/Qwen3.5-2B-FP8",
    "Qwen/Qwen3.5-4B": "Freesolo-Co/Qwen3.5-4B-FP8",
    "Qwen/Qwen3.5-9B": "Freesolo-Co/Qwen3.5-9B-FP8",
    "Qwen/Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B-FP8",
}

MODELS: dict[str, ModelInfo] = {
    "openbmb/MiniCPM5-1B": ModelInfo(
        id="openbmb/MiniCPM5-1B",
        display_name="MiniCPM5 1B",
        params="1.2B dense (Llama arch)",
        params_b=1.2,
        vocab_size=130_560,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        serving=ServingCapacity(
            gpu="L4",
            serve_model_id=SERVING_FP8_MODEL_REPOS["openbmb/MiniCPM5-1B"],
            max_loras=16,
            max_lora_rank=128,
            max_model_len=8192,
        ),
        thinking="hybrid",
        notes="On-device class SLM (131k ctx); standard Llama architecture.",
    ),
    "Qwen/Qwen3.5-0.8B": ModelInfo(
        id="Qwen/Qwen3.5-0.8B",
        display_name="Qwen3.5 0.8B",
        params="0.9B (text-only fine-tune)",
        params_b=0.9,
        vocab_size=248_320,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        serving=ServingCapacity(
            gpu="L4",
            serve_model_id=SERVING_FP8_MODEL_REPOS["Qwen/Qwen3.5-0.8B"],
            max_loras=16,
            max_lora_rank=128,
            max_model_len=8192,
        ),
        thinking="hybrid",
        notes="Smallest Qwen3.5; cheap smoke/dev runs with the modern arch.",
    ),
    "Qwen/Qwen3.5-2B": ModelInfo(
        id="Qwen/Qwen3.5-2B",
        display_name="Qwen3.5 2B",
        params="2.3B (text-only fine-tune)",
        params_b=2.3,
        vocab_size=248_320,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=16,
        recommended_gpu="RTX 4090",
        serving=ServingCapacity(
            gpu="L4",
            serve_model_id=SERVING_FP8_MODEL_REPOS["Qwen/Qwen3.5-2B"],
            max_loras=16,
            max_lora_rank=128,
            max_model_len=8192,
        ),
        thinking="hybrid",
    ),
    "Qwen/Qwen3.5-4B": ModelInfo(
        id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        params="4.7B (text-only fine-tune)",
        params_b=4.7,
        vocab_size=248_320,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
        serving=ServingCapacity(
            gpu="L4",
            serve_model_id=SERVING_FP8_MODEL_REPOS["Qwen/Qwen3.5-4B"],
            max_loras=16,
            max_lora_rank=64,
            max_model_len=8192,
            max_num_seqs=8,
            gpu_memory_utilization=0.98,
        ),
        thinking="hybrid",
        notes="Current-gen 4B. GRPO uses the sleep-mode memory recipe (hybrid arch needs "
        "extra engine state-cache); fused DeltaNet kernels ship in the default stack.",
    ),
    "Qwen/Qwen3.5-9B": ModelInfo(
        id="Qwen/Qwen3.5-9B",
        display_name="Qwen3.5 9B",
        params="9.7B (text-only fine-tune)",
        params_b=9.7,
        vocab_size=248_320,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=48,
        # NOT QLoRA: peft bnb merge during GRPO rollout diverges trainer precision -> TRL ratio collapses to 0.
        grpo_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
        serving=ServingCapacity(
            gpu="L4",
            serve_model_id=SERVING_FP8_MODEL_REPOS["Qwen/Qwen3.5-9B"],
            max_loras=16,
            max_lora_rank=64,
            max_model_len=8192,
            max_num_seqs=8,
            gpu_memory_utilization=0.98,
        ),
        thinking="hybrid",
        notes="bf16 LoRA. ~19 GB of weights; SFT fits a 48 GB card, while colocated GRPO "
        "(two bf16 copies + KV + the 248k-vocab fp32 logits) needs an 80 GB-class card "
        "(grpo_min_vram_gb floor).",
    ),
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B (MoE)",
        params="35B total / ~3B active (MoE)",
        # 35.0 not 35.95: the marketing figure tips the SFT equation over the B200 budget (see test_sft_equation_covers_honest_peak_across_seq_boundary).
        params_b=35.0,
        active_params_b=3.0,
        # Geometry for the SFT GC-off activation estimate (config.json text_config): 40 decoder
        # layers x 2048 hidden (hybrid GatedDeltaNet + full-attention, 256 experts / 8 active).
        num_layers=40,
        hidden_size=2048,
        vocab_size=248_320,
        algos=("sft", "grpo", "opd"),
        min_vram_gb=141,
        # Floor to 100 GB so SFT lands on H200, not the thin-margin consumer Blackwell or 80 GB H100.
        sft_min_vram_gb=100,
        # Floor also engages grpo_seq_escalation_gb: long (>16k) rollouts are rejected at parse time instead of OOMing in vLLM.
        grpo_min_vram_gb=180,
        # vLLM sleep mode HANGS the 35B colocate rollout (wake/reload stalls — live-confirmed, every
        # attempt). So GRPO is RESIDENT-ONLY: model_required_vram_gb sizes on the resident peak and a
        # config too long to fit resident is REJECTED at parse time (not routed to the hanging sleep).
        sleep_unsupported=True,
        quant="bf16",
        recommended_gpu="H200",
        serving=ServingCapacity(
            gpu="A100-80GB",
            serve_model_id=SERVING_FP8_MODEL_REPOS["Qwen/Qwen3.6-35B-A3B"],
            # rank-64 at only 6 hot slots: the fused-MoE LoRA buffer scales with
            # max_loras x rank x num_experts, so the A100-80GB ceiling is ~max_loras x rank = 384
            # (6 x 64 fits at 99.3% util; 16 x 64 OOMs on every single/multi GPU). Serving-validated.
            max_loras=6,
            max_lora_rank=64,
            max_model_len=8192,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.98,
        ),
        thinking="hybrid",
        min_disk_gb=200,
        notes="MoE (35B total / ~3B active), bf16 LoRA. SFT runs on the 141 GB H200 (the ~70 GB "
        "weights dominate; active-3B compute keeps activations/KV tiny, so context is ~unbounded by "
        "VRAM); colocated GRPO needs the 180 GB B200 (trainer + vLLM rollout = two 70 GB copies).",
    ),
}


def list_models() -> list[ModelInfo]:
    return sorted(MODELS.values(), key=lambda m: (m.min_vram_gb, m.id))


def get_model(model_id: str) -> ModelInfo:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        allowed = ", ".join(MODELS)
        raise ValueError(
            f"unsupported model {model_id!r}; choose one of: {allowed} — or set "
            f'model_policy = "allow" in the config to run any HF model that fits the GPU '
            f"(open-model policy)"
        ) from exc


def serving_lora_rank_cap(model: str | ModelInfo | None) -> int | None:
    """Return the model's serving LoRA rank cap, or None when Flash has no local cap.

    Serving capacity is model-specific: small serving models currently allow rank 64, while larger
    serving paths currently allow rank 32. Unknown/open-policy models intentionally return None instead
    of inheriting a global fallback.
    """
    if isinstance(model, ModelInfo):
        info = model
    elif isinstance(model, str) and model.strip():
        info = MODELS.get(model.strip())
    else:
        info = None
    if info is None or info.serving is None:
        return None
    return int(info.serving.max_lora_rank)


def vocab_size_for(model_id: str) -> int:
    """Curated vocab_size for a model, or the safe default for open-model-policy entries."""
    info = MODELS.get(model_id)
    return info.vocab_size if info is not None else _DEFAULT_VOCAB_SIZE


def resolve_model(
    model_id: str,
    algorithm: str,
    policy: str = "catalog",
    gpu: str | None = None,
) -> ModelInfo:
    """Resolve a model under the configured policy; "allow" accepts any HF model."""
    algo = normalize_algorithm(algorithm)
    if model_id in MODELS:
        return validate_model_for_algorithm(model_id, algo)
    if policy != "allow":
        return get_model(model_id)
    return _resolve_open_model(model_id, algo, gpu)


def _resolve_open_model(model_id: str, algo: str, gpu: str | None) -> ModelInfo:
    """Synthesize a ModelInfo for the open-model "allow" policy via a coarse HF VRAM-fit estimate."""
    from flash.engine.vram import check_fit

    est = check_fit(model_id, algo, gpu or DEFAULT_GPU)
    if est.verdict == "too_big":
        raise ValueError(
            f"{model_id} does not fit the requested GPU: {est.describe()}. "
            f"Pick a smaller model or a larger supported GPU."
        )
    if est.verdict in ("tight", "unknown"):
        print(f"warning: open-model policy: {est.describe()}")
    params = f"{est.params_b:.1f}B" if est.params_b else "unknown size"
    min_disk = int(est.params_b * 2) + 64 if est.params_b else 0
    return ModelInfo(
        id=model_id,
        display_name=model_id,
        params=params,
        # Carry the estimated/HF param count straight through (0.0 when size is unknown) so downstream
        # sizing reads ``params_b`` directly — no re-parsing the display string.
        params_b=est.params_b or 0.0,
        algos=ALGORITHMS,
        min_vram_gb=math.ceil(est.est_gb) if est.est_gb else 24,
        min_disk_gb=min_disk,
        recommended_gpu=gpu or DEFAULT_GPU,
        thinking="unknown",
        notes="unlisted model accepted via the open-model policy (not curated/validated)",
    )


def validate_model_for_algorithm(model_id: str, algorithm: str) -> ModelInfo:
    info = get_model(model_id)
    algo = normalize_algorithm(algorithm)
    # Each algorithm gates on its own capability entry (sft/grpo/opd), so a model must
    # explicitly opt into opd via its algos tuple — same contract as sft/grpo.
    if algo not in info.algos:
        allowed = ", ".join(info.algos)
        raise ValueError(f"{model_id} supports {allowed}, not {algo}")
    return info


def public_model_rows() -> list[dict[str, Any]]:
    return [m.to_dict() for m in list_models()]
