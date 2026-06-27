"""Curated model catalog for one-consumer-GPU LoRA jobs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

ALGORITHMS = ("sft", "grpo")


def normalize_algorithm(value: str) -> str:
    """Canonical (lowercased, validated) algorithm name."""
    value = (value or "grpo").lower()
    if value not in ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {value}; known: {', '.join(ALGORITHMS)}")
    return value


DEFAULT_GPU = "RTX 5090"

# Over-estimating is memory-safe (larger VRAM estimate, smaller cap); fallback = largest catalog vocab.
_DEFAULT_VOCAB_SIZE = 248_320


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    params: str
    algos: tuple[str, ...]
    min_vram_gb: int
    quant: str = "bf16"
    recommended_gpu: str = DEFAULT_GPU
    # 0 => GRPO uses min_vram_gb like SFT; set when colocated vLLM rollout needs a bigger card.
    grpo_min_vram_gb: int = 0
    # 0 => SFT sizes from param-based estimate; set only when a model must not down-route to the cheapest card.
    sft_min_vram_gb: int = 0
    notes: str = ""
    # 0 = platform default (64 GB) suffices. Runner raises gpu.disk_gb to at least this.
    min_disk_gb: int = 0
    # "none" / "hybrid" (Qwen3-style) / "always" (can't disable) / "unknown" (open-model policy)
    thinking: str = "none"
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    params_b: float = 0.0
    # MoE only: active params per token; 0.0 means dense (same as params_b).
    active_params_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

MODELS: dict[str, ModelInfo] = {
    "openbmb/MiniCPM5-1B": ModelInfo(
        id="openbmb/MiniCPM5-1B",
        display_name="MiniCPM5 1B",
        params="1.2B dense (Llama arch)",
        params_b=1.2,
        vocab_size=130_560,
        algos=("sft", "grpo"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
        notes="On-device class SLM (131k ctx); standard Llama architecture.",
    ),
    "Qwen/Qwen3.5-0.8B": ModelInfo(
        id="Qwen/Qwen3.5-0.8B",
        display_name="Qwen3.5 0.8B",
        params="0.9B (text-only fine-tune)",
        params_b=0.9,
        vocab_size=248_320,
        algos=("sft", "grpo"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
        notes="Smallest Qwen3.5; cheap smoke/dev runs with the modern arch.",
    ),
    "Qwen/Qwen3.5-2B": ModelInfo(
        id="Qwen/Qwen3.5-2B",
        display_name="Qwen3.5 2B",
        params="2.3B (text-only fine-tune)",
        params_b=2.3,
        vocab_size=248_320,
        algos=("sft", "grpo"),
        min_vram_gb=16,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
    ),
    "Qwen/Qwen3.5-4B": ModelInfo(
        id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        params="4.7B (text-only fine-tune)",
        params_b=4.7,
        vocab_size=248_320,
        algos=("sft", "grpo"),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
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
        algos=("sft", "grpo"),
        min_vram_gb=48,
        # NOT QLoRA: peft bnb merge during GRPO rollout diverges trainer precision -> TRL ratio collapses to 0.
        grpo_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
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
        vocab_size=248_320,
        algos=("sft", "grpo"),
        min_vram_gb=141,
        # Floor to 100 GB so SFT lands on H200, not the thin-margin consumer Blackwell or 80 GB H100.
        sft_min_vram_gb=100,
        # Floor also engages grpo_seq_escalation_gb: long (>16k) rollouts are rejected at parse time instead of OOMing in vLLM.
        grpo_min_vram_gb=180,
        quant="bf16",
        recommended_gpu="H200",
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
    required = "grpo" if algo == "grpo" else "sft"
    if required not in info.algos:
        allowed = ", ".join(info.algos)
        raise ValueError(f"{model_id} supports {allowed}, not {algo}")
    return info


def public_model_rows() -> list[dict[str, Any]]:
    return [m.to_dict() for m in list_models()]
