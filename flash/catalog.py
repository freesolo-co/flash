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


# The default GPU class a run lands on when none is pinned (also the open-model-policy
# sizing reference and the spec/from_dict fallback). The managed GPU class set (KNOWN)
# lives in providers.base; per-provider classes and pricing live under
# providers/{runpod,vast}. Defined above ModelInfo so it can back the recommended_gpu
# field default.
DEFAULT_GPU = "RTX 5090"

# Output vocab (== config.vocab_size, the lm_head / logits width — the PADDED model vocab,
# NOT the raw tokenizer token count). Sizes the GRPO fp32-logits VRAM term (engine.vram) and
# the per-device completion cap (engine.worker.rl_per_device_comps). This is the open-model
# fallback; curated per-model values live on each ModelInfo below and are read via
# vocab_size_for(). Over-estimating is the memory-SAFE direction (smaller cap, larger VRAM
# estimate), so the fallback is the largest catalog vocab.
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
    # GRPO needs more VRAM than SFT (a colocated vLLM rollout engine holds a second copy of
    # the weights + KV cache). 0 => GRPO uses ``min_vram_gb`` like SFT; set it when the GRPO
    # tier needs a bigger card than SFT (the colocate 2nd weight copy + KV pool). Consumed by
    # engine.vram.model_required_vram_gb.
    grpo_min_vram_gb: int = 0
    notes: str = ""
    # Worker container disk this model needs (GB). 0 = the platform default (64 GB)
    # suffices. The runner raises gpu.disk_gb to at least this, so big-checkpoint
    # models whose weights alone exceed 64 GB work out of the box.
    min_disk_gb: int = 0
    # Thinking/reasoning capability of the checkpoint's chat template:
    #   "none"    no <think> support (or a non-thinking variant) — `thinking = true` is
    #             rejected for these models
    #   "hybrid"  template honors enable_thinking (Qwen3-style hybrid reasoning)
    #   "always"  the model always emits reasoning; enable_thinking can't turn it off,
    #             so `thinking = true` is required
    #   "unknown" open-model-policy entries (capability not verified)
    thinking: str = "none"
    # Output vocab = config.vocab_size (lm_head / logits width, the padded model vocab — not
    # the raw tokenizer count). Drives the GRPO fp32-logits memory term and the per-device
    # completion cap. Curated per model below; defaults to the open-model fallback.
    vocab_size: int = _DEFAULT_VOCAB_SIZE
    # Total parameters in billions — the numeric model size the cost estimator reads directly
    # (no parsing of the ``params`` display string). Curated per catalog model below.
    params_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The default model Flash trains when a config omits one. A current-gen dense 4B
# (text-only fine-tune) on the modern worker stack — the safe out-of-the-box choice for
# the average developer. It is thinking-"hybrid"; the thinking flag now defaults ON.
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
    # ---- Qwen3.5 dense family: validated on the modern worker stack ----
    # (trl 1.x / vllm 0.19 / transformers 5.x). Trained + served TEXT-ONLY: the
    # checkpoints are natively multimodal, so LoRA excludes the vision tower and vLLM
    # loads language_model_only (see flash.engine.worker). Each entry passed a real
    # train+eval smoke on its recommended GPU (bench/results/phase1/).
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
        min_vram_gb=16,
        # MEMORY-OPTIMIZED: 4-bit NF4 frozen base + bf16 LoRA adapter (QLoRA). The base
        # drops from ~19 GB bf16 to ~5.3 GB, so colocated GRPO holds two 4-bit copies
        # (trainer + bnb-quantized vLLM rollout) instead of two bf16 copies -> it fits a
        # ~24-32 GB card instead of an 80 GB A100. NF4 is near-lossless for adapter training
        # (QLoRA paper + follow-ups), a small quality trade for a ~3x cheaper GPU. No GRPO
        # floor: the matrix sizes the (much smaller) 4-bit footprint directly.
        grpo_min_vram_gb=0,
        quant="4bit-qlora",
        recommended_gpu="RTX 5090",
        thinking="hybrid",
        notes="QLoRA (4-bit NF4 base + bf16 LoRA). GRPO's colocated vLLM rollout loads the "
        "base 4-bit via bitsandbytes too, so both copies are 4-bit -> fits ~24-32 GB "
        "instead of 80 GB bf16. ~near-lossless vs bf16 LoRA.",
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
    """Output vocab (== config.vocab_size, the lm_head / logits width) for a model — the
    number that sizes the GRPO fp32-logits VRAM term and the per-device completion cap.
    Returns the curated catalog value, else the safe default for open-model-policy entries.
    This is the PADDED model vocab, not the raw tokenizer token count."""
    info = MODELS.get(model_id)
    return info.vocab_size if info is not None else _DEFAULT_VOCAB_SIZE


def resolve_model(
    model_id: str,
    algorithm: str,
    policy: str = "catalog",
    gpu: str | None = None,
    *,
    skip_net: bool = False,
) -> ModelInfo:
    """Resolve a model under the configured policy.

    ``catalog`` (default): the model must be a curated catalog entry.
    ``allow``: any HF model is accepted; a coarse VRAM-fit estimate (HF safetensors
    metadata, no download) blocks only provably-impossible fits and warns on tight ones.

    ``skip_net=True`` sizes an unlisted "allow" model offline (no HF probe), threaded explicitly
    so ``flash train --cost`` parses the spec with zero network I/O. Catalog models never probe.
    """
    algo = normalize_algorithm(algorithm)
    if model_id in MODELS:
        return validate_model_for_algorithm(model_id, algo)
    if policy != "allow":
        # Reuse get_model's error (includes the open-model hint).
        return get_model(model_id)
    return _resolve_open_model(model_id, algo, gpu, skip_net=skip_net)


def _resolve_open_model(
    model_id: str, algo: str, gpu: str | None, *, skip_net: bool = False
) -> ModelInfo:
    """Synthesize a ModelInfo for the open-model "allow" policy from a coarse VRAM-fit
    estimate (HF safetensors metadata, no download). Blocks provably-impossible fits and
    warns on tight ones. Isolates the engine.vram dependency + disk-floor heuristic from
    the curated-catalog path in resolve_model. ``skip_net=True`` forces the offline (no-HF-probe)
    sizing for the cost estimator's parse."""
    from flash.engine.vram import check_fit

    est = check_fit(model_id, algo, gpu or DEFAULT_GPU, skip_net=skip_net)
    if est.verdict == "too_big":
        raise ValueError(
            f"{model_id} does not fit the requested GPU: {est.describe()}. "
            f"Pick a smaller model or a larger supported GPU."
        )
    if est.verdict in ("tight", "unknown"):
        print(f"warning: open-model policy: {est.describe()}")
    params = f"{est.params_b:.1f}B" if est.params_b else "unknown size"
    # Disk floor for the open model: a bf16 checkpoint is ~2 GB per billion params;
    # add worker-stack headroom so a large model that passes the VRAM check can't
    # provision a paid worker and then fail in prefetch_model when the checkpoint
    # overflows the 64 GB container default. 0 (unknown size) leaves the default
    # (the user can still raise it with gpu.disk_gb).
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
    # Catalog entries advertise the capability classes "sft" and "grpo": grpo needs the
    # colocated rollout engine, sft is trainer-only.
    required = "grpo" if algo == "grpo" else "sft"
    if required not in info.algos:
        allowed = ", ".join(info.algos)
        raise ValueError(f"{model_id} supports {allowed}, not {algo}")
    return info


def public_model_rows() -> list[dict[str, Any]]:
    return [m.to_dict() for m in list_models()]
