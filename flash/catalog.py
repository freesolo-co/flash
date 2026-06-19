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
# sizing reference and the spec/from_dict fallback). The validated GPU class set
# (SUPPORTED/is_validated) lives in providers.base; per-provider classes and pricing live
# under providers/{runpod,vast}. Defined above ModelInfo so it can back the
# recommended_gpu field default.
DEFAULT_GPU = "RTX 5090"


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
    # Requires the DISAGGREGATED (multi-GPU async) GRPO path: too large to colocate the trainer +
    # vLLM rollout on one GPU. A GRPO request for such a model must set ``[train].inference_gpus>0``
    # on a multi-GPU node (see engine.rollout_bench); colocate GRPO is rejected. SFT is unaffected.
    requires_disaggregated: bool = False
    # Attention-head count, used to REJECT an invalid tensor-parallel disaggregated split at SUBMIT
    # time (before renting): vLLM requires num_attention_heads % inference_gpus == 0 for TP, so e.g.
    # MiniCPM5-1B (16 heads) with inference_gpus=3 is impossible. 0 = unknown -> the submit-time
    # check is skipped and the worker's pre-server-boot guard (engine.worker.run_rl) is the catch-all
    # (it also covers open-model-policy runs, whose head count isn't known until the worker reads the
    # config). Declared for catalog entries whose head count is verified, so a known-invalid ratio
    # fails validation instead of charging the user for a node that can never run.
    num_attention_heads: int = 0

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
        algos=("sft", "grpo"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
        # 16 attention heads -> valid TP / inference_gpus are {1, 2, 4, 8}; a 1:3 (TP=3) split is
        # mathematically invalid (16 % 3 != 0) and is rejected at submit time (docs/async-rollout).
        num_attention_heads=16,
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
        algos=("sft", "grpo"),
        min_vram_gb=16,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
    ),
    "Qwen/Qwen3.5-4B": ModelInfo(
        id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        params="4.7B (text-only fine-tune)",
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
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B (MoE)",
        params="35B total / ~3B active (MoE)",
        algos=("sft", "grpo"),
        min_vram_gb=48,
        grpo_min_vram_gb=80,
        quant="4bit-qlora",
        recommended_gpu="A100 PCIe",
        thinking="hybrid",
        # The worker prefetches the FULL bf16 checkpoint into the HF cache before 4-bit load
        # (snapshot_download grabs every safetensor regardless of quant), and a 35B bf16
        # checkpoint is ~70 GB — it overflows the 60/64 GB container default and the run fails
        # in prefetch_model after a paid multi-GPU node is already provisioned. Floor the disk
        # the same way the open-model policy does (~2 GB/B weights + 64 GB worker-stack/cache
        # headroom): 35B -> ~134 GB; round to 140. The runner raises gpu.disk_gb to this.
        min_disk_gb=140,
        # Re-added for the DISAGGREGATED (multi-GPU async) GRPO path only: it OOMs when the trainer
        # and the vLLM rollout are colocated on one card. With a dedicated inference GPU (35B served
        # 4-bit) + a sharded trainer on the rest, it fits. GRPO colocate for it is rejected.
        requires_disaggregated=True,
        notes="MoE; GRPO requires the disaggregated multi-GPU node ([train].inference_gpus>0). "
        "The 35B is served 4-bit on the inference GPU while a sharded trainer runs on the rest.",
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


def resolve_model(
    model_id: str,
    algorithm: str,
    policy: str = "catalog",
    gpu: str | None = None,
) -> ModelInfo:
    """Resolve a model under the configured policy.

    ``catalog`` (default): the model must be a curated catalog entry.
    ``allow``: any HF model is accepted; a coarse VRAM-fit estimate (HF safetensors
    metadata, no download) blocks only provably-impossible fits and warns on tight ones.
    """
    algo = normalize_algorithm(algorithm)
    if model_id in MODELS:
        return validate_model_for_algorithm(model_id, algo)
    if policy != "allow":
        # Reuse get_model's error (includes the open-model hint).
        return get_model(model_id)
    return _resolve_open_model(model_id, algo, gpu)


def _resolve_open_model(model_id: str, algo: str, gpu: str | None) -> ModelInfo:
    """Synthesize a ModelInfo for the open-model "allow" policy from a coarse VRAM-fit
    estimate (HF safetensors metadata, no download). Blocks provably-impossible fits and
    warns on tight ones. Isolates the engine.vram dependency + disk-floor heuristic from
    the curated-catalog path in resolve_model."""
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
