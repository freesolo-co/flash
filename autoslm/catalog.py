"""Curated model catalog for one-consumer-GPU LoRA jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

ALGORITHMS = ("sft", "grpo")


def normalize_algorithm(value: str) -> str:
    """Canonical (lowercased, validated) algorithm name."""
    value = (value or "grpo").lower()
    if value not in ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {value}; known: {', '.join(ALGORITHMS)}")
    return value


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    params: str
    algos: tuple[str, ...]
    min_vram_gb: int
    quant: str = "bf16"
    recommended_gpu: str = "RTX 5090"
    experimental: bool = False
    notes: str = ""
    # Worker container disk this model needs (GB). 0 = the platform default (64 GB)
    # suffices. The orchestrator raises gpu.disk_gb to at least this, so big-checkpoint
    # models (MoE tiers whose bf16 weights alone exceed 64 GB) work out of the box.
    min_disk_gb: int = 0
    # Optional pre-quantized weights repo for the 4bit-qlora tier: the worker loads
    # these (~0.55 B/param) instead of quantizing the full bf16 checkpoint at load
    # (tokenizer/config still come from ``id``). Cuts the download ~3.5x and fits the
    # stock 64 GB disk. Only trusted/own exports belong here.
    quant_repo: str = ""
    # Thinking/reasoning capability of the checkpoint's chat template:
    #   "none"    no <think> support (or a non-thinking variant) — `thinking = true` is
    #             rejected for these models
    #   "hybrid"  template honors enable_thinking (Qwen3-style hybrid reasoning)
    #   "always"  the model always emits reasoning; enable_thinking can't turn it off,
    #             so `thinking = true` is required
    #   "unknown" open-model-policy entries (capability not verified)
    thinking: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


# The default model AutoSLM trains when a config omits one. A proven, dense, text-only
# instruction model that loads cleanly on the pinned worker stack (transformers<5,
# trl<0.24, vllm<0.11) — the safe out-of-the-box choice for the average developer.
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

MODELS: dict[str, ModelInfo] = {
    # ---- Supported tier: proven dense text models on the pinned worker stack ----
    "Qwen/Qwen3-0.6B": ModelInfo(
        id="Qwen/Qwen3-0.6B",
        display_name="Qwen3 0.6B",
        params="0.6B dense",
        algos=("sft", "grpo"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
        notes="Smallest real model; ideal for cheap smoke/dev runs.",
    ),
    "Qwen/Qwen3-4B-Instruct-2507": ModelInfo(
        id="Qwen/Qwen3-4B-Instruct-2507",
        display_name="Qwen3 4B Instruct 2507",
        params="4B dense",
        algos=("sft", "grpo"),
        min_vram_gb=24,
        recommended_gpu="RTX 4090",
        thinking="none",  # the non-thinking Instruct variant (no <think> in its template)
        notes="Default model: benchmark-proven, dense, loads on the pinned worker stack.",
    ),
    "Qwen/Qwen3-8B": ModelInfo(
        id="Qwen/Qwen3-8B",
        display_name="Qwen3 8B",
        params="8B dense",
        algos=("sft", "grpo"),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
        thinking="hybrid",
    ),
    "openbmb/MiniCPM5-1B": ModelInfo(
        id="openbmb/MiniCPM5-1B",
        display_name="MiniCPM5 1B",
        params="1.2B dense (Llama arch)",
        algos=("sft", "grpo"),
        min_vram_gb=12,
        recommended_gpu="RTX 4090",
        thinking="hybrid",
        notes="On-device class SLM (131k ctx); standard Llama architecture.",
    ),
    "Qwen/Qwen3-30B-A3B": ModelInfo(
        id="Qwen/Qwen3-30B-A3B",
        display_name="Qwen3 30B-A3B",
        params="30B total / 3B active MoE",
        algos=("sft",),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
        quant="4bit-qlora",
        experimental=True,
        thinking="hybrid",
        min_disk_gb=120,  # ~61 GB bf16 checkpoint + worker stack > the 64 GB default
        notes="Experimental SFT-only tier; all experts still occupy VRAM.",
    ),
    # ---- Qwen3.5 dense family: validated on the modern worker stack ----
    # (trl 1.x / vllm 0.19 / transformers 5.x). Trained + served TEXT-ONLY: the
    # checkpoints are natively multimodal, so LoRA excludes the vision tower and vLLM
    # loads language_model_only (see autoslm.engine.worker). Each entry passed a real
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
        algos=("sft",),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
        thinking="hybrid",
        notes="SFT at micro-batch 1 on a 5090; colocated GRPO does not fit in 32 GB bf16.",
    ),
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B",
        params="36B total / 3B active MoE",
        algos=("sft",),
        min_vram_gb=32,
        recommended_gpu="RTX 5090",
        quant="4bit-qlora",
        experimental=True,
        thinking="hybrid",
        min_disk_gb=160,  # ~72 GB bf16 checkpoint + worker stack + headroom
        notes="QLoRA SFT tier. AutoSLM provisions the bigger worker disk automatically "
        "(min_disk_gb; the 64 GB platform default can't hold the ~72 GB checkpoint). "
        "Validated end-to-end (train+eval) on A100 PCIe; 5090 works when its host "
        "pool bootstraps the stack (or with AUTOSLM_WORKER_IMAGE).",
    ),
}


def list_models(include_experimental: bool = False) -> list[ModelInfo]:
    models = MODELS.values()
    if not include_experimental:
        models = [m for m in models if not m.experimental]
    return sorted(models, key=lambda m: (m.experimental, m.min_vram_gb, m.id))


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

    from autoslm.engine.vram import check_fit

    est = check_fit(model_id, algo, gpu or "RTX 5090")
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
        algos=("sft", "grpo"),
        min_vram_gb=int(est.est_gb) if est.est_gb else 24,
        min_disk_gb=min_disk,
        recommended_gpu=gpu or "RTX 5090",
        experimental=True,
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


def public_model_rows(include_experimental: bool = False) -> list[dict]:
    return [m.to_dict() for m in list_models(include_experimental=include_experimental)]
