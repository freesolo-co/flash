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


# The default GPU class used as the open-model-policy
# sizing reference and the spec/from_dict fallback). The managed GPU class set (KNOWN)
# lives in providers.base; RunPod pricing lives under providers/runpod. Defined above
# ModelInfo so it can back the recommended_gpu field default.
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
    # Total parameters in billions — the numeric model size the cost estimator + VRAM equations read
    # DIRECTLY (no parsing of the ``params`` display string). Drives the memory/size terms (VRAM, disk,
    # download), which always size the FULL checkpoint. REQUIRED: every ModelInfo must state it — a
    # curated catalog model sets its true count, and the open-model policy passes the HF/estimated count
    # (or 0.0 when the size is genuinely "unknown size"). ``test_catalog`` asserts every curated MODELS
    # entry sets it > 0, so a new entry can never silently fall back to a parsed string again.
    params_b: float
    quant: str = "bf16"
    recommended_gpu: str = DEFAULT_GPU
    # GRPO needs more VRAM than SFT (a colocated vLLM rollout engine holds a second copy of
    # the weights + KV cache). 0 => GRPO uses ``min_vram_gb`` like SFT; set it when the GRPO
    # tier needs a bigger card than SFT (the colocate 2nd weight copy + KV pool). Consumed by
    # engine.vram.model_required_vram_gb.
    grpo_min_vram_gb: int = 0
    # SFT hard VRAM floor (GB). 0 => SFT sizes purely from the param-based estimate and is free to
    # down-route to a smaller validated card (the default — e.g. a 4B SFT estimates ~17 GB and rents
    # a 48 GB card, NOT its ``min_vram_gb`` reference). Set it ONLY when a curated model must not be
    # placed on the cheapest card the estimate would otherwise allow — e.g. a very large checkpoint
    # whose ~param-est margin over the frozen-weights floor is too thin on the next card down.
    # Consumed by engine.vram.model_required_vram_gb (the SFT analog of ``grpo_min_vram_gb``).
    sft_min_vram_gb: int = 0
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
    # Parameters ACTIVE per token in billions — only meaningful for an MoE, where a token routes
    # through a small subset of experts. The cost estimator's per-token FLOPs/step-time term reads
    # this (a token exercises only the active params), while VRAM/disk/download keep using the total
    # ``params_b``. 0.0 (the dense default) means "same as params_b" — every token hits every param.
    active_params_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The default model Flash trains when a config omits one. A current-gen dense 4B
# (text-only fine-tune) on the modern worker stack — the safe out-of-the-box choice for
# the average developer. It is thinking-"hybrid"; the thinking flag defaults OFF.
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
    # Qwen3.5 dense family: validated on the modern worker stack
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
        min_vram_gb=48,
        # bf16 LoRA (NOT QLoRA). 4-bit QLoRA was abandoned for the 9B because the GRPO vLLM
        # rollout MERGES the LoRA into the 4-bit base (peft bnb merge), and that rounding makes
        # the sampler policy diverge from the bf16 trainer -> TRL importance-sampling ratio
        # collapses to 0 (no learning) + runaway/non-terminating generations. bf16 keeps the
        # rollout and trainer in the same precision so GRPO actually learns. Costs a bigger GPU:
        # ~19 GB weights; SFT fits a 48 GB card, colocated GRPO (two bf16 copies + KV + the
        # 248k-vocab fp32 logits) needs an 80 GB class -> grpo_min_vram_gb floor below.
        grpo_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
        thinking="hybrid",
        notes="bf16 LoRA. ~19 GB of weights; SFT fits a 48 GB card, while colocated GRPO "
        "(two bf16 copies + KV + the 248k-vocab fp32 logits) needs an 80 GB-class card "
        "(grpo_min_vram_gb floor).",
    ),
    # ---- Qwen3.6 MoE: the big-checkpoint tier (H200 for SFT, B200 for GRPO) ----
    # 35B-A3B is a Mixture-of-Experts checkpoint: ~3B parameters are ACTIVE per token, but all 35B
    # are materialized on the GPU, so the MEMORY/disk/download terms size the FULL 35B (~70 GB bf16)
    # while the COMPUTE terms (activations, KV pool, rank-linear LoRA) size the ~3B active backbone
    # (engine.vram is MoE-aware via active_params_b). bf16 LoRA, NOT QLoRA — same reason as the 9B.
    # Because the resident weights dominate and the active compute is tiny, the GPU tier is set by
    # how many weight copies each algorithm holds, NOT by context length:
    #   * SFT — ONE ~70 GB copy + small active-compute (~82 GB peak, ~flat in context) -> fits the
    #     141 GB H200 with wide margin (context ~unbounded by VRAM). Live-validated on a B200; the
    #     H200 down-tier is the MoE-aware win (cheaper, plentiful stock).
    #   * GRPO — colocates the vLLM rollout, so TWO ~70 GB copies (trainer + engine) are resident at
    #     the rollout peak (~167 GB) -> needs the 180 GB B200; the H200 can't hold both. The MoE
    #     rollout weight-sync needed a fused-expert name fix (engine.worker.lora._remap_vl_sync_weights
    #     passes the multimodal ``model.language_model.*`` names through to vLLM's own mapper). Both
    #     single- and multi-turn GRPO live-validated on a B200.
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B (MoE)",
        params="35B total / ~3B active (MoE)",
        # TOTAL parameters (billions) the SFT VRAM equation + cost projection read. For an MoE
        # checkpoint the size term is the TOTAL count, not the ~3B active: download/VRAM/disk size the
        # FULL checkpoint that lands on the GPU (all experts are materialized). 35.0 is the CALIBRATED
        # total: the live-validated single-B200 SFT fit depends on it — the honest-peak equation lands
        # at the 180 GB B200's usable budget, and the marketing "~35.95B" figure tips it over (186 GB,
        # see test_sft_equation_covers_honest_peak_across_seq_boundary). Keep 35.0.
        params_b=35.0,
        # ~3B ACTIVE per token (the "A3B" in the name): a token routes through a small subset of
        # experts, so cost/step-time FLOPs scale with ~3B, not the 35B total. Without this the
        # estimator would price SFT as if every token exercised all 35B params — ~10x too slow/costly.
        active_params_b=3.0,
        vocab_size=248_320,
        algos=("sft", "grpo"),
        min_vram_gb=141,
        # Hard SFT floor: with MoE-aware sizing the SFT estimate is ~82 GB (the 70 GB resident weights
        # dominate; the active-3B activations/KV are tiny), which would otherwise down-route to the
        # 96 GB RTX Pro 6000 (consumer Blackwell, thin margin over the 70 GB base) or the 80 GB H100
        # (too tight). Floor to 100 GB so SFT lands on the 141 GB H200 — a datacenter card with wide
        # margin, ~$1.50/hr cheaper than the B200 and not needed here.
        sft_min_vram_gb=100,
        # GRPO floor = the 180 GB B200 (colocated GRPO holds two ~70 GB weight copies + a KV pool; the
        # 141 GB H200 can't hold the trainer + vLLM rollout). The base ~167 GB two-copy estimate already
        # routes GRPO to the B200, but setting the floor ALSO ENGAGES the long-context escalation —
        # model_required_vram_gb only adds grpo_seq_escalation_gb when a grpo floor is set. The
        # escalation keys on the ~3B ACTIVE params, so default/moderate GRPO still fits the B200 but a
        # long (>~16k-token, e.g. 32k) rollout is sized PAST 180 GB and rejected at parse time, instead
        # of booting a B200 and OOMing in vLLM's KV allocation.
        grpo_min_vram_gb=180,
        quant="bf16",
        recommended_gpu="H200",
        thinking="hybrid",
        # ~70 GB bf16 checkpoint. Peak disk = HF download (~70 GB) + Xet temp (~70 GB) + per-step
        # deployable-checkpoint saves; floor to 200 GB so the rent doesn't hit "No space left on
        # device" (the runner raises gpu.disk_gb to this out of the box).
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
    # Catalog entries advertise the capability classes "sft" and "grpo": grpo needs the
    # colocated rollout engine, sft is trainer-only.
    required = "grpo" if algo == "grpo" else "sft"
    if required not in info.algos:
        allowed = ", ".join(info.algos)
        raise ValueError(f"{model_id} supports {allowed}, not {algo}")
    return info


def public_model_rows() -> list[dict[str, Any]]:
    return [m.to_dict() for m in list_models()]
