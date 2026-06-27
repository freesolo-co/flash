"""Curated model catalog for one-consumer-GPU LoRA jobs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
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
    # Requires the DISAGGREGATED (multi-GPU async) GRPO path: too large to colocate the trainer +
    # vLLM rollout on one GPU. A GRPO request for such a model must set ``[train].inference_gpus>0``
    # on a multi-GPU node (see engine.rollout_bench.validate_disaggregated_requirement); colocate
    # GRPO is rejected at submit. SFT (no rollout engine) is unaffected.
    requires_disaggregated: bool = False
    # The disaggregated trainer cannot be REPLICATED across multiple trainer cards (plain DDP) for
    # this model: the worker's multi-trainer path replicates the whole policy on every trainer rank,
    # and a model this large (e.g. the 35B) blows host RAM / OOMs when loaded once per rank. Such a
    # model must use a SINGLE trainer card (train_gpus = [gpu].count - inference_gpus <= 1), i.e.
    # only 1:N ratios. Rejected at SUBMIT time (validate_disaggregated_requirement) so a multi-trainer
    # ratio fails before renting the paid node instead of crashing the DDP launch mid-run.
    single_trainer_only: bool = False
    # Mixture-of-experts checkpoint. Only MoE models can use the disaggregated rollout's DATA-parallel
    # mode (FLASH_DISAGG_PARALLEL=dp, tp=1 replicas): vLLM rejects offline data parallelism for DENSE
    # models, so the worker (engine.worker.run_rl) downgrades a dense ``dp`` request back to TENSOR
    # parallelism. The submit-time TP head-divisibility guard mirrors that downgrade. Declared per
    # entry so the schema knows, at submit, whether ``dp`` will really be honored.
    is_moe: bool = False
    # Natively-multimodal (vision-language) checkpoint that Flash trains/serves TEXT-ONLY. The
    # COLOCATE rollout engine skips the vision tower (patch_vllm_language_model_only), but the
    # DISAGGREGATED ``trl vllm-serve`` server exposes no language-model-only flag and would load the
    # full model incl. the tower. The only VL model VALIDATED for the disaggregated path is the
    # 35B-A3B (requires_disaggregated, served on H200-class GPUs where the tower fits); for every
    # other VL model the submit-time guard rejects an OPTIONAL disaggregated split. Declared
    # statically so the guard needs no network/config probe at submit.
    is_vl: bool = False
    # Attention-head count, used to REJECT an invalid tensor-parallel disaggregated split at SUBMIT
    # time (before renting): vLLM requires num_attention_heads % inference_gpus == 0 for TP, so e.g.
    # a 16-head model with inference_gpus=3 is impossible. 0 = unknown -> the submit-time check is
    # skipped and the worker's pre-server-boot guard is the catch-all.
    num_attention_heads: int = 0

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
    # ---- Qwen3.6 MoE: the DISAGGREGATED (multi-GPU async) GRPO tier ----
    # GRPO-only: the 35B OOMs when the bf16 trainer and the bf16 vLLM rollout share one card, so its
    # GRPO is rejected for the colocate path (requires_disaggregated) — it needs a dedicated inference
    # GPU + a single trainer card on a multi-GPU node (validate_disaggregated_requirement). bf16, NOT
    # QLoRA: 4-bit was abandoned (the GRPO vLLM rollout merges the LoRA into the 4-bit base, collapsing
    # the importance ratio -> no learning), same reason as the 9B. SFT is intentionally NOT advertised:
    # a 35B SFT's full-sequence activations exceed every validated card, so this entry trains GRPO only.
    "Qwen/Qwen3.6-35B-A3B": ModelInfo(
        id="Qwen/Qwen3.6-35B-A3B",
        display_name="Qwen3.6 35B-A3B (MoE)",
        params="35B total / ~3B active (MoE)",
        # TOTAL parameters (billions). For an MoE checkpoint the size term is the TOTAL count, not the
        # ~3B active: download/VRAM/disk size the FULL checkpoint that lands on the GPU (all experts
        # are materialized). Matches the dense siblings' convention (params "4.7B" -> params_b=4.7).
        params_b=35.0,
        algos=("grpo",),
        min_vram_gb=48,
        grpo_min_vram_gb=80,
        quant="bf16",
        recommended_gpu="A100 PCIe",
        thinking="hybrid",
        # Disaggregated double-load (trainer + a separate ``trl vllm-serve`` both materialize the
        # ~70 GB bf16 checkpoint) plus HF download + Xet temp + checkpoint saves push PEAK disk to
        # ~200 GB. Floor to the live-validated 300 GB so a paid multi-GPU rent doesn't hit "No space
        # left on device"; the runner raises gpu.disk_gb to this out of the box.
        min_disk_gb=300,
        # GRPO is DISAGGREGATED-ONLY: colocating the trainer + vLLM rollout on one card OOMs.
        requires_disaggregated=True,
        # MoE: the only catalog model that supports FLASH_DISAGG_PARALLEL=dp (vLLM rejects offline
        # data parallelism for dense models).
        is_moe=True,
        # 16 attention heads (text_config): a 1:3 split (inference_gpus=3) is invalid under TP
        # (16 % 3 != 0) and is rejected at submit.
        num_attention_heads=16,
        # VL (qwen3_5_moe) but the only VL model VALIDATED for the disaggregated rollout — it is
        # requires_disaggregated and runs on H200-class GPUs where the full model (incl. the vision
        # tower the server can't skip) fits and the consumer-card PTX issue doesn't apply.
        is_vl=True,
        # The disaggregated trainer is plain DDP (replicated per trainer rank); the 35B is too large
        # to load once PER trainer card (host-RAM/OOM), so only SINGLE-trainer ratios (1:N) are valid.
        single_trainer_only=True,
        notes="MoE; GRPO requires the disaggregated multi-GPU node ([train].inference_gpus>0), "
        "single-trainer (1:N) only. The 35B is served on a dedicated inference GPU while the "
        "bf16 LoRA trainer runs on ONE other card.",
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
    train=None,
) -> ModelInfo:
    """Resolve a model under the configured policy.

    ``catalog`` (default): the model must be a curated catalog entry.
    ``allow``: any HF model is accepted; a coarse VRAM-fit estimate (HF safetensors
    metadata, no download) blocks only provably-impossible fits and warns on tight ones.

    ``train`` (TrainSpec or raw [train] dict) supplies the disaggregated context: for a
    disaggregated GRPO run ([train].inference_gpus>0) the open-model fit check must use the
    per-GPU split sizing (allocator.required_vram_gb), not the colocate total, so a large HF
    model that fits as a split is not rejected as too_big here.
    """
    algo = normalize_algorithm(algorithm)
    if model_id in MODELS:
        return validate_model_for_algorithm(model_id, algo)
    if policy != "allow":
        # Reuse get_model's error (includes the open-model hint).
        return get_model(model_id)
    return _resolve_open_model(model_id, algo, gpu, train=train)


def _resolve_open_model(model_id: str, algo: str, gpu: str | None, *, train=None) -> ModelInfo:
    """Synthesize a ModelInfo for the open-model "allow" policy from a coarse VRAM-fit
    estimate (HF safetensors metadata, no download). Blocks provably-impossible fits and
    warns on tight ones. Isolates the engine.vram dependency + disk-floor heuristic from
    the curated-catalog path in resolve_model."""
    from flash.engine.vram import check_fit

    est = check_fit(model_id, algo, gpu or DEFAULT_GPU)
    # Disaggregated GRPO ([train].inference_gpus>0) splits the model across the node's GPUs, so
    # the per-card need is far below the colocate total. check_fit's verdict is colocated, so a
    # big HF model sized for a split would be wrongly rejected as too_big here — exactly the GPU
    # the disaggregated-aware resolve_gpu_policy/allocator already sized for. Re-check the fit
    # against the SAME disaggregated per-GPU estimate (allocator.required_vram_gb) the policy GPU
    # resolution and submit-time allocator use, so the two paths agree. (Falls back to the plain
    # colocate verdict when inference_gpus==0 or the param count is unreadable.)
    _ig = 0
    if train is not None:
        from flash.spec import strict_int

        _raw_ig = train.get("inference_gpus") if isinstance(train, dict) else getattr(train, "inference_gpus", 0)
        try:
            _ig = strict_int(_raw_ig or 0, name="train.inference_gpus", minimum=0)
        except (TypeError, ValueError):
            _ig = 0
    if est.verdict == "too_big" and _ig > 0 and est.est_gb:
        from flash.engine.vram import GPU_VRAM_GB
        from flash.providers.allocator import required_vram_gb

        disagg_need = required_vram_gb(model_id, algo, train=train)
        gpu_gb = GPU_VRAM_GB.get(gpu or DEFAULT_GPU, 32)
        if disagg_need <= gpu_gb * 1.15:
            # Fits as a disaggregated split on this per-role card: clear the colocate too_big and
            # let provisioning size the multi-GPU node. Surface the split need so the operator sees
            # the per-GPU figure rather than the (rejected) colocate total.
            print(
                f"warning: open-model policy ({model_id}): colocate estimate exceeds the GPU but a "
                f"disaggregated split ([train].inference_gpus={_ig}) needs ~{disagg_need} GB/GPU "
                f"<= {gpu or DEFAULT_GPU} ({gpu_gb} GB) — proceeding as a split rollout."
            )
            est = replace(est, verdict="tight")
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
    #
    # A DISAGGREGATED split (we just cleared too_big via inference_gpus>0 above) needs a far
    # higher floor than this colocate heuristic: the trainer AND a separate `trl vllm-serve`
    # process BOTH materialize the full bf16 checkpoint on the SAME node, and the HF download
    # (+ Xet temp + per-step checkpoint saves) pushes PEAK disk well past a single copy — the
    # exact failure mode the curated 35B entry floors to 300 GB for (~70 GB single checkpoint).
    # Mirror that ratio for unlisted split models so they don't provision a paid multi-GPU node
    # and then die with "No space left on device": the trainer + server materialize TWO bf16
    # copies (~4 GB/param) and the HF download lands a third (~2 GB/param) -> ~6 GB/param, plus
    # Xet-temp / per-step-checkpoint-save headroom (~96 GB). That reproduces the curated 35B
    # entry's validated 300 GB floor (6*35 + 96 ~= 306) and scales down for smaller splits.
    if est.params_b:
        # A disaggregated split materializes multiple full model copies on the same node
        # (trainer + `trl vllm-serve` + the HF download) regardless of the per-card VRAM
        # verdict, so the elevated disk floor must key off inference_gpus alone — a model
        # that comfortably "fits" the per-role card still needs the multi-copy headroom.
        _split = _ig > 0
        per_param = 6 if _split else 2
        headroom = 96 if _split else 64
        # ceil (not int/truncate): a fractional param estimate must round the floor UP so we never
        # under-provision disk by several GB — the exact failure ("No space left on device" mid-
        # download) this floor exists to prevent.
        min_disk = math.ceil(est.params_b * per_param) + headroom
    else:
        min_disk = 0
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
