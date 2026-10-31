"""Conservative architecture-aware VRAM fit estimation for single-GPU LoRA jobs."""

from __future__ import annotations

import json
import math
import os

_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
}

_BASE_OVERHEAD_GB = 4.0
_ACT_COEF = 0.12
_SFT_PER_DEVICE_BS_DEFAULT = 4
_VOCAB_DEFAULT = 248_320
# must track verl's `FusedLinearForPPO(chunk_size=...)` default
# (verl/utils/experimental/torch_functional.py). the child projects the vocab in chunks of this
# many token rows, so a smaller value here under-reserves and admits a job that then OOMs.
VERL_FUSED_CE_CHUNK_TOKENS = 512


def _sft_per_device_bs() -> int:
    """Worker's per-device SFT micro-batch cap — always the fixed default, never read from env."""
    return _SFT_PER_DEVICE_BS_DEFAULT


def sft_grad_accum(
    batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True
) -> tuple[int, int]:
    """(per-device micro-batch, grad-accum steps) for a requested global batch_size."""
    target = max(1, int(batch_size))
    per_device = sft_per_device(target, seq_len=seq_len, vocab=vocab, fused=fused)
    grad_accum = max(1, -(-target // per_device))  # ceil
    return per_device, grad_accum


# SFT uses fp32 AdamW because bitsandbytes fails on FSDP2 DTensors. GRPO's lower adapter
# allowance is part of its calibrated resident-peak model and must change with that calibration.
_LORA_PAGED_BYTES_PER_PARAM = 10.0
_LORA_ADAMW_BYTES_PER_PARAM = 16.0


def grpo_completion_len(max_tokens: int | None, thinking: bool) -> int:
    """The completion-token budget a GRPO run uses: an explicit ``max_tokens`` else the RL recipe
    default (thinking uses the longer ``max_completion_len_thinking``). Single source of truth for
    the three sites that must resolve the SAME integer (``_resolve_sequence_lengths``'s worker-side
    enforcement, ``grpo_rollout_seq_len``, and the spec-parse prompt-budget guard)."""
    from flash.engine.plan.recipe import RECIPE

    rl = RECIPE.rl
    return int(
        max_tokens or (rl.max_completion_len_thinking if thinking else rl.max_completion_len)
    )


def grpo_rollout_seq_len(
    max_length: int = 0,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> int:
    """vLLM engine context a GRPO run uses, mirroring run_rl() — shared by allocator, sleep gate, and KV budget."""
    from flash.engine.plan.recipe import RECIPE

    completion = grpo_completion_len(max_tokens, thinking)
    return int(max_length or max(1024, RECIPE.rl.max_prompt_len + completion))


def _legacy_lora_floor_gb(lora_rank: int, effective_params_b: float) -> float:
    """Measured floor retained until the shape equation has broader live-GPU calibration."""
    return (lora_rank / 16.0) * (0.3 + 0.04 * effective_params_b)


def _lora_parameter_count(
    lora_rank: int, model_info=None, *, tensor_parallel: int = 1
) -> int | None:
    """lora parameters resident on one tensor-parallel rank from concrete target geometry."""
    shapes = tuple(getattr(model_info, "lora_target_shapes", ()) or ())
    rank = int(lora_rank)
    if not shapes or rank <= 0:
        return None
    tp_size = max(1, int(tensor_parallel))
    if tp_size == 1:
        return rank * sum((int(i) + int(o)) * int(count) for i, o, count in shapes)
    # vllm shards one factor, except the current min-dim-1 gate keeps that factor whole. integer shard
    # sizing covers that catalog shape exactly; this is not a general replicated-layer classifier.
    return sum(
        rank * int(count) * (max(int(i), int(o)) + math.ceil(min(int(i), int(o)) / tp_size))
        for i, o, count in shapes
    )


def _lora_weight_memory_gb(
    lora_rank: int, model_info=None, *, tensor_parallel: int = 1
) -> float | None:
    """rank-local resident bf16 lora weights, or none without catalog target geometry."""
    params = _lora_parameter_count(lora_rank, model_info, tensor_parallel=tensor_parallel)
    return None if params is None else params * _BYTES_PER_PARAM["bf16"] / 1e9


def _lora_memory_gb(
    lora_rank: int,
    effective_params_b: float,
    algorithm: str,
    model_info=None,
) -> float:
    """Adapter, gradient, and optimizer memory for the actual PEFT all-linear targets."""
    floor = _legacy_lora_floor_gb(lora_rank, effective_params_b)
    trainable_params = _lora_parameter_count(lora_rank, model_info)
    if trainable_params is None:
        return floor
    bytes_per_param = (
        _LORA_PAGED_BYTES_PER_PARAM
        if (algorithm or "").lower() in ("grpo", "rl")
        else _LORA_ADAMW_BYTES_PER_PARAM
    )
    exact = trainable_params * bytes_per_param / 1e9
    return max(floor, exact)


_TRAIN_COEF = 0.27
_VLLM_COLOCATE_FLOOR_GB = 28.0
_LOGITS_BUDGET_GB = 6.0
# 16 B/elem: fp32 logits+grad + bf16 logits+grad + CE temp. 8 B/elem under-counts (live OOM confirmed).
_SFT_LOGITS_BYTES_PER_ELEM = 16.0
_SFT_CHUNKED_NLL_TOKENS = VERL_FUSED_CE_CHUNK_TOKENS
# thresholds for the worker gradient-checkpointing memory gate.
_CHECKPOINT_MIN_PARAMS_B = 3.0
_CHECKPOINT_LONG_CTX_TOKENS = 2048

# these model types have validated dense-logit-free fused SFT loss; others retain plain NLL.
# verl dispatches `qwen3_5` and `qwen3_5_moe` through its fused torch backend. liger remains off
# because it zeroed SFT LoRA gradients (GRAD-001). KimiVL exits before the fused patch, so do not
# add it without parity coverage.
_SFT_CHUNKED_NLL_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-35B-A3B",
    }
)


def sft_chunked_nll_enabled(model_id: str) -> bool:
    """whether the sft worker uses a dense-logit-free fused loss path."""
    return model_id in _SFT_CHUNKED_NLL_MODELS


def sft_logits_per_device_cap(seq_len: int, vocab: int) -> int:
    """Largest per-device SFT micro-batch whose un-fused fp32 logits fit _LOGITS_BUDGET_GB."""
    denom = max(1, int(seq_len)) * max(1, int(vocab)) * _SFT_LOGITS_BYTES_PER_ELEM
    return max(1, int(_LOGITS_BUDGET_GB * 1e9 / denom))


def sft_per_device(batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True) -> int:
    """Per-device SFT micro-batch: capped at 4, further vocab-sized when fused CE is off."""
    per_device = max(1, min(_SFT_PER_DEVICE_BS_DEFAULT, max(1, int(batch_size))))
    if not fused and seq_len and vocab:
        per_device = min(per_device, sft_logits_per_device_cap(seq_len, vocab))
    return per_device


def grpo_seq_escalation_gb(params_b: float | None, seq_len: int) -> int:
    """Extra GB for long-context GRPO beyond base footprint (9.7B fits 80 GB to seq 4096, OOMs at 8192)."""
    coef = 0.9
    if not params_b:
        return 0
    seq_thresh = 48_500.0 / params_b
    if seq_len <= seq_thresh:
        return 0
    return math.ceil(coef * params_b * (seq_len / seq_thresh - 1))


def _declares_linear_attention(model_info, model_id: str = "") -> bool:
    """True for a GDN hybrid, using the catalog rather than a network config fetch.

    A GDN worker rejects fp8 KV only when its rollout engine sleeps, because the crash is vllm's
    ``init_fp8_kv_scales`` on wake; a model the catalog pins resident keeps fp8. Resident models use their separately calibrated sizing floor. Fall back to catalog lookup
    by model id for pinned revisions because attention family is revision-independent.
    """
    if int(getattr(model_info, "num_linear_attention_layers", 0) or 0) > 0:
        return True
    if not model_id:
        return False
    from flash.core.catalog import MODELS

    catalog_info = MODELS.get(model_id)
    return int(getattr(catalog_info, "num_linear_attention_layers", 0) or 0) > 0


def _rollout_kv_floor_gb(
    params_b: float,
    vllm_max_len: int,
    concurrency: int,
    *,
    active_params_b: float | None = None,
    model_info=None,
    model_id: str = "",
    lora_rank: int = 0,
    rollout_cache_min_gb: float = 0.0,
) -> int:
    floor = grpo_kv_floor_gb(
        params_b,
        vllm_max_len,
        concurrency,
        active_params_b=active_params_b,
        model_info=model_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
        lora_rank=lora_rank,
    )
    from flash.providers.core.base import max_non_fp8_kv_vram_gb

    if _declares_linear_attention(model_info, model_id):
        return floor
    ceiling = max_non_fp8_kv_vram_gb()
    if floor <= ceiling:
        return floor
    # the bf16 floor exceeds the non-fp8 ceiling, so the run must land on an fp8-kv GPU. use the
    # (smaller) fp8 floor rather than the oversized bf16 floor, but keep the routed requirement
    # strictly above the ceiling: a fp8 floor that dips back under it would let routing pick a
    # bf16-kv GPU that then OOMs on the real bf16 cache the fp8 estimate did not reserve.
    fp8_floor = grpo_kv_floor_gb(
        params_b,
        vllm_max_len,
        concurrency,
        active_params_b=active_params_b,
        fp8_kv=True,
        model_info=model_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
        lora_rank=lora_rank,
    )
    return max(fp8_floor, ceiling + 1)


def estimate_vram_gb(
    params_b: float,
    algorithm: str,
    quant: str = "bf16",
    *,
    seq_len: int = 1024,
    max_tokens: int | None = None,
    lora_rank: int = 32,
    batch_size: int = 1,
    group_size: int = 8,
    thinking: bool = False,
    vocab: int = _VOCAB_DEFAULT,
    sleep_offload: bool = True,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    sft_fused_ce: bool | None = None,
    model_info=None,
    rollout_cache_min_gb: float = 0.0,
) -> float:
    """Estimated peak VRAM (GB) for a LoRA job on one GPU.

    MoE: active_params_b drives activations; weights and actual LoRA targets cover the full model.
    """
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    eff_b = float(active_params_b) if active_params_b else params_b
    algo = "grpo" if (algorithm or "").lower() in ("grpo", "rl") else "sft"
    width = math.sqrt(max(eff_b, 0.1))
    lora_opt = _lora_memory_gb(lora_rank, eff_b, algorithm, model_info)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    if algo == "grpo":
        # Sleep mode: peak = max(rollout, train). Resident: both live at once, peak = sum.
        kv = _resident_kv_gb(
            eff_b,
            seq_len,
            group_size,
            fp8_kv=fp8_kv,
            model_info=model_info,
            rollout_cache_min_gb=rollout_cache_min_gb,
        )
        if sleep_offload:
            # Mirror the worker's 1.5x allowance for geometry-based sizing or runtime floors.
            # Cap the parameter-only estimate, calibrated to measured train/OOM boundaries.
            arch_kv_known = (
                _architecture_kv_raw_gb(model_info, seq_len, group_size, fp8_kv) is not None
            )
            if (arch_kv_known or rollout_cache_min_gb) and kv > _KV_CAP:
                kv = 1.5 * kv
            else:
                kv = min(kv, _KV_CAP)
        rollout = weights + kv
        group_factor = max(1.0, (max(1, group_size) / 4.0) ** 0.5)
        think_factor = 1.3 if thinking else 1.0
        activations = _TRAIN_COEF * (seq_len / 1024.0) * width * group_factor * think_factor
        completion = max_tokens if max_tokens else min(seq_len, 1024)
        logits = min(completion * vocab * 4 / 1e9, _LOGITS_BUDGET_GB)
        train = activations + logits
        return base + (max(rollout, train) if sleep_offload else rollout + train)
    # direct callers default to the conservative plain-nll estimate. model_required_vram_gb passes
    # the worker's validated chunked-nll decision explicitly when it knows the model identity.
    fused = False if sft_fused_ce is None else bool(sft_fused_ce)
    pd = sft_per_device(batch_size, seq_len=seq_len, vocab=vocab, fused=fused)
    activations = _ACT_COEF * pd * (seq_len / 1024.0) * width
    # plain nll retains every sequence position. chunked nll retains at most one
    # VERL_FUSED_CE_CHUNK_TOKENS-token vocab projection at a time, independent of micro-batch and
    # context once the chunk is full.
    projected_tokens = min(pd * seq_len, _SFT_CHUNKED_NLL_TOKENS) if fused else pd * seq_len
    logits = projected_tokens * vocab * _SFT_LOGITS_BYTES_PER_ELEM / 1e9
    return base + activations + logits


# gc-off activation factors are empirical and geometry-specific; the estimate scales with
# layers*batch*seq*hidden. dense 65.0 is a live RTX 5090 fit against successful peaks. remeasure
# when changing the activation estimate or its safety margin.
#
# the MoE factor was 18.0 -- a guess, carried on the assumption that ~3B active params imply small
# activations. that is wrong: routing is per-token, so every one of the 40 layers still
# materializes a full activation set, and the wide expert stack makes each one large. 18.0 called a
# run that OOMs a fit, on BOTH an H200 and a B200.
#
# 196.9 is NOT a measured activation cost: it is the residual that lands this equation's TOTAL on
# the allocated-at-OOM boundary of a live B200 run. Re-derive it the same way, never by measuring
# activations alone. The run evidence, the unit trap, and the reason the naive
# "allocated minus resident" subtraction is wrong all live in the test that pins this boundary,
# test_moe_activation_constant_matches_the_live_b200_peak.
_GC_OFF_ACT_K_DENSE = 65.0
_GC_OFF_ACT_K_MOE = 196.9


def sft_gc_off_peak_gb(
    params_b: float,
    *,
    active_params_b: float | None,
    seq_len: int,
    hidden: int,
    num_layers: int,
    batch: int = _SFT_PER_DEVICE_BS_DEFAULT,
    lora_rank: int = 32,
    quant: str = "bf16",
    model_info=None,
) -> float:
    """Estimated peak VRAM (GB) for a dense-logit-free LoRA SFT step without checkpointing.

    Includes resident weights, optimizer/base, and all-layer activations; unknown geometry returns
    `inf`. MoE activations use hidden/layer geometry while weights retain full parameter size.
    """
    sizes = (params_b, hidden, num_layers, seq_len, batch, lora_rank)
    if any(not math.isfinite(value) or value <= 0 for value in sizes):
        return float("inf")
    if quant not in _BYTES_PER_PARAM or (
        active_params_b is not None
        and (not math.isfinite(active_params_b) or not 0 <= active_params_b <= params_b)
    ):
        return float("inf")
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    eff_b = float(active_params_b) if active_params_b else float(params_b)
    weights = float(params_b) * bpp
    lora_opt = _lora_memory_gb(lora_rank, eff_b, "sft", model_info)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    # a truthy active_params_b IS the sparse-MoE signal -- it is set only for a model whose active
    # parameter count differs from its total, which is what makes the per-layer FFN activation small.
    act_k = _GC_OFF_ACT_K_MOE if active_params_b else _GC_OFF_ACT_K_DENSE
    act = act_k * int(num_layers) * int(batch) * int(seq_len) * int(hidden) * 2.0 / 1e9
    return base + act


def sft_grad_checkpoint_can_disable(
    params_b: float,
    *,
    active_params_b: float | None,
    seq_len: int,
    hidden: int,
    num_layers: int,
    card_vram_gb: float,
    batch: int = _SFT_PER_DEVICE_BS_DEFAULT,
    lora_rank: int = 32,
    quant: str = "bf16",
    margin_gb: float | None = None,
    model_info=None,
) -> bool:
    """True when a dense-logit-free LoRA SFT step fits a card WITHOUT gradient checkpointing.

    Unknown geometry or insufficient headroom keeps checkpointing on. Dense workloads reserve
    8 GB beyond the estimated peak; MoE retains its 18 GB margin calibrated against live OOMs.
    These margins depend on the activation model, not a model ID, revision, or GPU family.
    """
    if margin_gb is None:
        margin_gb = 18.0 if active_params_b else 8.0
    if not math.isfinite(card_vram_gb) or card_vram_gb <= 0:
        return False
    if not math.isfinite(margin_gb) or margin_gb < 0:
        return False
    peak = sft_gc_off_peak_gb(
        params_b,
        active_params_b=active_params_b,
        seq_len=seq_len,
        hidden=hidden,
        num_layers=num_layers,
        batch=batch,
        lora_rank=lora_rank,
        quant=quant,
        model_info=model_info,
    )
    return peak + float(margin_gb) <= float(card_vram_gb)


def _config_geometry(config: dict) -> tuple[int, int, int, int]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return (
        int(text.get("vocab_size") or config.get("vocab_size") or 0),
        int(text.get("hidden_size") or config.get("hidden_size") or 0),
        int(text.get("num_hidden_layers") or config.get("num_hidden_layers") or 0),
        # QUERY heads. vllm tensor parallelism requires this to divide the card count handed to the
        # rollout engine, so it decides how wide a pinned run may be -- see
        # `allocator.geometry_safe_gpu_cap`.
        int(text.get("num_attention_heads") or config.get("num_attention_heads") or 0),
    )


def fetch_hf_model_geometry(
    model_id: str, revision: str = "", *, strict: bool = False
) -> tuple[float | None, int, int, int, int]:
    """Return revision-aware params and config geometry from Hugging Face."""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = os.environ.get("HF_TOKEN")
        info_kwargs = {"revision": revision} if revision else {}
        info = HfApi(token=token).model_info(
            model_id,
            expand=["safetensors"],
            **info_kwargs,
        )
        total = getattr(getattr(info, "safetensors", None), "total", None)
        params_b = float(total) / 1e9 if total else None
        download_kwargs = {"revision": revision} if revision else {}
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=token,
            **download_kwargs,
        )
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("model config is not an object")
        vocab, hidden, layers, heads = _config_geometry(config)
        return params_b, vocab, hidden, layers, heads
    except Exception as exc:
        if strict:
            raise ValueError(
                f"could not resolve revision-specific sizing metadata for model {model_id!r}"
            ) from exc
        return None, 0, 0, 0, 0


def _sizing_value(obj, key):
    """Return a sizing input from a mapping or object."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _optimizer_batch_value(train, algorithm: str):
    """Return the authored optimizer batch under whichever key this algorithm accepts.

    The optimizer batch is one sizing input here but two config keys: sft authors `batch_size`,
    grpo authors `prompts_per_step`, and the schema rejects each name under the other algorithm.
    Reading only `batch_size` therefore sized every rl run on the recipe default no matter what the
    user wrote, because an rl spec's `batch_size` is always None -- so a wide `prompts_per_step`
    silently under-provisioned the card instead of escalating it.

    Sizing runs before the train validators, so a spec carrying both keys cannot be assumed
    rejected yet; take the larger rather than trusting one, since under-sizing is the failure that
    OOMs a paid run and over-sizing only costs a bigger card.
    """
    from flash.core.catalog import optimizer_batch_key

    # the algorithm's OWN key comes from the one helper the writers use too (`RunConfig.train_knobs`
    # emits under it, ranking reads under it), so the read and write sides cannot drift apart.
    own = optimizer_batch_key(algorithm)
    names = (own, "batch_size") if own != "batch_size" else ("batch_size",)
    values = [
        v for v in (_positive_int_or_default(_sizing_value(train, n), None) for n in names) if v
    ]
    return max(values) if values else None


def _positive_int_or_default(v, default):
    """Return a positive integer or the provided fallback."""
    try:
        if isinstance(v, bool):
            return default
        f = float(v)
        return int(f) if math.isfinite(f) and f >= 1 else default
    except (TypeError, ValueError):
        return default


def _model_vram_need(
    params_b: float,
    algorithm: str,
    *,
    seq_len: int,
    max_tokens: int | None,
    lora_rank: int,
    batch_size: int,
    group_size: int,
    thinking: bool,
    sft_fused_ce: bool | None,
    headroom: float,
    quant: str = "bf16",
    vocab: int = _VOCAB_DEFAULT,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    model_info=None,
    rollout_cache_min_gb: float = 0.0,
) -> int:
    """Estimate the headroom-adjusted VRAM requirement."""
    est = estimate_vram_gb(
        params_b,
        algorithm,
        quant,
        seq_len=seq_len,
        max_tokens=max_tokens,
        lora_rank=lora_rank,
        batch_size=batch_size,
        group_size=group_size,
        thinking=thinking,
        vocab=vocab,
        active_params_b=active_params_b,
        fp8_kv=fp8_kv,
        sft_fused_ce=sft_fused_ce,
        model_info=model_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
    )
    return math.ceil(est * headroom)


def _catalog_model_required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    info,
    model_vocab: int,
    model_revision: str,
    seq_len: int,
    max_tokens: int | None,
    lora_rank: int,
    batch_size: int,
    group_size: int,
    thinking: bool,
    headroom: float,
    is_grpo: bool,
    sft_fused_ce: bool | None,
) -> int:
    """Size a catalog model with its run-specific floors."""
    params_b = info.params_b
    if model_revision:
        params_b, model_vocab = _validated_revision_geometry(model_id, model_revision, info)
    quant = getattr(info, "quant", "bf16") or "bf16"
    # Recipe sizing drops unverified layer geometry for pinned revisions. Custom training
    # uses the catalog adapter targets even for its validated immutable worker revision.
    # Dropping them would lose the full MoE expert-state allowance at higher ranks.
    sizing_info = info if not model_revision or algorithm == "custom" else None
    rollout_cache_min_gb = getattr(info, "rollout_cache_min_gb", 0.0)
    active_b = float(getattr(sizing_info, "active_params_b", 0.0) or 0.0)
    need = _model_vram_need(
        params_b or 4.0,
        algorithm,
        seq_len=seq_len,
        max_tokens=max_tokens,
        lora_rank=lora_rank,
        batch_size=batch_size,
        group_size=group_size,
        thinking=thinking,
        sft_fused_ce=sft_fused_ce,
        headroom=headroom,
        quant=quant,
        vocab=model_vocab,
        active_params_b=active_b,
        model_info=sizing_info,
        rollout_cache_min_gb=rollout_cache_min_gb,
    )
    floor = 0
    if is_grpo and getattr(info, "grpo_min_vram_gb", 0):
        floor = int(info.grpo_min_vram_gb)
    if not is_grpo and getattr(info, "sft_min_vram_gb", 0):
        floor = max(floor, int(info.sft_min_vram_gb))
    # Escalate on active_params_b for MoE: keying on total would over-reject (35B total's
    # threshold is below default rollout length); ~3B active gives ~16k headroom.
    if is_grpo and floor:
        if getattr(info, "sleep_unsupported", False):
            # sleep HANGS for this model, so size the resident peak with fp8 KV on sm100.
            # push nonresident fits past every GPU and reject them before the broken sleep path.
            resident_need = math.ceil(
                estimate_vram_gb(
                    params_b or 4.0,
                    "grpo",
                    quant,
                    seq_len=seq_len,
                    max_tokens=max_tokens,
                    lora_rank=lora_rank,
                    group_size=group_size,
                    thinking=thinking,
                    vocab=model_vocab,
                    sleep_offload=False,
                    active_params_b=active_b,
                    fp8_kv=True,
                    model_info=sizing_info,
                    rollout_cache_min_gb=rollout_cache_min_gb,
                )
                # 1.15 resident margin (NOT the looser 1.1 headroom) so the parse-time reject
                # lands at the SAME resident wall the worker gate enforces.
                * 1.15
            )
            floor = max(floor, resident_need)
            # ``need`` above was sized with the default sleep estimate. Now that the sleep rollout
            # honestly reserves the worker's colocate KV pool (max(_KV_CAP, 1.5 * arch KV)), that
            # estimate can exceed this resident wall and FALSELY reject a config that fits resident
            # on the big floor card. Sleep never runs for this model (it HANGS), so discard the
            # sleep sizing and size purely on the resident peak.
            need = floor
        else:
            floor += grpo_seq_escalation_gb(active_b or params_b, seq_len)
    need = max(need, floor)
    if is_grpo:
        floor_gb = 24 if (params_b or 0.0) <= 1.0 else int(_VLLM_COLOCATE_FLOOR_GB)
        need = max(need, floor_gb)
        # vLLM KV-cache init preflight: the card must leave a viable cache-block pool
        # under the colocate utilization cap, or the engine dies at init ("No available
        # memory for the cache blocks") on a card the training-peak estimate accepted.
        need = max(
            need,
            _rollout_kv_floor_gb(
                params_b or 4.0,
                seq_len,
                group_size,
                active_params_b=active_b,
                model_info=sizing_info,
                rollout_cache_min_gb=rollout_cache_min_gb,
                model_id=model_id,
                lora_rank=lora_rank,
            ),
        )
    return need


def model_required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    headroom: float = 1.1,
    model_revision: str = "",
) -> int:
    """Cheapest-sufficient VRAM (GB) for a specific run (allocator and provisional_gpu sizing)."""

    # Knob extraction must never crash: sizing runs before train validators; malformed values fall back.
    max_tokens = _positive_int_or_default(_sizing_value(train, "max_completion_tokens"), None)
    _algo = (algorithm or "").lower()
    if _algo in ("grpo", "rl"):
        _default_len = grpo_rollout_seq_len(0, max_tokens, thinking)
    elif _algo == "custom":
        _default_len = 2048
    else:
        from flash.engine.plan.recipe import RECIPE

        _default_len = int(RECIPE.sft.max_seq_len_thinking if thinking else RECIPE.sft.max_seq_len)
    seq_len = _positive_int_or_default(_sizing_value(train, "max_context_tokens"), _default_len)
    lora_rank = _positive_int_or_default(
        _sizing_value(train, "lora_rank"), 16 if _algo == "custom" else 32
    )
    batch_size_default = _sft_per_device_bs()
    group_size_default = 8
    group_size = _positive_int_or_default(_sizing_value(train, "group_size"), group_size_default)
    batch_size = _positive_int_or_default(_optimizer_batch_value(train, _algo), batch_size_default)
    if _algo == "custom":
        batch_size = 1  # TorchBackend processes rows sequentially, accumulating gradients.

    from flash.core.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    model_vocab = vocab_size_for(model_id)
    is_grpo = _algo in ("grpo", "rl")
    # the fused sft loss removes ignored positions before the lm head and projects valid tokens
    # without materializing dense logits. size validated qwen sft jobs without a [batch, seq, vocab]
    # term; models outside that validated set keep the conservative plain-nll estimate and cap.
    sft_fused_ce = None if is_grpo else (_algo == "sft" and sft_chunked_nll_enabled(model_id))
    if info is not None:
        return _catalog_model_required_vram_gb(
            model_id,
            algorithm,
            info=info,
            model_vocab=model_vocab,
            model_revision=model_revision,
            seq_len=seq_len,
            max_tokens=max_tokens,
            lora_rank=lora_rank,
            batch_size=batch_size,
            group_size=group_size,
            thinking=thinking,
            headroom=headroom,
            is_grpo=is_grpo,
            sft_fused_ce=sft_fused_ce,
        )
    # Only curated models are trainable, so `info` is never None here in practice. Kept as a
    # conservative default rather than a raise: this is a sizing helper on the allocation path, and
    # a caller probing a stale id should get the smallest managed card, not an exception.
    return 24


def resolve_params_b(model_id: str, revision: str = "") -> float | None:
    """Model size in billions, resolved the ONE way the worker and the cost estimator agree on.

    the curated catalog ``params_b``, or the real HF safetensors count for a PINNED revision (whose
    true size the catalog states only for the default revision). returns None for an uncataloged id,
    which only a stale caller can produce since submit rejects those, so callers degrade to the
    size-unknown path rather than raising on the allocation path. run_sft and run_rl both call this,
    so they can never drift.
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    if info is None:
        return None
    if revision:
        params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
        return params_b
    return info.params_b or None


# The KV-cache sizing helpers and the tuning constants they read live in
# `flash.engine.plan.kv_sizing`. Re-exported here because the fit estimators above read the
# constants too and `from flash.engine.plan.vram import colocate_kv_util` must keep working.
# The import sits at the bottom because those estimators are defined before this point.
from flash.engine.plan.kv_sizing import (  # noqa: E402,F401
    _KV_BLOCK_TOKENS,
    _KV_CAP,
    _KV_COEF,
    _KV_FRAGMENTATION_MARGIN,
    _KV_PROFILE_OVERHEAD_GB,
    _architecture_kv_raw_gb,
    _colocate_util_cap,
    _resident_kv_gb,
    colocate_kv_util,
    grpo_kv_floor_gb,
)

# The pinned-commit config probe lives in `flash.engine.plan.model_config_probe`, for the same
# file-size reason. Re-exported because `from flash.engine.plan.vram import
# _validated_revision_geometry` must keep working. NOTE that `_CONFIG_PROBE_MEMO` is
# deliberately NOT re-exported: an alias would be a SECOND name for one dict, and a test that
# swaps `vram._CONFIG_PROBE_MEMO` would leave the module actually reading the original --
# a silently ineffective patch. Reach for it at its defining module.
from flash.engine.plan.model_config_probe import (  # noqa: E402,F401
    _certify_model_config,
    _config_mismatches,
    _memoized_config_probe,
    _validated_revision_geometry,
    certified_attention_heads,
)
