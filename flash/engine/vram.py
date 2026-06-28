"""Coarse VRAM-fit estimation for one-consumer-GPU LoRA jobs (±20% heuristics)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _gpu_vram_table() -> dict[str, int]:
    try:
        from flash.providers.base import GPU_INFO

        return {name: info.vram_gb for name, info in GPU_INFO.items()}
    except Exception:
        return {"RTX 4090": 24, "RTX 5090": 32}


GPU_VRAM_GB = _gpu_vram_table()

_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
}

_BASE_OVERHEAD_GB = 4.0
_ACT_COEF = 0.12
_SFT_PER_DEVICE_BS_DEFAULT = 4


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


def sft_realized_batch(
    batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True
) -> int:
    """Realized SFT global batch (per_device x grad_accum) for a requested batch_size."""
    per_device, grad_accum = sft_grad_accum(batch_size, seq_len=seq_len, vocab=vocab, fused=fused)
    return per_device * grad_accum


_KV_COEF = 2.0
_KV_CAP = 8.0


def grpo_rollout_seq_len(
    max_length: int = 0,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> int:
    """vLLM engine context a GRPO run uses, mirroring run_rl() — shared by allocator, sleep gate, and KV budget."""
    from flash.engine.recipe import RECIPE

    rl = RECIPE.rl
    completion = int(
        max_tokens or (rl.max_completion_len_thinking if thinking else rl.max_completion_len)
    )
    return int(max_length or max(1024, rl.max_prompt_len + completion))


def _resident_kv_gb(params_b: float | None, vllm_max_len: int, num_generations: int = 8) -> float:
    """Resident KV (GB) for the engine context and generation group; shared by fit-gate and colocate budget."""
    width = math.sqrt(max(float(params_b or 1.0), 0.1))
    return _KV_COEF * (max(1, vllm_max_len) / 1024.0) * width * (max(1, num_generations) / 8.0)


def colocate_kv_util(
    params_b: float | None,
    vllm_max_len: int,
    total_vram_gb: float,
    sleep_mode: bool,
    num_generations: int = 8,
    active_params_b: float | None = None,
) -> float:
    """vllm_gpu_memory_utilization for the colocated GRPO rollout engine.

    ``gpu_memory_utilization`` is vLLM's WHOLE model-executor budget — its (2nd) bf16 weight copy PLUS
    the KV cache — so we budget BOTH (budgeting KV alone would starve the weights and, for big models,
    under-size the engine). The KV a GRPO rollout needs scales with the engine context AND the
    concurrent generation group (``num_generations`` simultaneous sequences), so we size the pool as
    ``_KV_COEF x seq x sqrt(params) x group/8`` with a 1.5x margin and an 8 GB floor — NOT capped, so
    long-context / large-group runs keep a big pool (the 0.45 utilization cap bounds it like the old
    blanket did). The old blanket sleep-path 0.45 reserved ~36 GB on an 80 GB A100 — MEASURED as the
    dominant resident allocation that set the GRPO step peak (~46 GB). BOTH paths budget the weight
    copy + KV; the non-sleep path uses the leaner resident-KV target (_KV_CAP). MEASURED at
    4B/group8/2k ctx: 0.25 util -> peak 46 -> 26 GB, reward byte-identical, train_wall neutral; a
    tighter 12 GB budget preempts, confirming this as the floor."""
    weights_gb = max(0.5, float(params_b or 1.0)) * 2.0  # vLLM's bf16 weight copy lives in the budget
    # MoE: the KV pool scales with the per-token COMPUTE width (the ~3B active backbone), NOT the 35B
    # total — exactly the split estimate_vram_gb/grpo_fits_resident use (active for KV, full for the
    # weight copy). Keying the KV off total here would budget a LARGER pool than the resident-fit gate
    # counted, so the gate could disable sleep while the engine reserves more KV than it sized — the
    # near-margin 35B-A3B GRPO mismatch. Dense models leave active_params_b unset -> full params_b.
    kv_params_b = float(active_params_b) if active_params_b else params_b
    if not sleep_mode:
        # Resident KV ON TOP of the weight copy: gpu_memory_utilization is the WHOLE executor budget,
        # so budgeting KV alone (the old _KV_CAP/total) starved the weights and vLLM raised "No
        # available memory for the cache blocks" on >=3B models whose weights exceed an 8 GB budget.
        # The KV must ALSO cover the rollout context -- a flat _KV_CAP starves the cache blocks on a
        # long-context run (vLLM's blocks must span vllm_max_model_length), so scale it with the
        # context + group (floored at _KV_CAP for the validated short-context lean point, bounded by
        # the 0.45 util cap below). Matches the resident-fit estimate (estimate_vram_gb sleep_offload
        # =False) so grpo_sleep_mode's gate and this budget size the SAME KV.
        kv_gb = max(_KV_CAP, _resident_kv_gb(kv_params_b, vllm_max_len, num_generations))
        return max(0.10, min(0.45, (weights_gb + kv_gb) / max(1.0, total_vram_gb)))
    # Sleep mode keeps a larger pool (1.5x margin): the engine is offloaded during the backward, so a
    # bigger rollout-phase KV does not compete with the training peak.
    kv_pool_gb = max(_KV_CAP, 1.5 * _resident_kv_gb(kv_params_b, vllm_max_len, num_generations))
    return min(0.45, (weights_gb + kv_pool_gb) / max(1.0, total_vram_gb))
_TRAIN_COEF = 0.27
# Small-model colocated GRPO floor: 0.8B OOMs 20 GB; 2B OOMs 24 GB -> both need 32 GB tier.
_VLLM_COLOCATE_FLOOR_GB = 28.0
_VOCAB_DEFAULT = 248_320
_LOGITS_BUDGET_GB = 6.0
# 16 B/elem: fp32 logits+grad + bf16 logits+grad + CE temp. 8 B/elem under-counts (live OOM confirmed).
_SFT_LOGITS_BYTES_PER_ELEM = 16.0
# Single source of truth for Liger gate — engine.worker.perf imports these.
_LIGER_MIN_PARAMS_B = 3.0
_LIGER_LONG_CTX_TOKENS = 2048


def sft_logits_fused(params_b: float | None, seq_len: int) -> bool:
    """True when the worker fuses SFT cross-entropy (>=3B model OR >=2048-token context)."""
    if seq_len >= _LIGER_LONG_CTX_TOKENS:
        return True
    return (params_b or 0.0) >= _LIGER_MIN_PARAMS_B


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


@dataclass(frozen=True)
class VramEstimate:
    params_b: float | None
    algorithm: str
    quant: str
    est_gb: float | None
    gpu: str
    gpu_gb: int
    verdict: str  # "fits" | "tight" | "too_big" | "unknown"

    def describe(self) -> str:
        if self.est_gb is None:
            return f"{self.gpu}: VRAM need unknown (could not read model size)"
        return (
            f"{self.gpu} ({self.gpu_gb} GB): estimated ~{self.est_gb:.0f} GB needed "
            f"({self.params_b:.1f}B params, {self.quant}, {self.algorithm}) -> {self.verdict}"
        )


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
    use_vllm: bool = True,
    vocab: int = _VOCAB_DEFAULT,
    sleep_offload: bool = True,
    active_params_b: float | None = None,
) -> float:
    """Estimated peak VRAM (GB) for a LoRA job on one GPU.

    MoE: active_params_b drives activations/KV/LoRA; weights term uses full params_b.
    """
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    eff_b = float(active_params_b) if active_params_b else params_b
    algo = "grpo" if (algorithm or "").lower() in ("grpo", "rl") else "sft"
    width = math.sqrt(max(eff_b, 0.1))
    lora_opt = (lora_rank / 16.0) * (0.3 + 0.04 * eff_b)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    if algo == "grpo":
        # Sleep mode: peak = max(rollout, train). Resident: both live at once, peak = sum.
        rollout = 0.0
        if use_vllm:
            if sleep_offload:
                rollout = weights + min(_KV_COEF * (seq_len / 1024.0) * width, _KV_CAP)
            else:
                rollout = weights + _resident_kv_gb(eff_b, seq_len, group_size)
        group_factor = max(1.0, (max(1, group_size) / 4.0) ** 0.5)
        think_factor = 1.3 if thinking else 1.0
        activations = _TRAIN_COEF * (seq_len / 1024.0) * width * group_factor * think_factor
        completion = max_tokens if max_tokens else min(seq_len, 1024)
        logits = min(completion * vocab * 4 / 1e9, _LOGITS_BUDGET_GB)
        train = activations + logits
        return base + (max(rollout, train) if sleep_offload else rollout + train)
    fused = sft_logits_fused(params_b, seq_len)
    pd = sft_per_device(batch_size, seq_len=seq_len, vocab=vocab, fused=fused)
    activations = _ACT_COEF * pd * (seq_len / 1024.0) * width
    # Don't clamp to budget: pd=1 is irreducible and the logits can exceed the budget at near-2048 ctx.
    logits = 0.0 if fused else pd * seq_len * vocab * _SFT_LOGITS_BYTES_PER_ELEM / 1e9
    return base + activations + logits


def grpo_fits_resident(
    model_id: str,
    *,
    seq_len: int = 1024,
    max_tokens: int | None = None,
    lora_rank: int = 32,
    group_size: int = 8,
    thinking: bool = False,
    card_vram_gb: float = 0.0,
    margin: float = 1.15,
) -> bool:
    """True when GRPO fits resident (no sleep-mode offload); False on unknown model/card (safe default)."""
    if not card_vram_gb or card_vram_gb <= 0:
        return False
    from flash.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    params_b = float(getattr(info, "params_b", 0.0) or 0.0) if info else 0.0
    if params_b <= 0:
        return False
    quant = (getattr(info, "quant", "bf16") or "bf16") if info else "bf16"
    # MoE: size the resident peak's COMPUTE terms (KV pool, activations, rank-linear LoRA) on the ~3B
    # ACTIVE backbone, exactly as model_required_vram_gb does — keying them on the 35B TOTAL inflates
    # the resident estimate above the card and wrongly forces vLLM sleep mode on a B200 MoE GRPO run,
    # where the sleep/wake cycle stalls the colocated rollout (the very failure this gate exists to
    # avoid). The ``weights`` term still sizes the full params_b. Dense models leave active_params_b
    # unset -> estimate_vram_gb falls back to params_b for every term (unchanged).
    active_b = float(getattr(info, "active_params_b", 0.0) or 0.0) if info else 0.0
    resident = estimate_vram_gb(
        params_b,
        "grpo",
        quant,
        seq_len=max(1, int(seq_len or 1024)),
        max_tokens=max_tokens,
        lora_rank=lora_rank,
        group_size=group_size,
        thinking=thinking,
        use_vllm=True,
        vocab=vocab_size_for(model_id),
        sleep_offload=False,
        active_params_b=active_b,
    )
    return resident * margin <= card_vram_gb


def model_required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    headroom: float = 1.1,
) -> int:
    """Cheapest-sufficient VRAM (GB) for a specific run (allocator and provisional_gpu sizing)."""

    # Knob extraction must never crash: sizing runs before train validators; malformed values fall back.
    def _g(obj, key):
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def _pos_int(v, default):
        try:
            if isinstance(v, bool):
                return default
            f = float(v)
            return int(f) if math.isfinite(f) and f >= 1 else default
        except (TypeError, ValueError):
            return default

    max_tokens = _pos_int(_g(train, "max_tokens"), None)
    if (algorithm or "").lower() in ("grpo", "rl"):
        _grpo_default_len = grpo_rollout_seq_len(0, max_tokens, thinking)
    else:
        _grpo_default_len = 1024
    seq_len = _pos_int(_g(train, "max_length"), _grpo_default_len)
    lora_rank = _pos_int(_g(train, "lora_rank"), 32)
    group_size = _pos_int(_g(train, "group_size"), 8)
    batch_size = _pos_int(_g(train, "batch_size"), _sft_per_device_bs())

    def _need(
        params_b: float,
        algorithm: str,
        *,
        quant: str = "bf16",
        use_vllm: bool = True,
        vocab: int = _VOCAB_DEFAULT,
        active_params_b: float | None = None,
    ) -> int:
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
            use_vllm=use_vllm,
            vocab=vocab,
            active_params_b=active_params_b,
        )
        return math.ceil(est * headroom)

    from flash.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    model_vocab = vocab_size_for(model_id)
    is_grpo = (algorithm or "").lower() in ("grpo", "rl")
    if info is not None:
        params_b = info.params_b  # curated, authoritative (required field) — no string parsing
        quant = getattr(info, "quant", "bf16") or "bf16"
        use_vllm = True
        active_b = float(getattr(info, "active_params_b", 0.0) or 0.0)
        need = _need(
            params_b or 4.0,
            algorithm,
            quant=quant,
            use_vllm=use_vllm,
            vocab=model_vocab,
            active_params_b=active_b,
        )
        floor = 0
        if is_grpo and getattr(info, "grpo_min_vram_gb", 0):
            floor = int(info.grpo_min_vram_gb)
        if not is_grpo and getattr(info, "sft_min_vram_gb", 0):
            floor = max(floor, int(info.sft_min_vram_gb))
        # Escalate on active_params_b for MoE: keying on total would over-reject (35B total's
        # threshold is below default rollout length); ~3B active gives ~16k headroom.
        if is_grpo and floor:
            floor += grpo_seq_escalation_gb(active_b or params_b, seq_len)
        need = max(need, floor)
        if is_grpo and use_vllm:
            floor_gb = 24 if (params_b or 0.0) <= 1.0 else int(_VLLM_COLOCATE_FLOOR_GB)
            need = max(need, floor_gb)
        return need
    params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return 24
    need = _need(params_b, "grpo", vocab=model_vocab)
    if is_grpo:
        need += grpo_seq_escalation_gb(params_b, seq_len)
    return need


def fetch_hf_params_b(model_id: str) -> float | None:
    """Total params (billions) from HF safetensors metadata; None on network/metadata failure."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            model_id, expand=["safetensors"]
        )
        total = getattr(getattr(info, "safetensors", None), "total", None)
        if total:
            return float(total) / 1e9
    except Exception:
        pass
    return None


def resolve_params_b(model_id: str) -> float | None:
    """Model size in billions, resolved the ONE way the worker and the cost estimator agree on:
    the curated catalog ``params_b`` (the required numeric field), else the real HF safetensors
    param count for an open-policy (uncataloged) model. Best-effort: returns None only
    when the model is uncataloged AND HF metadata is unavailable, so callers degrade to the
    size-unknown path (e.g. the fused-CE gate stays memory-safe, the colocate cap stays loose).
    The single source of truth for "how big is this model" -- run_sft, run_rl and cost.spec all
    call this so they can never drift."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None and info.params_b > 0:
        return info.params_b  # curated, authoritative (required field) — no string parsing
    return fetch_hf_params_b(model_id)


def check_fit(
    model_id: str,
    algorithm: str,
    gpu: str,
    quant: str = "bf16",
    params_b: float | None = None,
) -> VramEstimate:
    """Estimate whether ``model_id`` plausibly trains on ``gpu``; never raises."""
    gpu_gb = GPU_VRAM_GB.get(gpu, 32)
    if params_b is None:
        params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return VramEstimate(None, algorithm, quant, None, gpu, gpu_gb, "unknown")
    est = estimate_vram_gb(params_b, algorithm, quant)
    if est > gpu_gb * 1.15:
        verdict = "too_big"
    elif est > gpu_gb * 0.85:
        verdict = "tight"
    else:
        verdict = "fits"
    return VramEstimate(params_b, algorithm, quant, est, gpu, gpu_gb, verdict)
