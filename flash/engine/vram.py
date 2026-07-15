"""Coarse VRAM-fit estimation for one-consumer-GPU LoRA jobs (±20% heuristics)."""

from __future__ import annotations

import json
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


def opd_completion_len(max_tokens: int | None, thinking: bool) -> int:
    """The completion-token budget an OPD run uses: an explicit ``max_tokens`` else the OPD recipe
    default (thinking uses the longer ``max_completion_len_thinking``). Single source of truth for the
    four sites that must resolve the SAME integer — run_opd's knob resolution, ``opd_rollout_seq_len``,
    ``estimate_vram_gb``'s opd path, and the spec-parse prompt-budget guard."""
    from flash.engine.recipe import RECIPE

    opd = RECIPE.opd
    return int(max_tokens or (opd.max_completion_len_thinking if thinking else opd.max_completion_len))


def opd_rollout_seq_len(
    max_length: int = 0,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> int:
    """Sequence length an OPD run uses, mirroring run_opd()'s ``seq_cap``: the loss forward runs
    ``model(prompt_ids + student_ids)`` over prompt+completion, so size for both — else a raised
    ``max_tokens`` (unset ``max_length``) is sized as an SFT 1024-token job and OOMs on an
    under-sized GPU."""
    from flash.engine.recipe import RECIPE

    completion = opd_completion_len(max_tokens, thinking)
    return int(max_length or max(1024, RECIPE.opd.max_prompt_len + completion))


def opd_rollout_concurrency(prompts_per_step: int = 1, group_size: int = 1) -> int:
    """Concurrent student generations in one OPD vLLM rollout batch."""

    def _positive_int(value: object, default: int) -> int:
        try:
            if isinstance(value, bool):
                return default
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    return _positive_int(prompts_per_step, 1) * _positive_int(group_size, 1)


def opd_loss_microbatch(params_b: float, prompts_per_step: int = 1, group_size: int = 1) -> int:
    """Loss microbatch size used by OPD's dense-logit GKD backward.

    Keep this in lockstep with worker.opd._opd_loss_microbatch_size without importing the worker from
    the sizing path. Small/medium catalog models run up to four loss samples per forward; 35B-class
    models stay serial for VRAM safety.
    """
    total = opd_rollout_concurrency(prompts_per_step, group_size)
    params = float(params_b or 0.0)
    default = 4 if params and params <= 10.0 else 1
    return max(1, min(total, default))


def _resident_kv_gb(
    params_b: float | None, vllm_max_len: int, num_generations: int = 8, fp8_kv: bool = False
) -> float:
    """Resident KV GB for the rollout context and generation group."""
    width = math.sqrt(max(float(params_b or 1.0), 0.1))
    kv = _KV_COEF * (max(1, vllm_max_len) / 1024.0) * width * (max(1, num_generations) / 8.0)
    return kv * 0.5 if fp8_kv else kv


def _colocate_util_cap(weights_gb: float, total_vram_gb: float) -> float:
    """Utilization ceiling for the colocated vLLM executor budget.

    0.45 everywhere the blanket cap was tuned; lifted to 0.55 only when the executor carries a BIG
    weight copy (>=60 GB) AND the card leaves the trainer's resident copy room alongside the lifted
    budget (0.45*total - weights >= ~10 GB). See colocate_kv_util for the measured rationale."""
    big_weight_copy = weights_gb >= 60
    leaves_room_for_trainer_copy = (0.45 * total_vram_gb - weights_gb) >= 10.0
    return 0.55 if (big_weight_copy and leaves_room_for_trainer_copy) else 0.45


def colocate_kv_util(
    params_b: float | None,
    vllm_max_len: int,
    total_vram_gb: float,
    sleep_mode: bool,
    num_generations: int = 8,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
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
    weights_gb = (
        max(0.5, float(params_b or 1.0)) * 2.0
    )  # vLLM's bf16 weight copy lives in the budget
    # MoE: the KV pool scales with the per-token COMPUTE width (the ~3B active backbone), NOT the 35B
    # total — exactly the split estimate_vram_gb/grpo_fits_resident use (active for KV, full for the
    # weight copy). Keying the KV off total here would budget a LARGER pool than the resident-fit gate
    # counted, so the gate could disable sleep while the engine reserves more KV than it sized — the
    # near-margin 35B-A3B GRPO mismatch. Dense models leave active_params_b unset -> full params_b.
    kv_params_b = float(active_params_b) if active_params_b else params_b
    # Utilization ceiling. The blanket 0.45 was tuned on 80 GB cards, where it leaves healthy
    # headroom. On a BIG card carrying a BIG colocate weight copy (the 35B MoE on a 180 GB B200:
    # 70 GB vLLM weights + the trainer's 70 GB base both resident during rollout) it instead STARVES
    # the KV pool: 0.45 x 180 = 81 GB executor budget = 70 GB weights + only ~11 GB KV, with ~29 GB
    # of the card sitting unused. KV bounds rollout concurrency, so the cheap-active A3B can't fill
    # the card. Lift the cap to 0.55 ONLY when (a) the weight copy is big (>=60 GB) AND (b) the card
    # actually has room for it -- the trainer's resident weight copy must still fit alongside the 0.55
    # executor budget with margin: 0.45*total - weights >= ~10 GB. That holds on the 180 GB B200
    # (0.45*180 - 70 = 11) -> vLLM budget 99 GB = 70 weights + ~29 KV, resident 70 (trainer) + 99 =
    # 169 GB (~11 GB margin), but NOT the 141 GB H200 (0.45*141 - 70 = -6.5): there 0.55 x 141 = 78 GB
    # executor + the trainer's 70 GB copy = 148 GB > 141 GB -> overflow. So the H200 -- and ANY card
    # where the weight copy wouldn't fit alongside the 0.55 budget -- correctly STAYS at 0.45. Keying
    # off the real headroom (not a >=140 GB card threshold, which ALSO catches the H200) is what makes
    # the lift safe. The GRPO step's _GpuPeakSampler verifies the headroom holds. Every other
    # card/model is byte-for-byte unchanged at 0.45.
    _util_cap = _colocate_util_cap(weights_gb, total_vram_gb)
    if not sleep_mode:
        # Resident KV ON TOP of the weight copy: gpu_memory_utilization is the WHOLE executor budget,
        # so budgeting KV alone (the old _KV_CAP/total) starved the weights and vLLM raised "No
        # available memory for the cache blocks" on >=3B models whose weights exceed an 8 GB budget.
        # The KV must ALSO cover the rollout context -- a flat _KV_CAP starves the cache blocks on a
        # long-context run (vLLM's blocks must span vllm_max_model_length), so scale it with the
        # context + group (floored at _KV_CAP for the validated short-context lean point, bounded by
        # the 0.45 util cap below). Matches the resident-fit estimate (estimate_vram_gb sleep_offload
        # =False) so grpo_sleep_mode's gate and this budget size the SAME KV.
        kv_gb = max(
            _KV_CAP, _resident_kv_gb(kv_params_b, vllm_max_len, num_generations, fp8_kv=fp8_kv)
        )
        return max(0.10, min(_util_cap, (weights_gb + kv_gb) / max(1.0, total_vram_gb)))
    # Sleep mode keeps a larger pool (1.5x margin): the engine is offloaded during the backward, so a
    # bigger rollout-phase KV does not compete with the training peak.
    kv_pool_gb = max(
        _KV_CAP,
        1.5 * _resident_kv_gb(kv_params_b, vllm_max_len, num_generations, fp8_kv=fp8_kv),
    )
    return min(_util_cap, (weights_gb + kv_pool_gb) / max(1.0, total_vram_gb))


_TRAIN_COEF = 0.27
# Small-model colocated GRPO floor: 0.8B OOMs 20 GB; 2B OOMs 24 GB -> both need 32 GB tier.
_VLLM_COLOCATE_FLOOR_GB = 28.0
# OPD builds its resident vLLM engine after the HF/PEFT student is already loaded. A real
# Qwen3.5-2B OPD run with vLLM failed vLLM startup on a 32 GB RTX 5090 despite the raw equation
# estimating ~28 GB with headroom, because vLLM requires the requested executor budget to be free at
# init time. Keep 2B+ OPD off <=40 GB classes so the allocator picks an 80 GB-class card instead of a
# consumer GPU that passes the aggregate estimate but fails the vLLM free-memory preflight.
_OPD_VLLM_COLOCATE_FLOOR_GB = 41.0
_VOCAB_DEFAULT = 248_320
_LOGITS_BUDGET_GB = 6.0
# 16 B/elem: fp32 logits+grad + bf16 logits+grad + CE temp. 8 B/elem under-counts (live OOM confirmed).
_SFT_LOGITS_BYTES_PER_ELEM = 16.0
# Single source of truth for the SFT fused-CE gate. Keep the historical constant names because
# engine.worker.perf and tests import them.
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


def grpo_kv_floor_gb(
    params_b: float,
    vllm_max_len: int,
    group_size: int = 8,
    active_params_b: float | None = None,
) -> int:
    """Smallest card (GB) whose colocated vLLM executor budget still leaves a viable KV pool.

    ``colocate_kv_util`` caps ``gpu_memory_utilization`` at ~0.45 of the card, and that budget
    must carry vLLM's bf16 weight copy BEFORE any KV cache blocks. On a card sized only for the
    training peak, a long-context / large-group rollout can leave the pool so small that vLLM
    init fails with "No available memory for the cache blocks" — an OOM the preflight never
    caught (e.g. group_size=16 multi-turn rollouts on a 31 GB card). Require the capped budget
    to hold the weight copy plus at least HALF the concurrent group's working KV (full KV is a
    throughput target, not an init requirement; half keeps validated lean configs admitted
    while pushing the observed init-OOM shapes onto a bigger card or a parse-time reject).

    The utilization cap mirrors ``colocate_kv_util`` (0.45, lifted to 0.55 for big weight copies
    with headroom) so the floor never overestimates the card a large model actually needs."""
    kv_params_b = float(active_params_b) if active_params_b else float(params_b)
    weights_gb = max(0.5, float(params_b)) * 2.0
    need = weights_gb + 0.5 * _resident_kv_gb(kv_params_b, vllm_max_len, group_size)
    lower = math.ceil(need / 0.55)
    upper = math.ceil(need / 0.45)
    for total_vram_gb in range(lower, upper + 1):
        if _colocate_util_cap(weights_gb, total_vram_gb) * total_vram_gb >= need:
            return total_vram_gb
    return upper


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
    fp8_kv: bool = False,
    sft_fused_ce: bool | None = None,
) -> float:
    """Estimated peak VRAM (GB) for a LoRA job on one GPU.

    MoE: active_params_b drives activations/KV/LoRA; weights term uses full params_b.
    """
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    eff_b = float(active_params_b) if active_params_b else params_b
    is_opd = (algorithm or "").lower() == "opd"
    algo = "grpo" if (algorithm or "").lower() in ("grpo", "rl") else "sft"
    width = math.sqrt(max(eff_b, 0.1))
    lora_opt = (lora_rank / 16.0) * (0.3 + 0.04 * eff_b)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    if algo == "grpo":
        # Sleep mode: peak = max(rollout, train). Resident: both live at once, peak = sum.
        rollout = 0.0
        if use_vllm:
            _kv_fp8_factor = 0.5 if fp8_kv else 1.0  # fp8 KV halves bytes/token (cc>=8.9)
            if sleep_offload:
                rollout = weights + min(
                    _KV_COEF * (seq_len / 1024.0) * width * _kv_fp8_factor, _KV_CAP
                )
            else:
                rollout = weights + _resident_kv_gb(eff_b, seq_len, group_size, fp8_kv=fp8_kv)
        group_factor = max(1.0, (max(1, group_size) / 4.0) ** 0.5)
        think_factor = 1.3 if thinking else 1.0
        activations = _TRAIN_COEF * (seq_len / 1024.0) * width * group_factor * think_factor
        completion = max_tokens if max_tokens else min(seq_len, 1024)
        logits = min(completion * vocab * 4 / 1e9, _LOGITS_BUDGET_GB)
        train = activations + logits
        return base + (max(rollout, train) if sleep_offload else rollout + train)
    if is_opd:
        # OPD always samples the student through a resident colocated vLLM engine, while the HF/PEFT
        # trainer model stays resident for the GKD loss forward/backward. Budget the second bf16
        # weight copy plus a viable KV pool through the training step. ``use_vllm`` is intentionally
        # ignored for OPD; it only exists for legacy GRPO sizing callers.
        rollout_concurrency = opd_rollout_concurrency(batch_size, group_size)
        rollout = weights + max(
            _KV_CAP, _resident_kv_gb(eff_b, seq_len, rollout_concurrency, fp8_kv=fp8_kv)
        )
        # opd's gkd loss forward materializes DENSE logits — there is NO fused cross-entropy: the
        # student forward yields full-sequence bf16 logits [seq, vocab], then the completion rows are
        # gathered in fp32 [completion, vocab] for the logsumexp. The SFT fused path budgets ZERO
        # vocab logits for >=3B models, so a long-completion opd job (e.g. max_tokens=8192) would be
        # sized for an under-capacity card and OOM.
        #
        # Mirror opd_rollout_seq_len / run_opd's completion resolution: explicit max_tokens, else the
        # OPD recipe default (thinking uses the longer max_completion_len_thinking). A GRPO-style
        # min(seq_len, 1024) fallback would UNDER-budget a thinking opd job (1536-token completions).
        completion = opd_completion_len(max_tokens, thinking)
        # run_opd backprops a small loss microbatch. Budget that actual dense-logit microbatch, not
        # the full rollout/teacher batch and not a single sequence.
        loss_mb = opd_loss_microbatch(params_b, batch_size, group_size)
        activations = loss_mb * _ACT_COEF * (seq_len / 1024.0) * width
        # Dense-logit peak spans BOTH the forward and the loss BACKWARD (OPD has no fused CE):
        #   - forward:  bf16 full-sequence logits [seq, vocab]        (seq * 2)
        #               fp32 completion rows [completion, vocab] for logsumexp, saved for backward
        #                                                             (completion * 4)
        #   - backward: fp32 gradient of those completion rows [completion, vocab]
        #                                                             (completion * 4)
        #               bf16 gradient scattered back into the full logits [seq, vocab] to reach lm_head
        #                                                             (seq * 2)
        # All four coexist at the backward peak. The old formula counted only the two FORWARD buffers,
        # so a long-completion / large-vocab (248k) opd job under-budgeted the loss backward by
        # ~(completion*4 + seq*2)*vocab bytes and could route to a GPU that OOMs in OPD loss backward
        # (codex[bot]). Mirror the SFT dense-logit sizing by budgeting the backward buffers too.
        logits_fwd = loss_mb * (seq_len * 2 + completion * 4) * vocab
        logits_bwd = loss_mb * (seq_len * 2 + completion * 4) * vocab
        logits = (logits_fwd + logits_bwd) / 1e9
        return base + rollout + activations + logits
    # Actual TRL SFT keeps fused CE disabled (see worker/sft.py), so dense logits materialize even
    # for long-context / >=3B models. Callers can pass sft_fused_ce=True only for theoretical
    # comparisons; the default must mirror the worker.
    fused = False if sft_fused_ce is None else bool(sft_fused_ce)
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
    fp8_kv: bool = False,
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
        fp8_kv=fp8_kv,
    )
    return resident * margin <= card_vram_gb


# Activations retained for the backward when gradient checkpointing is OFF. With GC, each layer's
# intra-block activations are RECOMPUTED in the backward (one extra forward ~= +33% compute); with GC
# OFF they all stay resident -- so this term is ~num_layers x the per-layer (residual stream + qkv/out
# + MoE SwiGLU intermediate) activations per token. With FlashAttention / cuDNN-SDPA there is NO s^2
# attention-score tensor, and the GatedDeltaNet linear layers store only a small recurrent/conv state,
# so it is LINEAR in seq. K folds the per-layer multiplier (relative to hidden) into one constant. It
# is set ABOVE the Megatron FlashAttention factor (~34 bytes/(s.b.h) == K 17) so we OVER-reserve and
# keep GC ON rather than risk an OOM; the MoE's small active FFN (8 x 512 vs a dense 4 x 2048) makes
# the true value lower still. bf16 activations (2 bytes/elem). Calibrate K from the live H200 peak.
_GC_OFF_ACT_K = 18.0


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
) -> float:
    """Estimated peak VRAM (GB) for a FUSED-CE LoRA SFT step with gradient checkpointing OFF: the
    resident weights + optimizer/base + the no-recompute activations held across ALL ``num_layers``.
    Fused CE (chalk FLCE) is assumed, so there is no ``[B, T, vocab]`` logits term (the thing that
    made GC-off impossible at a 248k vocab). Unknown architecture dims -> ``inf`` (caller keeps GC on).

    MoE: the activation backbone scales with the model's real ``hidden`` x ``num_layers`` (geometry),
    NOT params_b -- the ~3B-active expert FFN is already folded into ``_GC_OFF_ACT_K``. ``weights``
    still reserves the FULL ``params_b`` (every expert is resident)."""
    if not (hidden and num_layers and seq_len):
        return float("inf")
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    eff_b = float(active_params_b) if active_params_b else float(params_b)
    weights = float(params_b) * bpp
    lora_opt = (lora_rank / 16.0) * (0.3 + 0.04 * eff_b)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    act = _GC_OFF_ACT_K * int(num_layers) * int(batch) * int(seq_len) * int(hidden) * 2.0 / 1e9
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
    margin_gb: float = 18.0,
) -> bool:
    """True when a FUSED-CE LoRA SFT step fits a ``card_vram_gb`` card WITHOUT gradient checkpointing,
    so GC -- a ~+33% recompute tax on every step -- can be turned off for the speed win.

    Conservative by construction: an unknown card / unknown architecture dims, or a peak that doesn't
    clear the ``margin_gb`` headroom, returns False (keep GC ON). The estimate over-reserves (high
    ``_GC_OFF_ACT_K`` + a fixed margin) precisely because the allocator sized the card for the
    GC-*on* peak -- this is the independent check that the larger GC-off peak still fits the REAL
    card (mirrors ``grpo_fits_resident``'s resident-fit guard)."""
    if not card_vram_gb or card_vram_gb <= 0 or not (hidden and num_layers and seq_len):
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
    )
    return peak + float(margin_gb) <= float(card_vram_gb)


def _config_geometry(config: dict) -> tuple[int, int, int]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return (
        int(text.get("vocab_size") or config.get("vocab_size") or 0),
        int(text.get("hidden_size") or config.get("hidden_size") or 0),
        int(text.get("num_hidden_layers") or config.get("num_hidden_layers") or 0),
    )


def fetch_hf_model_geometry(
    model_id: str, revision: str = "", *, strict: bool = False
) -> tuple[float | None, int, int, int]:
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
        vocab, hidden, layers = _config_geometry(config)
        return params_b, vocab, hidden, layers
    except Exception as exc:
        if strict:
            raise ValueError(
                f"could not resolve revision-specific sizing metadata for model {model_id!r}"
            ) from exc
        return None, 0, 0, 0


def _validated_revision_geometry(model_id: str, revision: str, info):
    params_b, vocab, hidden, layers = fetch_hf_model_geometry(model_id, revision, strict=True)
    # Revision-aware sizing is authoritative and must fail closed. When the pinned commit exposes no
    # parameter-count metadata (no safetensors.total), we cannot derive its size; silently reusing the
    # catalog default-revision count would size the exact-GPU preflight on weights the worker never loads,
    # the precise mis-provisioning this pin exists to prevent.
    if params_b is None:
        raise ValueError(
            f"model_revision for {model_id!r} exposes no parameter-count metadata "
            f"(no safetensors.total); cannot size the pinned revision"
        )
    mismatches: list[str] = []
    if info.params_b > 0:
        delta = abs(params_b - info.params_b) / info.params_b
        if delta > 0.05:
            mismatches.append("parameter count")
    if vocab and info.vocab_size and vocab != info.vocab_size:
        mismatches.append("vocabulary size")
    if hidden and info.hidden_size and hidden != info.hidden_size:
        mismatches.append("hidden size")
    if layers and info.num_layers and layers != info.num_layers:
        mismatches.append("layer count")
    if mismatches:
        raise ValueError(
            f"model_revision for {model_id!r} has geometry incompatible with the catalog: "
            f"{', '.join(mismatches)}"
        )
    return params_b, vocab or info.vocab_size


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
    def _get(obj, key):
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

    max_tokens = _pos_int(_get(train, "max_completion_tokens"), None)
    _algo = (algorithm or "").lower()
    if _algo in ("grpo", "rl"):
        _default_len = grpo_rollout_seq_len(0, max_tokens, thinking)
    elif _algo == "opd":
        # OPD generates on-policy like GRPO, so size for prompt+completion, not the SFT 1024 default.
        _default_len = opd_rollout_seq_len(0, max_tokens, thinking)
    else:
        _default_len = 1024
    seq_len = _pos_int(_get(train, "max_context_tokens"), _default_len)
    lora_rank = _pos_int(_get(train, "lora_rank"), 32)
    if _algo == "opd":
        from flash.engine.recipe import RECIPE

        batch_size_default = int(RECIPE.opd.prompts_per_step)
        group_size_default = int(RECIPE.opd.group_size)
    else:
        batch_size_default = _sft_per_device_bs()
        group_size_default = 8
    group_size = _pos_int(_get(train, "group_size"), group_size_default)
    batch_size = _pos_int(_get(train, "batch_size"), batch_size_default)

    def _need(
        params_b: float,
        algorithm: str,
        *,
        quant: str = "bf16",
        use_vllm: bool = True,
        vocab: int = _VOCAB_DEFAULT,
        active_params_b: float | None = None,
        fp8_kv: bool = False,
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
            fp8_kv=fp8_kv,
            sft_fused_ce=sft_fused_ce,
        )
        return math.ceil(est * headroom)

    def _opd_fp8_adjust(
        need: int,
        params_b: float,
        *,
        quant: str = "bf16",
        use_vllm: bool = True,
        vocab: int = _VOCAB_DEFAULT,
        active_params_b: float | None = None,
    ) -> int:
        """Re-size an OPD requirement with an fp8 KV cache once the run is provably modern-card-only.

        The colocated OPD vLLM rollout engine reserves an fp8 KV cache on cc >= 8.9 hardware
        (engine/worker/opd_vllm.py), but the estimate defaults to a bf16 KV pool. Any OPD run needing
        more VRAM than the biggest non-fp8 validated card (the 80 GB A100) can ONLY land on a modern
        (cc >= 8.9) card, so that bf16 pool is a phantom: it doubles the real KV and wrongly rejects
        full-context / grouped OPD configs on the 35B that actually fit a B200. Halve it — but only
        while the fp8-sized requirement still clears the non-fp8 ceiling, so the discount can never
        pull the run back onto a card that would NOT use fp8 (which would then OOM)."""
        from flash.providers.base import max_non_fp8_kv_vram_gb

        ceiling = max_non_fp8_kv_vram_gb()
        if need <= ceiling:
            return need
        fp8_need = _need(
            params_b,
            "opd",
            quant=quant,
            use_vllm=use_vllm,
            vocab=vocab,
            active_params_b=active_params_b,
            fp8_kv=True,
        )
        return fp8_need if fp8_need > ceiling else need

    from flash.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    model_vocab = vocab_size_for(model_id)
    is_grpo = _algo in ("grpo", "rl")
    is_opd = _algo == "opd"
    is_vllm_rollout = is_grpo or is_opd
    vllm_concurrency = (
        opd_rollout_concurrency(batch_size, group_size) if is_opd else group_size
    )
    # Worker parity: TRL SFT deliberately calls install_chalk_kernels(..., fused_ce=False), because
    # SFTTrainer.compute_loss reads outputs.logits and crashes when fused CE returns logits=None. So
    # every SFT run materializes dense logits, even for long context / >=3B models where the old
    # allocator assumed a fused CE path. Size SFT with fused CE OFF up front or long-context Qwen SFT
    # routes to consumer cards and OOMs on the first backward.
    sft_fused_ce = None if is_grpo else False
    if info is not None:
        params_b = info.params_b
        if model_revision:
            params_b, model_vocab = _validated_revision_geometry(model_id, model_revision, info)
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
        if is_opd:
            need = _opd_fp8_adjust(
                need,
                params_b or 4.0,
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
            if getattr(info, "sleep_unsupported", False):
                # Sleep is non-functional for this model (it HANGS) -> it MUST fit RESIDENT. Size the
                # requirement on the RESIDENT peak (engine live through the backward, fp8 KV on the big
                # floor card which is sm100) instead of the sleep estimate + grpo_seq_escalation_gb, so
                # a config too long to fit resident is pushed PAST every GPU and REJECTED at parse time
                # -- rather than admitted-then-HUNG in the broken sleep path between the resident wall
                # and the old ~16k sleep ceiling.
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
                        use_vllm=True,
                        vocab=model_vocab,
                        sleep_offload=False,
                        active_params_b=active_b,
                        fp8_kv=True,
                    )
                    # match grpo_fits_resident's 1.15 margin (NOT the looser 1.1 headroom) so the
                    # parse-time reject lands at the SAME resident wall the worker gate enforces.
                    * 1.15
                )
                floor = max(floor, resident_need)
            else:
                floor += grpo_seq_escalation_gb(active_b or params_b, seq_len)
        need = max(need, floor)
        if is_vllm_rollout and use_vllm:
            floor_gb = 24 if (params_b or 0.0) <= 1.0 else int(_VLLM_COLOCATE_FLOOR_GB)
            if is_opd and (params_b or 0.0) >= 2.0:
                floor_gb = max(floor_gb, int(_OPD_VLLM_COLOCATE_FLOOR_GB))
            need = max(need, floor_gb)
            # vLLM KV-cache init preflight: the card must leave a viable cache-block pool
            # under the colocate utilization cap, or the engine dies at init ("No available
            # memory for the cache blocks") on a card the training-peak estimate accepted.
            need = max(
                need,
                grpo_kv_floor_gb(
                    params_b or 4.0,
                    seq_len,
                    vllm_concurrency,
                    active_params_b=active_b,
                ),
            )
        return need
    params_b = (
        fetch_hf_params_b(model_id, revision=model_revision, strict=True)
        if model_revision
        else fetch_hf_params_b(model_id)
    )
    if params_b is None:
        return 24
    # Size the uncataloged (model_policy="allow") fallback with the ACTUAL algorithm, not a hardcoded
    # "grpo". The cataloged path above already threads `algorithm` through _need; do the same here so
    # open-model OPD uses its own dense-logit + colocated-vLLM estimator. The grpo-only sequence
    # escalation stays gated on is_grpo.
    need = _need(params_b, algorithm, vocab=model_vocab)
    if is_opd:
        need = _opd_fp8_adjust(need, params_b, vocab=model_vocab)
    if is_grpo:
        need += grpo_seq_escalation_gb(params_b, seq_len)
    if is_vllm_rollout:
        floor_gb = 24 if params_b <= 1.0 else int(_VLLM_COLOCATE_FLOOR_GB)
        if is_opd and params_b >= 2.0:
            floor_gb = max(floor_gb, int(_OPD_VLLM_COLOCATE_FLOOR_GB))
        need = max(need, floor_gb, grpo_kv_floor_gb(params_b, seq_len, vllm_concurrency))
    return need


def fetch_hf_params_b(
    model_id: str, revision: str = "", *, strict: bool = False
) -> float | None:
    """Total params in billions from revision-aware HF safetensors metadata."""
    try:
        from huggingface_hub import HfApi

        kwargs = {"revision": revision} if revision else {}
        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            model_id, expand=["safetensors"], **kwargs
        )
        total = getattr(getattr(info, "safetensors", None), "total", None)
        if total:
            return float(total) / 1e9
        if strict:
            raise ValueError("safetensors parameter metadata is unavailable")
    except Exception as exc:
        if strict:
            raise ValueError(
                f"could not resolve revision-specific parameter metadata for model {model_id!r}"
            ) from exc
    return None


def resolve_params_b(model_id: str, revision: str = "") -> float | None:
    """Model size in billions, resolved the ONE way the worker and the cost estimator agree on:
    the curated catalog ``params_b`` (the required numeric field), else the real HF safetensors
    param count for an open-policy (uncataloged) model. Best-effort: returns None only
    when the model is uncataloged AND HF metadata is unavailable, so callers degrade to the
    size-unknown path (e.g. the fused-CE gate stays memory-safe, the colocate cap stays loose).
    The single source of truth for "how big is this model" -- run_sft, run_rl and cost.spec all
    call this so they can never drift."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    if revision:
        if info is not None:
            params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
            return params_b
        return fetch_hf_params_b(model_id, revision=revision, strict=True)
    if info is not None and info.params_b > 0:
        return info.params_b
    return fetch_hf_params_b(model_id)


def check_fit(
    model_id: str,
    algorithm: str,
    gpu: str,
    quant: str = "bf16",
    params_b: float | None = None,
    model_revision: str = "",
) -> VramEstimate:
    """Estimate whether ``model_id`` plausibly trains on ``gpu``; never raises."""
    gpu_gb = GPU_VRAM_GB.get(gpu, 32)
    if params_b is None:
        if model_revision:
            try:
                params_b = fetch_hf_params_b(model_id, revision=model_revision, strict=True)
            except Exception:
                params_b = None
        else:
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
