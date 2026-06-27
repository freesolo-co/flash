"""Coarse VRAM-fit estimation for one-consumer-GPU LoRA jobs.

Used by the open-model policy (``model_policy = "allow"``) to sanity-check that an
unlisted HF model can plausibly run on the requested GPU before provisioning it.

These are deliberately coarse heuristics (documented ±20%): they exist to catch
*provably impossible* configurations (70B bf16 on a 24 GB card) and to warn on tight
fits — not to guarantee success. Calibrated against the measured catalog entries
(Qwen3-0.6B/4B/8B, Qwen3.5 dense).
"""

from __future__ import annotations

import math
import os
import re
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

# Fixed overheads (GB): CUDA context + activations w/ gradient checkpointing +
# LoRA params/grads/Adam states (tiny at rank<=64) + fragmentation headroom.
_BASE_OVERHEAD_GB = 4.0
# Activations with gradient checkpointing scale ~linearly with tokens-in-flight
# (batch x seq) and model width (~sqrt of params). Coef calibrated so 4.7B SFT at
# seq 32k / batch 1 lands ~22 GB (measured: fits a 32 GB 5090).
_ACT_COEF = 0.12
# SFT activations + logits peak on the worker's PER-DEVICE micro-batch, not [train].batch_size
# (which is the global/effective batch realized via gradient accumulation). The worker caps the
# micro-batch at 4 and, when the fused CE is off, vocab-sizes it further to the logits budget (see
# ``sft_per_device``). Mirror that here so an unset/long-context SFT run still reserves the
# micro-batch peak, and a large effective batch isn't mis-counted as resident VRAM (it's grad-accum,
# not in-flight activations).
_SFT_PER_DEVICE_BS_DEFAULT = 4


def _sft_per_device_bs() -> int:
    """The worker's BASE per-device SFT micro-batch cap (before the big-vocab logits cap layered on
    by ``sft_per_device``) — the activation-peak driver to size against.

    SFT micro-batch is a MANAGED default: the control plane no longer forwards
    ``SFT_PER_DEVICE_BS`` to the worker (build_worker_env dropped the tuning allowlist), and the
    worker's own process env never carries it, so the worker always runs the fixed default. The
    allocator must size against that SAME fixed value — reading the control-plane process env here
    would size a card for a micro-batch the worker never uses, under-routing an
    ``SFT_PER_DEVICE_BS=1`` operator env to a too-small GPU that then OOMs at the default
    micro-batch 4 (the asymmetry the env-knobs cleanup removed everywhere else)."""
    return _SFT_PER_DEVICE_BS_DEFAULT


def sft_grad_accum(
    batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True
) -> tuple[int, int]:
    """(per-device micro-batch, grad-accum steps) the worker realizes for a requested GLOBAL
    ``batch_size``: per_device capped at the micro-batch default (and ADDITIONALLY vocab-sized to
    the logits budget when the fused CE is off — see ``sft_per_device``), grad-accum CEIL'd so the
    realized global batch is never BELOW the request (e.g. batch 6 -> per_device 4 x
    ceil(6/4)=2 grad-accum = realized 8, >= 6).

    ``seq_len``/``vocab``/``fused`` are the big-vocab logits-cap inputs; omitted (or ``fused``) they
    reduce to the old fixed per-device cap, so existing callers are unchanged."""
    target = max(1, int(batch_size))
    per_device = sft_per_device(target, seq_len=seq_len, vocab=vocab, fused=fused)
    grad_accum = max(1, -(-target // per_device))  # ceil
    return per_device, grad_accum


def sft_realized_batch(
    batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True
) -> int:
    """The realized SFT global batch (per_device x grad_accum) for a requested ``batch_size`` —
    mirrors the worker so the cost step-count matches. Pass seq_len/vocab/fused to honor the
    big-vocab per-device cap (the cost path does); omitted, it's the old fixed-cap behavior."""
    per_device, grad_accum = sft_grad_accum(batch_size, seq_len=seq_len, vocab=vocab, fused=fused)
    return per_device * grad_accum


# Colocated-GRPO vLLM KV pool: grows with the engine's max context (seq) and model
# width, but vLLM bounds the pool to a fraction of the card and PAGES rather than OOMs,
# so it's capped (_KV_CAP) instead of growing without bound at long context.
_KV_COEF = 2.0
_KV_CAP = 8.0


def grpo_rollout_seq_len(
    max_length: int = 0,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> int:
    """The vLLM engine context a GRPO run ACTUALLY uses, mirroring run_rl(): the run's
    ``[train].max_length`` when set, else ``max(1024, RLConfig.max_prompt_len + completion)`` where
    ``completion`` is ``[train].max_tokens`` or the recipe's thinking/non-thinking default. The
    allocator sizing, the sleep-mode resident gate, and the colocate KV budget all resolve the SAME
    length here so a run whose max_length is unset is not sized as a 1024-token rollout while the
    worker launches a ~2368-token (3584 with thinking) engine."""
    from flash.engine.recipe import RECIPE

    rl = RECIPE.rl
    completion = int(
        max_tokens or (rl.max_completion_len_thinking if thinking else rl.max_completion_len)
    )
    return int(max_length or max(1024, rl.max_prompt_len + completion))


def _resident_kv_gb(params_b: float | None, vllm_max_len: int, num_generations: int = 8) -> float:
    """KV (GB) a colocated rollout engine holds RESIDENT for the engine context + generation group.
    Scales with BOTH (vLLM's cache blocks must cover ``vllm_max_model_length`` for ``num_generations``
    concurrent sequences) -- unlike the sleep-mode rollout estimate, which caps it (``_KV_CAP``)
    because the engine is offloaded during the backward there. Shared by the resident-fit estimate
    and the non-sleep colocate budget so the gate and the budget size the SAME KV."""
    width = math.sqrt(max(float(params_b or 1.0), 0.1))
    return _KV_COEF * (max(1, vllm_max_len) / 1024.0) * width * (max(1, num_generations) / 8.0)


def colocate_kv_util(
    params_b: float | None,
    vllm_max_len: int,
    total_vram_gb: float,
    sleep_mode: bool,
    num_generations: int = 8,
) -> float:
    """``vllm_gpu_memory_utilization`` for the colocated GRPO rollout engine, sized to the ACTUAL need
    rather than a blanket fraction of the card.

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
    if not sleep_mode:
        # Resident KV ON TOP of the weight copy: gpu_memory_utilization is the WHOLE executor budget,
        # so budgeting KV alone (the old _KV_CAP/total) starved the weights and vLLM raised "No
        # available memory for the cache blocks" on >=3B models whose weights exceed an 8 GB budget.
        # The KV must ALSO cover the rollout context -- a flat _KV_CAP starves the cache blocks on a
        # long-context run (vLLM's blocks must span vllm_max_model_length), so scale it with the
        # context + group (floored at _KV_CAP for the validated short-context lean point, bounded by
        # the 0.45 util cap below). Matches the resident-fit estimate (estimate_vram_gb sleep_offload
        # =False) so grpo_sleep_mode's gate and this budget size the SAME KV.
        kv_gb = max(_KV_CAP, _resident_kv_gb(params_b, vllm_max_len, num_generations))
        return max(0.10, min(0.45, (weights_gb + kv_gb) / max(1.0, total_vram_gb)))
    # Sleep mode keeps a larger pool (1.5x margin): the engine is offloaded during the backward, so a
    # bigger rollout-phase KV does not compete with the training peak.
    kv_pool_gb = max(_KV_CAP, 1.5 * _resident_kv_gb(params_b, vllm_max_len, num_generations))
    return min(0.45, (weights_gb + kv_pool_gb) / max(1.0, total_vram_gb))
# GRPO backward (activations + fp32 logits over the completion micro-batch) per unit
# context x model width. Grad checkpointing makes this MILD in seq -- calibrated to
# measured boundaries: 0.8B GRPO fits 24 GB up to seq 32k (seq ~free), while 4.7B GRPO
# steps off a 32 GB card between seq 16k and 32k. group size scales it sublinearly.
_TRAIN_COEF = 0.27
# Fixed floor for colocated-vLLM GRPO: the vLLM engine's CUDA context + KV pool (sized to the
# CARD's VRAM via gpu_util, not the model) + the 2nd resident weight copy is ~model-independent
# for small models and dominates their param estimate, so tiny/mid models all need the 32 GB tier.
# MEASURED at the default group_size=8: 0.8B GRPO OOMs a 20 GB card; 2B GRPO OOMs a 24 GB card
# (-> both need 32); 4B GRPO fits 32 (param est ~31 already clears this floor, so it's untouched).
_VLLM_COLOCATE_FLOOR_GB = 28.0
# Fallback output vocab (lm_head / logits width) for estimate_vram_gb when no model vocab is
# passed; the model-aware path (model_required_vram_gb) resolves the real per-model value
# from flash.catalog via vocab_size_for(). Mirrors catalog._DEFAULT_VOCAB_SIZE.
_VOCAB_DEFAULT = 248_320
# Matches the worker's logits budget (6 GB): the per-device fp32 logits are capped to this
# (rl_per_device_comps spills the rest into grad-accum), so the estimator never reserves above it.
_LOGITS_BUDGET_GB = 6.0

# SFT big-vocab logits: the SFT analog of the GRPO fp32-logits term above
# When the worker's fused cross-entropy (Liger) is OFF, an SFT forward materializes the FULL-sequence
# [per_device, seq_len, vocab] logits AND keeps their gradient live through the backward. At
# Qwen3.5's ~248k vocab this is the documented big-vocab SFT OOM driver (a 0.8B SFT OOM'd a 24 GB
# card). The worker fuses CE only for a >=3B model OR a >=2048-token context (mirrors
# engine.worker.perf._memory_mode); BELOW that the term is real and was previously ignored entirely.
# An SFT step holds, AT ONCE, the fp32 logits (4) + their fp32 grad (4) + the bf16 logits the model
# emits (2) + the bf16 grad (2) + the cross-entropy log_softmax temp (4) ~= 16 B/elem. (8 B/elem --
# fp32 logits+grad only -- UNDER-counted: a live 2B SFT seq1024 at per_device=2 peaked ~15.8 GiB and
# OOM'd a 16 GB card whose usable is ~15.6 GiB.) At 16 B/elem the per-device cap drops to 1 for a
# big-vocab un-fused SFT, so the worker materializes far less and the real peak clears even the
# tightest 16 GB card. The worker vocab-sizes the per-device micro-batch so these logits never
# exceed _LOGITS_BUDGET_GB while pd CAN be reduced; the estimator reserves the TRUE per-device-capped
# term (no budget clamp -- the irreducible pd=1 floor can exceed the budget at a near-2048 ctx) -- so
# the allocator provably covers the worker's real peak. VALIDATED by a live re-run.
_SFT_LOGITS_BYTES_PER_ELEM = 16.0
# Canonical fused-CE (Liger) gate thresholds: the worker fuses the SFT cross-entropy for a >=3B
# model OR a >=2048-token context. SINGLE SOURCE OF TRUTH -- engine.worker.perf imports these (its
# _LIGER_MIN_PARAMS / _LONG_CONTEXT_TOKENS derive from them) and sft_logits_fused mirrors the gate
# offline (no network AutoConfig probe) so the cost estimator stays deterministic.
_LIGER_MIN_PARAMS_B = 3.0
_LIGER_LONG_CTX_TOKENS = 2048


def sft_logits_fused(params_b: float | None, seq_len: int) -> bool:
    """Whether the worker fuses the SFT cross-entropy (Liger), so the [per_device, seq, vocab] logits
    never materialize. Mirrors engine.worker.perf._memory_mode without a network probe: fused for a
    >=3B model OR a >=2048-token context. (The worker image bakes liger-kernel, so True here means
    the fused kernel is actually used; if it were ever absent the per-device cap still bounds the
    logits.)"""
    if seq_len >= _LIGER_LONG_CTX_TOKENS:
        return True
    return (params_b or 0.0) >= _LIGER_MIN_PARAMS_B


def sft_logits_per_device_cap(seq_len: int, vocab: int) -> int:
    """Largest SFT per-device micro-batch whose un-fused [per_device, seq, vocab] fp32 logits (+grad)
    fit _LOGITS_BUDGET_GB. The SFT mirror of rl_per_device_comps' completion cap, sizing the FULL
    sequence (not just the completion): the worker spills the remainder into grad-accum so the
    realized global batch is unchanged, and the estimator reserves the same bounded term."""
    denom = max(1, int(seq_len)) * max(1, int(vocab)) * _SFT_LOGITS_BYTES_PER_ELEM
    return max(1, int(_LOGITS_BUDGET_GB * 1e9 / denom))


def sft_per_device(batch_size: int, *, seq_len: int = 0, vocab: int = 0, fused: bool = True) -> int:
    """The per-device SFT micro-batch the worker runs: the requested global batch capped at the
    micro-batch default (4) and ADDITIONALLY vocab-sized to the logits budget when the fused CE is
    OFF (small model AND short context) — so the big-vocab [per_device, seq, vocab] logits can't OOM
    the card. With seq_len/vocab unset (or fused), this remains the fixed cap."""
    per_device = max(1, min(_SFT_PER_DEVICE_BS_DEFAULT, max(1, int(batch_size))))
    if not fused and seq_len and vocab:
        per_device = min(per_device, sft_logits_per_device_cap(seq_len, vocab))
    return per_device


def grpo_seq_escalation_gb(params_b: float | None, seq_len: int) -> int:
    """Extra GB a long-context GRPO run needs beyond its base footprint.

    Big-model GRPO is tight: colocate holds 2 weight copies + a KV pool, so headroom shrinks
    with model size and long context overflows it. Calibrated on a bf16 9.7B GRPO run (RunPod):
    fits 80 GB to seq 4096 but OOMs at 8192. Safe headroom ~ 48500/params_b tokens; past that
    escalate, STEEPER for bigger models. Applies to both catalog and open-model GRPO so neither
    under-provisions.
    """
    coef = 0.9
    if not params_b:
        return 0
    seq_thresh = 48_500.0 / params_b
    if seq_len <= seq_thresh:
        return 0
    return math.ceil(coef * params_b * (seq_len / seq_thresh - 1))


def params_b_from_str(s: str | None) -> float | None:
    """Leading param count (billions) from a catalog ``params`` string, e.g.
    "4.7B (text-only fine-tune)" -> 4.7, "9.7B (text-only fine-tune)" -> 9.7."""
    if not s:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*B", s)
    return float(m.group(1)) if m else None


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
    """Estimated peak VRAM (GB) for a LoRA job on one GPU, over the full knob matrix.

    Terms (all in GB):
      weights      params x bytes/param (bf16=2)
      base         CUDA context + framework + fragmentation headroom
      lora_opt     LoRA adapter + grads + Adam states (rank-linear, model-scaled)
      activations  grad-checkpointed activations ~ batch x seq x sqrt(params)
      grpo only:
        +weights   colocated vLLM holds a 2nd resident weight copy at the rollout peak
                   (sleep mode offloads it BETWEEN steps, not during) -- skipped when
                   use_vllm is False (transformers generation, single copy)
        kv         vLLM KV pool ~ seq x sqrt(params)
        logits     fp32 logits [per_device_comps, completion, vocab]

    MoE: ``active_params_b`` (params ACTIVE per token; set for an A-xB MoE) drives the terms that
    scale with the dense BACKBONE / per-token compute -- activations, KV pool, and the
    rank-linear LoRA optimizer (LoRA targets the attention/router backbone, not the fused experts).
    The ``weights`` term stays on the FULL ``params_b`` (every expert is resident). So a 35B-A3B
    sizes its activations/KV like a ~3B model while still reserving the full 70 GB of weights --
    which is what lets it fit a much smaller card than its total size implies. ``None``/0 (dense)
    falls back to ``params_b`` for every term (unchanged)."""
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    # Backbone/compute scale: the active params for an MoE, else the full size.
    eff_b = float(active_params_b) if active_params_b else params_b
    algo = "grpo" if (algorithm or "").lower() in ("grpo", "rl") else "sft"
    width = math.sqrt(max(eff_b, 0.1))
    lora_opt = (lora_rank / 16.0) * (0.3 + 0.04 * eff_b)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    if algo == "grpo":
        # GRPO alternates two phases that DON'T peak together (sleep mode offloads the
        # vLLM engine during the backward), so peak = max(rollout, train), not the sum:
        #   rollout: colocated vLLM 2nd weight copy + KV pool (skipped if use_vllm=False)
        #   train:   backward activations + fp32 logits -- MILD in seq (grad ckpt)
        rollout = 0.0
        if use_vllm:
            if sleep_offload:
                # Sleep mode offloads the engine during the backward, so the rollout-phase KV (capped
                # at _KV_CAP) never competes with the training peak. This is the ALLOCATOR's estimate
                # (model_required_vram_gb) -- keep it calibrated; it sizes every GRPO allocation.
                rollout = weights + min(_KV_COEF * (seq_len / 1024.0) * width, _KV_CAP)
            else:
                # Resident: the engine stays live THROUGH the backward, so its KV (which must cover the
                # rollout context for the whole generation group) is held alongside training -- size it
                # to the real context, matching colocate_kv_util's non-sleep budget, instead of the
                # flat _KV_CAP (which let grpo_fits_resident wrongly admit long-context runs).
                rollout = weights + _resident_kv_gb(eff_b, seq_len, group_size)
        group_factor = max(1.0, (max(1, group_size) / 4.0) ** 0.5)
        think_factor = 1.3 if thinking else 1.0
        activations = _TRAIN_COEF * (seq_len / 1024.0) * width * group_factor * think_factor
        # fp32 logits [per_device, completion, vocab] are the documented GRPO OOM driver. The
        # worker MEMORY-CAPS per_device (rl_per_device_comps) so the live logits never exceed the
        # logits budget (6 GB) and the rest spills into grad-accum -- so the IRREDUCIBLE floor the
        # card must hold is the per_device=1 logits for the completion length: it scales with
        # max_tokens (NOT seq_len) and is capped at the budget. completion defaults to the recipe
        # budget (~min(seq_len, 1024)) when max_tokens is unset.
        completion = max_tokens if max_tokens else min(seq_len, 1024)
        logits = min(completion * vocab * 4 / 1e9, _LOGITS_BUDGET_GB)
        train = activations + logits
        # Sleep mode offloads the vLLM rollout engine during the backward, so rollout and train
        # don't peak together (peak = max). WITHOUT sleep the engine stays resident through the
        # backward, so both are live at once (peak = sum). sleep_offload=False sizes that resident
        # peak -- used by grpo_fits_resident to decide whether a run can skip sleep mode.
        return base + (max(rollout, train) if sleep_offload else rollout + train)
    # SFT: peak = base + activations + the big-vocab logits term. Both activations and logits are
    # driven by the worker's per-device micro-batch (capped at 4 AND vocab-sized to the logits budget
    # when the fused CE is off), NOT the global/effective batch_size (grad-accum realizes that). Use
    # the SAME ``sft_per_device`` the worker runs so the estimate tracks what actually executes.
    fused = sft_logits_fused(params_b, seq_len)
    pd = sft_per_device(batch_size, seq_len=seq_len, vocab=vocab, fused=fused)
    activations = _ACT_COEF * pd * (seq_len / 1024.0) * width
    # fp32-logits term: 0 when the worker fuses CE (>=3B model OR >=2048-token ctx, so the lm_head
    # never materializes [B,T,vocab]); else the [per_device, seq_len, vocab] logits the forward holds.
    # Reserve the TRUE per-device-capped value -- NOT clamped to the budget: the budget only chooses
    # ``pd`` (so pd>1 cases stay <= budget), but once pd floors to 1 the logits are IRREDUCIBLE and
    # can exceed the budget at a near-2048 ctx -- clamping there would under-reserve and OOM (the
    # worker can't go below pd=1). The SFT analog of the GRPO logits term, sized over the FULL seq_len
    # (SFT loss spans the sequence) -- the term the SFT estimate once ignored entirely.
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
    """Whether a colocated-vLLM GRPO run fits RESIDENT (no vLLM sleep-mode offload) on a card of
    ``card_vram_gb`` with a safety ``margin``. When it fits, sleep mode is unnecessary -- and the
    sleep/wake cycle is what stalls the large-model GRPO rollout -- so the worker can skip it.
    Conservative: an unknown card size or unknown model size returns False (keep the memory-safe
    sleep default)."""
    if not card_vram_gb or card_vram_gb <= 0:
        return False
    from flash.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    params_b = float(getattr(info, "params_b", 0.0) or 0.0) if info else 0.0
    if params_b <= 0:
        return False  # unknown size (open-model path) -> keep the safe default
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
    """Cheapest-sufficient VRAM (GB) for a specific run -- the matrix the allocator and
    ``provisional_gpu`` both size against.

    Catalog models size from their known param count + the run's actual knobs (``train``
    may be a TrainSpec, a dict, or None for recipe defaults). Curated GRPO floors
    (``grpo_min_vram_gb``) stay as HARD floors so we never under-provision a validated
    model; the matrix only ever sizes UP from there. Unlisted open models size from HF
    metadata, falling back to the 24 GB tier when the size can't be read.
    """

    # Best-effort knob extraction: this provisional sizing runs at parse time BEFORE the
    # dedicated train validators, so malformed knobs (nan/inf/strings/<=0) must fall back
    # to a default here, never crash -- config_schema raises the proper ConfigError next.
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
    # Default sequence length when [train].max_length is unset. For GRPO this must MIRROR what
    # run_rl() actually starts vLLM at — max(1024, RLConfig.max_prompt_len + completion) — not a
    # flat 1024, or the allocator can pick a GPU sized for 1024 tokens while the worker launches a
    # ~2368-token (3584 with thinking) engine and OOMs after provisioning. Completion = the run's
    # [train].max_tokens override, else the recipe's thinking/non-thinking completion default.
    if (algorithm or "").lower() in ("grpo", "rl"):
        # Same engine context run_rl() launches (max_length, else max(1024, prompt+completion)) via
        # the shared helper, so the allocator and the worker never disagree on the rollout length.
        _grpo_default_len = grpo_rollout_seq_len(0, max_tokens, thinking)
    else:
        _grpo_default_len = 1024
    seq_len = _pos_int(_g(train, "max_length"), _grpo_default_len)
    lora_rank = _pos_int(_g(train, "lora_rank"), 32)
    group_size = _pos_int(_g(train, "group_size"), 8)
    # Default to the worker's per-device SFT micro-batch (4): an unset
    # [train].batch_size still realizes that micro-batch on the worker, so size for it
    # rather than 1 (which would under-route a long-context SFT run to a too-small card).
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
        # estimate over the run's full knob matrix, then apply the safety headroom. Both the
        # catalog and open-model paths size through here so they stay in sync on the knob set.
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
    # Per-model output vocab (lm_head / logits width) sizes the fp32-logits term; resolved
    # from the catalog (curated value, else open-model fallback) instead of a hardcoded const.
    model_vocab = vocab_size_for(model_id)
    is_grpo = (algorithm or "").lower() in ("grpo", "rl")
    if info is not None:
        # Total weight size comes from the CURATED ``params_b`` (the single source of truth
        # resolve_params_b and the cost model read directly), falling back to the ``params`` display
        # string only when it's unset. Re-parsing the string is fragile for an MoE whose string lists
        # BOTH the total and the ~3B active count ("35B total / ~3B active"): a reordering could make
        # params_b_from_str pick up the active count and size the resident weights ~10x too small,
        # under-provisioning the card. The calibrated params_b (the MoE's 35.0) is authoritative.
        params_b = float(getattr(info, "params_b", 0.0) or 0.0) or params_b_from_str(info.params)
        quant = getattr(info, "quant", "bf16") or "bf16"
        # GRPO always runs the rollout on a colocated vLLM engine, so sizing must reserve room for
        # the 2nd (rollout) weight copy on the same card.
        use_vllm = True
        # MoE: the curated active-per-token count sizes the backbone/compute terms (activations, KV,
        # LoRA), while ``params_b`` still sizes the resident weights. 0/dense -> falls back to params_b.
        active_b = float(getattr(info, "active_params_b", 0.0) or 0.0)
        need = _need(
            params_b or 4.0,
            algorithm,
            quant=quant,
            use_vllm=use_vllm,
            vocab=model_vocab,
            active_params_b=active_b,
        )
        # Hard floor the param-based matrix can't see: a curated GRPO floor.
        floor = 0
        if is_grpo and getattr(info, "grpo_min_vram_gb", 0):
            floor = int(info.grpo_min_vram_gb)
        # SFT analog: a curated SFT floor pins a very large checkpoint to a bigger card than its raw
        # param estimate would pick (e.g. the 35B MoE's ~89 GB est would otherwise down-route to a
        # 96 GB card with a thin margin over its 70 GB frozen weights -> floor it to the 180 GB B200).
        if not is_grpo and getattr(info, "sft_min_vram_gb", 0):
            floor = max(floor, int(info.sft_min_vram_gb))
        # Big-model GRPO is TIGHT at its floor (2 weight copies + KV pool), so long context overflows
        # it -> escalate to a bigger tier (or, when already on the biggest card, push the requirement
        # PAST every GPU so the run is REJECTED at parse time instead of booting and OOMing in vLLM's
        # KV alloc). The KV/colocate pressure scales with the ATTENTION width — the ACTIVE params — so
        # escalate on active_params_b for an MoE; keying on the 35B TOTAL would over-reject (its dense
        # ~1385-token threshold sits below the default GRPO rollout length, breaking ordinary GRPO),
        # while the ~3B active gives ~16k tokens of headroom that still rejects a 32k run. Dense models
        # leave active_params_b unset -> falls back to params_b, unchanged. See grpo_seq_escalation_gb.
        if is_grpo and floor:
            floor += grpo_seq_escalation_gb(active_b or params_b, seq_len)
        need = max(need, floor)
        # vLLM-colocate floor: the engine (CUDA context + KV pool sized to the CARD's VRAM +
        # framework) + the 2nd resident weight copy add a ~constant the param estimate misses,
        # so small-model GRPO under-provisions. MEASURED at the default group_size=8: 0.8B GRPO
        # fits a 24 GB card but OOMs 20 (est ~18, ~6 GB headroom on 24); 2B GRPO OOMs a 24 GB
        # card (est ~20 but the colocate cost tips it past 24 -> needs the 32 tier). So sub-~1B
        # models floor at 24, while larger small-models that the param estimate still under-shoots
        # floor at the 32 tier. 4B+ already exceed this via their param estimate, so untouched.
        if is_grpo and use_vllm:
            floor_gb = 24 if (params_b or 0.0) <= 1.0 else int(_VLLM_COLOCATE_FLOOR_GB)
            need = max(need, floor_gb)
        return need
    # Unlisted open model: size from HF metadata (GRPO is the heavier phase).
    params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return 24
    # Open models size against the heavier GRPO phase regardless of the requested algorithm.
    need = _need(params_b, "grpo", vocab=model_vocab)
    # Same long-context GRPO escalation as the catalog path so a big open model isn't
    # under-provisioned at long context either.
    if is_grpo:
        need += grpo_seq_escalation_gb(params_b, seq_len)
    return need


def fetch_hf_params_b(model_id: str) -> float | None:
    """Total params (billions) from the HF API safetensors metadata (no download).

    Best-effort: returns ``None`` when the size can't be read (no network / no HF metadata),
    so callers fall back to the offline heuristic rather than failing.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            model_id, expand=["safetensors"]
        )
        total = getattr(getattr(info, "safetensors", None), "total", None)
        if total:
            return float(total) / 1e9
    except Exception:
        # Best-effort size probe (network/HF-metadata may be unavailable); fall through
        # to None so callers report "size unknown" rather than failing.
        pass
    return None


def resolve_params_b(model_id: str) -> float | None:
    """Model size in billions, resolved the ONE way the worker and the cost estimator agree on:
    the curated catalog ``params_b`` (else its ``params`` display string), else the real HF
    safetensors param count for an open-policy (uncataloged) model. Best-effort: returns None only
    when the model is uncataloged AND HF metadata is unavailable, so callers degrade to the
    size-unknown path (e.g. the fused-CE gate stays memory-safe, the colocate cap stays loose).
    The single source of truth for "how big is this model" -- run_sft, run_rl and cost.spec all
    call this so they can never drift."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        pb = getattr(info, "params_b", 0.0) or params_b_from_str(getattr(info, "params", None))
        if pb:
            return pb
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
