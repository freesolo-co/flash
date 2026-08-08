"""Conservative architecture-aware VRAM fit estimation for single-GPU LoRA jobs."""

from __future__ import annotations

import json
import math
import os


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
_VOCAB_DEFAULT = 248_320
# must track verl's `FusedLinearForPPO(chunk_size=...)` default
# (verl/utils/experimental/torch_functional.py). the child projects the vocab in chunks of this
# many token rows, so a smaller value here under-reserves and admits a job that then OOMs.
VERL_FUSED_CE_CHUNK_TOKENS = 512
OPD_CE_CHUNK_SIZE = VERL_FUSED_CE_CHUNK_TOKENS
_OPD_CE_PEAK_BYTES_PER_LOGIT = 16


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


_KV_COEF = 2.0
_KV_CAP = 8.0
_KV_BLOCK_TOKENS = 16
# vllm profile, graph, scheduler, and allocator costs beyond raw cache tensors. the 8 gb measured
# floor remains authoritative at short context; longer contexts retain 1.5 gb plus 25% fragmentation.
_KV_PROFILE_OVERHEAD_GB = 1.5
_KV_FRAGMENTATION_MARGIN = 1.25
# adapter bytes per trainable parameter. the 8-bit paged optimizer needs ~10; fp32 AdamW needs 16
# (4 param + 2 gradient + 8 moments + 2 bf16 shadow). SFT and OPD size on AdamW because verl builds
# torch.optim.AdamW for both -- OPD inherits verl's own default, and SFT names it explicitly since
# bitsandbytes' 8-bit blockwise update is not a distributed operator and dies on FSDP2's DTensor
# parameters. the backend is a [worker_env] property that never reaches this sizing path, so the
# heavier optimizer is the only safe basis: under-reserving OOMs a paid run.
#
# GRPO stays on the paged constant. verl GRPO also runs AdamW, so it is under-sized by the same
# 6 bytes/param -- but GRPO's estimate is already at the B200 ceiling (the 35B MoE at 4096 ctx sizes
# to 180 GB exactly), and widening it there REJECTS a configuration that runs today. that gap is real
# and tracked separately; correcting it needs the resident-peak model retuned, not a constant swap.
_LORA_PAGED_BYTES_PER_PARAM = 10.0
_LORA_ADAMW_BYTES_PER_PARAM = 16.0


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
    return int(
        max_tokens or (opd.max_completion_len_thinking if thinking else opd.max_completion_len)
    )


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
    """Loss microbatch size used by OPD's GKD backward.

    Keep this in lockstep with worker.opd._opd_loss_microbatch_size without importing the worker from
    the sizing path. Small/medium catalog models run up to four loss samples per forward; 35B-class
    models stay serial for VRAM safety.
    """
    total = opd_rollout_concurrency(prompts_per_step, group_size)
    params = float(params_b or 0.0)
    default = 4 if params and params <= 10.0 else 1
    return max(1, min(total, default))


def _legacy_lora_floor_gb(lora_rank: int, effective_params_b: float) -> float:
    """Measured floor retained until the shape equation has broader live-GPU calibration."""
    return (lora_rank / 16.0) * (0.3 + 0.04 * effective_params_b)


def _lora_memory_gb(
    lora_rank: int,
    effective_params_b: float,
    algorithm: str,
    model_info=None,
) -> float:
    """Adapter, gradient, and optimizer memory for the actual PEFT all-linear targets."""
    floor = _legacy_lora_floor_gb(lora_rank, effective_params_b)
    shapes = tuple(getattr(model_info, "lora_target_shapes", ()) or ())
    if not shapes:
        return floor
    target_dims = sum(
        (int(in_features) + int(out_features)) * int(count)
        for in_features, out_features, count in shapes
    )
    trainable_params = max(1, int(lora_rank)) * target_dims
    bytes_per_param = (
        _LORA_PAGED_BYTES_PER_PARAM
        if (algorithm or "").lower() in ("grpo", "rl")
        else _LORA_ADAMW_BYTES_PER_PARAM
    )
    exact = trainable_params * bytes_per_param / 1e9
    return max(floor, exact)


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, math.ceil(max(1, value) / multiple) * multiple)


def _architecture_kv_raw_gb(
    model_info,
    vllm_max_len: int,
    num_generations: int,
    fp8_kv: bool,
) -> float | None:
    """Raw per-layer attention KV plus recurrent-state pages from catalog geometry."""
    attention_layers = int(getattr(model_info, "num_attention_layers", 0) or 0)
    kv_heads = int(getattr(model_info, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(model_info, "head_dim", 0) or 0)
    if not (attention_layers and kv_heads and head_dim):
        return None

    sequences = max(1, int(num_generations))
    seq_len = max(1, int(vllm_max_len))
    kv_bytes = 1 if fp8_kv else 2
    attention_bytes_per_token = 2 * kv_heads * head_dim * kv_bytes
    attention_tokens = _round_up(seq_len, _KV_BLOCK_TOKENS)
    total_bytes = attention_layers * sequences * attention_tokens * attention_bytes_per_token

    linear_layers = int(getattr(model_info, "num_linear_attention_layers", 0) or 0)
    if linear_layers:
        key_heads = int(getattr(model_info, "linear_num_key_heads", 0) or 0)
        value_heads = int(getattr(model_info, "linear_num_value_heads", 0) or 0)
        key_dim = int(getattr(model_info, "linear_key_head_dim", 0) or 0)
        value_dim = int(getattr(model_info, "linear_value_head_dim", 0) or 0)
        conv_kernel = int(getattr(model_info, "linear_conv_kernel_dim", 0) or 0)
        if not all((key_heads, value_heads, key_dim, value_dim, conv_kernel)):
            # linear-attention dims are absent from the catalog: we can't size the recurrent/conv
            # state, but the standard attention KV already accumulated in total_bytes is a better
            # (if partial) estimate than the params-only legacy heuristic. The caller floors this
            # partial value with legacy so a hybrid model is never under-sized below the heuristic.
            return total_bytes / 1e9
        # vllm stores qwen gated-deltanet recurrent and convolution state in bf16 pages.
        state_elements = value_heads * key_dim * value_dim
        state_elements += (key_heads * key_dim + value_heads * value_dim) * conv_kernel
        state_bytes = state_elements * 2
        state_block_tokens = _round_up(
            math.ceil(state_bytes / attention_bytes_per_token), _KV_BLOCK_TOKENS
        )
        catalog_block = int(getattr(model_info, "mamba_block_size", 0) or 0)
        if fp8_kv and catalog_block:
            state_block_tokens = max(state_block_tokens, catalog_block)
        state_pages = math.ceil(seq_len / state_block_tokens)
        total_bytes += linear_layers * sequences * state_pages * state_bytes

    return total_bytes / 1e9


def _resident_kv_gb(
    params_b: float | None,
    vllm_max_len: int,
    num_generations: int = 8,
    fp8_kv: bool = False,
    model_info=None,
    preserve_legacy_floor: bool = False,
) -> float:
    """Profiled vLLM cache budget from architecture, with measured conservative floors."""
    width = math.sqrt(max(float(params_b or 1.0), 0.1))
    legacy = _KV_COEF * (max(1, vllm_max_len) / 1024.0) * width
    legacy *= max(1, num_generations) / 8.0
    if fp8_kv:
        legacy *= 0.5

    raw = _architecture_kv_raw_gb(model_info, vllm_max_len, num_generations, fp8_kv)
    if raw is None:
        return legacy
    profiled = max(_KV_CAP, _KV_PROFILE_OVERHEAD_GB + raw * _KV_FRAGMENTATION_MARGIN)
    # opd startup tiers and the resident-only 35b boundary are live-calibrated. retain their old
    # conservative pool until the architecture equation is validated on those exact gpu paths.
    # a hybrid model whose linear-attention dims are absent yields a PARTIAL estimate (attention KV
    # only); floor it with legacy so the dropped recurrent/conv state can't under-size it.
    _declares_linear = int(getattr(model_info, "num_linear_attention_layers", 0) or 0) > 0
    _linear_dims_known = all(
        int(getattr(model_info, _f, 0) or 0)
        for _f in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    )
    if (
        preserve_legacy_floor
        or getattr(model_info, "sleep_unsupported", False)
        or (_declares_linear and not _linear_dims_known)
    ):
        profiled = max(profiled, legacy)
    return profiled


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
    model_info=None,
    preserve_legacy_floor: bool = False,
) -> float:
    """vllm_gpu_memory_utilization for the colocated GRPO rollout engine.

    ``gpu_memory_utilization`` is vLLM's WHOLE model-executor budget — its (2nd) bf16 weight copy PLUS
    the KV cache — so we budget BOTH (budgeting KV alone would starve the weights and, for big models,
    under-size the engine). Curated models size attention KV and recurrent-state pages from catalog
    geometry, then retain a profiled overhead, fragmentation margin, and the measured 8 GB floor.
    Uncataloged models retain the legacy conservative equation. The old blanket sleep-path 0.45 reserved
    ~36 GB on an 80 GB A100, measured as the dominant resident allocation that set the GRPO step peak
    (~46 GB). Both paths budget the weight copy plus cache. At the validated 4B/group8/2k point, 0.25
    utilization reduced the peak from 46 GB to 26 GB with byte-identical reward and neutral train wall;
    a tighter 12 GB budget preempts, confirming the retained floor."""
    weights_gb = (
        max(0.5, float(params_b or 1.0)) * 2.0
    )  # vLLM's bf16 weight copy lives in the budget
    # catalog geometry is authoritative. active params remain only for the uncataloged fallback, while
    # the weight copy always uses total parameters.
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
        # =False) so the resident-fit gate and this budget size the SAME KV.
        kv_gb = max(
            _KV_CAP,
            _resident_kv_gb(
                kv_params_b,
                vllm_max_len,
                num_generations,
                fp8_kv=fp8_kv,
                model_info=model_info,
                preserve_legacy_floor=preserve_legacy_floor,
            ),
        )
        return max(0.10, min(_util_cap, (weights_gb + kv_gb) / max(1.0, total_vram_gb)))
    # Sleep mode keeps a larger pool (1.5x margin): the engine is offloaded during the backward, so a
    # bigger rollout-phase KV does not compete with the training peak.
    kv_pool_gb = max(
        _KV_CAP,
        1.5
        * _resident_kv_gb(
            kv_params_b,
            vllm_max_len,
            num_generations,
            fp8_kv=fp8_kv,
            model_info=model_info,
            preserve_legacy_floor=preserve_legacy_floor,
        ),
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
_LOGITS_BUDGET_GB = 6.0
# 16 B/elem: fp32 logits+grad + bf16 logits+grad + CE temp. 8 B/elem under-counts (live OOM confirmed).
_SFT_LOGITS_BYTES_PER_ELEM = 16.0
_SFT_CHUNKED_NLL_TOKENS = VERL_FUSED_CE_CHUNK_TOKENS
# shared thresholds for the independent liger and worker memory gates.
_LIGER_MIN_PARAMS_B = 3.0
_LIGER_LONG_CTX_TOKENS = 2048

# the fused, dense-logit-free sft loss is validated against these model families. other models keep
# plain nll until their output-head and backbone traversal are covered by the same parity tests.
# the implementation moved from trl's chunked_nll to verl's use_fused_kernels (set unconditionally
# in engine/worker/sft_train.py, with impl_backend=torch); the sizing property -- no dense
# [b, s, vocab] logits tensor -- is what this set gates, and it holds for both. liger is NOT part
# of that property: verl disables liger's fused linear ce, and liger zeroed the sft lora gradient
# (GRAD-001), so sft runs with use_liger=false.
#
# membership is decided by the checkpoint's ``model_type``, because that is what verl's
# patch_forward_with_backends dispatches on: ``qwen3_5`` and ``qwen3_5_moe`` both resolve to
# verl/models/transformers/qwen3_5.py's forward_with_torch_backend, which computes ce from hidden
# states via FusedLinearForPPO and never builds the dense tensor. every catalog model reports one of
# those two types (the 27B is ``qwen3_5``/``Qwen3_5ForConditionalGeneration`` despite its 3.6 name),
# so the set must list the whole sft-capable catalog -- an omission does not disable the fused
# kernel, it only makes the allocator reserve logits vram the run never allocates and collapse the
# micro-batch to 1. contrast kimi_vl, which returns before the fused patch ("Not support fused
# kernels for KimiVL") and would genuinely keep dense logits.
_SFT_CHUNKED_NLL_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-27B",
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


def grpo_kv_floor_gb(
    params_b: float,
    vllm_max_len: int,
    group_size: int = 8,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    model_info=None,
    preserve_legacy_floor: bool = False,
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
    need = weights_gb + 0.5 * _resident_kv_gb(
        kv_params_b,
        vllm_max_len,
        group_size,
        fp8_kv=fp8_kv,
        model_info=model_info,
        preserve_legacy_floor=preserve_legacy_floor,
    )
    lower = math.ceil(need / 0.55)
    upper = math.ceil(need / 0.45)
    for total_vram_gb in range(lower, upper + 1):
        if _colocate_util_cap(weights_gb, total_vram_gb) * total_vram_gb >= need:
            return total_vram_gb
    return upper


def _declares_linear_attention(model_info, model_id: str = "") -> bool:
    """True for a GDN hybrid, using the catalog rather than a network config fetch.

    The two vLLM rollout workers refuse an fp8 KV cache for these models (vllm's fp8 wake path
    init_fp8_kv_scales assumes a plain kv tensor and crashes on the hybrid cache), so sizing must
    not apply the fp8 discount to them or it reserves half the cache the run allocates. This is the
    offline half of engine/worker/packing.py:model_is_gdn_hybrid, which probes the HF config; the
    catalog carries num_linear_attention_layers for every model it routes.

    ``model_id`` is the fallback the pinned-revision path needs. A pinned commit drops to generic
    architecture sizing (``sizing_info = None``), which is right for geometry but wrong here:
    attention family is a property of the model, not of the commit, and the worker's runtime gate
    reads the pinned config and refuses fp8 all the same. Consulting the catalog by id keeps the two
    in agreement instead of handing pinned GDN runs a discount their cache never takes.
    """
    if int(getattr(model_info, "num_linear_attention_layers", 0) or 0) > 0:
        return True
    if not model_id:
        return False
    from flash.catalog import MODELS

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
    preserve_legacy_floor: bool = False,
) -> int:
    floor = grpo_kv_floor_gb(
        params_b,
        vllm_max_len,
        concurrency,
        active_params_b=active_params_b,
        model_info=model_info,
        preserve_legacy_floor=preserve_legacy_floor,
    )
    from flash.providers.base import max_non_fp8_kv_vram_gb

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
        preserve_legacy_floor=preserve_legacy_floor,
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
    use_vllm: bool = True,
    vocab: int = _VOCAB_DEFAULT,
    sleep_offload: bool = True,
    active_params_b: float | None = None,
    fp8_kv: bool = False,
    sft_fused_ce: bool | None = None,
    model_info=None,
) -> float:
    """Estimated peak VRAM (GB) for a LoRA job on one GPU.

    MoE: active_params_b drives activations; weights and actual LoRA targets cover the full model.
    """
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    eff_b = float(active_params_b) if active_params_b else params_b
    is_opd = (algorithm or "").lower() == "opd"
    algo = "grpo" if (algorithm or "").lower() in ("grpo", "rl") else "sft"
    width = math.sqrt(max(eff_b, 0.1))
    lora_opt = _lora_memory_gb(lora_rank, eff_b, algorithm, model_info)
    base = weights + _BASE_OVERHEAD_GB + lora_opt
    if algo == "grpo":
        # Sleep mode: peak = max(rollout, train). Resident: both live at once, peak = sum.
        rollout = 0.0
        if use_vllm:
            if sleep_offload:
                kv = _resident_kv_gb(
                    eff_b,
                    seq_len,
                    group_size,
                    fp8_kv=fp8_kv,
                    model_info=model_info,
                )
                # the sleep-mode worker (colocate_kv_util) budgets max(_KV_CAP, 1.5 * kv) at the real
                # group_size. only mirror that lift when kv is backed by real catalog geometry AND
                # exceeds _KV_CAP -- that is the finding's case (architecture-aware KV above the cap),
                # where the old min(kv, _KV_CAP) under-counted the pool by 4-11 gb and preflight
                # admitted a gpu the sleep executor then overran. the generic legacy KV formula
                # OVER-estimates long context, and its _KV_CAP clamp is what keeps the uncataloged
                # estimate on its measured train/OOM boundary, so keep min(kv, _KV_CAP) whenever the
                # geometry is absent (generic) or the arch-aware kv still fits the cap (short context).
                arch_kv_known = (
                    _architecture_kv_raw_gb(model_info, seq_len, group_size, fp8_kv) is not None
                )
                if arch_kv_known and kv > _KV_CAP:
                    rollout = weights + max(_KV_CAP, 1.5 * kv)
                else:
                    rollout = weights + min(kv, _KV_CAP)
            else:
                rollout = weights + _resident_kv_gb(
                    eff_b,
                    seq_len,
                    group_size,
                    fp8_kv=fp8_kv,
                    model_info=model_info,
                )
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
            _KV_CAP,
            _resident_kv_gb(
                eff_b,
                seq_len,
                rollout_concurrency,
                fp8_kv=fp8_kv,
                model_info=model_info,
                preserve_legacy_floor=True,
            ),
        )
        # text opd computes exact full-vocabulary ce in chunks of VERL_FUSED_CE_CHUNK_TOKENS rows.
        # the child projects prompt AND completion positions (verl's fused qwen forward takes no
        # `logits_to_keep` and does not slice hidden states), so the chunk cap, not the completion
        # length, is what bounds the projection. image samples still use the dense full-sequence
        # logits path and are processed one at a time. allocation does not know the dataset modality,
        # so reserve the larger of one text ce chunk and one dense image loss forward/backward peak.
        completion = opd_completion_len(max_tokens, thinking)
        loss_mb = opd_loss_microbatch(params_b, batch_size, group_size)
        activations = loss_mb * _ACT_COEF * (seq_len / 1024.0) * width
        ce_rows = min(OPD_CE_CHUNK_SIZE, loss_mb * completion)
        chunked_logits = ce_rows * vocab * _OPD_CE_PEAK_BYTES_PER_LOGIT / 1e9
        dense_image_logits = (seq_len * 4 + completion * 8) * vocab / 1e9
        logits = max(chunked_logits, dense_image_logits)
        return base + rollout + activations + logits
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
    revision: str = "",
    margin: float = 1.15,
) -> bool:
    """True when GRPO fits resident (no sleep-mode offload); False on unknown model/card (safe default)."""
    if not card_vram_gb or card_vram_gb <= 0:
        return False
    from flash.catalog import MODELS, vocab_size_for

    catalog_info = MODELS.get(model_id)
    if revision:
        params_b = float(resolve_params_b(model_id, revision=revision) or 0.0)
    else:
        params_b = float(getattr(catalog_info, "params_b", 0.0) or 0.0) if catalog_info else 0.0
    if params_b <= 0:
        return False
    quant = (getattr(catalog_info, "quant", "bf16") or "bf16") if catalog_info else "bf16"
    # pinned revisions use the conservative generic kv and lora geometry, matching the runtime budget.
    info = None if revision else catalog_info
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
        model_info=info,
    )
    return resident * margin <= card_vram_gb


# Activations retained for the backward when gradient checkpointing is OFF. With GC, each layer's
# intra-block activations are RECOMPUTED in the backward (one extra forward ~= +33% compute); with GC
# OFF they all stay resident -- so this term is ~num_layers x the per-layer (residual stream + qkv/out
# + MoE SwiGLU intermediate) activations per token. With FlashAttention / cuDNN-SDPA there is NO s^2
# attention-score tensor, and the GatedDeltaNet linear layers store only a small recurrent/conv state,
# so it is LINEAR in seq. K folds the per-layer multiplier (relative to hidden) into one constant.
# bf16 activations (2 bytes/elem).
#
# K is GEOMETRY-SPECIFIC, so there are two. What K multiplies is layers x batch x seq x hidden, i.e.
# it is the per-layer activation cost measured RELATIVE TO HIDDEN -- and a dense block stores far more
# of it per unit of hidden than a sparse-MoE block does, because the dense FFN intermediate is
# 4 x 2048 against the MoE's 8 x 512 active. One constant cannot describe both.
#
# DENSE: measured on a live RTX 5090, Qwen3.5-0.8B (hidden 1024, 24 layers), LoRA rank 32
# all-linear, bs 4, bf16, chunked dense-logit-free CE matching the worker's use_fused_kernels path:
#   seq 1024 -> 15.09 GB peak,  seq 2048 -> 28.18 GB peak,  seq 4096 -> OOM on 31.37 GB usable
# Fitting across seq points (the fixed base cancels) gives gc_off = 2.00 + 0.012785 * seq, i.e.
# K_dense = 65.0. Confirmed linear, not s^2: the gc-on slope is identical across both segments
# (2.262 GB per 1024 tokens) and the gc-off extra ratio is 1.981 against 2.000 ideal. Out-of-sample
# check the fit never saw: it predicts 54.4 GB at seq 4096 -> must OOM, and the card OOMed at
# 30.97 GB. The old shared 18.0 predicted 20.76 GB there, i.e. it would have called that run a FIT.
#
# SCOPE of that measurement: fit on ONE dense geometry (1024 hidden x 24 layers) and assumed
# invariant across dense sizes, which is what K being a per-layer-relative-to-hidden factor asserts.
# 130 bytes per (token, hidden-unit, layer) is ~3.8x the fused-Megatron figure, which is the expected
# direction for all-linear LoRA on HF -- every wrapped linear retains its input for the adapter
# backward, on top of HF's unfused per-block intermediates -- but the SIZE-invariance is untested.
# Anyone loosening the >= 120 GB card gate should re-measure at the geometry they are enabling rather
# than trust this extrapolation; at 27B it implies ~410 GB at seq 2048, far outside what was observed.
#
# MOE: 18.0 is UNVALIDATED against a live peak -- it is the pre-existing value, kept deliberately.
# It sits just above the Megatron FlashAttention factor (~34 bytes/(s.b.h) == K 17) so we over-reserve
# and keep GC ON rather than risk an OOM. The dense 65.0 must NOT be applied here: it would flip the
# 35B-A3B (the only model whose active_params_b is truthy, hence the only one that reaches this gate
# today) from GC-off to GC-on at seq >= 2048 and pay the ~+33% recompute tax on every step, on
# evidence gathered from a model whose FFN geometry it does not share. Calibrate from a live H200
# peak before changing it.
_GC_OFF_ACT_K_DENSE = 65.0
_GC_OFF_ACT_K_MOE = 18.0


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
    """Estimated peak VRAM (GB) for a dense-logit-free LoRA SFT step with gradient checkpointing OFF:
    the resident weights + optimizer/base + the no-recompute activations held across ALL ``num_layers``.
    Chunked or fused CE is assumed, so there is no ``[B, T, vocab]`` logits term (the thing that made
    GC-off impossible at a 248k vocab). Unknown architecture dims -> ``inf`` (caller keeps GC on).

    MoE: the activation backbone scales with the model's real ``hidden`` x ``num_layers`` (geometry),
    NOT params_b -- the ~3B-active expert FFN is already folded into ``_GC_OFF_ACT_K_MOE``. ``weights``
    still reserves the FULL ``params_b`` (every expert is resident)."""
    if not (hidden and num_layers and seq_len):
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
    margin_gb: float = 18.0,
    model_info=None,
) -> bool:
    """True when a dense-logit-free LoRA SFT step fits a ``card_vram_gb`` card WITHOUT gradient checkpointing,
    so GC -- a ~+33% recompute tax on every step -- can be turned off for the speed win.

    Conservative by construction: an unknown card / unknown architecture dims, or a peak that doesn't
    clear the ``margin_gb`` headroom, returns False (keep GC ON). The MoE estimate over-reserves
    (``_GC_OFF_ACT_K_MOE`` above the Megatron factor, plus a fixed margin) precisely because the
    allocator sized the card for the GC-*on* peak -- this is the independent check that the larger
    GC-off peak still fits the REAL card (mirrors ``grpo_fits_resident``'s resident-fit guard). The
    dense constant is a live measurement rather than a safety pad, so the ``margin_gb`` headroom is
    what carries the conservatism there."""
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
        model_info=model_info,
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
        # opd sizes prompt plus completion for the resident rollout path.
        _default_len = opd_rollout_seq_len(0, max_tokens, thinking)
    else:
        # sft trims rows to RECIPE.sft.max_seq_len_thinking when thinking is on (sft_max_length).
        # defaulting to the non-thinking cap here sizes activations for half the sequence the worker
        # actually trains on.
        from flash.engine.recipe import RECIPE

        _default_len = int(RECIPE.sft.max_seq_len_thinking if thinking else RECIPE.sft.max_seq_len)
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
        model_info=None,
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
            model_info=model_info,
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
        model_info=None,
    ) -> int:
        """Re-size an OPD requirement with an fp8 KV cache once the run is provably modern-card-only.

        The colocated OPD vLLM rollout engine reserves an fp8 KV cache on cc >= 8.9 hardware
        (engine/worker/opd_train.py sends engine_kwargs.vllm.kv_cache_dtype=fp8), but the estimate
        defaults to a bf16 KV pool. Any OPD run needing
        more VRAM than the biggest non-fp8 validated card (the 80 GB A100) can ONLY land on a modern
        (cc >= 8.9) card, so that bf16 pool is a phantom: it doubles the real KV and wrongly rejects
        full-context / grouped OPD configs on the 35B that actually fit a B200. Halve it — but only
        while the fp8-sized requirement still clears the non-fp8 ceiling, so the discount can never
        pull the run back onto a card that would NOT use fp8 (which would then OOM).

        Skipped entirely for GDN hybrids: both workers refuse fp8 KV for them (opd_train.py,
        rl_train.py), so their cache really is bf16 and the discount would admit a run onto a card
        that cannot hold it."""
        from flash.providers.base import max_non_fp8_kv_vram_gb

        if _declares_linear_attention(model_info, model_id):
            return need
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
            model_info=model_info,
        )
        return fp8_need if fp8_need > ceiling else need

    from flash.catalog import MODELS, vocab_size_for

    info = MODELS.get(model_id)
    model_vocab = vocab_size_for(model_id)
    is_grpo = _algo in ("grpo", "rl")
    is_opd = _algo == "opd"
    is_vllm_rollout = is_grpo or is_opd
    vllm_concurrency = opd_rollout_concurrency(batch_size, group_size) if is_opd else group_size
    # the fused sft loss removes ignored positions before the lm head and projects valid tokens
    # without materializing dense logits. size validated qwen sft jobs without a [batch, seq, vocab]
    # term; models outside that validated set keep the conservative plain-nll estimate and cap.
    sft_fused_ce = None if is_grpo else sft_chunked_nll_enabled(model_id)
    if info is not None:
        params_b = info.params_b
        if model_revision:
            params_b, model_vocab = _validated_revision_geometry(model_id, model_revision, info)
        quant = getattr(info, "quant", "bf16") or "bf16"
        use_vllm = True
        # pinned commits retain validated coarse geometry but use conservative generic architecture sizing.
        sizing_info = None if model_revision else info
        active_b = float(getattr(sizing_info, "active_params_b", 0.0) or 0.0)
        need = _need(
            params_b or 4.0,
            algorithm,
            quant=quant,
            use_vllm=use_vllm,
            vocab=model_vocab,
            active_params_b=active_b,
            model_info=sizing_info,
        )
        if is_opd:
            need = _opd_fp8_adjust(
                need,
                params_b or 4.0,
                quant=quant,
                use_vllm=use_vllm,
                vocab=model_vocab,
                active_params_b=active_b,
                model_info=sizing_info,
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
                        model_info=sizing_info,
                    )
                    # match grpo_fits_resident's 1.15 margin (NOT the looser 1.1 headroom) so the
                    # parse-time reject lands at the SAME resident wall the worker gate enforces.
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
                _rollout_kv_floor_gb(
                    params_b or 4.0,
                    seq_len,
                    vllm_concurrency,
                    active_params_b=active_b,
                    model_info=sizing_info,
                    model_id=model_id,
                    preserve_legacy_floor=is_opd,
                ),
            )
        return need
    # Only curated models are trainable, so `info` is never None here in practice. Kept as a
    # conservative default rather than a raise: this is a sizing helper on the allocation path, and
    # a caller probing a stale id should get the smallest managed card, not an exception.
    return 24


def fetch_hf_params_b(model_id: str, revision: str = "", *, strict: bool = False) -> float | None:
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
    the curated catalog ``params_b``, or the real HF safetensors count for a PINNED revision (the
    pinned commit's true size, which the catalog states only for the default revision). Returns
    None for an uncataloged id, which only a stale caller can produce -- submit rejects those --
    so callers degrade to the size-unknown path rather than raising on the allocation path.
    The single source of truth for "how big is this model" -- run_sft and run_rl both call this
    so they can never drift."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    if info is None:
        return None
    if revision:
        params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
        return params_b
    return info.params_b or None
