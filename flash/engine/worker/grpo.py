"""GRPO batch-sizing and no-op-failure helpers for the RL worker."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w


def compute_grpo_batching(prompts_per_step: int, group_size: int, per_device_comps: int) -> dict:
    """Translate prompts_per_step into a TRL GRPO batch config (TRL counts completions, not prompts).

    per_device sets peak VRAM; grad_accum carries the rest so unique prompts == prompts_per_step.
    per_device is shrunk to the largest divisor of target_comps so per_device * grad_accum is exact.
    """
    group_size = max(1, int(group_size))
    prompts_per_step = max(1, int(prompts_per_step))
    per_device = max(1, int(per_device_comps))
    target_comps = prompts_per_step * group_size
    per_device = max(1, min(per_device, target_comps))
    # Shrink to largest divisor of target_comps so grad_accum is exact (no over/under-shoot).
    while target_comps % per_device != 0:
        per_device -= 1
    grad_accum = max(1, target_comps // per_device)
    generations_per_step = per_device * grad_accum
    unique_prompts_per_step = generations_per_step // group_size
    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "generations_per_step": generations_per_step,
        "unique_prompts_per_step": unique_prompts_per_step,
        "divisible_by_group": (generations_per_step % group_size == 0),
    }


def resolve_grpo_prompts_per_step(requested: int, available_prompts: int) -> int:
    """Cap GRPO prompt batch to the retained dataset size to avoid zero-batch TRL errors."""
    requested = max(1, int(requested))
    available_prompts = int(available_prompts)
    if available_prompts <= 0:
        raise ValueError("GRPO needs at least one retained training prompt")
    return min(requested, available_prompts)


def build_grpo_prompt_dataset(prompts: list[dict]) -> tuple[list[dict], list]:
    """Arrow-safe GRPO dataset rows + parallel example list for reward_fn lookup.

    Dataset.from_list infers one type per field across ALL rows; mixed-type metadata
    (e.g. int vs str in the same field) causes ArrowInvalid at RL startup. Fix: keep only
    trivially-typed columns (prompt + integer example_idx); reward_fn maps the index back.
    """
    examples = [p["example"] for p in prompts]
    rows = []
    for i, p in enumerate(prompts):
        row = {"prompt": p["prompt"], "example_idx": i}
        if "image" in p:
            row["image"] = p["image"]
        if "images" in p:
            row["images"] = p["images"]
        rows.append(row)
    return rows, examples


# Measured plateau ceiling: throughput peaks at pd 8-16 and regresses at pd 32 (-20%).
_RL_PER_DEVICE_MAX = 16
# Reference seq length for activation/VRAM calibration.
_RL_ACT_SEQ_REF = 2048.0
# Calibrated: Qwen3.5-2B group8 OOMs 32GB at pd=8, fits at pd=4 (34.36 decimal GB / 8.053 = 4).
_RL_ACT_DIVISOR = 8.053
# Floor so Qwen3.5-0.8B on 24GB at seq<=1024 lands at pd=8 (measured safe, +12.6% over pd=4).
_RL_ACT_SEQ_SCALE_FLOOR = 0.63
# Never grow above the reference seq (long-seq growth unvalidated; regression is in tokens-in-flight).
_RL_ACT_SEQ_SCALE_CEIL = 1.0


def rl_per_device_comps(
    completion_len: int = 0,
    vocab: int = 248_320,
    *,
    use_vllm: bool = True,
    params_b: float | None = None,
    active_params_b: float | None = None,
    seq_len: int = 0,
    fused_logits: bool = False,
) -> int:
    """Per-device completion micro-batch for GRPO (TRL counts completions, not prompts).

    Grows into VRAM headroom on short-seq runs (measured +12.6% on 0.8B/24GB at seq=1024).
    Two caps: logits budget (~6GB hard ceiling on fp32 [per_device, completion, vocab]) and
    activation/VRAM cap (calibrated from live card + model width + seq scale).
    THINKING runs excluded from growth — long completions bypass the seq gate.
    Falls back to default (8, or 2 with thinking) with no live card.
    """
    default = 2 if _w.THINKING else 8

    _ovr = os.environ.get("FLASH_RL_PER_DEVICE_COMPS", "").strip()
    if _ovr:
        try:
            forced = int(_ovr)
            if forced >= 1:
                print(f"rl_per_device_comps: FLASH_RL_PER_DEVICE_COMPS override -> per_device={forced}")
                return forced
            print(f"rl_per_device_comps: ignoring non-positive FLASH_RL_PER_DEVICE_COMPS={_ovr!r}")
        except ValueError:
            print(f"rl_per_device_comps: ignoring non-integer FLASH_RL_PER_DEVICE_COMPS={_ovr!r}")

    logits_cap = _RL_PER_DEVICE_MAX
    if completion_len > 0 and not fused_logits:
        logits_cap = max(1, int(6.0e9 / (max(1, completion_len) * vocab * 4)))

    short_seq = (seq_len or _RL_ACT_SEQ_REF) < _RL_ACT_SEQ_REF

    vram_cap = None
    if use_vllm:
        try:
            import torch

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                _eff_b = float(active_params_b) if active_params_b else params_b
                width = (max(float(_eff_b), 0.1) ** 0.5) if _eff_b else 1.41
                seq_scale = min(
                    _RL_ACT_SEQ_SCALE_CEIL,
                    max(_RL_ACT_SEQ_SCALE_FLOOR, (seq_len or _RL_ACT_SEQ_REF) / _RL_ACT_SEQ_REF),
                )
                vram_cap = max(
                    1, int(vram_gb / (_RL_ACT_DIVISOR * (width / 1.41) * seq_scale))
                )
        except Exception as e:
            print("rl_per_device_comps colocate cap probe failed (keeping logits cap):", e)

    if vram_cap is None:
        return max(1, min(default, logits_cap))
    # THINKING excluded: long completions bypass the seq gate, so growth could silently OOM.
    ceiling = _RL_PER_DEVICE_MAX if (short_seq and not _w.THINKING) else default
    return max(1, min(ceiling, logits_cap, vram_cap))


def grpo_overrides() -> dict:
    """GRPO knobs from job spec's [train] table; omits unset fields so recipe defaults apply."""
    if not _w.JOB_SPEC:
        return {}
    train = _w.JOB_SPEC.train
    cfg = {
        "group_size": train.group_size,
        "temperature": train.temperature,
        "max_tokens": train.max_tokens,
        "kl_penalty_coef": train.kl_penalty_coef,
        "advantage_clip": train.advantage_clip,
        "thinking_length_penalty_coef": train.thinking_length_penalty_coef,
    }
    return {k: v for k, v in cfg.items() if v is not None}


def card_vram_gb() -> float | None:
    """Total VRAM in decimal GB (/1e9), or None if no live card.

    Uses /1e9 not GiB to match all other VRAM logic; binary GiB would under-report ~7%.
    Returns None (not 0.0) so callers keep their own no-card fallback.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return None


def resolve_grpo_sleep_mode() -> tuple[bool, int, float, bool]:
    """Resolve vLLM sleep-mode for colocated GRPO from JOB_SPEC + live card.

    Single source shared by run_rl and finalize_alloc_conf_for_sleep so they never drift.
    Returns (sleep_mode, ctx, card_gb, fp8_kv).
    """
    from flash.engine.recipe import RECIPE
    from flash.engine.worker.perf import grpo_sleep_mode

    spec = _w.JOB_SPEC
    train = spec.train if spec else None
    model_id = spec.model if spec else ""
    ctx = int(train.max_length if train and train.max_length else 0)
    gcfg = _w.grpo_overrides()
    group_size = int(gcfg.get("group_size") or RECIPE.rl.group_size)
    lora_rank = int(train.lora_rank) if train and train.lora_rank else 32
    card_gb = card_vram_gb() or 0.0
    fp8_kv = False
    try:
        import torch

        fp8_kv = bool(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
    except Exception:
        pass
    sleep_mode = grpo_sleep_mode(
        model_id,
        max_length=ctx,
        group_size=group_size,
        max_tokens=gcfg.get("max_tokens"),
        lora_rank=lora_rank,
        thinking=_w.THINKING,
        card_vram_gb=card_gb,
        fp8_kv=fp8_kv,
    )
    return sleep_mode, ctx, card_gb, fp8_kv


def _grpo_resume_already_complete(resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when this worker resumed a checkpoint already at the target step count.

    Such a resume does zero new steps (empty reward_history) yet IS fully trained — must not fail.
    """
    return bool(resume_ckpt) and target_steps > 0 and steps_run >= target_steps


def _grpo_is_no_op_failure(reward_history, resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when a GRPO run trained nothing (empty reward_history) and must fail loudly.

    Exception: a complete resume has empty fresh history but a fully-trained policy.
    """
    if reward_history:
        return False
    return not _grpo_resume_already_complete(resume_ckpt, target_steps, steps_run)


def grpo_mask_truncated_completions(train) -> bool:
    """Whether GRPO should drop truncated (non-EOS) completions from the loss.

    Default True — TRL's footgun defaults to False. GATED OFF when stop_sequences is set:
    stop-string rollouts don't end on EOS, so masking would wrongly drop every completion.
    """
    return not (train and train.stop_sequences)
