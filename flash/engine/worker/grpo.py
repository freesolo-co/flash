"""Pure GRPO batch-sizing + no-op-failure helpers for the RL worker path.

These translate an intended prompts/step into a TRL completion-batch config, cap the
per-device micro-batch to fit VRAM, and decide whether a finished GRPO run actually trained.
The only run-scoped state read here is ``THINKING`` (via the worker package at CALL time, so a
test's ``monkeypatch.setattr(worker, "JOB_SPEC", ...)`` + reload reaches ``rl_per_device_comps``)
and ``JOB_SPEC`` (``grpo_overrides``)."""

from __future__ import annotations

from flash.engine.worker._pkg import W as _w


def compute_grpo_batching(prompts_per_step: int, group_size: int, per_device_comps: int) -> dict:
    """Translate an intended ``prompts_per_step`` into a TRL GRPO batch configuration.

    TRL's GRPO batch sizing is denominated in **completions (prompt-completion pairs), not
    prompts**. The number of *unique prompts* optimized per step is

        (per_device_train_batch_size * gradient_accumulation_steps * num_processes)
        / num_generations

    So to actually optimize ``prompts_per_step`` prompts per step, the global *completion*
    batch must equal ``prompts_per_step * group_size``. We keep ``per_device`` small (it,
    not grad-accum, sets peak VRAM) and put the rest in gradient accumulation.

    The bug this fixes: ``grad_accum = prompts_per_step // per_device`` treated
    ``per_device_train_batch_size`` as a *prompt* count, omitting the ``* group_size``
    factor, so a run intended as 64 prompts/step actually optimized only
    ``64 / group_size = 8`` prompts/step (an 8x smaller effective batch).
    """
    import math

    group_size = max(1, int(group_size))
    prompts_per_step = max(1, int(prompts_per_step))
    per_device = max(1, int(per_device_comps))
    target_comps = prompts_per_step * group_size  # total completions / optimizer step
    # Never let the per-device completion micro-batch exceed the target completion batch:
    # a small prompts_per_step would otherwise overshoot it (mirrors run_sft's
    # `min(per_device_bs, effective_batch)`). No-op at the default (prompts_per_step=64).
    per_device = max(1, min(per_device, target_comps))
    grad_accum = max(1, target_comps // per_device)
    # TRL rejects a global completion batch (per_device * grad_accum) that is not
    # divisible by num_generations (= group_size), failing only AFTER the paid worker
    # is provisioned. per_device is the fixed VRAM knob, so round grad_accum UP to the
    # next multiple that makes the batch divisible (grad_accum must be a multiple of
    # group_size // gcd(per_device, group_size)). This only ever raises the effective
    # batch slightly; the common per_device|group_size cases are unchanged.
    accum_step = group_size // math.gcd(per_device, group_size)
    grad_accum = ((grad_accum + accum_step - 1) // accum_step) * accum_step
    generations_per_step = per_device * grad_accum
    unique_prompts_per_step = generations_per_step // group_size
    return {
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "generations_per_step": generations_per_step,
        "unique_prompts_per_step": unique_prompts_per_step,
        # TRL requires the global completion batch be divisible by num_generations.
        "divisible_by_group": (generations_per_step % group_size == 0),
    }


def resolve_grpo_prompts_per_step(requested: int, available_prompts: int) -> int:
    """Cap GRPO's prompt batch to the retained dataset size.

    TRL's GRPO dataloader can yield zero batches when the configured prompt batch is larger
    than the dataset that remains after prompt-budget filtering. That surfaces late as
    "There seems not to be a single sample in your epoch_iterator" and then our no-reward guard
    reports the wrong cause. Small smoke envs should still train; use every retained prompt per
    step instead of asking TRL for an impossible larger batch.
    """
    requested = max(1, int(requested))
    available_prompts = int(available_prompts)
    if available_prompts <= 0:
        raise ValueError("GRPO needs at least one retained training prompt")
    return min(requested, available_prompts)


def rl_per_device_comps(
    completion_len: int = 0,
    vocab: int = 248_320,
    *,
    use_vllm: bool = True,
    params_b: float | None = None,
) -> int:
    """Per-device *completion* micro-batch for GRPO (TRL counts completions, not prompts).

    This, not grad-accum, sets peak trainer VRAM: the logprob pass materializes fp32 logits
    of shape [per_device, completion_len, vocab]. At Qwen3.5's ~248k vocab a long completion is
    enormous (measured: per_device 8 x 4096 tok x 248k x 4 B = ~30 GiB single alloc -> OOMs
    a small card). So we MEMORY-CAP per_device to a logits budget (6 GB) for the
    given completion length, then push the difference into grad-accum
    (compute_grpo_batching) so the effective batch is unchanged. This keeps long-completion
    GRPO on a cheaper GPU.

    The logits budget is NOT the whole story: the per-device forward also holds the model's
    attention/activation memory (the Qwen3.5 GDN/FLA kernels peak per micro-batch even with
    grad checkpointing), which the logits term can't see. Under colocated vLLM (the rollout
    engine + its card-sized KV pool + a 2nd weight copy share the GPU) that activation peak is
    what OOMs a small card -- and Liger, which fuses away the logits, does NOT touch it.
    MEASURED: Qwen3.5-2B (width ~1.41) group8 seq2048 OOMs a 32 GB card at per_device=8 but
    TRAINS at 4. So for colocate, additionally cap per_device to the live card's VRAM scaled
    by model width (~sqrt(params)): ~vram_gb/8 at 2B-width, tightened for wider models (4B/9B).
    """
    # Default prompts/step; the auto-caps below (logits budget + colocate VRAM/width) handle OOM.
    base = 2 if _w.THINKING else 8
    if completion_len > 0:
        budget = 6.0 * 1e9
        cap = max(1, int(budget / (max(1, completion_len) * vocab * 4)))
        base = min(base, cap)
    if use_vllm:
        try:
            import torch

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                width = (max(float(params_b), 0.1) ** 0.5) if params_b else 1.41
                act_cap = max(1, int(vram_gb / (7.5 * (width / 1.41))))
                base = min(base, act_cap)
        except Exception as e:
            print("rl_per_device_comps colocate cap probe failed (keeping logits cap):", e)
    return max(1, base)


def grpo_overrides() -> dict:
    """The GRPO recipe knobs, read off the job spec's ``[train]`` table (``TrainSpec``).
    A field left unset (None) is omitted here so the recipe default applies downstream.

    Knobs: group_size, temperature, max_tokens (completion budget), kl_penalty_coef (the KL
    beta), advantage_clip (centered-advantage clip), and thinking_length_penalty_coef
    (a per-<think>-token reward deduction). These live in ``[train]`` — NOT in
    ``[environment.params]``, which is forwarded verbatim to the Freesolo env loader."""
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


def _grpo_resume_already_complete(resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when this worker resumed a checkpoint that already reached the target step count.

    Such a resume legitimately performs ZERO new optimizer steps (so the fresh hb_cb has an empty
    reward_history) yet the policy IS fully trained — it must NOT be flagged as a no-op failure.
    """
    return bool(resume_ckpt) and target_steps > 0 and steps_run >= target_steps


def _grpo_is_no_op_failure(reward_history, resume_ckpt, target_steps: int, steps_run: int) -> bool:
    """True when a GRPO run trained NOTHING and must fail loudly instead of reporting as done.

    An empty ``reward_history`` means the reward callback never fired — the rollout scored nothing
    (e.g. vLLM silently returning no completions), so no real training happened. The sole exception
    is a resume that already reached the target steps (see ``_grpo_resume_already_complete``): that
    has an empty fresh history but a fully-trained policy, so it is NOT a failure.
    """
    if reward_history:
        return False
    return not _grpo_resume_already_complete(resume_ckpt, target_steps, steps_run)
