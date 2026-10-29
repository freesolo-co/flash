"""GRPO dataset/config helpers shared by the RL worker.

verl derives its own batching and owns its rollout engine, so what remains here is the dataset
shaping and spec-to-knob translation the flash side still performs before handing off."""

from __future__ import annotations

from flash.engine.worker.runtime.state import WorkerContext


def resolve_grpo_prompts_per_step(requested: int, available_prompts: int) -> int:
    """Cap GRPO prompt batch to the retained dataset size so a step never gets an empty batch."""
    requested = max(1, int(requested))
    available_prompts = int(available_prompts)
    if available_prompts <= 0:
        raise ValueError("GRPO needs at least one retained training prompt")
    return min(requested, available_prompts)


def build_grpo_prompt_dataset(prompts: list[dict]) -> tuple[list[dict], list]:
    """Arrow-safe GRPO dataset rows + parallel example list for reward_fn lookup.

    Dataset.from_list infers one type per field across ALL rows; mixed-type metadata
    (e.g. int vs str in the same field) causes ArrowInvalid at RL startup. Fix: keep only
    trivially typed columns (prompt, integer example_idx, and optional image descriptor strings);
    reward_fn maps the index back.
    """
    examples = [p["example"] for p in prompts]
    rows = []
    for i, prompt in enumerate(prompts):
        row = {"prompt": prompt["prompt"], "example_idx": i}
        if "images" in prompt:
            row["images"] = list(prompt["images"])
        rows.append(row)
    return rows, examples


def grpo_overrides(*, context: WorkerContext) -> dict:
    """GRPO knobs from job spec's [train] table; omits unset fields so recipe defaults apply."""
    if not context.config.spec:
        return {}
    train = context.config.spec.train
    cfg = {
        "group_size": train.group_size,
        "temperature": train.temperature,
        "max_tokens": train.max_completion_tokens,
        "kl_penalty_coef": train.kl_penalty_coef,
        "thinking_length_penalty_coef": train.thinking_length_penalty_coef,
    }
    return {k: v for k, v in cfg.items() if v is not None}


def grpo_mask_truncated_completions(train) -> bool:
    """Resolve the researcher's single-turn GRPO length-truncation policy.

    Prefix rewards can train on length-limited samples without requiring EOS. An explicit
    setting takes precedence; otherwise length-limited responses are masked. Stop strings
    do not change this policy: the worker masks finish_reason="length", not "stop".
    """
    requested = getattr(train, "mask_truncated_completions", None)
    return True if requested is None else requested
