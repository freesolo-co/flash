"""derived grpo/opd prompt-budget data reported before paid training."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from flash.engine.plan.vram import (
    grpo_completion_len,
    grpo_rollout_seq_len,
    opd_completion_len,
    opd_rollout_seq_len,
)


class PromptBudget(TypedDict):
    """serialized prompt-budget descriptor carried by prepared and persisted runs."""

    algorithm: Literal["grpo", "opd", "rl"]
    engine_len: int
    max_completion: int
    prompt_budget: int
    context_source: Literal["authored", "recipe_default"]
    prompt_budget_is_upper_bound: Literal[True]
    warm_start_context: NotRequired[int]


def rl_prompt_budget(spec, *, warm_start_context: int | None = None) -> PromptBudget | None:
    """derive the pre-clamp prompt budget used by a grpo/opd worker.

    the worker clamps ``engine_len`` to the model architecture before subtracting the completion
    allowance, so this control-plane value is an upper bound. sft returns ``None`` because it
    truncates over-long rows and reports that behavior through its workload profile instead.
    """
    from typing import cast

    from flash.core.catalog import samples_on_policy

    raw_algorithm = getattr(spec, "algorithm", "")
    if not samples_on_policy(raw_algorithm):
        return None
    algorithm = cast("Literal['grpo', 'opd', 'rl']", raw_algorithm)
    train = spec.train
    authored = int(getattr(train, "max_context_tokens", 0) or 0)
    max_tokens = getattr(train, "max_completion_tokens", None)
    thinking = bool(getattr(spec, "thinking", False))
    if algorithm == "opd":
        max_completion = opd_completion_len(max_tokens, thinking)
        engine_len = opd_rollout_seq_len(authored, max_tokens, thinking)
    else:
        max_completion = grpo_completion_len(max_tokens, thinking)
        engine_len = grpo_rollout_seq_len(authored, max_tokens, thinking)
    budget: PromptBudget = {
        "algorithm": algorithm,
        "engine_len": int(engine_len),
        "max_completion": int(max_completion),
        "prompt_budget": int(engine_len) - int(max_completion),
        "context_source": "authored" if authored else "recipe_default",
        "prompt_budget_is_upper_bound": True,
    }
    source_context = int(warm_start_context or 0)
    if source_context > 0:
        budget["warm_start_context"] = source_context
    return budget
