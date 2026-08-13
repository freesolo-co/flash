"""What prompt budget a grpo/opd run will train against, and when that number is a surprise.

Split from ``vram.py`` because it is not sizing: nothing here decides hardware. It reports the
budget the rl workers derive so ``--dry-run`` and the run digest can show it before a gpu is paid
for, instead of the worker log being the first place it appears.
"""

from __future__ import annotations

from flash.engine.plan.vram import (
    grpo_completion_len,
    grpo_rollout_seq_len,
    opd_completion_len,
    opd_rollout_seq_len,
)


def rl_prompt_budget(spec, *, warm_start_context: int | None = None) -> dict | None:
    """The prompt budget a grpo/opd run will train against, derived exactly as its worker derives it.

    The workers do not truncate an over-budget prompt and do not raise on one: they DROP it
    (``train/rl/inputs.py``, ``opd_train_runner``). The filter selects on length, so what it removes
    is the long tail, and the surviving set is biased short. That makes the budget worth reporting
    before a gpu is paid for, not just in the worker log after allocation.

    ``context_source`` is the point. An omitted ``max_context_tokens`` does not mean "the model's
    context" -- it falls back to ``RECIPE.rl/opd.max_prompt_len + completion``, a recipe constant
    unrelated to the model or to the warm-start source, so a run continuing an 8192-token sft can
    silently train at 2048. ``authored`` vs ``recipe_default`` is what distinguishes a budget the
    user chose from one that defaulted out from under them.

    ``engine_len`` is the CONFIGURED length, before the workers clamp it to the model architecture
    (``clamp_engine_len``). That probe reads the hf config and cannot run on the control plane, and
    clamping only ever shrinks the value -- so the reported budget is an upper bound on the worker's,
    never an under-report that would make a too-small budget look adequate.

    None for sft, which truncates rather than dropping and reports its own
    ``truncated_examples``/``untruncated_max_length`` through the workload profile.
    """
    from flash.core.catalog import samples_on_policy

    algorithm = getattr(spec, "algorithm", "")
    if not samples_on_policy(algorithm):
        return None
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
    budget: dict[str, object] = {
        "algorithm": algorithm,
        "engine_len": int(engine_len),
        "max_completion": int(max_completion),
        "prompt_budget": int(engine_len) - int(max_completion),
        "context_source": "authored" if authored else "recipe_default",
    }
    source_context = int(warm_start_context or 0)
    if source_context > 0:
        budget["warm_start_context"] = source_context
    return budget


def rl_prompt_budget_warning(budget: object) -> str | None:
    """One user-facing line when a grpo/opd prompt budget defaulted instead of being chosen.

    Silent for an authored context: the user picked that number, and the drop behaviour it implies
    is theirs to own. Silent for sft (no budget descriptor at all).

    The line has to carry the drop semantics, not just the number. A budget reported alone reads as
    a capacity note, and the reason a defaulted one is dangerous is specifically that over-budget
    prompts are deleted from the dataset rather than shortened -- silently, and biased toward the
    longest examples.
    """
    if not isinstance(budget, dict) or budget.get("context_source") != "recipe_default":
        return None
    try:
        prompt_budget = int(budget["prompt_budget"])
        max_completion = int(budget["max_completion"])
    except (KeyError, TypeError, ValueError):
        return None
    algorithm = str(budget.get("algorithm") or "").upper() or "RL"
    message = (
        f"train.max_context_tokens is unset, so {algorithm} derives a {prompt_budget}-token prompt "
        f"budget from the recipe default (engine {budget.get('engine_len')} minus "
        f"max_completion_tokens {max_completion}), not from the model's context. Prompts longer "
        f"than {prompt_budget} tokens are DROPPED, not truncated, so the run trains on a shorter, "
        f"biased subset."
    )
    source_context = budget.get("warm_start_context")
    if isinstance(source_context, int) and not isinstance(source_context, bool):
        message += (
            f" The warm-start source trained at max_context_tokens={source_context}; it is NOT "
            f"inherited."
        )
    return message + " Set train.max_context_tokens explicitly to choose the budget."
