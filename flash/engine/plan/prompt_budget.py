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


def _cli_version() -> str:
    from flash import __version__

    return str(__version__)


def rl_prompt_budget(
    spec, *, warm_start_context: int | None = None, derived_by: str = ""
) -> dict | None:
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
        # the descriptor says this itself rather than leaving it to the warning. the warning is
        # silent for an authored context (nothing defaulted, so nothing to warn about), which left
        # the authored path publishing a bare integer that dry-run prints and reports validated --
        # the same overstatement, minus the caveat. engine_len is pre-clamp for every consumer, so
        # the qualifier belongs to the number, not to one of its readers.
        "prompt_budget_is_upper_bound": True,
    }
    source_context = int(warm_start_context or 0)
    if source_context > 0:
        budget["warm_start_context"] = source_context
    # only the cli marks itself. the server's descriptor is authoritative for the run it submitted,
    # so it carries no marker and its warning makes no version caveat.
    if derived_by:
        budget["derived_by"] = derived_by
    return budget


def rl_prompt_budget_warning(budget: object) -> str | None:
    """One user-facing line when a grpo/opd prompt budget defaulted instead of being chosen.

    Silent for an authored context: the user picked that number, and the drop behaviour it implies
    is theirs to own. Silent for sft (no budget descriptor at all).

    The line has to carry the drop semantics, not just the number. A budget reported alone reads as
    a capacity note, and the reason a defaulted one is dangerous is specifically that over-budget
    prompts are deleted from the dataset rather than shortened -- silently, and biased toward the
    longest examples.

    It also has to say "at most". ``engine_len`` is unclamped here (see ``rl_prompt_budget``), so a
    pinned revision whose ``max_position_embeddings`` is below the catalog cap makes the worker's
    real threshold LOWER than the number printed. Stating the budget as exact would tell a user that
    prompts under it survive, when some of them are still dropped.
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
        f"train.max_context_tokens is unset, so {algorithm} derives a prompt budget of at most "
        f"{prompt_budget} tokens from the recipe default (engine {budget.get('engine_len')} minus "
        f"max_completion_tokens {max_completion}), not from the model's context. The worker clamps "
        f"the engine length to the model architecture first, so its budget can be smaller than "
        f"this. Prompts over the budget are DROPPED, not truncated, so the run trains on a "
        f"shorter, biased subset."
    )
    # the local derivation reads THIS package's recipe constants. when the cli and the control
    # plane are on different releases the server's defaults can differ, and nothing in the
    # train-schema compatibility check compares recipe values -- so say which side computed the
    # number rather than presenting it as the submitted run's. the server-derived descriptor
    # carries no source, so this qualifies only the locally-derived one.
    if budget.get("derived_by") == "cli":
        message += (
            f" (Derived locally by this CLI {_cli_version()}; if the control plane is on a "
            f"different release its recipe default may differ.)"
        )
    source_context = budget.get("warm_start_context")
    if isinstance(source_context, int) and not isinstance(source_context, bool):
        # "was configured with", not "trained at". this is the source's AUTHORED value, and the
        # source's own worker clamped it to that model's architecture exactly as this run's will --
        # so claiming it trained there would state as fact a number we did not verify. the point of
        # the sentence is that a context exists and is not inherited, which survives the hedge.
        message += (
            f" The warm-start source was configured with max_context_tokens={source_context}; "
            f"it is NOT inherited."
        )
    return message + " Set train.max_context_tokens explicitly to choose the budget."
