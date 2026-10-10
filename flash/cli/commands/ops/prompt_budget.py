"""prompt-budget warnings and warm-start context lookup for training commands."""

from __future__ import annotations

import sys
from typing import TypeGuard, cast

from flash import __version__
from flash.cli.ui import render
from flash.engine.plan.prompt_budget import PromptBudget, rl_prompt_budget


def _positive_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _budget_from_status(status: object) -> PromptBudget | None:
    budget = status.get("prompt_budget") if isinstance(status, dict) else None
    if not isinstance(budget, dict):
        return None
    if budget.get("algorithm") not in {"grpo", "opd", "rl"}:
        return None
    if budget.get("context_source") not in {"authored", "recipe_default"}:
        return None
    if budget.get("prompt_budget_is_upper_bound") is not True:
        return None
    engine_len = budget.get("engine_len")
    max_completion = budget.get("max_completion")
    prompt_budget = budget.get("prompt_budget")
    if not _positive_int(engine_len):
        return None
    if not _positive_int(max_completion):
        return None
    if not _positive_int(prompt_budget):
        return None
    if prompt_budget != engine_len - max_completion:
        return None
    source_context = budget.get("warm_start_context")
    if source_context is not None and not _positive_int(source_context):
        return None
    return cast("PromptBudget", budget)


def prompt_budget_validation_suffix(status: object) -> str:
    """qualify a dry-run summary only when the server returned a budget."""
    return ", prompt budget (upper bound)" if _budget_from_status(status) is not None else ""


def warmstart_source_context(client, spec) -> int | None:
    """read a warm-start source's authored context through the cli's authenticated client."""
    from flash.core.catalog import samples_on_policy
    from flash.schema import parse_checkpoint_ref

    ref = getattr(spec.train, "init_from_adapter", "")
    if not ref or not samples_on_policy(spec.algorithm):
        return None
    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        return None
    try:
        source = client.get_run(parsed[0])
    # reporting is best-effort and must never block a valid submission
    except Exception:
        return None
    train = (source or {}).get("spec", {}).get("train", {}) if isinstance(source, dict) else {}
    context = train.get("max_context_tokens") if isinstance(train, dict) else None
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        return None
    return context


def _warmstart_context_sentence(context: object) -> str | None:
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        return None
    return (
        f"The warm-start source was configured with max_context_tokens={context}; "
        "it is NOT inherited."
    )


def prompt_budget_warning(
    budget: object,
    *,
    derived_locally: bool = False,
    include_warm_start_context: bool = True,
) -> str | None:
    """render the defaulted-budget warning for one prompt-budget descriptor."""
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
        "the engine length to the model architecture first, so its budget can be smaller than "
        "this. Prompts over the budget are DROPPED, not truncated, so the run trains on a "
        "shorter, biased subset."
    )
    if derived_locally:
        message += (
            f" (Derived locally by this CLI {__version__}; if the control plane is on a "
            "different release its recipe default may differ.)"
        )
    if include_warm_start_context:
        source = _warmstart_context_sentence(budget.get("warm_start_context"))
        if source:
            message += f" {source}"
    return message + " Set train.max_context_tokens explicitly to choose the budget."


def _print_warning(message: str | None) -> None:
    if message:
        print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def print_status_prompt_budget_warning(status: object) -> None:
    """print the server-derived warning carried by a submit or dry-run response."""
    _print_warning(prompt_budget_warning(_budget_from_status(status)))


def warn_before_paid_submit(client, spec) -> PromptBudget | None:
    """print the locally derived warning before create_run can start provisioning."""
    budget = rl_prompt_budget(spec, warm_start_context=warmstart_source_context(client, spec))
    _print_warning(prompt_budget_warning(budget, derived_locally=True))
    return budget


def print_warmstart_context_supplement(local_budget: PromptBudget | None, status: object) -> None:
    """print only context the server resolved after the generic pre-submit warning."""
    served_budget = _budget_from_status(status)
    if served_budget is None or (
        local_budget is not None and local_budget.get("warm_start_context")
    ):
        return
    _print_warning(_warmstart_context_sentence(served_budget.get("warm_start_context")))


def warn_cost_prompt_budget(spec) -> None:
    """print the local prompt-budget warning for the offline ``--cost`` path."""
    budget = rl_prompt_budget(spec)
    _print_warning(prompt_budget_warning(budget, derived_locally=True))
