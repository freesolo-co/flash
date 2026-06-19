"""Reward-function latency for GRPO.

Every GRPO step scores all ``prompts_per_step x group_size`` completions through the
run's verifiers environment. The per-completion cost is a property of the environment,
not the model -- an exact-match grader is ~free, while a sandboxed code-exec or an
LLM-judge reward can take seconds and *dominate* the step. The cost model infers a
per-completion latency from the environment slug (overridable per run).

These tiers are coarse priors; ``analytical`` calibrates the active value against the
real measured runs (see cost_estimator_results/real_runs/).
"""

from __future__ import annotations

# Seconds to grade one completion, by reward "weight".
REWARD_TIERS: dict[str, float] = {
    "trivial": 0.01,  # exact-match / regex / numeric check (gsm8k, math, bench)
    "light": 0.15,  # parsing + multi-field scoring, classification
    "medium": 0.6,  # light tool/eval, multi-step verifier
    "heavy": 3.0,  # sandboxed code execution, LLM-as-judge, web/tool rollout
}

# Environment-slug keyword -> tier (first match wins; checked in order).
_ENV_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("code", "swe", "exec", "contest", "leetcode"), "heavy"),
    (("judge", "chat", "ticket", "support", "rubric", "dialog"), "heavy"),
    (("search", "browse", "web", "tool", "agent", "linkd"), "medium"),
    (("sentiment", "classif", "extract", "ner", "label"), "light"),
    (("math", "gsm", "arith", "hendrycks", "aime", "bench", "mmlu"), "trivial"),
)

DEFAULT_REWARD_SECONDS = 0.3  # unknown env: assume a light-to-medium grader


def reward_seconds_per_completion(environment: str | None, override: float | None = None) -> float:
    """Per-completion reward latency (s): explicit override, else inferred from the env."""
    if override is not None:
        return max(0.0, override)
    if not environment:
        return DEFAULT_REWARD_SECONDS
    slug = environment.lower()
    for keywords, tier in _ENV_KEYWORDS:
        if any(k in slug for k in keywords):
            return REWARD_TIERS[tier]
    return DEFAULT_REWARD_SECONDS
