"""Bounded, credential-safe GRPO rollout samples for heartbeat diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from flash.diagnostics import sanitize_diagnostic

_PROMPT_TAIL_CHARS = 500
_COMPLETION_CHARS = 1000
_COMPLETION_TRUNCATION_MARKER = "\n[truncated]"
_DEFAULT_SAMPLE_LIMIT = 3
_MAX_SAMPLE_LIMIT = 4


def _sample_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = str(item.get("role") or "").strip()
                content = str(item.get("content") or "")
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value or "")


def build_rollout_sample(
    prompt: Any,
    completion: Any,
    reward: Any,
    generated_at_step: Any,
) -> dict[str, Any]:
    """Build one bounded rollout sample without retaining source example objects."""
    prompt_text = _sample_text(prompt)
    completion_text = _sample_text(completion)
    completion_was_truncated = len(completion_text) > _COMPLETION_CHARS
    completion_text = completion_text[:_COMPLETION_CHARS]
    if completion_was_truncated:
        completion_text += _COMPLETION_TRUNCATION_MARKER
    try:
        step = int(generated_at_step) if generated_at_step is not None else None
    except (TypeError, ValueError):
        step = None
    return {
        "prompt_tail": sanitize_diagnostic(prompt_text[-_PROMPT_TAIL_CHARS:]),
        "completion": sanitize_diagnostic(completion_text),
        "reward": float(reward),
        "generated_at_step": step,
    }


def select_rollout_samples(
    triples: Iterable[tuple[Any, Any, Any]],
    *,
    generated_at_step: Any = None,
    limit: int = _DEFAULT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    """Select deterministic samples, preferring one completion per distinct prompt."""
    bounded_limit = max(0, min(int(limit), _MAX_SAMPLE_LIMIT))
    if bounded_limit == 0:
        return []

    rows = list(triples)
    distinct: list[tuple[Any, Any, Any]] = []
    repeats: list[tuple[Any, Any, Any]] = []
    seen_prompts: set[str] = set()
    for row in rows:
        prompt_key = _sample_text(row[0])
        if prompt_key in seen_prompts:
            repeats.append(row)
        else:
            seen_prompts.add(prompt_key)
            distinct.append(row)

    selected = (distinct + repeats)[:bounded_limit]
    return [
        build_rollout_sample(prompt, completion, reward, generated_at_step)
        for prompt, completion, reward in selected
    ]
