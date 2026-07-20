"""Bounded, credential-safe GRPO rollout samples for heartbeat diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

from flash.diagnostics import neutralize_control_chars, sanitize_diagnostic

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


def _sanitize_rollout_text(text: str) -> str:
    redacted = sanitize_diagnostic(text, limit=sys.maxsize)
    return neutralize_control_chars(redacted)


def bound_rollout_prompt_tail(text: str) -> str:
    """Redact and neutralize a full prompt before retaining its bounded tail."""
    return _sanitize_rollout_text(text)[-_PROMPT_TAIL_CHARS:]


def bound_rollout_completion(text: str) -> str:
    """Redact and neutralize a full completion before retaining its bounded prefix."""
    safe_text = _sanitize_rollout_text(text)
    if len(safe_text) <= _COMPLETION_CHARS:
        return safe_text
    return safe_text[:_COMPLETION_CHARS] + _COMPLETION_TRUNCATION_MARKER


def build_rollout_sample(
    prompt: Any,
    completion: Any,
    reward: Any,
    generated_at_step: Any,
) -> dict[str, Any]:
    """Build one bounded rollout sample without retaining source example objects."""
    prompt_text = _sample_text(prompt)
    completion_text = _sample_text(completion)
    try:
        step = int(generated_at_step) if generated_at_step is not None else None
    except (TypeError, ValueError):
        step = None
    return {
        "prompt_tail": bound_rollout_prompt_tail(prompt_text),
        "completion": bound_rollout_completion(completion_text),
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
