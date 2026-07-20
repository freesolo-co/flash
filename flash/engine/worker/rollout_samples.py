"""Credential-safe rollout samples (full, untruncated) for heartbeat diagnostics.

A "sample" is one on-policy generation surfaced so a user can SEE what the model produced on a logged
training step (GRPO carries the scored ``reward``; OPD carries the distillation ``loss``). Completions
and prompts are shown in full — the only transformation is credential redaction and terminal
control-character neutralization, never length truncation. Sizes are already bounded by the training
config (``max_completion_tokens`` / the prompt budget), so full text stays sane on the wire.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

from flash.diagnostics import neutralize_control_chars, sanitize_diagnostic

# Exactly three samples per logged step, always. Not configurable.
_SAMPLE_LIMIT = 3
# The scalar a sample carries: GRPO reward, OPD distillation loss.
_SAMPLE_SCALARS = ("reward", "loss")


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


def sanitize_rollout_text(text: str) -> str:
    """Redact credentials and neutralize terminal control chars, preserving the full text.

    ``limit=sys.maxsize`` disables ``sanitize_diagnostic``'s length bound: samples are shown untruncated
    (redaction and control-char escaping are the only transformations)."""
    redacted = sanitize_diagnostic(text, limit=sys.maxsize)
    return neutralize_control_chars(redacted)


def build_rollout_sample(
    prompt: Any,
    completion: Any,
    *,
    reward: Any = None,
    loss: Any = None,
    generated_at_step: Any = None,
) -> dict[str, Any]:
    """Build one full, credential-safe rollout sample without retaining source example objects.

    Exactly one of ``reward`` (GRPO) or ``loss`` (OPD) is supplied and stored under its own key."""
    prompt_text = _sample_text(prompt)
    completion_text = _sample_text(completion)
    try:
        step = int(generated_at_step) if generated_at_step is not None else None
    except (TypeError, ValueError):
        step = None
    record: dict[str, Any] = {
        "prompt_tail": sanitize_rollout_text(prompt_text),
        "completion": sanitize_rollout_text(completion_text),
        "generated_at_step": step,
    }
    if reward is not None:
        record["reward"] = float(reward)
    if loss is not None:
        record["loss"] = float(loss)
    return record


def select_rollout_samples(
    triples: Iterable[tuple[Any, Any, Any]],
    *,
    generated_at_step: Any = None,
    scalar: str = "reward",
) -> list[dict[str, Any]]:
    """Select up to three deterministic samples, preferring one completion per distinct prompt.

    Each triple is ``(prompt, completion, value)``; ``value`` is stored under ``scalar`` — ``"reward"``
    for GRPO, ``"loss"`` for OPD."""
    if scalar not in _SAMPLE_SCALARS:
        raise ValueError(f"rollout sample scalar must be one of {_SAMPLE_SCALARS}, got {scalar!r}")

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

    selected = (distinct + repeats)[:_SAMPLE_LIMIT]
    return [
        build_rollout_sample(
            prompt,
            completion,
            generated_at_step=generated_at_step,
            **{scalar: value},
        )
        for prompt, completion, value in selected
    ]
