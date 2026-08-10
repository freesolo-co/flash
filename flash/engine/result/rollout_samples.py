"""Credential-safe rollout samples (full, untruncated) for heartbeat diagnostics.

A "sample" is one on-policy generation surfaced so a user can SEE what the model produced on a logged
training step (GRPO carries the scored ``reward``; OPD carries the distillation ``loss``). Text content
is shown in full with credential redaction and terminal control-character neutralization; non-text
multimodal parts are replaced with placeholders. Sizes are already bounded by the training config
(``max_completion_tokens`` / the prompt budget), so full text stays sane on the wire.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from typing import Any

from flash._internal.diagnostics import neutralize_control_chars, sanitize_diagnostic

# Exactly three samples per logged step, always. Not configurable.
_SAMPLE_LIMIT = 3
# The scalar a sample carries: GRPO reward, OPD distillation loss.
_SAMPLE_SCALARS = ("reward", "loss")


def sampled_completion_scalar(sample: dict) -> tuple[str, float] | None:
    """Read back the (key, finite value) of a sample's scalar, or ``None`` when it carries none.

    The inverse of what ``select_rollout_samples`` writes, and keyed off the same
    ``_SAMPLE_SCALARS``, so a reader can never go looking for a key the writer would not emit."""
    for key in _SAMPLE_SCALARS:
        if key not in sample:
            continue
        try:
            value = float(sample.get(key))
        except (TypeError, ValueError):
            return None
        return (key, value) if math.isfinite(value) else None
    return None


def _content_part_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return "<non-text>"

    part_type = str(value.get("type") or "").strip().lower()
    text = value.get("text")
    if part_type in {"text", "input_text", "output_text"} or (
        not part_type and isinstance(text, str)
    ):
        return text if isinstance(text, str) else ""
    if "image" in part_type:
        return "<image>"
    if "audio" in part_type:
        return "<audio>"
    if "video" in part_type:
        return "<video>"
    return "<non-text>"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_part_text(part) for part in value)
    if isinstance(value, dict):
        return _content_part_text(value)
    return str(value or "")


def _message_text(value: dict[str, Any]) -> str:
    role = str(value.get("role") or "").strip()
    content = _content_text(value.get("content"))
    return f"{role}: {content}" if role else content


def sample_completion_text(value: Any) -> str:
    """Flatten one prompt or completion to display text, whatever shape the backend produced.

    A completion is plain text (single-turn), one message dict, or a whole message list (a
    multi-turn transcript); non-text multimodal parts become placeholders. Public because the verl
    path logs a per-step preview of the same value it later publishes as a sample, and the two must
    agree on what the text of a rollout is."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _message_text(value) if "content" in value else _content_part_text(value)
    if isinstance(value, list):
        return "\n".join(
            _message_text(item)
            if isinstance(item, dict) and "content" in item
            else _content_part_text(item)
            for item in value
        )
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
    prompt_text = sample_completion_text(prompt)
    completion_text = sample_completion_text(completion)
    try:
        step = int(generated_at_step) if generated_at_step is not None else None
    except (TypeError, ValueError):
        step = None
    record: dict[str, Any] = {
        "prompt_tail": sanitize_rollout_text(prompt_text),
        "completion": sanitize_rollout_text(completion_text),
        "generated_at_step": step,
    }
    # this is the one place the scalar is coerced, so it is the one place a non-finite one can be
    # created. omit rather than publish it: json.dumps writes bare NaN/Infinity, which is not json,
    # and a strict reader rejects the whole heartbeat over it -- losing the step's other fields too.
    # a sample without its scalar is then skipped by the reader, so a diverged step publishes fewer
    # samples rather than a payload that will not parse.
    for key, value in (("reward", reward), ("loss", loss)):
        if value is None:
            continue
        scalar = float(value)
        if math.isfinite(scalar):
            record[key] = scalar
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
        prompt_key = sample_completion_text(row[0])
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
