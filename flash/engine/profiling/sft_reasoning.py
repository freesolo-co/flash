"""Thinking-SFT validation and per-row reasoning survival accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from flash.engine.profiling.reasoning_render import (
    marked_reasoning_end,
    reasoned_assistant_turns,
    reasoning_marker_prefix,
    reasoning_markers,
    strip_reasoning_markers,
    with_marked_reasoning,
)


def _require_reasoning_when_thinking(prompt_rows) -> None:
    """Every supervised assistant turn needs reasoning, including tool continuations.

    This source check catches empty targets before loading a tokenizer. Rendering must also
    preserve those blocks; authored reasoning alone does not establish the training contract.
    """
    for row_index, (_example, _prompt, completion, _coerced) in enumerate(prompt_rows):
        for turn_index, message in enumerate(completion):
            if message.get("role") == "assistant" and not reasoned_assistant_turns([message]):
                raise ValueError(
                    f"thinking = true, but selected SFT row {row_index + 1}, completion message "
                    f"{turn_index + 1} has no reasoning in its assistant turn. Every assistant "
                    "target needs reasoning (reasoning_content, or a <think>...</think> span "
                    "before the answer), including answers after tool calls; otherwise set "
                    "thinking = false."
                )


def _encoded_length(tokenizer, text: str) -> int:
    """Token count of ``text``, for either the batched or unbatched ``input_ids`` shape.

    A tokenizer handed a single string may answer with one id list or with a batch holding one
    row. Reading ``len()`` off the batch would count ROWS -- always 1 -- and silently declare every
    span to be within the cap.
    """
    ids = tokenizer(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


@dataclass(frozen=True)
class _RowReasoning:
    """One row's reasoning accounting, with the two causes of loss kept apart.

    They have opposite remedies: reasoning the TEMPLATE dropped is fixed by splitting a K-turn
    transcript into K single-turn rows, while reasoning the CAP cut is fixed by raising
    ``max_context_tokens``. Folding both into one survivor count makes the warning prescribe
    restructuring for a dataset whose only problem is that the row is too long.
    """

    authored_turns: int
    rendered_spans: int
    truncated_spans: int


def _row_reasoning(
    prompt_messages: list[dict],
    completion_messages: list[dict],
    *,
    render: Callable[[list[dict]], str],
    tokenizer,
    max_length: int,
) -> _RowReasoning:
    """One row's authored reasoning, and how much of it reaches the loss.

    The row is rendered once with each reasoning-authoring turn's reasoning marked. A turn survives
    exactly when its marker reaches the render: a marker rides only inside text the template chose
    to keep as THIS turn's reasoning, so a ``<think>`` an answer merely quotes carries none and a
    span in a system or user message -- never supervised, since the split is assistant-only -- cannot
    be credited either. Counting rendered tags instead gets both wrong, and scores the total-loss
    case as one survivor because the template injects an empty block on trailing assistant turns.

    A survivor also has to fit inside ``max_length``: the cap slices the token row, so a block past
    it never reaches the loss. Truncation is judged per turn rather than per row, because the cap
    usually cuts the answer tail while leaving an earlier reasoning block whole.

    Markers are stripped before measuring, so the length charged against the cap is the real render's
    rather than one inflated by the measurement's own bytes.
    """
    authored = reasoned_assistant_turns(completion_messages)
    if not authored:
        # nothing authored means nothing to lose, and the marked render would equal the full one.
        # skipped rather than rendered so a dataset with no reasoning pays nothing for this check.
        return _RowReasoning(0, 0, 0)
    prefix = reasoning_marker_prefix(render([*prompt_messages, *completion_messages]))
    marked = render([*prompt_messages, *with_marked_reasoning(completion_messages, prefix)])
    rendered = 0
    truncated = 0
    for marker in reasoning_markers(completion_messages, prefix):
        end = marked_reasoning_end(marked, marker)
        if end is None:
            continue
        rendered += 1
        through = strip_reasoning_markers(marked[:end], prefix)
        if _encoded_length(tokenizer, through) > max_length:
            truncated += 1
    return _RowReasoning(authored, rendered, truncated)


def _require_rendered_reasoning(reasoning: _RowReasoning, *, row_index: int) -> None:
    """Reject template loss separately from context-cap truncation."""
    if reasoning.rendered_spans < reasoning.authored_turns:
        raise ValueError(
            f"thinking = true, but the chat template dropped "
            f"{reasoning.authored_turns - reasoning.rendered_spans} of "
            f"{reasoning.authored_turns} authored reasoning blocks in selected SFT row "
            f"{row_index + 1}. Every assistant target must retain its reasoning after "
            "formatting. Split each K-turn transcript into K single-turn rows, with "
            "earlier turns in prompt_messages and one assistant target per row; otherwise "
            "set thinking = false."
        )
