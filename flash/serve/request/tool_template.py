"""the text the qwen tool template renders around an assistant turn's call blocks.

replay validation has to see the same prefix the model does. the parser selects the first complete
`<tool_call>` marker in the turn, so a marker anywhere ahead of the call blocks steals the parse,
and a marker the template discards must not be validated as though it survived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flash.content.reasoning_normalization import messages_for_chat_template
from flash.serve.contract.protocol import TEXT_TYPES

THINK_START, THINK_END = "<think>", "</think>"
_TOOL_RESPONSE_START, _TOOL_RESPONSE_END = "<tool_response>", "</tool_response>"


def _flatten_text(content: Any) -> Any:
    """supported text blocks reach the template as one concatenated string."""
    if not isinstance(content, list):
        return content
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") in TEXT_TYPES
        and isinstance(block.get("text"), str)
    )


def last_query_index(messages: Sequence[Mapping[str, Any]]) -> int:
    """index of the last ordinary user query, which gates whether reasoning is rendered."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        content = _flatten_text(messages[index].get("content"))
        text = content.strip() if isinstance(content, str) else ""
        # a synthesized tool-response turn is not a real query, so it does not close the span.
        if not (text.startswith(_TOOL_RESPONSE_START) and text.endswith(_TOOL_RESPONSE_END)):
            return index
    return len(messages) - 1


def rendered_turn_prefix(message: Mapping[str, Any], with_reasoning: bool) -> str:
    """the text the template renders before this turn's call blocks."""
    # deriving the prefix from the same normalization the prompt path applies is what keeps this
    # exact. validating the raw request fields instead would reject a turn whose markers the
    # template discards, and miss ones its own splitting introduces.
    normalized = messages_for_chat_template([dict(message)])[0]
    reasoning = normalized.get("reasoning_content")
    content = _flatten_text(normalized.get("content"))
    if not isinstance(reasoning, str):
        # the template gates on `reasoning_content is string`, so a non-string value is discarded
        # rather than stringified, and it falls back to splitting the content itself. keying this
        # on absence instead would skip that split whenever the field merely exists, rejecting a
        # turn whose markers the split discards.
        reasoning = None
        if isinstance(content, str) and THINK_END in content:
            # it takes reasoning from before the FIRST `</think>` but the answer from after the
            # LAST one, so the two splits are not symmetric and cannot share one partition.
            reasoning = content.partition(THINK_END)[0].rstrip("\n").rpartition(THINK_START)[2]
            reasoning = reasoning.lstrip("\n")
            content = content.rpartition(THINK_END)[2].lstrip("\n")
    prefix = ""
    # the `<think>` block is rendered only for turns after the last ordinary user query. an
    # earlier turn's reasoning is dropped, so including it would reject a replayable turn.
    if with_reasoning and isinstance(reasoning, str) and reasoning.strip():
        prefix += f"{THINK_START}\n{reasoning}\n{THINK_END}\n\n"
    # the separator is gated on the trimmed answer, so whitespace-only content renders as nothing
    # at all and must not contribute a prefix the model never sees.
    if isinstance(content, str) and content.strip():
        prefix += f"{content}\n\n"
    return prefix


__all__ = ["THINK_END", "THINK_START", "last_query_index", "rendered_turn_prefix"]
