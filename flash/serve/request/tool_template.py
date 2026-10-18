"""the text the qwen tool template renders around an assistant turn's call blocks.

replay validation has to see the same prefix the model does. the parser selects the first complete
`<tool_call>` marker in the turn, so a marker anywhere ahead of the call blocks steals the parse,
and a marker the template discards must not be validated as though it survived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flash.content.reasoning_normalization import messages_for_chat_template
from flash.serve.contract.protocol import IMAGE_TYPES, TEXT_TYPES

THINK_START, THINK_END = "<think>", "</think>"
_TOOL_RESPONSE_START, _TOOL_RESPONSE_END = "<tool_response>", "</tool_response>"
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _rendered_content(content: Any) -> Any:
    """the string the template's `render_content` macro builds from a block list.

    an image is emitted as a placeholder rather than dropped, and the macro tests for it before it
    tests for text. dropping it here would let a turn ending in an image read as though it ended
    with the text block before it. video is the macro's only other block kind, and the request
    boundary rejects it as unsupported before this runs, so it cannot reach here.
    """
    if not isinstance(content, list):
        return content
    rendered: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in IMAGE_TYPES:
            rendered.append(_IMAGE_PLACEHOLDER)
        elif block.get("type") in TEXT_TYPES and isinstance(block.get("text"), str):
            rendered.append(block["text"])
    return "".join(rendered)


def last_query_index(messages: Sequence[Mapping[str, Any]]) -> int:
    """index of the last ordinary user query, which gates whether reasoning is rendered."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        content = _rendered_content(messages[index].get("content"))
        text = content.strip() if isinstance(content, str) else ""
        # a synthesized tool-response turn is not a real query, so it does not close the span.
        if not (text.startswith(_TOOL_RESPONSE_START) and text.endswith(_TOOL_RESPONSE_END)):
            return index
    # the template's own initializer for this span, kept identical so gating agrees with it. when
    # no ordinary query exists the template renders nothing at all and raises instead, which the
    # prompt path already reports as a caller error, so there is no turn here to replay.
    return len(messages) - 1


def rendered_turn_prefix(message: Mapping[str, Any], with_reasoning: bool) -> str:
    """the text the template renders before this turn's call blocks."""
    # deriving the prefix from the same normalization the prompt path applies is what keeps this
    # exact. validating the raw request fields instead would reject a turn whose markers the
    # template discards, and miss ones its own splitting introduces.
    normalized = messages_for_chat_template([dict(message)])[0]
    reasoning = normalized.get("reasoning_content")
    content = _rendered_content(normalized.get("content"))
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
    # the template trims both before emitting them, so surrounding whitespace never reaches the
    # model and must not be measured against the replay ceiling as though it did.
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    content = content.strip() if isinstance(content, str) else ""
    prefix = ""
    # the `<think>` block is rendered only for turns after the last ordinary user query. an
    # earlier turn's reasoning is dropped, so including it would reject a replayable turn.
    if with_reasoning and reasoning:
        prefix += f"{THINK_START}\n{reasoning}\n{THINK_END}\n\n"
    if content:
        prefix += f"{content}\n\n"
    return prefix


__all__ = ["THINK_END", "THINK_START", "last_query_index", "rendered_turn_prefix"]
