"""stdlib-only canonical reasoning normalization shared with verl child runtimes."""

from __future__ import annotations

from typing import Any

_TEXT_BLOCK_TYPES = frozenset({"text", "input_text"})


def _canonical_leading_thinking(content: str) -> tuple[str, str] | None:
    """split one unambiguous leading ``<think>`` span from its answer."""
    if not content.startswith("<think>"):
        return None
    if content.count("<think>") != 1 or content.count("</think>") != 1:
        return None
    close = content.find("</think>", len("<think>"))
    if close < 0:
        return None
    reasoning = content[len("<think>") : close].strip("\n")
    answer = content[close + len("</think>") :].lstrip("\n")
    return reasoning, answer


def _normalize_block_content(content: list[Any]) -> tuple[str, list[Any]] | None:
    """extract reasoning from one contiguous leading text-block run."""
    text_blocks: list[dict[str, Any]] = []
    for block in content:
        if (
            not isinstance(block, dict)
            or block.get("type") not in _TEXT_BLOCK_TYPES
            or not isinstance(block.get("text"), str)
        ):
            break
        text_blocks.append(block)
    if not text_blocks:
        return None
    split = _canonical_leading_thinking("".join(block["text"] for block in text_blocks))
    if split is None:
        return None
    reasoning, answer = split
    remaining = list(content[len(text_blocks) :])
    if answer:
        first = dict(text_blocks[0])
        first["text"] = answer
        remaining.insert(0, first)
    return reasoning, remaining


def messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """expose one canonical leading reasoning span through the template field."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        split: tuple[str, Any] | None = None
        if copied.get("role") == "assistant" and "reasoning_content" not in copied:
            if isinstance(content, str):
                split = _canonical_leading_thinking(content)
            elif isinstance(content, list):
                split = _normalize_block_content(content)
        if split is not None:
            copied["reasoning_content"], copied["content"] = split
        normalized.append(copied)
    return normalized


__all__ = ["messages_for_chat_template"]
