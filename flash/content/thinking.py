"""Pure ``<think>`` parsing, shared by the single-turn and multi-turn grading paths.

These are plain string functions with no torch/worker dependency, so both
:mod:`flash.engine.worker.model.decoding` (single-turn, GPU-side) and :mod:`flash.envs.adapter`
(multi-turn, importable from the CLI) can reach the same implementation. Keeping one copy is the
point: the two paths previously handed graders different shapes of the same completion, so an
environment could grade correctly in one mode and silently mis-grade in the other.
"""

from __future__ import annotations

from typing import Any


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


def messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose one canonical inline reasoning span through Qwen3.8's template field.

    Normalize only an assistant message that starts with exactly one balanced ``<think>`` pair.
    Quoted, close-only, malformed, and repeated markers remain literal content. An explicit
    ``reasoning_content`` field remains authoritative, including an empty string.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        split = (
            _canonical_leading_thinking(content)
            if copied.get("role") == "assistant"
            and "reasoning_content" not in copied
            and isinstance(content, str)
            else None
        )
        if split is not None:
            copied["reasoning_content"], copied["content"] = split
        normalized.append(copied)
    return normalized


def strip_think(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """Drop <think> reasoning spans before grading. Uses LAST </think> (answer extraction);
    unclosed reasoning returns "" to score 0 (reward pressure to finish within budget)."""
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    # No </think>: check prompt-opened before model-opened — else an echoed <think> leaks pre-think text.
    if prompt_opened_thinking:
        return ""
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
    return completion


def thinking_text(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """Extract reasoning text for scorers; ``None`` means no thinking span was present."""
    if completion is None:
        return None
    if prompt_opened_thinking:
        text = completion.rsplit("</think>", 1)[0] if "</think>" in completion else completion
        open_idx = text.find("<think>")
        if open_idx != -1 and not text[:open_idx].strip():
            text = text[open_idx + len("<think>") :]
        return text

    segments: list[str] = []
    pos = 0
    while True:
        open_idx = completion.find("<think>", pos)
        close_idx = completion.find("</think>", pos)
        if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
            segments.append(completion[pos:close_idx])
            pos = close_idx + len("</think>")
            continue
        if open_idx == -1:
            break
        start = open_idx + len("<think>")
        close_idx = completion.find("</think>", start)
        if close_idx == -1:
            segments.append(completion[start:])
            break
        segments.append(completion[start:close_idx])
        pos = close_idx + len("</think>")
    if not segments:
        return None
    return "\n".join(segments)


__all__ = ["messages_for_chat_template", "strip_think", "thinking_text"]
