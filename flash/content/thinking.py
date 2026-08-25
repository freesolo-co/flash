"""Pure ``<think>`` parsing, shared by the single-turn and multi-turn grading paths.

These are plain string functions with no torch/worker dependency, so both
:mod:`flash.engine.worker.model.decoding` (single-turn, GPU-side) and :mod:`flash.envs.adapter`
(multi-turn, importable from the CLI) can reach the same implementation. Keeping one copy is the
point: the two paths previously handed graders different shapes of the same completion, so an
environment could grade correctly in one mode and silently mis-grade in the other.
"""

from __future__ import annotations

from typing import Any

from flash.content.reasoning_normalization import messages_for_chat_template


def _messages_for_content_only_serialization(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inline reasoning that ``preserve_thinking=False`` keeps for content-only consumers."""
    last_user = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    serialized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        reasoning = copied.get("reasoning_content")
        if index > last_user and copied.get("role") == "assistant":
            prefix = (
                f"<think>\n{reasoning.strip()}\n</think>\n\n"
                if isinstance(reasoning, str)
                else "<think>\n\n</think>\n\n"
            )
            content = copied.get("content")
            if isinstance(content, list):
                copied["content"] = [{"type": "text", "text": prefix}, *content]
            elif content is None:
                copied["content"] = prefix
            elif isinstance(content, str):
                copied["content"] = prefix + content
        serialized.append(copied)
    return serialized


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
