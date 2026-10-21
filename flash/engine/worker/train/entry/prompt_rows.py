"""canonical parent-side prompt rows shared by grpo and opd parquet writers."""

from __future__ import annotations

from typing import Any

_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content", "reasoning_content"})
_TEXT_BLOCK_TYPES = frozenset({"text", "input_text"})


def canonical_prompt_messages(
    messages: list[dict[str, Any]], *, multimodal: bool
) -> list[dict[str, str]]:
    """validate messages and emit arrow-safe rows with string-only content."""
    if not isinstance(messages, list):
        raise ValueError("prompt messages must be a list")
    normalized: list[dict[str, str]] = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"prompt message {position} must be an object")
        extras = sorted(key for key in message if key not in _ALLOWED_MESSAGE_KEYS)
        if extras:
            raise ValueError(f"prompt message {position} has unsupported fields {extras}")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"prompt message {position} has an invalid role")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if "reasoning_content" in message and not isinstance(reasoning, str):
            raise ValueError("prompt reasoning_content must be text")
        if isinstance(content, str):
            rendered = content
        elif (
            content is None
            and "content" in message
            and role == "assistant"
            and isinstance(reasoning, str)
        ):
            rendered = ""
        elif isinstance(content, list):
            parts: list[str] = []
            for block_index, block in enumerate(content):
                if not isinstance(block, dict):
                    raise ValueError(
                        f"prompt message {position} content block {block_index} must be an object"
                    )
                block_type = block.get("type")
                if block_type in _TEXT_BLOCK_TYPES:
                    text = block.get("text")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"prompt message {position} text block {block_index} must contain text"
                        )
                    parts.append(text)
                elif block_type == "image" and multimodal:
                    parts.append("<image>")
                else:
                    raise ValueError(
                        f"prompt message {position} has unsupported content block type "
                        f"{block_type!r}"
                    )
            rendered = "".join(parts)
        else:
            raise ValueError(f"prompt message {position} content must be text or content blocks")
        row = {"role": role, "content": rendered}
        if isinstance(reasoning, str):
            row["reasoning_content"] = reasoning
        normalized.append(row)
    return normalized


def prompt_rows_include_reasoning(rows: list[dict[str, Any]]) -> bool:
    """scan every prompt message and validate optional reasoning fields."""
    include_reasoning = False
    for row in rows:
        for message in row.get("prompt", []):
            if "reasoning_content" not in message:
                continue
            if not isinstance(message["reasoning_content"], str):
                raise ValueError("prompt reasoning_content must be text")
            include_reasoning = True
    return include_reasoning


def prompt_message_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """return the shared arrow feature for a prompt message struct."""
    from datasets import Value

    prompt: dict[str, Any] = {
        "role": Value("string"),
        "content": Value("string"),
    }
    if prompt_rows_include_reasoning(rows):
        prompt["reasoning_content"] = Value("string")
    return prompt


__all__ = ["canonical_prompt_messages", "prompt_message_features", "prompt_rows_include_reasoning"]
