"""OpenAI tool-call message fields in a multi-turn transcript for an environment that declares tools.

An environment without tools keeps the role/content transcript and its rejection of every other
field. One that declares tools renders them natively, the way serving does, so its transcript may
also carry the two shapes serving accepts in history: an assistant turn's ``tool_calls`` and a
tool result's ``tool_call_id``/``name``. Only those fields, on those roles, are admitted here; the
call arguments stay the JSON string the wire carries, and are decoded into the mapping the chat
template iterates only at render time, exactly as serving's ``detached_template_messages`` does.
"""

from __future__ import annotations

import json
from typing import Any

TOOL_MESSAGE_KEYS = frozenset({"tool_calls", "tool_call_id", "name"})


def _call_text(value, *, source: str, position: int, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} message {position} {field} must be a nonempty string")
    return value


def _validated_tool_calls(calls, *, source: str, position: int) -> list[dict[str, Any]]:
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{source} message {position} tool_calls must be a nonempty list")
    validated = []
    for index, call in enumerate(calls):
        field = f"tool_calls[{index}]"
        if not isinstance(call, dict) or set(call) - {"id", "type", "function"}:
            raise ValueError(
                f"{source} message {position} {field} must contain only id, type, and function"
            )
        if call.get("type") != "function":
            raise ValueError(f"{source} message {position} {field}.type must be function")
        function = call.get("function")
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ValueError(
                f"{source} message {position} {field}.function must contain exactly name and "
                "arguments"
            )
        validated.append(
            {
                "id": _call_text(
                    call.get("id"), source=source, position=position, field=f"{field}.id"
                ),
                "type": "function",
                "function": {
                    "name": _call_text(
                        function["name"], source=source, position=position, field=f"{field}.name"
                    ),
                    # the wire form: a json object encoded as a string. checked here so a reply the
                    # template could not render fails on its own message, not inside jinja.
                    "arguments": _json_object_text(
                        function["arguments"], source=source, position=position, field=field
                    ),
                },
            }
        )
    return validated


def _json_object_text(value, *, source: str, position: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} message {position} {field}.arguments must be a json string")
    try:
        decoded = json.loads(value)
    except ValueError as error:
        raise ValueError(
            f"{source} message {position} {field}.arguments is not valid json"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{source} message {position} {field}.arguments must encode an object")
    return value


def tool_message_fields(message: dict, *, source: str, position: int) -> dict[str, Any]:
    """the validated tool fields one message carries, or {} when it carries none.

    ``tool_calls`` belongs to an assistant turn and ``tool_call_id``/``name`` to a tool result,
    matching serving's ``validate_tool_history``; on any other role they are rejected rather than
    carried, because the template would silently drop them.
    """
    role = message.get("role")
    fields: dict[str, Any] = {}
    if message.get("tool_calls") is not None:
        if role != "assistant":
            raise ValueError(f"{source} message {position} tool_calls require the assistant role")
        fields["tool_calls"] = _validated_tool_calls(
            message["tool_calls"], source=source, position=position
        )
    for key in ("tool_call_id", "name"):
        if message.get(key) is None:
            continue
        if role != "tool":
            raise ValueError(f"{source} message {position} {key} requires the tool role")
        fields[key] = _call_text(message[key], source=source, position=position, field=key)
    return fields


def template_tool_messages(messages: list[dict]) -> list[dict]:
    """decode call arguments into the mapping the qwen template iterates, as serving does."""
    if not any(message.get("tool_calls") for message in messages):
        return messages
    from flash.serve.request.tool_calls import detached_template_messages

    return detached_template_messages(messages)


__all__ = ["TOOL_MESSAGE_KEYS", "template_tool_messages", "tool_message_fields"]
