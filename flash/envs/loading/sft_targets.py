"""The one parser for a dataset row's message-shaped SFT ``output``.

A row's gold completion may be authored as a message list or as ``{"messages": [...]}``. Both the
control-plane quote (flash/engine/profiling/dataset_profile.py, which runs at preview and at
submit) and the worker (flash/envs/loading/adapter.py) read it through this module, so a row the
preview accepts is a row the worker trains on.

The rules are those of ``freesolo.datasets.target_messages`` -- every entry an object, a
non-empty string role, content a string or null only beside ``tool_calls`` -- plus one Flash
extension: content may be a list of content-block objects. That is how a multi-turn trajectory
carries a screenshot observation in a user turn (text + image blocks). Which roles may carry
images is decided later, identically for both paths, by the shared SFT workload preparation
(flash/engine/profiling/sft_workload.py): images in user turns are model input, images in
assistant or tool turns are refused there.
"""

from __future__ import annotations

import json
from typing import Any


def is_message_target(value: object) -> bool:
    """Whether ``output`` is message-shaped (a message list or a ``messages`` container)."""
    if isinstance(value, dict):
        return "messages" in value
    return isinstance(value, list) and any(isinstance(message, dict) for message in value)


def parse_message_target(value: object, *, row_id: Any = None) -> list[dict]:
    """Copy and validate a message-shaped ``output``; raise ``ValueError`` naming the defect."""
    where = f"sft output for row id {row_id!r}"
    source = "output"
    if isinstance(value, dict):
        sibling_keys = sorted(str(key) for key in value if key != "messages")
        if sibling_keys:
            raise ValueError(
                f"{where} has 'messages' alongside sibling keys {sibling_keys}; expected exactly "
                "{'messages': [...]}"
            )
        value = value["messages"]
        source = "'messages'"
    if not isinstance(value, list):
        raise ValueError(
            f"{where} has a {source} value of type {type(value).__name__}; expected a list of "
            "message objects"
        )
    if not value:
        raise ValueError(f"{where} has an empty {source} list; expected at least one message")
    invalid = [index for index, message in enumerate(value) if not isinstance(message, dict)]
    if invalid:
        raise ValueError(
            f"{where} contains non-object {source} entries at indexes {invalid}; expected a list "
            "of message objects"
        )
    messages: list[dict] = []
    for index, item in enumerate(value):
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{where} messages[{index}].role must be a non-empty string")
        content = item.get("content")
        if isinstance(content, list):
            if not content or not all(isinstance(block, dict) for block in content):
                raise ValueError(
                    f"{where} messages[{index}].content list must be a non-empty list of "
                    "content-block objects"
                )
        elif content is not None and not isinstance(content, str):
            raise ValueError(
                f"{where} messages[{index}].content must be a string, a content-block list, or null"
            )
        elif content is None and not item.get("tool_calls"):
            raise ValueError(
                f"{where} messages[{index}].content must be a string or a content-block list "
                "(or null only with tool_calls)"
            )
        message = dict(item)
        message["role"] = role.strip()
        messages.append(message)
    return messages


def decode_tool_call_arguments(messages: list[dict], *, row_id: Any = None) -> list[dict]:
    """Decode JSON-string tool-call arguments into the mapping the chat template iterates.

    OpenAI-format data carries ``tool_calls[].function.arguments`` as a JSON string, while the
    Qwen3.5 template iterates ``arguments|items``. Every SFT render (preview quote and worker alike,
    through flash/engine/profiling/sft_workload.py) reads rows through this, so both accept the
    string form and refuse one that does not encode a JSON object, naming the row.
    """
    decoded: list[dict] = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            message = {**message, "tool_calls": [dict(call) for call in calls]}
            for position, call in enumerate(message["tool_calls"]):
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
                    continue
                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    arguments = None
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"sft row id {row_id!r} messages[{index}].tool_calls[{position}].function."
                        "arguments must be a JSON object or a string encoding one"
                    )
                call["function"] = {**function, "arguments": arguments}
        decoded.append(message)
    return decoded
