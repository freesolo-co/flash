"""the message fields a grpo episode can start from, shared by the worker and `flash env test`."""

from __future__ import annotations

# the grpo prompt parquet rows (flash/engine/worker/train/entry/prompt_rows.py) carry exactly these.
PROMPT_MESSAGE_KEYS = frozenset({"role", "content", "reasoning_content"})


def require_plain_prompt_turns(messages: object, *, example_id: object) -> None:
    """name the first start-of-episode field the grpo prompt rows cannot carry.

    canonical_prompt_messages refuses it too, but only after the chat template has already
    rendered the prompt -- where a native tool call with JSON-string arguments fails inside jinja as
    "Can only get item pairs from a mapping." with no hint of which row or field.
    """
    if not isinstance(messages, list):
        return
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        extras = sorted(key for key in message if key not in PROMPT_MESSAGE_KEYS)
        if extras:
            raise ValueError(
                f"GRPO prompt for example {example_id!r}: message {position} "
                f"(role {message.get('role')!r}) carries {extras}. GRPO starts every episode from "
                "plain role/content turns, so start_episode()/build_prompt_messages() cannot "
                "replay native tool-call history (tool_calls, tool_call_id, name). Render that "
                "history into the turns' text, or start the episode before the first tool call."
            )


__all__ = ["PROMPT_MESSAGE_KEYS", "require_plain_prompt_turns"]
