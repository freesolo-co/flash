"""Parsing of the chat/adapter targets a user types, and the request body they build.

Split out of ``flash.client.http`` to keep that module under the file-size gate. These are pure
functions over strings and dicts -- no transport, no client state -- so they read and test on their
own. ``http`` re-exports them, which is what keeps ``from flash.client.http import ...`` working.

``ClientError`` is imported inside each function rather than at module scope: it is defined in
``http``, which imports this module, so a top-level import would be circular. The functions here
already late-import ``flash.schema`` for the same shape of reason, so this stays consistent.
"""

from __future__ import annotations

from typing import Any

_CHAT_STEP_SELECTOR_CAPABILITY = "chat_step_selector_v1"


def _validate_chat_messages(messages: list[dict]) -> None:
    from flash.client.http import ClientError

    if not isinstance(messages, list):
        raise ClientError("chat messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ClientError(f"chat messages[{index}] must be an object")


def _parse_chat_target(target: str) -> tuple[str, str | None, int | None]:
    from flash.client.http import ClientError
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    revision = parse_adapter_revision(target)
    if revision is not None:
        return revision[0], target.strip(), None
    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError(
            "invalid run id: expected a bare RUN_ID, RUN_ID/step-N, or a full immutable adapter "
            "revision"
        )
    run_id, step = parsed
    return run_id, None, step


def _prepare_chat_request(
    target: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
) -> tuple[str, dict[str, Any]]:
    base_run_id, adapter_revision, step = _parse_chat_target(target)
    _validate_chat_messages(messages)
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
    if adapter_revision is not None:
        body["adapter_revision"] = adapter_revision
    elif step is not None:
        body["step"] = step
    return base_run_id, body


def _parse_adapter_target(target: str) -> tuple[str, int | None]:
    from flash.client.http import ClientError
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError(
            "invalid adapter id: expected RUN_ID for the final adapter or RUN_ID/step-N "
            "for a saved checkpoint"
        )
    return parsed
