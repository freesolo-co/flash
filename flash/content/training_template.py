"""Chat-template options shared by supervised and reinforcement learning."""

from __future__ import annotations

from typing import Any


def training_template_kwargs(thinking: bool, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Shared chat-template options for SFT rows and RL prompts and reply glue.

    Tools are normalized wire declarations. Omit the key for tool-free renders.
    """
    kwargs: dict[str, Any] = {"enable_thinking": thinking, "preserve_thinking": False}
    if tools is not None:
        kwargs["tools"] = tools
    return kwargs
