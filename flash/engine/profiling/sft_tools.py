"""tool declarations an SFT row trains against.

A tool-calling agent is served with ``tools=``, and the chat template renders those declarations
into the prompt (Qwen3.5 prepends a ``# Tools`` block to the system turn). A row rendered without
them trains on a prompt the deployed adapter never sees. A row may therefore carry ``tools`` -- the
same per-example field as OpenAI fine-tuning data -- which is validated and normalized exactly as a
chat request's ``tools`` are (``text_tokenizer`` renders ``tools_wire(normalize_tools(...))``), then
rendered into every training view of the row. A row without ``tools`` renders as before.
"""

from __future__ import annotations

from typing import Any

from flash.serve.request.tool_calls import normalize_tools, tools_wire


def sft_row_tools(example) -> list[dict[str, Any]] | None:
    """The row's tools in the form serving renders them, or None when it declares none."""
    if not isinstance(example, dict) or example.get("tools") is None:
        return None
    try:
        return tools_wire(normalize_tools(example["tools"]))
    except ValueError as exc:
        raise ValueError(f"sft tools: {exc}") from exc


def sft_template_kwargs(thinking: bool, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Chat-template kwargs for every training view of a row.

    The transcript, the prompt boundary, the reasoning probe and the loss mask's field probe all
    render with these, so the supervised tokens are located in exactly the text that is trained on.
    Without tools this is the pre-existing thinking-only render.
    """
    kwargs: dict[str, Any] = {"enable_thinking": thinking, "preserve_thinking": False}
    if tools is not None:
        kwargs["tools"] = tools
    return kwargs


def reject_image_row_tools(example, *, image_bearing: bool) -> None:
    """Refuse tools on an image row before any tokenizer or processor is resolved.

    The image pipeline tokenizes through the processor's own render, so accepting them would silently
    train the row without the declarations it was given. Validating here, while rows are collected,
    also makes a malformed declaration fail with its own message rather than as a capability error.
    """
    if sft_row_tools(example) is not None and image_bearing:
        raise ValueError("sft tools are not supported on image-bearing rows")
