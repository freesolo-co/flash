"""Bounded, data-uri-only image handling for chat requests."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException, status

from flash.serve.request import validation as _shared
from flash.serving.src.engine.model_config import image_limit_for, supports_image_input
from flash.serving.src.io.schemas import AdapterRecord

_MAX_IMAGES = _shared.MAX_IMAGES
_MAX_COMPRESSED_BYTES = _shared.MAX_COMPRESSED_BYTES
_MAX_TOTAL_COMPRESSED_BYTES = _shared.MAX_TOTAL_COMPRESSED_BYTES
_MAX_DIMENSION = _shared.MAX_DIMENSION
_MAX_TOTAL_DECODED_BYTES = _shared.MAX_TOTAL_DECODED_BYTES
_MAX_SOURCE_CHARS = _shared.MAX_SOURCE_CHARS
_MAX_PIXELS = _shared.MAX_PIXELS


class MultimodalRequestError(ValueError):
    """A client-visible invalid multimodal request."""


def has_image_blocks(messages: Any) -> bool:
    """Return whether a message list cheaply contains a recognized image block type."""
    return _shared.has_image_blocks(messages, sequence_types=list)


def normalize_chat_messages(
    messages: Any,
    *,
    supports_images: bool,
    image_limit: int | None,
) -> list[dict[str, Any]] | None:
    """Validate chat messages and return normalized text, or None for an image request.

    Image normalization lifts data URIs into a separate source channel. Returning that normalized
    list without its sources would discard every image, so the engine receives and re-normalizes the
    original image-bearing messages after this boundary validation succeeds.
    """
    template_messages, sources = _normalize_messages(messages)
    if not sources:
        return template_messages
    if not supports_images:
        raise MultimodalRequestError("the resolved base model does not support image input")
    _shared.close_images(_decode_images(sources, image_limit=image_limit))
    return None


async def _prepare_generate_request(payload: Any, target: AdapterRecord) -> None:
    messages = getattr(payload, "messages", None)
    if not isinstance(messages, list):
        return
    # validate every message list, not only ones carrying list-form content. `generaterequest`
    # only checks that this is a list of dicts, and the engine hands it straight to the chat
    # template, so a string-content request skipping this bypassed both halves of the shared
    # contract: a malformed `{"role": "user"}` rendered as an empty prompt and billed the
    # completion, and `developer` was never rewritten to `system` before gpu dispatch.
    try:
        normalized = await asyncio.to_thread(
            normalize_chat_messages,
            messages,
            supports_images=supports_image_input(target.base_model),
            image_limit=image_limit_for(target.base_model),
        )
    except MultimodalRequestError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    # write back only for text-only lists, where the normalized form is self-contained and the
    # role rewrite would otherwise be lost. image lists return none: their normalized form has
    # the sources stripped out, so the engine must re-normalize the original.
    if normalized is not None:
        payload.messages = normalized


def prepare_multimodal_request(
    messages: Any,
    *,
    image_limit: int | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Return processor-ready messages and detached RGB images after full validation."""
    template_messages, sources = _normalize_messages(messages)
    return template_messages, _decode_images(sources, image_limit=image_limit)


def _normalize_messages(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    return _shared.normalize_messages(
        messages,
        sequence_types=list,
        sequence_error="messages must be a list of message objects",
        error_type=MultimodalRequestError,
        max_source_chars=_MAX_SOURCE_CHARS,
    )


def _decode_images(sources: list[str], *, image_limit: int | None) -> list[Any]:
    return _shared.decode_images(
        sources,
        image_limit=image_limit,
        max_images=_MAX_IMAGES,
        max_compressed_bytes=_MAX_COMPRESSED_BYTES,
        max_total_compressed_bytes=_MAX_TOTAL_COMPRESSED_BYTES,
        max_dimension=_MAX_DIMENSION,
        max_pixels=_MAX_PIXELS,
        max_total_decoded_bytes=_MAX_TOTAL_DECODED_BYTES,
        error_type=MultimodalRequestError,
    )
