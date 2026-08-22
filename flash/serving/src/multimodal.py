"""Bounded, data-uri-only image handling for chat requests."""

from __future__ import annotations

from typing import Any

from flash.serve import request_validation as _shared

_MAX_IMAGES = _shared.MAX_IMAGES
_MAX_COMPRESSED_BYTES = _shared.MAX_COMPRESSED_BYTES
_MAX_TOTAL_COMPRESSED_BYTES = _shared.MAX_TOTAL_COMPRESSED_BYTES
_MAX_DIMENSION = _shared.MAX_DIMENSION
_MAX_TOTAL_DECODED_BYTES = _shared.MAX_TOTAL_DECODED_BYTES
_MAX_SOURCE_CHARS = _shared.MAX_SOURCE_CHARS
_MODE_BYTES_PER_PIXEL = _shared.MODE_BYTES_PER_PIXEL
_RGB_BYTES_PER_PIXEL = _shared.RGB_BYTES_PER_PIXEL
_WORST_BYTES_PER_PIXEL = _shared.WORST_BYTES_PER_PIXEL
_MAX_PIXELS = _shared.MAX_PIXELS
_IMAGE_TYPES = _shared.IMAGE_TYPES
_TEXT_TYPES = _shared.TEXT_TYPES
_ALLOWED_ROLES = _shared.ALLOWED_ROLES
_MIME_TO_FORMAT = _shared.MIME_TO_FORMAT
_DATA_URI_RE = _shared.DATA_URI_RE


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


def _decode_data_uri(source: str, index: int) -> tuple[bytes, str]:
    return _shared.decode_data_uri(
        source,
        index,
        max_compressed_bytes=_MAX_COMPRESSED_BYTES,
        error_type=MultimodalRequestError,
    )


def _decoded_bytes(mode: str, pixels: int) -> int:
    return _shared.decoded_bytes(mode, pixels)


def _validate_dimensions(width: int, height: int, index: int) -> int:
    return _shared.validate_dimensions(
        width,
        height,
        index,
        max_dimension=_MAX_DIMENSION,
        max_pixels=_MAX_PIXELS,
        error_type=MultimodalRequestError,
    )
