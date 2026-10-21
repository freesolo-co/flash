"""bounded data-uri image handling with pillow imported only on demand."""

from __future__ import annotations

from typing import Any

from flash.serve.request import validation as _shared

from .errors import MultimodalRequestError

_MAX_IMAGES = _shared.MAX_IMAGES
_MAX_COMPRESSED_BYTES = _shared.MAX_COMPRESSED_BYTES
_MAX_TOTAL_COMPRESSED_BYTES = _shared.MAX_TOTAL_COMPRESSED_BYTES
_MAX_DIMENSION = _shared.MAX_DIMENSION
_MAX_TOTAL_DECODED_BYTES = _shared.MAX_TOTAL_DECODED_BYTES
_MAX_SOURCE_CHARS = _shared.MAX_SOURCE_CHARS
_MAX_PIXELS = _shared.MAX_PIXELS


def has_image_blocks(messages: Any) -> bool:
    """return whether a message list contains a recognized image block."""
    return _shared.has_image_blocks(messages, sequence_types=(list, tuple))


def validate_messages(messages: Any) -> None:
    """raise unless every message satisfies the rules generation will later assume.

    This deliberately discards the normalized result: normalization replaces image blocks with a
    bare `{"type": "image"}` and carries payloads separately. Only the text branch may adopt that
    result, past the image dispatch in `PromptPreparer`.
    """
    _normalize_messages(messages)


def normalize_text_messages(messages: Any) -> list[dict[str, Any]]:
    """return template-ready messages for a request that carries no image blocks.

    The accepted vocabulary is wider than one chat template's. `developer` becomes `system`, and
    `input_text` becomes `text`, so the validated request and rendered prompt remain identical.
    Image payloads are not decoded here, keeping text requests free of base64 and pillow work.
    """
    normalized, _sources = _normalize_messages(messages)
    return normalized


def prepare_multimodal_request(
    messages: Any,
    *,
    image_limit: int | None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """return processor-ready messages and detached rgb images."""
    template_messages, sources = _normalize_messages(messages)
    return template_messages, _decode_images(sources, image_limit=image_limit)


def _normalize_messages(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    return _shared.normalize_messages(
        messages,
        sequence_types=(list, tuple),
        sequence_error="messages must be a sequence of message objects",
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
