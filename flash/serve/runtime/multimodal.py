"""bounded data-uri image handling with pillow imported only on demand."""

from __future__ import annotations

import base64
import binascii
import io
import re
import warnings
from typing import Any

from .errors import MultimodalRequestError

_MAX_IMAGES = 4
_MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_COMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_DIMENSION = 8192
_MAX_TOTAL_DECODED_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_CHARS = len("data:image/webp;base64,") + 4 * ((_MAX_COMPRESSED_BYTES + 2) // 3)
_IMAGE_TYPES = frozenset({"image_url", "input_image", "image"})
_TEXT_TYPES = frozenset({"text", "input_text"})
_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MIME_TO_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
_DATA_URI_RE = re.compile(r"\Adata:(image/[^;,]+);base64,(.*)\Z", re.DOTALL)
_MODE_BYTES_PER_PIXEL = {
    "1": 1,
    "L": 1,
    "P": 1,
    "I;16": 2,
    "I;16B": 2,
    "I;16L": 2,
    "LA": 2,
    "La": 2,
    "RGB": 3,
    "YCbCr": 3,
    "LAB": 3,
    "HSV": 3,
    "RGBA": 4,
    "RGBa": 4,
    "CMYK": 4,
    "I": 4,
    "F": 4,
}
_RGB_BYTES_PER_PIXEL = 3
# decoding one image costs its own buffer plus a transient RGB conversion, so the worst mode
# sets the bytes-per-pixel a pixel budget has to assume.
_WORST_BYTES_PER_PIXEL = max(_MODE_BYTES_PER_PIXEL.values()) + 2 * _RGB_BYTES_PER_PIXEL

# derived, not chosen: a pixel count the decoded-memory guard would always reject is not a limit,
# it is a promise the validator breaks. training (`train/core/child/glue.py`) and the hosted
# serving validator advertise the same pair and derive it the same way -- a hardcoded cap here
# accepted images both of those reject, so one deployable image contract had two answers.
#
# there is deliberately no cumulative pixel cap. what a set of images costs is decoded memory,
# which depends on each image's mode and on their order -- neither of which a sum of pixel counts
# can see. with this per-image cap, `_MAX_IMAGES` of them cannot reach the old 33_554_432 total
# anyway, so that guard could never fire. the cumulative decoded-memory guard bounds it exactly.
_MAX_PIXELS = _MAX_TOTAL_DECODED_BYTES // _WORST_BYTES_PER_PIXEL


def has_image_blocks(messages: Any) -> bool:
    """return whether a message list contains a recognized image block."""
    if not isinstance(messages, list | tuple):
        return False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        if any(
            isinstance(block, dict) and block.get("type") in _IMAGE_TYPES
            for block in message["content"]
        ):
            return True
    return False


def validate_messages(messages: Any) -> None:
    """raise unless every message satisfies the rules generation will later assume.

    This is the boundary check, and it deliberately discards the normalized result rather than
    returning it: normalization replaces each image block with a bare `{"type": "image"}` and
    carries the payload out separately, so a caller that swapped in the normalized messages would
    drop every image. Only the text branch may adopt them, which is why the rewrite lives in
    `normalize_text_messages` and is applied in `PromptPreparer`, past the image dispatch.
    """

    _normalize_messages(messages)


def normalize_text_messages(messages: Any) -> list[dict[str, Any]]:
    """return template-ready messages for a request that carries no image blocks.

    `PromptPreparer.prepare` dispatches on `has_image_blocks`, so until now only image-bearing
    requests were normalized. A text-only request went straight to `apply_chat_template`, which is
    a jinja renderer, not a validator: a missing `content` rendered as the empty string and
    generated from an empty prompt (200 with garbage), while a non-string `content` or an unknown
    `role` raised a `TemplateError` from outside `_rejection_as_prompt_error` and was answered 503
    -- telling the caller the service is down and inviting a retry that must fail identically.

    Validating alone is not enough, because this vocabulary is wider than any one chat template's.
    `developer` is accepted here and rewritten to `system`, and `input_text` blocks are rewritten
    to `text`; Qwen3.5's template raises `Unexpected message role` on the former and only some
    templates read the latter. Returning the rewritten messages -- rather than checking a copy and
    rendering the original -- is what makes the accepted spelling and the rendered prompt the same
    request. Image *payloads* are decoded in `_decode_images`, which this does not call, so a
    text-only request still costs no base64 work and touches no pillow.
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
    if not isinstance(messages, list | tuple):
        raise MultimodalRequestError("messages must be a sequence of message objects")
    normalized: list[dict[str, Any]] = []
    sources: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise MultimodalRequestError(f"message {message_index} must be an object")
        role = message.get("role")
        if role == "developer":
            role = "system"
            message = {**message, "role": role}
        if role not in _ALLOWED_ROLES:
            raise MultimodalRequestError(
                f"message {message_index} role must be system, user, assistant, or tool"
            )
        content = message.get("content")
        if isinstance(content, str):
            normalized.append(dict(message))
            continue
        if content is None and role == "assistant" and "tool_calls" in message:
            _validate_tool_calls(message["tool_calls"], message_index)
            normalized.append(dict(message))
            continue
        if not isinstance(content, list):
            raise MultimodalRequestError(
                f"message {message_index} content must be a string or a list of content blocks"
            )
        blocks = _normalize_blocks(content, role, message_index, sources)
        normalized.append({**message, "content": blocks})
    return normalized, sources


def _validate_tool_calls(tool_calls: Any, message_index: int) -> None:
    """require the shape every tool-aware template iterates over.

    This branch exists to let an assistant message carry `tool_calls` *instead of* content, but it
    only tested that the key was present, so `tool_calls: 1` or `"x"` reached the chat template and
    raised a jinja `UndefinedError` from outside `_rejection_as_prompt_error` -- answered 503, which
    tells the caller to retry a request that must fail identically.

    Deliberately structural and no deeper. A template's real requirements are its own: Qwen3.5
    renders `[{"function": {"name": "f"}}]` but raises `TypeError` on
    `[{"function": {"name": "f", "arguments": "{}"}}]`, even though a string `arguments` is what the
    OpenAI schema specifies. Encoding that here would bind this shared vocabulary to one template
    and reject payloads a different one accepts, so this rejects only what no template can consume.
    An empty list is rejected too: it means there are no calls, which leaves `content: null` with
    nothing to render and is the empty-prompt case this branch is not for.
    """

    if not isinstance(tool_calls, list) or not tool_calls:
        raise MultimodalRequestError(
            f"message {message_index} tool_calls must be a nonempty list of call objects"
        )
    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            raise MultimodalRequestError(
                f"message {message_index} tool call {call_index} must be an object"
            )


def _normalize_blocks(
    content: list[Any],
    role: str,
    message_index: int,
    sources: list[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, dict):
            raise MultimodalRequestError(
                f"message {message_index} content block {block_index} must be an object"
            )
        block_type = block.get("type")
        if block_type in _TEXT_TYPES:
            text = block.get("text")
            if not isinstance(text, str):
                raise MultimodalRequestError(
                    f"message {message_index} content block {block_index} text must be a string"
                )
            normalized.append({"type": "text", "text": text})
        elif block_type in _IMAGE_TYPES:
            if role != "user":
                raise MultimodalRequestError("image blocks are allowed only in user messages")
            sources.append(_image_source(block, message_index, block_index))
            normalized.append({"type": "image"})
        else:
            raise MultimodalRequestError(
                f"message {message_index} content block {block_index} has unsupported type "
                f"{block_type!r}"
            )
    return normalized


def _image_source(block: dict[str, Any], message_index: int, block_index: int) -> str:
    candidates: list[Any] = []
    for key in ("url", "image_url", "input_image", "image"):
        if key not in block or block[key] is None:
            continue
        source = block[key]
        if isinstance(source, dict):
            source = source.get("url")
        candidates.append(source)
    if len(candidates) != 1 or not isinstance(candidates[0], str):
        raise MultimodalRequestError(
            f"message {message_index} content block {block_index} must contain exactly one image source"
        )
    source = candidates[0]
    if len(source) > _MAX_SOURCE_CHARS:
        raise MultimodalRequestError("image source exceeds the per-image encoded-size limit")
    return source


def _decode_images(sources: list[str], *, image_limit: int | None) -> list[Any]:
    limit = _MAX_IMAGES if image_limit is None else min(image_limit, _MAX_IMAGES)
    if len(sources) > limit:
        raise MultimodalRequestError(f"at most {limit} images are allowed per request")
    images: list[Any] = []
    total_compressed = 0
    total_pixels = 0
    try:
        for index, source in enumerate(sources):
            data, declared_format = _decode_data_uri(source, index)
            total_compressed += len(data)
            if total_compressed > _MAX_TOTAL_COMPRESSED_BYTES:
                raise MultimodalRequestError("images exceed the total compressed-byte limit")
            image, pixels = _load_image(
                data,
                declared_format,
                index,
                prior_pixels=total_pixels,
            )
            total_pixels += pixels
            images.append(image)
        return images
    except BaseException:
        _close_images(images)
        raise


def _decode_data_uri(source: str, index: int) -> tuple[bytes, str]:
    if source.startswith(("http:", "https:", "file:", "ftp:")):
        raise MultimodalRequestError(
            f"image {index} must use a data uri; remote and file urls are not allowed"
        )
    if not source.startswith("data:"):
        raise MultimodalRequestError(f"image {index} must use a data uri")
    match = _DATA_URI_RE.fullmatch(source)
    if match is None:
        if ";base64," not in source:
            raise MultimodalRequestError(f"image {index} data uri must use base64 encoding")
        raise MultimodalRequestError(
            f"image {index} must use the exact data:image/...;base64,... format"
        )
    mime, encoded = match.groups()
    declared_format = _MIME_TO_FORMAT.get(mime)
    if declared_format is None:
        raise MultimodalRequestError(f"image {index} uses unsupported mime type {mime!r}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MultimodalRequestError(f"image {index} contains invalid base64 data") from exc
    if len(data) > _MAX_COMPRESSED_BYTES:
        raise MultimodalRequestError(f"image {index} exceeds the compressed-byte limit")
    return data, declared_format


def _decoded_bytes(mode: str, pixels: int) -> int:
    return pixels * (_MODE_BYTES_PER_PIXEL.get(mode, 4) + 2 * _RGB_BYTES_PER_PIXEL)


def _load_image(
    data: bytes,
    declared_format: str,
    index: int,
    *,
    prior_pixels: int,
) -> tuple[Any, int]:
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                _validate_image_metadata(probe, declared_format, index, prior_pixels)
                probe.verify()
            with Image.open(io.BytesIO(data)) as loaded:
                pixels = _validate_image_metadata(loaded, declared_format, index, prior_pixels)
                loaded.load()
                ImageOps.exif_transpose(loaded, in_place=True)
                rgb = loaded.convert("RGB")
                try:
                    rgb.load()
                    return rgb.copy(), pixels
                finally:
                    rgb.close()
    except MultimodalRequestError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MultimodalRequestError(f"image {index} is a decompression bomb") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MultimodalRequestError(f"image {index} is invalid or truncated") from exc


def _validate_image_metadata(
    image: Any,
    declared_format: str,
    index: int,
    prior_pixels: int,
) -> int:
    if image.format != declared_format:
        raise MultimodalRequestError(
            f"image {index} mime type does not match its {image.format or 'unknown'} format"
        )
    if getattr(image, "n_frames", 1) != 1:
        raise MultimodalRequestError(f"image {index} must be a static single-frame image")
    pixels = _validate_dimensions(*image.size, index)
    peak_bytes = _RGB_BYTES_PER_PIXEL * prior_pixels + _decoded_bytes(image.mode, pixels)
    if peak_bytes > _MAX_TOTAL_DECODED_BYTES:
        raise MultimodalRequestError("images exceed the total decoded-memory limit")
    return pixels


def _validate_dimensions(width: int, height: int, index: int) -> int:
    if width <= 0 or height <= 0:
        raise MultimodalRequestError(f"image {index} has zero dimensions")
    if width > _MAX_DIMENSION or height > _MAX_DIMENSION:
        raise MultimodalRequestError(f"image {index} exceeds the dimension limit")
    pixels = width * height
    if pixels > _MAX_PIXELS:
        raise MultimodalRequestError(f"image {index} exceeds the pixel limit")
    return pixels


def _close_images(images: list[Any] | tuple[Any, ...]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if close is not None:
            close()
