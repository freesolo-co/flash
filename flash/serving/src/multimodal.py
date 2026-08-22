"""Bounded, data-uri-only image handling for chat requests."""

from __future__ import annotations

import base64
import binascii
import io
import re
import warnings
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

_MAX_IMAGES = 4
_MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_COMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_DIMENSION = 8192
_MAX_TOTAL_DECODED_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_CHARS = len("data:image/webp;base64,") + 4 * ((_MAX_COMPRESSED_BYTES + 2) // 3)

# bytes per pixel of a decoded pillow image, by mode, used to bound decode memory.
# covers the modes png/webp/jpeg decode to; unknown modes fall back to a conservative 4.
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
# bytes per pixel that a decoded image occupies once it is the resident rgb copy we keep.
_RGB_BYTES_PER_PIXEL = 3
# decoding one image costs its own buffer plus a transient RGB conversion, so the worst mode
# sets the bytes-per-pixel a pixel budget has to assume.
_WORST_BYTES_PER_PIXEL = max(_MODE_BYTES_PER_PIXEL.values()) + 2 * _RGB_BYTES_PER_PIXEL

# the per-image pixel cap is derived from the memory budget rather than chosen: a pixel count the
# decoded-memory guard would always reject is not a limit, it is a promise the validator breaks.
# training advertises the same pair and derives it the same way.
#
# there is deliberately no cumulative pixel cap. what a set of images costs is decoded memory, and
# that depends on each image's decoded mode and on their order -- neither of which a sum of pixel
# counts can see. so every mode-blind total is wrong in one direction: low enough to be safe for
# four RGBA images and it rejects the same pixel count in cheaper modes, high enough to admit
# those and it can never fire. the cumulative decoded-memory guard below bounds it exactly.
_MAX_PIXELS = _MAX_TOTAL_DECODED_BYTES // _WORST_BYTES_PER_PIXEL
_IMAGE_TYPES = frozenset({"image_url", "input_image", "image"})
_TEXT_TYPES = frozenset({"text", "input_text"})
_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_DATA_URI_RE = re.compile(r"\Adata:(image/[^;,]+);base64,(.*)\Z", re.DOTALL)


class MultimodalRequestError(ValueError):
    """A client-visible invalid multimodal request."""


def has_image_blocks(messages: Any) -> bool:
    """Return whether a message list cheaply contains a recognized image block type."""
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") in _IMAGE_TYPES:
                return True
    return False


def normalize_chat_messages(
    messages: Any,
    *,
    supports_images: bool,
    image_limit: int | None,
) -> list[dict[str, Any]] | None:
    """Validate chat messages and return a template-ready list, or None to keep the original.

    Text-only lists come back normalized, because role rewrites (`developer` -> `system`) only
    take effect if the caller templates what was validated.

    Image-bearing lists come back as None. `_normalize_messages` strips each image block down to
    a bare `{"type": "image"}` and lifts the data uri into a separate `sources` channel, so that
    list is only meaningful alongside the decoded images. Handing it back would discard the
    sources while still looking like an image request to `has_image_blocks`, and the engine --
    which re-normalizes from the original itself -- would then fail on a block with no source.
    """
    template_messages, sources = _normalize_messages(messages)
    if not sources:
        # text-only: nothing to decode, and no image capability to resolve. returning here keeps
        # the shared shape/role checks universal while leaving the expensive image path untouched.
        return template_messages
    if not supports_images:
        raise MultimodalRequestError("the resolved base model does not support image input")
    images = _decode_images(sources, image_limit=image_limit)
    for image in images:
        image.close()
    # validated only. the engine re-normalizes the original list together with its sources.
    return None


def prepare_multimodal_request(
    messages: Any,
    *,
    image_limit: int | None,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Return processor-ready messages and detached RGB images after full validation."""
    template_messages, sources = _normalize_messages(messages)
    return template_messages, _decode_images(sources, image_limit=image_limit)


def _normalize_messages(messages: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(messages, list):
        raise MultimodalRequestError("messages must be a list of message objects")

    normalized: list[dict[str, Any]] = []
    sources: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise MultimodalRequestError(f"message {message_index} must be an object")
        role = message.get("role")
        if role == "developer":
            # openai's `developer` role is the successor to `system`; map it to system so image
            # chats accept it (these models have no distinct developer role), matching text requests.
            role = "system"
            message = {**message, "role": "system"}
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

        normalized_blocks: list[dict[str, Any]] = []
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
                normalized_blocks.append({"type": "text", "text": text})
                continue
            if block_type in _IMAGE_TYPES:
                if role != "user":
                    raise MultimodalRequestError("image blocks are allowed only in user messages")
                sources.append(_image_source(block, message_index, block_index))
                normalized_blocks.append({"type": "image"})
                continue
            raise MultimodalRequestError(
                f"message {message_index} content block {block_index} has unsupported type "
                f"{block_type!r}"
            )
        normalized.append({**message, "content": normalized_blocks})
    return normalized, sources


def _validate_tool_calls(tool_calls: Any, message_index: int) -> None:
    """require the shape every tool-aware template iterates over.

    This branch lets an assistant message carry `tool_calls` *instead of* content, but only tested
    that the key was present, so `tool_calls: 1` reached the chat template and raised a jinja error
    from outside the rejection handler -- answered 503, telling the caller to retry a request that
    must fail identically. `flash/serve/runtime/multimodal.py` already rejects these; this keeps the
    two serving paths answering the same question the same way.

    Deliberately structural and no deeper: a template's real requirements are its own, so this
    rejects only what no template can consume. An empty list is rejected too -- it means there are
    no calls, leaving `content: null` with nothing to render.
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
            f"message {message_index} content block {block_index} must contain exactly one image "
            "source"
        )
    source = candidates[0]
    if len(source) > _MAX_SOURCE_CHARS:
        raise MultimodalRequestError("image source exceeds the per-image encoded-size limit")
    return source


def _decode_images(sources: list[str], *, image_limit: int | None) -> list[Image.Image]:
    limit = _MAX_IMAGES if image_limit is None else min(image_limit, _MAX_IMAGES)
    if len(sources) > limit:
        raise MultimodalRequestError(f"at most {limit} images are allowed per request")

    images: list[Image.Image] = []
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
        for image in images:
            image.close()
        raise


def _decode_data_uri(source: str, index: int) -> tuple[bytes, str]:
    if source.startswith(("http:", "https:", "file:", "ftp:")):
        raise MultimodalRequestError(
            f"image {index} must use a data URI; remote and file URLs are not allowed"
        )
    if not source.startswith("data:"):
        raise MultimodalRequestError(f"image {index} must use a data URI")
    match = _DATA_URI_RE.fullmatch(source)
    if match is None:
        if ";base64," not in source:
            raise MultimodalRequestError(f"image {index} data URI must use base64 encoding")
        raise MultimodalRequestError(
            f"image {index} must use the exact data:image/...;base64,... format"
        )
    mime, encoded = match.groups()
    declared_format = _MIME_TO_FORMAT.get(mime)
    if declared_format is None:
        raise MultimodalRequestError(f"image {index} uses unsupported MIME type {mime!r}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MultimodalRequestError(f"image {index} contains invalid base64 data") from exc
    if len(data) > _MAX_COMPRESSED_BYTES:
        raise MultimodalRequestError(f"image {index} exceeds the compressed-byte limit")
    return data, declared_format


def _decoded_bytes(mode: str, pixels: int) -> int:
    # bound the peak load+convert allocation for one image: at `return rgb.copy()` three buffers
    # are live at once before `loaded`/`rgb` are released - the original decoded image (mode
    # bytes/pixel), the converted rgb image, and its returned copy - so charge mode + two rgb buffers.
    return pixels * (_MODE_BYTES_PER_PIXEL.get(mode, 4) + 2 * _RGB_BYTES_PER_PIXEL)


def _load_image(
    data: bytes,
    declared_format: str,
    index: int,
    *,
    prior_pixels: int,
) -> tuple[Image.Image, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                actual_format = probe.format
                if actual_format != declared_format:
                    raise MultimodalRequestError(
                        f"image {index} MIME type does not match its {actual_format or 'unknown'} format"
                    )
                if getattr(probe, "n_frames", 1) != 1:
                    raise MultimodalRequestError(
                        f"image {index} must be a static single-frame image"
                    )
                width, height = probe.size
                pixels = _validate_dimensions(width, height, index)
                decoded_bytes = _decoded_bytes(probe.mode, pixels)
                # earlier images are resident only as their rgb copies (3 bytes/pixel); this image
                # adds its transient load+convert peak on top. bound that coexisting worst case.
                if _RGB_BYTES_PER_PIXEL * prior_pixels + decoded_bytes > _MAX_TOTAL_DECODED_BYTES:
                    raise MultimodalRequestError("images exceed the total decoded-memory limit")
                probe.verify()

            with Image.open(io.BytesIO(data)) as loaded:
                if loaded.format != declared_format:
                    raise MultimodalRequestError(
                        f"image {index} MIME type does not match its format"
                    )
                if getattr(loaded, "n_frames", 1) != 1:
                    raise MultimodalRequestError(
                        f"image {index} must be a static single-frame image"
                    )
                width, height = loaded.size
                pixels = _validate_dimensions(width, height, index)
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


def _validate_dimensions(width: int, height: int, index: int) -> int:
    if width <= 0 or height <= 0:
        raise MultimodalRequestError(f"image {index} has zero dimensions")
    if width > _MAX_DIMENSION or height > _MAX_DIMENSION:
        raise MultimodalRequestError(f"image {index} exceeds the dimension limit")
    pixels = width * height
    if pixels > _MAX_PIXELS:
        raise MultimodalRequestError(f"image {index} exceeds the pixel limit")
    return pixels
