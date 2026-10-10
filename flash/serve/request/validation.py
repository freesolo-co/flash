"""dependency-light request validation shared by both serving implementations."""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import warnings
from collections.abc import Mapping
from typing import Any

from flash.serve.contract.protocol import (
    IMAGE_TYPES,
    TEXT_TYPES,
    reject_non_finite_json_constant,
)
from flash.serve.request.tool_calls import validate_tool_history

MAX_IMAGES = 4
MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
MAX_TOTAL_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_DIMENSION = 8192
MAX_TOTAL_DECODED_BYTES = 64 * 1024 * 1024
MAX_SOURCE_CHARS = len("data:image/webp;base64,") + 4 * ((MAX_COMPRESSED_BYTES + 2) // 3)
MAX_MESSAGE_NODES = 4096
MAX_MESSAGE_DEPTH = 256
ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
MIME_TO_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
DATA_URI_RE = re.compile(r"\Adata:(image/[^;,]+);base64,(.*)\Z", re.DOTALL)

# bytes per pixel of a decoded pillow image. these are the modes png, webp, and jpeg decode to;
# unknown modes use a conservative four bytes per pixel.
MODE_BYTES_PER_PIXEL = {
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
RGB_BYTES_PER_PIXEL = 3
# one decode can hold the original image, its rgb conversion, and the detached rgb copy together.
WORST_BYTES_PER_PIXEL = max(MODE_BYTES_PER_PIXEL.values()) + 2 * RGB_BYTES_PER_PIXEL

# this cap is derived rather than chosen. a pixel count the decoded-memory guard always rejects is
# not a real limit, and a lower cap rejects images the memory guard can safely admit. there is no
# cumulative pixel cap because decoded mode and image order determine memory; the cumulative
# decoded-memory guard below accounts for the actual coexisting buffers.
MAX_PIXELS = MAX_TOTAL_DECODED_BYTES // WORST_BYTES_PER_PIXEL

CONSTRAINT_KEYS = ("json", "regex", "choice", "json_object")
REMOVED_KEYS = frozenset({"grammar", "structural_tag"})
OPTION_KEYS = ("disable_any_whitespace", "disable_additional_properties", "whitespace_pattern")
CONSTRAINT_ALIASES = {"json_schema": "json", "schema": "json", "choices": "choice"}
OFF_STRINGS = frozenset({"", "none", "text"})
JSON_OBJECT_STRINGS = frozenset({"json", "json_object"})
ALLOWED_KEYS_HINT = (
    f"allowed keys: {', '.join(CONSTRAINT_KEYS)} "
    f"(aliases: {', '.join(sorted(CONSTRAINT_ALIASES))}), options: {', '.join(OPTION_KEYS)}"
)

ErrorType = type[Exception]
_COPYABLE_MESSAGE_SCALARS = (bool, float, int, str, type(None))


def detached_messages(
    messages: Any,
    *,
    sequence_types: type | tuple[type, ...],
    sequence_error: str,
    error_type: ErrorType,
) -> list[dict[str, Any]]:
    if not isinstance(messages, sequence_types):
        raise error_type(sequence_error)
    detached: list[dict[str, Any]] = []
    try:
        iterator = iter(messages)
    except Exception as exc:
        raise error_type("messages contain an unsupported value") from exc
    stack: list[tuple[Any, Any, int | None, bool]] = [(iterator, detached, None, True)]
    active: set[int] = {id(messages)}
    nodes = 1
    while stack:
        iterator, target, identity, top_level = stack[-1]
        try:
            entry = next(iterator)
        except StopIteration:
            stack.pop()
            active.discard(identity)
            continue
        except Exception as exc:
            raise error_type("messages contain an unsupported value") from exc
        key = None
        item = entry
        if isinstance(target, dict):
            try:
                key, item = entry
            except Exception as exc:
                raise error_type("messages contain an unsupported value") from exc
            if type(key) is not str:
                raise error_type("messages contain an unsupported value")
        nodes += 1
        if nodes > MAX_MESSAGE_NODES:
            raise error_type("messages exceed the supported complexity")
        if top_level and not isinstance(item, Mapping):
            raise error_type(f"message {len(target)} must be an object")
        if isinstance(item, Mapping):
            if len(stack) > MAX_MESSAGE_DEPTH:
                raise error_type("messages exceed the supported complexity")
            copied: dict[str, Any] = {}
            _append_detached(target, key, copied)
            nested = _message_items(item)
        elif isinstance(item, list | tuple):
            if len(stack) > MAX_MESSAGE_DEPTH:
                raise error_type("messages exceed the supported complexity")
            copied = []
            _append_detached(target, key, copied)
            nested = _message_values(item, error_type)
        elif type(item) in _COPYABLE_MESSAGE_SCALARS:
            _append_detached(target, key, item)
            continue
        else:
            raise error_type("messages contain an unsupported value")
        identity = id(item)
        if identity in active:
            raise error_type("messages must not contain recursive containers")
        active.add(identity)
        stack.append((nested, copied, identity, False))
    return detached


def _append_detached(target: Any, key: str | None, value: Any) -> None:
    if isinstance(target, list):
        target.append(value)
    else:
        target[key] = value


def _message_items(value: Mapping[Any, Any]) -> Any:
    for key in value:
        yield key, value[key]


def _message_values(value: list[Any] | tuple[Any, ...], error_type: ErrorType) -> Any:
    try:
        return iter(value)
    except Exception as exc:
        raise error_type("messages contain an unsupported value") from exc


def has_image_blocks(messages: Any, *, sequence_types: type | tuple[type, ...]) -> bool:
    if not isinstance(messages, sequence_types):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") in IMAGE_TYPES
            for block in message["content"]
        )
        for message in messages
    )


def normalize_messages(
    messages: Any,
    *,
    sequence_types: type | tuple[type, ...],
    sequence_error: str,
    error_type: ErrorType,
    max_source_chars: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(messages, sequence_types):
        raise error_type(sequence_error)
    normalized: list[dict[str, Any]] = []
    sources: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise error_type(f"message {message_index} must be an object")
        role = message.get("role")
        if role == "developer":
            # image models have no distinct developer role, so use the same system-role rewrite as
            # text requests. returning this rewritten message keeps validation and rendering aligned.
            role = "system"
            message = {**message, "role": role}
        if role not in ALLOWED_ROLES:
            raise error_type(
                f"message {message_index} role must be system, user, assistant, or tool"
            )
        content = message.get("content")
        if isinstance(content, str):
            normalized.append(dict(message))
            continue
        if content is None and role == "assistant" and "tool_calls" in message:
            normalized.append(dict(message))
            continue
        if not isinstance(content, list):
            raise error_type(
                f"message {message_index} content must be a string or a list of content blocks"
            )
        normalized.append(
            {
                **message,
                "content": normalize_blocks(
                    content,
                    role,
                    message_index,
                    sources,
                    error_type=error_type,
                    max_source_chars=max_source_chars,
                ),
            }
        )
    validate_tool_history(normalized, error_type=error_type)
    return normalized, sources


def normalize_blocks(
    content: list[Any],
    role: str,
    message_index: int,
    sources: list[str],
    *,
    error_type: ErrorType,
    max_source_chars: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, dict):
            raise error_type(
                f"message {message_index} content block {block_index} must be an object"
            )
        block_type = block.get("type")
        if block_type in TEXT_TYPES:
            text = block.get("text")
            if not isinstance(text, str):
                raise error_type(
                    f"message {message_index} content block {block_index} text must be a string"
                )
            normalized.append({"type": "text", "text": text})
        elif block_type in IMAGE_TYPES:
            if role != "user":
                raise error_type("image blocks are allowed only in user messages")
            sources.append(
                image_source(
                    block,
                    message_index,
                    block_index,
                    error_type=error_type,
                    max_source_chars=max_source_chars,
                )
            )
            normalized.append({"type": "image"})
        else:
            raise error_type(
                f"message {message_index} content block {block_index} has unsupported type "
                f"{block_type!r}"
            )
    return normalized


def image_source(
    block: dict[str, Any],
    message_index: int,
    block_index: int,
    *,
    error_type: ErrorType,
    max_source_chars: int,
) -> str:
    candidates: list[Any] = []
    for key in ("url", "image_url", "input_image", "image"):
        if key not in block or block[key] is None:
            continue
        source = block[key]
        if isinstance(source, dict):
            source = source.get("url")
        candidates.append(source)
    if len(candidates) != 1 or not isinstance(candidates[0], str):
        raise error_type(
            f"message {message_index} content block {block_index} must contain exactly one image source"
        )
    source = candidates[0]
    if len(source) > max_source_chars:
        raise error_type("image source exceeds the per-image encoded-size limit")
    return source


def decode_images(
    sources: list[str],
    *,
    image_limit: int | None,
    max_images: int,
    max_compressed_bytes: int,
    max_total_compressed_bytes: int,
    max_dimension: int,
    max_pixels: int,
    max_total_decoded_bytes: int,
    error_type: ErrorType,
) -> list[Any]:
    limit = max_images if image_limit is None else min(image_limit, max_images)
    if len(sources) > limit:
        raise error_type(f"at most {limit} images are allowed per request")
    images: list[Any] = []
    total_compressed = 0
    total_pixels = 0
    try:
        for index, source in enumerate(sources):
            data, declared_format = decode_data_uri(
                source,
                index,
                max_compressed_bytes=max_compressed_bytes,
                error_type=error_type,
            )
            total_compressed += len(data)
            if total_compressed > max_total_compressed_bytes:
                raise error_type("images exceed the total compressed-byte limit")
            image, pixels = load_image(
                data,
                declared_format,
                index,
                prior_pixels=total_pixels,
                max_dimension=max_dimension,
                max_pixels=max_pixels,
                max_total_decoded_bytes=max_total_decoded_bytes,
                error_type=error_type,
            )
            total_pixels += pixels
            images.append(image)
        return images
    except BaseException:
        close_images(images)
        raise


def decode_data_uri(
    source: str,
    index: int,
    *,
    max_compressed_bytes: int,
    error_type: ErrorType,
) -> tuple[bytes, str]:
    if source.startswith(("http:", "https:", "file:", "ftp:")):
        raise error_type(f"image {index} must use a data URI; remote and file URLs are not allowed")
    if not source.startswith("data:"):
        raise error_type(f"image {index} must use a data URI")
    match = DATA_URI_RE.fullmatch(source)
    if match is None:
        if ";base64," not in source:
            raise error_type(f"image {index} data URI must use base64 encoding")
        raise error_type(f"image {index} must use the exact data:image/...;base64,... format")
    mime, encoded = match.groups()
    declared_format = MIME_TO_FORMAT.get(mime)
    if declared_format is None:
        raise error_type(f"image {index} uses unsupported MIME type {mime!r}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise error_type(f"image {index} contains invalid base64 data") from exc
    if len(data) > max_compressed_bytes:
        raise error_type(f"image {index} exceeds the compressed-byte limit")
    return data, declared_format


def decoded_bytes(mode: str, pixels: int) -> int:
    return pixels * (MODE_BYTES_PER_PIXEL.get(mode, 4) + 2 * RGB_BYTES_PER_PIXEL)


def load_image(
    data: bytes,
    declared_format: str,
    index: int,
    *,
    prior_pixels: int,
    max_dimension: int,
    max_pixels: int,
    max_total_decoded_bytes: int,
    error_type: ErrorType,
) -> tuple[Any, int]:
    # pillow stays lazy so importing the policy-neutral runtime does not load image dependencies.
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                _validate_image_metadata(
                    probe,
                    declared_format,
                    index,
                    prior_pixels,
                    max_dimension=max_dimension,
                    max_pixels=max_pixels,
                    max_total_decoded_bytes=max_total_decoded_bytes,
                    error_type=error_type,
                )
                probe.verify()
            with Image.open(io.BytesIO(data)) as loaded:
                pixels = _validate_image_metadata(
                    loaded,
                    declared_format,
                    index,
                    prior_pixels,
                    max_dimension=max_dimension,
                    max_pixels=max_pixels,
                    max_total_decoded_bytes=max_total_decoded_bytes,
                    error_type=error_type,
                )
                loaded.load()
                ImageOps.exif_transpose(loaded, in_place=True)
                rgb = loaded.convert("RGB")
                try:
                    rgb.load()
                    return rgb.copy(), pixels
                finally:
                    rgb.close()
    except error_type:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise error_type(f"image {index} is a decompression bomb") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise error_type(f"image {index} is invalid or truncated") from exc


def _validate_image_metadata(
    image: Any,
    declared_format: str,
    index: int,
    prior_pixels: int,
    *,
    max_dimension: int,
    max_pixels: int,
    max_total_decoded_bytes: int,
    error_type: ErrorType,
) -> int:
    if image.format != declared_format:
        raise error_type(
            f"image {index} MIME type does not match its {image.format or 'unknown'} format"
        )
    if getattr(image, "n_frames", 1) != 1:
        raise error_type(f"image {index} must be a static single-frame image")
    pixels = validate_dimensions(
        *image.size,
        index,
        max_dimension=max_dimension,
        max_pixels=max_pixels,
        error_type=error_type,
    )
    peak_bytes = RGB_BYTES_PER_PIXEL * prior_pixels + decoded_bytes(image.mode, pixels)
    if peak_bytes > max_total_decoded_bytes:
        raise error_type("images exceed the total decoded-memory limit")
    return pixels


def validate_dimensions(
    width: int,
    height: int,
    index: int,
    *,
    max_dimension: int,
    max_pixels: int,
    error_type: ErrorType,
) -> int:
    if width <= 0 or height <= 0:
        raise error_type(f"image {index} has zero dimensions")
    if width > max_dimension or height > max_dimension:
        raise error_type(f"image {index} exceeds the dimension limit")
    pixels = width * height
    if pixels > max_pixels:
        raise error_type(f"image {index} exceeds the pixel limit")
    return pixels


def close_images(images: list[Any] | tuple[Any, ...]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if close is not None:
            close()


def decode_json(value: str) -> Any:
    return json.loads(value, parse_constant=reject_non_finite_json_constant)


def normalize_structured_outputs(
    value: Any,
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if value is False:
        return {}
    if value is True:
        raise error_type(
            "structured outputs spec cannot be `true`: pass a constraint "
            f'(e.g. {{"json": <schema>}}); {ALLOWED_KEYS_HINT}'
        )
    if isinstance(value, str):
        return _normalize_structured_string(
            value,
            error_type=error_type,
            validate_decoded_dicts=validate_decoded_dicts,
        )
    if isinstance(value, dict):
        return _normalize_structured_dict(
            value,
            error_type=error_type,
            validate_decoded_dicts=validate_decoded_dicts,
        )
    raise error_type(
        "structured outputs spec must be a dict, a JSON string, or an off marker "
        f"(null/false/'none'/'text'), got {type(value).__name__}"
    )


def _normalize_structured_string(
    value: str,
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> dict[str, Any]:
    keyword = value.strip().lower()
    if keyword in OFF_STRINGS:
        return {}
    if keyword in JSON_OBJECT_STRINGS:
        return {"json_object": True}
    # any other string must encode a dict form. this supports string-only transports while keeping
    # bad specs at the cpu validation boundary instead of the gpu grammar compiler.
    try:
        parsed = decode_json(value)
    except ValueError as exc:
        raise error_type(
            "structured outputs string must be 'json'/'json_object', an off marker "
            f"(''/'none'/'text'), or valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise error_type(
            "structured outputs JSON string must decode to an object (a JSON schema or a "
            f"constraint spec), got {type(parsed).__name__}"
        )
    return _normalize_structured_dict(
        parsed,
        error_type=error_type,
        validate_decoded_dicts=validate_decoded_dicts,
    )


def _normalize_structured_dict(
    value: dict[str, Any],
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> dict[str, Any]:
    # null means unset for spec-shaped dicts, matching structuredoutputsparams defaults.
    data = {key: item for key, item in value.items() if item is not None}
    if not data:
        # the empty dict is an explicit-off marker and must survive the gpu's second validation pass.
        return {}
    removed = sorted(key for key in data if key in REMOVED_KEYS)
    if removed:
        raise error_type(
            f"structured outputs {', '.join(removed)} constraint(s) are not supported; "
            f"use one of {', '.join(CONSTRAINT_KEYS)}"
        )
    if any(
        key in CONSTRAINT_KEYS or key in CONSTRAINT_ALIASES or key in OPTION_KEYS for key in data
    ):
        return _normalize_structured_canonical(
            data,
            error_type=error_type,
            validate_decoded_dicts=validate_decoded_dicts,
        )
    # nulls inside a raw schema are schema content, so validate and return the original dict.
    return {
        "json": _coerce_json_schema(
            value,
            error_type=error_type,
            validate_decoded_dicts=validate_decoded_dicts,
        )
    }


def _normalize_structured_canonical(
    data: dict[str, Any],
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for key, raw in data.items():
        canonical = CONSTRAINT_ALIASES.get(key, key)
        if canonical in CONSTRAINT_KEYS:
            if canonical in constraints:
                raise error_type(
                    f"structured outputs spec sets {canonical!r} twice "
                    f"(via {key!r} and an alias); {ALLOWED_KEYS_HINT}"
                )
            constraints[canonical] = _validate_structured_constraint(
                canonical,
                raw,
                error_type=error_type,
                validate_decoded_dicts=validate_decoded_dicts,
            )
        elif canonical in OPTION_KEYS:
            options[canonical] = _validate_structured_option(
                canonical,
                raw,
                error_type=error_type,
            )
        else:
            raise error_type(f"unknown structured outputs key {key!r}; {ALLOWED_KEYS_HINT}")
    if len(constraints) != 1:
        found = ", ".join(sorted(constraints)) or "none"
        raise error_type(
            "structured outputs spec must set exactly one constraint of "
            f"{', '.join(CONSTRAINT_KEYS)}; got {found}"
        )
    # canonical ordering keeps serialized payloads stable across both validation passes.
    return {**constraints, **{key: options[key] for key in OPTION_KEYS if key in options}}


def _validate_structured_constraint(
    key: str,
    value: Any,
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> Any:
    if key == "json":
        return _coerce_json_schema(
            value,
            error_type=error_type,
            validate_decoded_dicts=validate_decoded_dicts,
        )
    if key == "regex":
        if not isinstance(value, str) or not value.strip():
            raise error_type(
                f"structured outputs {key!r} must be a non-empty string, got {type(value).__name__}"
            )
        return value
    if key == "choice":
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            raise error_type(
                f"structured outputs 'choice' must be a non-empty list of strings, got {value!r}"
            )
        if not all(isinstance(item, str) for item in value):
            raise error_type(
                f"structured outputs 'choice' entries must all be strings, got {value!r}"
            )
        return value
    assert key == "json_object"
    if value is not True:
        raise error_type(
            "structured outputs 'json_object' must be true (to disable structured outputs "
            "send null, false, or 'none')"
        )
    return True


def _validate_structured_option(key: str, value: Any, *, error_type: ErrorType) -> Any:
    if key == "whitespace_pattern":
        if not isinstance(value, str) or not value:
            raise error_type(
                f"structured outputs option {key!r} must be a non-empty string, "
                f"got {type(value).__name__}"
            )
        return value
    if not isinstance(value, bool):
        raise error_type(
            f"structured outputs option {key!r} must be a boolean, got {type(value).__name__}"
        )
    return value


def _coerce_json_schema(
    value: Any,
    *,
    error_type: ErrorType,
    validate_decoded_dicts: bool,
) -> dict[str, Any]:
    if isinstance(value, dict):
        # hosted rejects decoded nan and infinity after strict outer json parsing. runtime has not
        # been traced end to end for that behavior, so this remains the sole policy difference.
        if validate_decoded_dicts:
            try:
                decode_json(json.dumps(value))
            except ValueError as exc:
                raise error_type(f"structured outputs 'json' is not valid JSON: {exc}") from exc
        return value
    if isinstance(value, str):
        try:
            parsed = decode_json(value)
        except ValueError as exc:
            raise error_type(f"structured outputs 'json' string is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise error_type(
                "structured outputs 'json' string must decode to a JSON schema object, "
                f"got {type(parsed).__name__}"
            )
        return parsed
    raise error_type(
        "structured outputs 'json' must be a JSON schema object (or a JSON string of one), "
        f"got {type(value).__name__}"
    )
