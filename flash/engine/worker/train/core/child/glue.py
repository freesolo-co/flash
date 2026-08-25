"""child-side multi-turn rollout primitives shared by the OPD and GRPO verl agent loops.

stdlib only at import time. no verl import and no flash import, so the parent can copy this file
into a verl child's workdir the same way it copies the loop modules themselves. image decoding loads
Pillow only when a validated dynamic image reply arrives.

everything here is algorithm-neutral: turning an environment reply into the exact tokens the chat
template would have produced, deciding whether an assistant turn terminated or was truncated, and
the two small accounting helpers the loops share. what is NOT here is either loop's body -- OPD
distils each turn against a teacher and returns one output per turn, GRPO scores the episode and
returns one output per episode, and those differences are the point.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import io
import json
import warnings
from typing import Any
from uuid import uuid4

from flash_reasoning_normalization import (
    messages_for_chat_template as _messages_for_chat_template,
)

_ALLOWED_MESSAGE_KEYS = frozenset({"role", "content", "reasoning_content"})
_PROBE_PREFIX = "flash-env-glue-probe"
# the media block types verl's own dataset parser substitutes for a placeholder, and the exact set
# it then extracts (rl_dataset.py `_build_messages`: `<image>`/`<video>`/`<audio>` become blocks of
# the matching type, and `_process_multi_modal_info` returns images, videos, audios). duplicated
# here as literals rather than imported from flash.content.multimodal because this module is copied
# into the verl child, where flash is not importable -- see the module docstring.
_MEDIA_BLOCK_TYPES = frozenset({"image", "video", "audio"})


_IMAGE_DATA_URI_HEADERS = {
    "data:image/jpeg;base64": "JPEG",
    "data:image/png;base64": "PNG",
    "data:image/webp;base64": "WEBP",
}
_MAX_IMAGES_PER_REPLY = 4
_MAX_IMAGE_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_IMAGE_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_WIDTH = 8192
_MAX_IMAGE_HEIGHT = 8192
_MAX_TOTAL_DECODED_BYTES = 64 * 1024 * 1024
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

# derived from the memory budget rather than chosen independently, matching the parent's
# `flash.content.multimodal`: a pixel count the decoded-memory guard would always reject is not a
# limit. this file cannot import flash, so the derivation is repeated rather than shared -- the
# inputs above are the same, so the two land on the same numbers. there is no cumulative pixel cap
# for the same reason as the parent: decoded memory depends on mode and order, so no mode-blind
# total can track it, and the aggregate decoded-byte guard below already bounds it exactly.
_MAX_IMAGE_PIXELS = _MAX_TOTAL_DECODED_BYTES // _WORST_BYTES_PER_PIXEL


def normalize_token_ids(value) -> list[int]:
    """normalize tokenizer outputs to one flat list of integer token ids."""
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("tokenizer output must be list-like")
    return [int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in value]


def content_block_text(content, *, source: str, position: int) -> str:
    """flatten one openai-style content block list to the text the transcript represents.

    an image-bearing prompt does NOT reach the child as a string. verl's RLHFDataset rewrites the
    parquet's string content into blocks -- it splits on the `<image>` placeholder and replaces that
    segment with an image block (rl_dataset.py `_build_messages`) -- so `raw_prompt` arrives as
    [{"type": "image", ...}, {"type": "text", "text": ...}] for exactly the multimodal rows flash
    writes. validating that shape as text-only would reject every image episode at turn one.

    the media blocks are dropped rather than rendered here: they are carried separately, as decoded
    pixels in `multi_modal_data`, and the prompt ids come from the chat template applied to the
    ORIGINAL block list. this text view exists for the transcript and the inter-turn glue, which are
    token-level text operations. a block whose type is neither text nor known media is an error
    rather than a silent drop -- it would mean the model saw something this transcript cannot show.
    """
    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError(
                f"{source} message {position} has a content block that is not an object"
            )
        block_type = block.get("type")
        if block_type in _MEDIA_BLOCK_TYPES:
            continue
        if block_type != "text":
            raise ValueError(
                f"{source} message {position} has an unsupported content block type "
                f"{block_type!r}; a multi-turn transcript can represent text and media blocks only"
            )
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{source} message {position} has a text block without text")
        parts.append(text)
    return "".join(parts)


def validate_structured_messages(messages: list[dict], *, source: str) -> list[dict]:
    """validate and canonicalize role/content messages without dropping media placement."""
    if not isinstance(messages, list):
        raise ValueError(f"{source} messages must be a list")
    normalized: list[dict[str, Any]] = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source} message {position} must be an object")
        extras = sorted(key for key in message if key not in _ALLOWED_MESSAGE_KEYS)
        if extras:
            raise ValueError(
                f"{source} message {position} carries unsupported transcript metadata {extras}; "
                "tool names, call ids, tool calls, and other message fields cannot be represented "
                "in a role/content multi-turn transcript"
            )
        role = message.get("role")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{source} message {position} has an invalid role")
        if "reasoning_content" in message and not isinstance(reasoning, str):
            raise ValueError(f"{source} message {position} reasoning_content must be text")
        metadata = {"reasoning_content": reasoning} if reasoning is not None else {}
        if isinstance(content, str):
            normalized.append({"role": role, "content": content, **metadata})
            continue
        if not isinstance(content, list):
            raise ValueError(
                f"{source} message {position} content must be text or content blocks for multi-turn"
            )
        blocks: list[dict[str, Any]] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValueError(
                    f"{source} message {position} has a content block that is not an object"
                )
            block_type = block.get("type")
            allowed_keys = (
                {"type", block_type} if block_type in _MEDIA_BLOCK_TYPES else {"type", "text"}
            )
            extras = sorted(
                key for key, value in block.items() if key not in allowed_keys and value is not None
            )
            if extras:
                raise ValueError(
                    f"{source} message {position} content block {block_index} carries unsupported "
                    f"fields {extras}"
                )
            if block_type in _MEDIA_BLOCK_TYPES:
                blocks.append({"type": block_type})
            elif block_type == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if not text:
                    continue
                if blocks and blocks[-1].get("type") == "text":
                    blocks[-1]["text"] += text
                else:
                    blocks.append({"type": "text", "text": text})
            else:
                raise ValueError(
                    f"{source} message {position} has an unsupported content block at index "
                    f"{block_index}"
                )
        normalized.append({"role": role, "content": blocks, **metadata})
    return normalized


def validate_transcript_messages(
    messages: list[dict], *, source: str, allow_content_blocks: bool = False
) -> list[dict]:
    """require the exact role/content transcript shape the child rollout loops can represent.

    ``allow_content_blocks`` flattens openai-style block lists to their text instead of rejecting
    them. it is OFF by default and opted into only by a caller that has already extracted the media
    out of the ORIGINAL blocks, because dropping to text is lossless only once the pixels are held
    somewhere else. a caller that has not done that extraction must keep raising: silently training
    on a caption whose image was discarded is the failure this default exists to prevent.
    """
    if not isinstance(messages, list):
        raise ValueError(f"{source} messages must be a list")
    normalized = []
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source} message {position} must be an object")
        extras = sorted(key for key in message if key not in _ALLOWED_MESSAGE_KEYS)
        if extras:
            raise ValueError(
                f"{source} message {position} carries unsupported transcript metadata {extras}; "
                "tool names, call ids, tool calls, and other message fields cannot be represented "
                "in a role/content multi-turn transcript"
            )
        role = message.get("role")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{source} message {position} has an invalid role")
        if "reasoning_content" in message and not isinstance(reasoning, str):
            raise ValueError(f"{source} message {position} reasoning_content must be text")
        if allow_content_blocks and isinstance(content, list):
            # a multimodal prompt arrives as content BLOCKS, not a string; flatten to the text this
            # transcript can represent. the media rides separately as decoded pixels.
            content = content_block_text(content, source=source, position=position)
        elif not isinstance(content, str):
            raise ValueError(f"{source} message {position} content must be text for multi-turn")
        normalized_message = {"role": role, "content": content}
        if reasoning is not None:
            normalized_message["reasoning_content"] = reasoning
        normalized.append(normalized_message)
    return normalized


def _tokenize_text(tokenizer, text: str) -> list[int]:
    return normalize_token_ids(tokenizer(text, add_special_tokens=False))


def _unique_glue_probe(messages: list[dict]) -> str:
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    while True:
        probe = f"{_PROBE_PREFIX}-{uuid4().hex}"
        if probe not in serialized:
            return probe


def dedup_seam_terminator(response_ids: list[int], glue_ids: list[int]) -> list[int]:
    """keep the sampled terminator and drop the duplicate leading glue copy."""
    if response_ids and glue_ids and response_ids[-1] == glue_ids[0]:
        return glue_ids[1:]
    return glue_ids


class EnvGlueTokenizer:
    """derive exact inter-turn glue without re-rendering accepted transcript history."""

    def __init__(self, tokenizer, *, thinking: bool, cache_size: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.thinking = bool(thinking)
        self.cache_size = int(cache_size)
        self.cache: dict[str, list[int]] = {}

    def __call__(self, env_messages: list[dict]) -> list[int]:
        messages = validate_transcript_messages(env_messages, source="environment reply")
        key = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        probe = _unique_glue_probe(messages)
        text = self.tokenizer.apply_chat_template(
            _messages_for_chat_template([{"role": "assistant", "content": probe}, *messages]),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.thinking,
            preserve_thinking=False,
        )
        first = text.find(probe)
        if first == -1 or text.find(probe, first + len(probe)) != -1:
            raise ValueError(
                "multi-turn rollout could not uniquely locate the assistant-content probe in the "
                "chat template; exact inter-turn glue cannot be recovered"
            )
        glue_ids = _tokenize_text(self.tokenizer, text[first + len(probe) :])
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = list(glue_ids)
        return glue_ids


def _hash_framed(digest, name: str, value: bytes) -> None:
    encoded_name = name.encode("ascii")
    digest.update(len(encoded_name).to_bytes(4, "big"))
    digest.update(encoded_name)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _hash_processor_field(digest, field_name: str, value) -> None:
    if (
        not hasattr(value, "dtype")
        or not hasattr(value, "shape")
        or not callable(getattr(value, "tobytes", None))
    ):
        raise ValueError(
            f"processor image output {field_name!r} must expose dtype, shape, and tobytes"
        )
    if bool(getattr(value.dtype, "hasobject", False)):
        raise ValueError(f"processor image output {field_name!r} cannot use object dtype")
    try:
        shape = tuple(int(dimension) for dimension in value.shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"processor image output {field_name!r} has an invalid shape") from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"processor image output {field_name!r} has an invalid shape")
    try:
        data = value.tobytes(order="C")
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(
            f"processor image output {field_name!r} cannot provide C-contiguous bytes"
        ) from error
    if not isinstance(data, bytes):
        raise ValueError(f"processor image output {field_name!r} returned non-bytes from tobytes")
    shape_bytes = len(shape).to_bytes(4, "big") + b"".join(
        dimension.to_bytes(8, "big") for dimension in shape
    )
    _hash_framed(digest, f"{field_name}.dtype", str(value.dtype).encode("ascii"))
    _hash_framed(digest, f"{field_name}.shape", shape_bytes)
    _hash_framed(digest, f"{field_name}.data", data)


def processor_image_digest(processor, image) -> str:
    """hash one image's exact model-facing processor output."""
    image_processor = getattr(processor, "image_processor", None)
    if not callable(image_processor):
        raise ValueError("image media requires a processor with an image_processor")
    try:
        model_inputs = image_processor(images=[image], return_tensors="np")
    except Exception as error:
        raise ValueError("processor image preprocessing failed") from error
    digest = hashlib.sha256()
    digest.update(b"flash-processor-image-v1\0")
    for field_name in ("pixel_values", "image_grid_thw"):
        try:
            value = model_inputs[field_name]
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(
                f"processor image output is missing required field {field_name!r}"
            ) from error
        _hash_processor_field(digest, field_name, value)
    return digest.hexdigest()


def processor_image_digests(processor, images) -> list[str]:
    images = list(images or ())
    if not images:
        return []
    return [processor_image_digest(processor, image) for image in images]


def _validate_reply_image_count(messages: list[dict], image_data_uris: list[str]) -> None:
    image_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type")
            if block_type == "image":
                image_count += 1
            elif block_type in {"video", "audio"}:
                raise ValueError("environment reply image glue cannot carry video or audio media")
    if image_count != len(image_data_uris):
        raise ValueError(
            "environment reply image block count does not match its transported image data"
        )


def _validate_dynamic_image_dimensions(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("environment reply image width and height must be positive")
    if width > _MAX_IMAGE_WIDTH or height > _MAX_IMAGE_HEIGHT:
        raise ValueError(
            f"environment reply image dimensions {width}x{height} exceed the "
            f"{_MAX_IMAGE_WIDTH}x{_MAX_IMAGE_HEIGHT} limit"
        )
    pixels = width * height
    if pixels > _MAX_IMAGE_PIXELS:
        raise ValueError(
            f"environment reply image has {pixels} pixels, exceeding the "
            f"{_MAX_IMAGE_PIXELS}-pixel limit"
        )
    return pixels


def _decode_canonical_image_data_uri(uri: str) -> tuple[bytes, str]:
    if not isinstance(uri, str):
        raise ValueError("environment reply image transport must use canonical data uris")
    header, separator, payload = uri.partition(",")
    expected_format = _IMAGE_DATA_URI_HEADERS.get(header)
    if not separator or expected_format is None:
        if uri.startswith(("http://", "https://")):
            raise ValueError("environment reply image transport cannot use remote media")
        if uri.startswith("data:"):
            raise ValueError(
                "environment reply image transport must use canonical base64 PNG, JPEG, or WebP "
                "data uris"
            )
        raise ValueError("environment reply image transport must use canonical data uris")
    max_encoded_size = ((_MAX_IMAGE_SOURCE_BYTES + 2) // 3) * 4
    if len(payload) > max_encoded_size:
        raise ValueError(
            f"environment reply image source exceeds the {_MAX_IMAGE_SOURCE_BYTES}-byte limit"
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("environment reply image data uri contains invalid base64 data") from error
    if len(data) > _MAX_IMAGE_SOURCE_BYTES:
        raise ValueError(
            f"environment reply image source exceeds the {_MAX_IMAGE_SOURCE_BYTES}-byte limit"
        )
    if base64.b64encode(data).decode("ascii") != payload:
        raise ValueError("environment reply image transport data uri is not canonical")
    return data, expected_format


def _inspect_dynamic_image(data: bytes, expected_format: str) -> tuple[int, int]:
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                if image_format != expected_format:
                    raise ValueError(
                        "environment reply image data uri MIME type does not match its image format"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("environment reply image must be a static single-frame image")
                pixels = _validate_dynamic_image_dimensions(image.width, image.height)
                decoded_peak_bytes = pixels * (
                    _MODE_BYTES_PER_PIXEL.get(image.mode, 4) + 2 * _RGB_BYTES_PER_PIXEL
                )
                image.verify()
        return pixels, decoded_peak_bytes
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("environment reply image source is not a valid image") from error


def _decode_validated_dynamic_image(data: bytes, expected_format: str):
    from PIL import Image, ImageOps

    converted = None
    try:
        with Image.open(io.BytesIO(data)) as image:
            if str(image.format or "").upper() != expected_format:
                raise ValueError(
                    "environment reply image data uri MIME type does not match its image format"
                )
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("environment reply image must be a static single-frame image")
            _validate_dynamic_image_dimensions(image.width, image.height)
            image.load()
            ImageOps.exif_transpose(image, in_place=True)
            _validate_dynamic_image_dimensions(image.width, image.height)
            converted = image.convert("RGB")
            converted.load()
        return converted
    except ValueError:
        if converted is not None:
            converted.close()
        raise
    except Exception as error:
        if converted is not None:
            converted.close()
        raise ValueError("environment reply image source is not a valid image") from error


def _decode_image_data_uris(image_data_uris: list[str]) -> list:
    if len(image_data_uris) > _MAX_IMAGES_PER_REPLY:
        raise ValueError(f"environment reply exceeds the {_MAX_IMAGES_PER_REPLY}-image limit")
    prepared: list[tuple[bytes, str]] = []
    total_source_bytes = 0
    prior_pixels = 0
    for uri in image_data_uris:
        data, expected_format = _decode_canonical_image_data_uri(uri)
        total_source_bytes += len(data)
        if total_source_bytes > _MAX_TOTAL_IMAGE_SOURCE_BYTES:
            raise ValueError("environment reply images exceed the aggregate source-byte limit")
        pixels, decoded_peak_bytes = _inspect_dynamic_image(data, expected_format)
        if _RGB_BYTES_PER_PIXEL * prior_pixels + decoded_peak_bytes > _MAX_TOTAL_DECODED_BYTES:
            raise ValueError("environment reply images exceed the aggregate decoded-byte limit")
        prior_pixels += pixels
        prepared.append((data, expected_format))

    images = []
    try:
        for data, expected_format in prepared:
            images.append(_decode_validated_dynamic_image(data, expected_format))
        return images
    except Exception:
        for image in images:
            with contextlib.suppress(Exception):
                image.close()
        raise


def _processor_glue_ids(
    processor,
    tokenizer,
    messages: list[dict],
    images: list,
    *,
    apply_chat_template_kwargs: dict,
    mm_processor_kwargs: dict,
) -> list[int]:
    """derive exact append-only glue with the processor that expands the new images."""
    structured = validate_structured_messages(messages, source="environment reply")
    probe = _unique_glue_probe(structured)
    rendered = processor.apply_chat_template(
        [{"role": "assistant", "content": probe}, *structured],
        tokenize=False,
        add_generation_prompt=True,
        **apply_chat_template_kwargs,
    )
    first = rendered.find(probe)
    if first < 0 or rendered.find(probe, first + len(probe)) >= 0:
        raise ValueError(
            "multi-turn rollout could not uniquely locate the assistant-content probe in the "
            "processor chat template; exact inter-turn glue cannot be recovered"
        )
    suffix = rendered[first + len(probe) :]
    model_inputs = processor(
        text=[suffix],
        images=images,
        videos=None,
        return_tensors=None,
        **mm_processor_kwargs,
    )
    return normalize_token_ids(model_inputs)


def parent_image_digests(
    processor,
    image_descriptors: list[str] | tuple[str, ...],
    package_root: str | None,
) -> list[str]:
    from flash.content.multimodal import decode_image_descriptors

    images = decode_image_descriptors(list(image_descriptors), package_root)
    try:
        return processor_image_digests(processor, images)
    finally:
        for image in images:
            image.close()


def parent_environment_glue(
    processor,
    tokenizer,
    messages: list[dict],
    image_descriptors: list[str] | tuple[str, ...],
    package_root: str | None,
    *,
    thinking: bool,
) -> tuple[list[int], list[str]]:
    """derive the authenticated parent copy of one reply's glue and ordered image digests."""
    structured = validate_structured_messages(messages, source="environment reply")
    if not image_descriptors:
        transcript = validate_transcript_messages(
            structured,
            source="environment reply",
            allow_content_blocks=True,
        )
        return EnvGlueTokenizer(tokenizer, thinking=thinking)(transcript), []
    if processor is None:
        raise ValueError("environment reply carries images but the parent has no processor")
    from flash.content.multimodal import decode_image_descriptors

    images = decode_image_descriptors(list(image_descriptors), package_root)
    try:
        token_ids = _processor_glue_ids(
            processor,
            tokenizer,
            structured,
            images,
            apply_chat_template_kwargs={
                "enable_thinking": bool(thinking),
                "preserve_thinking": False,
            },
            mm_processor_kwargs={},
        )
        return token_ids, processor_image_digests(processor, images)
    finally:
        for image in images:
            image.close()


class EnvironmentReplyGlue:
    __slots__ = ("image_digests", "images", "token_ids")

    def __init__(self, token_ids: list[int], images: list, image_digests: list[str]) -> None:
        self.token_ids = list(token_ids)
        self.images = list(images)
        self.image_digests = list(image_digests)


class EnvGlueProcessor:
    """derive text or processor-aware environment glue and decode only newly transported images."""

    def __init__(self, loop_self, *, thinking: bool) -> None:
        self.loop_self = loop_self
        self.text = EnvGlueTokenizer(loop_self.tokenizer, thinking=thinking)
        self.apply_chat_template_kwargs = dict(
            getattr(loop_self, "apply_chat_template_kwargs", {}) or {}
        )
        self.apply_chat_template_kwargs["enable_thinking"] = bool(thinking)
        self.apply_chat_template_kwargs["preserve_thinking"] = False

    async def __call__(
        self,
        messages: list[dict],
        image_data_uris: list[str] | None = None,
    ) -> EnvironmentReplyGlue:
        image_data_uris = list(image_data_uris or ())
        structured = validate_structured_messages(messages, source="environment reply")
        _validate_reply_image_count(structured, image_data_uris)
        if not image_data_uris:
            text_messages = validate_transcript_messages(
                structured,
                source="environment reply",
                allow_content_blocks=True,
            )
            return EnvironmentReplyGlue(self.text(text_messages), [], [])
        processor = getattr(self.loop_self, "processor", None)
        if processor is None:
            raise ValueError("environment reply carries images but the rollout has no processor")
        images = _decode_image_data_uris(image_data_uris)
        try:
            mm_processor_kwargs = self.loop_self._get_mm_processor_kwargs(None)
            token_ids = _processor_glue_ids(
                processor,
                self.loop_self.tokenizer,
                structured,
                images,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
            )
            return EnvironmentReplyGlue(
                token_ids, images, processor_image_digests(processor, images)
            )
        except Exception:
            for image in images:
                with contextlib.suppress(Exception):
                    image.close()
            raise


def validate_glue_template(tokenizer, *, thinking: bool) -> None:
    """fail before rollout when assistant content cannot round-trip through the template."""
    EnvGlueTokenizer(tokenizer, thinking=thinking)(
        [{"role": "user", "content": "flash multi-turn glue validation"}]
    )


def trim_trailing_stop(
    tokenizer, response_ids, stop_text: str, stop_sequences
) -> tuple[list[int], str]:
    """Drop a trailing stop delimiter from BOTH the sampled ids and the decoded text (token-level).

    HF ``stop_strings`` halts only AFTER the delimiter text is emitted, so a run with e.g. ``[train]
    stop_sequences=["</answer>"]`` would otherwise score/distil the delimiter the user asked to stop
    at.
    """
    ids = [int(token_id) for token_id in response_ids]
    # Pick the LONGEST configured stop that is a trailing match (the earliest stop boundary in the
    # text). Overlapping delimiters like ["\n", "\n\n"] would otherwise trim only the first-listed
    # shorter suffix off a "\n\n" tail, leaving one newline for the teacher to score/distil; taking
    # the longest match removes the whole delimiter in one shot regardless of config order.
    stop = max(
        (value for value in stop_sequences if value and stop_text.endswith(value)),
        key=len,
        default="",
    )
    if not stop:
        return ids, tokenizer.decode(ids, skip_special_tokens=True)
    keep_length = len(stop_text) - len(stop)
    # Locate the kept prefix by scanning from the END on the SAME (special-tokens-included) decode
    # keep_length is measured in, so a special-token delimiter's id(s) are dropped correctly. Scanning
    # from the END is O(dropped * completion): the delimiter is short so only a handful of trailing
    # tokens drop; decoding growing prefixes from the START would be O(completion^2).
    kept = len(ids)
    while kept > 0 and len(tokenizer.decode(ids[:kept], skip_special_tokens=False)) > keep_length:
        kept -= 1
    # Return the kept ids decoded WITHOUT special tokens -- the teacher/alignment text. Decoding the
    # kept ids (not slicing stop_text at keep_length) keeps the teacher-scored text and the student
    # ids identical even when the stop starts INSIDE the final sampled token (that token decodes to
    # e.g. "B</answer>"): the whole token is excluded from `kept`, so the fused "B" is dropped rather
    # than left dangling to desync the loss alignment / token count.
    return ids[:kept], tokenizer.decode(ids[:kept], skip_special_tokens=True)


def prepare_assistant_turn(
    tokenizer,
    token_ids: list[int],
    *,
    stop_reason: str | None,
    max_tokens: int,
    eos_token_ids: frozenset[int],
    stop_sequences: tuple[str, ...],
) -> dict[str, Any]:
    """apply the shared termination, stop trimming, empty, and replacement-char gates."""
    raw_ids = [int(token_id) for token_id in token_ids]
    stop_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    ended_by_eos = bool(eos_token_ids and not eos_token_ids.isdisjoint(raw_ids))
    ended_by_stop = any(value and stop_text.endswith(value) for value in stop_sequences)
    ended_before_cap = stop_reason == "completed" and len(raw_ids) < int(max_tokens)
    terminated = ended_by_eos or ended_by_stop or ended_before_cap
    if stop_reason == "aborted":
        # an aborted rollout is truncated REGARDLESS of what signals appear inside the sampled
        # ids: the validator requires termination == "truncated" for truncated turns, so the
        # label must not leak eos/stop from partial content.
        terminated = False
        ended_by_eos = ended_by_stop = ended_before_cap = False
    termination = (
        "eos"
        if ended_by_eos
        else "stop"
        if ended_by_stop
        else "accepted_stop"
        if ended_before_cap
        else "truncated"
    )
    response_ids = raw_ids
    completion_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    # trim a trailing stop suffix whenever one ended the text — including when an EOS is ALSO
    # present (the label prefers eos, but the bridge's eos validation requires the trimmed span
    # to match; leaving the stop text in place desyncs response_ids from raw_response_ids).
    if terminated and ended_by_stop and not ended_by_eos:
        response_ids, completion_text = trim_trailing_stop(
            tokenizer, response_ids, stop_text, stop_sequences
        )
    if not terminated:
        skip_reason = "truncated_rollout"
        truncated = True
    elif not completion_text.strip():
        skip_reason = "empty_completion"
        truncated = False
    elif "�" in completion_text:
        skip_reason = "replacement_char"
        truncated = False
    else:
        skip_reason = ""
        truncated = False
    return {
        "raw_response_ids": raw_ids,
        "response_ids": response_ids,
        "completion_text": completion_text,
        "termination": termination,
        "stop_reason": stop_reason,
        "max_tokens": int(max_tokens),
        "truncated": truncated,
        "skip_reason": skip_reason,
    }


def turn_is_unusable(turn: dict[str, Any]) -> bool:
    """Whether the environment neither saw nor scored this turn.

    The bridge returns before ``record_model_turn`` for a truncated, empty, or replacement-char
    turn, so it never enters environment state and earns no reward. Both consequences follow from
    this one predicate: the turn's tokens must stay out of the loss (a zeroed ``response_mask``,
    the same way environment glue is excluded), and it must NOT take a turn span (``score_rollouts``
    rejects a span/reward count mismatch and drops the whole group to episode credit).
    """
    return bool(turn["truncated"] or turn["skip_reason"])


def sum_preemptions(current: int, value: int | None) -> int:
    if value is None:
        return current
    if current < 0:
        return int(value)
    return current + int(value)


async def run_executor_call(loop, callback):
    """finish a bridge request before propagating task cancellation."""
    task = asyncio.ensure_future(loop.run_in_executor(None, callback))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await asyncio.shield(task)
        raise


def multi_modal_image_count(multi_modal_data) -> int:
    """how many images a verl multimodal payload carries.

    single-turn and multi-turn both report this count to the bridge, which compares it against the
    frozen parent prompt. two counters would let the two paths disagree about the same payload, so
    this is the one definition: verl carries "image" (singular) for a single-image row and "images"
    for a list, and a bare non-sequence value under either key is one image.
    """
    if not multi_modal_data:
        return 0
    if not isinstance(multi_modal_data, dict):
        raise TypeError("verl multimodal data must be a mapping")
    images = multi_modal_data.get("images")
    if images is None:
        images = multi_modal_data.get("image")
    if images is None:
        return 0
    if not isinstance(images, (list, tuple)):
        return 1
    try:
        return len(images)
    except TypeError as error:
        raise TypeError("verl multimodal images must be a sized collection") from error


class EpisodePrompt:
    """the tokenized prompt plus cumulative decoded media used by every generation."""

    __slots__ = (
        "audios",
        "image_digests",
        "images",
        "mm_processor_kwargs",
        "prompt_ids",
        "structured_messages",
        "videos",
    )

    def __init__(
        self, processor, multi_modal_data, mm_processor_kwargs, prompt_ids, structured_messages
    ):
        multi_modal_data = dict(multi_modal_data or {})
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.prompt_ids = list(prompt_ids)
        self.structured_messages = structured_messages
        self.images = list(multi_modal_data.get("images") or ())
        self.videos = list(multi_modal_data.get("videos") or ())
        self.audios = list(multi_modal_data.get("audios") or ())
        self.image_digests = processor_image_digests(processor, self.images)

    def image_count(self) -> int:
        return len(self.images)

    def append_images(self, images: list, image_digests: list[str]) -> None:
        if len(images) != len(image_digests):
            raise ValueError("new image count does not match its digest count")
        self.images.extend(images)
        self.image_digests.extend(image_digests)

    def media_snapshot(self) -> dict:
        snapshot = {}
        if self.images:
            snapshot["images"] = list(self.images)
        if self.videos:
            snapshot["videos"] = list(self.videos)
        if self.audios:
            snapshot["audios"] = list(self.audios)
        return snapshot


async def prepare_episode_prompt(loop_self, raw_prompt) -> EpisodePrompt:
    """extract media and ids before building the canonical structured prompt view.

    order matters: an image prompt arrives as content blocks, so the media and block-rendered ids
    must be captured before structured validation. the canonical view preserves media placement for
    bridge authentication and rejects unsupported blocks and message metadata before rollout.
    """
    messages = [dict(message) for message in raw_prompt]
    for message in messages:
        # arrow materializes an omitted nullable struct field as null; restore omission before the
        # canonical validator so authored non-string reasoning metadata still fails in the parent.
        if message.get("reasoning_content") is None:
            message.pop("reasoning_content", None)
    multi_modal_data = await loop_self.process_multi_modal_info(messages)
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")
    audios = multi_modal_data.get("audios")
    structured = validate_structured_messages(messages, source="initial prompt")
    mm_processor_kwargs = loop_self._get_mm_processor_kwargs(audios)
    prompt_ids = await loop_self.apply_chat_template(
        messages,
        images=images,
        videos=videos,
        audios=audios,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    return EpisodePrompt(
        getattr(loop_self, "processor", None),
        multi_modal_data,
        mm_processor_kwargs,
        prompt_ids,
        structured,
    )
