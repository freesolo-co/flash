"""Image normalization and bounded lazy decoding for managed training."""

from __future__ import annotations

import base64
import io
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.content import image_descriptors as _image_descriptors
from flash.content.image_descriptors import ValidatedImageDescriptor

MAX_IMAGES_PER_EXAMPLE = 4
MAX_IMAGE_SOURCE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_SOURCE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_WIDTH = 8192
MAX_IMAGE_HEIGHT = 8192
MAX_TOTAL_DECODED_BYTES = 64 * 1024 * 1024
# derived from the memory budget rather than chosen independently: with a separate number the
# two disagreed, and an image under the advertised pixel cap could still be rejected for decoded
# memory (a 4K RGB screenshot needed ~71 MiB against a 64 MiB budget). serving advertises this
# same pair, so both repositories derive it the same way.
#
# there is deliberately no cumulative pixel cap. decoded memory is what a set of images actually
# costs, and it depends on each image's decoded mode and on their order -- neither of which a sum
# of pixel counts can see. every mode-blind total therefore has to be wrong in one direction: a
# total low enough to be safe for four RGBA images rejects the same pixel count spread across
# cheaper modes, and one high enough to admit those never fires at all. the cumulative
# decoded-memory guard below already bounds the real resource exactly, per image and in order.
MAX_IMAGE_PIXELS = MAX_TOTAL_DECODED_BYTES // _image_descriptors.WORST_BYTES_PER_PIXEL
MAX_DATA_URI_HEADER_BYTES = 1024
MAX_IMAGE_DESCRIPTOR_BYTES = 64 * 1024 * 1024
MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES = 64 * 1024 * 1024
_IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})
_IMAGE_TARGET_BLOCK_TYPES = frozenset({*_IMAGE_BLOCK_TYPES, "output_image"})
_TEXT_BLOCK_TYPES = frozenset({"text", "input_text"})
_IMAGE_DETAIL_VALUES = frozenset({"auto", "low", "high"})
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
IMAGE_TEACHER_PLACEHOLDER = "<|media_pad|>"
# the model's real image-expansion token, as opposed to the placeholder above. one definition so
# the renderer's rejection and the teacher client's drop guard cannot drift apart.
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass(frozen=True)
class NormalizedImages:
    messages: list[dict]
    descriptors: list[str]


@dataclass(frozen=True)
class NormalizedEnvironmentReply:
    messages: list[dict]
    descriptors: tuple[str, ...]
    data_uris: tuple[str, ...]


def _image_limits() -> _image_descriptors.ImageDescriptorLimits:
    """Rebuild the limits per call so the module constants stay late-bound.

    this looks like a frozen value that could be built once at import, but the constants above are
    the documented override point: the limit tests `monkeypatch.setattr` them on this module and
    expect the next normalize call to honour the new value. binding a module-level singleton would
    freeze the import-time numbers and leave those tests green against the original limits, which is
    the worst failure mode available here -- a caps test that no longer tests the cap.
    """
    return _image_descriptors.ImageDescriptorLimits(
        max_images=MAX_IMAGES_PER_EXAMPLE,
        max_source_bytes=MAX_IMAGE_SOURCE_BYTES,
        max_total_source_bytes=MAX_TOTAL_IMAGE_SOURCE_BYTES,
        max_width=MAX_IMAGE_WIDTH,
        max_height=MAX_IMAGE_HEIGHT,
        max_pixels=MAX_IMAGE_PIXELS,
        max_total_decoded_bytes=MAX_TOTAL_DECODED_BYTES,
        max_data_uri_header_bytes=MAX_DATA_URI_HEADER_BYTES,
        max_descriptor_bytes=MAX_IMAGE_DESCRIPTOR_BYTES,
    )


def _is_pil_image(value: object) -> bool:
    try:
        from PIL import Image

        return isinstance(value, Image.Image)
    except ImportError:
        return False


def _descriptor_size(value: str) -> int:
    return _image_descriptors.descriptor_size(value)


def _descriptor(kind: str, value: str) -> str:
    return _image_descriptors.descriptor(
        kind,
        value,
        max_descriptor_bytes=MAX_IMAGE_DESCRIPTOR_BYTES,
    )


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    return _image_descriptors.decode_data_uri(uri, _image_limits())


def _check_source_size(size: int) -> None:
    _image_descriptors.check_source_size(size, max_source_bytes=MAX_IMAGE_SOURCE_BYTES)


def _package_image_path(value: str, package_root: Path | None) -> Path:
    return _image_descriptors.package_image_path(value, package_root, _image_limits())


def _validate_dimensions(width: int, height: int) -> int:
    return _image_descriptors.validate_dimensions(width, height, _image_limits())


def inspect_image_bytes(data: bytes) -> tuple[int, int, int]:
    """validate one image and return its decoded peak bytes, width, and height."""
    return _image_descriptors.inspect_image_bytes(data, _image_limits())


def _pil_descriptor(image: object) -> str:
    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    _validate_dimensions(width, height)
    out = io.BytesIO()
    image.save(out, format="PNG")
    data = out.getvalue()
    _check_source_size(len(data))
    return _descriptor("bytes", base64.b64encode(data).decode("ascii"))


def _canonical_data_uri(data: bytes, image_format: str) -> str:
    return _image_descriptors.canonical_data_uri(data, image_format)


def normalize_image_source(
    source: object,
    package_root: str | Path | None,
    *,
    defer_validation: bool = False,
) -> str:
    """Convert one supported image source to an Arrow-safe descriptor.

    ``defer_validation`` skips the per-source pixel inspection for EVERY source kind, for a caller
    that validates the resulting descriptors itself. Profiling does exactly that, through one cached
    pass that bounds total decode work; inspecting here as well would decode each image twice and
    charge only some kinds to that budget. Every other caller validates eagerly.
    """
    root = Path(package_root) if package_root is not None else None
    if isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, bytes):
        _check_source_size(len(source))
        descriptor = _descriptor("bytes", base64.b64encode(source).decode("ascii"))
        if not defer_validation:
            validate_image_descriptor_data(descriptor, source)
        return descriptor
    if _is_pil_image(source):
        return _pil_descriptor(source)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("image source must be a data URI, bytes, PIL image, or relative path")
    stripped = source.strip()
    if stripped.startswith("data:"):
        data, expected_format = _decode_data_uri(source)
        descriptor = _descriptor("data_uri", _canonical_data_uri(data, expected_format))
        if not defer_validation:
            validate_image_descriptor_data(descriptor, data)
        return descriptor
    parsed = urllib.parse.urlparse(stripped)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        # flash never fetches a user-supplied URL server-side. download the image ahead of time
        # and reference it as a relative path in the environment package, or embed it as a data
        # URI, so the training input is fixed at submit time rather than at training time.
        raise ValueError(
            "remote image URLs are not supported; include the image in the environment package "
            "as a relative path, or embed it as a data URI"
        )
    if scheme == "file":
        raise ValueError("file:// image URLs are not supported")
    if scheme:
        raise ValueError(f"unsupported image URL scheme: {scheme}")
    path = _package_image_path(stripped, root)
    descriptor = _descriptor("path", path.relative_to(root.resolve()).as_posix())
    if not defer_validation:
        validate_image_descriptor_data(descriptor, path.read_bytes())
    return descriptor


def _validate_image_detail(value: object, block_type: str) -> None:
    if value is not None and value not in _IMAGE_DETAIL_VALUES:
        raise ValueError(f"malformed {block_type} block: detail must be 'auto', 'low', or 'high'")


def _image_source_from_block(block: dict) -> object | None:
    block_type = block.get("type")
    source_keys = {
        "image_url": ("image_url",),
        "input_image": ("image_url", "input_image", "image", "url"),
        # source is a package-local authoring form retained for existing environments.
        "image": ("image", "image_url", "url", "source"),
    }.get(block_type)
    if source_keys is None:
        return None
    _validate_image_detail(block.get("detail"), str(block_type))
    candidates = [block[key] for key in source_keys if key in block and block[key] is not None]
    if not candidates and block_type == "image":
        return None
    if len(candidates) != 1:
        raise ValueError(f"malformed {block_type} block: expected exactly one image source")
    value = candidates[0]
    if isinstance(value, dict):
        extra = set(value) - {"url", "detail"}
        if extra or "url" not in value or not isinstance(value["url"], str):
            raise ValueError(
                f"malformed {block_type} block: expected image URL object with a url field"
            )
        _validate_image_detail(value.get("detail"), str(block_type))
        value = value["url"]
    return value


def _top_level_sources(record: dict) -> list[object]:
    sources: list[object] = []
    if "image" in record and record["image"] is not None:
        sources.append(record["image"])
    if "images" in record and record["images"] is not None:
        images = record["images"]
        if not isinstance(images, (list, tuple)):
            raise ValueError("top-level images must be a list or tuple")
        image_count = len(sources) + len(images)
        if image_count > MAX_IMAGES_PER_EXAMPLE:
            raise ValueError(
                f"example contains {image_count} images, exceeding the "
                f"{MAX_IMAGES_PER_EXAMPLE}-image limit"
            )
        sources.extend(images)
    return sources


def record_has_images(record: dict, messages: list[dict] | None = None) -> bool:
    if record.get("image") is not None or record.get("images") not in (None, [], ()):
        return True
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if content_has_images(content):
            return True
    return False


def content_has_images(content: object) -> bool:
    """Whether one message's ``content`` carries a prompt image block."""
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES for block in content
    )


def completion_has_images(messages: object) -> bool:
    """Whether an sft completion contains any sdk input or output image block spelling."""
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") in _IMAGE_TARGET_BLOCK_TYPES
            for block in message["content"]
        )
        for message in messages
    )


def text_only_prompt_messages(messages: list[dict]) -> list[dict]:
    """Drop image content blocks while preserving the teacher's message order and text."""
    stripped: list[dict] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = "".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") in _TEXT_BLOCK_TYPES
                and isinstance(block.get("text"), str)
            )
        stripped.append(copied)
    return stripped


def _reject_literal_image_placeholder(
    text: str,
    message_index: int,
    markers: tuple[str, ...] = (IMAGE_TEACHER_PLACEHOLDER, IMAGE_PAD_TOKEN),
) -> None:
    """Reject prompt text that already contains a reserved image marker.

    two distinct markers, one reason. the rendered prompt marks image positions with
    ``IMAGE_TEACHER_PLACEHOLDER`` and the teacher client splits on every occurrence to pair each
    one with an image, so a placeholder in the USER'S OWN text is indistinguishable from one this
    renderer inserted and would be paired with an image that does not exist.

    ``IMAGE_PAD_TOKEN`` is the model's real special token: whatever the provider expands an image
    into. the silent-drop guard counts its runs in the returned prompt ids and requires at least
    one run per supplied image. text containing the literal token encodes to that same id, so it
    contributes a run the renderer never produced -- and a run from text can make up for an image
    the provider silently dropped, which is exactly the failure the guard exists to catch. keeping
    it out of the source text is what makes "one run per image" mean what it says.

    ``markers`` narrows the set for callers where only one of the two is live. local sft training
    never renders the teacher placeholder, so a literal ``<|media_pad|>`` there is ordinary text
    for a qwen tokenizer and rejecting it would be a false refusal.
    """
    for marker in markers:
        if marker in text:
            raise ValueError(
                f"message {message_index} text contains the reserved image marker "
                f"{marker!r}; remove it from the prompt"
            )


def image_teacher_prompt_messages(messages: list[dict], descriptor_count: int) -> list[dict]:
    """Render normalized image blocks as Kimi media placeholders without changing text order."""
    from flash.content.thinking import _messages_for_content_only_serialization

    messages = _messages_for_content_only_serialization(messages)
    rendered: list[dict] = []
    image_count = 0
    for message_index, message in enumerate(messages):
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            # text blocks since the last image, joined and checked as one run. a marker split
            # across adjacent text blocks ("<|media_" then "pad|>") is invisible per block and
            # only becomes literal once concatenated, so the run is the unit that has to be
            # validated. an image block ends a run: the renderer's own placeholder goes there,
            # and user text on either side of it cannot reach across to form a marker.
            text_run: list[str] = []
            for block_index, block in enumerate(content):
                if not isinstance(block, dict):
                    raise ValueError(
                        f"malformed content block at message {message_index}, index {block_index}: "
                        "expected an object"
                    )
                block_type = block.get("type")
                if block_type in _TEXT_BLOCK_TYPES and isinstance(block.get("text"), str):
                    _reject_literal_image_placeholder(block["text"], message_index)
                    text_run.append(block["text"])
                    parts.append(block["text"])
                elif block_type in _IMAGE_BLOCK_TYPES:
                    _reject_literal_image_placeholder("".join(text_run), message_index)
                    text_run = []
                    parts.append(IMAGE_TEACHER_PLACEHOLDER)
                    image_count += 1
                else:
                    raise ValueError(
                        f"unsupported content block type {block_type!r} at message "
                        f"{message_index}, index {block_index}"
                    )
            _reject_literal_image_placeholder("".join(text_run), message_index)
            copied["content"] = "".join(parts)
        elif content is None:
            copied["content"] = ""
        elif isinstance(content, str):
            _reject_literal_image_placeholder(content, message_index)
        else:
            raise ValueError(f"message {message_index} content must be text or content blocks")
        rendered.append(copied)
    if image_count != descriptor_count:
        raise ValueError(
            f"image teacher prompt contains {image_count} placeholder(s), expected "
            f"{descriptor_count} normalized image descriptor(s)"
        )
    return rendered


def _canonical_message_role(role: object, message_index: int) -> str:
    if role == "developer":
        return "system"
    if role not in _ALLOWED_MESSAGE_ROLES:
        raise ValueError(f"message {message_index} role must be system, user, assistant, or tool")
    return str(role)


def normalize_prompt_images(
    record: dict,
    messages: list[dict],
    package_root: str | Path | None,
    *,
    defer_validation: bool = False,
) -> NormalizedImages:
    """Normalize prompt image blocks and top-level fields without mutating the record.

    ``defer_validation`` is forwarded to :func:`normalize_image_source`; see it for when a caller
    may skip the per-source pixel inspection.
    """
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("prompt messages must be a list of message objects")
    pending = list(_top_level_sources(record))
    descriptors: list[str] = []
    source_bytes = 0
    descriptor_bytes = 0

    def add_descriptor(source: object) -> None:
        nonlocal descriptor_bytes, source_bytes
        image_count = len(descriptors) + 1
        if image_count > MAX_IMAGES_PER_EXAMPLE:
            raise ValueError(
                f"example contains {image_count} images, exceeding the "
                f"{MAX_IMAGES_PER_EXAMPLE}-image limit"
            )
        descriptor = normalize_image_source(
            source,
            package_root,
            # the finished descriptor list is validated once below so aggregate limits and
            # mime/format agreement are checked over the exact cumulative image set.
            defer_validation=True,
        )
        descriptor_bytes += _descriptor_size(descriptor)
        if descriptor_bytes > MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES:
            raise ValueError(
                f"example encoded image descriptors total {descriptor_bytes} bytes, exceeding the "
                f"{MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES}-byte limit"
            )
        source_bytes += _descriptor_source_size(descriptor, package_root)
        if source_bytes > MAX_TOTAL_IMAGE_SOURCE_BYTES:
            raise ValueError(
                f"example image sources total {source_bytes} bytes, exceeding the "
                f"{MAX_TOTAL_IMAGE_SOURCE_BYTES}-byte limit"
            )
        descriptors.append(descriptor)

    normalized: list[dict] = []
    first_user = None
    saw_image_block = False
    for message_index, message in enumerate(messages):
        copied = dict(message)
        role = _canonical_message_role(copied.get("role"), message_index)
        copied["role"] = role
        if role == "user" and first_user is None:
            first_user = message_index
        content = copied.get("content")
        if isinstance(content, str):
            copied["content"] = [{"type": "text", "text": content}]
            normalized.append(copied)
            continue
        if content is None:
            copied["content"] = []
            normalized.append(copied)
            continue
        if not isinstance(content, list):
            raise ValueError(
                f"message {message_index} content must be text or a list of content blocks"
            )
        blocks: list[dict] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValueError(
                    f"malformed content block at message {message_index}, index {block_index}: expected an object"
                )
            block_type = block.get("type")
            if block_type in _TEXT_BLOCK_TYPES:
                if not isinstance(block.get("text"), str):
                    raise ValueError(
                        f"malformed text block at message {message_index}, index {block_index}: missing text"
                    )
                blocks.append({"type": "text", "text": block["text"]})
                continue
            if block_type not in _IMAGE_BLOCK_TYPES:
                raise ValueError(
                    f"unsupported content block type {block_type!r} at message {message_index}, index {block_index}"
                )
            if role != "user":
                raise ValueError("image blocks are allowed only in user messages")
            saw_image_block = True
            source = _image_source_from_block(block)
            if source is None:
                if not pending:
                    raise ValueError(
                        f"image placeholder at message {message_index}, index {block_index} has no matching source"
                    )
                source = pending.pop(0)
            add_descriptor(source)
            blocks.append({"type": "image"})
        copied["content"] = blocks
        normalized.append(copied)
    if pending:
        if saw_image_block:
            raise ValueError(
                f"example has {len(pending)} extra top-level image source(s) with no matching placeholder"
            )
        if first_user is None:
            raise ValueError("top-level image fields require at least one user prompt message")
        content = normalized[first_user].get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif content is None:
            content = []
        elif not isinstance(content, list):
            raise ValueError("user prompt content must be text or content blocks")
        image_blocks = []
        for source in pending:
            add_descriptor(source)
            image_blocks.append({"type": "image"})
        normalized[first_user]["content"] = [*content, *image_blocks]
    # checked here, on the finished blocks, rather than inside the loop above: string content never
    # enters that loop, and the top-level-image path appends to a message the loop already passed.
    # this is the one point every shape has converged to the same form.
    #
    # the unit is a RUN of consecutive text blocks, not one block, for the reason the teacher
    # renderer already documents: "<|image_" and "pad|>" are each harmless alone and only become
    # the reserved marker once the template concatenates them. an image block ends a run, since the
    # processor puts its own expansion there and user text cannot reach across it.
    for message_index, message in enumerate(normalized):
        text_run: list[str] = []
        for block in message["content"]:
            if block.get("type") == "text":
                text_run.append(block["text"])
                continue
            _reject_literal_image_placeholder("".join(text_run), message_index, (IMAGE_PAD_TOKEN,))
            text_run = []
        _reject_literal_image_placeholder("".join(text_run), message_index, (IMAGE_PAD_TOKEN,))
    if not defer_validation:
        validate_image_descriptors(descriptors, package_root)
    return NormalizedImages(normalized, descriptors)


def normalize_environment_reply(
    messages: list[dict],
    package_root: str | Path | None,
    cumulative_descriptors: list[str] | tuple[str, ...],
) -> NormalizedEnvironmentReply:
    """normalize one visible environment reply and validate its cumulative image context."""
    normalized = normalize_prompt_images({}, messages, package_root, defer_validation=True)
    descriptors = tuple(normalized.descriptors)
    cumulative = [*cumulative_descriptors, *descriptors]
    validated = validate_image_descriptors(cumulative, package_root)
    data_uris = tuple(item.data_uri for item in validated[len(cumulative_descriptors) :])
    messages_out = (
        normalized.messages if descriptors else text_only_prompt_messages(normalized.messages)
    )
    return NormalizedEnvironmentReply(messages_out, descriptors, data_uris)


def _descriptor_source_size(descriptor: str, package_root: str | Path | None) -> int:
    return _image_descriptors.descriptor_source_size(
        descriptor,
        package_root,
        _image_limits(),
    )


def read_descriptor_source(descriptor: str, package_root: str | Path | None) -> bytes:
    """read one descriptor's raw source bytes under the per-source size limit."""
    return _image_descriptors.read_descriptor_source(
        descriptor,
        package_root,
        _image_limits(),
    )


def validate_image_descriptor_data(
    descriptor: str, data: bytes, *, decode_pixels: bool = True
) -> ValidatedImageDescriptor:
    """validate bytes already read from one normalized descriptor."""
    return _image_descriptors.validate_image_descriptor_data(
        descriptor,
        data,
        _image_limits(),
        decode_pixels=decode_pixels,
        pixel_decoder=decode_descriptor_pixels,
    )


def validate_image_descriptor(
    descriptor: str,
    package_root: str | Path | None,
    *,
    decode_pixels: bool = True,
) -> ValidatedImageDescriptor:
    """validate one normalized descriptor and return its canonical data uri metadata."""
    return _image_descriptors.validate_image_descriptor(
        descriptor,
        package_root,
        _image_limits(),
        decode_pixels=decode_pixels,
        pixel_decoder=decode_descriptor_pixels,
    )


def validate_image_descriptors(
    descriptors: list[str] | tuple[str, ...], package_root: str | Path | None
) -> list[ValidatedImageDescriptor]:
    """validate one cumulative descriptor set under the shared image admission limits."""
    return _image_descriptors.validate_image_descriptors(
        descriptors,
        package_root,
        _image_limits(),
        pixel_decoder=decode_descriptor_pixels,
    )


def image_descriptors_to_data_uris(
    descriptors: list[str] | tuple[str, ...], package_root: str | Path | None
) -> list[str]:
    """convert normalized descriptors to canonical data uris under aggregate limits."""
    return [item.data_uri for item in validate_image_descriptors(descriptors, package_root)]


def messages_with_image_data_uris(
    messages: list[dict],
    descriptors: list[str] | tuple[str, ...],
    package_root: str | Path | None,
) -> list[dict]:
    """Bind descriptors to normalized image blocks as canonical data-URI image_url blocks."""
    uri_iter = iter(image_descriptors_to_data_uris(descriptors, package_root))
    prepared: list[dict] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                copied_block = dict(block)
                if copied_block.get("type") == "image":
                    try:
                        uri = next(uri_iter)
                    except StopIteration as exc:
                        raise ValueError(
                            "normalized messages contain more image blocks than descriptors"
                        ) from exc
                    copied_block = {"type": "image_url", "image_url": {"url": uri}}
                blocks.append(copied_block)
            copied["content"] = blocks
        prepared.append(copied)
    try:
        next(uri_iter)
    except StopIteration:
        return prepared
    raise ValueError("normalized image descriptors contain no matching image block")


def _decode_image_bytes(data: bytes):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    converted = None
    try:
        with Image.open(io.BytesIO(data)) as image:
            _validate_dimensions(image.width, image.height)
            image.load()
            ImageOps.exif_transpose(image, in_place=True)
            _validate_dimensions(image.width, image.height)
            converted = image.convert("RGB")
            converted.load()
            return converted
    except ValueError:
        if converted is not None:
            converted.close()
        raise
    except Exception as exc:
        if converted is not None:
            converted.close()
        raise ValueError("image source is not a valid image") from exc


def messages_with_decoded_images(messages: list[dict], images: list[object]) -> list[dict]:
    """Bind decoded images into a message list's image blocks, in order.

    Shared by sft, opd, and grpo: each algorithm decodes the row's descriptors itself and then
    needs the same binding into the chat-template input.
    """
    image_iter = iter(images)
    prepared = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                copied_block = dict(block)
                if copied_block.get("type") == "image":
                    copied_block["image"] = next(image_iter)
                blocks.append(copied_block)
            copied["content"] = blocks
        prepared.append(copied)
    try:
        next(image_iter)
    except StopIteration:
        return prepared
    raise ValueError("unused decoded image while preparing multimodal sft tokens")


def decode_descriptor_pixels(data: bytes) -> None:
    """Fully decode one image's pixels for validation, then release it.

    Header inspection accepts an image whose pixel data is truncated or corrupt; only a full decode
    finds that, and the caller wants the failure, not the image.
    """
    image = _decode_image_bytes(data)
    image.close()


def decode_image_descriptors(
    descriptors: list[str], package_root: str | Path | None
) -> list[object]:
    """decode one row to fully loaded rgb images under aggregate limits."""
    if len(descriptors) > MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"example contains {len(descriptors)} images, exceeding the {MAX_IMAGES_PER_EXAMPLE}-image limit"
        )
    validated = validate_image_descriptors(descriptors, package_root)
    prepared = [item.data for item in validated]
    images: list[Any] = []
    try:
        images.extend(_decode_image_bytes(data) for data in prepared)
    except Exception:
        for image in images:
            image.close()
        raise
    return images


def resolve_image_pad_token_id(processor, tok) -> int:
    """Resolve the Qwen VL image-pad token id without assuming a model-specific constant."""

    def valid_token_id(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            return None
        return token_id if token_id >= 0 else None

    token_id = valid_token_id(getattr(processor, "image_token_id", None))
    if token_id is not None:
        return token_id

    convert = getattr(tok, "convert_tokens_to_ids", None)
    if callable(convert):
        image_token = getattr(processor, "image_token", None)
        candidates = [image_token] if isinstance(image_token, str) else []
        candidates.append("<|image_pad|>")
        for token in candidates:
            try:
                token_id = valid_token_id(convert(token))
            except Exception:
                token_id = None
            if token_id is not None:
                return token_id

    raise ValueError("could not resolve a valid image-pad token id from the processor or tokenizer")


def validate_multimodal_training(model_id: str, algorithm: str, teacher_model: str | None) -> None:
    from flash.core.catalog import supports_image_training
    from flash.engine.plan.recipe import resolve_teacher, teacher_supports_images

    if not supports_image_training(model_id):
        raise ValueError(f"{model_id} does not support image-bearing training records")
    if algorithm == "opd":
        teacher = resolve_teacher(teacher_model)
        if not teacher_supports_images(teacher.alias):
            from flash.engine.plan.recipe import image_capable_teacher_aliases

            choices = " or ".join(f'"{alias}"' for alias in image_capable_teacher_aliases())
            raise ValueError(
                f"image-bearing opd requires [train] teacher_model = {choices}; "
                f"the selected teacher {teacher.alias!r} cannot see images"
            )


def validate_image_observation_environment(env: object, spec: object) -> None:
    """Require image-capable training inputs for an env that can emit image observations."""
    if not bool(getattr(env, "image_observations", False)):
        return
    train = getattr(spec, "train", None)
    validate_multimodal_training(
        str(getattr(spec, "model", "")),
        str(getattr(spec, "algorithm", "")),
        getattr(train, "teacher_model", None),
    )


def message_content_text(content: object) -> str:
    """The text of one message's ``content``: the string itself, or its openai-style text blocks.

    Any other shape (null tool-call content, image-only blocks) yields ``""``. This is the single
    definition of "what the text of a message is" -- graders, replay, and reward all read a
    completion through it, so an image-only or tool-call turn cannot mean one thing in one path and
    something else in another."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in _TEXT_BLOCK_TYPES
        )
    return ""


def _record_messages(record: dict) -> list[dict]:
    value = record.get("input")
    if isinstance(value, list) and all(isinstance(message, dict) for message in value):
        return value
    return []


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid packaged dataset JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"packaged dataset row {line_number} must be an object")
                rows.append(row)
        return rows
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid packaged dataset JSON at {path}") from exc
    if isinstance(value, dict):
        value = value.get("records", value.get("data"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"packaged dataset at {path} must contain a list of objects")
    return value


def preflight_validate_image_opd(spec, *, scan_packaged_environment: bool = True) -> None:
    """validate statically discoverable image opd datasets before gpu allocation."""
    if getattr(spec, "algorithm", "") != "opd":
        return
    environment = getattr(spec, "environment", None)
    params = dict(getattr(environment, "params", None) or {})
    records = params.get("records")
    if records is not None:
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            return
    else:
        if not scan_packaged_environment:
            return
        from flash.envs.loading.loader import (
            _packaged_dataset_file,
            _resolve_environment_reference,
            _resolve_path_arg,
            _validate_packaged_dataset_split,
        )

        env = spec.environment
        reference = _resolve_environment_reference(env.id, env.resolved_sha or None)
        reference_path = Path(reference)
        base_dir = reference_path.parent if reference_path.exists() else Path.cwd()
        source = params.get("dataset_path")
        if source:
            source = _resolve_path_arg(source, base_dir)
            path = Path(source) if isinstance(source, str) else None
        else:
            split = params.get("split")
            split = split.strip() if isinstance(split, str) else "train"
            split = _validate_packaged_dataset_split(split or "train")
            path = _packaged_dataset_file(base_dir, split)
        if path is None or not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            return
        records = _read_json_records(path)
    train = getattr(spec, "train", None)
    max_examples = int(getattr(train, "max_examples", 0) or 0)
    if max_examples > 0:
        records = records[:max_examples]
    for record in records:
        if record_has_images(record, _record_messages(record)):
            validate_multimodal_training(
                str(getattr(spec, "model", "")),
                "opd",
                getattr(train, "teacher_model", None),
            )
            return
