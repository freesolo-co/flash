"""Image normalization and bounded lazy decoding for managed training."""

from __future__ import annotations

import base64
import binascii
import io
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

MAX_IMAGES_PER_EXAMPLE = 8
MAX_IMAGE_SOURCE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_IMAGE_SOURCE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_WIDTH = 8192
MAX_IMAGE_HEIGHT = 8192
MAX_IMAGE_PIXELS = 40_000_000
MAX_TOTAL_DECODED_BYTES = 128 * 1024 * 1024
MAX_DATA_URI_HEADER_BYTES = 1024
MAX_IMAGE_DESCRIPTOR_BYTES = 64 * 1024 * 1024
MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES = 64 * 1024 * 1024
_IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})
IMAGE_TEACHER_PLACEHOLDER = "<|media_pad|>"
# the model's real image-expansion token, as opposed to the placeholder above. one definition so
# the renderer's rejection and the teacher client's drop guard cannot drift apart.
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass(frozen=True)
class NormalizedImages:
    messages: list[dict]
    descriptors: list[str]


def _is_pil_image(value: object) -> bool:
    try:
        from PIL import Image

        return isinstance(value, Image.Image)
    except ImportError:
        return False


def _descriptor_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _check_descriptor_size(descriptor: str) -> None:
    size = _descriptor_size(descriptor)
    if size > MAX_IMAGE_DESCRIPTOR_BYTES:
        raise ValueError(
            f"encoded image descriptor exceeds the {MAX_IMAGE_DESCRIPTOR_BYTES}-byte limit "
            f"({size} bytes)"
        )


def _descriptor(kind: str, value: str) -> str:
    descriptor = json.dumps({"kind": kind, "value": value}, separators=(",", ":"), sort_keys=True)
    _check_descriptor_size(descriptor)
    return descriptor


def _parse_descriptor(raw: str) -> tuple[str, str]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid internal image descriptor") from exc
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        raise ValueError("invalid internal image descriptor")
    kind = value["kind"]
    payload = value["value"]
    if not isinstance(kind, str) or not isinstance(payload, str):
        raise ValueError("invalid internal image descriptor")
    return kind, payload


def _data_uri_bytes(uri: str) -> bytes:
    comma = uri.find(",")
    if comma < 0:
        raise ValueError("image data URI must use an image media type")
    header = uri[:comma]
    if len(header.encode("utf-8")) > MAX_DATA_URI_HEADER_BYTES:
        raise ValueError(
            f"image data URI header exceeds the {MAX_DATA_URI_HEADER_BYTES}-byte limit"
        )
    if not header.lower().startswith("data:image/"):
        raise ValueError("image data URI must use an image media type")
    payload = uri[comma + 1 :]
    try:
        if ";base64" in header.lower():
            max_encoded_size = ((MAX_IMAGE_SOURCE_BYTES + 2) // 3) * 4
            if len(payload) > max_encoded_size:
                _check_source_size(MAX_IMAGE_SOURCE_BYTES + 1)
            data = base64.b64decode(payload, validate=True)
        else:
            if len(payload) > MAX_IMAGE_SOURCE_BYTES * 3:
                _check_source_size(MAX_IMAGE_SOURCE_BYTES + 1)
            data = urllib.parse.unquote_to_bytes(payload)
    except binascii.Error as exc:
        raise ValueError("malformed image data URI") from exc
    _check_source_size(len(data))
    return data


def _check_source_size(size: int) -> None:
    if size > MAX_IMAGE_SOURCE_BYTES:
        raise ValueError(
            f"image source exceeds the {MAX_IMAGE_SOURCE_BYTES}-byte limit ({size} bytes)"
        )


def _package_image_path(value: str, package_root: Path | None) -> Path:
    if package_root is None:
        raise ValueError("relative image paths require an extracted environment package root")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("image paths must be relative to the environment package root")
    root = package_root.resolve()
    dataset_root = (root / "dataset").resolve()
    resolved = (root / candidate).resolve()
    if resolved != dataset_root and dataset_root not in resolved.parents:
        raise ValueError("image paths must stay inside the environment package dataset/ directory")
    if not resolved.is_file():
        raise ValueError(f"image path does not exist: {value!r}")
    _check_source_size(resolved.stat().st_size)
    return resolved


def _validate_dimensions(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise ValueError(
            f"image dimensions {width}x{height} exceed the {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT} limit"
        )
    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(f"image has {pixels} pixels, exceeding the {MAX_IMAGE_PIXELS}-pixel limit")
    return pixels * 3


def inspect_image_bytes(data: bytes) -> tuple[int, int, int]:
    """Validate one image's header and return its (decoded_rgb_bytes, width, height).

    Reads the header only: the pixels are not decoded here.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            return (
                _validate_dimensions(image.width, image.height),
                image.width,
                image.height,
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc


def _inspect_image_bytes(data: bytes) -> int:
    return inspect_image_bytes(data)[0]


def _pil_descriptor(image: object) -> str:
    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    _validate_dimensions(width, height)
    out = io.BytesIO()
    image.save(out, format="PNG")
    data = out.getvalue()
    _check_source_size(len(data))
    return _descriptor("bytes", base64.b64encode(data).decode("ascii"))


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
        if not defer_validation:
            _inspect_image_bytes(source)
        return _descriptor("bytes", base64.b64encode(source).decode("ascii"))
    if _is_pil_image(source):
        return _pil_descriptor(source)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("image source must be a data URI, bytes, PIL image, relative path, or URL")
    value = source.strip()
    parsed = urllib.parse.urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme == "data":
        data = _data_uri_bytes(value)
        if not defer_validation:
            _inspect_image_bytes(data)
        return _descriptor("data_uri", value)
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
    path = _package_image_path(value, root)
    descriptor = _descriptor("path", path.relative_to(root.resolve()).as_posix())
    if not defer_validation:
        _inspect_image_bytes(path.read_bytes())
    return descriptor


def _image_source_from_block(block: dict) -> object | None:
    block_type = block.get("type")
    if block_type == "image":
        value = next(
            (block[key] for key in ("image", "source", "image_url", "url") if key in block),
            None,
        )
    elif block_type == "image_url":
        value = block.get("image_url", block.get("url"))
    elif block_type == "input_image":
        value = block.get("image_url", block.get("image", block.get("url")))
    else:
        return None
    if isinstance(value, dict):
        extra = set(value) - {"url", "detail"}
        if extra or "url" not in value:
            raise ValueError(
                f"malformed {block_type} block: expected image URL object with a url field"
            )
        value = value["url"]
    if value is None and block_type != "image":
        raise ValueError(f"malformed {block_type} block: missing image source")
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
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES for block in content
        ):
            return True
    return False


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
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        stripped.append(copied)
    return stripped


def _reject_literal_image_placeholder(text: str, message_index: int) -> None:
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
    """
    for marker in (IMAGE_TEACHER_PLACEHOLDER, IMAGE_PAD_TOKEN):
        if marker in text:
            raise ValueError(
                f"message {message_index} text contains the reserved image marker "
                f"{marker!r}; remove it from the prompt"
            )


def image_teacher_prompt_messages(messages: list[dict], descriptor_count: int) -> list[dict]:
    """Render normalized image blocks as Kimi media placeholders without changing text order."""
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
                if block_type == "text" and isinstance(block.get("text"), str):
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
            defer_validation=defer_validation,
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
        if str(copied.get("role") or "").lower() == "user" and first_user is None:
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
            if block_type == "text":
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
    return NormalizedImages(normalized, descriptors)


def _descriptor_source_size(descriptor: str, package_root: str | Path | None) -> int:
    kind, value = _parse_descriptor(descriptor)
    if kind == "bytes":
        return len(base64.b64decode(value, validate=True))
    if kind == "data_uri":
        return len(_data_uri_bytes(value))
    if kind == "path":
        return (
            _package_image_path(value, Path(package_root) if package_root else None).stat().st_size
        )
    raise ValueError("invalid internal image descriptor kind")


def read_descriptor_source(descriptor: str, package_root: str | Path | None) -> bytes:
    """Read one descriptor's raw source bytes under the per-source size limit."""
    kind, value = _parse_descriptor(descriptor)
    if kind == "bytes":
        try:
            data = base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid internal byte image descriptor") from exc
    elif kind == "data_uri":
        data = _data_uri_bytes(value)
    elif kind == "path":
        data = _package_image_path(value, Path(package_root) if package_root else None).read_bytes()
    else:
        raise ValueError("invalid internal image descriptor kind")
    _check_source_size(len(data))
    return data


def _base64_image_data_uri(data: bytes) -> tuple[str, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            decoded_bytes = _validate_dimensions(image.width, image.height)
            media_type = Image.MIME.get(str(image.format or "").upper())
            if not media_type:
                media_type = image.get_format_mimetype()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc
    if not isinstance(media_type, str) or not media_type.startswith("image/"):
        raise ValueError("image source has an unsupported media type")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}", decoded_bytes


def image_descriptors_to_data_uris(
    descriptors: list[str] | tuple[str, ...], package_root: str | Path | None
) -> list[str]:
    """Convert normalized descriptors to image data URIs under existing aggregate limits."""
    if len(descriptors) > MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"example contains {len(descriptors)} images, exceeding the "
            f"{MAX_IMAGES_PER_EXAMPLE}-image limit"
        )
    uris: list[str] = []
    source_bytes = 0
    decoded_bytes = 0
    for descriptor in descriptors:
        data = read_descriptor_source(descriptor, package_root)
        source_bytes += len(data)
        if source_bytes > MAX_TOTAL_IMAGE_SOURCE_BYTES:
            raise ValueError(
                f"example image sources exceed the {MAX_TOTAL_IMAGE_SOURCE_BYTES}-byte limit"
            )
        uri, image_decoded_bytes = _base64_image_data_uri(data)
        decoded_bytes += image_decoded_bytes
        if decoded_bytes > MAX_TOTAL_DECODED_BYTES:
            raise ValueError(
                f"example decoded images exceed the {MAX_TOTAL_DECODED_BYTES}-byte limit"
            )
        uris.append(uri)
    return uris


def _decode_image_bytes(data: bytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    converted = None
    try:
        with Image.open(io.BytesIO(data)) as image:
            _validate_dimensions(image.width, image.height)
            image.load()
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
    prepared: list[bytes] = []
    source_bytes = 0
    decoded_bytes = 0
    for descriptor in descriptors:
        data = read_descriptor_source(descriptor, package_root)
        source_bytes += len(data)
        if source_bytes > MAX_TOTAL_IMAGE_SOURCE_BYTES:
            raise ValueError(
                f"example image sources exceed the {MAX_TOTAL_IMAGE_SOURCE_BYTES}-byte limit"
            )
        decoded_bytes += _inspect_image_bytes(data)
        if decoded_bytes > MAX_TOTAL_DECODED_BYTES:
            raise ValueError(
                f"example decoded images exceed the {MAX_TOTAL_DECODED_BYTES}-byte limit"
            )
        prepared.append(data)
    images = []
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
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def assistant_completion_text(completion: object) -> str:
    if not isinstance(completion, list):
        return str(completion or "")
    for message in reversed(completion):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            # a content shape with no text at all (null tool-call content) keeps scanning earlier
            # assistant turns rather than resolving to "" here.
            if isinstance(content, (str, list)):
                return message_content_text(content)
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
        from flash.envs.loader import (
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
            # only `multi_turn` is a preflight signal. the authoritative source is the env CLASS
            # (adapter.py sets multi_turn from isinstance(sdk_env, EnvironmentMultiTurn)), which
            # this path deliberately cannot see: it reads the dataset file rather than importing
            # user environment code before allocation. so a class-based multi-turn env is caught
            # by the worker's fail-closed guard instead. `max_turns` must NOT reject here -- it is
            # a turn CAP that single-turn envs may legitimately carry, and rejecting on it fails
            # jobs the worker would happily run.
            if params.get("multi_turn") is True:
                raise ValueError("multi-turn image-bearing opd is not supported")
            return
