"""Image normalization and bounded lazy decoding for managed training."""

from __future__ import annotations

import base64
import binascii
import http.client
import io
import ipaddress
import json
import os
import socket
import ssl
import urllib.parse
from collections.abc import Callable
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
REMOTE_IMAGE_ENV = "FLASH_MULTIMODAL_ALLOW_REMOTE_IMAGES"
_REMOTE_TIMEOUT_SECONDS = 10.0
_IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})


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


def _remote_enabled() -> bool:
    return os.environ.get(REMOTE_IMAGE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _validate_remote_url(url: str) -> tuple[urllib.parse.SplitResult, str, int, tuple[str, ...]]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("remote image URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("remote image URLs must not include credentials")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("remote image URL must include a hostname")
    try:
        hostname = hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if scheme == "https" else 80)
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"failed to resolve remote image host: {hostname}") from exc
    addresses = []
    for family, _socktype, _proto, _canonname, sockaddr in resolved:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            raise ValueError(f"remote image host resolved to an unsupported address family: {hostname}")
        address = str(sockaddr[0]).split("%", 1)[0]
        try:
            globally_routable = ipaddress.ip_address(address).is_global
        except ValueError as exc:
            raise ValueError(f"remote image host resolved to an invalid address: {hostname}") from exc
        if not globally_routable:
            raise ValueError(
                f"remote image host must resolve only to globally routable addresses: {hostname}"
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ValueError(f"remote image host did not resolve to an IP address: {hostname}")
    return parsed, hostname, port, tuple(addresses)


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


def _inspect_image_bytes(data: bytes) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            return _validate_dimensions(image.width, image.height)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc


def _pil_descriptor(image: object) -> str:
    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    _validate_dimensions(width, height)
    out = io.BytesIO()
    image.save(out, format="PNG")
    data = out.getvalue()
    _check_source_size(len(data))
    return _descriptor("bytes", base64.b64encode(data).decode("ascii"))


def normalize_image_source(source: object, package_root: str | Path | None) -> str:
    """Convert one supported image source to an Arrow-safe descriptor."""
    root = Path(package_root) if package_root is not None else None
    if isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, bytes):
        _check_source_size(len(source))
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
        _inspect_image_bytes(data)
        return _descriptor("data_uri", value)
    if scheme in {"http", "https"}:
        if not _remote_enabled():
            raise ValueError(
                f"remote image URLs are disabled; set {REMOTE_IMAGE_ENV}=1 only for a trusted dataset"
            )
        data = _read_remote(value)
        _inspect_image_bytes(data)
        return _descriptor("bytes", base64.b64encode(data).decode("ascii"))
    if scheme == "file":
        raise ValueError("file:// image URLs are not supported")
    if scheme:
        raise ValueError(f"unsupported image URL scheme: {scheme}")
    path = _package_image_path(value, root)
    _inspect_image_bytes(path.read_bytes())
    return _descriptor("path", path.relative_to(root.resolve()).as_posix())


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
            raise ValueError(f"malformed {block_type} block: expected image URL object with a url field")
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
            isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES
            for block in content
        ):
            return True
    return False


def normalize_prompt_images(
    record: dict,
    messages: list[dict],
    package_root: str | Path | None,
) -> NormalizedImages:
    """Normalize prompt image blocks and top-level fields without mutating the record."""
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
        descriptor = normalize_image_source(source, package_root)
        descriptor_bytes += _descriptor_size(descriptor)
        if descriptor_bytes > MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES:
            raise ValueError(
                f"example encoded image descriptors total {descriptor_bytes} bytes, exceeding the "
                f"{MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES}-byte limit"
            )
        source_bytes += descriptor_source_size(descriptor, package_root)
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


def descriptor_source_size(descriptor: str, package_root: str | Path | None) -> int:
    kind, value = _parse_descriptor(descriptor)
    if kind == "bytes":
        return len(base64.b64decode(value, validate=True))
    if kind == "data_uri":
        return len(_data_uri_bytes(value))
    if kind == "path":
        return _package_image_path(value, Path(package_root) if package_root else None).stat().st_size
    raise ValueError("invalid internal image descriptor kind")


def _remote_host_header(hostname: str, port: int, scheme: str) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if scheme == "https" else 80
    return host if port == default_port else f"{host}:{port}"


def _fetch_remote_address(
    parsed: urllib.parse.SplitResult, hostname: str, port: int, address: str
) -> bytes:
    connection = http.client.HTTPConnection(address, port, timeout=_REMOTE_TIMEOUT_SECONDS)
    try:
        if parsed.scheme.lower() == "https":
            raw_socket = socket.create_connection((address, port), timeout=_REMOTE_TIMEOUT_SECONDS)
            try:
                connection.sock = ssl.create_default_context().wrap_socket(
                    raw_socket, server_hostname=hostname
                )
            except Exception:
                raw_socket.close()
                raise
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET",
            target,
            headers={
                "Host": _remote_host_header(hostname, port, parsed.scheme.lower()),
                "User-Agent": "freesolo-flash-image-loader",
            },
        )
        response = connection.getresponse()
        try:
            if 300 <= response.status < 400:
                raise ValueError("remote image redirects are not allowed")
            if response.status < 200 or response.status >= 300:
                raise ValueError(f"remote image request failed with HTTP {response.status}")
            declared = response.headers.get("Content-Length")
            if declared:
                _check_source_size(int(declared))
            data = response.read(MAX_IMAGE_SOURCE_BYTES + 1)
        finally:
            response.close()
    finally:
        connection.close()
    _check_source_size(len(data))
    return data


def _read_remote(url: str) -> bytes:
    parsed, hostname, port, addresses = _validate_remote_url(url)
    last_error = None
    for address in addresses:
        try:
            return _fetch_remote_address(parsed, hostname, port, address)
        except ValueError:
            raise
        except Exception as exc:
            last_error = exc
    raise ValueError(f"failed to fetch remote image: {url}") from last_error


def _read_descriptor_source(descriptor: str, package_root: str | Path | None) -> bytes:
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


def _decode_image_bytes(data: bytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            _validate_dimensions(image.width, image.height)
            image.load()
            return image.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc


def decode_image_descriptor(descriptor: str, package_root: str | Path | None):
    """Decode and validate one descriptor at batch access time."""
    data = _read_descriptor_source(descriptor, package_root)
    decoded_bytes = _inspect_image_bytes(data)
    return _decode_image_bytes(data), len(data), decoded_bytes


def decode_image_descriptors(
    descriptors: list[str], package_root: str | Path | None
) -> list[object]:
    if len(descriptors) > MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"example contains {len(descriptors)} images, exceeding the {MAX_IMAGES_PER_EXAMPLE}-image limit"
        )
    prepared: list[bytes] = []
    source_bytes = 0
    decoded_bytes = 0
    for descriptor in descriptors:
        data = _read_descriptor_source(descriptor, package_root)
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
    return [_decode_image_bytes(data) for data in prepared]


def decode_arrow_examples(examples: list[dict], package_root: str | Path | None) -> list[dict]:
    decoded = []
    for example in examples:
        row = dict(example)
        row["images"] = decode_image_descriptors(list(row.get("images") or []), package_root)
        decoded.append(row)
    return decoded


def filter_vlm_sft_rows(
    rows: list[dict], collator: Callable[[list[dict]], dict], special_ids: set[int]
) -> tuple[list[dict], int, int, int]:
    """Filter empty or fully truncated targets using the processor-expanded labels."""
    kept = []
    dropped = 0
    masked_tokens = 0
    total_tokens = 0
    for row in rows:
        batch = collator([row])
        labels = batch["labels"][0]
        attention = batch.get("attention_mask")
        labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        if attention is None:
            active = [1] * len(labels_list)
        else:
            mask = attention[0]
            active = mask.tolist() if hasattr(mask, "tolist") else list(mask)
        has_target = any(
            is_active and label != -100 and label not in special_ids
            for label, is_active in zip(labels_list, active, strict=True)
        )
        if not has_target:
            dropped += 1
            continue
        kept.append(row)
        total_tokens += sum(bool(value) for value in active)
        masked_tokens += sum(
            bool(is_active) and label == -100
            for label, is_active in zip(labels_list, active, strict=True)
        )
    return kept, dropped, masked_tokens, total_tokens


class ArrowSafeVisionCollator:
    """Decode Arrow-safe descriptors just before TRL vision collation."""

    def __init__(
        self,
        processor,
        max_length: int,
        package_root: str | Path | None,
        delegate: Callable[[list[dict]], dict] | None = None,
    ):
        if delegate is None:
            from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

            delegate = DataCollatorForVisionLanguageModeling(
                processor=processor,
                max_length=max_length,
                completion_only_loss=True,
            )
        self._delegate = delegate
        self._package_root = package_root

    def __call__(self, examples: list[dict]) -> dict:
        return self._delegate(decode_arrow_examples(examples, self._package_root))


def lazy_image_dataset_transform(package_root: str | Path | None) -> Callable[[dict], dict]:
    """Build a Hugging Face dataset transform that decodes only requested rows."""

    def transform(batch: dict) -> dict:
        image_rows = batch.get("images")
        if image_rows is None:
            return batch
        batch = dict(batch)
        batch["images"] = [decode_image_descriptors(list(row or []), package_root) for row in image_rows]
        return batch

    return transform


def processor_prompt_token_count(
    processor,
    messages: list[dict],
    descriptors: list[str],
    package_root: str | Path | None,
    *,
    tools: list | None = None,
    enable_thinking: bool = False,
) -> int:
    """Count the exact processor-expanded prompt tokens used by TRL GRPO."""
    from trl.data_utils import prepare_multimodal_messages

    images = decode_image_descriptors(descriptors, package_root)
    prompt = prepare_multimodal_messages(messages, images=images)
    result = processor.apply_chat_template(
        conversation=[prompt],
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=enable_thinking,
    )
    input_ids = result["input_ids"]
    if input_ids and isinstance(input_ids[0], int):
        return len(input_ids)
    return len(input_ids[0])


def validate_multimodal_training(model_id: str, algorithm: str, *, multi_turn: bool = False) -> None:
    from flash.catalog import supports_image_training

    if not supports_image_training(model_id):
        raise ValueError(f"{model_id} does not support image-bearing training records")
    if algorithm == "opd":
        raise ValueError("opd does not support image-bearing records; use sft or single-turn grpo")
    if algorithm == "grpo" and multi_turn:
        raise ValueError("multimodal multi-turn grpo is not supported; use single-turn grpo or image sft")


def assistant_completion_text(completion: object) -> str:
    if not isinstance(completion, list):
        return str(completion or "")
    for message in reversed(completion):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(block.get("text") or "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
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


def preflight_reject_image_opd(spec) -> None:
    """Reject statically discoverable image OPD datasets before GPU allocation."""
    if getattr(spec, "algorithm", "") != "opd":
        return
    params = dict(getattr(getattr(spec, "environment", None), "params", None) or {})
    records = params.get("records")
    if records is not None:
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            return
    else:
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
    for index, record in enumerate(records):
        if record_has_images(record, _record_messages(record)):
            raise ValueError(
                f"opd does not support image-bearing records (found an image in dataset row {index}); "
                "use sft or single-turn grpo"
            )
