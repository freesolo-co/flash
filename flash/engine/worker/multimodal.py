"""Multimodal training-row helpers shared by SFT and GRPO.

Flash environments expose OpenAI-style chat messages. This module normalizes image-bearing
messages into the shape TRL's VLM trainers understand: chat content keeps lightweight
``{"type": "image"}`` placeholders, while the actual PIL image objects live in ``image`` or
``images`` dataset columns.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from flash.catalog import supports_multimodal

_IMAGE_BLOCK_TYPES = {"image", "input_image", "image_url"}
_TEXT_BLOCK_TYPES = {"text", "input_text"}
_DEFAULT_IMAGE_MAX_BYTES = 20 * 1024 * 1024
_DEFAULT_IMAGE_MAX_PIXELS = 64_000_000
_DEFAULT_IMAGE_TOKEN_RESERVE = 1024


def model_supports_images(model_id: str) -> bool:
    return supports_multimodal(model_id)


def _image_values_from_example(example: dict | None) -> list[Any]:
    if not isinstance(example, dict):
        return []
    out: list[Any] = []
    if example.get("image") is not None:
        out.append(example["image"])
    images = example.get("images")
    if images is not None:
        if isinstance(images, (str, bytes, bytearray)) or not isinstance(images, Iterable):
            out.append(images)
        else:
            out.extend(list(images))
    return [v for v in out if v is not None]


def _block_image_payload(block: dict) -> Any | None:
    if block.get("image") is not None:
        return block["image"]
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    if image_url is not None:
        return image_url
    if block.get("url") is not None:
        return block["url"]
    return None


def _base_dirs_from_env() -> tuple[Path, ...]:
    raw = os.environ.get("FLASH_IMAGE_BASE_DIR", "")
    dirs = [Path.cwd()]
    dirs.extend(Path(part.strip()) for part in raw.split(os.pathsep) if part.strip())
    return tuple(dirs)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _image_max_bytes() -> int:
    return _env_int("FLASH_IMAGE_MAX_BYTES", _DEFAULT_IMAGE_MAX_BYTES, minimum=1)


def image_token_reserve() -> int:
    return _env_int("FLASH_IMAGE_TOKEN_RESERVE", _DEFAULT_IMAGE_TOKEN_RESERVE, minimum=0)


def multimodal_token_estimate(text: str, tokenizer, image_count: int) -> int:
    """Conservative context estimate for VLM prompts.

    Model processors differ in whether image patch tokens are visible through ``input_ids`` before
    collation/rollout. Reserve a fixed budget per image so prompt filtering and cost estimates do
    not count image rows as text-only.
    """
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = getattr(tokenized, "input_ids", None)
    if input_ids is None and isinstance(tokenized, dict):
        input_ids = tokenized.get("input_ids")
    text_tokens = len(input_ids or [])
    return text_tokens + max(0, int(image_count)) * image_token_reserve()


def _configure_image_limits(Image) -> None:
    Image.MAX_IMAGE_PIXELS = _env_int(
        "FLASH_IMAGE_MAX_PIXELS", _DEFAULT_IMAGE_MAX_PIXELS, minimum=1
    )


def _check_image_size(size: int | None, *, label: str) -> None:
    if size is not None and size > _image_max_bytes():
        raise ValueError(f"{label} exceeds FLASH_IMAGE_MAX_BYTES={_image_max_bytes()}")


def _read_limited_response(resp) -> bytes:
    content_length = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
    if content_length:
        try:
            declared = int(content_length)
        except (TypeError, ValueError):
            pass
        else:
            _check_image_size(declared, label="remote image")
    limit = _image_max_bytes()
    data = bytearray()
    while True:
        chunk = resp.read(min(1024 * 1024, limit + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError(f"remote image exceeds FLASH_IMAGE_MAX_BYTES={limit}")
    return bytes(data)


def _allowed_base_dirs(base_dirs: tuple[Path, ...] | None = None) -> tuple[Path, ...]:
    return tuple(
        Path(base).expanduser().resolve(strict=False) for base in (base_dirs or _base_dirs_from_env())
    )


def _is_under_base(path: Path, bases: tuple[Path, ...]) -> bool:
    try:
        return any(path == base or path.is_relative_to(base) for base in bases)
    except AttributeError:  # pragma: no cover - Python 3.11+ in CI/worker
        return any(path == base or base in path.parents for base in bases)


def _local_image_paths(value: str, *, base_dirs: tuple[Path, ...] | None = None) -> list[Path]:
    parsed = urllib.parse.urlparse(value)
    candidate = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme == "file" else Path(value)
    bases = _allowed_base_dirs(base_dirs)
    paths = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if not _is_under_base(resolved, bases):
            continue
        out.append(resolved)
    return out


def _load_image(value: Any, *, base_dirs: tuple[Path, ...] | None = None):
    """Load a PIL image from a path, URL, data URI, raw bytes, or pass through a PIL image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - worker image carries pillow
        raise RuntimeError(
            "image-bearing training rows require Pillow; install pillow in the worker image "
            "or add it to the environment pip requirements"
        ) from exc
    _configure_image_limits(Image)

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        _check_image_size(len(value), label="image bytes")
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if hasattr(value, "read"):
        return Image.open(value).convert("RGB")
    if not isinstance(value, str):
        raise TypeError(f"unsupported image value {type(value).__name__}; expected path, URL, bytes, or PIL.Image")

    text = value.strip()
    if not text:
        raise ValueError("empty image reference")
    if text.startswith("data:image/"):
        _head, _sep, payload = text.partition(",")
        if not _sep:
            raise ValueError("invalid data URI image reference")
        data = base64.b64decode(payload)
        _check_image_size(len(data), label="data URI image")
        return Image.open(io.BytesIO(data)).convert("RGB")

    parsed = urllib.parse.urlparse(text)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(text, timeout=60.0) as resp:
            return Image.open(io.BytesIO(_read_limited_response(resp))).convert("RGB")
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"unsupported image URL scheme {parsed.scheme!r}")

    paths = _local_image_paths(text, base_dirs=base_dirs)
    for path in paths:
        if path.is_file():
            _check_image_size(path.stat().st_size, label=f"image file {path}")
            return Image.open(path).convert("RGB")
    searched = ", ".join(str(p) for p in paths)
    allowed = ", ".join(str(p) for p in _allowed_base_dirs(base_dirs))
    raise FileNotFoundError(
        f"image file not found or outside FLASH_IMAGE_BASE_DIR: {text!r} "
        f"(searched {searched or '<none>'}; allowed roots {allowed})"
    )


def _load_images(values: Iterable[Any], *, base_dirs: tuple[Path, ...] | None = None) -> list:
    return [_load_image(v, base_dirs=base_dirs) for v in values]


def has_image_input(messages: list[dict] | None = None, example: dict | None = None) -> bool:
    return bool(image_input_count(messages, example))


def image_input_count(messages: list[dict] | None = None, example: dict | None = None) -> int:
    top_level = len(_image_values_from_example(example))
    block_count = 0
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and str(block.get("type") or "").lower() in _IMAGE_BLOCK_TYPES:
                    block_count += 1
    return block_count or top_level


def _normalize_content(content: Any, *, block_payloads: list[Any], empty_image_blocks: list[int]) -> Any:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    out = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        kind = str(block.get("type") or "").lower()
        if kind in _IMAGE_BLOCK_TYPES:
            payload = _block_image_payload(block)
            if payload is None:
                empty_image_blocks.append(len(out))
            else:
                block_payloads.append(payload)
            out.append({"type": "image"})
            continue
        if kind in _TEXT_BLOCK_TYPES and "text" in block:
            out.append({"type": "text", "text": str(block.get("text") or "")})
            continue
        out.append(dict(block))
    return out


def normalize_messages_with_images(
    messages: list[dict],
    example: dict | None = None,
    *,
    base_dirs: tuple[Path, ...] | None = None,
) -> tuple[list[dict], list]:
    """Return ``(messages, images)`` with image placeholders and loaded PIL images.

    If an example carries top-level ``image``/``images`` but its messages do not include an image
    placeholder, the image placeholders are prepended to the first user message. That keeps simple
    text environments compatible with image JSONL rows.
    """
    top_values = _image_values_from_example(example)
    block_values: list[Any] = []
    empty_image_blocks: list[int] = []
    normalized: list[dict] = []
    for message in messages:
        m = dict(message)
        m["content"] = _normalize_content(
            m.get("content"), block_payloads=block_values, empty_image_blocks=empty_image_blocks
        )
        normalized.append(m)

    placeholder_count = len(block_values) + len(empty_image_blocks)
    values: list[Any]
    if placeholder_count:
        values = list(block_values)
        missing = placeholder_count - len(values)
        if missing:
            if len(top_values) < missing:
                raise ValueError(
                    f"message has {missing} image placeholder(s) without matching top-level image data"
                )
            values.extend(top_values[:missing])
    else:
        values = list(top_values)
        if values:
            placeholders = [{"type": "image"} for _ in values]
            injected = False
            for message in normalized:
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        message["content"] = [*placeholders, *content]
                    else:
                        message["content"] = [*placeholders, {"type": "text", "text": str(content or "")}]
                    injected = True
                    break
            if not injected and normalized:
                content = normalized[0].get("content")
                normalized[0]["content"] = [*placeholders, {"type": "text", "text": str(content or "")}]

    return normalized, _load_images(values, base_dirs=base_dirs)


def multimodal_sft_row(
    prompt_messages: list[dict],
    completion_messages: list[dict],
    example: dict,
    *,
    base_dirs: tuple[Path, ...] | None = None,
) -> dict:
    prompt, prompt_images = normalize_messages_with_images(prompt_messages, example, base_dirs=base_dirs)
    completion, completion_images = normalize_messages_with_images(
        completion_messages, {}, base_dirs=base_dirs
    )
    images = [*prompt_images, *completion_images]
    return {"prompt": prompt, "completion": completion, "images": images}


def multimodal_grpo_prompt_row(
    prompt_messages: list[dict],
    example: dict,
    *,
    base_dirs: tuple[Path, ...] | None = None,
) -> dict:
    prompt, images = normalize_messages_with_images(prompt_messages, example, base_dirs=base_dirs)
    return {"prompt": prompt, "example": example, "images": images}


def message_text(messages: list[dict]) -> str:
    """Lossy text rendering used only for token-budget estimates and warnings."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    kind = str(block.get("type") or "").lower()
                    if kind in _IMAGE_BLOCK_TYPES:
                        parts.append("<image>")
                    elif "text" in block:
                        parts.append(str(block.get("text") or ""))
                elif block is not None:
                    parts.append(str(block))
        elif content is not None:
            parts.append(str(content))
    return "\n".join(p for p in parts if p)
