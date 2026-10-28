"""Decode the parent's staged rollout images without an intermediate resize."""

from pathlib import Path
from urllib.parse import unquote, urlparse


def initial_prompt_images(messages: list[dict]) -> list | None:
    """Return Flash image pixels, or None when the prompt has no image blocks.

    The parent freezes tokens and media digests from the original RGB pixels. verl's
    default Qwen vision loader resizes first, which changes those processor outputs.
    Staged file URIs are worker-owned PNGs; authored external URLs remain unsupported.
    """
    from flash.content.multimodal import (
        MAX_IMAGE_SOURCE_BYTES,
        MAX_IMAGES_PER_EXAMPLE,
        decode_image_descriptors,
        normalize_image_source,
    )

    descriptors = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"video", "audio"}:
                raise ValueError("Flash multi-turn prompts support images, not video or audio")
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            if len(descriptors) >= MAX_IMAGES_PER_EXAMPLE:
                raise ValueError("initial rollout prompt exceeds the image count limit")
            source = block.get("image")
            if isinstance(source, str) and urlparse(source).scheme == "file":
                parsed = urlparse(source)
                if parsed.netloc or parsed.query or parsed.fragment:
                    raise ValueError("staged rollout image must be a local file URI")
                with Path(unquote(parsed.path)).open("rb") as handle:
                    source = handle.read(MAX_IMAGE_SOURCE_BYTES + 1)
            descriptors.append(normalize_image_source(source, None, defer_validation=True))
    return decode_image_descriptors(descriptors, None) if descriptors else None
