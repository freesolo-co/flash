"""Decode the parent's staged rollout images without an intermediate resize."""

from pathlib import Path
from urllib.parse import unquote, urlparse

from flash.content import image_descriptors, multimodal


def _image_data(source, limits) -> tuple[bytes, int, str | None]:
    """Return image bytes and the bytes charged to the authored-source budget."""
    if isinstance(source, str) and urlparse(source).scheme == "file":
        parsed = urlparse(source)
        if parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("staged rollout image must be a local file URI")
        # Parent-owned PNGs can be larger than their admitted JPEG/WebP sources.
        # Bound the artifact read separately; decoded dimensions/memory are checked below.
        with Path(unquote(parsed.path)).open("rb") as handle:
            data = handle.read(limits.max_total_decoded_bytes + 1)
        if len(data) > limits.max_total_decoded_bytes:
            raise ValueError("staged rollout image exceeds the artifact byte limit")
        return data, 0, "PNG"
    descriptor = multimodal.normalize_image_source(source, None, defer_validation=True)
    data = image_descriptors.read_descriptor_source(descriptor, None, limits)
    kind, value = image_descriptors.parse_descriptor(descriptor)
    expected_format = (
        image_descriptors.decode_data_uri(value, limits)[1] if kind == "data_uri" else None
    )
    return data, len(data), expected_format


def initial_prompt_images(messages: list[dict]) -> list | None:
    """Return Flash image pixels, or None when the prompt has no image blocks.

    The parent freezes tokens and media digests from the original RGB pixels. verl's
    default Qwen vision loader resizes first, which changes those processor outputs.
    Staged file URIs are worker-owned PNGs; authored external URLs remain unsupported.
    """
    limits = multimodal._image_limits()
    images = []
    source_bytes = prior_pixels = 0
    try:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"video", "audio"}:
                    raise ValueError("Flash multi-turn prompts support images, not video or audio")
                if not isinstance(block, dict) or block.get("type") != "image":
                    continue
                if len(images) >= limits.max_images:
                    raise ValueError("initial rollout prompt exceeds the image count limit")
                data, admitted_bytes, expected_format = _image_data(block.get("image"), limits)
                source_bytes += admitted_bytes
                if source_bytes > limits.max_total_source_bytes:
                    raise ValueError("initial rollout image sources exceed the source byte limit")
                peak_bytes, width, height, _ = image_descriptors.inspect_image_metadata(
                    data, limits, expected_format=expected_format
                )
                if (
                    image_descriptors.RGB_BYTES_PER_PIXEL * prior_pixels + peak_bytes
                    > limits.max_total_decoded_bytes
                ):
                    raise ValueError("initial rollout decoded images exceed the memory limit")
                images.append(multimodal._decode_image_bytes(data))
                prior_pixels += width * height
    except Exception:
        for image in images:
            image.close()
        raise
    return images or None
