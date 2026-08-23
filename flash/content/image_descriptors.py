"""bounded parsing and validation for managed image descriptors."""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXIF_ORIENTATION_TAG = 274
_EXIF_TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})
_MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_FORMAT_TO_MIME = {image_format: mime for mime, image_format in _MIME_TO_FORMAT.items()}
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
RGB_BYTES_PER_PIXEL = 3
# decoding one image costs its own buffer plus a transient RGB conversion, so the worst mode
# sets the bytes-per-pixel a pixel budget has to assume. exported because the per-image pixel cap
# is derived from it: a cap the decoded-memory guard would always reject is not a limit.
WORST_BYTES_PER_PIXEL = max(_MODE_BYTES_PER_PIXEL.values()) + 2 * RGB_BYTES_PER_PIXEL


@dataclass(frozen=True)
class ImageDescriptorLimits:
    max_images: int
    max_source_bytes: int
    max_total_source_bytes: int
    max_width: int
    max_height: int
    max_pixels: int
    max_total_decoded_bytes: int
    max_data_uri_header_bytes: int
    max_descriptor_bytes: int


@dataclass(frozen=True)
class ValidatedImageDescriptor:
    data: bytes
    data_uri: str
    source_bytes: int
    decoded_peak_bytes: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def descriptor_size(value: str) -> int:
    return len(value.encode("utf-8"))


def descriptor(kind: str, value: str, *, max_descriptor_bytes: int) -> str:
    encoded = json.dumps({"kind": kind, "value": value}, separators=(",", ":"), sort_keys=True)
    size = descriptor_size(encoded)
    if size > max_descriptor_bytes:
        raise ValueError(
            f"encoded image descriptor exceeds the {max_descriptor_bytes}-byte limit ({size} bytes)"
        )
    return encoded


def parse_descriptor(raw: str) -> tuple[str, str]:
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


def check_source_size(size: int, *, max_source_bytes: int) -> None:
    if size > max_source_bytes:
        raise ValueError(f"image source exceeds the {max_source_bytes}-byte limit ({size} bytes)")


def decode_data_uri(uri: str, limits: ImageDescriptorLimits) -> tuple[bytes, str]:
    comma = uri.find(",")
    header = uri if comma < 0 else uri[:comma]
    if len(header.encode("utf-8")) > limits.max_data_uri_header_bytes:
        raise ValueError(
            f"image data URI header exceeds the {limits.max_data_uri_header_bytes}-byte limit"
        )
    match = _DATA_URI_RE.fullmatch(uri)
    if match is None:
        if not uri.startswith("data:"):
            raise ValueError("image source must use a data URI")
        if ";base64," not in uri:
            raise ValueError("image data URI must use base64 encoding")
        raise ValueError("image data URI must use the exact data:image/...;base64,... format")
    mime, payload = match.groups()
    expected_format = _MIME_TO_FORMAT.get(mime)
    if expected_format is None:
        raise ValueError(f"image data URI uses unsupported MIME type {mime!r}")
    max_encoded_size = ((limits.max_source_bytes + 2) // 3) * 4
    if len(payload) > max_encoded_size:
        check_source_size(
            limits.max_source_bytes + 1,
            max_source_bytes=limits.max_source_bytes,
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data URI contains invalid base64 data") from exc
    check_source_size(len(data), max_source_bytes=limits.max_source_bytes)
    return data, expected_format


def package_image_path(
    value: str, package_root: Path | None, limits: ImageDescriptorLimits
) -> Path:
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
    check_source_size(resolved.stat().st_size, max_source_bytes=limits.max_source_bytes)
    return resolved


def validate_dimensions(width: int, height: int, limits: ImageDescriptorLimits) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if width > limits.max_width or height > limits.max_height:
        raise ValueError(
            f"image dimensions {width}x{height} exceed the "
            f"{limits.max_width}x{limits.max_height} limit"
        )
    pixels = width * height
    if pixels > limits.max_pixels:
        raise ValueError(
            f"image has {pixels} pixels, exceeding the {limits.max_pixels}-pixel limit"
        )
    return pixels


def _oriented_dimensions(image: Any, width: int, height: int) -> tuple[int, int]:
    try:
        orientation = image.getexif().get(_EXIF_ORIENTATION_TAG)
    except (AttributeError, TypeError, ValueError, SyntaxError):
        orientation = None
    if orientation in _EXIF_TRANSPOSED_ORIENTATIONS:
        return height, width
    return width, height


def inspect_image_metadata(
    data: bytes,
    limits: ImageDescriptorLimits,
    *,
    expected_format: str | None = None,
) -> tuple[int, int, int, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in _FORMAT_TO_MIME:
                    raise ValueError("image source must be a static PNG, JPEG, or WebP image")
                if expected_format is not None and image_format != expected_format:
                    raise ValueError(
                        f"image data URI MIME type does not match its {image_format} format"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("image source must be a static single-frame image")
                stored_width, stored_height = image.size
                pixels = validate_dimensions(stored_width, stored_height, limits)
                decoded_peak_bytes = pixels * (
                    _MODE_BYTES_PER_PIXEL.get(image.mode, 4) + 2 * RGB_BYTES_PER_PIXEL
                )
                image.verify()
            with Image.open(io.BytesIO(data)) as orientation_probe:
                width, height = _oriented_dimensions(
                    orientation_probe,
                    stored_width,
                    stored_height,
                )
                validate_dimensions(width, height, limits)
            return decoded_peak_bytes, width, height, image_format
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc


def inspect_image_bytes(data: bytes, limits: ImageDescriptorLimits) -> tuple[int, int, int]:
    decoded_peak_bytes, width, height, _image_format = inspect_image_metadata(data, limits)
    return decoded_peak_bytes, width, height


def canonical_data_uri(data: bytes, image_format: str) -> str:
    media_type = _FORMAT_TO_MIME.get(image_format)
    if media_type is None:
        raise ValueError("image source must be a static PNG, JPEG, or WebP image")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def descriptor_source_size(
    encoded: str,
    package_root: str | Path | None,
    limits: ImageDescriptorLimits,
) -> int:
    kind, value = parse_descriptor(encoded)
    if kind == "bytes":
        return len(base64.b64decode(value, validate=True))
    if kind == "data_uri":
        return len(decode_data_uri(value, limits)[0])
    if kind == "path":
        root = Path(package_root) if package_root else None
        return package_image_path(value, root, limits).stat().st_size
    raise ValueError("invalid internal image descriptor kind")


def read_descriptor_source(
    encoded: str,
    package_root: str | Path | None,
    limits: ImageDescriptorLimits,
) -> bytes:
    kind, value = parse_descriptor(encoded)
    if kind == "bytes":
        try:
            data = base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid internal byte image descriptor") from exc
    elif kind == "data_uri":
        data = decode_data_uri(value, limits)[0]
    elif kind == "path":
        root = Path(package_root) if package_root else None
        data = package_image_path(value, root, limits).read_bytes()
    else:
        raise ValueError("invalid internal image descriptor kind")
    check_source_size(len(data), max_source_bytes=limits.max_source_bytes)
    return data


def validate_image_descriptor_data(
    encoded: str,
    data: bytes,
    limits: ImageDescriptorLimits,
    *,
    decode_pixels: bool,
    pixel_decoder: Callable[[bytes], None],
) -> ValidatedImageDescriptor:
    kind, value = parse_descriptor(encoded)
    expected_format = decode_data_uri(value, limits)[1] if kind == "data_uri" else None
    decoded_peak_bytes, width, height, image_format = inspect_image_metadata(
        data,
        limits,
        expected_format=expected_format,
    )
    if decode_pixels:
        pixel_decoder(data)
    return ValidatedImageDescriptor(
        data=data,
        data_uri=canonical_data_uri(data, image_format),
        source_bytes=len(data),
        decoded_peak_bytes=decoded_peak_bytes,
        width=width,
        height=height,
    )


def validate_image_descriptor(
    encoded: str,
    package_root: str | Path | None,
    limits: ImageDescriptorLimits,
    *,
    decode_pixels: bool,
    pixel_decoder: Callable[[bytes], None],
) -> ValidatedImageDescriptor:
    data = read_descriptor_source(encoded, package_root, limits)
    return validate_image_descriptor_data(
        encoded,
        data,
        limits,
        decode_pixels=decode_pixels,
        pixel_decoder=pixel_decoder,
    )


def validate_image_descriptors(
    descriptors: list[str] | tuple[str, ...],
    package_root: str | Path | None,
    limits: ImageDescriptorLimits,
    *,
    pixel_decoder: Callable[[bytes], None],
) -> list[ValidatedImageDescriptor]:
    if len(descriptors) > limits.max_images:
        raise ValueError(
            f"example contains {len(descriptors)} images, exceeding the "
            f"{limits.max_images}-image limit"
        )
    validated: list[ValidatedImageDescriptor] = []
    source_bytes = 0
    prior_pixels = 0
    for encoded in descriptors:
        item = validate_image_descriptor(
            encoded,
            package_root,
            limits,
            decode_pixels=False,
            pixel_decoder=pixel_decoder,
        )
        source_bytes += item.source_bytes
        if source_bytes > limits.max_total_source_bytes:
            raise ValueError(
                f"example image sources exceed the {limits.max_total_source_bytes}-byte limit"
            )
        decoded_bytes = RGB_BYTES_PER_PIXEL * prior_pixels + item.decoded_peak_bytes
        if decoded_bytes > limits.max_total_decoded_bytes:
            raise ValueError(
                f"example decoded images exceed the {limits.max_total_decoded_bytes}-byte limit"
            )
        prior_pixels += item.pixels
        validated.append(item)
    for item in validated:
        pixel_decoder(item.data)
    return validated
