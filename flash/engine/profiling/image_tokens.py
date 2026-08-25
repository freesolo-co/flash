"""torch-free image token accounting for control-plane sft profiling.

one rendered image placeholder expands to the published vision grid size:

    pad_run = (resized_h // patch_size) * (resized_w // patch_size) // merge_size**2

the upstream differential pins the resize policy, and pixels are fully loaded before quoting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from flash.content.image_descriptors import RGB_BYTES_PER_PIXEL
from flash.content.multimodal import (
    MAX_IMAGES_PER_EXAMPLE,
    MAX_TOTAL_DECODED_BYTES,
    MAX_TOTAL_IMAGE_SOURCE_BYTES,
    decode_descriptor_pixels,
    read_descriptor_source,
    validate_image_descriptor_data,
)

# qwen VL publishes its pixel budget under `size`, older revisions under min_pixels/max_pixels.
# these defaults are the transformers fallbacks, used only when the config declares neither.
_DEFAULT_MIN_PIXELS = 56 * 56
_DEFAULT_MAX_PIXELS = 14 * 14 * 4 * 1280
# the same bound transformers enforces: beyond it smart_resize cannot hold the aspect ratio.
_MAX_ASPECT_RATIO = 200
# profile-wide unique full-decode work; per-row decoded bytes keep their separate bound.
MAX_PROFILE_DECODED_WORK_BYTES = MAX_TOTAL_DECODED_BYTES * MAX_IMAGES_PER_EXAMPLE


@dataclass(frozen=True)
class ImageDescriptorMetadata:
    source_bytes: int
    decoded_peak_bytes: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass
class ImageProfileValidationState:
    """one profile's descriptor cache and cumulative decode budget.

    a descriptor repeated across rows is validated and decoded ONCE; every occurrence still counts
    toward the per-row limits below, so repetition cannot be used to smuggle a row past them.
    """

    descriptor_metadata: dict[str, ImageDescriptorMetadata] = field(default_factory=dict)
    decoded_work_bytes: int = 0


def image_descriptor_metadata(
    descriptors: list[str],
    package_root: str | Path | None,
    validation_state: ImageProfileValidationState,
    *,
    profile_decoded_work_limit: int,
) -> list[ImageDescriptorMetadata]:
    """return fully validated dimensions while bounding unique profile decode work."""
    if len(descriptors) > MAX_IMAGES_PER_EXAMPLE:
        raise ValueError(
            f"example contains {len(descriptors)} images, exceeding the {MAX_IMAGES_PER_EXAMPLE}-image limit"
        )
    pending: dict[str, tuple[bytes, ImageDescriptorMetadata]] = {}
    metadata = []
    source_bytes = 0
    prior_pixels = 0
    new_decoded_work = 0
    for descriptor in descriptors:
        item = validation_state.descriptor_metadata.get(descriptor)
        if item is None:
            prepared = pending.get(descriptor)
            if prepared is None:
                data = read_descriptor_source(descriptor, package_root)
                validated = validate_image_descriptor_data(descriptor, data, decode_pixels=False)
                item = ImageDescriptorMetadata(
                    source_bytes=validated.source_bytes,
                    decoded_peak_bytes=validated.decoded_peak_bytes,
                    width=validated.width,
                    height=validated.height,
                )
                pending[descriptor] = data, item
                new_decoded_work += item.decoded_peak_bytes
            else:
                item = prepared[1]
        metadata.append(item)
        source_bytes += item.source_bytes
        if source_bytes > MAX_TOTAL_IMAGE_SOURCE_BYTES:
            raise ValueError(
                f"example image sources exceed the {MAX_TOTAL_IMAGE_SOURCE_BYTES}-byte limit"
            )
        decoded_bytes = RGB_BYTES_PER_PIXEL * prior_pixels + item.decoded_peak_bytes
        if decoded_bytes > MAX_TOTAL_DECODED_BYTES:
            raise ValueError(
                f"example decoded images exceed the {MAX_TOTAL_DECODED_BYTES}-byte limit"
            )
        prior_pixels += item.pixels

    prospective_decoded_work = validation_state.decoded_work_bytes + new_decoded_work
    if prospective_decoded_work > profile_decoded_work_limit:
        raise ValueError(
            f"profile decoded image work exceeds the {profile_decoded_work_limit}-byte limit"
        )

    # fully decode each new descriptor before committing anything: a header that parses but whose
    # pixels are corrupt must fail the row, and must leave neither the cache nor the budget touched.
    for data, _item in pending.values():
        decode_descriptor_pixels(data)
    validation_state.descriptor_metadata.update(
        {descriptor: item for descriptor, (_data, item) in pending.items()}
    )
    validation_state.decoded_work_bytes = prospective_decoded_work
    return metadata


class ImageGeometryUnavailable(ValueError):
    """the model's published preprocessor config does not describe its image geometry."""

    def __init__(self, message: str, *, plane_fault: bool = False) -> None:
        super().__init__(message)
        self.plane_fault = plane_fault


@dataclass(frozen=True)
class ImageGeometry:
    """the published vision geometry a checkpoint expands images with."""

    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """resize dimensions to the qwen vision factor and pixel budget."""
    if height <= 0 or width <= 0:
        raise ValueError("image width and height must be positive")
    if max(height, width) / min(height, width) > _MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {_MAX_ASPECT_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _positive_json_int(config: dict, field: str, *, required: bool = False) -> int | None:
    if field not in config or config[field] is None:
        if required:
            raise ImageGeometryUnavailable(
                f"preprocessor config must declare {field} as a positive JSON integer"
            )
        return None
    value = config[field]
    if type(value) is not int or value <= 0:
        raise ImageGeometryUnavailable(
            f"preprocessor config must declare {field} as a positive JSON integer"
        )
    return value


def geometry_from_preprocessor_config(config: object) -> ImageGeometry:
    """read published patch, merge, and pixel-budget geometry."""
    if not isinstance(config, dict):
        raise ImageGeometryUnavailable("preprocessor config is not an object")
    patch_size = _positive_json_int(config, "patch_size", required=True)
    merge_size = _positive_json_int(config, "merge_size", required=True)
    assert patch_size is not None
    assert merge_size is not None
    size_value = config.get("size")
    if size_value is None:
        size: dict = {}
    elif isinstance(size_value, dict):
        size = size_value
    else:
        raise ImageGeometryUnavailable("preprocessor config field size must be an object")
    published_min = _positive_json_int(config, "min_pixels")
    published_max = _positive_json_int(config, "max_pixels")
    size_min = _positive_json_int(size, "shortest_edge")
    size_max = _positive_json_int(size, "longest_edge")
    min_pixels = published_min or size_min or _DEFAULT_MIN_PIXELS
    max_pixels = published_max or size_max or _DEFAULT_MAX_PIXELS
    if min_pixels > max_pixels:
        raise ImageGeometryUnavailable(
            f"preprocessor config declares min_pixels {min_pixels} above max_pixels {max_pixels}"
        )
    return ImageGeometry(
        patch_size=patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def load_image_geometry(model_id: str, revision: str = "") -> ImageGeometry:
    """Fetch the model's published image geometry from the hub without importing torch."""
    import os

    from huggingface_hub import hf_hub_download

    from flash.engine.support.huggingface import hub_error_transience, model_revision_kwargs

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="preprocessor_config.json",
            token=os.environ.get("HF_TOKEN"),
            **model_revision_kwargs(revision),
        )
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except ImageGeometryUnavailable:
        raise
    except Exception as exc:
        transience = hub_error_transience(exc)
        raise ImageGeometryUnavailable(
            f"could not read the image geometry published by {model_id!r}; an image-bearing sft "
            "dataset cannot be quoted without it",
            # a downloaded local config that cannot be read is plane state, while generic worker
            # prefetch os errors remain non-retriable at their caller boundary.
            plane_fault=transience is True or (transience is None and isinstance(exc, OSError)),
        ) from exc
    return geometry_from_preprocessor_config(config)


def image_pad_tokens(width: int, height: int, geometry: ImageGeometry) -> int:
    """The number of pad tokens one image of this size expands to."""
    resized_h, resized_w = smart_resize(
        height,
        width,
        geometry.factor,
        geometry.min_pixels,
        geometry.max_pixels,
    )
    patches = (resized_h // geometry.patch_size) * (resized_w // geometry.patch_size)
    return patches // (geometry.merge_size * geometry.merge_size)


def descriptor_pad_tokens(
    descriptors: list[str],
    package_root: str | Path | None,
    geometry: ImageGeometry,
    validation_state: ImageProfileValidationState,
) -> list[int]:
    """return each descriptor occurrence's validated pad-token run length."""
    metadata = image_descriptor_metadata(
        descriptors,
        package_root,
        validation_state,
        profile_decoded_work_limit=MAX_PROFILE_DECODED_WORK_BYTES,
    )
    return [image_pad_tokens(item.width, item.height, geometry) for item in metadata]


def expand_image_pad_runs(
    input_ids: list[int], pad_token_id: int, pad_counts: list[int]
) -> list[int]:
    """expand each rendered image placeholder to its processor pad run."""
    expanded: list[int] = []
    seen = 0
    for token_id in input_ids:
        if token_id == pad_token_id:
            if seen >= len(pad_counts):
                raise ValueError(
                    "the rendered prompt contains more image placeholders than the example "
                    "supplied images"
                )
            expanded.extend([pad_token_id] * pad_counts[seen])
            seen += 1
        else:
            expanded.append(token_id)
    if seen != len(pad_counts):
        raise ValueError(
            f"the rendered prompt contains {seen} image placeholder(s), expected {len(pad_counts)}"
        )
    return expanded
