"""torch-free image token accounting for control-plane sft profiling.

the control plane quotes sft from the packaged dataset in-process, and its `[server]` extra
installs transformers and pillow but not torch. the VL `AutoProcessor` cannot load there at all:
transformers resolves it through a torchvision-backed image processor, so importing it raises
`ImportError: ... requires the Torchvision library` before any tokenization happens.

a quote does not need pixels, only token counts. the plain tokenizer renders an image content
block into `<|vision_start|><|image_pad|><|vision_end|>` and emits exactly ONE `<|image_pad|>` per
image; the processor differs only in expanding that single placeholder into the run of pad tokens
the vision tower will actually occupy. that run length is arithmetic over the image's dimensions,
which pillow reads from the header alone:

    resized  = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)
    patches  = (resized_h // patch_size) * (resized_w // patch_size)
    pad_run  = patches // merge_size ** 2

`smart_resize` is the qwen VL resize policy, reimplemented here as pure ``math``. it is duplicated
rather than imported because the upstream module that defines it is the same one that pulls in
torchvision -- the dependency there is for resampling PIXELS, which this module never does.
`test_image_tokens.py` pins this implementation against transformers' own `smart_resize` so the
copy cannot drift silently.

the geometry comes from the model's published `preprocessor_config.json`, never hardcoded: a
checkpoint that changes its patch or merge size changes the quote with it.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

# qwen VL publishes its pixel budget under `size`, older revisions under min_pixels/max_pixels.
# these defaults are the transformers fallbacks, used only when the config declares neither.
_DEFAULT_PATCH_SIZE = 16
_DEFAULT_MERGE_SIZE = 2
_DEFAULT_MIN_PIXELS = 56 * 56
_DEFAULT_MAX_PIXELS = 14 * 14 * 4 * 1280
# the same bound transformers enforces: beyond it smart_resize cannot hold the aspect ratio.
_MAX_ASPECT_RATIO = 200


class ImageGeometryUnavailable(ValueError):
    """the model's published preprocessor config does not describe its image geometry."""


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
    """qwen VL's resize policy: dimensions divisible by ``factor``, pixels inside the budget.

    a pure-arithmetic copy of transformers' `smart_resize`, pinned against it by test.
    """
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


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def geometry_from_preprocessor_config(config: dict) -> ImageGeometry:
    """Read the published patch/merge sizes and pixel budget from a preprocessor config."""
    if not isinstance(config, dict):
        raise ImageGeometryUnavailable("preprocessor config is not an object")
    patch_size = _positive_int(config.get("patch_size")) or _DEFAULT_PATCH_SIZE
    merge_size = _positive_int(config.get("merge_size")) or _DEFAULT_MERGE_SIZE
    size = config.get("size")
    size = size if isinstance(size, dict) else {}
    # `size` is the current spelling; min_pixels/max_pixels the older one. prefer whichever the
    # checkpoint actually published rather than assuming a shape.
    min_pixels = (
        _positive_int(config.get("min_pixels"))
        or _positive_int(size.get("shortest_edge"))
        or _DEFAULT_MIN_PIXELS
    )
    max_pixels = (
        _positive_int(config.get("max_pixels"))
        or _positive_int(size.get("longest_edge"))
        or _DEFAULT_MAX_PIXELS
    )
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

    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename="preprocessor_config.json",
            token=os.environ.get("HF_TOKEN"),
            **({"revision": revision} if revision else {}),
        )
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except ImageGeometryUnavailable:
        raise
    except Exception as exc:
        raise ImageGeometryUnavailable(
            f"could not read the image geometry published by {model_id!r}; an image-bearing sft "
            "dataset cannot be quoted without it"
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


def image_dimensions(data: bytes) -> tuple[int, int]:
    """Read one image's (width, height) from its bytes without decoding the pixels."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a [server] dependency
        raise RuntimeError("Pillow is required for multimodal image training") from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise ValueError("image source is not a valid image") from exc


def descriptor_pad_tokens(
    descriptors: list[str],
    package_root: str | Path | None,
    geometry: ImageGeometry,
) -> list[int]:
    """The pad-token run length each normalized descriptor expands to, in order."""
    from flash.content.multimodal import read_descriptor_source

    counts = []
    for descriptor in descriptors:
        width, height = image_dimensions(read_descriptor_source(descriptor, package_root))
        counts.append(image_pad_tokens(width, height, geometry))
    return counts


def expand_image_pad_runs(
    input_ids: list[int], pad_token_id: int, pad_counts: list[int]
) -> list[int]:
    """Expand each single rendered pad placeholder into the run the processor would produce.

    The plain tokenizer emits exactly one ``<|image_pad|>`` per image block. The processor emits
    ``pad_counts[i]`` of them for image ``i``. Rewriting the placeholder in place reproduces the
    processor's token sequence exactly, which is what makes the completion mask -- derived by
    comparing prompt and full ids -- land on the same boundary the worker will compute.
    """
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
