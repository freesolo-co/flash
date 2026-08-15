"""torch-free image token accounting used to quote image-bearing sft on the control plane."""

from __future__ import annotations

import io
import json

import pytest

from flash.engine.profiling.image_tokens import (
    ImageGeometry,
    ImageGeometryUnavailable,
    descriptor_pad_tokens,
    expand_image_pad_runs,
    geometry_from_preprocessor_config,
    image_pad_tokens,
    smart_resize,
)

# the geometry every image-capable catalog model publishes today (patch 16, merge 2, and the
# pixel budget from `size`). read from config in production; pinned here so the expectations below
# are readable numbers rather than a second copy of the lookup.
QWEN_GEOMETRY = ImageGeometry(patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216)


def _png_bytes(width: int, height: int) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (width, height), (12, 34, 56)).save(out, format="PNG")
    return out.getvalue()


class TestSmartResizeMatchesTransformers:
    """The resize policy is duplicated from transformers; pin the copy against the original.

    The upstream module cannot be imported without torchvision, which is exactly why the policy is
    reimplemented. Where it IS importable (the dev/gpu extras), the two must agree exactly -- that
    is what stops this copy drifting from the behavior the GPU worker will actually apply.
    """

    def test_matches_the_upstream_policy_across_a_size_grid(self):
        upstream = pytest.importorskip(
            "transformers.models.qwen2_vl.image_processing_qwen2_vl",
            reason="needs torchvision, which the control plane deliberately lacks",
        )
        sizes = [
            (56, 56),
            (64, 64),
            (112, 84),
            (224, 224),
            (300, 200),
            (1, 1),
            (17, 133),
            (640, 480),
            (1024, 768),
            (33, 33),
            (4000, 60),
            (200, 300),
            (97, 53),
            (8192, 8192),
        ]
        for width, height in sizes:
            assert smart_resize(
                height,
                width,
                QWEN_GEOMETRY.factor,
                QWEN_GEOMETRY.min_pixels,
                QWEN_GEOMETRY.max_pixels,
            ) == upstream.smart_resize(
                height,
                width,
                factor=QWEN_GEOMETRY.factor,
                min_pixels=QWEN_GEOMETRY.min_pixels,
                max_pixels=QWEN_GEOMETRY.max_pixels,
            ), f"resize policy drifted from transformers at {width}x{height}"


class TestImagePadTokens:
    """Pad-run lengths, measured against a real Qwen3-VL processor.

    The expectations are the processor's own output, captured from
    `AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")` counting `<|image_pad|>` ids in the
    tokenized row. They are ground truth, not a restatement of the formula.
    """

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (56, 56, 64),
            (64, 64, 64),
            (112, 84, 70),
            (224, 224, 64),
            (300, 200, 70),
            (1, 1, 64),
            (17, 133, 69),
            (640, 480, 300),
            (1024, 768, 768),
            (33, 33, 64),
            (4000, 60, 250),
            (200, 300, 70),
            (97, 53, 66),
        ],
    )
    def test_reproduces_the_processors_pad_run(self, width, height, expected):
        assert image_pad_tokens(width, height, QWEN_GEOMETRY) == expected

    def test_a_larger_image_never_costs_fewer_tokens(self):
        counts = [image_pad_tokens(w, w, QWEN_GEOMETRY) for w in (64, 256, 512, 1024, 2048)]
        assert counts == sorted(counts)

    def test_rejects_a_degenerate_size(self):
        with pytest.raises(ValueError, match="positive"):
            image_pad_tokens(0, 10, QWEN_GEOMETRY)

    def test_rejects_an_aspect_ratio_the_policy_cannot_hold(self):
        # transformers raises here too; matching it keeps the quote from inventing a number the
        # worker would refuse to train on.
        with pytest.raises(ValueError, match="aspect ratio"):
            image_pad_tokens(4000, 3, QWEN_GEOMETRY)


class TestGeometryFromConfig:
    def test_reads_the_published_geometry(self):
        geometry = geometry_from_preprocessor_config(
            {
                "patch_size": 16,
                "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            }
        )
        assert geometry == QWEN_GEOMETRY

    def test_accepts_the_older_min_max_pixels_spelling(self):
        geometry = geometry_from_preprocessor_config(
            {"patch_size": 14, "merge_size": 2, "min_pixels": 3136, "max_pixels": 1003520}
        )
        assert (geometry.patch_size, geometry.min_pixels, geometry.max_pixels) == (
            14,
            3136,
            1003520,
        )

    def test_a_different_patch_size_changes_the_quote(self):
        # the geometry is read, never hardcoded: a checkpoint that changes its patch size has to
        # move the token count with it, or the quote silently describes a different model.
        coarse = ImageGeometry(patch_size=32, merge_size=2, min_pixels=65536, max_pixels=16777216)
        assert image_pad_tokens(640, 480, coarse) != image_pad_tokens(640, 480, QWEN_GEOMETRY)

    def test_rejects_an_inverted_pixel_budget(self):
        with pytest.raises(ImageGeometryUnavailable, match="min_pixels"):
            geometry_from_preprocessor_config({"min_pixels": 4096, "max_pixels": 1024})

    def test_rejects_a_non_object_config(self):
        with pytest.raises(ImageGeometryUnavailable):
            geometry_from_preprocessor_config([])  # type: ignore[arg-type]


class TestExpandImagePadRuns:
    PAD = 248056

    def test_expands_each_placeholder_in_place(self):
        assert expand_image_pad_runs([1, self.PAD, 2, self.PAD, 3], self.PAD, [3, 2]) == [
            1,
            self.PAD,
            self.PAD,
            self.PAD,
            2,
            self.PAD,
            self.PAD,
            3,
        ]

    def test_leaves_a_text_only_row_untouched(self):
        assert expand_image_pad_runs([1, 2, 3], self.PAD, []) == [1, 2, 3]

    def test_rejects_more_placeholders_than_images(self):
        with pytest.raises(ValueError, match="more image placeholders"):
            expand_image_pad_runs([self.PAD, self.PAD], self.PAD, [4])

    def test_rejects_fewer_placeholders_than_images(self):
        # an image the renderer never placed would be paid for in the quote and absent from
        # training, so the mismatch has to be loud rather than dropped.
        with pytest.raises(ValueError, match="expected 2"):
            expand_image_pad_runs([self.PAD], self.PAD, [4, 4])


class TestDescriptorPadTokens:
    def test_counts_each_descriptor_without_decoding_pixels(self):
        from flash.content.multimodal import normalize_image_source

        descriptors = [
            normalize_image_source(_png_bytes(640, 480), None),
            normalize_image_source(_png_bytes(56, 56), None),
        ]
        assert descriptor_pad_tokens(descriptors, None, QWEN_GEOMETRY) == [300, 64]

    def test_rejects_a_descriptor_that_is_not_an_image(self):
        descriptor = json.dumps({"kind": "bytes", "value": "bm90YW5pbWFnZQ=="})
        with pytest.raises(ValueError, match="not a valid image"):
            descriptor_pad_tokens([descriptor], None, QWEN_GEOMETRY)
