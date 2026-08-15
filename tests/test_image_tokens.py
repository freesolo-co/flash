"""torch-free image token accounting used to quote image-bearing sft on the control plane."""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest

from flash.engine.profiling.image_tokens import (
    ImageGeometry,
    ImageGeometryUnavailable,
    descriptor_pad_tokens,
    expand_image_pad_runs,
    geometry_from_preprocessor_config,
    image_pad_tokens,
    load_image_geometry,
    smart_resize,
)

# the geometry every image-capable catalog model publishes today (patch 16, merge 2, and the
# pixel budget from `size`). read from config in production; pinned here so the expectations below
# are readable numbers rather than a second copy of the lookup.
QWEN_GEOMETRY = ImageGeometry(patch_size=16, merge_size=2, min_pixels=65536, max_pixels=16777216)
PROCESSOR_PAD_CASES = [
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
]


def _png_bytes(width: int, height: int) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (width, height), (12, 34, 56)).save(out, format="PNG")
    return out.getvalue()


def _truncated_bmp_bytes(width: int, height: int) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (width, height), (12, 34, 56)).save(out, format="BMP")
    return out.getvalue()[:-10]


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
        for width, height, _expected in PROCESSOR_PAD_CASES:
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

    @pytest.mark.parametrize(("width", "height", "expected"), PROCESSOR_PAD_CASES)
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
    def test_accepts_explicit_positive_json_integers(self):
        geometry = geometry_from_preprocessor_config(
            {
                "patch_size": 16,
                "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            }
        )
        assert geometry == QWEN_GEOMETRY

    @pytest.mark.parametrize("field", ["patch_size", "merge_size"])
    def test_rejects_missing_geometry(self, field):
        config = {"patch_size": 16, "merge_size": 2}
        del config[field]

        with pytest.raises(ImageGeometryUnavailable, match=field):
            geometry_from_preprocessor_config(config)

    @pytest.mark.parametrize("field", ["patch_size", "merge_size"])
    @pytest.mark.parametrize("value", [0, -1, True, "16", 16.5])
    def test_rejects_non_positive_or_non_integer_geometry(self, field, value):
        config = {"patch_size": 16, "merge_size": 2, field: value}

        with pytest.raises(ImageGeometryUnavailable, match=field):
            geometry_from_preprocessor_config(config)

    @pytest.mark.parametrize(
        ("container", "field"),
        [
            ("root", "min_pixels"),
            ("root", "max_pixels"),
            ("size", "shortest_edge"),
            ("size", "longest_edge"),
        ],
    )
    @pytest.mark.parametrize("value", [0, -1, True, "65536", 65536.0])
    def test_rejects_invalid_published_pixel_budgets(self, container, field, value):
        config = {"patch_size": 16, "merge_size": 2}
        target = config if container == "root" else config.setdefault("size", {})
        target[field] = value

        with pytest.raises(ImageGeometryUnavailable, match=field):
            geometry_from_preprocessor_config(config)

    def test_rejects_an_invalid_size_budget_even_when_the_root_budget_wins(self):
        with pytest.raises(ImageGeometryUnavailable, match="shortest_edge"):
            geometry_from_preprocessor_config(
                {
                    "patch_size": 16,
                    "merge_size": 2,
                    "min_pixels": 65536,
                    "size": {"shortest_edge": "ignored-if-not-validated"},
                }
            )

    @pytest.mark.parametrize("value", [[], "pixels", 1, False])
    def test_rejects_a_present_non_object_size(self, value):
        with pytest.raises(ImageGeometryUnavailable, match="size must be an object"):
            geometry_from_preprocessor_config({"patch_size": 16, "merge_size": 2, "size": value})

    def test_none_pixel_budget_fields_use_the_existing_defaults(self):
        geometry = geometry_from_preprocessor_config(
            {
                "patch_size": 16,
                "merge_size": 2,
                "min_pixels": None,
                "max_pixels": None,
                "size": None,
            }
        )
        assert (geometry.min_pixels, geometry.max_pixels) == (3136, 1003520)

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
            geometry_from_preprocessor_config(
                {
                    "patch_size": 16,
                    "merge_size": 2,
                    "min_pixels": 4096,
                    "max_pixels": 1024,
                }
            )

    def test_rejects_a_non_object_config(self):
        with pytest.raises(ImageGeometryUnavailable):
            geometry_from_preprocessor_config([])  # type: ignore[arg-type]


class TestImageGeometrySubmissionFailures:
    @staticmethod
    def _submission_status(exc: Exception) -> int:
        from flash.server.routes.runs import _submit_failure_http_error

        return _submit_failure_http_error(exc).status_code

    @staticmethod
    def _load_failure(monkeypatch, failure: Exception) -> ImageGeometryUnavailable:
        def fail_download(**_kwargs):
            raise failure

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_download)
        with pytest.raises(ImageGeometryUnavailable) as excinfo:
            load_image_geometry("org/model")
        return excinfo.value

    @pytest.mark.parametrize(
        ("status", "expected_plane_fault", "expected_http_status"),
        [
            (401, False, 400),
            (403, False, 400),
            (404, False, 400),
            (429, True, 503),
            (500, True, 503),
            (503, True, 503),
        ],
    )
    def test_hub_http_status_distinguishes_permanent_and_transient_failures(
        self, monkeypatch, status, expected_plane_fault, expected_http_status
    ):
        from huggingface_hub.errors import HfHubHTTPError

        response = SimpleNamespace(
            status_code=status,
            headers={},
            request=SimpleNamespace(),
        )
        failure = self._load_failure(
            monkeypatch,
            HfHubHTTPError("hub request failed", response=response),
        )

        assert failure.plane_fault is expected_plane_fault
        assert self._submission_status(failure) == expected_http_status

    def test_network_timeout_is_a_plane_fault(self, monkeypatch):
        import httpx

        failure = self._load_failure(monkeypatch, httpx.ReadTimeout("timed out"))

        assert failure.plane_fault is True
        assert self._submission_status(failure) == 503

    def test_local_cache_miss_is_a_plane_fault(self, monkeypatch):
        from huggingface_hub.errors import LocalEntryNotFoundError

        failure = self._load_failure(
            monkeypatch,
            LocalEntryNotFoundError("cache entry is unavailable"),
        )

        assert failure.plane_fault is True
        assert self._submission_status(failure) == 503

    def test_local_file_io_failure_is_a_plane_fault(self, monkeypatch, tmp_path):
        missing_path = tmp_path / "missing-preprocessor-config.json"
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **_kwargs: str(missing_path),
        )

        with pytest.raises(ImageGeometryUnavailable) as excinfo:
            load_image_geometry("org/model")

        assert excinfo.value.plane_fault is True
        assert self._submission_status(excinfo.value) == 503

    def test_missing_hub_file_is_a_submission_error(self, monkeypatch):
        from huggingface_hub.errors import EntryNotFoundError

        failure = self._load_failure(
            monkeypatch,
            EntryNotFoundError("preprocessor_config.json is missing"),
        )

        assert failure.plane_fault is False
        assert self._submission_status(failure) == 400

    @pytest.mark.parametrize(
        "contents",
        [
            "{not-json",
            json.dumps([]),
            json.dumps({"patch_size": 16, "merge_size": 0}),
        ],
    )
    def test_malformed_or_invalid_config_is_a_submission_error(
        self, monkeypatch, tmp_path, contents
    ):
        config_path = tmp_path / "preprocessor_config.json"
        config_path.write_text(contents)
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **_kwargs: str(config_path),
        )

        with pytest.raises(ImageGeometryUnavailable) as excinfo:
            load_image_geometry("org/model")

        assert excinfo.value.plane_fault is False
        assert self._submission_status(excinfo.value) == 400


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
    def test_counts_each_fully_validated_descriptor(self):
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

    def test_rejects_a_truncated_payload_that_still_has_valid_dimensions(self):
        image_module = pytest.importorskip("PIL.Image")
        data = _truncated_bmp_bytes(64, 64)
        with image_module.open(io.BytesIO(data)) as image:
            assert image.size == (64, 64)
        descriptor = json.dumps({"kind": "bytes", "value": base64.b64encode(data).decode("ascii")})

        with pytest.raises(ValueError, match="not a valid image"):
            descriptor_pad_tokens([descriptor], None, QWEN_GEOMETRY)

    def test_rejects_aggregate_decoded_bytes_before_full_decode(self, monkeypatch):
        from flash.content import multimodal

        data = _png_bytes(10, 10)
        descriptors = [multimodal.normalize_image_source(data, None) for _ in range(2)]
        monkeypatch.setattr(multimodal, "MAX_TOTAL_DECODED_BYTES", 599)
        monkeypatch.setattr(
            multimodal,
            "_decode_image_bytes",
            lambda _data: (_ for _ in ()).throw(AssertionError("full decode reached")),
        )

        with pytest.raises(ValueError, match="decoded images"):
            descriptor_pad_tokens(descriptors, None, QWEN_GEOMETRY)
