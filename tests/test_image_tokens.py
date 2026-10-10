"""torch-free image token accounting used to quote image-bearing sft on the control plane."""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

from flash.content import multimodal
from flash.engine.profiling import image_tokens
from flash.engine.profiling.image_tokens import (
    ImageGeometry,
    ImageGeometryUnavailable,
    ImageProfileValidationState,
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


def _truncated_png_bytes(width: int, height: int) -> bytes:
    """PNG bytes whose header parses but whose pixels cannot be fully verified."""
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (width, height), (12, 34, 56)).save(out, format="PNG")
    truncated = out.getvalue()[:-10]
    with image_module.open(io.BytesIO(truncated)) as image:
        assert image.size == (width, height)
    return truncated


def _hub_http_error(status: int) -> HfHubHTTPError:
    response = SimpleNamespace(status_code=status, headers={}, request=SimpleNamespace())
    return HfHubHTTPError("hub request failed", response=response)


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
    `AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")` counting `<|image_pad|>` ids in the
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

    @pytest.mark.parametrize(
        ("config", "match"),
        [
            pytest.param(
                {"min_pixels": 65536, "size": {"shortest_edge": "ignored-if-not-validated"}},
                "shortest_edge",
                id="invalid-size-budget-even-when-the-root-budget-wins",
            ),
            pytest.param(
                {"min_pixels": 4096, "max_pixels": 1024},
                "min_pixels",
                id="inverted-pixel-budget",
            ),
            *(
                pytest.param({"size": value}, "size must be an object", id=f"non-object-size-{i}")
                for i, value in enumerate([[], "pixels", 1, False])
            ),
        ],
    )
    def test_rejects_an_unusable_published_budget(self, config, match):
        with pytest.raises(ImageGeometryUnavailable, match=match):
            geometry_from_preprocessor_config({"patch_size": 16, "merge_size": 2, **config})

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
        ("raised", "expected_plane_fault"),
        [
            # the hub reports both classes through one exception type, so only the status separates
            # "this model cannot be quoted" from "the hub is briefly unreachable".
            *(
                pytest.param(_hub_http_error(status), plane_fault, id=f"hub-{status}")
                for status, plane_fault in [
                    (401, False),
                    (403, False),
                    (404, False),
                    (408, True),
                    (425, True),
                    (429, True),
                    (500, True),
                    (503, True),
                ]
            ),
            pytest.param(httpx.ReadTimeout("timed out"), True, id="network-timeout"),
            pytest.param(
                LocalEntryNotFoundError("cache entry is unavailable"), True, id="cache-miss"
            ),
            pytest.param(
                EntryNotFoundError("preprocessor_config.json is missing"), False, id="no-hub-file"
            ),
        ],
    )
    def test_a_failed_download_is_classified_by_whether_retrying_could_help(
        self, monkeypatch, raised, expected_plane_fault
    ):
        failure = self._load_failure(monkeypatch, raised)

        assert failure.plane_fault is expected_plane_fault
        assert self._submission_status(failure) == (503 if expected_plane_fault else 400)

    @pytest.mark.parametrize(
        ("contents", "expected_plane_fault"),
        [
            # a download that "succeeded" but left nothing readable is the plane's own fault; a file
            # that read fine and is simply not usable geometry is the submission's.
            pytest.param(None, True, id="downloaded-path-does-not-exist"),
            pytest.param("{not-json", False, id="not-json"),
            pytest.param(json.dumps([]), False, id="not-an-object"),
            pytest.param(json.dumps({"patch_size": 16, "merge_size": 0}), False, id="bad-geometry"),
        ],
    )
    def test_a_downloaded_config_is_classified_by_whether_it_could_be_read(
        self, monkeypatch, tmp_path, contents, expected_plane_fault
    ):
        config_path = tmp_path / "preprocessor_config.json"
        if contents is not None:
            config_path.write_text(contents)
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda **_kwargs: str(config_path),
        )

        with pytest.raises(ImageGeometryUnavailable) as excinfo:
            load_image_geometry("org/model")

        assert excinfo.value.plane_fault is expected_plane_fault
        assert self._submission_status(excinfo.value) == (503 if expected_plane_fault else 400)


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
        descriptors = [
            multimodal.normalize_image_source(_png_bytes(640, 480), None),
            multimodal.normalize_image_source(_png_bytes(56, 56), None),
        ]
        assert descriptor_pad_tokens(
            descriptors, None, QWEN_GEOMETRY, ImageProfileValidationState()
        ) == [300, 64]

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(b"notanimage", id="not-an-image-at-all"),
            # a truncated bitmap still parses a valid header, so a dimensions-only check would
            # accept it and the worker would then fail on a row the quote already charged for.
            pytest.param(_truncated_png_bytes(64, 64), id="truncated-with-valid-dimensions"),
        ],
    )
    def test_rejects_a_payload_that_cannot_be_fully_decoded(self, payload):
        descriptor = json.dumps(
            {"kind": "bytes", "value": base64.b64encode(payload).decode("ascii")}
        )
        with pytest.raises(ValueError, match="not a valid image"):
            descriptor_pad_tokens([descriptor], None, QWEN_GEOMETRY, ImageProfileValidationState())

    def test_rejects_a_deferred_truncated_path_without_committing_state(
        self, tmp_path, monkeypatch
    ):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        corrupt_path = dataset / "corrupt.png"
        corrupt_path.write_bytes(_truncated_png_bytes(64, 64))
        reads = []
        real_read_bytes = type(corrupt_path).read_bytes

        def count_read(path):
            if path.resolve() == corrupt_path.resolve():
                reads.append(path.resolve())
            return real_read_bytes(path)

        monkeypatch.setattr(type(corrupt_path), "read_bytes", count_read)
        state = ImageProfileValidationState()
        normalized = multimodal.normalize_prompt_images(
            {"image": "dataset/corrupt.png"},
            [{"role": "user", "content": "describe"}],
            tmp_path,
            defer_validation=True,
        )

        assert reads == []
        with pytest.raises(ValueError, match="not a valid image"):
            descriptor_pad_tokens(normalized.descriptors, tmp_path, QWEN_GEOMETRY, state)

        assert reads == [corrupt_path.resolve()]
        assert state.descriptor_metadata == {}
        assert state.decoded_work_bytes == 0

    def test_rejects_aggregate_decoded_bytes_before_full_decode(self, monkeypatch):
        data = _png_bytes(10, 10)
        descriptors = [multimodal.normalize_image_source(data, None) for _ in range(2)]
        monkeypatch.setattr(image_tokens, "MAX_TOTAL_DECODED_BYTES", 599)
        monkeypatch.setattr(
            image_tokens,
            "decode_descriptor_pixels",
            lambda _data: (_ for _ in ()).throw(AssertionError("full decode reached")),
        )

        with pytest.raises(ValueError, match="decoded images"):
            descriptor_pad_tokens(descriptors, None, QWEN_GEOMETRY, ImageProfileValidationState())

    def test_source_limit_stops_before_reading_the_next_distinct_descriptor(self, monkeypatch):
        data = _png_bytes(10, 10)
        first = multimodal.normalize_image_source(data, None)
        second = multimodal.normalize_image_source(_png_bytes(11, 10), None)
        descriptors = [first, first, second]
        reads = []
        real_read = image_tokens.read_descriptor_source

        def record_read(descriptor, package_root):
            reads.append(descriptor)
            return real_read(descriptor, package_root)

        monkeypatch.setattr(image_tokens, "MAX_TOTAL_IMAGE_SOURCE_BYTES", len(data) * 2 - 1)
        monkeypatch.setattr(image_tokens, "read_descriptor_source", record_read)
        monkeypatch.setattr(
            image_tokens,
            "decode_descriptor_pixels",
            lambda _data: (_ for _ in ()).throw(AssertionError("full decode reached")),
        )

        with pytest.raises(ValueError, match="image sources"):
            descriptor_pad_tokens(descriptors, None, QWEN_GEOMETRY, ImageProfileValidationState())

        assert reads == [first]

    def test_failed_row_commits_no_partial_profile_cache_accounting(self, monkeypatch):
        valid = multimodal.normalize_image_source(_png_bytes(10, 10), None)
        corrupt_data = _truncated_png_bytes(64, 64)
        corrupt = json.dumps(
            {"kind": "bytes", "value": base64.b64encode(corrupt_data).decode("ascii")}
        )
        successful_decodes = []
        real_decode = multimodal._decode_image_bytes

        def count_successful_decode(data):
            image = real_decode(data)
            successful_decodes.append(image.size)
            image.close()

        monkeypatch.setattr(image_tokens, "decode_descriptor_pixels", count_successful_decode)
        state = ImageProfileValidationState()

        with pytest.raises(ValueError, match="not a valid image"):
            descriptor_pad_tokens([valid, corrupt], None, QWEN_GEOMETRY, state)

        assert state.descriptor_metadata == {}
        assert state.decoded_work_bytes == 0
        assert descriptor_pad_tokens([valid], None, QWEN_GEOMETRY, state) == [64]
        assert successful_decodes == [(10, 10)]
        assert state.decoded_work_bytes == 900
