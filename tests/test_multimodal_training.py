from __future__ import annotations

import base64
import dataclasses
import io
import json
import threading
import urllib.parse
from types import SimpleNamespace

import pytest

import flash.engine.worker.train.entry.rl_train_runner as rl_train_runner
import flash.engine.worker.train.rl.launch.inputs as rl_inputs
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.lifecycle as runner_lifecycle
from flash.content import image_descriptors as _image_descriptors
from flash.content import multimodal as mm
from tests._helpers.profile import attach_sft_profile

# one text row and one image row: the shape a ceiling-based quote cannot describe.
_MIXED_RECORDS = [
    {"input": "text", "output": "answer"},
    {"input": "image", "output": "red", "image": "dataset/red.png"},
]


def _png_bytes(color=(255, 0, 0), size=(2, 2)) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", size, color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _oriented_jpeg_bytes() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    source = image_module.new("RGB", (3, 2))
    source.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    exif = image_module.Exif()
    exif[274] = 6
    out = io.BytesIO()
    source.save(out, format="JPEG", quality=100, subsampling=0, exif=exif)
    source.close()
    return out.getvalue()


def _package(tmp_path):
    root = tmp_path / "env"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    image = dataset / "red.png"
    image.write_bytes(_png_bytes())
    return root, image


def test_normalizes_openai_blocks_and_top_level_images_without_mutating_record(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    record = {
        "input": "describe the colors",
        "image": "dataset/red.png",
        "images": [_data_uri(data)],
        "reward_metadata": {"expected": "red"},
    }
    original = dict(record)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "image", "image": data},
                {"type": "input_image", "input_image": _data_uri(data), "detail": "low"},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]

    normalized = mm.normalize_prompt_images(record, messages, root)

    assert record == original
    assert len(normalized.descriptors) == 4
    assert [block["type"] for block in normalized.messages[0]["content"]] == [
        "text",
        "image",
        "image",
        "image",
        "image",
    ]
    assert all(isinstance(value, str) for value in normalized.descriptors)
    assert record["reward_metadata"] == {"expected": "red"}


def test_mixed_placeholders_keep_source_order_and_reject_extra_top_level_images(tmp_path):
    root, _image = _package(tmp_path)
    red = _png_bytes((255, 0, 0))
    green = _png_bytes((0, 255, 0))
    blue = _png_bytes((0, 0, 255))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image_url", "image_url": _data_uri(green)},
                {"type": "image"},
            ],
        }
    ]

    normalized = mm.normalize_prompt_images({"images": [red, blue]}, messages, root)
    decoded = mm.decode_image_descriptors(normalized.descriptors, root)
    assert [image.getpixel((0, 0)) for image in decoded] == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    extra_messages = [{"role": "user", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="extra top-level image"):
        mm.normalize_prompt_images({"images": [red, blue]}, extra_messages, root)


def test_supported_source_forms_decode_from_arrow_safe_descriptors(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    root, _image = _package(tmp_path)
    data = _png_bytes()
    pil = image_module.open(io.BytesIO(data))
    sources = [data, bytearray(data), pil, _data_uri(data), "dataset/red.png"]

    descriptors = [mm.normalize_image_source(source, root) for source in sources]

    assert all(isinstance(value, str) for value in descriptors)
    assert all(set(json.loads(value)) == {"kind", "value"} for value in descriptors)
    decoded = [mm.decode_image_descriptors([descriptor], root)[0] for descriptor in descriptors]
    try:
        assert [image.size for image in decoded] == [(2, 2)] * len(sources)
    finally:
        for image in decoded:
            image.close()


def test_image_descriptors_convert_to_base64_data_uris_without_paths(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptors = [
        mm.normalize_image_source(data, root),
        mm.normalize_image_source("dataset/red.png", root),
    ]

    uris = mm.image_descriptors_to_data_uris(descriptors, root)

    assert len(uris) == 2
    assert all(uri.startswith("data:image/png;base64,") for uri in uris)
    assert all("dataset/red.png" not in uri for uri in uris)
    assert [base64.b64decode(uri.split(",", 1)[1], validate=True) for uri in uris] == [
        data,
        data,
    ]


def test_cumulative_descriptor_validation_and_canonical_messages_are_immutable(tmp_path):
    root, _image = _package(tmp_path)
    first = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "first"},
                    {"type": "input_image", "input_image": _data_uri(_png_bytes())},
                ],
            }
        ],
        root,
    )
    snapshot = json.loads(json.dumps(first.messages))
    second = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": _data_uri(_png_bytes((0, 255, 0)))},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
        root,
    )
    descriptors = [*first.descriptors, *second.descriptors]

    validated = mm.validate_image_descriptors(descriptors, root)
    messages = mm.messages_with_image_data_uris(
        [*first.messages, {"role": "assistant", "content": "ok"}, *second.messages],
        descriptors,
        root,
    )

    assert first.messages == snapshot
    assert [item.pixels for item in validated] == [4, 4]
    assert [
        block["image_url"]["url"]
        for message in messages
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "image_url"
    ] == [item.data_uri for item in validated]


def test_data_uri_requires_strict_base64_and_mime_format_agreement(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    percent_uri = "data:image/png," + urllib.parse.quote_from_bytes(data)

    with pytest.raises(ValueError, match="base64 encoding"):
        mm.normalize_image_source(percent_uri, root)
    with pytest.raises(ValueError, match="invalid base64"):
        mm.normalize_image_source("data:image/png;base64,a G==", root)
    with pytest.raises(ValueError, match="unsupported MIME"):
        mm.normalize_image_source(
            "data:image/jpg;base64," + base64.b64encode(data).decode("ascii"), root
        )
    with pytest.raises(ValueError, match="MIME type does not match"):
        mm.normalize_image_source(
            "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), root
        )


@pytest.mark.parametrize("padding", [" ", "\n"])
def test_data_uri_rejects_leading_and_trailing_whitespace(tmp_path, padding):
    root, _image = _package(tmp_path)
    uri = _data_uri(_png_bytes())

    with pytest.raises(ValueError, match="data URI"):
        mm.normalize_image_source(padding + uri, root)
    with pytest.raises(ValueError, match=r"base64|exact data"):
        mm.normalize_image_source(uri + padding, root)


@pytest.mark.parametrize(
    ("image_format", "mime"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_data_uri_accepts_exact_serving_formats(tmp_path, image_format, mime):
    image_module = pytest.importorskip("PIL.Image")
    root, _image = _package(tmp_path)
    out = io.BytesIO()
    image_module.new("RGB", (2, 2), "red").save(out, format=image_format)
    uri = f"data:{mime};base64," + base64.b64encode(out.getvalue()).decode("ascii")

    descriptor = mm.normalize_image_source(uri, root)

    assert mm.image_descriptors_to_data_uris([descriptor], root)[0].startswith(
        f"data:{mime};base64,"
    )


def test_decode_applies_exif_orientation_and_reports_canonical_dimensions(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image_ops = pytest.importorskip("PIL.ImageOps")
    root, _image = _package(tmp_path)
    data = _oriented_jpeg_bytes()
    uri = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    descriptor = mm.normalize_image_source(uri, root)

    validated = mm.validate_image_descriptors([descriptor], root)[0]
    decoded = mm.decode_image_descriptors([descriptor], root)[0]
    with image_module.open(io.BytesIO(data)) as encoded:
        expected = image_ops.exif_transpose(encoded).convert("RGB")
    try:
        assert (validated.width, validated.height) == (2, 3)
        assert decoded.size == (2, 3)
        assert decoded.tobytes() == expected.tobytes()
    finally:
        decoded.close()
        expected.close()


def test_data_uri_rejects_oversized_header_before_descriptor_storage(tmp_path):
    root, _image = _package(tmp_path)
    header = "data:image/png;name=" + ("x" * mm.MAX_DATA_URI_HEADER_BYTES) + ";base64"
    uri = header + "," + base64.b64encode(_png_bytes()).decode("ascii")

    with pytest.raises(ValueError, match="header exceeds"):
        mm.normalize_image_source(uri, root)


def test_normalization_bounds_aggregate_encoded_descriptors(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptor_size = len(mm.normalize_image_source(_data_uri(data), root).encode("utf-8"))
    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES", descriptor_size * 2 - 1)

    with pytest.raises(ValueError, match="encoded image descriptors total"):
        mm.normalize_prompt_images(
            {"images": [_data_uri(data), _data_uri(data)]},
            [{"role": "user", "content": "compare"}],
            root,
        )


def test_relative_paths_are_confined_to_packaged_dataset_directory(tmp_path):
    root, _image = _package(tmp_path)
    outside = root / "outside.png"
    outside.write_bytes(_png_bytes())
    escaped = root.parent / "escaped.png"
    escaped.write_bytes(_png_bytes())
    (root / "dataset" / "escape-link.png").symlink_to(escaped)

    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("outside.png", root)
    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("../escaped.png", root)
    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("dataset/escape-link.png", root)
    with pytest.raises(ValueError, match="relative"):
        mm.normalize_image_source(str(outside.resolve()), root)
    with pytest.raises(ValueError, match="file://"):
        mm.normalize_image_source(outside.resolve().as_uri(), root)


@pytest.mark.parametrize(
    "url",
    ["http://images.example/red.png", "https://images.example/red.png"],
)
def test_remote_image_urls_are_always_rejected(monkeypatch, url):
    """Flash never fetches a user-supplied URL server-side, and nothing can re-enable it.

    The rejection is unconditional by construction: there is no env flag, no argument, and no
    module attribute left to flip, so a dataset carrying a remote URL fails at normalization
    rather than turning the trainer into an SSRF vector.
    """
    with pytest.raises(ValueError, match="remote image URLs are not supported"):
        mm.normalize_image_source(url, None)

    # the fetch machinery itself is gone, not merely gated
    for removed in ("_read_remote", "_remote_enabled", "_validate_remote_url", "REMOTE_IMAGE_ENV"):
        assert not hasattr(mm, removed)


def test_malformed_blocks_fail_clearly(tmp_path):
    root, _image = _package(tmp_path)
    with pytest.raises(ValueError, match="expected an object"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": ["bad"]}], root)
    with pytest.raises(ValueError, match="missing text"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": [{"type": "text"}]}], root)
    with pytest.raises(ValueError, match="exactly one image source"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": [{"type": "image_url"}]}], root)
    with pytest.raises(ValueError, match="unsupported content block"):
        mm.normalize_prompt_images(
            {}, [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], root
        )


_SDK_IMAGE_SOURCE_FORMS = [
    pytest.param("image_url", "image_url", id="image-url"),
    pytest.param("input_image", "image_url", id="input-image-image-url"),
    pytest.param("input_image", "input_image", id="input-image-input-image"),
    pytest.param("input_image", "image", id="input-image-image"),
    pytest.param("input_image", "url", id="input-image-url"),
    pytest.param("image", "image", id="image-image"),
    pytest.param("image", "image_url", id="image-image-url"),
    pytest.param("image", "url", id="image-url-key"),
]
_SDK_IMAGE_TARGET_FORMS = [
    *_SDK_IMAGE_SOURCE_FORMS,
    pytest.param("output_image", "image_url", id="output-image"),
]


@pytest.mark.parametrize(("block_type", "source_key"), _SDK_IMAGE_SOURCE_FORMS)
@pytest.mark.parametrize("object_source", [False, True], ids=["string", "object"])
def test_accepts_every_freesolo_041_image_source_spelling(block_type, source_key, object_source):
    uri = _data_uri(_png_bytes())
    source = {"url": uri, "detail": "auto"} if object_source else uri
    block = {"type": block_type, source_key: source}
    if block_type == "input_image":
        block["detail"] = "high"

    normalized = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    block,
                ],
            }
        ],
        None,
    )

    assert normalized.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image"},
            ],
        }
    ]
    assert len(normalized.descriptors) == 1


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_restricts_image_blocks_to_user_messages(role):
    with pytest.raises(ValueError, match="only in user messages"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": role,
                    "content": [{"type": "image_url", "image_url": _data_uri(_png_bytes())}],
                }
            ],
            None,
        )


def test_developer_role_is_canonicalized_to_system():
    normalized = mm.normalize_prompt_images(
        {},
        [
            {"role": "developer", "content": "be concise"},
            {"role": "user", "content": "answer"},
        ],
        None,
    )

    assert normalized.messages == [
        {"role": "system", "content": [{"type": "text", "text": "be concise"}]},
        {"role": "user", "content": [{"type": "text", "text": "answer"}]},
    ]


def test_developer_role_cannot_carry_images():
    with pytest.raises(ValueError, match="only in user messages"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "developer",
                    "content": [{"type": "image_url", "image_url": _data_uri(_png_bytes())}],
                }
            ],
            None,
        )


@pytest.mark.parametrize("role", [" user ", "User", "", "moderator", None])
def test_rejects_noncanonical_or_unsupported_roles(role):
    with pytest.raises(ValueError, match="role must be system, user, assistant, or tool"):
        mm.normalize_prompt_images(
            {},
            [{"role": role, "content": "hello"}],
            None,
        )


def test_rejects_ambiguous_or_invalid_sdk_image_sources():
    uri = _data_uri(_png_bytes())
    with pytest.raises(ValueError, match="exactly one image source"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": uri,
                            "input_image": uri,
                        }
                    ],
                }
            ],
            None,
        )
    with pytest.raises(ValueError, match="detail must"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "input_image": uri, "detail": "tiny"}],
                }
            ],
            None,
        )


def test_count_source_byte_pixel_and_decoded_byte_limits(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    messages = [{"role": "user", "content": "what color?"}]

    assert mm.MAX_IMAGES_PER_EXAMPLE == 4
    assert mm.MAX_IMAGE_SOURCE_BYTES == 8 * 1024 * 1024
    assert mm.MAX_TOTAL_IMAGE_SOURCE_BYTES == 16 * 1024 * 1024
    assert mm.MAX_IMAGE_PIXELS == 6_710_886
    assert mm.MAX_TOTAL_DECODED_BYTES == 64 * 1024 * 1024

    monkeypatch.setattr(mm, "MAX_IMAGES_PER_EXAMPLE", 1)
    with pytest.raises(ValueError, match="image limit"):
        mm.normalize_prompt_images({"images": [data, data]}, messages, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", len(data) - 1)
    with pytest.raises(ValueError, match="source exceeds"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", 3)
    with pytest.raises(ValueError, match="source exceeds"):
        mm.normalize_image_source(_data_uri(b"four"), root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(mm, "MAX_IMAGES_PER_EXAMPLE", 4)
    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_SOURCE_BYTES", len(data))
    with pytest.raises(ValueError, match="image sources total"):
        mm.normalize_prompt_images({"images": [data, data]}, messages, root)

    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_SOURCE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(mm, "MAX_IMAGE_WIDTH", 1)
    with pytest.raises(ValueError, match="dimensions"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_WIDTH", 8192)
    monkeypatch.setattr(mm, "MAX_IMAGE_PIXELS", 3)
    with pytest.raises(ValueError, match="pixel limit"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_PIXELS", 6_710_886)
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", 1)
    descriptor = mm.normalize_image_source(data, root)
    with pytest.raises(ValueError, match="decoded images"):
        mm.decode_image_descriptors([descriptor], root)


def test_an_image_at_the_advertised_pixel_cap_survives_the_decoded_memory_guard(
    monkeypatch, tmp_path
):
    """A real image exactly at the pixel cap must pass the memory guard that runs after it.

    The two limits are enforced in sequence, so a pixel cap the memory guard would always reject
    is not a limit at all -- it is a promise the validator breaks. That was the live bug: with the
    cap chosen independently of the budget, a 4K RGB screenshot (8.3 MP) sat under an advertised
    16.7 MP cap and still failed at ~71 MiB against 64 MiB.

    This pushes an actual image through `validate_image_descriptors` rather than asserting the
    arithmetic. Asserting `MAX_IMAGE_PIXELS * WORST_BYTES_PER_PIXEL <= MAX_TOTAL_DECODED_BYTES`
    looks like it pins the derivation, but for `cap = budget // worst` that inequality is a
    property of floor division -- it holds for every budget and every factor, so it stays green
    even when the guard's own formula is changed out from under it. Only a real decode can tell
    whether the number the guard computes agrees with the number the cap was derived from.

    The cap is scaled down here (via the real limit constants) so the test decodes a small image
    instead of allocating 6.7 MP; the ratio between cap and budget is what is under test, and it
    is preserved exactly.
    """
    root, _image = _package(tmp_path)
    worst = _image_descriptors.WORST_BYTES_PER_PIXEL

    # a 4x4 RGBA image: the worst decoded mode, so its real peak is the one the cap assumes.
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGBA", (4, 4), (255, 0, 0, 255)).save(out, format="PNG")
    data = out.getvalue()
    pixels = 16

    # the cap, derived exactly as the module derives it, for a budget this image sits at.
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", pixels * worst)
    monkeypatch.setattr(mm, "MAX_IMAGE_PIXELS", mm.MAX_TOTAL_DECODED_BYTES // worst)
    assert pixels == mm.MAX_IMAGE_PIXELS

    # an image exactly at the cap must be accepted: it passes the pixel check by equality, and
    # its real decoded peak must fit the budget the cap came from.
    descriptor = mm.normalize_image_source(data, root)
    assert mm.decode_image_descriptors([descriptor], root)

    # and the cap is not safe merely by being tiny -- one pixel more must not fit the budget.
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", (pixels - 1) * worst)
    with pytest.raises(ValueError, match="decoded images"):
        mm.decode_image_descriptors([mm.normalize_image_source(data, root)], root)


def test_no_cumulative_pixel_cap_shadows_the_decoded_memory_budget():
    """Image sets are bounded by decoded memory, never by a total pixel count.

    A sum of pixels cannot bound decoded memory: cost depends on each image's decoded mode (1 to 4
    bytes per pixel) and on their order, and a pixel count carries neither. Every mode-blind total
    is therefore wrong in one direction. One low enough to be safe for four RGBA images rejects
    the same pixel count spread across cheaper modes; one high enough to admit those can never
    fire, because the memory guard rejects first. Concretely, with a 64 MiB budget the memory
    guard can never admit more than 19,984,521 total pixels, so any cap at or above that is dead
    code and any cap below it rejects sets the budget permits.

    The per-image cap is different and is kept: one image in the worst mode is exactly the case
    `WORST_BYTES_PER_PIXEL` describes, so there the derivation is sound.
    """
    assert not hasattr(mm, "MAX_TOTAL_IMAGE_PIXELS")
    assert not any(
        field.name == "max_total_pixels"
        for field in dataclasses.fields(_image_descriptors.ImageDescriptorLimits)
    )


def test_total_decoded_budget_is_checked_before_any_image_load_or_conversion(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptors = [mm.normalize_image_source(data, root), mm.normalize_image_source(data, root)]
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", 12)
    decode_calls = []

    def fail_decode(_data):
        decode_calls.append(True)
        raise AssertionError("image load and conversion must not start before aggregate preflight")

    monkeypatch.setattr(mm, "_decode_image_bytes", fail_decode)

    with pytest.raises(ValueError, match="decoded images"):
        mm.decode_image_descriptors(descriptors, root)
    assert decode_calls == []


def test_a_row_whose_completion_truncated_away_is_dropped_not_trained_on():
    """A row that keeps no unmasked, non-special target token teaches nothing and must be dropped.

    `sft_max_len` truncates from the right, so a long prompt can leave a row whose completion is
    gone entirely: every position is either prompt (loss_mask 0) or a structural special token.
    Training on it is not merely wasted -- the row still contributes its prompt to the batch, so the
    reported mask ratio and token counts describe a dataset the model never learned from, and a
    dataset that truncates away *every* completion must abort rather than run to completion having
    learned nothing.

    This was `filter_vlm_sft_rows`, which measured the same thing off the vision collator's labels.
    verl pre-tokenizes to `input_ids`/`loss_mask` instead, so the check moved to `has_real_target`
    -- same invariant, different representation, and nothing covered it after the move.
    """
    from flash.engine.worker.entry.sft import has_real_target

    eos = 2
    special = {eos}
    # a real target: one unmasked token that is not a special.
    assert has_real_target([9, 9, 7, eos], [0, 0, 1, 1], special)
    # completion truncated away: every unmasked position is prompt.
    assert not has_real_target([9, 9, 7, eos], [0, 0, 0, 0], special)
    # content-free completion: unmasked, but only the structural end token.
    assert not has_real_target([9, eos], [0, 1], special)
    # the mask -- not the token -- decides: the same ids masked as prompt are not a target.
    assert not has_real_target([7, 7], [0, 0], special)
    # and a special id that is NOT registered special still counts as real content.
    assert has_real_target([9, eos], [0, 1], set())


def test_grpo_rows_retain_arrow_safe_images_and_reward_examples(tmp_path):
    pytest.importorskip("datasets")
    from datasets import Dataset

    from flash.engine.worker.train.rl.launch.config import build_grpo_prompt_dataset

    root, _image = _package(tmp_path)
    descriptor = mm.normalize_image_source("dataset/red.png", root)
    text_example = {"input": "text?", "metadata": {"mixed": "text"}}
    image_example = {"input": "color?", "metadata": {"mixed": 7}}
    prompts = [
        {
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "text?"}]}],
            "images": [],
            "example": text_example,
        },
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}]}],
            "images": [descriptor],
            "example": image_example,
        },
    ]
    rows, examples = build_grpo_prompt_dataset(prompts)
    dataset = Dataset.from_list(rows)

    assert dataset[0]["images"] == []
    assert dataset[1]["images"] == [descriptor]
    assert examples == [text_example, image_example]
    assert examples[rows[1]["example_idx"]] is image_example


def test_sft_rejects_image_completion_when_prompt_is_text_only():
    from flash.engine.worker.entry.sft import _reject_image_completion

    completion = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "image_url", "image_url": {"url": "https://images.example/x.png"}},
            ],
        }
    ]

    with pytest.raises(ValueError, match="image-bearing SFT completions are not supported"):
        _reject_image_completion(completion, image_bearing=False)


@pytest.mark.parametrize(("block_type", "source_key"), _SDK_IMAGE_TARGET_FORMS)
@pytest.mark.parametrize("image_bearing", [False, True], ids=["text-prompt", "image-prompt"])
def test_sft_rejects_every_sdk_image_target_spelling(block_type, source_key, image_bearing):
    from flash.engine.worker.entry.sft import _reject_image_completion

    completion = [
        {
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "answer"},
                {"type": block_type, source_key: _data_uri(_png_bytes())},
            ],
        }
    ]

    with pytest.raises(ValueError, match="image-bearing SFT completions are not supported"):
        _reject_image_completion(completion, image_bearing=image_bearing)


def test_user_image_prompt_with_text_assistant_target_is_accepted():
    from flash.engine.worker.entry.sft import _reject_image_completion

    normalized = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "input_image": _data_uri(_png_bytes())},
                ],
            }
        ],
        None,
    )
    completion = [{"role": "assistant", "content": [{"type": "input_text", "text": "red"}]}]

    _reject_image_completion(completion, image_bearing=True)

    assert len(normalized.descriptors) == 1
    assert mm.text_only_prompt_messages(completion) == [{"role": "assistant", "content": "red"}]


def test_sft_mixed_text_completion_shapes_are_arrow_safe():
    import inspect

    pytest.importorskip("datasets")
    from datasets import Dataset

    from flash.engine.profiling import sft_workload

    completions = [
        [{"role": "assistant", "content": "red"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "blue"}]}],
    ]
    rows = [
        {
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "color?"}]}],
            "completion": mm.text_only_prompt_messages(completion),
            "images": [],
        }
        for completion in completions
    ]

    dataset = Dataset.from_list(rows)

    assert [row["completion"][0]["content"] for row in dataset] == ["red", "blue"]
    # the normalizer must actually be applied on the training path, not merely importable: a mixed
    # str/list `content` column makes Arrow infer a struct type and drops one shape at write time.
    assert (
        "completion_messages = text_only_prompt_messages(completion_messages)"
        in inspect.getsource(sft_workload)
    )


def test_image_teacher_prompt_uses_one_media_pad_per_descriptor_in_order():
    messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "middle"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        },
    ]

    rendered = mm.image_teacher_prompt_messages(messages, 2)

    assert rendered == [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": "before<|media_pad|>middle<|media_pad|>after",
        },
    ]
    assert rendered[1]["content"].count(mm.IMAGE_TEACHER_PLACEHOLDER) == 2
    with pytest.raises(ValueError, match="expected 1 normalized image descriptor"):
        mm.image_teacher_prompt_messages(messages, 1)


def test_image_teacher_prompt_rejects_the_placeholder_in_user_text():
    # the placeholder marks image positions and the client splits on EVERY occurrence, so one
    # sitting in the user's own text would be paired with an image that does not exist. reject it
    # here, naming the offending message, rather than failing downstream as a count mismatch that
    # reads like an internal bug. covers both the block-list and plain-string content shapes.
    blocks = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"what is {mm.IMAGE_TEACHER_PLACEHOLDER} here?"},
                {"type": "image"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.image_teacher_prompt_messages(blocks, 1)

    plain = [
        {"role": "system", "content": f"never emit {mm.IMAGE_TEACHER_PLACEHOLDER}"},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.image_teacher_prompt_messages(plain, 1)


def test_image_teacher_prompt_rejects_the_image_pad_token_in_user_text():
    # <|image_pad|> is the model's REAL expansion token, and the silent-drop guard counts its runs
    # in the returned prompt ids, requiring at least one run per supplied image. text containing the
    # literal token encodes to that same id, so it contributes a run the renderer never produced --
    # and that run can cover for an image the provider silently dropped, defeating the exact guard
    # the image path depends on. keeping it out of source text is what makes the count meaningful.
    blocks = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"the marker is {mm.IMAGE_PAD_TOKEN} ok"},
                {"type": "image"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.image_teacher_prompt_messages(blocks, 1)

    plain = [
        {"role": "system", "content": f"never emit {mm.IMAGE_PAD_TOKEN}"},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.image_teacher_prompt_messages(plain, 1)


def test_image_teacher_prompt_rejects_a_marker_split_across_adjacent_text_blocks():
    """A marker assembled from fragments is still a marker once the blocks are joined.

    Per-block validation cannot see this: "<|media_" and "pad|>" are each harmless text, and the
    reserved marker exists only in the concatenation. That is why the check runs over each RUN of
    consecutive text blocks rather than over one block at a time.

    The last case is the reason the unit is a run and not the whole joined string: an image block
    between the fragments puts the renderer's own placeholder there, so the user's text never
    forms a marker and the prompt is legitimate. A whole-string check would reject it, and a
    count-based check would accept the split-after-image case above it.
    """
    for fragments in (
        ["<|media_", "pad|>"],
        ["<|image_", "pad|>"],
        ["<|med", "ia_p", "ad|>"],
    ):
        split = [
            {
                "role": "user",
                "content": [{"type": "text", "text": piece} for piece in fragments]
                + [{"type": "image"}],
            }
        ]
        with pytest.raises(ValueError, match="reserved image marker"):
            mm.image_teacher_prompt_messages(split, 1)

    after_image = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "<|media_"},
                {"type": "text", "text": "pad|>"},
            ],
        }
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.image_teacher_prompt_messages(after_image, 1)

    around_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|media_"},
                {"type": "image"},
                {"type": "text", "text": "pad|>"},
            ],
        }
    ]
    rendered = mm.image_teacher_prompt_messages(around_image, 1)
    assert rendered[0]["content"] == f"<|media_{mm.IMAGE_TEACHER_PLACEHOLDER}pad|>"


def test_text_only_prompt_messages_drops_images_and_preserves_text_order():
    image_module = pytest.importorskip("PIL.Image")
    pil = image_module.new("RGB", (1, 1), "red")
    messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": [
                {"type": "text", "text": "first "},
                {"type": "image", "image": pil},
                {"type": "image_url", "image_url": {"url": "dataset/red.png"}},
                {"type": "text", "text": "second"},
                {"type": "input_image", "image_url": "dataset/blue.png"},
            ],
        },
        {"role": "user", "content": [{"type": "image", "image": "dataset/only.png"}]},
    ]

    stripped = mm.text_only_prompt_messages(messages)

    assert stripped == [
        {"role": "system", "content": "rules"},
        {"role": "user", "name": "viewer", "content": "first second"},
        {"role": "user", "content": ""},
    ]
    assert messages[1]["content"][1]["image"] is pil


def test_multimodal_algorithm_validation_requires_a_vision_teacher_after_model_validation():
    mm.validate_multimodal_training("Qwen/Qwen3.5-9B", "sft", None)
    mm.validate_multimodal_training("Qwen/Qwen3.5-9B", "grpo", None)
    mm.validate_multimodal_training("Qwen/Qwen3.5-9B", "opd", "qwen3-vl-235b")
    with pytest.raises(
        ValueError,
        match=r"requires.*qwen3-vl-235b.*selected teacher \'glm-5\.2\' cannot see images",
    ):
        mm.validate_multimodal_training("Qwen/Qwen3.5-9B", "opd", "glm-5.2")
    with pytest.raises(ValueError, match="does not support"):
        mm.validate_multimodal_training(
            "meta-llama/Llama-3.2-1B",
            "opd",
            "glm-5.2",
        )


def test_native_single_turn_image_grpo_suppresses_image_pad_generation():
    """An image run must not be able to generate the image-pad token itself. verl generates in its
    own subprocess, so the ban is injected as a rollout shim rather than a generate kwarg."""
    import inspect

    # the shim's own rendering is covered in test_rl_train.py; what belongs here is the multimodal
    # wiring -- the pad id comes from the PROCESSOR (a text run resolves none) and reaches the shim.
    resolver = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert "image_pad_token_id = resolve_image_pad_token_id(processor, tok)" in resolver

    entry = inspect.getsource(rl_train_runner._write_rl_plugin_config)
    assert '"image_pad_token_id": inp["image_pad_token_id"]' in entry


def test_image_opd_preflight_rejects_packaged_dataset_before_allocation(tmp_path):
    root, _image = _package(tmp_path)
    env_file = root / "environment.py"
    env_file.write_text("def load_environment(**kwargs):\n    return None\n")
    (root / "dataset" / "train.jsonl").write_text(
        json.dumps({"input": "color?", "output": "red", "image": "dataset/red.png"}) + "\n"
    )
    environment = SimpleNamespace(id=str(env_file), resolved_sha="", params={})
    supported = SimpleNamespace(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(teacher_model="qwen3-vl-235b"),
    )
    mm.preflight_validate_image_opd(supported)

    text_teacher = SimpleNamespace(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(teacher_model="kimi-k3"),
    )
    with pytest.raises(ValueError, match="selected teacher 'kimi-k3' cannot see images"):
        mm.preflight_validate_image_opd(text_teacher)

    unsupported = SimpleNamespace(
        model="meta-llama/Llama-3.2-1B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(teacher_model="qwen3-vl-235b"),
    )
    with pytest.raises(ValueError, match="does not support image-bearing"):
        mm.preflight_validate_image_opd(unsupported)


def test_image_opd_preflight_allows_inline_multi_turn_images_when_capabilities_match():
    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=SimpleNamespace(
            id="local",
            params={
                "multi_turn": True,
                "records": [
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": [{"type": "image"}],
                            }
                        ],
                        "image": _data_uri(_png_bytes()),
                    }
                ],
            },
        ),
        train=SimpleNamespace(teacher_model="qwen3-vl-235b"),
    )

    mm.preflight_validate_image_opd(spec, scan_packaged_environment=False)


def test_image_opd_preflight_allows_max_turns_on_a_single_turn_env():
    # max_turns is a turn CAP, not a multi-turn declaration: the worker derives multi_turn from the
    # env CLASS, never from params. rejecting on max_turns here would fail a job the worker runs.
    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        environment=SimpleNamespace(
            id="local",
            params={
                "max_turns": 4,
                "records": [
                    {
                        "input": [{"role": "user", "content": [{"type": "image"}]}],
                        "image": _data_uri(_png_bytes()),
                    }
                ],
            },
        ),
        train=SimpleNamespace(teacher_model="qwen3-vl-235b"),
    )

    mm.preflight_validate_image_opd(spec)


@pytest.mark.parametrize("record_source", ["inline", "packaged"])
def test_image_opd_preflight_limits_scan_to_max_examples(tmp_path, record_source):
    records = [
        {"input": "text only", "output": "answer"},
        {"input": "image", "output": "red", "image": "dataset/red.png"},
    ]
    if record_source == "inline":
        environment = SimpleNamespace(
            id="local",
            resolved_sha="",
            params={"records": records},
        )
    else:
        root = tmp_path / "env"
        (root / "dataset").mkdir(parents=True)
        env_file = root / "environment.py"
        env_file.write_text("def load_environment(**kwargs):\n    return None\n")
        (root / "dataset" / "train.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        environment = SimpleNamespace(id=str(env_file), resolved_sha="", params={})
    spec = SimpleNamespace(
        model="meta-llama/Llama-3.2-1B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(max_examples=1),
    )

    mm.preflight_validate_image_opd(spec)


@pytest.mark.parametrize("background", [False, True])
def test_image_opd_submit_preflight_rejects_text_teacher_before_state_mutation(
    monkeypatch, tmp_path, background
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))

    def fail(*args, **kwargs):
        raise AssertionError("rejected submit must not mutate warm-start state or reach providers")

    monkeypatch.setattr(runner_preparation, "_mark_warmstart_source", fail)
    monkeypatch.setattr(runner_lifecycle, "_run_job", fail)
    monkeypatch.setattr(runner_lifecycle, "_run_job_background", fail)
    monkeypatch.setattr(threading, "Thread", fail)

    spec = JobSpec.from_dict(
        {
            "run_id": f"image-opd-{'background' if background else 'sync'}",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "opd",
            "environment": {
                "id": "local",
                "params": {
                    "records": [{"input": "color?", "output": "red", "image": "dataset/red.png"}]
                },
            },
            "train": {"epochs": 1, "max_examples": 1, "teacher_model": "kimi-k3"},
        }
    )

    with pytest.raises(ValueError, match="selected teacher 'kimi-k3' cannot see images"):
        runner_submit.submit_job(spec, background=background)
    with pytest.raises(FileNotFoundError):
        runner_status.get_status(spec.run_id)


def test_grpo_prices_the_full_context_budget_for_image_and_mixed_rows():
    """An image prompt occupies its context budget, so grpo prices the budget, not the text length.

    sft used to be asserted here against the same ceiling. It no longer prices from the ceiling at
    all: the workload profile measures the tokens the rows actually produce, which for a mixed
    dataset is the whole point of measuring. The companion test below holds the multimodal half of
    that -- an image sft run cannot be quoted from an assumed context.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec

    grpo_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "environment": {"id": "local", "params": {"records": _MIXED_RECORDS}},
            "train": {
                "epochs": 1,
                "max_examples": 2,
                "max_context_tokens": 2048,
                "max_completion_tokens": 64,
            },
        }
    )

    assert runconfig_from_spec(grpo_spec).seq_len == 2048


def test_image_sft_cannot_be_priced_from_an_assumed_context():
    """The failure mode this replaces: quoting a mixed dataset off max_context_tokens.

    Image rows and text rows produce wildly different token counts, so a ceiling-based quote for a
    mixed dataset is a guess wearing an exact number. Pricing now requires the profile that
    tokenized these exact rows, and without one the quote fails rather than defaulting.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import runconfig_from_spec
    from flash.engine.profiling.workload_profile import WorkloadProfileMismatch

    sft_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "environment": {"id": "local", "params": {"records": _MIXED_RECORDS}},
            "train": {"epochs": 1, "max_examples": 2, "max_context_tokens": 1536},
        }
    )

    with pytest.raises(WorkloadProfileMismatch):
        runconfig_from_spec(sft_spec)

    # with the measurement attached, the priced length is the profile's, never the config ceiling.
    priced = runconfig_from_spec(attach_sft_profile(sft_spec))
    assert priced.seq_len == attach_sft_profile(sft_spec).workload_profile["max_length"]


def test_catalog_image_capability_does_not_change_public_rows():
    from flash.core.catalog import public_model_rows, supports_image_training

    assert supports_image_training("Qwen/Qwen3.5-9B")
    assert supports_image_training("Qwen/Qwen3.8-27B")
    assert not supports_image_training("meta-llama/Llama-3.2-1B")
    forbidden = {"modalities", "multimodal", "supports_images", "image_training"}
    assert all(not (forbidden & set(row)) for row in public_model_rows())
