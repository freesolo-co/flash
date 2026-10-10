from __future__ import annotations

import asyncio
import base64
import copy
import importlib
import io
import os
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from flash.content import multimodal as mm

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_IMAGE_CASES = [
    ((56, 56), "red", 64),
    ((300, 200), "green", 70),
    ((640, 480), "blue", 300),
    ((1024, 768), "yellow", 768),
]
_PROBE = "flash-incremental-image-glue-probe"


def _flat_ids(value) -> list[int]:
    if isinstance(value, Mapping) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token_id) for token_id in value]


def _image_bytes(size: tuple[int, int], color: str) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def _nonuniform_32_png() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (32, 32))
    image.putdata(
        [
            (
                (x * 17 + y * 3) % 256,
                (x * 5 + y * 19) % 256,
                (x * 11 + y * 7) % 256,
            )
            for y in range(32)
            for x in range(32)
        ]
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    image.close()
    return out.getvalue()


def _qwen_sized_image(processor, image):
    encoded = processor.image_processor(images=[image], return_tensors="np")
    _, grid_height, grid_width = encoded["image_grid_thw"][0].tolist()
    patch_size = int(processor.image_processor.patch_size)
    return image.resize((grid_width * patch_size, grid_height * patch_size))


def _image_pad_runs(token_ids: list[int], image_pad_id: int) -> list[int]:
    runs = []
    index = 0
    while index < len(token_ids):
        if token_ids[index] != image_pad_id:
            index += 1
            continue
        end = index + 1
        while end < len(token_ids) and token_ids[end] == image_pad_id:
            end += 1
        runs.append(end - index)
        index = end
    return runs


def _find_unique_subsequence(values: list[int], needle: list[int]) -> int:
    positions = [
        index
        for index in range(len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]
    if len(positions) != 1:
        raise AssertionError(f"processor probe occurred {len(positions)} times")
    return positions[0]


def _processor_ids(processor, messages: list[dict]) -> list[int]:
    return _flat_ids(
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )
    )


def _parent_glue_ids(processor, messages: list[dict], descriptors: list[str]) -> list[int]:
    canonical = mm.messages_with_image_data_uris(messages, descriptors, None)
    probe_messages = [{"role": "assistant", "content": _PROBE}, *canonical]
    rendered = processor.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    first = rendered.find(_PROBE)
    if first < 0 or rendered.find(_PROBE, first + len(_PROBE)) >= 0:
        raise AssertionError("processor did not render the glue probe exactly once")
    suffix = rendered[first + len(_PROBE) :]
    images = mm.decode_image_descriptors(descriptors, None)
    try:
        return _flat_ids(
            processor(
                text=[suffix],
                images=images,
                videos=None,
                return_tensors=None,
            )
        )
    finally:
        for image in images:
            image.close()


def _child_glue_ids(processor, messages: list[dict], descriptors: list[str]) -> list[int]:
    canonical = mm.messages_with_image_data_uris(messages, descriptors, None)
    token_ids = _processor_ids(
        processor,
        [{"role": "assistant", "content": _PROBE}, *canonical],
    )
    probe_ids = _flat_ids(processor.tokenizer(_PROBE, add_special_tokens=False))
    start = _find_unique_subsequence(token_ids, probe_ids)
    return token_ids[start + len(probe_ids) :]


@pytest.fixture(scope="module")
def real_processor():
    if os.environ.get("FLASH_IMAGE_INCREMENTAL_PROCESSOR") != "1":
        pytest.skip(
            "real incremental image processor gate disabled; set "
            "FLASH_IMAGE_INCREMENTAL_PROCESSOR=1"
        )
    try:
        importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        pytest.fail(
            f"real processor gate requires cached dependencies for "
            f"{MODEL_ID}@{MODEL_REVISION}: {exc}",
            pytrace=False,
        )
    try:
        return transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.fail(
            f"failed to load cached real processor {MODEL_ID}@{MODEL_REVISION}: {exc}",
            pytrace=False,
        )


def test_dynamic_image_decoder_accepts_only_canonical_managed_data_uris():
    from PIL import Image

    from flash.engine.worker.train.core.child.glue import _decode_image_data_uris

    supported = []
    for image_format, media_type in (
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
    ):
        out = io.BytesIO()
        Image.new("RGB", (3, 2), "purple").save(out, format=image_format)
        supported.append(
            f"data:{media_type};base64,{base64.b64encode(out.getvalue()).decode('ascii')}"
        )
    decoded = _decode_image_data_uris(supported)
    try:
        assert [image.mode for image in decoded] == ["RGB", "RGB", "RGB"]
        assert [image.size for image in decoded] == [(3, 2), (3, 2), (3, 2)]
    finally:
        for image in decoded:
            image.close()

    png_payload = supported[0].split(",", 1)[1]
    rejected = [
        ("https://images.example/reply.png", "remote media"),
        (f"data:image/gif;base64,{png_payload}", "canonical base64 PNG, JPEG, or WebP"),
        ("data:image/png;base64,not base64", "invalid base64"),
        (f"data:image/png,{png_payload}", "canonical base64 PNG, JPEG, or WebP"),
        (f"data:image/jpeg;base64,{png_payload}", "MIME type does not match"),
    ]
    for uri, message in rejected:
        with pytest.raises(ValueError, match=message):
            _decode_image_data_uris([uri])


def test_dynamic_image_decoder_matches_parent_retained_rgb_budget(monkeypatch):
    from flash.engine.worker.train.core.child import glue as child_glue

    uris = [
        "data:image/png;base64," + base64.b64encode(_image_bytes((4, 4), color)).decode("ascii")
        for color in ("red", "blue")
    ]
    monkeypatch.setattr(child_glue, "_MAX_TOTAL_DECODED_BYTES", 200)
    decoded = child_glue._decode_image_data_uris(uris)
    try:
        assert [image.size for image in decoded] == [(4, 4), (4, 4)]
    finally:
        for image in decoded:
            image.close()


def test_dynamic_image_decoder_rejects_memory_before_pixel_load(monkeypatch):
    from PIL import Image, PngImagePlugin

    from flash.engine.worker.train.core.child import glue as child_glue

    out = io.BytesIO()
    Image.new("RGBA", (4, 4), (1, 2, 3, 4)).save(out, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    monkeypatch.setattr(child_glue, "_MAX_TOTAL_DECODED_BYTES", 100)

    def reject_load(_image):
        raise AssertionError("pixel load occurred before the decoded-memory gate")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", reject_load)
    with pytest.raises(ValueError, match="aggregate decoded-byte limit"):
        child_glue._decode_image_data_uris([uri])


def test_dynamic_image_decoder_closes_partial_results_on_failure(monkeypatch):
    from flash.engine.worker.train.core.child import glue as child_glue

    valid_uri = "data:image/png;base64," + base64.b64encode(_image_bytes((2, 2), "red")).decode(
        "ascii"
    )
    captured = []
    decode_one = child_glue._decode_validated_dynamic_image

    def recording_decode(data, expected_format):
        if captured:
            raise ValueError("injected second image decode failure")
        image = decode_one(data, expected_format)
        captured.append(image)
        return image

    monkeypatch.setattr(child_glue, "_decode_validated_dynamic_image", recording_decode)
    with pytest.raises(ValueError, match="injected second image decode failure"):
        child_glue._decode_image_data_uris([valid_uri, valid_uri])
    assert len(captured) == 1
    with pytest.raises(ValueError, match="closed image"):
        captured[0].getpixel((0, 0))


def test_dynamic_image_transport_requires_exact_block_count():
    from flash.engine.worker.train.core.child.glue import EnvGlueProcessor

    class _Loop:
        def __init__(self):
            self.processor = object()
            self.tokenizer = object()
            self.apply_chat_template_kwargs = {}

    glue = EnvGlueProcessor(_Loop(), thinking=False)
    message = [{"role": "user", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="image block count"):
        asyncio.run(glue(message, []))
    with pytest.raises(ValueError, match="image block count"):
        asyncio.run(glue([{"role": "user", "content": "text"}], ["data:image/png;base64,"]))


def test_source_shipped_glue_imports_under_a_flat_child_name():
    from flash.engine.worker.train.core.child import glue as child_glue

    spec = importlib.util.spec_from_file_location("flash_env_glue_flat_test", child_glue.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EnvGlueProcessor.__name__ == "EnvGlueProcessor"


def test_processor_digest_detects_a_qwen_intermediate_resize(real_processor):
    from flash.engine.worker.train.core.child.glue import (
        parent_image_digests,
        processor_image_digests,
    )

    descriptor = mm.normalize_image_source(_nonuniform_32_png(), None)
    raw = mm.decode_image_descriptors([descriptor], None)
    try:
        resized = [_qwen_sized_image(real_processor, raw[0])]
        try:
            assert raw[0].size == (32, 32)
            assert resized[0].size != raw[0].size
            assert parent_image_digests(real_processor, [descriptor], None) != (
                processor_image_digests(real_processor, resized)
            )
        finally:
            resized[0].close()
    finally:
        raw[0].close()

    class _TextOnlyProcessor:
        @property
        def image_processor(self):
            raise AssertionError("text-only digest path touched the image processor")

    assert processor_image_digests(_TextOnlyProcessor(), []) == []


def test_processor_digest_separates_images_that_differ_only_by_grid(real_processor):
    """a transposed solid image has identical pixel bytes but a different patch grid.

    the digest must authenticate image_grid_thw, not pixel_values alone, or two
    differently shaped observations collide and media identity stops binding order.
    """
    from flash.engine.worker.train.core.child.glue import parent_image_digests

    portrait = mm.normalize_image_source(_image_bytes((56, 112), "red"), None)
    landscape = mm.normalize_image_source(_image_bytes((112, 56), "red"), None)

    encoded = []
    for descriptor in (portrait, landscape):
        decoded = mm.decode_image_descriptors([descriptor], None)
        try:
            fields = real_processor.image_processor(images=[decoded[0]], return_tensors="np")
            encoded.append(
                (
                    fields["pixel_values"].shape,
                    fields["pixel_values"].tobytes(order="C"),
                    fields["image_grid_thw"].tolist(),
                )
            )
        finally:
            decoded[0].close()

    # the collision must remain constructible, otherwise this test proves nothing.
    assert encoded[0][0] == encoded[1][0]
    assert encoded[0][1] == encoded[1][1]
    assert encoded[0][2] != encoded[1][2]

    assert (
        parent_image_digests(real_processor, [portrait], None)[0]
        != parent_image_digests(real_processor, [landscape], None)[0]
    )


def test_real_processor_preserves_dynamic_reply_pixels_and_media_identity(real_processor):
    from flash.engine.worker.train.core.child.glue import (
        EnvGlueProcessor,
        parent_image_digests,
    )
    from flash.engine.worker.train.opd.child.multiturn import _validate_opd_reply_media
    from flash.engine.worker.train.rl.child.multiturn import _validate_grpo_reply_media

    class _ProcessorProbe:
        def __init__(self, delegate):
            self.delegate = delegate
            self.mm_processor_calls = []

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def __call__(self, *args, **kwargs):
            self.mm_processor_calls.append(kwargs.pop("flash_probe", None))
            return self.delegate(*args, **kwargs)

    class _Loop:
        def __init__(self, processor):
            self.processor = _ProcessorProbe(processor)
            self.tokenizer = processor.tokenizer
            self.apply_chat_template_kwargs = {}
            self.process_multi_modal_info_calls = 0

        def _get_mm_processor_kwargs(self, _audio_data=None):
            return {"flash_probe": True}

        async def process_multi_modal_info(self, _messages):
            self.process_multi_modal_info_calls += 1
            raise AssertionError("dynamic reply images reached process_multi_modal_info")

    observation = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect the nonuniform image"},
                    {"type": "input_image", "input_image": _nonuniform_32_png()},
                ],
            }
        ],
        None,
    )
    loop = _Loop(real_processor)
    result = asyncio.run(
        EnvGlueProcessor(loop, thinking=False)(
            observation.messages,
            mm.image_descriptors_to_data_uris(observation.descriptors, None),
        )
    )
    try:
        expected_digests = parent_image_digests(real_processor, observation.descriptors, None)
        assert result.token_ids == _parent_glue_ids(
            real_processor,
            observation.messages,
            observation.descriptors,
        )
        assert [image.size for image in result.images] == [(32, 32)]
        assert result.image_digests == expected_digests
        assert loop.processor.mm_processor_calls == [True]
        assert loop.process_multi_modal_info_calls == 0

        prompt = SimpleNamespace(image_digests=[])
        step = {"image_count": 1, "image_digests": expected_digests}
        _validate_opd_reply_media(prompt, result, step)
        _validate_grpo_reply_media(prompt, result, step)
        assert result.images[0].getpixel((7, 11))
    finally:
        for image in result.images:
            image.close()


def test_real_processor_proves_incremental_four_image_glue(real_processor, monkeypatch):
    processor = real_processor
    image_pad_id = mm.resolve_image_pad_token_id(processor, processor.tokenizer)
    cumulative_messages: list[dict] = []
    cumulative_descriptors: list[str] = []
    snapshots: list[tuple[list[dict], list[str]]] = []
    expected_runs: list[int] = []
    processor_calls = 0

    for index, (size, color, expected_run) in enumerate(_IMAGE_CASES):
        observation = mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"observation {index + 1}"},
                        {"type": "input_image", "input_image": _image_bytes(size, color)},
                    ],
                }
            ],
            None,
        )
        if index:
            parent_glue = _parent_glue_ids(
                processor,
                observation.messages,
                observation.descriptors,
            )
            child_glue = _child_glue_ids(
                processor,
                observation.messages,
                observation.descriptors,
            )
            assert parent_glue == child_glue
            cumulative_messages.append({"role": "assistant", "content": f"answer {index}"})
        cumulative_messages.extend(copy.deepcopy(observation.messages))
        cumulative_descriptors.extend(observation.descriptors)
        canonical = mm.messages_with_image_data_uris(
            cumulative_messages,
            cumulative_descriptors,
            None,
        )
        token_ids = _processor_ids(processor, canonical)
        processor_calls += 1
        expected_runs.append(expected_run)

        assert len(cumulative_descriptors) == index + 1
        assert _image_pad_runs(token_ids, image_pad_id) == expected_runs
        snapshots.append((copy.deepcopy(cumulative_messages), list(cumulative_descriptors)))

    assert len(set(expected_runs)) == 4
    for messages, descriptors in snapshots:
        assert messages == cumulative_messages[: len(messages)]
        assert descriptors == cumulative_descriptors[: len(descriptors)]

    with pytest.raises(ValueError, match="only in user messages"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "input_image", "input_image": _image_bytes((8, 8), "black")}
                    ],
                }
            ],
            None,
        )

    fifth = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [{"type": "input_image", "input_image": _image_bytes((8, 8), "black")}],
            }
        ],
        None,
    )
    calls_before_fifth = processor_calls
    with pytest.raises(ValueError, match="4-image limit"):
        mm.validate_image_descriptors(
            [*cumulative_descriptors, *fifth.descriptors],
            None,
        )
    assert processor_calls == calls_before_fifth

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network reached")),
    )
    with pytest.raises(ValueError, match="remote image URLs are not supported"):
        mm.normalize_prompt_images(
            {},
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "input_image": "https://images.example/remote.png",
                        }
                    ],
                }
            ],
            None,
        )
    assert processor_calls == calls_before_fifth


def test_child_glue_image_limits_match_the_parent_normalizer():
    """The child's copied limits must stay equal to the parent's.

    `glue.py` is source-copied into the verl child, where `flash` is not importable, so it restates
    the image limits as literals instead of importing them (see its module docstring). that is the
    right call, but it leaves two independent copies of one contract: raising a cap in
    `flash.content.multimodal` alone would let the parent normalize and ship an image the child then
    rejects mid-rollout, and lowering one alone would silently re-validate already-accepted media.
    nothing else pins the two together, so pin them here.
    """
    from flash.content import image_descriptors as _image_descriptors
    from flash.engine.worker.train.core.child import glue as child_glue

    assert mm.MAX_IMAGES_PER_EXAMPLE == child_glue._MAX_IMAGES_PER_REPLY
    assert mm.MAX_IMAGE_SOURCE_BYTES == child_glue._MAX_IMAGE_SOURCE_BYTES
    assert mm.MAX_TOTAL_IMAGE_SOURCE_BYTES == child_glue._MAX_TOTAL_IMAGE_SOURCE_BYTES
    assert mm.MAX_IMAGE_WIDTH == child_glue._MAX_IMAGE_WIDTH
    assert mm.MAX_IMAGE_HEIGHT == child_glue._MAX_IMAGE_HEIGHT
    assert mm.MAX_IMAGE_PIXELS == child_glue._MAX_IMAGE_PIXELS
    assert mm.MAX_TOTAL_DECODED_BYTES == child_glue._MAX_TOTAL_DECODED_BYTES

    # the caps above are derived from the per-mode byte table, so equal caps do not by themselves
    # prove the two copies agree: adding a mode to one table only (say `"RGBX": 4`) leaves the max
    # -- and therefore every cap -- unchanged while the per-image decoded peaks silently diverge.
    # pin the inputs, not just the outputs.
    assert _image_descriptors._MODE_BYTES_PER_PIXEL == child_glue._MODE_BYTES_PER_PIXEL
    assert _image_descriptors.RGB_BYTES_PER_PIXEL == child_glue._RGB_BYTES_PER_PIXEL
    assert _image_descriptors.WORST_BYTES_PER_PIXEL == child_glue._WORST_BYTES_PER_PIXEL
