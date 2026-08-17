from __future__ import annotations

import asyncio
import copy
import importlib
import io
import os
from collections.abc import Mapping

import pytest

from flash.content import multimodal as mm

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
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


def test_processor_digests_match_raw_and_qwen_sized_images(real_processor):
    from flash.engine.worker.train.core.child.glue import (
        parent_image_digests,
        processor_image_digests,
    )

    digests = []
    for size, color in (((32, 32), "red"), ((300, 200), "green")):
        descriptor = mm.normalize_image_source(_image_bytes(size, color), None)
        raw = mm.decode_image_descriptors([descriptor], None)
        try:
            resized = [_qwen_sized_image(real_processor, raw[0])]
            try:
                parent = parent_image_digests(real_processor, [descriptor], None)
                child = processor_image_digests(real_processor, resized)
            finally:
                resized[0].close()
        finally:
            raw[0].close()
        assert parent == child
        digests.extend(parent)

    other = mm.normalize_image_source(_image_bytes((32, 32), "blue"), None)
    assert parent_image_digests(real_processor, [other], None)[0] != digests[0]

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


def test_real_processor_drives_the_shared_dynamic_child_glue(real_processor):
    from flash.engine.worker.train.core.child.glue import (
        EnvGlueProcessor,
        parent_image_digests,
    )

    class _Loop:
        def __init__(self, processor):
            self.processor = processor
            self.tokenizer = processor.tokenizer
            self.apply_chat_template_kwargs = {}

        def _get_mm_processor_kwargs(self, _audio_data=None):
            return {}

        async def process_multi_modal_info(self, messages):
            uris = [
                block["image"]
                for message in messages
                for block in message.get("content", [])
                if isinstance(block, dict) and block.get("type") == "image"
            ]
            descriptors = [mm.normalize_image_source(uri, None) for uri in uris]
            decoded = mm.decode_image_descriptors(descriptors, None)
            try:
                return {"images": [_qwen_sized_image(self.processor, image) for image in decoded]}
            finally:
                for image in decoded:
                    image.close()

    loop = _Loop(real_processor)
    glue = EnvGlueProcessor(loop, thinking=False)
    cumulative_images = []
    cumulative_digests = []
    snapshots = []

    for index, (size, color, _expected_run) in enumerate(_IMAGE_CASES):
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
        result = asyncio.run(
            glue(
                observation.messages,
                mm.image_descriptors_to_data_uris(observation.descriptors, None),
            )
        )
        expected = _parent_glue_ids(
            real_processor,
            observation.messages,
            observation.descriptors,
        )
        assert result.token_ids == expected
        assert len(result.images) == 1
        assert result.image_digests == parent_image_digests(
            real_processor, observation.descriptors, None
        )
        cumulative_images.extend(result.images)
        cumulative_digests.extend(result.image_digests)
        snapshots.append((list(cumulative_images), list(cumulative_digests)))
        assert len(cumulative_images) == index + 1
        assert len(set(cumulative_digests)) == index + 1

    for images, digests in snapshots:
        assert images == cumulative_images[: len(images)]
        assert digests == cumulative_digests[: len(digests)]
    for image in cumulative_images:
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
