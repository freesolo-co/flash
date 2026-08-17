from __future__ import annotations

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
