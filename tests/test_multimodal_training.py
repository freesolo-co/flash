from __future__ import annotations

import inspect

from flash.catalog import MODELS, supports_multimodal
from flash.engine.worker import grpo, rl, sft
from flash.engine.worker.multimodal import (
    has_image_input,
    message_text,
    model_supports_images,
    multimodal_grpo_prompt_row,
    multimodal_sft_row,
)

_RED_DOT = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/lR50uwAAAABJRU5ErkJggg=="
)


def _prompt():
    return [{"role": "user", "content": "What color is the square?"}]


def _completion():
    return [{"role": "assistant", "content": "red"}]


def _example():
    return {"input": "What color is the square?", "image": _RED_DOT, "output": "red"}


def test_catalog_labels_every_multimodal_model():
    expected = {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    actual = {model_id for model_id, info in MODELS.items() if supports_multimodal(info)}
    assert actual == expected
    assert supports_multimodal("openbmb/MiniCPM5-1B") is False

    for model_id, info in MODELS.items():
        row = info.to_dict()
        assert row["modalities"] == info.modalities
        assert row["multimodal"] is (model_id in expected)


def test_top_level_image_injects_chat_placeholder_and_loads_image():
    row = multimodal_sft_row(_prompt(), _completion(), _example())
    assert row["prompt"][0]["content"][0] == {"type": "image"}
    assert row["prompt"][0]["content"][1] == {
        "type": "text",
        "text": "What color is the square?",
    }
    assert row["images"][0].size == (1, 1)
    assert message_text(row["prompt"]) == "<image>\nWhat color is the square?"


def test_explicit_image_block_uses_top_level_image_data():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Name the color."}],
        }
    ]
    row = multimodal_grpo_prompt_row(messages, _example())
    assert row["prompt"] == messages
    assert row["images"][0].size == (1, 1)
    assert has_image_input(messages, {}) is True


def test_every_catalog_multimodal_model_accepts_image_rows():
    image_models = [model_id for model_id, info in MODELS.items() if supports_multimodal(info)]
    assert image_models
    for model_id in image_models:
        assert model_supports_images(model_id) is True
        sft_row = multimodal_sft_row(_prompt(), _completion(), _example())
        grpo_row = multimodal_grpo_prompt_row(_prompt(), _example())
        assert sft_row["prompt"][0]["content"][0]["type"] == "image"
        assert grpo_row["prompt"][0]["content"][0]["type"] == "image"


def test_sft_multimodal_path_is_wired_to_trl_vlm_support():
    src = inspect.getsource(sft.run_sft)
    assert "AutoProcessor.from_pretrained" in src
    assert "multimodal_sft_row(" in src
    assert "processing_class=processor or tok" in src
    assert "make_lora(model_id, multimodal=_multimodal)" in src
    assert "token packing disabled" in src


def test_grpo_multimodal_path_is_wired_to_trl_vlm_support():
    src = inspect.getsource(rl.run_rl)
    assert "AutoProcessor.from_pretrained" in src
    assert "multimodal_grpo_prompt_row(" in src
    assert "processing_class=processor or tok" in src
    assert "_init_adapter_model(model_id, multimodal=_multimodal)" in src
    assert "if not _multimodal:\n            patch_vllm_language_model_only(model_id)" in src


def test_grpo_prompt_dataset_preserves_image_columns_without_rich_examples():
    rows, examples = grpo.build_grpo_prompt_dataset(
        [{"prompt": "p", "image": "img", "example": {"metadata": {"mixed": object()}}}]
    )
    assert rows == [{"prompt": "p", "example_idx": 0, "image": "img"}]
    assert examples[0]["metadata"]["mixed"] is not None
