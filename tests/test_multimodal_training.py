from __future__ import annotations

import base64
import inspect
from typing import ClassVar

import pytest

from flash.catalog import MODELS, supports_multimodal
from flash.engine.worker import grpo, rl, sft
from flash.engine.worker.multimodal import (
    _load_image,
    _read_limited_response,
    has_image_input,
    image_input_count,
    message_text,
    model_supports_images,
    multimodal_grpo_prompt_row,
    multimodal_sft_row,
    multimodal_token_estimate,
)
from flash.envs.adapter import FreesoloEnvironment, load_freesolo_environment

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


def test_freesolo_canonical_record_preserves_image_fields():
    record = FreesoloEnvironment._canonical_record(
        {
            "id": "one",
            "input": "What color?",
            "output": "red",
            "image": _RED_DOT,
            "images": [_RED_DOT],
            "metadata": {"split": "train"},
            "custom": {"kept": True},
        }
    )
    assert record["image"] == _RED_DOT
    assert record["images"] == [_RED_DOT]
    assert record["custom"] == {"kept": True}


def test_local_image_loading_is_confined_to_base_dirs(tmp_path):
    pytest.importorskip("PIL")
    allowed_dir = tmp_path / "images"
    allowed_dir.mkdir()
    allowed = allowed_dir / "red.png"
    outside = tmp_path / "secret.png"
    payload = base64.b64decode(_RED_DOT.split(",", 1)[1])
    allowed.write_bytes(payload)
    outside.write_bytes(payload)

    image = _load_image("red.png", base_dirs=(allowed_dir,))
    assert image.size == (1, 1)
    with pytest.raises(FileNotFoundError, match="outside FLASH_IMAGE_BASE_DIR"):
        _load_image("../secret.png", base_dirs=(allowed_dir,))
    with pytest.raises(FileNotFoundError, match="outside FLASH_IMAGE_BASE_DIR"):
        _load_image(str(outside), base_dirs=(allowed_dir,))


def test_remote_image_reader_enforces_byte_cap(monkeypatch):
    class Response:
        headers: ClassVar[dict[str, str]] = {"Content-Length": "4"}

        def read(self, _n):
            return b""

    monkeypatch.setenv("FLASH_IMAGE_MAX_BYTES", "3")
    with pytest.raises(ValueError, match="FLASH_IMAGE_MAX_BYTES=3"):
        _read_limited_response(Response())


def test_multimodal_token_estimate_reserves_image_tokens(monkeypatch):
    class Tokenizer:
        def __call__(self, _text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [1, 2, 3]}

    monkeypatch.setenv("FLASH_IMAGE_TOKEN_RESERVE", "7")
    assert multimodal_token_estimate("hello", Tokenizer(), image_count=2) == 17
    assert image_input_count(_prompt(), _example()) == 1


def test_every_catalog_multimodal_model_accepts_image_rows():
    image_models = [model_id for model_id, info in MODELS.items() if supports_multimodal(info)]
    assert image_models
    for model_id in image_models:
        assert model_supports_images(model_id) is True
        sft_row = multimodal_sft_row(_prompt(), _completion(), _example())
        grpo_row = multimodal_grpo_prompt_row(_prompt(), _example())
        assert sft_row["prompt"][0]["content"][0]["type"] == "image"
        assert grpo_row["prompt"][0]["content"][0]["type"] == "image"


def test_mixed_multimodal_sft_rows_are_arrow_compatible():
    from datasets import Dataset

    text_row = multimodal_sft_row(
        [{"role": "user", "content": "Text-only question?"}],
        _completion(),
        {"input": "Text-only question?", "output": "red"},
    )
    image_row = multimodal_sft_row(_prompt(), _completion(), _example())

    ds = Dataset.from_list([text_row, image_row])
    assert ds.column_names == ["prompt", "completion", "images"]
    assert ds[0]["images"] == []
    assert ds[0]["prompt"][0]["content"] == [{"type": "text", "text": "Text-only question?"}]
    assert ds[1]["images"][0].size == (1, 1)


def test_mixed_multimodal_grpo_rows_are_arrow_compatible():
    from datasets import Dataset

    prompts = [
        multimodal_grpo_prompt_row(
            [{"role": "user", "content": "Text-only question?"}],
            {"input": "Text-only question?", "output": "red"},
        ),
        multimodal_grpo_prompt_row(_prompt(), _example()),
    ]
    rows, examples = grpo.build_grpo_prompt_dataset(prompts)

    ds = Dataset.from_list(rows)
    assert ds.column_names == ["prompt", "example_idx", "images"]
    assert ds[0]["images"] == []
    assert ds[1]["images"][0].size == (1, 1)
    assert examples[1]["image"] == _RED_DOT


def test_sft_multimodal_path_is_wired_to_trl_vlm_support():
    src = inspect.getsource(sft.run_sft)
    assert "AutoProcessor.from_pretrained" in src
    assert "multimodal_sft_row(" in src
    assert "processing_class=processor or tok" in src
    assert "make_lora(model_id, multimodal=_multimodal)" in src
    assert "token packing disabled" in src
    assert "dropped_empty_targets" in src
    assert "multimodal_token_estimate" in src
    assert "base_dirs=image_base_dirs" in src


def test_grpo_multimodal_path_is_wired_to_trl_vlm_support():
    src = inspect.getsource(rl.run_rl)
    assert "AutoProcessor.from_pretrained" in src
    assert "multimodal_grpo_prompt_row(" in src
    assert "processing_class=processor or tok" in src
    assert "_init_adapter_model(model_id, multimodal=_multimodal)" in src
    assert "if not _multimodal:\n            patch_vllm_language_model_only(model_id)" in src
    assert "multimodal_token_estimate" in src
    assert "not conversational and not _multimodal" in src
    assert "multimodal GRPO currently supports single-turn image prompts only" in src
    assert "base_dirs=image_base_dirs" in src


def test_freesolo_env_threads_image_base_dirs():
    src = inspect.getsource(load_freesolo_environment)
    assert "image_base_dirs = [base_dir]" in src
    assert "image_base_dirs=tuple" in src


def test_grpo_prompt_dataset_preserves_image_columns_without_rich_examples():
    rows, examples = grpo.build_grpo_prompt_dataset(
        [{"prompt": "p", "image": "img", "example": {"metadata": {"mixed": object()}}}]
    )
    assert rows == [{"prompt": "p", "example_idx": 0, "image": "img"}]
    assert examples[0]["metadata"]["mixed"] is not None
