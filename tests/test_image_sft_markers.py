from __future__ import annotations

import json
from typing import ClassVar

import pytest

from flash.content import multimodal as mm
from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.sft_workload import prepare_sft_workload
from flash.engine.worker.entry.sft import _reject_image_completion

_VALID_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.parametrize(
    ("record_images", "content"),
    [
        pytest.param(
            False,
            [{"type": "image"}, {"type": "text", "text": f"m {mm.IMAGE_PAD_TOKEN}"}],
            id="blocks",
        ),
        pytest.param(True, f"m {mm.IMAGE_PAD_TOKEN}", id="string-content-top-level-image"),
        pytest.param(
            False,
            [
                {"type": "image"},
                {"type": "text", "text": "<|image_"},
                {"type": "text", "text": "pad|>"},
            ],
            id="marker-split-across-adjacent-text-blocks",
        ),
        pytest.param(
            False,
            [
                {"type": "image"},
                {"type": "text", "text": "<|ima"},
                {"type": "text", "text": "ge_p"},
                {"type": "text", "text": "ad|>"},
            ],
            id="marker-split-three-ways",
        ),
    ],
)
def test_normalization_rejects_the_pad_token_in_image_prompt_text(record_images, content):
    record = {"image": _VALID_PNG_DATA_URI} if record_images else {}
    messages = [{"role": "user", "content": content}]
    if not record_images:
        messages[0]["content"] = [
            {
                "type": "image_url",
                "image_url": {"url": _VALID_PNG_DATA_URI},
            }
            if block.get("type") == "image"
            else block
            for block in content
        ]
    with pytest.raises(ValueError, match="reserved image marker"):
        mm.normalize_prompt_images(record, messages, None)


def test_normalization_allows_marker_fragments_separated_by_an_image_block():
    normalized = mm.normalize_prompt_images(
        {},
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|image_"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _VALID_PNG_DATA_URI},
                    },
                    {"type": "text", "text": "pad|>"},
                ],
            }
        ],
        None,
    )
    assert len(normalized.descriptors) == 1
    assert [block["type"] for block in normalized.messages[0]["content"]] == [
        "text",
        "image",
        "text",
    ]


def test_normalization_keeps_the_teacher_placeholder_out_of_the_local_sft_check():
    normalized = mm.normalize_prompt_images(
        {"image": _VALID_PNG_DATA_URI},
        [{"role": "user", "content": f"literally {mm.IMAGE_TEACHER_PLACEHOLDER} here"}],
        None,
    )
    assert len(normalized.descriptors) == 1
    assert normalized.messages[0]["content"][0]["text"].endswith("here")


class _ToolCallChatTemplate:
    chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}{{ message['tool_calls'][0]['function']['name'] }}"
        "{{ message['tool_calls'][0]['function']['arguments'] }}<|im_end|>\n{% endfor %}"
    )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_kwargs):
        assert not tokenize
        rendered = []
        for message in messages:
            body = str(message.get("content") or "")
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                body += str(function.get("name", ""))
                body += str(function.get("arguments", ""))
            rendered.append(f"<|im_start|>{message.get('role')}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        pytest.param("prompt", "name", mm.IMAGE_PAD_TOKEN, id="prompt-function-name"),
        pytest.param(
            "prompt",
            "arguments",
            json.dumps({mm.IMAGE_PAD_TOKEN: "value"}),
            id="prompt-argument-name",
        ),
        pytest.param("completion", "name", mm.IMAGE_PAD_TOKEN, id="completion-function-name"),
        pytest.param(
            "completion",
            "arguments",
            json.dumps({"outer": {"inner": mm.IMAGE_PAD_TOKEN}}),
            id="completion-nested-argument-value",
        ),
    ],
)
def test_image_sft_rejects_image_pad_from_rendered_tool_call_fields(location, field, value):
    function = {"name": "lookup", "arguments": json.dumps({"city": "london"})}
    function[field] = value
    tool_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": function}],
    }
    prompt = [tool_message] if location == "prompt" else [{"role": "user", "content": "go"}]
    completion = (
        [tool_message] if location == "completion" else [{"role": "assistant", "content": "ok"}]
    )

    with pytest.raises(ValueError, match="reserved image marker"):
        _reject_image_completion(
            completion,
            image_bearing=True,
            source_messages=[*prompt, *completion],
            template_source=_ToolCallChatTemplate(),
        )


def test_image_sft_rejects_image_pad_split_across_adjacent_rendered_tool_fields():
    completion = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "<|image_", "arguments": "pad|>"},
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="reserved image marker"):
        _reject_image_completion(
            completion,
            image_bearing=True,
            source_messages=completion,
            template_source=_ToolCallChatTemplate(),
        )


def test_image_sft_rendered_tool_call_guard_allows_ordinary_fields_and_media_pad():
    completion = [
        {
            "role": "assistant",
            "content": f"literal {mm.IMAGE_TEACHER_PLACEHOLDER}",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps(
                            {"marker": mm.IMAGE_TEACHER_PLACEHOLDER, "city": "london"}
                        ),
                    },
                }
            ],
        }
    ]
    _reject_image_completion(
        completion,
        image_bearing=True,
        source_messages=[{"role": "user", "content": "go"}, *completion],
        template_source=_ToolCallChatTemplate(),
    )


def test_image_sft_completion_text_carrying_the_pad_token_is_rejected():
    with pytest.raises(ValueError, match="reserved image marker"):
        _reject_image_completion(
            [{"role": "assistant", "content": f"it shows {mm.IMAGE_PAD_TOKEN} a cat"}],
            image_bearing=True,
        )
    _reject_image_completion(
        [{"role": "assistant", "content": "it shows a cat"}],
        image_bearing=True,
    )


class _RegisteredImagePadTokenizer:
    eos_token = "|"
    eos_token_id = 2
    pad_token = None
    pad_token_id = 0
    all_special_ids: ClassVar[list[int]] = [0, 1, 2, 248056]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        **_kwargs,
    ):
        assert not tokenize
        assert enable_thinking is False
        rendered = "".join(
            f"<{message['role']}>{message.get('content') or ''}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def convert_tokens_to_ids(self, token):
        return 248056 if token == mm.IMAGE_PAD_TOKEN else 1

    def convert_ids_to_tokens(self, token_id):
        return mm.IMAGE_PAD_TOKEN if token_id == 248056 else "[UNK]"

    def get_vocab(self):
        return {"[UNK]": 1, "|": 2, mm.IMAGE_PAD_TOKEN: 248056}

    def __call__(self, texts, *, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._encode(text) for text in texts]
        if truncation:
            assert max_length is not None
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}

    @staticmethod
    def _encode(text: str) -> list[int]:
        ids = []
        remaining = text
        while remaining:
            marker = remaining.find(mm.IMAGE_PAD_TOKEN)
            if marker < 0:
                ids.extend(3 + ord(char) for char in remaining)
                break
            ids.extend(3 + ord(char) for char in remaining[:marker])
            ids.append(248056)
            remaining = remaining[marker + len(mm.IMAGE_PAD_TOKEN) :]
        return ids


class _TextOnlyEnvironment:
    package_root = None
    multi_turn = False

    def __init__(self, prompt: str, completion: str):
        self._row = {"prompt": prompt, "completion": completion}

    def dataset(self):
        return [self._row]

    def prompt_messages(self, row):
        return [{"role": "user", "content": row["prompt"]}]

    def sft_completion(self, row):
        return [{"role": "assistant", "content": row["completion"]}]


def _prepare_text_only_row(prompt: str, completion: str):
    tokenizer = _RegisteredImagePadTokenizer()
    spec = JobSpec(
        model="test/model",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(id="local", resolved_sha="b" * 40),
        train=TrainSpec(epochs=1, batch_size=1, max_context_tokens=512, max_examples=1),
        workload_profile_input_digest="0" * 64,
        workload_profile_producer_version="test",
    )
    prepared = prepare_sft_workload(
        spec,
        _TextOnlyEnvironment(prompt, completion),
        tokenizer_loader=lambda _model, _revision: tokenizer,
        producer_version="test",
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )
    assert prepared.multimodal is False
    assert tokenizer.convert_tokens_to_ids(mm.IMAGE_PAD_TOKEN) == 248056
    return prepared.rows[0]


def test_text_only_prompt_allows_registered_image_pad_with_normal_masking():
    row = _prepare_text_only_row(f"literal {mm.IMAGE_PAD_TOKEN} prompt", "answer")
    positions = [index for index, token_id in enumerate(row["input_ids"]) if token_id == 248056]

    assert len(positions) == 1
    assert row["loss_mask"][positions[0]] == 0
    assert sum(row["loss_mask"]) > 0


def test_text_only_completion_allows_registered_image_pad_with_normal_supervision():
    row = _prepare_text_only_row("prompt", f"before {mm.IMAGE_PAD_TOKEN} after")
    positions = [index for index, token_id in enumerate(row["input_ids"]) if token_id == 248056]

    assert len(positions) == 1
    assert row["loss_mask"][positions[0]] == 1
    assert sum(row["loss_mask"]) > 1
