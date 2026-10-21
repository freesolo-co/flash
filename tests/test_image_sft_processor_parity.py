"""the torch-free image estimate must match a pinned real-processor oracle."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from flash.content.multimodal import IMAGE_PAD_TOKEN, normalize_image_source
from flash.engine.profiling import sft_image_rows
from flash.engine.profiling.image_tokens import (
    ImageProfileValidationState,
    geometry_from_preprocessor_config,
    image_pad_tokens,
)
from flash.engine.profiling.sft_image_rows import (
    _registered_token_id,
    estimate_sft_image_row,
    process_sft_image_row,
)

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ORACLE_PATH = Path(__file__).parent / "fixtures" / "image_sft_processor_oracle.json"
BASE_GEOMETRY = {
    "patch_size": 16,
    "merge_size": 2,
    "size": {"shortest_edge": 65536, "longest_edge": 16777216},
}
CASE_SOURCES = {
    "truncated-role-fallback": {
        "sizes": [[56, 56]],
        "question": "tiny",
        "completion": [{"role": "assistant", "content": "the answer"}],
        "max_length": 81,
        "geometry": BASE_GEOMETRY,
    },
    "ordered-multi-image": {
        "sizes": [[300, 200], [56, 56]],
        "question": "compare these",
        "completion": [{"role": "assistant", "content": "the answer"}],
        "max_length": 8192,
        "geometry": BASE_GEOMETRY,
    },
    "max-pixel-resize": {
        "sizes": [[640, 480]],
        "question": "large",
        "completion": [{"role": "assistant", "content": "the answer"}],
        "max_length": 8192,
        "geometry": {
            "patch_size": 16,
            "merge_size": 2,
            "size": {"shortest_edge": 65536, "longest_edge": 65536},
        },
    },
    "multi-turn-assistant-mask": {
        "sizes": [[56, 56]],
        "question": "look",
        "completion": [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "environment reply"},
            {"role": "assistant", "content": "second"},
        ],
        "max_length": 8192,
        "geometry": BASE_GEOMETRY,
    },
}


def _canonical(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _descriptor(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (17, 99, 200)).save(buffer, format="PNG")
    return normalize_image_source(buffer.getvalue(), None)


def _prompt(source: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in source["sizes"]),
                {"type": "text", "text": source["question"]},
            ],
        }
    ]


def _render_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        f"<|vision_start|>{IMAGE_PAD_TOKEN}<|vision_end|>"
        if block.get("type") == "image"
        else str(block.get("text", ""))
        for block in content
    )


def _render_messages(messages: list[dict], *, add_generation_prompt: bool) -> str:
    rendered = []
    for index, message in enumerate(messages):
        body = _render_content(message.get("content"))
        if message.get("role") == "assistant" and index == len(messages) - 1:
            body = f"<think>\n\n</think>\n\n{body}"
        rendered.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
    if add_generation_prompt:
        rendered.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(rendered)


def _expand_runs(values: list) -> list[int]:
    expanded = []
    for value in values:
        if isinstance(value, list):
            item, count = value
            expanded.extend([item] * count)
        else:
            expanded.append(value)
    return expanded


def _compress_runs(values: list[int]) -> list:
    compressed = []
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end] == values[index]:
            end += 1
        count = end - index
        compressed.append(values[index] if count == 1 else [values[index], count])
        index = end
    return compressed


class _OracleTokenizer:
    """replay captured ids only for the exact prompt/full render each call requires."""

    chat_template = "<|im_start|><|im_end|>"
    unk_token_id = -1

    def __init__(self, oracle: dict, case: dict):
        self._case = case
        self._tokens = oracle["tokens"]
        self._pieces = {int(key): value for key, value in oracle["pieces"].items()}

    def get_vocab(self):
        return dict(self._tokens)

    def get_chat_template(self):
        return self.chat_template

    def convert_tokens_to_ids(self, token):
        return self._tokens.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self._pieces.get(token_id)

    def decode(self, token_ids):
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_kwargs):
        rendered = _render_messages(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return rendered
        boundary = "prompt" if add_generation_prompt else "full"
        if rendered != self._case[f"rendered_{boundary}"]:
            raise AssertionError(f"oracle received the wrong messages for the {boundary} boundary")
        return {"input_ids": list(self._case[f"base_{boundary}"])}


@pytest.fixture(scope="module")
def processor_oracle():
    return json.loads(ORACLE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_name", CASE_SOURCES)
def test_estimator_matches_the_pinned_processor_oracle(processor_oracle, case_name):
    source = CASE_SOURCES[case_name]
    case = processor_oracle["cases"][case_name]
    assert processor_oracle["provenance"]["case_source_sha256"] == _digest(CASE_SOURCES)
    assert processor_oracle["provenance"]["preprocessor_geometry_sha256"] == _digest(BASE_GEOMETRY)
    tokenizer = _OracleTokenizer(processor_oracle, case)
    descriptors = [_descriptor(width, height) for width, height in source["sizes"]]

    actual = estimate_sft_image_row(
        tokenizer,
        _prompt(source),
        source["completion"],
        descriptors,
        package_root=None,
        geometry=geometry_from_preprocessor_config(source["geometry"]),
        validation_state=ImageProfileValidationState(),
        max_length=source["max_length"],
        thinking=False,
    )

    assert actual == (
        _expand_runs(case["input_ids"]),
        _expand_runs(case["loss_mask"]),
        b"",
        case["untruncated_length"],
        case["role_aware"],
    )


def test_oracle_pins_model_revision_and_image_pad_registration(processor_oracle):
    provenance = processor_oracle["provenance"]
    assert {
        "model": provenance["model"],
        "revision": provenance["revision"],
        "processor_revision": provenance["processor_revision"],
        "tokenizer_revision": provenance["tokenizer_revision"],
    } == {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "processor_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    }
    assert provenance["image_pad_registration"] == {
        "token": IMAGE_PAD_TOKEN,
        "id": 248056,
        "convert_ids_to_tokens": IMAGE_PAD_TOKEN,
        "vocab_aliases": [IMAGE_PAD_TOKEN],
    }
    assert processor_oracle["tokens"][IMAGE_PAD_TOKEN] == 248056


def test_oracle_cases_exercise_each_claimed_parity_boundary(processor_oracle):
    cases = processor_oracle["cases"]
    truncated = cases["truncated-role-fallback"]
    assert truncated["untruncated_length"] > len(_expand_runs(truncated["input_ids"]))
    assert truncated["role_aware"] is False

    pad_id = processor_oracle["tokens"][IMAGE_PAD_TOKEN]
    ordered_runs = [
        value[1]
        for value in cases["ordered-multi-image"]["input_ids"]
        if isinstance(value, list) and value[0] == pad_id
    ]
    assert ordered_runs == [70, 64]

    source = CASE_SOURCES["max-pixel-resize"]
    resized = geometry_from_preprocessor_config(source["geometry"])
    published = geometry_from_preprocessor_config(BASE_GEOMETRY)
    resized_count = sum(
        value[1]
        for value in cases["max-pixel-resize"]["input_ids"]
        if isinstance(value, list) and value[0] == pad_id
    )
    assert resized_count == image_pad_tokens(640, 480, resized)
    assert resized_count != image_pad_tokens(640, 480, published)


def test_oracle_rejects_prompt_and_full_message_swaps(processor_oracle):
    source = CASE_SOURCES["multi-turn-assistant-mask"]
    case = processor_oracle["cases"]["multi-turn-assistant-mask"]
    tokenizer = _OracleTokenizer(processor_oracle, case)
    prompt = _prompt(source)
    full = [*prompt, *source["completion"]]

    with pytest.raises(AssertionError, match="wrong messages for the full boundary"):
        tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=False)
    with pytest.raises(AssertionError, match="wrong messages for the prompt boundary"):
        tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=True)


class _CanonicalMismatchTokenizer(PreTrainedTokenizerFast):
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if ids == 1:
            return "<alias>"
        return super().convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)


class _DuplicateAliasTokenizer(PreTrainedTokenizerFast):
    def get_vocab(self):
        return {**super().get_vocab(), "<alias>": 1}


class _InvalidIdTokenizer(PreTrainedTokenizerFast):
    invalid_id = None

    def convert_tokens_to_ids(self, tokens):
        if tokens == IMAGE_PAD_TOKEN:
            return self.invalid_id
        return super().convert_tokens_to_ids(tokens)


def _minimal_hf_tokenizer(
    *, with_image_token: bool, tokenizer_type=PreTrainedTokenizerFast
) -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0}
    if with_image_token:
        vocab[IMAGE_PAD_TOKEN] = 1
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    kwargs = {"additional_special_tokens": [IMAGE_PAD_TOKEN]} if with_image_token else {}
    return tokenizer_type(tokenizer_object=backend, unk_token="[UNK]", **kwargs)


def test_registered_token_id_accepts_one_exact_canonical_registration():
    tokenizer = _minimal_hf_tokenizer(with_image_token=True)
    assert _registered_token_id(tokenizer, IMAGE_PAD_TOKEN) == 1


@pytest.mark.parametrize(
    "tokenizer",
    [
        pytest.param(_minimal_hf_tokenizer(with_image_token=False), id="unknown-token-collision"),
        pytest.param(
            _minimal_hf_tokenizer(
                with_image_token=True,
                tokenizer_type=_CanonicalMismatchTokenizer,
            ),
            id="canonical-token-mismatch",
        ),
        pytest.param(
            _minimal_hf_tokenizer(
                with_image_token=True,
                tokenizer_type=_DuplicateAliasTokenizer,
            ),
            id="duplicate-id-alias",
        ),
    ],
)
def test_registered_token_id_rejects_ambiguous_hf_tokenizers(tokenizer):
    with pytest.raises(ValueError, match=r"exact token|resolves to|shared by"):
        _registered_token_id(tokenizer, IMAGE_PAD_TOKEN)


@pytest.mark.parametrize("invalid_id", [True, "1", -1])
def test_registered_token_id_rejects_invalid_id_shapes(invalid_id):
    tokenizer = _minimal_hf_tokenizer(
        with_image_token=True,
        tokenizer_type=_InvalidIdTokenizer,
    )
    tokenizer.invalid_id = invalid_id
    with pytest.raises(ValueError, match=r"exact token|resolves to|shared by"):
        _registered_token_id(tokenizer, IMAGE_PAD_TOKEN)


def test_estimator_rejects_unknown_fallback_before_decoding_descriptors(monkeypatch):
    missing = _minimal_hf_tokenizer(with_image_token=False)
    present = _minimal_hf_tokenizer(with_image_token=True)
    decoded = []

    def descriptor_tokens(*_args, **_kwargs):
        decoded.append(True)
        raise AssertionError("descriptor decoded")

    monkeypatch.setattr(sft_image_rows, "descriptor_pad_tokens", descriptor_tokens)
    common = {
        "prompt_messages": [],
        "completion_messages": [],
        "descriptors": ["not-a-descriptor"],
        "package_root": None,
        "geometry": geometry_from_preprocessor_config(BASE_GEOMETRY),
        "validation_state": ImageProfileValidationState(),
        "max_length": 16,
        "thinking": False,
    }

    with pytest.raises(ValueError, match="does not define the image placeholder"):
        estimate_sft_image_row(missing, **common)
    assert decoded == []
    with pytest.raises(AssertionError, match="descriptor decoded"):
        estimate_sft_image_row(present, **common)
    assert decoded == [True]


def _ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _published_geometry(config: dict) -> dict:
    return {
        "patch_size": config["patch_size"],
        "merge_size": config["merge_size"],
        "size": {
            "shortest_edge": config["size"]["shortest_edge"],
            "longest_edge": config["size"]["longest_edge"],
        },
    }


def _load_live_image_stack():
    import transformers
    from huggingface_hub import hf_hub_download

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    path = hf_hub_download(MODEL_ID, "preprocessor_config.json", revision=MODEL_REVISION)
    return processor, tokenizer, json.loads(Path(path).read_text(encoding="utf-8"))


def capture_processor_oracle(processor, tokenizer, preprocessor_config: dict) -> dict:
    import transformers

    cases = {}
    unique_ids = set()
    original_size = (
        processor.image_processor.size.shortest_edge,
        processor.image_processor.size.longest_edge,
    )
    try:
        for case_name, source in CASE_SOURCES.items():
            geometry = source["geometry"]["size"]
            processor.image_processor.size.shortest_edge = geometry["shortest_edge"]
            processor.image_processor.size.longest_edge = geometry["longest_edge"]
            descriptors = [_descriptor(width, height) for width, height in source["sizes"]]
            prompt = _prompt(source)
            full = [*prompt, *source["completion"]]
            common = {"tokenize": True, "return_dict": True, "enable_thinking": False}
            base_full = _ids(
                dict(
                    tokenizer.apply_chat_template(
                        full,
                        add_generation_prompt=False,
                        **common,
                    )
                )["input_ids"]
            )
            base_prompt = _ids(
                dict(
                    tokenizer.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        **common,
                    )
                )["input_ids"]
            )
            input_ids, mask, _blob, length, role_aware = process_sft_image_row(
                processor,
                prompt,
                source["completion"],
                descriptors,
                package_root=None,
                max_length=source["max_length"],
                thinking=False,
            )
            unique_ids.update(input_ids)
            cases[case_name] = {
                "base_full": base_full,
                "base_prompt": base_prompt,
                "rendered_full": tokenizer.apply_chat_template(
                    full,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                ),
                "rendered_prompt": tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "input_ids": _compress_runs(input_ids),
                "loss_mask": _compress_runs(mask),
                "untruncated_length": length,
                "role_aware": role_aware,
            }
    finally:
        (
            processor.image_processor.size.shortest_edge,
            processor.image_processor.size.longest_edge,
        ) = original_size
    geometry = _published_geometry(preprocessor_config)
    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    return {
        "provenance": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "processor_class": type(processor).__name__,
            "processor_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "preprocessor_geometry_sha256": _digest(geometry),
            "case_source_sha256": _digest(CASE_SOURCES),
            "image_pad_registration": {
                "token": IMAGE_PAD_TOKEN,
                "id": image_pad_id,
                "convert_ids_to_tokens": tokenizer.convert_ids_to_tokens(image_pad_id),
                "vocab_aliases": sorted(
                    token
                    for token, token_id in tokenizer.get_vocab().items()
                    if token_id == image_pad_id
                ),
            },
            "capture_command": (
                "FLASH_IMAGE_PROCESSOR_LIVE=1 uv run --extra gpu python "
                "tests/capture_image_sft_processor_oracle.py"
            ),
        },
        "tokens": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in (IMAGE_PAD_TOKEN, "<|im_start|>", "<|im_end|>")
        },
        "pieces": {str(token_id): tokenizer.decode([token_id]) for token_id in sorted(unique_ids)},
        "cases": cases,
    }


@pytest.fixture(scope="module")
def optional_image_stack():
    if os.environ.get("FLASH_IMAGE_PROCESSOR_LIVE") != "1":
        pytest.skip("live image processor parity disabled; set FLASH_IMAGE_PROCESSOR_LIVE=1")
    pytest.importorskip("torch", reason="the vl processor cannot load without torch")
    pytest.importorskip("torchvision", reason="the vl image processor requires torchvision")
    try:
        return _load_live_image_stack()
    except Exception as exc:  # pragma: no cover - optional network integration
        pytest.skip(f"could not load {MODEL_ID} image stack: {exc}")


def test_optional_live_processor_verifies_the_checked_oracle(
    optional_image_stack, processor_oracle
):
    assert capture_processor_oracle(*optional_image_stack) == processor_oracle
