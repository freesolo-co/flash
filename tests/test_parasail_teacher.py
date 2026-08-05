from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flash.engine.recipe import TEACHER_MODELS, resolve_teacher
from flash.engine.worker.teacher import TeacherClient, TeacherError, TeacherScore
from flash.engine.worker.teacher_encoding import (
    EncodedTeacherToken,
    _byte_piece_offsets,
    _exact_source_spans,
)


class _FixedTokenizer:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def encode(self, _text):
        return list(self.tokens)


def _tokens():
    return [
        EncodedTeacherToken(10, 0, 1),
        EncodedTeacherToken(20, 1, 5),
        EncodedTeacherToken(30, 5, 6),
    ]


def _usage():
    return {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _token_keyed_response():
    return {
        "choices": [
            {
                "prompt_token_ids": [10, 20, 30],
                "token_ids": [40],
                "prompt_logprobs": [
                    None,
                    {"20": {"logprob": -0.5, "decoded_token": "ignored"}},
                    {"30": {"logprob": -0.2}},
                ],
            }
        ],
        "usage": _usage(),
    }


def _positional_response():
    return {
        "choices": [
            {
                "prompt_token_ids": [10, 20, 30],
                "token_ids": [40],
                "logprobs": {"token_logprobs": [None, -0.5, -0.2, -0.1]},
            }
        ],
        "usage": _usage(),
    }


def _client(response, capture=None):
    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-glm-52",
        tokenizer=_FixedTokenizer(_tokens()),
    )

    def post(path, body):
        if capture is not None:
            capture.update({"path": path, "body": body})
        return response

    client._post = post
    return client


def test_training_guide_lists_only_the_managed_teacher_aliases():
    guide = (Path(__file__).parents[1] / "flash" / "cli" / "TRAINING.md").read_text()

    assert "glm-5.2 (default) | kimi-k3 |" in guide
    assert "qwen3.5-397b-a17b (key stays managed)" in guide
    assert "kimi-k2.6" not in guide
    assert "The control-plane broker owns `PARASAIL_API_KEY`" in guide
    assert "`FLASH_CONTROL_PANEL_URL` and an attempt-scoped `FLASH_TEACHER_CAPABILITY`" in guide
    assert "platform's Parasail key is used only inside the paid OPD worker" not in guide


def test_catalog_contains_exactly_three_parasail_aliases():
    assert set(TEACHER_MODELS) == {"kimi-k3", "glm-5.2", "qwen3.5-397b-a17b"}
    assert resolve_teacher("").alias == "glm-5.2"
    assert {model.model_id for model in TEACHER_MODELS.values()} == {
        "parasail-kimi-k3-fast",
        "parasail-glm-52",
        "parasail-qwen35-397b-a17b",
    }
    for rejected in (
        "kimi-k2.6",
        "deepseek-v4-pro",
        "moonshotai/Kimi-K3",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "parasail-glm-52",
        "accounts/fireworks/models/glm-5p2",
    ):
        with pytest.raises(ValueError, match="not a supported teacher"):
            resolve_teacher(rejected)


def test_pinned_tokenizer_hashes_are_complete():
    assert dict(TEACHER_MODELS["kimi-k3"].tokenizer_files) == {
        "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
        "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    }
    assert dict(TEACHER_MODELS["glm-5.2"].tokenizer_files)["tokenizer.json"] == (
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
    )
    assert dict(TEACHER_MODELS["qwen3.5-397b-a17b"].tokenizer_files)["tokenizer.json"] == (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    )


def test_normalized_offsets_expand_to_exact_source_coverage():
    tokens = _exact_source_spans("é", [7], [(0, 1)])
    assert tokens == [EncodedTeacherToken(7, 0, 2)]


def test_multibyte_split_tokens_keep_byte_safe_overlapping_character_spans():
    spans = _byte_piece_offsets("🙂x", [b"\xf0", b"\x9f\x99\x82", b"x"])
    assert spans == [(0, 1), (0, 1), (1, 2)]
    tokens = _exact_source_spans("🙂x", [1, 2, 3], spans)
    assert [(token.start, token.end) for token in tokens] == spans


def test_token_keyed_response_scores_boundary_crossing_token_and_bills_one_output():
    capture = {}
    scored = _client(_token_keyed_response(), capture).score("P: ", "hi!")

    assert capture == {
        "path": "/v1/teacher/completions",
        "body": {
            "model": "parasail-glm-52",
            "prompt": "P: hi!",
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "return_token_ids": True,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        },
    }
    assert isinstance(scored, TeacherScore)
    assert isinstance(scored.tokens, tuple)
    assert [(token.text, token.start, token.end, token.logprob) for token in scored.tokens] == [
        (": hi", 0, 2, -0.5),
        ("!", 2, 3, -0.2),
    ]
    assert scored.input_tokens == 3
    assert scored.output_tokens == 1
    with pytest.raises(FrozenInstanceError):
        scored.input_tokens = 0
    unbilled = scored.without_billing()
    assert unbilled.tokens is scored.tokens
    assert (unbilled.input_tokens, unbilled.output_tokens) == (0, 0)
    assert (scored.input_tokens, scored.output_tokens) == (3, 1)


def test_positional_glm_response_normalizes_only_after_id_parity():
    scored = _client(_positional_response()).score("P: ", "hi!")
    assert [token.logprob for token in scored.tokens] == [-0.5, -0.2]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response["choices"].append({}), "exactly one choice"),
        (
            lambda response: response["choices"][0].update(prompt_token_ids=[10, 99, 30]),
            "do not match",
        ),
        (lambda response: response["choices"][0].update(token_ids=[]), "generated token"),
        (lambda response: response.update(usage=None), "missing usage"),
        (
            lambda response: response["usage"].update(completion_tokens=0, total_tokens=3),
            "usage does not match",
        ),
        (
            lambda response: response["choices"][0]["prompt_logprobs"][1]["20"].update(
                logprob=None
            ),
            "nonnumeric",
        ),
        (
            lambda response: response["choices"][0]["prompt_logprobs"][1]["20"].update(logprob=0.2),
            "positive",
        ),
    ],
)
def test_response_contract_rejects_malformed_values(mutate, message):
    response = _token_keyed_response()
    mutate(response)
    with pytest.raises(TeacherError, match=message) as error:
        _client(response).score("P: ", "hi!")
    assert error.value.permanent is True


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_response_contract_rejects_nonfinite_scores(score):
    response = _token_keyed_response()
    response["choices"][0]["prompt_logprobs"][1]["20"]["logprob"] = score
    with pytest.raises(TeacherError, match="nonfinite"):
        _client(response).score("P: ", "hi!")


def test_multimodal_scoring_surface_is_removed():
    assert not hasattr(_client(_token_keyed_response()), "score_many_multimodal")
