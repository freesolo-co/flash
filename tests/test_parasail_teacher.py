from __future__ import annotations

import io
import json
import math
import sqlite3
import time
import urllib.error
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flash.engine.plan.recipe import TEACHER_MODELS, resolve_teacher
from flash.engine.worker.teacher.client import (
    TeacherClient,
    TeacherError,
    TeacherScore,
    _chat_messages,
)
from flash.engine.worker.teacher.encoding import (
    EncodedTeacherToken,
    _byte_piece_offsets,
    _exact_source_spans,
)


class _FixedTokenizer:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def encode(self, _text):
        return list(self.tokens)


class _MultimodalTokenizer:
    def encode(self, text):
        if text == "<|image_pad|>":
            return [EncodedTeacherToken(151655, 0, len(text))]
        if text == "<|im_end|>":
            return [EncodedTeacherToken(151645, 0, len(text))]
        # the assistant header's trailing newline, which a new-turn completion is encoded after
        if text == "\n":
            return [EncodedTeacherToken(198, 0, 1)]
        if text == "red":
            return [
                EncodedTeacherToken(201, 0, 1),
                EncodedTeacherToken(202, 1, 3),
            ]
        # "red" does not merge with the header newline, so the joined encoding is the plain
        # concatenation and the prefix-diff tail is exactly the completion's own ids.
        if text == "\nred":
            return [
                EncodedTeacherToken(198, 0, 1),
                EncodedTeacherToken(201, 1, 2),
                EncodedTeacherToken(202, 2, 4),
            ]
        raise AssertionError(f"unexpected tokenizer input: {text!r}")


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


def _multimodal_response(prompt_ids=None):
    # the completion sits at the END of the supplied assistant turn, closed by <|im_end|> (151645)
    # and followed by the chat template's own generation header. that trailing header is what the
    # live route really returns, and scoring must not mistake it for the completion.
    prompt_ids = prompt_ids or [10, 151655, 11, 201, 202, 151645, 198]
    return {
        "choices": [{"index": 0, "token_ids": [999], "message": {"role": "assistant"}}],
        "prompt_token_ids": prompt_ids,
        "prompt_logprobs": [
            None,
            *[
                {str(token_id): {"logprob": -0.1 * index, "rank": 1}}
                for index, token_id in enumerate(prompt_ids[1:], 1)
            ],
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "total_tokens": len(prompt_ids) + 1,
        },
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


def _multimodal_client(response, capture=None):
    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_MultimodalTokenizer(),
    )

    def post(path, body):
        if capture is not None:
            capture.update({"path": path, "body": body})
        return response

    client._post = post
    return client


def test_training_guide_lists_only_the_managed_teacher_aliases():
    guide = (Path(__file__).parents[1] / "flash" / "cli" / "scaffold" / "TRAINING.md").read_text()

    assert "glm-5.2 (default) | kimi-k3 |" in guide
    assert "qwen3.5-397b-a17b | deepseek-v4-pro |" in guide
    assert "#                                                       # qwen3-vl-235b" in guide
    assert "kimi-k2.6" not in guide
    assert "The control-plane broker owns `PARASAIL_API_KEY`" in guide
    assert "`FLASH_PUBLIC_URL` and an attempt-scoped `FLASH_TEACHER_CAPABILITY`" in guide
    assert "platform's Parasail key is used only inside the paid OPD worker" not in guide


def test_catalog_contains_exactly_five_parasail_aliases():
    assert set(TEACHER_MODELS) == {
        "kimi-k3",
        "glm-5.2",
        "qwen3.5-397b-a17b",
        "deepseek-v4-pro",
        "qwen3-vl-235b",
    }
    assert resolve_teacher("").alias == "glm-5.2"
    assert {model.model_id for model in TEACHER_MODELS.values()} == {
        "parasail-kimi-k3-fast",
        "parasail-glm-52",
        "parasail-qwen35-397b-a17b",
        "parasail-deepseek-v4-pro",
        "parasail-qwen3-vl-235b-a22b-instruct",
    }
    # qwen3.5 is a vision model despite the alias carrying no "-vl": the checkpoint is a
    # ForConditionalGeneration with a vision_config, and the served route tokenizes an image and
    # answers questions about it. asserting the exact set keeps a text-only teacher from being
    # marked image-capable, which would not fail loudly -- a text route accepts the request and
    # drops the image.
    assert {alias for alias, model in TEACHER_MODELS.items() if model.supports_images} == {
        "qwen3-vl-235b",
        "qwen3.5-397b-a17b",
    }
    assert {alias for alias, model in TEACHER_MODELS.items() if not model.supports_images} == {
        "kimi-k3",
        "glm-5.2",
        "deepseek-v4-pro",
    }
    for rejected in (
        "kimi-k2.6",
        "deepseek-ai/DeepSeek-V4-Pro",
        "moonshotai/Kimi-K3",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        "parasail-glm-52",
        "parasail-deepseek-v4-pro",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/glm-5p2",
    ):
        with pytest.raises(ValueError, match="not a supported teacher"):
            resolve_teacher(rejected)


def test_removed_qwen3_vl_8b_alias_is_not_resolvable():
    with pytest.raises(ValueError, match="not a supported teacher"):
        resolve_teacher("qwen3-vl-8b")


def test_image_capable_aliases_come_from_the_catalog_in_order():
    """The image-teacher list a user is shown must be derived, not written out beside the check."""
    from flash.engine.plan.recipe import image_capable_teacher_aliases

    aliases = image_capable_teacher_aliases()
    assert aliases == ("qwen3.5-397b-a17b", "qwen3-vl-235b"), aliases
    # catalog order, so the message is stable rather than set-iteration order.
    assert list(aliases) == [a for a in TEACHER_MODELS if TEACHER_MODELS[a].supports_images]


def test_image_capable_teachers_declare_the_image_pad_token():
    """A teacher may only be flagged image-capable if its own tokenizer can express an image.

    `_score_one_multimodal` raises a permanent error unless the teacher tokenizer encodes
    `<|image_pad|>` as exactly one token, so flagging a teacher whose pinned tokenizer lacks that
    marker would fail every image rollout after the gpu is already rented. This reads the pinned
    tokenizer contract in the catalog rather than the network.
    """
    for alias, teacher in TEACHER_MODELS.items():
        if not teacher.supports_images:
            continue
        assert teacher.tokenizer_kind == "tokenizer_json", (
            f"{alias} is flagged image-capable but pins a {teacher.tokenizer_kind} tokenizer; the "
            "multimodal scoring path needs the image-pad marker as a single token."
        )
        assert "vl" in teacher.tokenizer_repo.lower() or "qwen3.5" in teacher.tokenizer_repo.lower()


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
    assert dict(TEACHER_MODELS["deepseek-v4-pro"].tokenizer_files)["tokenizer.json"] == (
        "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
    )
    assert TEACHER_MODELS["qwen3-vl-235b"].tokenizer_revision == (
        "710c13861be6c466e66de3f484069440b8f31389"
    )
    assert dict(TEACHER_MODELS["qwen3-vl-235b"].tokenizer_files)["tokenizer.json"] == (
        "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
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


def test_multimodal_scoring_builds_chat_body_and_reads_top_level_prompt_scores():
    capture = {}
    image_uri = "data:image/png;base64,aW1hZ2U="
    scored = _multimodal_client(_multimodal_response(), capture).score_many_multimodal(
        [
            (
                [
                    {"role": "system", "content": "rules"},
                    {
                        "role": "user",
                        "content": "before<|media_pad|>after",
                    },
                ],
                "red",
                [image_uri],
            )
        ]
    )[0]

    assert capture == {
        "path": "/v1/teacher/chat_completions",
        "body": {
            "model": "parasail-qwen3-vl-235b-a22b-instruct",
            "messages": [
                {"role": "system", "content": "rules"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": "after"},
                    ],
                },
                {"role": "assistant", "content": "red"},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "seed": 0,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        },
    }
    assert [(token.text, token.start, token.end, token.logprob) for token in scored.tokens] == [
        ("r", 0, 1, -0.30000000000000004),
        ("ed", 1, 3, -0.4),
    ]
    assert (scored.input_tokens, scored.output_tokens) == (7, 1)


def test_multimodal_scoring_rejects_prompt_ids_unchanged_by_image():
    response = _multimodal_response([10, 11, 201, 202, 151645, 198])

    with pytest.raises(TeacherError, match="silently dropped") as error:
        _multimodal_client(response).score_many_multimodal(
            [
                (
                    [{"role": "user", "content": "<|media_pad|>"}],
                    "red",
                    ["data:image/png;base64,aW1hZ2U="],
                )
            ]
        )

    assert error.value.permanent is True


def test_multimodal_completion_ids_must_be_present():
    response = _multimodal_response([10, 151655, 11, 203, 204, 151645, 198])

    with pytest.raises(TeacherError, match="does not end the supplied assistant turn") as error:
        _multimodal_client(response).score_many_multimodal(
            [
                (
                    [{"role": "user", "content": "<|media_pad|>"}],
                    "red",
                    ["data:image/png;base64,aW1hZ2U="],
                )
            ]
        )

    assert error.value.permanent is True


def test_multimodal_scoring_anchors_to_the_supplied_turn_not_a_quoting_prompt():
    # a valid rollout whose PROMPT quotes the answer: the completion ids appear twice. requiring
    # global uniqueness would permanently fail it, so anchor to the turn the completion closes.
    # the scored logprobs must come from that occurrence, not the earlier quote.
    prompt_ids = [10, 151655, 201, 202, 11, 201, 202, 151645, 198]
    scored = _multimodal_client(_multimodal_response(prompt_ids)).score_many_multimodal(
        [
            (
                [{"role": "user", "content": "<|media_pad|>"}],
                "red",
                ["data:image/png;base64,aW1hZ2U="],
            )
        ]
    )[0]

    # logprob is -0.1 * index, so the trailing pair (indices 5, 6) is distinguishable from the
    # leading one (indices 2, 3). asserting the VALUES is what proves which run was scored.
    assert [round(token.logprob, 4) for token in scored.tokens] == [-0.5, -0.6]


def test_multimodal_scoring_ignores_the_trailing_generation_header():
    # the defect this anchoring exists for. the chat route appends its own
    # "<|im_start|>assistant\n" after our supplied turn, and a completion can collide with those
    # template ids -- confirmed live, where a completion of "assistant" returned
    # [..., 77091, 151645, 198, 151644, 77091, 198]. picking the LAST matching run scores fixed
    # template tokens as the model's answer, and every structural check still passes, so the
    # corruption is silent. the completion here is a single id (77091) that appears both where the
    # student answered (index 3) and inside the trailing header (index 6).
    class _CollidingTokenizer:
        def encode(self, text):
            table = {
                "<|image_pad|>": [EncodedTeacherToken(151655, 0, len(text))],
                "<|im_end|>": [EncodedTeacherToken(151645, 0, len(text))],
                "\n": [EncodedTeacherToken(198, 0, 1)],
                "assistant": [EncodedTeacherToken(77091, 0, len("assistant"))],
                # no merge across the header boundary here, so the tail is just 77091
                "\nassistant": [
                    EncodedTeacherToken(198, 0, 1),
                    EncodedTeacherToken(77091, 1, 10),
                ],
            }
            if text in table:
                return list(table[text])
            raise AssertionError(f"unexpected tokenizer input: {text!r}")

    prompt_ids = [10, 151655, 11, 77091, 151645, 198, 151644, 77091, 198]
    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_CollidingTokenizer(),
    )
    client._post = lambda path, body: _multimodal_response(prompt_ids)

    scored = client.score_many_multimodal(
        [
            (
                [{"role": "user", "content": "<|media_pad|>"}],
                "assistant",
                ["data:image/png;base64,aW1hZ2U="],
                False,
            )
        ]
    )[0]

    # logprob is -0.1 * index: the real answer sits at index 3 (-0.3), the header's copy at
    # index 7 (-0.7). asserting the VALUE is what distinguishes the two.
    assert [round(token.logprob, 4) for token in scored.tokens] == [-0.3]


def test_multimodal_scoring_rejects_a_partially_dropped_image():
    # two images supplied, one expanded. a raw pad COUNT cannot catch this -- the surviving image
    # alone contributes more pads than the image count -- so this is the case that proves the
    # guard counts per-image runs.
    prompt_ids = [10, 151655, 151655, 151655, 11, 201, 202, 151645, 198]
    response = _multimodal_response(prompt_ids)

    with pytest.raises(TeacherError, match="fewer images") as error:
        _multimodal_client(response).score_many_multimodal(
            [
                (
                    [{"role": "user", "content": "<|media_pad|><|media_pad|>"}],
                    "red",
                    ["data:image/png;base64,aW1hZ2U=", "data:image/png;base64,aW1hZ2Uy"],
                )
            ]
        )

    assert error.value.permanent is True


def test_multimodal_scoring_admits_two_genuinely_expanded_images():
    # the paired control: the same two-image request, both expanded, must be ADMITTED. a guard
    # that rejects valid traffic is as broken as one that never fires.
    prompt_ids = [10, 151655, 151655, 11, 151655, 151655, 12, 201, 202, 151645, 198]
    scored = _multimodal_client(_multimodal_response(prompt_ids)).score_many_multimodal(
        [
            (
                [{"role": "user", "content": "<|media_pad|><|media_pad|>"}],
                "red",
                ["data:image/png;base64,aW1hZ2U=", "data:image/png;base64,aW1hZ2Uy"],
            )
        ]
    )[0]

    assert len(scored.tokens) == 2


def test_multimodal_drop_guard_ignores_pads_inside_the_completion():
    # the completion is SAMPLED by the student, and a vision student can emit the literal
    # image-pad token. counting pads over the whole prompt lets that token stand in for an image
    # the provider actually dropped: two supplied, one expanded, and the completion's own pad
    # supplies the second run. the guard passes and the OPD target is image-unconditioned.
    # rejecting the marker in prompt text does not cover it -- the completion is appended after
    # that check. so pads are counted over the prompt region only.
    class _PadInCompletionTokenizer:
        def encode(self, text):
            table = {
                "<|image_pad|>": [EncodedTeacherToken(151655, 0, len("<|image_pad|>"))],
                "<|im_end|>": [EncodedTeacherToken(151645, 0, len("<|im_end|>"))],
                "\n": [EncodedTeacherToken(198, 0, 1)],
                "\n<|image_pad|>x": [
                    EncodedTeacherToken(198, 0, 1),
                    EncodedTeacherToken(151655, 1, 14),
                    EncodedTeacherToken(120, 14, 15),
                ],
            }
            if text in table:
                return list(table[text])
            raise AssertionError(f"unexpected tokenizer input: {text!r}")

    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_PadInCompletionTokenizer(),
    )
    # one real image expanded (index 1-2), then the completion's own pad at index 4. counting the
    # whole list yields 2 runs and admits a request that dropped an image.
    prompt_ids = [10, 151655, 151655, 11, 151655, 120, 151645, 198]
    client._post = lambda path, body: _multimodal_response(prompt_ids)

    with pytest.raises(TeacherError, match="fewer images") as error:
        client.score_many_multimodal(
            [
                (
                    [{"role": "user", "content": "<|media_pad|><|media_pad|>"}],
                    "<|image_pad|>x",
                    [
                        "data:image/png;base64,aW1hZ2U=",
                        "data:image/png;base64,aW1hZ2Uy",
                    ],
                    False,
                )
            ]
        )

    assert error.value.permanent is True


def test_multimodal_completion_opens_a_new_assistant_turn():
    # the student froze its prompt with add_generation_prompt=True, so the sampled tokens follow a
    # NEW assistant boundary. gluing them onto an assistant turn from the environment's own history
    # would score them as a continuation of that turn -- a silently different conditioning prefix,
    # which produces plausible logprobs against the wrong context rather than an error.
    messages = _chat_messages(
        [
            {"role": "user", "content": "<|media_pad|>"},
            {"role": "assistant", "content": "Earlier reply."},
        ],
        "NEW",
        ["data:image/png;base64,aW1hZ2U="],
    )

    assert messages[-2] == {"role": "assistant", "content": "Earlier reply."}
    assert messages[-1] == {"role": "assistant", "content": "NEW"}


def test_multimodal_thinking_prefill_continues_the_same_assistant_turn():
    # the ONE case that legitimately continues the trailing turn: the student sampled AFTER the
    # synthetic prefill inside that same assistant turn, so the teacher must condition identically.
    messages = _chat_messages(
        [
            {"role": "user", "content": "<|media_pad|>"},
            {"role": "assistant", "content": "<think>\n"},
        ],
        "NEW",
        ["data:image/png;base64,aW1hZ2U="],
        continue_final_assistant=True,
    )

    assert [m["role"] for m in messages].count("assistant") == 1
    assert messages[-1] == {"role": "assistant", "content": "<think>\nNEW"}


def test_multimodal_completion_is_encoded_after_its_assistant_prefix():
    # BPE merges across the prefix/completion boundary: a completion starting with "\n" placed
    # after a thinking prefill ending in "\n" becomes ONE token. encoding the completion alone then
    # yields ids that never occur in the provider's rendered prompt, so the contiguous-run lookup
    # either rejects a valid rollout or matches the standalone ids somewhere earlier and silently
    # scores the wrong text. this stub reproduces the merge: 198 + 198 -> 271.
    class _MergingTokenizer:
        def encode(self, text):
            if text == "<|image_pad|>":
                return [EncodedTeacherToken(151655, 0, len(text))]
            if text == "<|im_end|>":
                return [EncodedTeacherToken(151645, 0, len(text))]
            table = {
                "<think>\n": [EncodedTeacherToken(151667, 0, 7), EncodedTeacherToken(198, 7, 8)],
                "\nred": [EncodedTeacherToken(198, 0, 1), EncodedTeacherToken(1151, 1, 4)],
                # the merged rendering: the two newlines collapse into 271
                "<think>\n\nred": [
                    EncodedTeacherToken(151667, 0, 7),
                    EncodedTeacherToken(271, 7, 9),
                    EncodedTeacherToken(1151, 9, 12),
                ],
            }
            if text in table:
                return list(table[text])
            raise AssertionError(f"unexpected tokenizer input: {text!r}")

    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_MergingTokenizer(),
    )
    # the provider's rendered prompt contains the MERGED ids [271, 1151]; the standalone encoding
    # [198, 1151] does not occur anywhere in it. scoring must still find the completion, so this
    # only passes if the request path encodes in context rather than in isolation.
    prompt_ids = [10, 151655, 11, 271, 1151, 151645, 198]
    client._post = lambda path, body: _multimodal_response(prompt_ids)

    scored = client.score_many_multimodal(
        [
            (
                [
                    {"role": "user", "content": "<|media_pad|>"},
                    {"role": "assistant", "content": "<think>\n"},
                ],
                "\nred",
                ["data:image/png;base64,aW1hZ2U="],
                True,
            )
        ]
    )[0]

    # two scored tokens means the merged run was located; standalone encoding would raise instead.
    assert len(scored.tokens) == 2


def test_multimodal_new_turn_completion_merges_with_the_assistant_header():
    # the same merge happens with NO thinking prefill. the chat template renders
    # "<|im_start|>assistant\n", so a completion starting with "\n" merges against the header's
    # own newline exactly as it would against a prefill. measured on the pinned Qwen3-VL
    # tokenizer, isolated encoding loses 5 of 8 whitespace-leading completions: "\nred" encodes
    # to [198, 1151] while the render contains [..., 77091, 271, 1151]. encoding a new turn in
    # isolation therefore rejects those rollouts permanently.
    class _HeaderMergingTokenizer:
        def encode(self, text):
            table = {
                "<|image_pad|>": [EncodedTeacherToken(151655, 0, len(text))],
                "<|im_end|>": [EncodedTeacherToken(151645, 0, len(text))],
                "\n": [EncodedTeacherToken(198, 0, 1)],
                "\nred": [EncodedTeacherToken(198, 0, 1), EncodedTeacherToken(1151, 1, 4)],
                # header newline + completion newline collapse into 271, exactly as BPE does
                "\n\nred": [
                    EncodedTeacherToken(271, 0, 2),
                    EncodedTeacherToken(1151, 2, 5),
                ],
            }
            if text in table:
                return list(table[text])
            raise AssertionError(f"unexpected tokenizer input: {text!r}")

    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_HeaderMergingTokenizer(),
    )
    # what the provider really returns: the merged [271, 1151]. the isolated encoding [198, 1151]
    # never occurs, so this only passes if a new turn is encoded after the header's newline.
    prompt_ids = [10, 151655, 11, 271, 1151, 151645, 198]
    client._post = lambda path, body: _multimodal_response(prompt_ids)

    scored = client.score_many_multimodal(
        [
            (
                [{"role": "user", "content": "<|media_pad|>"}],
                "\nred",
                ["data:image/png;base64,aW1hZ2U="],
                False,
            )
        ]
    )[0]

    # two scored tokens means the MERGED run [271, 1151] was located. the isolated encoding
    # [198, 1151] does not occur in prompt_ids at all, so this raises without the header prefix.
    assert len(scored.tokens) == 2
    # the spans still address the COMPLETION, not the header: the merged token clamps to 0 and the
    # last token ends at len("\nred"), so the caller's slice of completion_text stays exact.
    assert (scored.tokens[0].start, scored.tokens[-1].end) == (0, 4)


def test_multimodal_encoding_rejects_a_boundary_that_eats_multiple_prefix_tokens():
    # real BPE only merges the token straddling the seam, so the tail starts at most one token
    # inside the prefix. a tokenizer that rewrote MORE would return tokens whose text is largely
    # prefill, and scoring those as the completion would attribute the prefill's logprobs to
    # sampled tokens -- wrong in the silent direction, so fail closed instead.
    class _OvermergingTokenizer:
        def encode(self, text):
            table = {
                "AB": [
                    EncodedTeacherToken(1, 0, 1),
                    EncodedTeacherToken(2, 1, 2),
                    EncodedTeacherToken(3, 2, 3),
                ],
                # one shared token, then a single token swallowing the rest of the prefix plus C
                "ABC": [
                    EncodedTeacherToken(1, 0, 1),
                    EncodedTeacherToken(9, 1, 3),
                    EncodedTeacherToken(4, 3, 4),
                ],
            }
            if text in table:
                return list(table[text])
            raise AssertionError(f"unexpected tokenizer input: {text!r}")

    client = TeacherClient(
        "capability",
        "https://broker.example",
        "parasail-qwen3-vl-235b-a22b-instruct",
        tokenizer=_OvermergingTokenizer(),
    )

    with pytest.raises(TeacherError, match="rewrote more than one prefix token"):
        client._encode_completion_in_context(
            [
                {"role": "user", "content": "<|media_pad|>"},
                {"role": "assistant", "content": "AB"},
            ],
            "C",
            continue_final_assistant=True,
        )


def test_score_many_caps_in_flight_requests_at_the_measured_ceiling():
    # the multi-turn path hands score_many a WHOLE EPISODE (up to OPD_MAX_EPISODE_TURNS = 64) in
    # one call rather than pre-slicing it, so score_many's own pool is now the only thing standing
    # between an episode and the provider. OPD_TEACHER_SCORING_CONCURRENCY is a measured
    # rejection-free ceiling and a shed request is a LOST TEACHER SCORE, so assert the observable
    # property -- peak simultaneous requests -- rather than the pool's declared max_workers.
    import threading
    import time

    from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

    client = _client(_token_keyed_response())
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def score_one(_prompt, _completion):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # hold each call open so every worker the pool is willing to run overlaps here.
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return TeacherScore(tokens=[], input_tokens=3, output_tokens=1)

    client._score_one = score_one
    items = [
        (f"prompt-{index}", f"completion-{index}")
        for index in range(2 * OPD_TEACHER_SCORING_CONCURRENCY)
    ]

    results = client.score_many(items)

    assert len(results) == len(items)
    assert peak <= OPD_TEACHER_SCORING_CONCURRENCY


def test_score_many_stops_billing_requests_after_one_fails():
    """A failed teacher request must not drag the rest of the episode through the provider.

    These are PAID requests and the caller retries the whole OPD attempt on a raise, so every
    request submitted after the failure is billed for a result nobody reads. The pre-PR code
    pre-sliced the episode into waves, which capped that waste at one wave.

    The SLOW FIRST REQUEST is the point. Consuming results in input order cannot report the failure
    until request 0 returns, by which time the whole episode has been sent: measured 64/64 that
    way, against ~one pool width here. Teacher latency tracks completion length, so an uneven
    episode is the normal case, not a corner.
    """
    import threading
    import time

    from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY

    client = _client(_token_keyed_response())
    lock = threading.Lock()
    executed = []

    def score_one(prompt, _completion):
        with lock:
            executed.append(prompt)
        time.sleep(0.5 if prompt == "prompt-0" else 0.01)
        if prompt == "prompt-5":
            raise RuntimeError("teacher exploded")
        return TeacherScore(tokens=[], input_tokens=3, output_tokens=1)

    client._score_one = score_one
    total = 2 * OPD_TEACHER_SCORING_CONCURRENCY
    items = [(f"prompt-{index}", f"completion-{index}") for index in range(total)]

    with pytest.raises(RuntimeError, match="teacher exploded"):
        client.score_many(items)

    assert len(executed) < total, "the whole episode was billed after the failure"
    # measured 37 of 64, stable across runs. asserted as a CEILING of one pool width past the
    # failure point, not an equality: the exact count depends on how many workers have picked up
    # work when the raise lands. an input-order consumer scores 64 here and fails this.
    assert len(executed) <= OPD_TEACHER_SCORING_CONCURRENCY + 6, len(executed)


# --------------------------------------------------------------------------------------------------
# transient-failure retry through the real broker and ledger
# --------------------------------------------------------------------------------------------------
class _WholeTextTokenizer:
    def encode(self, text):
        return [EncodedTeacherToken(7, 0, len(text))]


class _StaticResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


class _LedgerBrokerTransport:
    """Drive the worker client through the real broker and sqlite ledger in-process."""

    def __init__(
        self, token, *, drop_responses=0, strip_error_bodies=0, proxy_error_bodies=0, proxy_body=b""
    ):
        self.token = token
        self.drop_responses = drop_responses
        self.strip_error_bodies = strip_error_bodies
        # an intermediary that answers with its OWN json rather than deleting the body. this is
        # the shape that is not covered by `strip_error_bodies`: the body parses, so a check that
        # asks only "did a body arrive" reads it as the broker's verdict.
        self.proxy_error_bodies = proxy_error_bodies
        self.proxy_body = proxy_body
        self.request_ids = []

    def urlopen(self, request, *, timeout):
        del timeout
        from flash.server.domain.teacher import broker as teacher_broker

        request_id = dict(request.header_items())["X-flash-teacher-request-id"]
        self.request_ids.append(request_id)
        try:
            response = teacher_broker.complete_teacher_request(
                capability_token=self.token,
                request_id=request_id,
                raw_body=request.data,
            )
        except teacher_broker.TeacherBrokerError as error:
            body = json.dumps(error.payload()).encode()
            if self.strip_error_bodies > 0:
                self.strip_error_bodies -= 1
                body = b""
            elif self.proxy_error_bodies > 0:
                self.proxy_error_bodies -= 1
                body = self.proxy_body
            raise urllib.error.HTTPError(
                request.full_url,
                error.status_code,
                error.code,
                {},
                io.BytesIO(body),
            ) from None
        if self.drop_responses > 0:
            self.drop_responses -= 1
            raise ConnectionResetError("worker-side response loss")
        return _StaticResponse(json.dumps(response).encode())


def _parasail_success():
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "prompt_token_ids": [7],
                    "token_ids": [8],
                    "prompt_logprobs": [{"7": {"logprob": -0.1, "decoded_token": "answer"}}],
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


@pytest.fixture
def broker_ledger(monkeypatch, tmp_path):
    from flash.server.domain.teacher import broker as teacher_broker
    from flash.server.platform import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key("test-owner-key")
    db.record_run("run-1", owner["id"])
    monkeypatch.setattr(teacher_broker, "_require_current_attempt", lambda _capability: None)
    monkeypatch.setenv("PARASAIL_API_KEY", "control-plane-only-canary")
    return db.issue_teacher_capability(
        run_id="run-1",
        attempt=2,
        teacher_alias="glm-5.2",
        provider=teacher_broker.PARASAIL_PROVIDER,
        model="parasail-glm-52",
        scoring_mode=teacher_broker.PARASAIL_SCORING_MODE,
        expires_at=time.time() + 600,
        limits={
            "max_requests": 8,
            "max_score_items": 8,
            "max_request_bytes": teacher_broker.MAX_REQUEST_BODY_BYTES,
            "max_response_bytes": teacher_broker.MAX_RESPONSE_BODY_BYTES,
            "max_concurrency": 2,
            "max_upstream_attempts": teacher_broker.MAX_UPSTREAM_ATTEMPTS,
            "max_request_tokens": 128,
            "max_total_tokens": 512,
        },
    )


def _ledger_client(
    token, *, drop_responses=0, strip_error_bodies=0, proxy_error_bodies=0, proxy_body=b""
):
    client = TeacherClient(
        token,
        "https://plane.example",
        "parasail-glm-52",
        tokenizer=_WholeTextTokenizer(),
    )
    transport = _LedgerBrokerTransport(
        token,
        drop_responses=drop_responses,
        strip_error_bodies=strip_error_bodies,
        proxy_error_bodies=proxy_error_bodies,
        proxy_body=proxy_body,
    )
    client._transport = transport
    return client, transport


def _ledger_row(request_id):
    from flash.server.platform import db

    connection = sqlite3.connect(db.DB_PATH)
    row = connection.execute(
        "SELECT state, upstream_attempt_count, provider_status, error_class, "
        "input_tokens, output_tokens FROM teacher_score_requests WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    connection.close()
    return row


def test_provider_500_is_terminal_and_never_redispatches(broker_ledger, monkeypatch):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 500, b"provider incident"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger)

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(dispatches) == 1
    assert len(transport.request_ids) == 1
    assert sleeps == []
    state, attempts, status, error_class, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts, status, error_class) == ("outcome_unknown", 1, 500, "permanent")
    assert (input_tokens, output_tokens) == (0, 0)


def test_provider_429_is_retried_with_bounded_backoff(broker_ledger, monkeypatch):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    outcomes = [(429, b"rate limited"), (429, b"rate limited"), (200, _parasail_success())]
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return outcomes[len(dispatches) - 1]

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger)

    scored = client.score("question", "answer")

    assert scored.input_tokens == 1
    assert len(dispatches) == 3
    assert len(set(transport.request_ids)) == 1
    # exponential backoff between attempts, not a hot loop against a rate-limited provider.
    assert sleeps == [2.0, 4.0]
    state, attempts, _status, _error, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts) == ("succeeded", 3)
    assert (input_tokens, output_tokens) == (1, 1)


def test_rate_limit_survives_a_stripped_error_body(broker_ledger, monkeypatch):
    """An upstream 429 stays retryable even when the broker's error body is destroyed in transit.

    A live OPD run died here: an intermediary replaced the broker's structured 502 body with 16
    bytes of `error code: 502`, so the worker fell back to `broker_http_error (permanent)` and
    killed a run the provider had merely rate-limited. The retry signal therefore travels in the
    status line, which intermediaries preserve, and this asserts the whole path end to end with
    the body stripped on the first attempt.
    """
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    outcomes = [(429, b"rate limited"), (200, _parasail_success())]
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return outcomes[len(dispatches) - 1]

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger, strip_error_bodies=1)

    scored = client.score("question", "answer")

    assert scored.input_tokens == 1
    # the bodyless rejection was retried rather than treated as terminal, and the retry is the
    # same logical request, so the provider is not billed for two.
    assert len(dispatches) == 2
    assert len(set(transport.request_ids)) == 1
    assert sleeps == [2.0]


def _worker_client_after_http_error(monkeypatch, *, status, reason, body):
    from flash.engine.worker.teacher import client as worker_teacher

    request_ids = []

    def urlopen(_transport, request, timeout=None):
        del timeout
        request_ids.append(dict(request.header_items())["X-flash-teacher-request-id"])
        if len(request_ids) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                reason,
                {},
                io.BytesIO(body),
            )
        return _StaticResponse(_parasail_success())

    sleeps = []
    monkeypatch.setattr(worker_teacher._ThreadLocalHttpsTransport, "urlopen", urlopen)
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client = worker_teacher.TeacherClient(
        "capability-value",
        "https://broker.example",
        "parasail-glm-52",
        tokenizer=_WholeTextTokenizer(),
    )
    return client, request_ids, sleeps


@pytest.mark.parametrize(
    "ledger_code",
    ["outcome_unknown", "provider_rejected", "provider_contract_error"],
)
def test_bodyless_terminal_409_fails_closed_without_retry(monkeypatch, ledger_code):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker
    from flash.server.platform import db

    broker_error = teacher_broker._map_ledger_error(
        db.TeacherLedgerError(ledger_code), "request-terminal-conflict-001"
    )
    assert broker_error.status_code == 409
    assert broker_error.retryable is False
    client, request_ids, sleeps = _worker_client_after_http_error(
        monkeypatch,
        status=broker_error.status_code,
        reason=broker_error.code,
        body=b"",
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(request_ids) == 1
    assert sleeps == []


def test_bodyless_request_in_progress_409_fails_closed(monkeypatch):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker
    from flash.server.platform import db

    broker_error = teacher_broker._map_ledger_error(
        db.TeacherLedgerError("request_in_progress", retryable=True),
        "request-in-progress-001",
    )
    assert broker_error.status_code == 409
    assert broker_error.retryable is True
    client, request_ids, sleeps = _worker_client_after_http_error(
        monkeypatch,
        status=broker_error.status_code,
        reason=broker_error.code,
        body=b"",
    )

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(request_ids) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("classification", "code", "should_retry"),
    [
        ("transient", "request_in_progress", True),
        ("permanent", "provider_rejected", False),
    ],
)
def test_classified_409_obeys_the_body(monkeypatch, classification, code, should_retry):
    from flash.engine.worker.teacher import client as worker_teacher

    body = json.dumps({"error": {"code": code, "classification": classification}}).encode()
    client, request_ids, sleeps = _worker_client_after_http_error(
        monkeypatch, status=409, reason="conflict", body=body
    )

    if should_retry:
        assert client.score("question", "answer").input_tokens == 1
        assert len(request_ids) == 2
        assert len(set(request_ids)) == 1
        assert sleeps == [2.0]
    else:
        with pytest.raises(worker_teacher.TeacherError) as error:
            client.score("question", "answer")
        assert error.value.permanent is True
        assert len(request_ids) == 1
        assert sleeps == []


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_stripped_ambiguous_5xx_stays_terminal_and_never_redispatches(
    broker_ledger, monkeypatch, status
):
    """A bodyless 5xx must stay permanent: it can arrive after the provider began work.

    This is the safety half of the rule above. Widening the bodyless rescue to "any 5xx" would
    make the run survive, at the cost of dispatching and paying for the same teacher request
    twice, so an ambiguous status must remain terminal even though the body is gone.
    """
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return status, b"upstream failure"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger, strip_error_bodies=1)

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(dispatches) == 1
    assert len(transport.request_ids) == 1
    assert sleeps == []
    state, attempts, ledger_status, error_class, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts, ledger_status, error_class) == (
        "outcome_unknown",
        1,
        status,
        "permanent",
    )
    assert (input_tokens, output_tokens) == (0, 0)


def test_ambiguous_provider_transport_failure_is_terminal_without_redispatch(
    broker_ledger, monkeypatch
):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        raise ConnectionResetError("provider connection dropped mid-call")

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger)

    with pytest.raises(worker_teacher.TeacherError) as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(dispatches) == 1
    assert len(transport.request_ids) == 1
    assert sleeps == []
    state, attempts, status, error_class, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts, status, error_class) == (
        "outcome_unknown",
        1,
        None,
        "provider_transport_ambiguous",
    )
    assert (input_tokens, output_tokens) == (0, 0)


def test_lost_worker_response_replays_the_succeeded_result_without_a_second_bill(
    broker_ledger, monkeypatch
):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 200, _parasail_success()

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    # the broker settles the request as succeeded, then the worker loses the response in
    # transit. the retry with the same request_id must replay from the ledger, not redispatch.
    client, transport = _ledger_client(broker_ledger, drop_responses=1)

    scored = client.score("question", "answer")

    assert scored.input_tokens == 1
    assert len(dispatches) == 1
    assert len(transport.request_ids) == 2
    assert len(set(transport.request_ids)) == 1
    state, attempts, _status, _error, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts) == ("succeeded", 1)
    assert (input_tokens, output_tokens) == (1, 1)


def test_provider_401_rejection_stays_permanent_and_never_retries(broker_ledger, monkeypatch):
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 401, b"invalid provider key"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _seconds: None)
    client, transport = _ledger_client(broker_ledger)

    with pytest.raises(TeacherError, match="provider_rejected") as error:
        client.score("question", "answer")

    assert error.value.permanent is True
    assert len(dispatches) == 1
    assert len(transport.request_ids) == 1
    state, attempts, status, error_class, input_tokens, output_tokens = _ledger_row(
        transport.request_ids[0]
    )
    assert (state, attempts, status, error_class) == ("provider_rejected", 1, 401, "permanent")
    assert (input_tokens, output_tokens) == (0, 0)


@pytest.mark.parametrize(
    ("label", "proxy_body"),
    [
        # the shape the finding names: a generic gateway body with no code and no classification.
        ("message only", b'{"error": {"message": "rate limited"}}'),
        # and one that names a code but still never classifies.
        ("code without classification", b'{"error": {"code": "gateway_rate_limited"}}'),
    ],
)
def test_rate_limit_survives_a_proxy_substituted_error_body(
    broker_ledger, monkeypatch, label, proxy_body
):
    """A safe-to-retry 429 stays retryable when an intermediary supplies its OWN JSON body.

    `strip_error_bodies` covers the body being DELETED. This is the other half: the body parses as
    JSON and carries an `error` object, but it never classifies. Treating any dict as the broker's
    structured verdict suppresses the status-only fallback, so the default `permanent` aborts a
    paid OPD run the broker meant the worker to retry against the same request id.
    """
    del label
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    outcomes = [(429, b"rate limited"), (200, _parasail_success())]
    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return outcomes[len(dispatches) - 1]

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    sleeps = []
    monkeypatch.setattr(worker_teacher.time, "sleep", sleeps.append)
    client, transport = _ledger_client(broker_ledger, proxy_error_bodies=1, proxy_body=proxy_body)

    scored = client.score("question", "answer")

    assert scored.input_tokens == 1
    # retried, and as the same logical request so the provider is billed once.
    assert len(dispatches) == 2
    assert len(set(transport.request_ids)) == 1
    assert sleeps == [2.0]


def test_proxy_body_that_does_classify_permanent_is_still_obeyed(broker_ledger, monkeypatch):
    """The relaxation must not make a real `permanent` verdict retryable.

    A body that actually classifies stays authoritative, including when it says the failure is
    terminal on a status the bodyless fallback would otherwise rescue.
    """
    from flash.engine.worker.teacher import client as worker_teacher
    from flash.server.domain.teacher import broker as teacher_broker

    dispatches = []

    def dispatch(*_args):
        dispatches.append(1)
        return 429, b"rate limited"

    monkeypatch.setattr(teacher_broker, "_provider_post", dispatch)
    monkeypatch.setattr(worker_teacher.time, "sleep", lambda _s: None)
    client, _transport = _ledger_client(
        broker_ledger,
        proxy_error_bodies=1,
        proxy_body=b'{"error": {"code": "upstream_rejected", "classification": "permanent"}}',
    )

    with pytest.raises(worker_teacher.TeacherError, match="permanent"):
        client.score("question", "answer")
    assert len(dispatches) == 1
