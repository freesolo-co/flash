"""Serving hands reasoning back in ``reasoning_content``, split out of ``content``.

A thinking chat template renders the OPENING ``<think>`` into the prompt, so the model samples only
the closing tag. The flash side must fold the two fields back into one balanced string: reading
``content`` alone drops the whole reasoning phase, and reading the raw text alone yields a stray
``</think>`` with no opener.
"""

from __future__ import annotations

from typing import ClassVar

import flash.serve.deploy as deploy


def _sse(*deltas: dict) -> list[str]:
    import json

    lines = [f"data: {json.dumps({'choices': [{'delta': d}]})}" for d in deltas]
    lines.append("data: [DONE]")
    return lines


def test_split_reasoning_is_folded_back_into_a_balanced_block():
    message = {"content": "the answer", "reasoning_content": "weighing it up"}
    assert deploy._balanced_thinking_content(message) == "<think>weighing it up</think>the answer"


def test_content_without_reasoning_is_returned_verbatim():
    # absent and null both mean serving never split the tags out: nothing to fold back in.
    assert deploy._balanced_thinking_content({"content": "plain answer"}) == "plain answer"
    assert deploy._balanced_thinking_content({"content": "plain", "reasoning_content": None}) == (
        "plain"
    )
    assert deploy._balanced_thinking_content({}) == ""


def test_an_explicitly_empty_reasoning_string_still_gets_a_balanced_pair():
    """`""` is not the same as absent: it means the model closed its block before answering.

    Collapsing the two loses a real distinction. A thinking consumer splits the answer on
    `</think>`, so returning the bare answer here makes a valid completion look like it has no
    answer at all -- a thinking structured-output deployment then fails its smoke test despite
    the model having answered correctly.
    """
    assert (
        deploy._balanced_thinking_content({"content": "plain", "reasoning_content": ""})
        == "<think></think>plain"
    )


def test_content_that_already_closes_a_block_is_not_nested_again():
    # a serving build that leaves the tags inline must not gain a second, outer block -- but it
    # must still gain the OPENER, which the prompt swallowed. the old expectation returned this
    # string verbatim, which left `reasoned</think>answer`: a close with nothing opening it, i.e.
    # exactly the malformed completion this helper exists to repair.
    message = {"content": "reasoned</think>answer", "reasoning_content": "reasoned"}
    assert deploy._balanced_thinking_content(message) == "<think>reasoned</think>answer"


def test_a_bare_inline_close_is_reopened_rather_than_treated_as_balanced():
    """`</think>answer` has a closer and no opener, so `"</think>" in content` is not "balanced".

    The opening tag came from the prompt, so a serving build that also leaves the sampled close in
    `content` produces this shape. Treating the stray closer as proof of a balanced block hands
    CLI and API consumers the very completion this helper repairs.
    """
    message = {"content": "</think>answer", "reasoning_content": "reasoned"}
    out = deploy._balanced_thinking_content(message)
    assert out == "<think>reasoned</think>answer"
    assert out.count("<think>") == 1
    assert out.count("</think>") == 1


def test_a_real_opener_is_left_alone():
    # a genuinely balanced block must pass through untouched, with no second pair.
    message = {"content": "<think>r</think>answer", "reasoning_content": "r"}
    assert deploy._balanced_thinking_content(message) == "<think>r</think>answer"


def test_an_answer_mentioning_the_close_tag_keeps_its_reasoning():
    """A close tag mid-answer is answer text, not the block delimiter.

    Splitting on the first `</think>` anywhere rewrote such an answer into the reasoning slot and
    DROPPED the real reasoning: `<think>answer about </think> tags` -- the reasoning "r" is gone and
    the answer is now the reasoning. The streaming path already only accepts a close at the head of
    the first content delta (`test_streamed_content_keeps_a_close_tag_that_is_not_the_block_
    delimiter`); this is its non-streaming twin.
    """
    message = {"content": "answer about </think> tags", "reasoning_content": "r"}
    out = deploy._balanced_thinking_content(message)

    assert out == "<think>r</think>answer about </think> tags"
    # the reasoning survives, and the answer is not promoted into the reasoning slot.
    assert out.startswith("<think>r</think>")
    # and both paths agree on this input.
    streamed = "".join(
        deploy._openai_stream_content(
            iter(_sse({"reasoning_content": "r"}, {"content": "answer about </think> tags"}))
        )
    )
    assert out == streamed


def test_an_answer_that_is_the_literal_open_tag_is_still_closed():
    """A structured answer may BE `<think>` -- a `choice` constraint equal to it, or JSON quoting it.

    Asking only whether the opener appears anywhere reported that answer as an already-balanced
    block and returned it with no close tag, after which
    `flash/server/routes/serving.py::_thinking_structured_answer` rejects the deployment smoke.
    Balance means an opener that precedes a close, not the mere presence of the substring.
    """
    message = {"content": "<think>", "reasoning_content": "r"}
    out = deploy._balanced_thinking_content(message)

    assert out == "<think>r</think><think>"
    # the answer survives after the delimiter, which is what the smoke's answer split reads.
    assert out.split("</think>", 1)[1] == "<think>"


def test_a_legacy_inline_block_without_the_split_field_is_still_balanced():
    """Version skew: control plane upgraded ahead of the serving backend.

    A backend predating the split leaves `reasoned</think>answer` in `content` with no
    `reasoning_content` field at all. With no field to contradict it an unbalanced close can only be
    the sampled delimiter, so the block is re-opened around the text before it rather than handed
    back malformed.
    """
    out = deploy._balanced_thinking_content({"content": "reasoned</think>answer"})

    assert out == "<think>reasoned</think>answer"
    assert out.count("<think>") == 1
    assert out.count("</think>") == 1


def test_a_legacy_answer_without_any_block_is_still_returned_verbatim():
    # the skew branch must not invent a block for a plain answer that never had one.
    assert deploy._balanced_thinking_content({"content": "plain answer"}) == "plain answer"


def test_payload_balancing_rewrites_every_choice_in_place():
    payload = {
        "choices": [
            {"message": {"content": "a", "reasoning_content": "r1"}},
            {"message": {"content": "b", "reasoning_content": "r2"}},
        ]
    }
    deploy._balance_thinking_payload(payload)
    assert payload["choices"][0]["message"]["content"] == "<think>r1</think>a"
    assert payload["choices"][1]["message"]["content"] == "<think>r2</think>b"
    # the split fields survive for callers that want them.
    assert payload["choices"][0]["message"]["reasoning_content"] == "r1"


def test_payload_balancing_tolerates_shapes_it_cannot_rewrite():
    for payload in (None, [], "text", {}, {"choices": None}, {"choices": [{}, {"message": None}]}):
        deploy._balance_thinking_payload(payload)  # must not raise


def test_streamed_reasoning_opens_and_closes_around_the_answer():
    lines = _sse(
        {"reasoning_content": "weigh"},
        {"reasoning_content": "ing"},
        {"content": "the "},
        {"content": "answer"},
    )
    assert "".join(deploy._openai_stream_content(iter(lines))) == (
        "<think>weighing</think>the answer"
    )


def test_streamed_answer_without_reasoning_is_untouched():
    lines = _sse({"content": "just "}, {"content": "text"})
    assert "".join(deploy._openai_stream_content(iter(lines))) == "just text"


def test_streamed_inline_close_is_not_duplicated_by_the_synthetic_one():
    """The streaming twin of `test_content_that_already_closes_a_block_is_not_nested_again`.

    A compatibility build can emit reasoning on its own delta field AND retain the sampled close
    at the head of the first content delta. Synthesising a second one produced
    `<think>reasoned</think></think>answer`, which leaves `flash models chat` output and any
    downstream parser malformed.
    """
    lines = _sse({"reasoning_content": "reasoned"}, {"content": "</think>answer"})
    out = "".join(deploy._openai_stream_content(iter(lines)))
    assert out == "<think>reasoned</think>answer"
    assert out.count("</think>") == 1


def test_streamed_content_keeps_a_close_tag_that_is_not_the_block_delimiter():
    # only a close at the HEAD of the first content delta is the sampled delimiter. one appearing
    # later is answer text, and stripping it would corrupt the answer.
    lines = _sse({"reasoning_content": "r"}, {"content": "answer about </think> tags"})
    assert "".join(deploy._openai_stream_content(iter(lines))) == (
        "<think>r</think>answer about </think> tags"
    )


def test_streamed_reasoning_cut_off_mid_block_is_still_closed():
    # generation hit the length cap inside the reasoning block: an unbalanced OPENER is the same
    # defect as the unbalanced closer, mirrored, so the stream must still close it.
    lines = _sse({"reasoning_content": "thinking hard"})
    assert "".join(deploy._openai_stream_content(iter(lines))) == "<think>thinking hard</think>"


def test_streamed_block_opens_once_across_many_reasoning_deltas():
    lines = _sse(*[{"reasoning_content": c} for c in "abc"], {"content": "z"})
    streamed = "".join(deploy._openai_stream_content(iter(lines)))
    assert streamed == "<think>abc</think>z"
    assert streamed.count("<think>") == 1
    assert streamed.count("</think>") == 1


def test_non_streaming_chat_balances_before_returning(monkeypatch):
    captured = {}

    class _Resp:
        headers: ClassVar[dict] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "answer", "reasoning_content": "reasoned"},
                        "finish_reason": "stop",
                    }
                ]
            }

    class _Client:
        def post(self, url, **kwargs):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr(deploy, "serving_openai_base_url", lambda: "https://serve.example/v1")
    monkeypatch.setattr(deploy, "_internal_key_header", lambda: {})
    monkeypatch.setattr(deploy, "_chat_http_client", lambda: _Client())

    result = deploy.chat("run-1", [{"role": "user", "content": "hi"}], thinking=True)
    content = result["choices"][0]["message"]["content"]
    assert content == "<think>reasoned</think>answer"
    # the balanced string is what the deployment smoke greps for its thinking-tag telemetry.
    assert "<think>" in content
    assert "</think>" in content
