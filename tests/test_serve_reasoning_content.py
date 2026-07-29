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

    lines = [f'data: {json.dumps({"choices": [{"delta": d}]})}' for d in deltas]
    lines.append("data: [DONE]")
    return lines


def test_split_reasoning_is_folded_back_into_a_balanced_block():
    message = {"content": "the answer", "reasoning_content": "weighing it up"}
    assert deploy._balanced_thinking_content(message) == "<think>weighing it up</think>the answer"


def test_content_without_reasoning_is_returned_verbatim():
    assert deploy._balanced_thinking_content({"content": "plain answer"}) == "plain answer"
    assert deploy._balanced_thinking_content({"content": "plain", "reasoning_content": ""}) == (
        "plain"
    )
    assert deploy._balanced_thinking_content({"content": "plain", "reasoning_content": None}) == (
        "plain"
    )
    assert deploy._balanced_thinking_content({}) == ""


def test_content_that_already_closes_a_block_is_not_nested_again():
    # a serving build that leaves the tags inline must not gain a second, outer block.
    message = {"content": "reasoned</think>answer", "reasoning_content": "reasoned"}
    assert deploy._balanced_thinking_content(message) == "reasoned</think>answer"


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
