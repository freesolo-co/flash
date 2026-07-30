"""Serving hands reasoning back in ``reasoning_content``, split out of ``content``.

A thinking chat template renders the OPENING ``<think>`` into the prompt, so the model samples only
the closing tag. The flash side must fold the two fields back into one balanced string: reading
``content`` alone drops the whole reasoning phase, and reading the raw text alone yields a stray
``</think>`` with no opener.

Folding is gated on the request's own ``thinking`` flag rather than inferred from the text. Without
that gate an ordinary answer containing a literal ``</think>`` was rewritten into a synthetic
reasoning block, on a path that also backs the public non-streaming chat route.
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
    assert (
        deploy._balanced_thinking_content(message, thinking=True)
        == "<think>weighing it up</think>the answer"
    )


def test_content_without_reasoning_is_returned_verbatim():
    # absent and null both mean serving never split the tags out: nothing to fold back in.
    assert (
        deploy._balanced_thinking_content({"content": "plain answer"}, thinking=True)
        == "plain answer"
    )
    assert deploy._balanced_thinking_content(
        {"content": "plain", "reasoning_content": None}, thinking=True
    ) == ("plain")
    assert deploy._balanced_thinking_content({}, thinking=True) == ""


def test_an_explicitly_empty_reasoning_string_still_gets_a_balanced_pair():
    """`""` is not the same as absent: it means the model closed its block before answering.

    Collapsing the two loses a real distinction. A thinking consumer splits the answer on
    `</think>`, so returning the bare answer here makes a valid completion look like it has no
    answer at all -- a thinking structured-output deployment then fails its smoke test despite
    the model having answered correctly.
    """
    assert (
        deploy._balanced_thinking_content({"content": "plain", "reasoning_content": ""},
                                          thinking=True)
        == "<think></think>plain"
    )


def test_content_that_already_closes_a_block_is_not_nested_again():
    # a serving build that leaves the tags inline must not gain a second, outer block -- but it
    # must still gain the OPENER, which the prompt swallowed. the old expectation returned this
    # string verbatim, which left `reasoned</think>answer`: a close with nothing opening it, i.e.
    # exactly the malformed completion this helper exists to repair.
    message = {"content": "reasoned</think>answer", "reasoning_content": "reasoned"}
    assert deploy._balanced_thinking_content(message, thinking=True) == "<think>reasoned</think>answer"


def test_a_bare_inline_close_is_reopened_rather_than_treated_as_balanced():
    """`</think>answer` has a closer and no opener, so `"</think>" in content` is not "balanced".

    The opening tag came from the prompt, so a serving build that also leaves the sampled close in
    `content` produces this shape. Treating the stray closer as proof of a balanced block hands
    CLI and API consumers the very completion this helper repairs.
    """
    message = {"content": "</think>answer", "reasoning_content": "reasoned"}
    out = deploy._balanced_thinking_content(message, thinking=True)
    assert out == "<think>reasoned</think>answer"
    assert out.count("<think>") == 1
    assert out.count("</think>") == 1


def test_whitespace_before_the_sampled_close_is_still_the_block_delimiter():
    """The model samples its close on its own line as often as not.

    Requiring an exactly empty prefix rejected `\\n</think>answer`, fell through to wrapping the
    whole string, and produced `<think>reasoned</think>\\n</think>answer` -- two close tags, with
    the real answer stranded behind the first one. `_thinking_structured_answer` splits on that
    first close, so the smoke then read `\\n</think>answer` as the answer (cursor).
    """
    message = {"content": "\n</think>answer", "reasoning_content": "reasoned"}
    out = deploy._balanced_thinking_content(message, thinking=True)

    assert out == "<think>reasoned</think>answer"
    assert out.count("</think>") == 1
    # the shape the smoke actually reads.
    assert out.split("</think>", 1)[1] == "answer"


def test_a_real_opener_is_left_alone():
    # a genuinely balanced block must pass through untouched, with no second pair.
    message = {"content": "<think>r</think>answer", "reasoning_content": "r"}
    assert deploy._balanced_thinking_content(message, thinking=True) == "<think>r</think>answer"


def test_an_inline_pair_that_is_not_the_reasoning_keeps_the_real_reasoning():
    """An answer may contain a full literal pair without that pair being the reasoning block.

    Treating ANY balanced pair as proof the block was already folded returned the answer unchanged
    and DROPPED `reasoning_content` entirely -- the whole reasoning phase, silently. Only a pair
    whose body equals the field is the same block emitted twice (codex[bot], cursor).
    """
    message = {"content": "the tag is <think>like this</think> ok", "reasoning_content": "reasoned"}
    out = deploy._balanced_thinking_content(message, thinking=True)

    assert out == "<think>reasoned</think>the tag is <think>like this</think> ok"
    # the reasoning survives, and the answer is intact after the delimiter.
    assert "reasoned" in out
    assert out.split("</think>", 1)[1] == "the tag is <think>like this</think> ok"


def test_an_answer_that_is_a_whole_literal_pair_keeps_the_real_reasoning():
    # the degenerate case of the above: the answer is EXACTLY a pair, e.g. a `choice` constraint.
    message = {"content": "<think>x</think>", "reasoning_content": "the real reasoning"}
    out = deploy._balanced_thinking_content(message, thinking=True)

    assert out == "<think>the real reasoning</think><think>x</think>"
    assert "the real reasoning" in out


def test_an_answer_mentioning_the_close_tag_keeps_its_reasoning():
    """A close tag mid-answer is answer text, not the block delimiter.

    Splitting on the first `</think>` anywhere rewrote such an answer into the reasoning slot and
    DROPPED the real reasoning: `<think>answer about </think> tags` -- the reasoning "r" is gone and
    the answer is now the reasoning. The streaming path already only accepts a close at the head of
    the first content delta (`test_streamed_content_keeps_a_close_tag_that_is_not_the_block_
    delimiter`); this is its non-streaming twin.
    """
    message = {"content": "answer about </think> tags", "reasoning_content": "r"}
    out = deploy._balanced_thinking_content(message, thinking=True)

    assert out == "<think>r</think>answer about </think> tags"
    # the reasoning survives, and the answer is not promoted into the reasoning slot.
    assert out.startswith("<think>r</think>")
    # and both paths agree on this input.
    streamed = "".join(
        deploy._openai_stream_content(
            iter(_sse({"reasoning_content": "r"}, {"content": "answer about </think> tags"})),
            thinking=True,
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
    out = deploy._balanced_thinking_content(message, thinking=True)

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
    out = deploy._balanced_thinking_content({"content": "reasoned</think>answer"}, thinking=True)

    assert out == "<think>reasoned</think>answer"
    assert out.count("<think>") == 1
    assert out.count("</think>") == 1


def test_a_legacy_answer_without_any_block_is_still_returned_verbatim():
    # the skew branch must not invent a block for a plain answer that never had one.
    assert (
        deploy._balanced_thinking_content({"content": "plain answer"}, thinking=True)
        == "plain answer"
    )


def test_a_non_thinking_response_is_never_rewritten():
    """The request's own flag decides, not the shape of the text.

    Inferring reasoning-ness from content rewrote an ordinary answer that merely contains the
    literal tag -- documentation, a structured field quoting it -- into a synthetic reasoning
    block, and `_balance_thinking_payload` ran on EVERY `chat()` response, including the public
    non-thinking route (codex[bot], cursor).
    """
    for message in (
        {"content": "foo</think>bar"},
        {"content": "</think>answer"},
        {"content": "answer", "reasoning_content": "reasoned"},
        {"content": "<think>x</think>", "reasoning_content": "r"},
    ):
        assert deploy._balanced_thinking_content(message, thinking=False) == message["content"]


def test_payload_balancing_rewrites_every_choice_in_place():
    payload = {
        "choices": [
            {"message": {"content": "a", "reasoning_content": "r1"}},
            {"message": {"content": "b", "reasoning_content": "r2"}},
        ]
    }
    deploy._balance_thinking_payload(payload, thinking=True)
    assert payload["choices"][0]["message"]["content"] == "<think>r1</think>a"
    assert payload["choices"][1]["message"]["content"] == "<think>r2</think>b"
    # the split fields survive for callers that want them.
    assert payload["choices"][0]["message"]["reasoning_content"] == "r1"


def test_payload_balancing_is_a_no_op_outside_thinking_mode():
    payload = {"choices": [{"message": {"content": "a</think>b", "reasoning_content": "r"}}]}
    deploy._balance_thinking_payload(payload, thinking=False)
    assert payload["choices"][0]["message"]["content"] == "a</think>b"


def test_payload_balancing_tolerates_shapes_it_cannot_rewrite():
    for payload in (None, [], "text", {}, {"choices": None}, {"choices": [{}, {"message": None}]}):
        deploy._balance_thinking_payload(payload, thinking=True)  # must not raise


def test_streamed_reasoning_opens_and_closes_around_the_answer():
    lines = _sse(
        {"reasoning_content": "weigh"},
        {"reasoning_content": "ing"},
        {"content": "the "},
        {"content": "answer"},
    )
    assert "".join(deploy._openai_stream_content(iter(lines), thinking=True)) == (
        "<think>weighing</think>the answer"
    )


def test_streamed_answer_without_reasoning_is_untouched():
    lines = _sse({"content": "just "}, {"content": "text"})
    assert "".join(deploy._openai_stream_content(iter(lines), thinking=True)) == "just text"
    assert "".join(deploy._openai_stream_content(iter(lines), thinking=False)) == "just text"


def test_a_non_thinking_stream_yields_each_delta_as_it_arrives():
    # no block can exist outside thinking mode, so nothing is ever held back: the deltas a caller
    # prints must arrive one at a time, not batched at the end.
    lines = _sse({"content": "a"}, {"content": "b"})
    assert list(deploy._openai_stream_content(iter(lines), thinking=False)) == ["a", "b"]


def test_a_split_stream_releases_the_answer_incrementally():
    # once a reasoning delta proves the backend splits, the legacy hold is dropped and the answer
    # streams delta by delta rather than accumulating to the end.
    lines = _sse({"reasoning_content": "r"}, {"content": "an"}, {"content": "swer"})
    assert list(deploy._openai_stream_content(iter(lines), thinking=True)) == [
        "<think>",
        "r",
        "</think>",
        "an",
        "swer",
    ]


def test_streamed_inline_close_is_not_duplicated_by_the_synthetic_one():
    """The streaming twin of `test_content_that_already_closes_a_block_is_not_nested_again`.

    A compatibility build can emit reasoning on its own delta field AND retain the sampled close
    at the head of the first content delta. Synthesising a second one produced
    `<think>reasoned</think></think>answer`, which leaves `flash models chat` output and any
    downstream parser malformed.
    """
    lines = _sse({"reasoning_content": "reasoned"}, {"content": "</think>answer"})
    out = "".join(deploy._openai_stream_content(iter(lines), thinking=True))
    assert out == "<think>reasoned</think>answer"
    assert out.count("</think>") == 1


def test_streamed_content_keeps_a_close_tag_that_is_not_the_block_delimiter():
    # only a close at the HEAD of the first content delta is the sampled delimiter. one appearing
    # later is answer text, and stripping it would corrupt the answer.
    lines = _sse({"reasoning_content": "r"}, {"content": "answer about </think> tags"})
    assert "".join(deploy._openai_stream_content(iter(lines), thinking=True)) == (
        "<think>r</think>answer about </think> tags"
    )


def test_streamed_reasoning_cut_off_mid_block_is_still_closed():
    # generation hit the length cap inside the reasoning block: an unbalanced OPENER is the same
    # defect as the unbalanced closer, mirrored, so the stream must still close it.
    lines = _sse({"reasoning_content": "thinking hard"})
    assert (
        "".join(deploy._openai_stream_content(iter(lines), thinking=True))
        == "<think>thinking hard</think>"
    )


def test_streamed_block_opens_once_across_many_reasoning_deltas():
    lines = _sse(*[{"reasoning_content": c} for c in "abc"], {"content": "z"})
    streamed = "".join(deploy._openai_stream_content(iter(lines), thinking=True))
    assert streamed == "<think>abc</think>z"
    assert streamed.count("<think>") == 1
    assert streamed.count("</think>") == 1


def test_a_legacy_inline_stream_is_reopened_like_its_non_streaming_twin():
    """Version skew reaches the streaming path too, and it repaired nothing.

    A backend predating the split streams `reasoned</think>answer` on `content` with no reasoning
    delta at all. The non-streaming path re-opened around that delimiter and this one did not, so
    the two disagreed on exactly one payload shape and `flash models chat` printed a close tag with
    no opener (cursor).
    """
    lines = _sse({"content": "reasoned</think>answer"})
    streamed = "".join(deploy._openai_stream_content(iter(lines), thinking=True))
    twin = deploy._balanced_thinking_content({"content": "reasoned</think>answer"}, thinking=True)

    assert streamed == "<think>reasoned</think>answer"
    assert streamed == twin


def test_a_legacy_inline_stream_delimiter_split_across_deltas_is_still_found():
    # the sampled close is tokenised, so it straddles delta boundaries routinely. matching only
    # within a single delta would miss it and leave the block unopened.
    lines = _sse({"content": "reasoned</th"}, {"content": "ink>answer"})
    assert (
        "".join(deploy._openai_stream_content(iter(lines), thinking=True))
        == "<think>reasoned</think>answer"
    )


def test_a_thinking_stream_that_never_closes_is_not_wrapped():
    # no delimiter ever arrives, so nothing marked a reasoning phase. wrapping here would label a
    # whole valid answer as reasoning, and the smoke's answer split would find nothing after the
    # close tag and reject a working deployment.
    lines = _sse({"content": "plain "}, {"content": "answer"})
    assert "".join(deploy._openai_stream_content(iter(lines), thinking=True)) == "plain answer"


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


def test_non_thinking_chat_returns_serving_content_unchanged(monkeypatch):
    """`chat()` balanced every response regardless of the flag it was already given.

    A non-thinking answer quoting the tag came back rewritten into a reasoning block, on the path
    that also serves the public non-streaming chat route (codex[bot]).
    """

    class _Resp:
        headers: ClassVar[dict] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "foo</think>bar"}}]}

    class _Client:
        def post(self, url, **kwargs):
            return _Resp()

    monkeypatch.setattr(deploy, "serving_openai_base_url", lambda: "https://serve.example/v1")
    monkeypatch.setattr(deploy, "_internal_key_header", lambda: {})
    monkeypatch.setattr(deploy, "_chat_http_client", lambda: _Client())

    result = deploy.chat("run-1", [{"role": "user", "content": "hi"}])
    assert result["choices"][0]["message"]["content"] == "foo</think>bar"
