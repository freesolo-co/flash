"""Serving returns reasoning in ``reasoning_content``, separate from ``content``.

Flash restores the prompt-supplied opening ``<think>`` and folds both fields into a balanced string.
Only the request's ``thinking`` flag enables folding, so literal tags in normal answers stay intact.
``PARITY_CASES`` requires streaming output to match the non-streaming public route.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

import flash.serve.deployment.deploy as deploy
import flash.serve.request.streaming as serving_streaming
import flash.serve.request.thinking as serving_thinking
import flash.serve.request.transport as serving_transport
from flash.client.http import ClientError


def _sse(*deltas: dict) -> list[str]:
    lines: list[str] = []
    for delta in deltas:
        lines.extend([f"data: {json.dumps({'choices': [{'delta': delta}]})}", ""])
    lines.extend(["data: [DONE]", ""])
    return lines


def _folded(message: dict, *, thinking: bool = True) -> str:
    return serving_thinking._balanced_thinking_content(message, thinking=thinking)


def _streamed(*deltas: dict, thinking: bool = True) -> str:
    return "".join(serving_streaming._openai_stream_content(iter(_sse(*deltas)), thinking=thinking))


def _split(message: dict) -> tuple[dict, ...]:
    """``message`` as serving would stream it: the reasoning field, then the answer."""
    deltas: list[dict] = []
    if isinstance(message.get("reasoning_content"), str):
        deltas.append({"reasoning_content": message["reasoning_content"]})
    if message.get("content"):
        deltas.append({"content": message["content"]})
    return tuple(deltas)


# --------------------------------------------------------------------------------------------
# the fold, on payloads the streaming path cannot receive in this shape.

FOLD_CASES: dict[str, tuple[dict, str]] = {
    "split-fields": (
        {"content": "the answer", "reasoning_content": "weighing it up"},
        "<think>weighing it up</think>the answer",
    ),
    # absent and null both mean serving never split the tags out: nothing to fold back in, and the
    # skew branch must not invent a block for an answer that never had one.
    "no-reasoning-field": ({"content": "plain answer"}, "plain answer"),
    "null-reasoning": ({"content": "plain", "reasoning_content": None}, "plain"),
    "empty-message": ({}, ""),
    # `""` is not absent: the model closed its block before answering. a thinking consumer splits
    # the answer on the close tag, so returning the bare answer makes a valid completion look like
    # it has no answer at all, and the structured smoke then fails a checkpoint that answered.
    "empty-reasoning": ({"content": "plain", "reasoning_content": ""}, "<think></think>plain"),
    # a build that leaves the sampled close inline must not gain a second, outer block -- but it
    # must still gain the OPENER, which the prompt swallowed. the model samples that close on its
    # own line as often as not, so a whitespace prefix is still the delimiter.
    "retained-close": (
        {"content": "reasoned</think>answer", "reasoning_content": "reasoned"},
        "<think>reasoned</think>answer",
    ),
    "bare-retained-close": (
        {"content": "</think>answer", "reasoning_content": "reasoned"},
        "<think>reasoned</think>answer",
    ),
    "retained-close-behind-newline": (
        {"content": "\n</think>answer", "reasoning_content": "reasoned"},
        "<think>reasoned</think>answer",
    ),
    "already-balanced": (
        {"content": "<think>r</think>answer", "reasoning_content": "r"},
        "<think>r</think>answer",
    ),
    # an answer may hold a full literal pair without that pair being the reasoning block. treating
    # ANY balanced pair as proof the block was folded dropped `reasoning_content` entirely. only a
    # pair whose body equals the field is the same block emitted twice.
    "answer-holds-its-own-pair": (
        {"content": "the tag is <think>like this</think> ok", "reasoning_content": "reasoned"},
        "<think>reasoned</think>the tag is <think>like this</think> ok",
    ),
    "answer-is-a-whole-pair": (
        {"content": "<think>x</think>", "reasoning_content": "the real reasoning"},
        "<think>the real reasoning</think><think>x</think>",
    ),
    # balance means an opener that PRECEDES a close, not the mere presence of the substring: a
    # `choice` constraint may be the literal open tag, and returning it unclosed fails the smoke.
    "answer-is-the-open-tag": (
        {"content": "<think>", "reasoning_content": "r"},
        "<think>r</think><think>",
    ),
    # body equality alone mistook a pair inside the ANSWER for the reasoning emitted twice, so an
    # empty-reasoning JSON answer quoting the tag came back with no reasoning block at all.
    "empty-reasoning-answer-quotes-the-pair": (
        {"content": '{"tag":"<think></think>"}', "reasoning_content": ""},
        '<think></think>{"tag":"<think></think>"}',
    ),
    # the duplicate check compares the span stripped, so a trailing newline inside the repeat is
    # the same block rather than a second one.
    "duplicate-past-trailing-newline": (
        {"content": "<think>reasoned\n</think>answer", "reasoning_content": "reasoned"},
        "<think>reasoned\n</think>answer",
    ),
}


@pytest.mark.parametrize(("message", "expected"), FOLD_CASES.values(), ids=FOLD_CASES)
def test_the_fold_rebuilds_a_balanced_block(message, expected):
    assert _folded(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        {"content": "foo</think>bar"},
        {"content": "</think>answer"},
        {"content": "answer", "reasoning_content": "reasoned"},
        {"content": "<think>x</think>", "reasoning_content": "r"},
    ],
)
def test_a_non_thinking_response_is_never_rewritten(message):
    # the request's own flag decides, not the shape of the text. inferring reasoning-ness from
    # content rewrote an ordinary answer that merely contains the literal tag -- documentation, a
    # structured field quoting it -- into a synthetic reasoning block, and
    # `_balance_thinking_payload` runs on EVERY `chat()` response, including the public
    # non-thinking route.
    assert _folded(message, thinking=False) == message["content"]


# --------------------------------------------------------------------------------------------
# payloads both paths receive. each defect below was a disagreement between them on one payload,
# so the assertion that matters is not what either returns alone but that they return the same.

PARITY_CASES: dict[str, tuple[dict, str]] = {
    # a close tag mid-answer is answer text, not the block delimiter. splitting on the first one
    # anywhere promoted the answer into the reasoning slot and DROPPED the real reasoning.
    "close-tag-mid-answer": (
        {"content": "answer about </think> tags", "reasoning_content": "r"},
        "<think>r</think>answer about </think> tags",
    ),
    # version skew, control plane ahead of serving: no `reasoning_content` field at all. with
    # nothing to contradict it an unbalanced close can only be the sampled delimiter.
    "legacy-inline-block": (
        {"content": "reasoned</think>answer"},
        "<think>reasoned</think>answer",
    ),
    # ... and a pair further down the answer must not disarm that re-open. asking whether the close
    # was already matched, with any pair anywhere answering yes, switched off the whole skew fix.
    "legacy-close-with-answer-side-pair": (
        {"content": "reasoned</think>answer <think>x</think> more"},
        "<think>reasoned</think>answer <think>x</think> more",
    ),
    # a `choice` constraint whose value is the literal close tag is byte-identical to a retained
    # delimiter. stripping it rewrote the response to reasoning-only content and the smoke rejected
    # a checkpoint that had answered its grammar correctly. what separates them is what follows.
    "answer-is-the-close-tag": (
        {"content": "</think>", "reasoning_content": "why"},
        "<think>why</think></think>",
    ),
    # at eof, a complete inline block matching `reasoning_content` is treated as a duplicate. the
    # same bytes could be a literal answer, but folding to one block fails the deployment smoke closed
    # instead of accepting a response with no distinguishable answer.
    "terminal-duplicate": (
        {"content": "<think>why</think>", "reasoning_content": "why"},
        "<think>why</think>",
    ),
    # the same repeat without its opener, which the prompt carried. the EOF predicate keyed on a
    # balanced pair and so missed this form, appending it as an answer: a stop firing right after
    # the reasoning then passed the smoke and activated a deployment that never answered.
    "terminal-duplicate-bare": (
        {"content": "why</think>", "reasoning_content": "why"},
        "<think>why</think>",
    ),
    # ... which keys on the body matching. an answer that is its own literal pair is not the
    # repeat, so it must survive rather than be swallowed.
    "terminal-pair-that-is-not-the-reasoning": (
        {"content": "<think>other</think>", "reasoning_content": "why"},
        "<think>why</think><think>other</think>",
    ),
    # an explicit opener says the block is present; its body must then match the reasoning to be
    # the repeat. removing the opener left an empty prefix, read as the bare delimiter, and the
    # streaming path deleted the answer's own pair while the fold kept it.
    "answer-leading-empty-pair": (
        {"content": "<think></think>answer", "reasoning_content": "why"},
        "<think>why</think><think></think>answer",
    ),
    "empty-reasoning-duplicated-inline": (
        {"content": "<think></think>answer", "reasoning_content": ""},
        "<think></think>answer",
    ),
}


@pytest.mark.parametrize(("message", "expected"), PARITY_CASES.values(), ids=PARITY_CASES)
def test_both_paths_agree_on_the_same_payload(message, expected):
    assert _folded(message) == expected
    assert _streamed(*_split(message)) == expected


# --------------------------------------------------------------------------------------------
# the streaming path, on delta shapes the fold has no equivalent for.

STREAM_CASES: dict[str, tuple[tuple[dict, ...], str]] = {
    "reasoning-then-answer": (
        (
            {"reasoning_content": "weigh"},
            {"reasoning_content": "ing"},
            {"content": "the "},
            {"content": "answer"},
        ),
        "<think>weighing</think>the answer",
    ),
    "block-opens-once-across-reasoning-deltas": (
        ({"reasoning_content": "a"}, {"reasoning_content": "b"}, {"content": "z"}),
        "<think>ab</think>z",
    ),
    # no delimiter ever arrives, so nothing marked a reasoning phase. wrapping here would label a
    # whole valid answer as reasoning, and the smoke's answer split would find nothing behind the
    # close tag and reject a working deployment.
    "no-delimiter-is-not-wrapped": (({"content": "plain "}, {"content": "answer"}), "plain answer"),
    # generation hit the length cap inside the block: an unbalanced OPENER is the same defect as
    # the unbalanced closer, mirrored, so the stream must still close it.
    "reasoning-cut-off-mid-block": (
        ({"reasoning_content": "thinking hard"},),
        "<think>thinking hard</think>",
    ),
    # `""` means the block closed immediately, which the fold emits a pair for. reading falsiness
    # made the two paths disagree on the same payload.
    "empty-reasoning": (
        ({"reasoning_content": ""}, {"content": "answer"}),
        "<think></think>answer",
    ),
    # a compatibility build retains the sampled close at the head of the first content delta.
    # synthesising a second one left `<think>reasoned</think></think>answer`. the tag is tokenised,
    # so it straddles delta boundaries routinely, and its leading newline may be a delta of its own
    # -- an empty `strip()` failed the "could this still be the tag" test and released it as answer.
    "retained-close": (
        ({"reasoning_content": "reasoned"}, {"content": "</think>answer"}),
        "<think>reasoned</think>answer",
    ),
    "retained-close-split": (
        ({"reasoning_content": "reasoned"}, {"content": "</th"}, {"content": "ink>answer"}),
        "<think>reasoned</think>answer",
    ),
    "retained-close-behind-newline": (
        ({"reasoning_content": "reasoned"}, {"content": "\n</think>answer"}),
        "<think>reasoned</think>answer",
    ),
    "retained-close-after-newline-delta": (
        ({"reasoning_content": "reasoned"}, {"content": "\n"}, {"content": "</think>answer"}),
        "<think>reasoned</think>answer",
    ),
    "retained-close-after-newline-delta-split": (
        (
            {"reasoning_content": "reasoned"},
            {"content": "\n"},
            {"content": "</th"},
            {"content": "ink>answer"},
        ),
        "<think>reasoned</think>answer",
    ),
    "retained-close-after-newline-delta-split-twice": (
        (
            {"reasoning_content": "reasoned"},
            {"content": "\n"},
            {"content": "<"},
            {"content": "/think>"},
            {"content": "answer"},
        ),
        "<think>reasoned</think>answer",
    ),
    # holding for the answer must not turn into keeping a delimiter that really was retained.
    "retained-close-answer-arrives-late": (
        ({"reasoning_content": "why"}, {"content": "</think>"}, {"content": "answer"}),
        "<think>why</think>answer",
    ),
    # the buffered close belongs to the block that was open when it arrived. a later non-empty
    # reasoning delta opens a NEW block, and the stale buffer was never cleared -- so end of stream
    # flushed it behind the new block's close, an extra tag the fold never produces.
    "retained-close-then-later-reasoning": (
        ({"reasoning_content": "why"}, {"content": "</think>"}, {"reasoning_content": "more"}),
        "<think>why</think><think>more</think>",
    ),
    # the legacy delimiter is split across deltas too, so matching within one delta would miss it.
    "legacy-delimiter-split": (
        ({"content": "reasoned</th"}, {"content": "ink>answer"}),
        "<think>reasoned</think>answer",
    ),
    "legacy-close-with-answer-side-pair-split": (
        ({"content": "reasoned</think>answer <think>x"}, {"content": "</think> more"}),
        "<think>reasoned</think>answer <think>x</think> more",
    ),
    # a legacy stream carrying its own opener is already balanced; re-opening around its close
    # nested one block inside another on the public chat_stream path.
    "legacy-stream-already-balanced": (
        ({"content": "<think>reasoned</think>answer"},),
        "<think>reasoned</think>answer",
    ),
    # the reasoning on the field AND repeated inline ahead of the retained close. the streaming
    # helper recognised only whitespace before that close, so it returned the content whole and the
    # caller received the reasoning twice and the close twice.
    "duplicate-reasoning-inline": (
        ({"reasoning_content": "reasoned"}, {"content": "reasoned</think>answer"}),
        "<think>reasoned</think>answer",
    ),
    "duplicate-reasoning-split-body": (
        (
            {"reasoning_content": "reasoned"},
            {"content": "reas"},
            {"content": "oned</think>"},
            {"content": "answer"},
        ),
        "<think>reasoned</think>answer",
    ),
    "duplicate-reasoning-sampled-newline": (
        ({"reasoning_content": "reasoned"}, {"content": "reasoned\n</think>answer"}),
        "<think>reasoned</think>answer",
    ),
    "duplicate-reasoning-split-everywhere": (
        (
            {"reasoning_content": "reasoned"},
            {"content": "reasoned"},
            {"content": "\n"},
            {"content": "</think>"},
            {"content": "answer"},
        ),
        "<think>reasoned</think>answer",
    ),
    # the duplicate rule keys on the reasoning being repeated BEFORE the close, so an answer that
    # merely shares a prefix with it must arrive whole rather than be held for a delimiter that is
    # not coming.
    "answer-shares-a-prefix-with-the-reasoning": (
        ({"reasoning_content": "rea"}, {"content": "reactor design"}),
        "<think>rea</think>reactor design",
    ),
    # the repeat can carry its own opener, and is subject to the same delta boundaries.
    "duplicate-block-with-opener": (
        ({"reasoning_content": "why"}, {"content": "<think>why</think>answer"}),
        "<think>why</think>answer",
    ),
    "duplicate-block-with-opener-split": (
        ({"reasoning_content": "why"}, {"content": "<think>why<"}, {"content": "/think>a"}),
        "<think>why</think>a",
    ),
    "terminal-duplicate-split": (
        ({"reasoning_content": "why"}, {"content": "<think>why<"}, {"content": "/think>"}),
        "<think>why</think>",
    ),
    "terminal-duplicate-bare-split": (
        ({"reasoning_content": "why"}, {"content": "why<"}, {"content": "/think>"}),
        "<think>why</think>",
    ),
    # `_delimiter_may_complete` gave up on an empty body and released the head as answer, so the
    # pair streamed twice -- while the same bytes in ONE delta folded correctly.
    "empty-duplicate-pair-split": (
        ({"reasoning_content": ""}, {"content": "<think></"}, {"content": "think>answer"}),
        "<think></think>answer",
    ),
    # a backend whose stream schema serializes the empty field on every delta reaches the open
    # branch again after the first answer delta closed the block. the reasoning phase happens once,
    # so the block does too.
    "empty-reasoning-repeated-on-every-delta": (
        ({"reasoning_content": "", "content": "a"}, {"reasoning_content": "", "content": "b"}),
        "<think></think>ab",
    ),
    # that guard is scoped to the EMPTY field on purpose: reasoning arriving after the answer began
    # still carries text, and suppressing its tags would stream that text as answer.
    "late-nonempty-reasoning": (
        ({"reasoning_content": "why"}, {"content": "a"}, {"reasoning_content": "more"}),
        "<think>why</think>a<think>more</think>",
    ),
    # the retained close is recognised by comparing the buffer against the reasoning it follows, so
    # that reasoning must be the CURRENT block's. accumulating it across blocks left the second
    # block's repeat compared against both bodies joined, which matched neither, and the delimiter
    # streamed a second time -- two openers against three closes.
    "second-block-keeps-its-own-retained-close": (
        (
            {"reasoning_content": "a"},
            {"content": "x"},
            {"reasoning_content": "b"},
            {"content": "b</think>answer"},
        ),
        "<think>a</think>x<think>b</think>answer",
    ),
    "second-block-terminal-duplicate": (
        (
            {"reasoning_content": "a"},
            {"content": "x"},
            {"reasoning_content": "b"},
            {"content": "<think>b</think>"},
        ),
        "<think>a</think>x<think>b</think>",
    ),
}


@pytest.mark.parametrize(("deltas", "expected"), STREAM_CASES.values(), ids=STREAM_CASES)
def test_the_stream_rebuilds_a_balanced_block(deltas, expected):
    assert _streamed(*deltas) == expected


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (({"content": "just "}, {"content": "text"}), "just text"),
        # wrapping the field in synthetic tags for a request that never asked for a reasoning
        # phase is the defect the flag exists to prevent, which the streaming path kept doing.
        (({"reasoning_content": "reasoned"}, {"content": "answer"}), "answer"),
    ],
)
def test_a_non_thinking_stream_is_never_rewritten(deltas, expected):
    assert _streamed(*deltas, thinking=False) == expected


# --------------------------------------------------------------------------------------------
# delivery, which the joined-string cases above cannot see.


def test_a_non_thinking_stream_yields_each_delta_as_it_arrives():
    # no block can exist outside thinking mode, so nothing is ever held back: the deltas a caller
    # prints must arrive one at a time, not batched at the end.
    lines = _sse({"content": "a"}, {"content": "b"})
    assert list(serving_streaming._openai_stream_content(iter(lines), thinking=False)) == ["a", "b"]


def test_an_engine_error_raises_after_yielding_partial_content():
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'partial'}}]})}",
        "",
        f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'error'}], 'error': {'message': 'engine stream failed', 'type': 'engine_error', 'code': 500}})}",
        "",
    ]
    stream = serving_streaming._openai_stream_content(iter(lines), thinking=False)

    assert next(stream) == "partial"
    with pytest.raises(ClientError, match="engine stream failed"):
        next(stream)


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (
            [
                f"data: {json.dumps({'error': {'message': 'engine stream failed'}})}",
                "",
            ],
            "engine stream failed",
        ),
        (["data: {", ""], "invalid openai sse json"),
        ([], r"terminal \[DONE\]"),
    ],
    ids=["engine-error", "malformed-sse", "premature-eof"],
)
def test_a_reasoning_block_closes_before_stream_errors(tail: list[str], message: str) -> None:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'why'}}]})}",
        "",
        *tail,
    ]
    stream = deploy._openai_stream_content(iter(lines), thinking=True)

    assert next(stream) == "<think>"
    assert next(stream) == "why"
    assert next(stream) == "</think>"
    with pytest.raises(ClientError, match=message):
        next(stream)


def test_a_split_stream_releases_the_answer_incrementally():
    # once a reasoning delta proves the backend splits, the legacy hold is dropped and the answer
    # streams delta by delta rather than accumulating to the end.
    lines = _sse({"reasoning_content": "r"}, {"content": "an"}, {"content": "swer"})
    assert list(serving_streaming._openai_stream_content(iter(lines), thinking=True)) == [
        "<think>",
        "r",
        "</think>",
        "an",
        "swer",
    ]


def test_the_held_stream_does_not_rescan_its_whole_buffer_per_delta():
    """A legacy inline stream with no early close holds every delta, and used to rescan them all.

    Token-sized deltas made the search quadratic in the completion length, and the public chat route
    caps no `max_tokens`, so a long generation burned that time before streaming a byte.

    Asserted as characters scanned rather than wall-clock, which would be a flaky way to ask the same
    question. Counting `str.find` calls alone would not fail either -- the old code called it once
    per delta too. What changed is how much each call reads, so that is what is measured.
    """
    scanned = 0
    real_find_delimiter = serving_thinking._find_delimiter

    def _counting_find(buffer: str, start: int) -> int:
        # measured through the source's own named seam, because instrumenting the buffer is not
        # possible from here: the loop coerces each delta with `str()` before appending it, so a
        # `str` subclass never survives into the buffer being searched.
        nonlocal scanned
        scanned += max(0, len(buffer) - start)
        return real_find_delimiter(buffer, start)

    def _scan_for(deltas: int) -> int:
        nonlocal scanned
        scanned = 0
        lines = _sse(*({"content": "tok "} for _ in range(deltas)))
        out = "".join(
            serving_streaming._openai_stream_content(
                iter(lines), thinking=True, find_delimiter=_counting_find
            )
        )
        assert out == "tok " * deltas
        return scanned

    small = _scan_for(500)
    large = _scan_for(1000)
    # linear: twice the deltas scans about twice the characters. quadratic would be about four
    # times, so the midpoint separates them with room for the tail rescan on either side.
    assert large < small * 3, f"{small} -> {large} chars scanned looks quadratic"


def test_the_closing_buffer_does_not_rescan_itself_per_delta():
    """The post-reasoning buffer grows the same way the held one does, and rescanned itself too.

    A compatibility build repeats a long `reasoning_content` inline over token-sized deltas before
    sending the retained close, so the buffer the delimiter search runs on grows a token at a time
    with no answer released until the tag lands. Measured as the held twin is, and for the same
    reason.
    """
    scanned = 0
    real_find_delimiter = serving_thinking._find_delimiter

    def _counting_find(buffer: str, start: int) -> int:
        nonlocal scanned
        scanned += max(0, len(buffer) - start)
        return real_find_delimiter(buffer, start)

    def _scan_for(deltas: int) -> int:
        nonlocal scanned
        scanned = 0
        reasoning = "tok " * deltas
        stream = [{"reasoning_content": reasoning}]
        stream += [{"content": "tok "} for _ in range(deltas)]
        stream += [{"content": "</think>"}, {"content": "answer"}]
        out = "".join(
            serving_streaming._openai_stream_content(
                iter(_sse(*stream)), thinking=True, find_delimiter=_counting_find
            )
        )
        assert out == f"<think>{reasoning}</think>answer"
        return scanned

    small = _scan_for(500)
    large = _scan_for(1000)
    # the count must come from the buffer this test is about. without this the ratio below would
    # read an uninstrumented path as zero and zero, which no growth assertion can tell from linear.
    assert small > 0, "the closing buffer was never searched through the measured seam"
    assert large < small * 3, f"{small} -> {large} chars scanned looks quadratic"


# --------------------------------------------------------------------------------------------
# the payload rewrite, and the public chat route it runs on.


def test_payload_balancing_rewrites_every_choice_in_place():
    payload = {
        "choices": [
            {"message": {"content": "a", "reasoning_content": "r1"}},
            {"message": {"content": "b", "reasoning_content": "r2"}},
        ]
    }
    serving_thinking._balance_thinking_payload(payload, thinking=True)
    assert payload["choices"][0]["message"]["content"] == "<think>r1</think>a"
    assert payload["choices"][1]["message"]["content"] == "<think>r2</think>b"
    # the split fields survive for callers that want them.
    assert payload["choices"][0]["message"]["reasoning_content"] == "r1"


def test_payload_balancing_is_a_no_op_outside_thinking_mode():
    payload = {"choices": [{"message": {"content": "a</think>b", "reasoning_content": "r"}}]}
    serving_thinking._balance_thinking_payload(payload, thinking=False)
    assert payload["choices"][0]["message"]["content"] == "a</think>b"


def test_payload_balancing_tolerates_shapes_it_cannot_rewrite():
    for payload in (None, [], "text", {}, {"choices": None}, {"choices": [{}, {"message": None}]}):
        serving_thinking._balance_thinking_payload(payload, thinking=True)  # must not raise


def _stub_serving(monkeypatch, message: dict) -> None:
    class _Resp:
        headers: ClassVar[dict] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

    class _Client:
        def post(self, url, **kwargs):
            return _Resp()

    monkeypatch.setattr(
        serving_transport, "serving_openai_base_url", lambda: "https://serve.example/v1"
    )
    monkeypatch.setattr(serving_transport, "_internal_key_header", dict)
    monkeypatch.setattr(serving_transport, "_chat_http_client", lambda: _Client())


def test_non_streaming_chat_balances_before_returning(monkeypatch):
    _stub_serving(monkeypatch, {"content": "answer", "reasoning_content": "reasoned"})

    result = deploy.chat(
        "run-1/final",
        [{"role": "user", "content": "hi"}],
        org_id="org-1",
        thinking=True,
    )

    # the balanced string is what the deployment smoke greps for its thinking-tag telemetry.
    assert result["choices"][0]["message"]["content"] == "<think>reasoned</think>answer"


def test_non_thinking_chat_returns_serving_content_unchanged(monkeypatch):
    # `chat()` balanced every response regardless of the flag it was already given, so a
    # non-thinking answer quoting the tag came back rewritten into a reasoning block -- on the path
    # that also serves the public non-streaming chat route.
    _stub_serving(monkeypatch, {"content": "foo</think>bar"})

    result = deploy.chat("run-1/final", [{"role": "user", "content": "hi"}], org_id="org-1")

    assert result["choices"][0]["message"]["content"] == "foo</think>bar"


def test_non_thinking_chat_preserves_tool_only_null_content(monkeypatch) -> None:
    tool_calls = [{"type": "function", "function": {"name": "weather", "arguments": "{}"}}]
    _stub_serving(monkeypatch, {"content": None, "tool_calls": tool_calls})

    result = deploy.chat("run-1/final", [{"role": "user", "content": "hi"}], org_id="org-1")

    assert result["choices"][0]["message"] == {"content": None, "tool_calls": tool_calls}
