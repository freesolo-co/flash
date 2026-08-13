"""Pin the offline thinking-template fake against the real Qwen3.5 chat template.

``tests/test_sft_workload.py`` measures reasoning loss through ``ThinkingTokenizer``, a
transcription of Qwen3.5's template. A fake is a claim about the real template, and the whole
warning rests on that claim, so it is checked here against the shipped tokenizer rather than
trusted. Downloads a tokenizer, so it is opt-in like the other live tests.
"""

from __future__ import annotations

import os

import pytest

from flash.engine.profiling.workload_profile import (
    count_rendered_reasoning_spans,
    reasoned_assistant_turns,
    reasoning_marker_prefix,
    reasoning_span_texts,
    reasoning_spans,
    with_marked_reasoning,
)

pytestmark = pytest.mark.live

# spelled out rather than imported from the module under test: a test that borrowed the parser's
# own constant would still pass if that constant were wrong.
TURN_END = "<|im_end|>"

# the smallest catalog model that ships the thinking template; the template is shared across the
# Qwen3.5 family, so the 0.8B render is the same rule the 4B and 27B students apply.
MODEL = "Qwen/Qwen3.5-0.8B"

MULTITURN = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "board"},
    {"role": "assistant", "content": "<think>first</think>a1"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>second</think>a2"},
    {"role": "user", "content": "next"},
    {"role": "assistant", "content": "<think>third</think>a3"},
]


@pytest.fixture(scope="module")
def tokenizer():
    if not os.environ.get("FLASH_LIVE"):
        pytest.skip("set FLASH_LIVE=1 to download the tokenizer")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL)


def _render(tokenizer, messages, *, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def test_the_real_template_drops_all_but_the_final_turns_reasoning(tokenizer) -> None:
    """The measurement the warning reports, taken from the shipped template."""
    rendered = _render(tokenizer, MULTITURN)

    assert reasoned_assistant_turns(MULTITURN) == 3
    assert count_rendered_reasoning_spans(rendered) == 1
    # the reasoning that survived is the LAST turn's, not an arbitrary one
    assert "third" in rendered
    assert "first" not in rendered
    assert "second" not in rendered


def test_the_real_template_injects_an_empty_think_block_with_no_authored_reasoning(
    tokenizer,
) -> None:
    """Why survival counts non-empty spans: the raw tag is present either way.

    This is the input that loses ALL of its reasoning, and a raw ``count("<think>")`` would score
    it as one survivor -- the smallest apparent loss on the worst transcript.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>first</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    rendered = _render(tokenizer, messages)

    assert "<think>" in rendered
    assert count_rendered_reasoning_spans(rendered) == 0
    assert reasoned_assistant_turns(messages) == 1


def test_the_real_template_keeps_reasoning_authored_in_final_position(tokenizer) -> None:
    """The control: the shape the warning tells users to restructure into loses nothing."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "board"},
        {"role": "assistant", "content": "<think>only</think>a"},
    ]
    rendered = _render(tokenizer, messages)

    assert reasoned_assistant_turns(messages) == 1
    assert count_rendered_reasoning_spans(rendered) == 1
    assert "only" in rendered


def test_the_real_template_passes_a_prompt_think_span_through_verbatim(tokenizer) -> None:
    """A ``<think>`` span written into a PROMPT is content, and never a supervised reasoning block.

    The template's reasoning split is ASSISTANT-only, so a literal ``<think>...</think>`` written
    into a system prompt survives into the render verbatim while never being supervised. It is not
    emitted in the template's reasoning position, so it is not a block the template owns: were it
    counted, that span would offset reasoning stripped from the target and silence the warning.
    """
    prompt_messages = [
        {"role": "system", "content": "answer as <think>reasoning</think>answer"},
        {"role": "user", "content": "board"},
    ]
    prompt = _render(tokenizer, prompt_messages, add_generation_prompt=True)

    # the text survives verbatim, but it is prompt content, not a reasoning block
    assert "<think>reasoning</think>" in prompt
    assert count_rendered_reasoning_spans(prompt) == 0


def test_the_real_template_strips_a_prompt_turn_that_the_prompt_only_render_keeps(
    tokenizer,
) -> None:
    """Why the baseline is a re-render of the whole row, not the prompt render.

    The same assistant turn is trailing in the prompt-only render and interior in the full render,
    so its span exists in one and not the other. Subtracting the prompt's count would remove a span
    the full render never had, cancelling a target block that did survive.
    """
    prompt_messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>promptreason</think>a1"},
    ]
    completion_messages = [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<think>targetreason</think>a2"},
    ]
    prompt = _render(tokenizer, prompt_messages, add_generation_prompt=True)
    full = _render(tokenizer, [*prompt_messages, *completion_messages])

    # the prompt render keeps the prior turn's reasoning; the full render strips it
    assert "promptreason" in prompt
    assert count_rendered_reasoning_spans(prompt) == 1
    assert "promptreason" not in full
    # only the target's own reasoning survives, and prompt-count subtraction would erase it
    assert count_rendered_reasoning_spans(full) == 1
    assert "targetreason" in full


def test_the_real_template_renders_two_empty_blocks_for_two_bare_trailing_turns(tokenizer) -> None:
    """The input that makes a delimiter-crossing span pattern merge two empty blocks into one.

    Both blocks are empty, so the correct count is zero; a pattern that lets the first closing
    tag's ``<`` satisfy its non-whitespace requirement reports one survivor instead.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>first</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "assistant", "content": "a3"},
    ]
    rendered = _render(tokenizer, messages)

    assert rendered.count("<think>") == 2
    assert count_rendered_reasoning_spans(rendered) == 0
    assert reasoned_assistant_turns(messages) == 1


def test_the_marked_render_keeps_the_full_renders_span_sequence(tokenizer) -> None:
    """The property the survivor count rests on, checked against the shipped template.

    Marking changes reasoning TEXT only, so the template sees identical roles in identical
    positions and both renders carry the same spans in the same order. That is what lets the marked
    render say which turn owns each span while the real render says where that span ends; if
    marking could add, drop, or reorder a span the two would no longer address the same thing.
    """
    full = _render(tokenizer, MULTITURN)
    prefix = reasoning_marker_prefix(full)
    marked = _render(tokenizer, with_marked_reasoning(MULTITURN, prefix))

    assert count_rendered_reasoning_spans(marked) == count_rendered_reasoning_spans(full) == 1
    # the answers and turn structure are untouched: only reasoning text differs
    for answer in ("a1", "a2", "a3"):
        assert answer in marked
    assert marked.count("<|im_start|>") == full.count("<|im_start|>")
    # exactly the surviving turn is identified, and it is the LAST assistant turn (index 6) -- the
    # template strips the earlier two, so their markers never reach the render
    spans = reasoning_span_texts(marked)
    assert f"{prefix}6 " in spans[0]
    assert not any(f"{prefix}{index} " in spans[0] for index in (2, 4))


def test_a_quoted_think_span_never_carries_a_marker(tokenizer) -> None:
    """A ``<think>`` the answer merely QUOTES is answer text, and is not a reasoning span at all.

    This turn supplies its reasoning through ``reasoning_content`` and quotes the format in its
    answer. The template renders the quote verbatim into the ANSWER, after the closer and the blank
    line that end the reasoning block, so only the field's reasoning is a block the template owns.
    Counting the quote would credit a survivor that is not reasoning, overstating what reaches the
    loss and suppressing the very drop warning this measurement exists to raise.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "answer shows <think>example</think> format",
            "reasoning_content": "real reasoning",
        },
    ]
    full = _render(tokenizer, messages)
    # the full render carries BOTH the field's reasoning and the answer's quote, verbatim
    assert "real reasoning" in full
    assert "<think>example</think>" in full
    # ... but only the first is structurally a reasoning block
    assert count_rendered_reasoning_spans(full) == 1

    prefix = reasoning_marker_prefix(full)
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))
    spans = reasoning_span_texts(marked)

    assert count_rendered_reasoning_spans(marked) == 1
    # the one span is the reasoning, and it carries the marker; the quote is answer text
    assert [prefix in span for span in spans] == [True]
    assert "example" not in spans[0]


def test_reasoning_quoting_an_assistant_header_is_found_whole(tokenizer) -> None:
    """Reasoning that quotes a turn header must not cut its own span short.

    The header characters are ordinary text inside a block, so a scan that anchors on the header
    alone finds a second, spurious turn start mid-block and reports the surrounding real block as
    missing entirely. That is the opposite of a spurious extra span: the block silently vanishes
    from the survival count and the warning fires on a run that lost nothing. Only the template can
    place a header after the previous turn's terminator, which is what separates the two.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "a1",
            "reasoning_content": "a turn starts with <|im_start|>assistant\n<think>\n like so",
        },
    ]
    full = _render(tokenizer, messages)
    # the quote really is in the render, character for character
    assert "<|im_start|>assistant\n<think>\n like so" in full
    # and the turn is still ONE block, found whole rather than cut at the quote
    assert count_rendered_reasoning_spans(full) == 1

    prefix = reasoning_marker_prefix(full)
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))
    spans = reasoning_span_texts(marked)

    assert [prefix in span for span in spans] == [True]
    assert "like so" in spans[0]


def test_a_reasoning_span_stops_at_its_own_turn_not_a_later_tool_message(tokenizer) -> None:
    """A block ends at its own turn even when no LATER turn opens reasoning of its own.

    A tool response does not reset ``last_query_index``, so the assistant before it keeps its
    reasoning -- and if the scan only stops at the next REASONING turn, a trailing tool message
    leaves it unbounded. The span then runs through the answer, the turn terminator, and into the
    tool output, where a quoted closer ends it. A context cap between the real closer and that
    quoted one reports intact reasoning as truncated.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "REAL"},
        # tool output that happens to contain the template's own closer layout
        {"role": "user", "content": "<tool_response>before\n</think>\n\nafter</tool_response>"},
    ]
    full = _render(tokenizer, messages)
    spans = reasoning_spans(full)

    assert len(spans) == 1
    body = full[spans[0][0] : spans[0][1]]
    assert "REAL" in body
    # the block stops at its own closer: it does not reach the answer, the terminator, or the tool
    assert body == "<think>\nREAL\n</think>\n\n"
    assert "after" not in body
    assert TURN_END not in body


def test_the_fake_renders_a_tool_role_message_the_way_the_template_does(tokenizer) -> None:
    """The offline fake must agree with the template on the tool shape, not just on span counts.

    A ``role="tool"`` message does not render as a ``tool`` header: the template wraps it in a USER
    turn carrying ``<tool_response>``. That is precisely why a tool turn does not reset
    ``last_query_index`` and the assistant before it keeps its reasoning. A fake emitting a header
    the template never produces can agree on counts by luck while disagreeing on the layout the
    span rules actually read, and the offline suite validates the warning against that fake.
    """
    from tests.test_sft_workload import ThinkingTokenizer

    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "R1"},
        {"role": "tool", "content": "TOOLOUT"},
        {"role": "assistant", "content": "a2", "reasoning_content": "R2"},
    ]
    real = _render(tokenizer, messages)
    fake = ThinkingTokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )

    # byte-identical, not merely span-equivalent
    assert fake == real
    # and the shape is the wrapped user turn, with no tool header anywhere
    assert "<|im_start|>user\n<tool_response>" in real
    assert "<|im_start|>tool" not in real


def test_a_quoted_terminator_does_not_lose_the_block_when_a_turn_follows(tokenizer) -> None:
    """A quoted ``<|im_end|>`` must not end the scan when a REAL turn follows.

    The horizon match consumes this turn's terminator, so its start IS that terminator. Searching
    for a terminator strictly below the horizon therefore cannot see the real one and finds only
    the quoted one, which sits in front of the block's closer -- so the block is dropped entirely
    and an intact survivor reports as lost. The turn that follows is what makes this reachable: with
    nothing after it, the backwards search from the end finds the real terminator anyway.
    """
    messages = [
        {"role": "user", "content": "u1"},
        # quotes the terminator, and is NOT the last turn
        {"role": "assistant", "content": "a1", "reasoning_content": "mentions <|im_end|> inline"},
        {"role": "assistant", "content": "a2", "reasoning_content": "SECOND"},
    ]
    full = _render(tokenizer, messages)
    spans = reasoning_span_texts(full)

    # both turns follow the last user message, so the template keeps both blocks
    assert len(spans) == 2
    assert "mentions <|im_end|> inline" in spans[0]
    assert "SECOND" in spans[1]


def test_reasoning_the_scan_cannot_resolve_is_not_reported_as_a_template_drop(tokenizer) -> None:
    """Reasoning the template KEPT must never be reported as reasoning the template dropped.

    Reasoning containing a verbatim ``<|im_end|>`` newline plus a header is byte-identical to a real
    turn boundary, so no rule reading the render alone can resolve the block. That is a limit of the
    scan, not a loss: the template kept the reasoning. The distinction matters outwardly, because
    the drop warning tells the user to split their transcript into single-turn rows -- advice that
    is simply wrong for a row whose reasoning survived, and that no rerun would talk them out of.

    So the contract asserted here is the outward one: the row goes unreported rather than reported
    wrongly. A future scan that resolves this shape should make the assertion below stronger, not
    fail it.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "a1",
            # the control-token layout written out in full, not merely mentioned
            "reasoning_content": "see <|im_end|>\n<|im_start|>user\n then more",
        },
    ]
    full = _render(tokenizer, messages)

    # the template kept the reasoning verbatim: this row lost nothing
    assert "see <|im_end|>" in full
    # so whatever the scan resolves, it must not claim the block was dropped
    spans = reasoning_span_texts(full)
    assert spans == [] or any("then more" in span for span in spans)


def test_the_real_template_prefers_reasoning_content_over_an_inline_span(tokenizer) -> None:
    """Pins the ``reasoning_content`` branch of ``reasoned_assistant_turns`` to the template."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "hidden"},
    ]
    rendered = _render(tokenizer, messages)

    assert reasoned_assistant_turns(messages) == 1
    assert count_rendered_reasoning_spans(rendered) == 1
    assert "hidden" in rendered
