"""Pin the offline thinking-template fake against the immutable Qwen3.8 chat template.

``tests/test_sft_workload.py`` measures reasoning loss through ``ThinkingTokenizer``, a
transcription of Flash's prompt contract. A fake is a claim about the real template and the warning
rests on it, so it is checked against the shipped tokenizer rather than trusted. Downloads a
tokenizer, so it is opt-in like the other live tests.

Survival is measured the way the profiler measures it -- mark the reasoning, render, ask which
markers arrived -- so these exercise the real decision procedure, not a paraphrase of it.
"""

from __future__ import annotations

import os

import pytest

from flash.content.thinking import messages_for_chat_template
from flash.engine.profiling.workload_profile import (
    marked_reasoning_end,
    reasoned_assistant_turns,
    reasoning_marker_prefix,
    reasoning_markers,
    strip_reasoning_markers,
    with_marked_reasoning,
)

pytestmark = pytest.mark.live

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

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
    return transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def _render(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages_for_chat_template(messages),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=False,
    )


def _survivors(tokenizer, messages):
    """Which turns' reasoning the REAL template keeps, by the profiler's own marker procedure."""
    prefix = reasoning_marker_prefix(_render(tokenizer, messages))
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))
    markers = reasoning_markers(messages, prefix)
    return [marked_reasoning_end(marked, marker) is not None for marker in markers]


def test_the_real_template_drops_all_but_the_final_turns_reasoning(tokenizer) -> None:
    """The measurement the warning reports, taken from the shipped template."""
    rendered = _render(tokenizer, MULTITURN)

    assert reasoned_assistant_turns(MULTITURN) == 3
    assert _survivors(tokenizer, MULTITURN) == [False, False, True]
    # the reasoning that survived is the LAST turn's, not an arbitrary one
    assert "third" in rendered
    assert "first" not in rendered
    assert "second" not in rendered


def test_the_real_template_injects_an_empty_think_block_with_no_authored_reasoning(
    tokenizer,
) -> None:
    """Why survival is asked of a marker: the raw tag is present either way. This input loses ALL of
    its reasoning, and a raw ``count("<think>")`` would score it as one survivor -- the smallest
    apparent loss on the worst transcript. No marker arrives, so the marker procedure reports the
    total loss.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>first</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    assert "<think>" in _render(tokenizer, messages)
    assert reasoned_assistant_turns(messages) == 1
    assert _survivors(tokenizer, messages) == [False]


def test_the_real_template_keeps_reasoning_authored_in_final_position(tokenizer) -> None:
    """The control: the shape the warning tells users to restructure into loses nothing."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "board"},
        {"role": "assistant", "content": "<think>only</think>a"},
    ]

    assert reasoned_assistant_turns(messages) == 1
    assert _survivors(tokenizer, messages) == [True]
    assert "only" in _render(tokenizer, messages)


def test_a_think_span_quoted_by_an_answer_never_carries_a_marker(tokenizer) -> None:
    """A ``<think>`` the ANSWER quotes renders a real block, and must never be credited. The quoting
    turn is stripped, the final turn is kept, and any procedure counting rendered tags reports no
    loss.

    The quote sits in the turn that KEEPS its reasoning, because a stripped turn is split at the
    last ``<think>`` and the quote would be discarded along with the reasoning -- leaving nothing
    for a counting procedure to miscredit, and so nothing for this test to catch.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>early</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "reasoning_content": "late", "content": "a <think>tag</think> here"},
    ]
    prefix = reasoning_marker_prefix(_render(tokenizer, messages))
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))

    assert reasoned_assistant_turns(messages) == 2
    assert _survivors(tokenizer, messages) == [False, True]
    # the quoted span is in the render, and carries no marker: the marker is in the real reasoning
    assert "a <think>tag</think> here" in marked
    assert f"late{prefix}" in marked


def test_reasoning_writing_out_the_turn_layout_is_still_credited(tokenizer) -> None:
    """Control tokens inside reasoning are content, and the real template keeps the turn. The bytes
    are identical to a turn boundary, so nothing reading the rendered text alone can tell them
    apart. The marker can, because it rides inside whatever the template kept.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "reasoning_content": "before <|im_end|>\n<|im_start|>user\n after",
            "content": "ans",
        },
    ]

    assert reasoned_assistant_turns(messages) == 1
    assert _survivors(tokenizer, messages) == [True]


def test_consecutive_trailing_assistant_turns_both_keep_their_reasoning(tokenizer) -> None:
    """The rule is ``loop.index0 > ns.last_query_index``, not "the final turn". A fake paraphrasing
    it as "the last message" would keep one block where the template keeps two.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>first</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<think>second</think>a2"},
        {"role": "assistant", "content": "<think>third</think>a3"},
    ]

    assert _survivors(tokenizer, messages) == [False, True, True]


def test_a_tool_response_turn_does_not_reset_the_reasoning_window(tokenizer) -> None:
    """A ``tool`` message renders as a ``<tool_response>`` user turn and does NOT end the window. A
    fake emitting an ``<|im_start|>tool`` header would render a turn the template never produces and
    could agree by accident while disagreeing on the rule.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>first</think>a1"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "<think>second</think>a2"},
    ]

    assert _survivors(tokenizer, messages) == [True, True]


def test_the_real_template_prefers_reasoning_content_over_an_inline_span(tokenizer) -> None:
    """The field wins, and ``content`` stays whole as the answer. A fake that always split
    ``content`` at a ``<think>`` tag would tear an answer apart at a tag it merely quotes.
    """
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "reasoning_content": "field", "content": "the <think>tag</think> x"},
    ]
    rendered = _render(tokenizer, messages)

    assert reasoned_assistant_turns(messages) == 1
    assert _survivors(tokenizer, messages) == [True]
    # the answer survives whole, quoted tag included
    assert "the <think>tag</think> x" in rendered


def test_a_marked_render_differs_from_the_real_one_only_by_its_markers(tokenizer) -> None:
    """The measurement must not perturb what it measures. Only reasoning TEXT changes, so the
    template sees identical roles in identical positions. Stripping the markers back out must return
    the real render byte for byte -- otherwise the token length charged against the cap is the
    MARKED render's, and truncation is judged against a row training never sees.
    """
    for messages in (
        MULTITURN,
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "<think>r</think>a"}],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "reasoning_content": "r", "content": "a"},
        ],
        # the template renders the reasoning through `|trim`. a marker appended AFTER trailing
        # whitespace shields that run from the trim, so the marked render keeps bytes the real one
        # drops and tokenizes longer through the closing tag -- a cap at the real block's end then
        # reports a block that fits as cut. these are the shapes that catch it.
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "reasoning_content": "r" + " " * 64, "content": "a"},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "reasoning_content": "r\n", "content": "a"},
        ],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "<think>r" + " " * 8 + "</think>a"},
        ],
    ):
        full = _render(tokenizer, messages)
        prefix = reasoning_marker_prefix(full)
        marked = _render(tokenizer, with_marked_reasoning(messages, prefix))

        assert strip_reasoning_markers(marked, prefix) == full


def test_a_blocks_end_offset_lands_past_its_own_closing_tag(tokenizer) -> None:
    """Truncation is judged against an offset running through the block's closer, and no further.
    Measuring to the blank line before the answer would call a block whose every reasoning token was
    retained truncated whenever the cap lands just past the closing tag.
    """
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "<think>reasoned</think>the answer"},
    ]
    prefix = reasoning_marker_prefix(_render(tokenizer, messages))
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))
    (marker,) = reasoning_markers(messages, prefix)

    end = marked_reasoning_end(marked, marker)
    assert end is not None
    through = strip_reasoning_markers(marked[:end], prefix)
    assert through.endswith("</think>")
    # the answer is past the boundary, so it is not charged against the reasoning's own budget
    assert "the answer" not in through


def test_a_closer_quoted_inside_reasoning_does_not_bound_the_block_short(tokenizer) -> None:
    """The closing tag is found FORWARD of the marker, so quoted closers lie behind it. A block
    bounded at a quoted closer ends too early, and a cap between that closer and the real one scores
    a block training cuts as fully retained -- the direction that hides the loss.
    """
    messages = [
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "reasoning_content": "start\n</think>\n\n middle <|im_end|>\n<|im_start|>user\n end",
            "content": "ANSWER",
        },
    ]
    prefix = reasoning_marker_prefix(_render(tokenizer, messages))
    marked = _render(tokenizer, with_marked_reasoning(messages, prefix))
    (marker,) = reasoning_markers(messages, prefix)

    end = marked_reasoning_end(marked, marker)
    assert end is not None
    through = strip_reasoning_markers(marked[:end], prefix)
    # every quoted fragment is inside the block, so the whole of it is charged against the cap
    assert "start" in through
    assert "middle" in through
    assert "end" in through
    assert "ANSWER" not in through
