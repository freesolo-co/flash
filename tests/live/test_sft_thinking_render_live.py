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
)

pytestmark = pytest.mark.live

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
    """Why rendered spans are counted net of the prompt render.

    The template's reasoning split is ASSISTANT-only, so a literal ``<think>...</think>`` written
    into a system prompt survives into the render while never being supervised. Counting the full
    render would let that span offset reasoning stripped from the target and silence the warning.
    """
    prompt_messages = [
        {"role": "system", "content": "answer as <think>reasoning</think>answer"},
        {"role": "user", "content": "board"},
    ]
    prompt = _render(tokenizer, prompt_messages, add_generation_prompt=True)

    assert count_rendered_reasoning_spans(prompt) == 1
    assert "<think>reasoning</think>" in prompt


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
