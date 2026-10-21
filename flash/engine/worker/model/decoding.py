"""Decoding parity helpers: prompt rendering + <think> handling."""

from __future__ import annotations

import flash.engine.worker.runtime.state as _worker_state

# the pure parsers live in flash.content.thinking so the multi-turn grading path (flash.envs.adapter, which
# must stay importable without torch) reaches the same implementation. re-exported here because this
# module is the worker's public decoding surface.
from flash.content.thinking import strip_think as strip_think
from flash.content.thinking import thinking_text as thinking_text


def prompt_opens_thinking(prompt: str | None) -> bool:
    """True when the rendered prompt ends with an unclosed <think> — derived from the prompt itself,
    NOT the enable_thinking flag (an uncurated template may ignore it)."""
    if not prompt:
        return False
    return prompt.rstrip().endswith("<think>")


def graded_text(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """Answer text extracted for grading; reward state may still carry the raw completion."""
    return (
        strip_think(completion, prompt_opened_thinking=prompt_opened_thinking)
        if _worker_state.THINKING
        else completion
    )


def think_token_count(
    completion: str | None, tokenizer, *, prompt_opened_thinking: bool = False
) -> int:
    """Tokens in the FIRST <think> span (for reward deduction). Uses FIRST </think>, unlike
    strip_think which uses LAST — intentional: this measures reasoning, that extracts the answer."""
    if not completion:
        return 0
    open_idx = completion.find("<think>")
    close_idx = completion.find("</think>")
    if not prompt_opened_thinking and open_idx == -1 and close_idx == -1:
        return 0
    if prompt_opened_thinking:
        # A <think> echoed at the very start (whitespace-only before it) is the opener, not content.
        opens_in_completion = open_idx != -1 and not completion[:open_idx].strip()
    else:
        opens_in_completion = open_idx != -1 and (close_idx == -1 or open_idx < close_idx)
    start = open_idx + len("<think>") if opens_in_completion else 0
    end = close_idx if close_idx != -1 else len(completion)
    think_text = completion[start:end] if end >= start else completion[start:]
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
