"""Decoding parity helpers: prompt rendering + <think> handling."""

from __future__ import annotations

from flash.engine.worker._pkg import W as _w


def render_prompt(tokenizer, item) -> str:
    item = item if isinstance(item, dict) else {"question": item}
    msgs = _w.require_active_env().prompt_messages(item)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
    )


def prompt_opens_thinking(prompt: str | None) -> bool:
    """True when the rendered prompt ends with an unclosed <think> — derived from the prompt itself,
    NOT the enable_thinking flag (an uncurated template may ignore it)."""
    if not prompt:
        return False
    return prompt.rstrip().endswith("<think>")


def strip_think(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """Drop <think> reasoning spans before grading. Uses LAST </think> (answer extraction);
    unclosed reasoning returns "" to score 0 (reward pressure to finish within budget)."""
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    # No </think>: check prompt-opened before model-opened — else an echoed <think> leaks pre-think text.
    if prompt_opened_thinking:
        return ""
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
    return completion


def graded_text(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """What the env grader sees: strips <think> blocks when THINKING is on."""
    return (
        strip_think(completion, prompt_opened_thinking=prompt_opened_thinking)
        if _w.THINKING
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
