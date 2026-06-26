"""Decoding parity helpers: prompt rendering + <think> handling.

Render with the model's own chat template and one run-wide thinking flag (off by default),
so SFT targets and RL rollouts use identical prompt formatting within a run. Run-scoped state
(``THINKING`` and the active env) is read through the worker package at CALL time so a test's
``monkeypatch.setattr(worker, "THINKING"/"JOB_SPEC", ...)`` reaches these readers.
"""

from __future__ import annotations

from flash.engine.worker._pkg import W as _w


def render_prompt(tokenizer, item) -> str:
    item = item if isinstance(item, dict) else {"question": item}
    msgs = _w.require_active_env().prompt_messages(item)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
    )


def strip_think(completion: str | None) -> str | None:
    """Drop <think>...</think> reasoning before the environment grades/rewards a
    thinking-mode completion.

    - closed block(s): keep only the text after the LAST </think>. This also covers
      always-thinking templates that pre-open <think> inside the generation prompt,
      whose completions contain </think> with no opening tag.
    - unclosed <think> (completion budget exhausted): keep only the pre-think text
      (usually empty), so answer extraction fails and the completion scores 0 —
      deliberate reward pressure to close thinking within budget, and it keeps a
      last-number fallback from matching numbers inside the reasoning.
    - no tags: unchanged.
    """
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
    return completion


def graded_text(completion: str | None) -> str | None:
    """What the env grader/reward sees: thinking runs strip <think> blocks first (a
    completion whose reasoning never closes grades 0 — see strip_think). Applied once
    here, before ACTIVE_ENV.grade/reward, so it works for every environment."""
    return strip_think(completion) if _w.THINKING else completion


def think_token_count(completion: str | None, tokenizer) -> int:
    """Number of tokens inside the completion's <think>...</think> span (0 if none).

    Used for the thinking-length reward deduction: long reasoning is penalized in
    proportion to the tokens it spent, mirroring the SDK's thinking_length_penalty_coef.
    """
    if not completion or "<think>" not in completion:
        return 0
    after = completion.split("<think>", 1)[1]
    think_text = after.split("</think>", 1)[0] if "</think>" in after else after
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
