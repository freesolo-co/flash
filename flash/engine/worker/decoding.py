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


def think_token_count(
    completion: str | None, tokenizer, *, prompt_opened_thinking: bool = False
) -> int:
    """Number of reasoning tokens in the completion's FIRST reasoning span (0 if none).

    Used for the thinking-length reward deduction: long reasoning is penalized in
    proportion to the tokens it spent, mirroring the SDK's thinking_length_penalty_coef.

    Counts the FIRST reasoning span only — which is the whole reasoning for a hybrid-thinking
    model (it reasons once, then answers). Handles the ways such a model surfaces that span:
      1. Self-contained block — the completion holds the whole ``<think>...</think>`` span
         (the model opened and closed the tag itself); counted up to the first ``</think>``.
      2. Prompt-opened block — the chat template appended ``<think>\\n`` to the *prompt*
         (Qwen3.5 / MiniCPM hybrid thinking with ``enable_thinking=true``), so the
         completion starts mid-reasoning and only carries the closing ``</think>``. The
         reasoning is then everything before the first ``</think>``.
      3. Prompt-opened but UNCLOSED — same prompt pre-open as (2), but the completion ran out of
         ``max_tokens`` before ever emitting ``</think>``, so it carries NEITHER tag. Only reachable
         when ``prompt_opened_thinking`` is set (the caller knows thinking was enabled): the WHOLE
         completion is unterminated reasoning. Without this the LONGEST prompt-opened rambles — the
         exact case the penalty targets — would score 0. When ``prompt_opened_thinking`` is False a
         tag-less completion is plain (non-thinking) text and counts 0.
    Without case 2 the penalty silently no-ops for the common enable_thinking=true path.
    Any later ``<think>`` blocks (uncommon — a malformed re-open) are NOT added to the count.
    """
    if not completion:
        return 0
    if "<think>" in completion:  # case 1: model emitted the opening tag
        after = completion.split("<think>", 1)[1]
        think_text = after.split("</think>", 1)[0] if "</think>" in after else after
    elif "</think>" in completion:  # case 2: prompt opened <think>; count up to the close
        think_text = completion.split("</think>", 1)[0]
    elif prompt_opened_thinking:  # case 3: prompt opened <think>, never closed (ran out of tokens)
        think_text = completion
    else:
        return 0
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
