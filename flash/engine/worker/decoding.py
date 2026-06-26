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


def prompt_opens_thinking(prompt: str | None) -> bool:
    """True when the rendered PROMPT pre-opens a <think> block it never closes — i.e. the model's
    completion is expected to start mid-reasoning (hybrid-thinking templates append ``<think>`` after
    the generation prompt when ``enable_thinking=true``).

    Derived from the rendered prompt itself, NOT from the enable_thinking flag: an uncurated model's
    template (``thinking="unknown"``) may ignore that flag and never pre-open the tag, in which case a
    tagless completion is a normal answer — it must NOT be treated, penalized, or stripped as
    unterminated reasoning. Callers pass the result as ``prompt_opened_thinking`` to
    ``think_token_count`` / ``graded_text`` / ``strip_think``.
    """
    if not prompt:
        return False
    open_idx = prompt.rfind("<think>")
    if open_idx == -1:
        return False
    # ``<think>`` is not a substring of ``</think>``, so rfind finds only real opening tags. The last
    # opener pre-opens reasoning iff no ``</think>`` follows it.
    return prompt.rfind("</think>") < open_idx


def strip_think(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """Drop <think>...</think> reasoning before the environment grades/rewards a
    thinking-mode completion.

    - closed block(s): keep only the text after the LAST </think>. This also covers
      always-thinking templates that pre-open <think> inside the generation prompt,
      whose completions contain </think> with no opening tag.
    - unclosed <think> (completion budget exhausted): keep only the pre-think text
      (usually empty), so answer extraction fails and the completion scores 0 —
      deliberate reward pressure to close thinking within budget, and it keeps a
      last-number fallback from matching numbers inside the reasoning.
    - tagless completion under a prompt-opened <think> (``prompt_opened_thinking``): the
      generation ran out of budget before emitting EITHER tag, so the whole completion is
      unterminated reasoning — return "" so the env grades nothing (scores 0), the same
      pressure as an unclosed in-band <think>. Without this, an env with a raw-text answer
      fallback could reward the reasoning ramble. Gated on the caller confirming the prompt
      actually pre-opened the tag (not merely that THINKING was set), so a non-thinking
      template's normal answer is left untouched.
    - no tags (and no prompt pre-open): unchanged.
    """
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
    if prompt_opened_thinking:
        return ""
    return completion


def graded_text(completion: str | None, *, prompt_opened_thinking: bool = False) -> str | None:
    """What the env grader/reward sees: thinking runs strip <think> blocks first (a
    completion whose reasoning never closes grades 0 — see strip_think). Applied once
    here, before ACTIVE_ENV.grade/reward, so it works for every environment.

    ``prompt_opened_thinking`` (whether the rendered prompt actually pre-opened <think>) is
    forwarded to strip_think so a TAGLESS unclosed completion is hidden from the env too — not
    just penalized in think_token_count — closing the raw-text-fallback reward leak."""
    return (
        strip_think(completion, prompt_opened_thinking=prompt_opened_thinking)
        if _w.THINKING
        else completion
    )


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
    Case 1 vs 2 is decided by tag ORDER, not mere presence: a prompt-opened completion that closes
    its reasoning and then echoes a literal/malformed ``<think>`` in the answer must still count the
    span up to the FIRST ``</think>`` (case 2) — anchoring on the echoed opener would count the wrong
    span. Any later ``<think>`` blocks (uncommon — a malformed re-open) are NOT added to the count.
    """
    if not completion:
        return 0
    open_idx = completion.find("<think>")
    close_idx = completion.find("</think>")
    if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
        # case 1: the model emitted its OWN opening <think> before any close — count that span.
        after = completion[open_idx + len("<think>") :]
        think_text = after.split("</think>", 1)[0] if "</think>" in after else after
    elif close_idx != -1:
        # case 2: prompt-opened <think> — the completion starts mid-reasoning and only carries the
        # close. Count up to the FIRST </think>; a later literal <think> in the answer is NOT the
        # opener (tag order, not presence).
        think_text = completion[:close_idx]
    elif prompt_opened_thinking:  # case 3: prompt opened <think>, never closed (ran out of tokens)
        think_text = completion
    else:
        return 0
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
