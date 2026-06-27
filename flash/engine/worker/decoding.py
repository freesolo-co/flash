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

    A hybrid template pre-opens reasoning by appending ``<think>`` as the TRAILING generation-prompt
    suffix, so we require the rendered prompt to END with ``<think>`` (ignoring trailing whitespace).
    Anchoring on the suffix — not a scan of the whole prompt — avoids a false positive when an
    earlier user/system/example message merely *contains* an unclosed literal ``<think>`` that the
    template never turned into an assistant prefill.
    """
    if not prompt:
        return False
    # ``"</think>".endswith("<think>")`` is False (the chars before ``think>`` differ), so a prompt
    # ending in a CLOSED block is correctly not treated as pre-opened.
    return prompt.rstrip().endswith("<think>")


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
    - unclosed under a prompt-opened <think> (``prompt_opened_thinking``, no </think> anywhere): the
      generation ran out of budget before closing, so the WHOLE completion is unterminated reasoning
      — even if it redundantly echoed another <think> inside it — so return "" and the env grades
      nothing (scores 0), the same pressure as an unclosed in-band <think>. Without this, an env with
      a raw-text answer fallback could reward the reasoning ramble. Gated on the caller confirming the
      prompt actually pre-opened the tag (not merely that THINKING was set), so a non-thinking
      template's normal answer is left untouched.
    - no tags (and no prompt pre-open): unchanged.
    """
    if completion is None:
        return None
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    # No </think>: reasoning never closed. A prompt-opened block means the whole completion is
    # unterminated reasoning (incl. any echoed <think>), so hide it BEFORE the model-opened branch —
    # else an echoed <think> would leak the pre-think text to the env.
    if prompt_opened_thinking:
        return ""
    if "<think>" in completion:
        return completion.split("<think>", 1)[0]
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
    When ``prompt_opened_thinking`` is set the reasoning ALWAYS begins at the completion's first
    token (the prompt pre-opened ``<think>``), so cases 2/3 span from the start to the first
    ``</think>`` (or the whole completion if it never closes) — INCLUDING any ``<think>`` the model
    redundantly echoed *inside* the reasoning. The lone exception is a ``<think>`` re-emitted at the
    very START (only whitespace before it): that leading tag is the opener, not content, so it's
    skipped. Anchoring on a mid-reasoning echoed opener (e.g. ``reason 42 <think> more </think> ans``)
    would wrongly count only the post-echo sliver instead of the full pre-opened span.
    When ``prompt_opened_thinking`` is False, case 1 vs 2 is decided by tag ORDER, not mere presence:
    a completion that closes its reasoning and then echoes a literal/malformed ``<think>`` in the
    answer must still count the span up to the FIRST ``</think>`` (case 2) — anchoring on the echoed
    opener would count the wrong span. Any later ``<think>`` blocks (a malformed re-open) are NOT
    added to the count.
    """
    if not completion:
        return 0
    open_idx = completion.find("<think>")
    close_idx = completion.find("</think>")
    if prompt_opened_thinking:
        # cases 2 & 3, prompt pre-opened: reasoning starts at the completion's FIRST token. Count from
        # there to the first </think> (or the whole completion when it never closes — budget ran out),
        # INCLUDING any <think> the model echoed mid-reasoning. The one exception: a <think> re-emitted
        # at the very START is the opener, not content — skip past it so its tokens aren't counted.
        start = (
            open_idx + len("<think>")
            if open_idx != -1 and not completion[:open_idx].strip()
            else 0
        )
        think_text = (
            completion[start:close_idx]
            if close_idx != -1 and close_idx >= start
            else completion[start:]
        )
    elif close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
        # case 2 (flag not passed): a hybrid template pre-opened <think> in the PROMPT, so the
        # completion starts mid-reasoning and carries only the close. Count up to the FIRST </think>;
        # a later literal <think> in the answer is NOT the opener (tag order, not presence).
        think_text = completion[:close_idx]
    elif open_idx != -1 and close_idx != -1:
        # case 1: the model emitted its OWN opening <think> before the close — count between them.
        think_text = completion[open_idx + len("<think>") : close_idx]
    elif open_idx != -1:
        # the model opened <think>, never closed it, and the prompt did NOT pre-open — count after the
        # opener (the unclosed model-emitted span).
        think_text = completion[open_idx + len("<think>") :]
    else:
        return 0
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
