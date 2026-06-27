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

    - closed block(s): keep only the text after the LAST </think> (so a model that
      prematurely closes then RE-OPENS reasoning still grades on its final answer). This
      also covers always-thinking templates that pre-open <think> inside the generation
      prompt, whose completions contain </think> with no opening tag. NB this uses the LAST
      </think> by design, whereas think_token_count counts up to the FIRST — the two
      intentionally differ (answer extraction vs reasoning measurement).
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
    """Number of reasoning tokens in the completion's FIRST ``<think>`` span (0 if none).

    Drives the thinking-length reward deduction (mirrors the SDK's thinking_length_penalty_coef):
    long reasoning is penalized in proportion to the tokens it spent. A hybrid-thinking model
    reasons once then answers, so the FIRST span IS the whole reasoning — it is bounded at the
    FIRST ``</think>`` on purpose, so a later ``</think>`` echoed in the *answer* is never counted.
    (``strip_think`` deliberately uses the LAST ``</think>`` instead — see its docstring — because
    the two answer different questions: this counts the reasoning, that extracts the final answer.
    The first-vs-last difference is intentional; do not "unify" them.)

    ``prompt_opened_thinking`` (the rendered PROMPT itself pre-opened ``<think>`` — not merely that
    THINKING was set; see ``prompt_opens_thinking``) decides where the reasoning STARTS:
      - pre-opened: at the completion's first token (the opener lives in the prompt), so an
        unclosed, tag-less ramble that ran out of ``max_tokens`` is still all reasoning and is
        counted in full. A ``<think>`` the model echoes is content, NOT the opener — except one
        re-emitted at the very START (whitespace-only before it), which is skipped as the opener.
      - not pre-opened: at the model's OWN opening ``<think>`` (decided by tag ORDER — a
        close-before-open completion is prompt-style reasoning). A tag-less completion here is a
        plain (non-thinking) answer and counts 0 — never over-penalize an uncurated template.
    """
    if not completion:
        return 0
    open_idx = completion.find("<think>")
    close_idx = completion.find("</think>")
    # No tags at all and the prompt did not pre-open: a plain (non-thinking) answer — count nothing.
    if not prompt_opened_thinking and open_idx == -1 and close_idx == -1:
        return 0
    # The reasoning is the single span [start, end). END = the FIRST </think> (a later one is an
    # answer echo), or the whole completion when reasoning never closes (budget ran out). START =
    # just past the genuine OPENING <think>, else the completion's first token (prompt pre-opened).
    if prompt_opened_thinking:
        # Pre-opened: reasoning starts at the first token; only a <think> echoed at the very START
        # (whitespace-only before it) is the opener — a mid-reasoning echo is content, not a tag.
        opens_in_completion = open_idx != -1 and not completion[:open_idx].strip()
    else:
        # Not pre-opened: the model's own <think> is the opener only when it precedes the close
        # (tag ORDER, not mere presence — a close-before-open completion is prompt-style reasoning).
        opens_in_completion = open_idx != -1 and (close_idx == -1 or open_idx < close_idx)
    start = open_idx + len("<think>") if opens_in_completion else 0
    end = close_idx if close_idx != -1 else len(completion)
    think_text = completion[start:end] if end >= start else completion[start:]
    if not think_text:
        return 0
    return len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
