"""OPD run-setup helpers: thinking-mode prefill, student LoRA construction, in-loop progress pings,
and no-signal skip formatting. Extracted from ``opd`` (which re-imports them so they stay importable
from it) to keep ``run_opd`` focused on the training loop."""

from __future__ import annotations

import contextlib
from collections import Counter

from flash.engine.worker._pkg import W as _w


def _thinking_prefill_text(tok) -> str:
    """The trailing text a thinking-mode chat template opens after the generation prompt (Qwen's
    ``<think>\\n``), i.e. the delta between the enable_thinking=True and =False renders. Returns "" when
    thinking is off or the template ignores enable_thinking (the two renders match), so callers can
    unconditionally append it to the teacher prompt for student/teacher conditioning parity."""
    if not _w.THINKING:
        return ""
    probe = [{"role": "user", "content": ""}]
    with contextlib.suppress(Exception):
        base = tok.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        think = tok.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if think == base:
            return ""  # template ignores enable_thinking -> plain "Assistant: " already matches
        # The thinking render opens a reasoning block the non-thinking render doesn't, but it is NOT
        # always a pure suffix (the old think.startswith(base) test). Compute the longest common PREFIX
        # and SUFFIX of the two renders; the thinking render's UNIQUE MIDDLE is the opener.
        p = 0
        m = min(len(base), len(think))
        while p < m and base[p] == think[p]:
            p += 1
        s = 0
        while s < len(base) - p and s < len(think) - p and base[-1 - s] == think[-1 - s]:
            s += 1
        think_mid = think[p : len(think) - s]
        base_mid = base[p : len(base) - s]
        # CLOSED-BLOCK hybrid recovery, checked BEFORE the think_mid early-return: enable_thinking=False
        # force-CLOSES the block (base's unique middle is a closing tag "</think>...") while the shared
        # prefix already opened "<think>", which the student pre-fills. Recover the OPEN-block opener from
        # the think render so the teacher conditions on the same open block instead of base's closed one.
        # This MUST run before `if think_mid` because a base that closes the block right after the opener
        # leaves a non-empty WHITESPACE remainder in think_mid (base "<think></think>" vs think
        # "<think>\n" -> think_mid "\n"), which the early-return would otherwise hand back in place of the
        # real "<think>\n" opener (codex[bot]). The base "<think>\n\n</think>" / think "<think>\n" shape
        # (think_mid EMPTY) is the SAME recovery. lstrip absorbs intra-block whitespace before the closing
        # tag so detection still fires; we return think[cut:] (the thinking-side opener), so the strip only
        # affects DETECTION, not the returned opener. If the opener isn't in the shared prefix, fall
        # through: return the think_mid delta ("" only when the model opens <think> inside the completion).
        base_mid_tag = base_mid.lstrip()
        if base_mid_tag.startswith("</") and ">" in base_mid_tag:
            open_tag = (
                "<" + base_mid_tag[2 : base_mid_tag.index(">") + 1]
            )  # "</think>..." -> "<think>"
            cut = think.rfind(open_tag, 0, p)
            if cut != -1:
                return think[cut:]  # e.g. "<think>\n"
        if think_mid:
            return think_mid  # opener appended, or inserted before shared trailing template text
    return ""


def _student_model(model_id, mik, device):
    """Build the trainable student LoRA and return ``(model, rollout_model_source)``.

    Warm-starts from ``train.init_from_adapter`` when set — continuing a prior run's adapter (e.g. an
    SFT checkpoint), the same path GRPO uses via ``_init_adapter_model`` — otherwise a fresh LoRA on
    the base. This makes an SFT->opd pipeline a genuine continuation (the opd stage keeps the SFT
    behavior) rather than silently restarting from base.

    Warm-start (``init_peft is None``) returns a trainable PeftModel that CONTINUES the prior adapter
    in place — VL and non-VL alike — so the saved/deployed adapter is that same rank-r adapter on the
    catalog base (no merge, no recombine). Only the FRESH-LoRA path (``init_peft`` is a config) builds
    a new adapter here, loading the full multimodal model for VL so all-linear LoRA sees every target.
    """
    init_model, init_peft = _w._init_adapter_model(model_id)
    if init_peft is None:
        # init_model is already a trainable PeftModel continuing the prior (e.g. SFT) adapter.
        return init_model.to(device), model_id
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    model_cls = AutoModelForCausalLM
    if _w.is_vl_checkpoint(model_id):
        # VL checkpoints are trained/served on the full multimodal tree, including visual linears, so
        # a fresh LoRA must target that same tree (parity with the warm-start / serving module set).
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText

    # init_model is the base id (fresh run). VL runs load the full multimodal model so all-linear LoRA
    # sees every target; non-VL runs keep the lighter causal-LM loader.
    base = model_cls.from_pretrained(init_model, trust_remote_code=True, **mik).to(device)
    return get_peft_model(base, init_peft), str(init_model)


def _opd_progress(step: int, done: int) -> None:
    """Emit the in-loop non-liveness ``opd_step`` progress ping. Pollers ignore liveness heartbeats, so
    a long teacher-bound step (the bounded-concurrency teacher scoring of a large batch/group, or a
    slow/retrying Fireworks endpoint that stalls even the overlapped calls; also the serial on-policy
    generation preceding it) — or a stretch where every sample skips (empty completions, no teacher
    signal) — would otherwise emit only
    liveness pings and trip the training stall window. Report ``step`` = optimizer updates COMPLETED so
    far (opt_steps), NOT the loop index: opd_step is step-gated in the poller (_poll.STEP_GATED_STAGES),
    so while the FIRST optimizer step is still accumulating (opt_steps==0) these pings keep the WIDE
    setup grace and must not flip a still-running first step into the tight training window; once a real
    update has landed (opt_steps>=1) they tighten it as intended, and the throttle bounds the HF upload
    rate."""
    _w.heartbeat("opd_step", step=step, samples_done=done)


def _format_skip_counts(counts: Counter[str]) -> str:
    """Compact human-readable summary for OPD no-signal diagnostics."""

    if not counts:
        return "unknown=0"
    return ", ".join(f"{k}={counts[k]}" for k in sorted(counts))
