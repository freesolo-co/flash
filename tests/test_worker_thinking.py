"""CPU dry-run of the worker's thinking-mode behavior (heavy deps mocked).

Covers the run-wide THINKING flag: strip_think extracts the answer text from
<think>-wrapped completions (an unclosed block extracts no final answer even when
the gold number appears inside the reasoning), and the thinking-aware GRPO
micro-batch default kicks in.
"""

from __future__ import annotations

import importlib
import json
import os

import pytest

_WORKER_ENV = (
    "HF_REPO",
    "RUN_MODE",
    "PHASE",
    "SEED",
    "FLASH_THINKING",
    "FLASH_JOB_SPEC_JSON",
    "FLASH_JOB_SPEC_PATH",
    "RL_PER_DEVICE_PROMPTS",
)


def _set_thinking_worker_env():
    saved = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update({"HF_REPO": "", "RUN_MODE": "rl", "PHASE": "rl", "SEED": "0"})
    # thinking is a run-config field (TOML `thinking`), not an env knob: drive it via the JobSpec.
    os.environ["FLASH_JOB_SPEC_JSON"] = json.dumps(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "thinking": True,
            "environment": {"id": "stub/env"},
        }
    )
    for k in ("FLASH_THINKING", "FLASH_JOB_SPEC_PATH", "RL_PER_DEVICE_PROMPTS"):
        os.environ.pop(k, None)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_strip_think_unit():
    import flash.engine.worker as ne

    assert ne.strip_think(None) is None
    assert ne.strip_think("no tags, answer 42") == "no tags, answer 42"
    assert ne.strip_think("<think>reason 99</think>\\boxed{42}") == "\\boxed{42}"
    # multiple blocks: only text after the LAST </think> survives
    assert ne.strip_think("<think>a</think>mid<think>b</think> ans 7") == " ans 7"
    # always-thinking templates pre-open <think> in the prompt: completions carry only
    # a closing tag
    assert ne.strip_think("reasoning...</think>\\boxed{5}") == "\\boxed{5}"
    # unclosed <think> (completion budget exhausted): pre-think text only
    assert ne.strip_think("preamble<think>still going 42") == "preamble"
    assert ne.strip_think("<think>still going 42") == ""


def test_thinking_text_unit():
    import flash.engine.worker as ne

    assert ne.thinking_text(None) is None
    assert ne.thinking_text("no tags, answer 42") is None
    assert ne.thinking_text("<think>reason 99</think>\\boxed{42}") == "reason 99"
    assert ne.thinking_text("<think>a</think>mid<think>b</think> ans 7") == "a\nb"
    assert ne.thinking_text("reasoning...</think>\\boxed{5}") == "reasoning..."
    assert (
        ne.thinking_text("reasoning...</think>\\boxed{5}", prompt_opened_thinking=True)
        == "reasoning..."
    )
    assert ne.thinking_text("preamble<think>still going 42") == "still going 42"
    assert ne.thinking_text("<think>still going 42") == "still going 42"


def test_thinking_budget_selection(monkeypatch):
    # A JobSpec with an env id makes the worker resolve ACTIVE_ENV at import; stub the loader so
    # this CPU dry-run doesn't load the Freesolo env. We only exercise THINKING resolution here.
    # (the per-device micro-batch half of this test went with trl: verl bounds the backward pass by
    # TOKENS via use_dynamic_bsz + ppo_max_token_len_per_gpu, so there is no per-device sequence
    # count to select. see test_rl_train's batch-shape and token-budget coverage.)
    monkeypatch.setattr("flash.envs.registry.load_environment", lambda *a, **k: object())
    saved = _set_thinking_worker_env()
    import flash.engine.worker as ne

    try:
        importlib.reload(ne)
        assert ne.THINKING is True
    finally:
        _restore_env(saved)
    # thinking off: a JobSpec with thinking=false -> original (larger) micro-batch
    os.environ["FLASH_JOB_SPEC_JSON"] = json.dumps(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "thinking": False,
            "environment": {"id": "stub/env"},
        }
    )
    try:
        importlib.reload(ne)
        assert ne.THINKING is False
    finally:
        os.environ.pop("FLASH_JOB_SPEC_JSON", None)
        importlib.reload(ne)


def test_grpo_prompts_per_step_caps_to_available_dataset():
    import flash.engine.worker as ne

    importlib.reload(ne)
    assert ne.resolve_grpo_prompts_per_step(requested=64, available_prompts=3) == 3
    assert ne.resolve_grpo_prompts_per_step(requested=2, available_prompts=10) == 2
    with pytest.raises(ValueError, match="at least one retained training prompt"):
        ne.resolve_grpo_prompts_per_step(requested=64, available_prompts=0)
