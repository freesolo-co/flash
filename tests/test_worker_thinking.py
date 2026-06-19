"""CPU dry-run of the worker's thinking-mode behavior (heavy deps mocked).

Covers the run-wide THINKING flag: strip_think drops <think> blocks before the env
reward sees a completion (an unclosed block scores 0 even when the gold number
appears inside the reasoning), and the thinking-aware GRPO micro-batch default kicks
in. strip_think is applied once in worker.graded_text before the env rewards — so it
works for every environment.
"""

from __future__ import annotations

import importlib
import json
import os

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


def test_thinking_budget_selection(monkeypatch):
    # A JobSpec with an env id makes the worker resolve ACTIVE_ENV at import; stub the loader so
    # this CPU dry-run doesn't reach the Prime Hub. We only exercise THINKING / micro-batch here.
    monkeypatch.setattr("flash.envs.registry.load_environment", lambda *a, **k: object())
    saved = _set_thinking_worker_env()
    import flash.engine.worker as ne

    try:
        importlib.reload(ne)
        assert ne.THINKING is True
        # GRPO micro-batch shrinks (logits VRAM scales with seq len); effective batch
        # is preserved through grad-accum
        assert ne.rl_per_device_comps() == 2
        b = ne.compute_grpo_batching(64, 8, ne.rl_per_device_comps())
        assert b["unique_prompts_per_step"] == 64
        assert b["divisible_by_group"]
        # RL_PER_DEVICE_PROMPTS is no longer an override — the default + auto-caps stand.
        os.environ["RL_PER_DEVICE_PROMPTS"] = "4"
        assert ne.rl_per_device_comps() == 2
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
        assert ne.rl_per_device_comps() == 8
    finally:
        os.environ.pop("FLASH_JOB_SPEC_JSON", None)
        importlib.reload(ne)


def test_grpo_batching_rounds_up_to_divisible():
    # A small/non-divisible override (prompts=5, group=3, per_device=8) used to leave
    # the global completion batch (8) indivisible by num_generations (3); TRL rejects
    # that only AFTER the paid worker is provisioned. compute_grpo_batching now rounds
    # grad_accum up so the batch is always divisible by group_size.
    import flash.engine.worker as ne

    importlib.reload(ne)
    for prompts, group, per_device in [(5, 3, 8), (64, 8, 2), (64, 8, 8), (7, 5, 4), (1, 6, 8)]:
        b = ne.compute_grpo_batching(prompts, group, per_device)
        assert b["divisible_by_group"], (prompts, group, per_device)
        assert b["generations_per_step"] % group == 0
        assert b["unique_prompts_per_step"] >= 1
        # never shrinks below the requested prompts/step (rounding only goes up)
        assert (
            b["unique_prompts_per_step"] >= prompts or b["generations_per_step"] >= prompts * group
        )


def test_grpo_batching_caps_per_device_at_target():
    # A small prompts_per_step must not be overshot by an oversized per-device completion
    # micro-batch: the global completion batch is capped at prompts_per_step * group_size,
    # so the per-device micro-batch is clamped down to the target (mirrors run_sft).
    import flash.engine.worker as ne

    importlib.reload(ne)
    # per_device (8) far exceeds target completions (1 * 2 = 2): must clamp, not overshoot.
    b = ne.compute_grpo_batching(prompts_per_step=1, group_size=2, per_device_comps=8)
    assert b["per_device_train_batch_size"] <= 2
    assert b["generations_per_step"] <= 2
    assert b["unique_prompts_per_step"] == 1
    # The common default stays a no-op: per_device passes through unchanged.
    b = ne.compute_grpo_batching(prompts_per_step=64, group_size=8, per_device_comps=2)
    assert b["per_device_train_batch_size"] == 2
    assert b["unique_prompts_per_step"] == 64


def test_rl_per_device_comps_colocated_flag(monkeypatch):
    """The colocate activation cap (which shrinks per_device for a colocated rollout engine) must be
    gated by ``colocated``: in disaggregated mode the engine is on a separate GPU, so passing
    colocated=False must NOT apply that cap (else grad-accum inflates and the split's throughput is
    cancelled). The logits-budget cap (a real per-device VRAM term) still applies in both modes.

    The cap is CUDA-gated, so on a CPU runner force the cap branch via a fake torch to prove the
    flag actually gates it: colocated=True clamps to the tiny fake-VRAM act_cap; colocated=False
    keeps the (larger) base/logits value."""
    import types

    import flash.engine.worker as ne

    importlib.reload(ne)
    monkeypatch.delenv("RL_PER_DEVICE_PROMPTS", raising=False)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _i: types.SimpleNamespace(total_memory=8 * 1024**3),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    colocated = ne.rl_per_device_comps(use_vllm=True, colocated=True, params_b=4.0)
    disagg = ne.rl_per_device_comps(use_vllm=True, colocated=False, params_b=4.0)
    # disaggregated skips the activation cap, so it is at least as large as the colocated value
    assert disagg >= colocated
    # and on this tiny fake card the cap actually bites, so the two genuinely differ
    assert disagg > colocated
