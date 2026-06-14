"""CPU dry-run of the worker's thinking-mode behavior (heavy deps mocked).

Covers the run-wide THINKING flag: strip_think drops <think> blocks before the env
reward sees a completion (an unclosed block scores 0 even when the gold number
appears inside the reasoning), and the thinking-aware GRPO micro-batch default kicks
in. strip_think is applied once in worker.graded_text before the env rewards — so it
works for every environment.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_WORKER_ENV = (
    "HF_REPO",
    "RUN_MODE",
    "PHASE",
    "SEED",
    "AUTOSLM_THINKING",
    "AUTOSLM_JOB_SPEC_JSON",
    "AUTOSLM_JOB_SPEC_PATH",
    "RL_PER_DEVICE_PROMPTS",
)


def _set_thinking_worker_env():
    saved = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update({"HF_REPO": "", "RUN_MODE": "rl", "PHASE": "rl", "SEED": "0"})
    os.environ["AUTOSLM_THINKING"] = "1"  # no-JobSpec env fallback (bench path)
    for k in ("AUTOSLM_JOB_SPEC_JSON", "AUTOSLM_JOB_SPEC_PATH", "RL_PER_DEVICE_PROMPTS"):
        os.environ.pop(k, None)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_strip_think_unit():
    import autoslm.engine.worker as ne

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


def test_thinking_budget_selection():
    saved = _set_thinking_worker_env()
    import autoslm.engine.worker as ne

    try:
        importlib.reload(ne)
        assert ne.THINKING is True
        # GRPO micro-batch shrinks (logits VRAM scales with seq len); effective batch
        # is preserved through grad-accum
        assert ne.rl_per_device_comps() == 2
        b = ne.compute_grpo_batching(64, 8, ne.rl_per_device_comps())
        assert b["unique_prompts_per_step"] == 64
        assert b["divisible_by_group"]
        # env override still wins
        os.environ["RL_PER_DEVICE_PROMPTS"] = "4"
        assert ne.rl_per_device_comps() == 4
    finally:
        _restore_env(saved)
        importlib.reload(ne)
    # thinking off: original default micro-batch
    assert ne.THINKING is False
    assert ne.rl_per_device_comps() == 8


def test_grpo_batching_rounds_up_to_divisible():
    # A small/non-divisible override (prompts=5, group=3, per_device=8) used to leave
    # the global completion batch (8) indivisible by num_generations (3); TRL rejects
    # that only AFTER the paid worker is provisioned. compute_grpo_batching now rounds
    # grad_accum up so the batch is always divisible by group_size.
    import autoslm.engine.worker as ne

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
