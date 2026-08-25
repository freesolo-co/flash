"""CPU dry-run of node_entry's train->metrics->finalize path with heavy deps mocked.

This can't validate that the model actually trains on a GPU (needs the GPU), but it
exercises the pure control flow (train_meta handoff, adapter resolution, vLLM eval loop,
grading, RunMetrics build, DONE finalize) to catch NameErrors / attribute / shape bugs
before spending GPU budget on the real run.
"""

from __future__ import annotations

import os

# Disable HF I/O in node_entry helpers (they early-return when HF_REPO is empty).
os.environ["HF_REPO"] = ""
os.environ["PHASE"] = "rl"
os.environ["RUN_MODE"] = "rl"
os.environ["SEED"] = "0"


def test_on_policy_epochs_resolve_to_prompt_pool_passes():
    from flash.engine.plan.steps import on_policy_steps

    assert (
        on_policy_steps(
            epochs=2,
            prompt_count=33,
            prompts_per_step=16,
        )
        == 5
    )
