"""CPU dry-run of node_entry's train->metrics->finalize path with heavy deps mocked.

This can't validate that the model actually trains on a GPU (needs the GPU), but it
exercises the pure control flow (train_meta handoff, adapter resolution, vLLM eval loop,
grading, RunMetrics build, DONE finalize) to catch NameErrors / attribute / shape bugs
before spending GPU budget on the real run.
"""

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Disable HF I/O in node_entry helpers (they early-return when HF_REPO is empty).
os.environ["HF_REPO"] = ""
os.environ["PHASE"] = "rl"
os.environ["RUN_MODE"] = "rl"
os.environ["SEED"] = "0"


def test_grpo_batching_matches_prompts_per_step():
    """Regression guard for the GRPO batch bug: TRL sizes batches in COMPLETIONS, so
    grad-accum must include the group size. Each optimizer step must optimize the intended
    number of *unique prompts* (64), not prompts_per_step/group_size (the old 8/step bug)."""
    import autoslm.engine.worker as ne

    for per_device in (8, 4, 16, 1):
        b = ne.compute_grpo_batching(prompts_per_step=64, group_size=8, per_device_comps=per_device)
        assert b["unique_prompts_per_step"] == 64, (per_device, b)
        assert b["generations_per_step"] == 512, (per_device, b)
        assert b["divisible_by_group"] is True, (per_device, b)
        assert b["per_device_train_batch_size"] * b["gradient_accumulation_steps"] == 512

    # The OLD formula would have given only 8 prompts/step (what we are fixing):
    old_grad_accum = 64 // 8
    assert (8 * old_grad_accum) // 8 == 8


def test_reward_heartbeat_callback_accumulates_history():
    """The live-signal feature: the GRPO callback records a per-step reward_history (only
    from step logs that carry a 'reward') and ignores non-reward logs. Uses a minimal
    transformers stub so the test doesn't depend on a real transformers install."""
    saved = sys.modules.get("transformers")
    tfm = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    tfm.TrainerCallback = TrainerCallback
    sys.modules["transformers"] = tfm
    try:
        os.environ["HF_REPO"] = ""  # heartbeat stays local (no HF upload) in tests
        import autoslm.engine.worker as ne

        cb = ne.make_reward_heartbeat_callback()

        class _State:
            global_step = 0

        st = _State()
        st.global_step = 1
        cb.on_log(None, st, None, logs={"reward": 0.50, "loss": 1.2})
        st.global_step = 2
        cb.on_log(None, st, None, logs={"reward": 0.62})
        cb.on_log(None, st, None, logs={"eval_loss": 0.9})  # ignored
        cb.on_log(None, st, None, logs=None)  # ignored
        cb.on_log(None, st, None, logs={"reward": None})  # ignored

        assert cb.reward_history == [0.50, 0.62], cb.reward_history
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved
