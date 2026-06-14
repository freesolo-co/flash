"""CPU dry-run of node_entry's eval->metrics->finalize path with heavy deps mocked.

This can't validate that the model actually trains on a GPU (needs the GPU), but it
exercises the pure control flow (train_meta handoff, adapter resolution, vLLM eval loop,
grading, RunMetrics build, DONE finalize) to catch NameErrors / attribute / shape bugs
before spending GPU budget on the real run.
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Disable HF I/O in node_entry helpers (they early-return when HF_REPO is empty).
os.environ["HF_REPO"] = ""
os.environ["PHASE"] = "rl"
os.environ["RUN_MODE"] = "eval_after"
os.environ["SEED"] = "0"
os.environ["EVAL_NUM"] = "4"


def _install_mocks():
    # --- mock vllm ---
    vllm = types.ModuleType("vllm")

    class _Out:
        def __init__(self, text, n):
            self.text = text
            self.token_ids = list(range(n))

    class _Req:
        def __init__(self, text, n):
            self.outputs = [_Out(text, n)]

    class LLM:
        def __init__(self, *a, **k):
            self._calls = 0

        def generate(self, prompts, sp, lora_request=None):
            # base run returns a wrong answer; adapter run returns correct \boxed{}
            self._calls += 1
            if lora_request is None:
                return [_Req("the answer is 0", 5) for _ in prompts]
            return [_Req("reasoning... \\boxed{42}", 8) for _ in prompts]

    class SamplingParams:
        def __init__(self, *a, **k):
            pass

    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    lora_mod = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, *a, **k):
            pass

    req_mod.LoRARequest = LoRARequest
    sys.modules["vllm"] = vllm
    sys.modules["vllm.lora"] = lora_mod
    sys.modules["vllm.lora.request"] = req_mod

    # --- mock transformers.AutoTokenizer ---
    tfm = types.ModuleType("transformers")

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **k):
            return msgs[-1]["content"]

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    tfm.AutoTokenizer = _Tok
    sys.modules["transformers"] = tfm

    # --- mock torch: report a ready CUDA device so wait_for_gpu() passes instantly ---
    torch = types.ModuleType("torch")
    torch.zeros = lambda *a, **k: 0  # 0 + 1 in wait_for_gpu's probe works on a plain int
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        get_device_name=lambda i=0: "MockGPU",
        device_count=lambda: 1,
    )
    sys.modules["torch"] = torch


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


def test_supported_config_kwargs_filters_unknown():
    """Version-safety: unknown GRPOConfig kwargs are dropped (not crash); known recipe
    knobs pass through."""
    import dataclasses

    import autoslm.engine.worker as ne

    @dataclasses.dataclass
    class FakeGRPOConfig:
        learning_rate: float = 1e-5
        lr_scheduler_type: str = "linear"
        beta: float = 0.0

    kept = ne.supported_config_kwargs(
        FakeGRPOConfig,
        {
            "learning_rate": 2e-5,
            "lr_scheduler_type": "constant",
            "beta": 0.0,
            "scale_rewards": "none",
            "loss_type": "dr_grpo",  # absent in this fake version
        },
    )
    assert kept == {"learning_rate": 2e-5, "lr_scheduler_type": "constant", "beta": 0.0}


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


def test_eval_after_flow():
    _install_mocks()
    import autoslm.engine.worker as ne

    # avoid network: fake the active env's eval dataset (gold=42)
    ne.ACTIVE_ENV.dataset = lambda split, *a, **k: [
        {"question": "q1", "gold": "42", "solution": "x\n#### 42"} for _ in range(4)
    ]

    # write a train_meta as the fine-tuning process would
    os.makedirs("/tmp/rl_seed0/adapter", exist_ok=True)
    with open("/tmp/rl_seed0/adapter/adapter_config.json", "w") as f:
        f.write("{}")
    with open("/tmp/train_meta.json", "w") as f:
        json.dump(
            {
                "phase": "rl",
                "adapter_dir": "/tmp/rl_seed0/adapter",
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "train_wall": 1000.0,
                "setup_seconds": 60.0,
                "train_tokens": 0,
                "generated_tokens": 12_000_000,
                "notes": {"steps": 150},
            },
            f,
        )

    # run the fresh-process eval flow
    for p in ("/tmp/metrics.json", "/tmp/DONE"):
        if os.path.exists(p):
            os.remove(p)
    ne.run_eval_after()

    assert os.path.exists("/tmp/metrics.json"), "metrics.json not written"
    assert os.path.exists("/tmp/DONE"), "DONE sentinel not written"
    with open("/tmp/metrics.json") as f:
        m = json.load(f)
    assert m["arm"] == "runpod"
    assert m["phase"] == "rl"
    assert abs(m["base_eval_acc"] - 0.0) < 1e-9, m["base_eval_acc"]  # base wrong -> 0
    assert abs(m["trained_eval_acc"] - 1.0) < 1e-9, m["trained_eval_acc"]  # adapter \boxed{42} -> 1
    assert m["generated_tokens"] == 12_000_000
    assert m["train_throughput_toks_per_s"] > 0
    print(
        "dry-run metrics:",
        {
            k: m[k]
            for k in ("base_eval_acc", "trained_eval_acc", "generated_tokens", "wall_seconds")
        },
    )


if __name__ == "__main__":
    import traceback

    try:
        test_eval_after_flow()
        print("PASS test_eval_after_flow")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
