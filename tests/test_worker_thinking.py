"""CPU dry-run of the worker's thinking-mode behavior (heavy deps mocked).

Exercises the run-wide THINKING flag end to end through run_eval_after: prompt
rendering passes enable_thinking=True, grading strips <think> blocks (an unclosed
block scores 0 even when the gold number appears inside the reasoning), budgets pick
the thinking-aware defaults, and the metrics notes record the thinking flag.

Adapted to the env-based worker: eval data + grading come from ACTIVE_ENV (the gsm8k
built-in here), and strip_think is applied once in worker.graded_text before the env
grades — so it works for every environment.
"""

import importlib
import json
import os
import sys
import types
from typing import ClassVar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_WORKER_ENV = (
    "HF_REPO",
    "RUN_MODE",
    "PHASE",
    "SEED",
    "EVAL_NUM",
    "AUTOSLM_THINKING",
    "AUTOSLM_JOB_SPEC_JSON",
    "AUTOSLM_JOB_SPEC_PATH",
    "EVAL_MAX_NEW_TOKENS",
    "EVAL_MAX_MODEL_LEN",
    "RL_PER_DEVICE_PROMPTS",
)


def _set_thinking_worker_env():
    saved = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update(
        {"HF_REPO": "", "RUN_MODE": "eval_after", "PHASE": "rl", "SEED": "0", "EVAL_NUM": "4"}
    )
    os.environ["AUTOSLM_THINKING"] = "1"  # no-JobSpec env fallback (bench path)
    for k in (
        "AUTOSLM_JOB_SPEC_JSON",
        "AUTOSLM_JOB_SPEC_PATH",
        "EVAL_MAX_NEW_TOKENS",
        "EVAL_MAX_MODEL_LEN",
        "RL_PER_DEVICE_PROMPTS",
    ):
        os.environ.pop(k, None)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _install_mocks(max_new: int):
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
            pass

        def generate(self, prompts, sp, lora_request=None):
            if lora_request is None:
                # base: thinking never closes and saturates the cap; the gold number
                # appears INSIDE the reasoning (must not be grabbed by the fallback)
                return [_Req("<think> clearly the answer is 42, but", max_new) for _ in prompts]
            # adapter: closed think block + clean boxed answer
            return [_Req("<think>scratch 99</think>\\boxed{42}", 50) for _ in prompts]

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

    tfm = types.ModuleType("transformers")

    class _Tok:
        chat_template_kwargs_seen: ClassVar[list[dict]] = []

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **k):
            type(self).chat_template_kwargs_seen.append(dict(k))
            return msgs[-1]["content"]

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    tfm.AutoTokenizer = _Tok
    sys.modules["transformers"] = tfm

    torch = types.ModuleType("torch")
    torch.zeros = lambda *a, **k: 0
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        get_device_name=lambda i=0: "MockGPU",
        device_count=lambda: 1,
    )
    sys.modules["torch"] = torch
    return _Tok


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
        # thinking-aware defaults
        assert ne.eval_max_new_tokens() == ne.RECIPE.eval.max_new_tokens_thinking == 2048
        assert ne.eval_max_model_len(2048) == ne.RECIPE.rl.max_prompt_len + 2048 + 128
        # GRPO micro-batch shrinks (logits VRAM scales with seq len); effective batch
        # is preserved through grad-accum
        assert ne.rl_per_device_comps() == 2
        b = ne.compute_grpo_batching(64, 8, ne.rl_per_device_comps())
        assert b["unique_prompts_per_step"] == 64
        assert b["divisible_by_group"]
        # env overrides still win
        os.environ["EVAL_MAX_NEW_TOKENS"] = "333"
        os.environ["EVAL_MAX_MODEL_LEN"] = "4096"
        os.environ["RL_PER_DEVICE_PROMPTS"] = "4"
        assert ne.eval_max_new_tokens() == 333
        assert ne.eval_max_model_len(333) == 4096
        assert ne.rl_per_device_comps() == 4
    finally:
        _restore_env(saved)
        importlib.reload(ne)
    # thinking off: original defaults (and the old 2048 floor) are untouched
    assert ne.THINKING is False
    assert ne.eval_max_new_tokens() == ne.RECIPE.eval.max_new_tokens == 512
    assert ne.eval_max_model_len(512) == 2048
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


def test_thinking_eval_after_flow():
    saved = _set_thinking_worker_env()
    tok_cls = _install_mocks(max_new=2048)
    import autoslm.engine.worker as ne

    try:
        importlib.reload(ne)
        assert ne.THINKING is True
        # Inject a tiny eval split via the active (gsm8k) environment; grading is the
        # real env grader operating on the strip_think'd text.
        ne.ACTIVE_ENV.dataset = lambda split: [{"question": "q1", "gold": "42"} for _ in range(4)]

        os.makedirs("/tmp/rl_seed0/adapter", exist_ok=True)
        with open("/tmp/rl_seed0/adapter/adapter_config.json", "w") as f:
            f.write("{}")
        with open("/tmp/train_meta.json", "w") as f:
            json.dump(
                {
                    "phase": "rl",
                    "adapter_dir": "/tmp/rl_seed0/adapter",
                    "model_id": "Qwen/Qwen3-0.6B",
                    "train_wall": 1000.0,
                    "setup_seconds": 60.0,
                    "train_tokens": 0,
                    "generated_tokens": 1_000_000,
                    "notes": {"steps": 20, "thinking": True},
                },
                f,
            )
        for p in ("/tmp/metrics.json", "/tmp/DONE", "/tmp/generations.jsonl"):
            if os.path.exists(p):
                os.remove(p)

        ne.run_eval_after()

        with open("/tmp/metrics.json") as f:
            m = json.load(f)
        # base never closes <think> (gold inside the reasoning) -> graded 0
        assert m["base_eval_acc"] == 0.0, m["base_eval_acc"]
        # adapter closes the block and answers \boxed{42} -> graded 1
        assert m["trained_eval_acc"] == 1.0, m["trained_eval_acc"]
        # env-rendered prompts; the run-wide thinking flag is recorded
        assert m["notes"]["renderer"] == "autoslm_env"
        assert m["notes"]["thinking"] is True
        # prompts were rendered WITH thinking
        assert any(k.get("enable_thinking") is True for k in tok_cls.chat_template_kwargs_seen)
        # gens keep the RAW completion and surface cap saturation per-completion
        with open("/tmp/generations.jsonl") as f:
            gens = [json.loads(line) for line in f]
        assert all(gg["completion"].startswith("<think>") for gg in gens)
        assert all(gg["truncated"] is False for gg in gens)  # adapter gens: 50 < cap
        assert all(gg["n_tokens"] == 50 for gg in gens)
    finally:
        _restore_env(saved)
        for mod in ("vllm", "vllm.lora", "vllm.lora.request", "transformers", "torch"):
            sys.modules.pop(mod, None)
        importlib.reload(ne)
