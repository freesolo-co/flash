"""CPU dry-run of the worker's standalone eval (`slm eval`) path with GPU deps mocked."""

import importlib
import json
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["HF_REPO"] = ""
os.environ["RUN_MODE"] = "eval_only"
os.environ["PHASE"] = "eval"
os.environ["SEED"] = "0"
os.environ["EVAL_NUM"] = "4"


def _install_mocks():
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
            return [_Req("the answer is 0", 5) for _ in prompts]

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
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **k):
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


def test_eval_only_flow():
    _install_mocks()
    import autoslm.engine.worker as ne

    importlib.reload(ne)
    # avoid network: fake the active env's eval dataset (gold=42)
    ne.ACTIVE_ENV.dataset = lambda split, *a, **k: [
        {"question": "q", "gold": "42", "solution": "x\n#### 42"} for _ in range(4)
    ]

    for p in ("/tmp/metrics.json", "/tmp/DONE"):
        if os.path.exists(p):
            os.remove(p)

    ne.run_eval_only()

    assert os.path.exists("/tmp/metrics.json")
    assert os.path.exists("/tmp/DONE")
    with open("/tmp/metrics.json") as f:
        m = json.load(f)
    assert m["arm"] == "runpod"
    assert m["phase"] == "eval"
    assert m["base_eval_acc"] == 0.0  # base returns wrong answer
    assert m["trained_eval_acc"] is None  # no adapter provided
