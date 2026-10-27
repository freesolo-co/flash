"""Paid isolated checks of the exact hosted 0.8B/L40S and 4B/H100 engine shapes.

Run with modal run scripts/qualify_small_qwen_serving.py::main.
No production router, adapter registry, or deployment is changed.
"""

import asyncio
import json
import time
from pathlib import Path

import modal

from flash.serving.app.modal_app import image

app = modal.App("flash-small-qwen-qualification")
cache = modal.Volume.from_name("flash-small-qwen-qualification-cache", create_if_missing=True)


async def qualify(model, expected_gpu):
    import torch
    from safetensors.torch import save_file
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from flash.core.catalog import get_model
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl

    started = time.monotonic()
    engine = _LoraEngineImpl()
    engine.base_model = model
    await engine._load()
    health = engine._health()
    if expected_gpu not in health["device_name"]:
        raise RuntimeError(f"Wrong serving GPU: {health['device_name']}")
    payload = {
        "model_id": model,
        "messages": [{"role": "user", "content": "Reply with only the word READY."}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    base = await engine._generate(
        payload,
        {
            "model_id": model,
            "repo_id": model,
            "base_model": model,
            "thinking": False,
            "serve_base_model": True,
        },
    )
    if not base.get("text"):
        raise RuntimeError("Base serving returned no content")

    # Exercise adapter loading at the advertised rank; zeros preserve the base function.
    info = get_model(model)
    rank = info.serving.max_lora_rank
    root = Path("/tmp/qualification-adapter")
    await asyncio.to_thread(root.mkdir, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": rank,
                "lora_dropout": 0,
                "bias": "none",
                "target_modules": ["q_proj"],
            }
        )
    )
    name = "base_model.model.model.language_model.layers.3.self_attn.q_proj"
    save_file(
        {
            name + ".lora_A.weight": torch.zeros(rank, info.hidden_size, dtype=torch.bfloat16),
            name + ".lora_B.weight": torch.zeros(
                2 * info.num_attention_heads * info.head_dim, rank, dtype=torch.bfloat16
            ),
        },
        str(root / "adapter_model.safetensors"),
    )
    request = LoRARequest("qualification", 1, str(root))
    await engine.engine.add_lora(request)
    prompt = engine.tokenizer.apply_chat_template(
        payload["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    last = None
    async for output in engine.engine.generate(
        prompt, SamplingParams(temperature=0, max_tokens=32), "adapter-canary", lora_request=request
    ):
        last = output
    if last is None or not last.outputs[0].text:
        raise RuntimeError("Adapter serving returned no content")
    return {
        "model": model,
        "health": health,
        "seconds": time.monotonic() - started,
        "base_text": base["text"],
        "adapter_text": last.outputs[0].text,
        "adapter_rank": rank,
    }


@app.function(image=image, gpu="L40S", timeout=2700, volumes={"/vol/hosting-cache": cache})
async def small():
    return await qualify("Qwen/Qwen3.5-0.8B", "L40S")


@app.function(image=image, gpu="H100!", timeout=2700, volumes={"/vol/hosting-cache": cache})
async def medium():
    return await qualify("Qwen/Qwen3.5-4B", "H100")


@app.local_entrypoint()
def main(output: str = "results/small-qwen-serving.json"):
    calls = [small.spawn(), medium.spawn()]
    results = [call.get() for call in calls]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
