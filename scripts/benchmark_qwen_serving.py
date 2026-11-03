"""Paid, isolated throughput sweep; never deploys or routes customer traffic.

Run: modal run scripts/benchmark_qwen_serving.py --base-model Qwen/Qwen3.5-9B-Base
Each sequence cap gets a fresh single-use GPU. Results exclude startup and report
aggregate output tokens / wall seconds for a concurrent wave, not summed per-request
rates. Unique cache salts prevent repeated prompts from inflating prefix-cache gains.
The adapter check uses a synthetic zero q_proj LoRA at the supported rank; it is a
kernel/memory smoke test, not a quality or full multimodal adapter qualification.
"""

import asyncio
import json
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.modal_app import engine_secrets, hf_cache_volume, image
from flash.serving.src.engine.model_config import base_models, gpu_for

app = modal.App("flash-qwen-throughput-qualification")


@app.function(
    image=image,
    secrets=engine_secrets,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    timeout=2400,
    volumes={"/vol/hosting-cache": hf_cache_volume},
)
async def measure(model, sequence_cap):
    from vllm import SamplingParams

    from flash.serving.src.engine import model_config
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl

    model_config._BY_MODEL[model]["engine"]["max_num_seqs"] = sequence_cap
    engine = _LoraEngineImpl()
    engine.base_model = model
    start = time.monotonic()
    await engine._load()
    boot = time.monotonic() - start
    tokens = engine.tokenizer.encode(
        "A checklist for testing reliable software systems. " * 5000, add_special_tokens=False
    )

    async def wave(concurrency, prompt_length, lora=None):
        async def request():
            request_id = "bench-" + uuid.uuid4().hex
            start = time.monotonic()
            first = None
            last = None
            stream = engine.engine.generate(
                {"prompt_token_ids": tokens[:prompt_length], "cache_salt": request_id},
                SamplingParams(temperature=0, max_tokens=256, ignore_eos=True),
                request_id,
                lora_request=lora,
            )
            try:
                async for output in stream:
                    last = output
                    if first is None and output.outputs and output.outputs[0].token_ids:
                        first = time.monotonic() - start
            finally:
                await stream.aclose()
                if last is None or not last.finished:
                    await engine.engine.abort(request_id)
            if last is None or not last.finished or len(last.outputs[0].token_ids) != 256:
                raise RuntimeError("incomplete benchmark generation")
            return {
                "tokens": len(last.outputs[0].token_ids),
                "ttft": first,
                "elapsed": time.monotonic() - start,
            }

        start = time.monotonic()
        rows = await asyncio.gather(*(request() for _ in range(concurrency)))
        elapsed = time.monotonic() - start
        return {
            "concurrency": concurrency,
            "prompt_tokens": prompt_length,
            "output_tokens": sum(r["tokens"] for r in rows),
            "elapsed": elapsed,
            "output_tokens_per_second": sum(r["tokens"] for r in rows) / elapsed,
            "max_ttft": max(r["ttft"] for r in rows),
            "mean_ttft": sum(r["ttft"] for r in rows) / len(rows),
            "lora": lora is not None,
        }

    results = []
    for length in (512, 4096):
        for concurrency in (1, 6, 8, 16, 32):
            for repeat in range(2):
                row = await wave(concurrency, length)
                row["repeat"] = repeat
                results.append(row)
                print("BENCH", json.dumps({"sequence_cap": sequence_cap, **row}), flush=True)
    # Maximum-rank q_proj adapter smoke/load, with the same batching workload.
    import torch
    from safetensors.torch import save_file
    from vllm.lora.request import LoRARequest

    from flash.core.catalog import get_model

    info = get_model(model)
    rank = info.serving.max_lora_rank
    root = Path("/tmp/throughput-adapter")
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
    lora = LoRARequest("throughput", 1, str(root))
    await engine.engine.add_lora(lora)
    for length in (512, 4096, 32768):
        row = await wave(sequence_cap, length, lora)
        results.append(row)
        print("BENCH", json.dumps({"sequence_cap": sequence_cap, **row}), flush=True)
    result = {
        "model": model,
        "sequence_cap": sequence_cap,
        "boot_seconds": boot,
        "health": engine._health(),
        "measurements": results,
    }
    engine._liveness_task.cancel()
    await asyncio.gather(engine._liveness_task, return_exceptions=True)
    engine.engine.shutdown()
    return result


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-9B-Base",
    caps: str = "8,16,32",
    output: str = "results/qwen-throughput.json",
):
    if base_model not in base_models():
        raise ValueError(f"Unknown hosted Qwen model: {base_model}")
    sequence_caps = [int(value) for value in caps.split(",")]
    if not sequence_caps or any(value < 1 or value > 64 for value in sequence_caps):
        raise ValueError("Sequence caps must be between 1 and 64")
    gpu = gpu_for(base_model)
    benchmark = measure.with_options(gpu="H100!" if gpu == "H100" else gpu)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for cap in sequence_caps:
        results.append(benchmark.remote(base_model, cap))
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved {base_model} sequence-cap {cap} measurements to {path}", flush=True)
