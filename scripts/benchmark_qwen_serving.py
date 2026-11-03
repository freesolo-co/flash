"""Paid, isolated throughput sweep; never deploys or routes customer traffic.

Run: modal run scripts/benchmark_qwen_serving.py --base-model Qwen/Qwen3.5-9B-Base
Use --caps and --concurrency to compare scheduler caps at the same offered load.
Use --profile stress to add mixed lengths and a full-cap 32k LoRA burst.
QWEN_BENCH_IMAGE_ID optionally reuses an already-built serving image.
Each sequence cap gets a fresh single-use GPU. Results exclude startup and report
aggregate output tokens / wall seconds for a concurrent wave, not summed per-request
rates. Unique cache salts prevent repeated prompts from inflating prefix-cache gains.
The adapter check uses a synthetic zero q_proj LoRA at the supported rank; it is a
kernel/memory smoke test, not a quality or full multimodal adapter qualification.
"""

import asyncio
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.modal_app import engine_secrets, hf_cache_volume, image
from flash.serving.src.engine.model_config import base_models, gpu_for

app = modal.App("flash-qwen-throughput-qualification")
if image_id := os.environ.get("QWEN_BENCH_IMAGE_ID"):
    image = modal.Image.from_id(image_id).add_local_python_source("flash")


@app.function(
    image=image,
    secrets=engine_secrets,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    timeout=3600,
    volumes={"/vol/hosting-cache": hf_cache_volume},
)
async def measure(model, sequence_cap, offered_concurrency, profile):
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

    def gpu_memory():
        # vLLM owns a child process, so parent torch allocator statistics miss its VRAM.
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()

    def percentile(values, fraction):
        ordered = sorted(values)
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    async def wave(concurrency, prompt_length, lora=None):
        async def request(index):
            length = (
                prompt_length[index % len(prompt_length)]
                if isinstance(prompt_length, list)
                else prompt_length
            )
            request_id = "bench-" + uuid.uuid4().hex
            start = time.monotonic()
            first = None
            last = None
            stream = engine.engine.generate(
                {"prompt_token_ids": tokens[:length], "cache_salt": request_id},
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

        memory_before = gpu_memory()
        start = time.monotonic()
        async with asyncio.timeout(240):
            rows = await asyncio.gather(*(request(index) for index in range(concurrency)))
        elapsed = time.monotonic() - start
        return {
            "concurrency": concurrency,
            "prompt_tokens": prompt_length,
            "output_tokens": sum(r["tokens"] for r in rows),
            "elapsed": elapsed,
            "output_tokens_per_second": sum(r["tokens"] for r in rows) / elapsed,
            "max_ttft": max(r["ttft"] for r in rows),
            "mean_ttft": sum(r["ttft"] for r in rows) / len(rows),
            "p50_ttft": percentile([r["ttft"] for r in rows], 0.5),
            "p95_ttft": percentile([r["ttft"] for r in rows], 0.95),
            "p50_latency": percentile([r["elapsed"] for r in rows], 0.5),
            "p95_latency": percentile([r["elapsed"] for r in rows], 0.95),
            "gpu_memory_mib_before": memory_before,
            "gpu_memory_mib_after": gpu_memory(),
            "completed_requests": len(rows),
            "lora": lora is not None,
        }

    results = []
    for length in () if profile == "stress" else (512, 4096):
        for concurrency in offered_concurrency:
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
        # Keep the long-context smoke bounded. A full high-cap 32k wave is a
        # separate overload experiment, not representative of mixed production input.
        row = await wave(min(sequence_cap, 8) if length == 32768 else sequence_cap, length, lora)
        results.append(row)
        print("BENCH", json.dumps({"sequence_cap": sequence_cap, **row}), flush=True)
    if profile == "stress":
        for length in ([512] * 11 + [4096] * 4 + [32768], 32768):
            row = await wave(sequence_cap, length, lora)
            results.append(row)
            print("BENCH", json.dumps({"sequence_cap": sequence_cap, **row}), flush=True)
    result = {
        "model": model,
        "sequence_cap": sequence_cap,
        "profile": profile,
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
    caps: str = "32,64,128",
    concurrency: str = "1,16,32,64,128",
    profile: str = "sweep",
    output: str = "results/qwen-throughput.json",
):
    if base_model not in base_models():
        raise ValueError(f"Unknown hosted Qwen model: {base_model}")
    if profile not in ("sweep", "stress"):
        raise ValueError("Profile must be sweep or stress")
    sequence_caps = [int(value) for value in caps.split(",")]
    if not sequence_caps or any(value < 1 or value > 128 for value in sequence_caps):
        raise ValueError("Sequence caps must be between 1 and 128")
    offered_concurrency = [int(value) for value in concurrency.split(",")]
    if not offered_concurrency or any(value < 1 or value > 128 for value in offered_concurrency):
        raise ValueError("Offered concurrency must be between 1 and 128")
    gpu = gpu_for(base_model)
    benchmark = measure.with_options(gpu="H100!" if gpu == "H100" else gpu, region="us")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for cap in sequence_caps:
        results.append(benchmark.remote(base_model, cap, offered_concurrency, profile))
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved {base_model} sequence-cap {cap} measurements to {path}", flush=True)
