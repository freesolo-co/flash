"""Paid, isolated throughput sweep; never deploys or routes customer traffic.

Run: modal run scripts/benchmark_qwen_serving.py --base-model Qwen/Qwen3.5-9B-Base
Use --caps and --concurrency to compare scheduler caps at the same offered load.
Use --profile stress to add mixed lengths and a full-cap 32k LoRA burst.
Use --profile sweep-stress for both, and --token-budget to vary chunked prefill.
Use --gpu-override to compare hardware without changing production routing.
Use --hot-loras to measure a smaller GPU-resident adapter cache (zero keeps the catalog).
Use --checkpoint with --checkpoint-revision for an immutable quantized candidate.
QWEN_BENCH_IMAGE_ID optionally reuses an already-built serving image.
--hold-gpu restarts engines within one allocation instead of requeueing between caps.
By default each sequence cap gets a fresh single-use GPU. Results exclude startup and report
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


async def generation_smokes(engine):
    """Exercise ordinary detokenization and the preserved vision path separately from rates."""
    from vllm import SamplingParams

    prompts = {"text": "The capital of France is"}
    if engine.processor is not None:
        from PIL import Image

        prompts["image"] = {
            "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Describe the image.<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {"image": Image.new("RGB", (64, 64), "blue")},
        }
    outputs = {}
    for kind, prompt in prompts.items():
        last = None
        async with asyncio.timeout(120):
            async for output in engine.engine.generate(
                prompt, SamplingParams(temperature=0, max_tokens=64), "smoke-" + uuid.uuid4().hex
            ):
                last = output
        if last is None or not last.finished or not last.outputs[0].token_ids:
            raise RuntimeError(f"Incomplete {kind} generation smoke")
        outputs[kind] = last.outputs[0].text
    print("GENERATION_SMOKES", json.dumps(outputs), flush=True)
    return outputs


@app.function(
    image=image,
    secrets=engine_secrets,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    timeout=3600,
    volumes={"/vol/hosting-cache": hf_cache_volume},
)
async def measure(
    model,
    sequence_cap,
    offered_concurrency,
    profile,
    token_budget,
    gpu_override="",
    hot_loras=0,
    checkpoint="",
    checkpoint_revision="",
):
    from vllm import SamplingParams

    from flash.serving.src.engine import model_config
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl

    model_config._BY_MODEL[model]["engine"]["max_num_seqs"] = sequence_cap
    if gpu_override:
        model_config._BY_MODEL[model]["gpu"] = gpu_override
    if hot_loras:
        model_config._BY_MODEL[model]["engine"]["max_loras"] = hot_loras
    if checkpoint:
        model_config._BY_MODEL[model]["engine"].update(
            serve_model_id=checkpoint, model_revision=checkpoint_revision, quantization=None
        )
    if token_budget:
        model_config._BY_MODEL[model]["engine"]["max_num_batched_tokens"] = token_budget
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
        # Full-cap 32k bursts can take minutes on smaller GPUs. Measure that
        # latency explicitly instead of confusing a four-minute cutoff with OOM.
        async with asyncio.timeout(600):
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
    if profile in ("stress", "sweep-stress"):
        for length in ([512] * 11 + [4096] * 4 + [32768], 32768):
            row = await wave(sequence_cap, length, lora)
            results.append(row)
            print("BENCH", json.dumps({"sequence_cap": sequence_cap, **row}), flush=True)
    result = {
        "model": model,
        "sequence_cap": sequence_cap,
        "profile": profile,
        "token_budget_override": token_budget or None,
        "gpu_override": gpu_override or None,
        "hot_loras_override": hot_loras or None,
        "checkpoint_override": checkpoint or None,
        "checkpoint_revision": checkpoint_revision or None,
        "boot_seconds": boot,
        "generation_smokes": await generation_smokes(engine),
        "health": engine._health(),
        "measurements": results,
    }
    engine._liveness_task.cancel()
    await asyncio.gather(engine._liveness_task, return_exceptions=True)
    engine.engine.shutdown()
    return result


@app.function(
    image=image,
    secrets=engine_secrets,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    timeout=7200,
    volumes={"/vol/hosting-cache": hf_cache_volume},
)
async def measure_matrix(
    model,
    sequence_caps,
    offered_concurrency,
    profile,
    token_budget,
    gpu_override="",
    hot_loras=0,
    checkpoint="",
    checkpoint_revision="",
):
    import gc

    results = []
    for cap in sequence_caps:
        result = await measure.local(
            model,
            cap,
            offered_concurrency,
            profile,
            token_budget,
            gpu_override,
            hot_loras,
            checkpoint,
            checkpoint_revision,
        )
        results.append(result)
        print("CAP_RESULT", json.dumps(result), flush=True)
        # Each vLLM EngineCore is a separate child process and shutdown waits for
        # it to exit. Reclaim parent references, then reject a leaked GPU allocation
        # instead of allowing it to bias the next engine's memory profiling.
        gc.collect()
        await asyncio.sleep(1)
        used = (
            await asyncio.to_thread(
                subprocess.check_output,
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
        ).strip()
        print("GPU_AFTER_SHUTDOWN_MIB", used, flush=True)
        if any(int(value) > 1024 for value in used.splitlines()):
            raise RuntimeError("Engine shutdown retained more than 1 GiB of GPU memory")
    return results


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-9B-Base",
    caps: str = "32,64,128",
    concurrency: str = "1,16,32,64,128",
    profile: str = "sweep",
    token_budget: int = 0,
    gpu_override: str = "",
    hot_loras: int = 0,
    checkpoint: str = "",
    checkpoint_revision: str = "",
    hold_gpu: bool = False,
    output: str = "results/qwen-throughput.json",
):
    if base_model not in base_models():
        raise ValueError(f"Unknown hosted Qwen model: {base_model}")
    if profile not in ("sweep", "stress", "sweep-stress"):
        raise ValueError("Profile must be sweep, stress, or sweep-stress")
    if token_budget and not 2048 <= token_budget <= 32768:
        raise ValueError("Token budget must be 0 (catalog default) or between 2048 and 32768")
    sequence_caps = [int(value) for value in caps.split(",")]
    if not sequence_caps or any(value < 1 or value > 128 for value in sequence_caps):
        raise ValueError("Sequence caps must be between 1 and 128")
    offered_concurrency = [int(value) for value in concurrency.split(",")]
    if not offered_concurrency or any(value < 1 or value > 128 for value in offered_concurrency):
        raise ValueError("Offered concurrency must be between 1 and 128")
    if gpu_override and gpu_override not in ("L4", "L40S", "H100", "H200"):
        raise ValueError("GPU override must be L4, L40S, H100, or H200")
    if hot_loras not in (0, 1, 2, 4, 8, 16):
        raise ValueError("Hot LoRAs must be 0 (catalog default), 1, 2, 4, 8, or 16")
    if bool(checkpoint) != bool(checkpoint_revision):
        raise ValueError("Checkpoint and immutable revision must be provided together")
    if checkpoint_revision and (
        len(checkpoint_revision) != 40
        or any(c not in "0123456789abcdef" for c in checkpoint_revision)
    ):
        raise ValueError("Checkpoint revision must be a full lowercase commit SHA")
    gpu = gpu_override or gpu_for(base_model)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hold_gpu:
        benchmark = measure_matrix.with_options(gpu="H100!" if gpu == "H100" else gpu, region="us")
        results = benchmark.remote(
            base_model,
            sequence_caps,
            offered_concurrency,
            profile,
            token_budget,
            gpu_override,
            hot_loras,
            checkpoint,
            checkpoint_revision,
        )
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved {base_model} matrix measurements to {path}", flush=True)
        return
    benchmark = measure.with_options(gpu="H100!" if gpu == "H100" else gpu, region="us")
    results = []
    for cap in sequence_caps:
        print(
            f"Starting {base_model}: cap={cap}, profile={profile}, token_budget={token_budget}",
            flush=True,
        )
        results.append(
            benchmark.remote(
                base_model,
                cap,
                offered_concurrency,
                profile,
                token_budget,
                gpu_override,
                hot_loras,
                checkpoint,
                checkpoint_revision,
            )
        )
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved {base_model} sequence-cap {cap} measurements to {path}", flush=True)
