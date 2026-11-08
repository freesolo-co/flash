"""Isolated A100 FP8-weight qualification; never changes production placement."""

import asyncio
import json
import time
import uuid
from pathlib import Path

import modal

app = modal.App("flash-a100-fp8-qualification")
if modal.is_local():
    from flash.serving.app.modal_app import engine_secrets, hf_cache_volume
    from flash.serving.lambda_cloud.config import RUNTIME_IMAGE

    image = (
        modal.Image.from_registry(RUNTIME_IMAGE)
        .env(
            {
                "HF_HOME": "/vol/hosting-cache",
                "HF_HUB_CACHE": "/vol/hosting-cache/hub",
                "VLLM_CACHE_ROOT": "/vol/hosting-cache/vllm",
                "HF_HUB_DISABLE_XET": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_MOE_USE_DEEP_GEMM": "0",
                # Trusted local test callback only; never enabled in hosted serving.
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        )
        .add_local_dir(Path(__file__).resolve().parent.parent / "flash", "/app/flash", copy=True)
    )
else:
    image = None
    engine_secrets = []
    hf_cache_volume = modal.Volume.from_name("freesolo-lora-serving-hf-cache")


def inspect_quantization(worker):
    from collections import Counter

    model = worker.model_runner.model
    methods = Counter()
    dtypes = Counter()
    expert_dtypes = Counter()
    examples = []
    for name, module in model.named_modules():
        scheme = getattr(module, "scheme", None)
        if scheme is not None:
            methods["scheme:" + type(scheme).__name__] += 1
        method = getattr(module, "quant_method", None)
        if method is not None:
            kind = type(method).__name__
            methods[kind] += 1
            kernel = getattr(method, "fp8_linear", None)
            if kernel is not None:
                methods["kernel:" + type(kernel).__name__] += 1
            if ("Fp8" in kind or "FP8" in kind or "Compressed" in kind) and len(examples) < 8:
                examples.append(
                    {
                        "module": name,
                        "method": kind,
                        "use_marlin": getattr(method, "use_marlin", None),
                    }
                )
    for name, value in model.named_parameters():
        dtypes[str(value.dtype)] += value.numel()
        if name.rsplit(".", 1)[-1] in {"w13_weight", "w2_weight"}:
            expert_dtypes[str(value.dtype)] += value.numel()
    return {
        "quant_methods": dict(methods),
        "parameter_elements_by_dtype": dict(dtypes),
        "expert_weight_elements_by_dtype": dict(expert_dtypes),
        "examples": examples,
    }


async def lora_smoke(engine, model, rank):
    import torch
    from safetensors.torch import save_file
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    from flash.core.catalog import get_model

    info = get_model(model)
    root = Path("/tmp/a100-fp8-adapter")
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
    adapter = LoRARequest("a100-fp8-smoke", 1, str(root))
    await engine.engine.add_lora(adapter)
    last = None
    async for output in engine.engine.generate(
        "The capital of France is",
        SamplingParams(temperature=0, max_tokens=32),
        uuid.uuid4().hex,
        lora_request=adapter,
    ):
        last = output
    if last is None or not last.finished or not last.outputs[0].token_ids:
        raise RuntimeError("LoRA generation incomplete")
    return {
        "rank": rank,
        "target": "q_proj",
        "text": last.outputs[0].text,
        "synthetic_zero_adapter": True,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=engine_secrets,
    volumes={"/vol/hosting-cache": hf_cache_volume},
    max_containers=2,
    single_use_containers=True,
    timeout=3600,
)
async def qualify(model, fp8_revision="", selective=False, online_dense=False, tensor_parallel=1):
    import torch
    from vllm import SamplingParams

    from flash.serving.src.engine import model_config
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl
    from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload
    from flash.serving.src.store import settings as serving_settings

    serving_settings.TENSOR_PARALLEL_SIZE = tensor_parallel
    start = time.monotonic()
    result = {
        "model": model,
        "tensor_parallel": tensor_parallel,
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "scope": f"FP8 candidate weights / 16-bit activations; {tensor_parallel} x A100 80GB; catalog context and LoRA buffers unchanged",
    }
    config = model_config._BY_MODEL[model]
    config["gpu"] = "A100-80GB"
    if model in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"):
        config["engine"]["quantization"] = "fp8"
    if model == "Qwen/Qwen3.6-35B-A3B":
        assert len(fp8_revision) == 40
        config["engine"].update(
            serve_model_id=model + "-FP8", model_revision=fp8_revision, quantization="fp8"
        )
    if online_dense and model in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"):
        config["engine"].update(
            serve_model_id=model,
            model_revision=config["engine"]["tokenizer_revision"],
            quantization="fp8",
        )
    if (selective and model in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B")) or online_dense:
        from flash.serving.src.engine import boot

        original_args = boot.engine_args_for
        exclusions = ["re:.*visual.*", "re:.*lm_head$", "re:.*in_proj_(a|b|ba)$"]

        def selective_args(*args, **kwargs):
            values = original_args(*args, **kwargs)
            values["quantization"] = "fp8_per_tensor"
            values["quantization_config"] = {"ignore": exclusions}
            return values

        boot.engine_args_for = selective_args
        result["fp8_exclusions"] = exclusions
    result["settings"] = model_config.engine_overrides_for(model)
    engine = _LoraEngineImpl()
    engine.base_model = model
    engine.pinned_gpu = "A100-80GB"
    try:
        async with asyncio.timeout(3000):
            await engine._load()
        result["ready_seconds"] = time.monotonic() - start
        result["health"] = engine._health()
        result["quantization_inspection"] = await engine.engine.collective_rpc(inspect_quantization)
        methods = result["quantization_inspection"][0]["quant_methods"]
        if not any(("fp8" in name.lower() or "w8a" in name.lower()) for name in methods):
            raise RuntimeError("Loaded worker has no verified FP8 quantization method")
        record = internal_adapter_payload(
            AdapterRecord(
                model_id=model,
                repo_id=model,
                base_model=model,
                thinking=False,
                serve_base_model=True,
            )
        )
        payload = {
            "model_id": model,
            "messages": [{"role": "user", "content": "Reply with the word hello."}],
            "temperature": 0,
            "max_tokens": 32,
        }
        result["buffered"] = await engine._generate(
            payload, record_dict=record, generation_id=uuid.uuid4().hex
        )
        events = [
            e
            async for e in engine._stream_generate(
                payload, record_dict=record, generation_id=uuid.uuid4().hex
            )
        ]
        if not result["buffered"].get("text") or not events or events[-1].get("type") != "final":
            raise RuntimeError("Buffered or streaming inference incomplete")
        result["stream_final"] = events[-1]
        # A bounded 4k decode checks more than the tiny greeting; native context remains configured.
        tokens = engine.tokenizer.encode("Testing inference with quantized weights. " * 1000)[:4096]
        last = None
        async for output in engine.engine.generate(
            {"prompt_token_ids": tokens},
            SamplingParams(max_tokens=32, temperature=0),
            uuid.uuid4().hex,
        ):
            last = output
        assert last is not None
        assert last.finished
        assert last.outputs[0].token_ids
        result["four_k_decode"] = {
            "prompt_tokens": len(tokens),
            "output_tokens": len(last.outputs[0].token_ids),
        }
        result["lora_qproj_smoke"] = await lora_smoke(
            engine, model, result["settings"]["max_lora_rank"]
        )
        result["capital_sanity_passed"] = "paris" in result["lora_qproj_smoke"]["text"].lower()
        if not result["capital_sanity_passed"]:
            raise RuntimeError("Execution succeeded but zero-adapter factual-output sanity failed")
        if not model.endswith("-Base"):
            result["greeting_sanity_passed"] = (
                "hello" in result["buffered"]["text"].lower()
                and "hello" in events[-1]["choices"][0]["text"].lower()
            )
            if not result["greeting_sanity_passed"]:
                raise RuntimeError("Execution succeeded but instruction-output sanity failed")
        result["status"] = "passed_base_and_qproj_lora"
        result["limitations"] = [
            "Synthetic zero q_proj adapter only; no full-expert or trained-adapter quality qualification",
            "No full 256k input or quality benchmark",
            "Modal hardware test; not Lambda provisioning qualification",
        ]
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__ + ": " + str(exc)[:1500]
    finally:
        await engine._shutdown()
    result["elapsed_seconds"] = time.monotonic() - start
    return result


@app.local_entrypoint()
def main(
    output: str,
    models: str = "",
    selective: bool = False,
    online_dense: bool = False,
    concurrency: int = 2,
    tensor_parallel: int = 1,
):
    from huggingface_hub import model_info

    from flash.serving.src.engine.model_config import base_models

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    selected = models.split(",") if models else base_models()
    revision = (
        model_info("Qwen/Qwen3.6-35B-A3B-FP8").sha if "Qwen/Qwen3.6-35B-A3B" in selected else ""
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if concurrency not in (1, 2):
        raise ValueError("qualification concurrency must be one or two")
    if tensor_parallel not in (1, 2):
        raise ValueError("qualification supports one or two GPUs")
    function = qualify if tensor_parallel == 1 else qualify.with_options(gpu="A100-80GB:2")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {
            pool.submit(
                function.remote, model, revision, selective, online_dense, tensor_parallel
            ): model
            for model in selected
        }
        for future in as_completed(pending):
            model = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "model": model,
                    "status": "remote_failed",
                    "error": type(exc).__name__ + ": " + str(exc)[:1500],
                }
            (folder / (model.rsplit("/", 1)[-1] + ".json")).write_text(json.dumps(result, indent=2))
            print("QUALIFICATION_RESULT", json.dumps(result), flush=True)
