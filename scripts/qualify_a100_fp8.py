"""Isolated A100 FP8-weight qualification; never changes production placement."""

import asyncio
import json
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.modal_app import engine_secrets, hf_cache_volume
from flash.serving.lambda_cloud.config import RUNTIME_IMAGE
from flash.serving.src.engine.model_config import base_models

app = modal.App("flash-a100-fp8-qualification")
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
        }
    )
    .add_local_python_source("flash")
)


def inspect_quantization(worker):
    from collections import Counter

    model = worker.model_runner.model
    methods = Counter()
    dtypes = Counter()
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
    for value in model.parameters():
        dtypes[str(value.dtype)] += value.numel()
    return {
        "quant_methods": dict(methods),
        "parameter_elements_by_dtype": dict(dtypes),
        "examples": examples,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=engine_secrets,
    volumes={"/vol/hosting-cache": hf_cache_volume},
    max_containers=1,
    single_use_containers=True,
    timeout=3600,
)
async def qualify(model, fp8_revision=""):
    import torch
    from vllm import SamplingParams

    from flash.serving.src.engine import model_config
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl
    from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload

    start = time.monotonic()
    result = {
        "model": model,
        "device": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "scope": "FP8 weights / 16-bit activations; single A100 80GB; catalog context and LoRA buffers unchanged",
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
        result["status"] = "passed_base_inference"
        result["limitations"] = [
            "No loaded LoRA adapter or full-expert LoRA qualification",
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
def main(output: str, models: str = ""):
    from huggingface_hub import model_info

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    selected = models.split(",") if models else base_models()
    revision = (
        model_info("Qwen/Qwen3.6-35B-A3B-FP8").sha if "Qwen/Qwen3.6-35B-A3B" in selected else ""
    )
    for model in selected:
        try:
            result = qualify.remote(model, revision)
        except Exception as exc:
            result = {
                "model": model,
                "status": "remote_failed",
                "error": type(exc).__name__ + ": " + str(exc)[:1500],
            }
        (folder / (model.rsplit("/", 1)[-1] + ".json")).write_text(json.dumps(result, indent=2))
        print("QUALIFICATION_RESULT", json.dumps(result), flush=True)
