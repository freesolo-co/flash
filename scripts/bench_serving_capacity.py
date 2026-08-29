#!/usr/bin/env python3
"""measure real serving capacity on a blackwell card: allocator readings, then a load sweep.

this is a NONSHIPPING benchmark app. it deploys under its own app name so it can never overwrite
``freesolo-lora-serving`` (whose name is hardcoded with no deploy-time knob), and it reuses the
production ``engine_args_for`` so the numbers describe the config that actually ships rather than a
hand-rolled approximation.

what it reports, per (gpu, max_num_seqs) cell:

* the gdn prefill backend vllm actually chose -- ``flashinfer`` or ``triton``. on sm10.x this is
  gated by ``_is_libs_cu13_install_intact()`` and falls back SILENTLY, so an unrepaired blackwell
  boot bills the blackwell rate for the slower kernel. the sweep ABORTS on triton rather than
  publishing numbers for a kernel we would not ship.
* the allocator's own kv-cache sizing (``num_gpu_blocks`` x ``block_size``), which is the reading
  the derived per-sequence memory model could not supply.
* throughput and latency under a fixed offered load, so ``max_num_seqs`` can be chosen from a knee
  rather than from arithmetic.

usage (deploy is separate so the image build is not billed as gpu time)::

    modal run scripts/bench_serving_capacity.py::probe --gpu B200 --base-model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import modal

from flash.serving.app.modal_app import image as _serving_image

APP_NAME = "flash-serving-capacity-bench"

app = modal.App(APP_NAME, image=_serving_image)

_HF_SECRET = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if os.environ.get("HF_TOKEN")
    else []
)

# a probe boots one engine and exits. the timeout bounds a runaway boot into a bounded bill; the
# 35b's measured init is ~1010s, so 2400s leaves room without allowing an unbounded stall.
_PROBE_TIMEOUT_SECONDS = 2400


def _gdn_backend_decision() -> dict[str, Any]:
    """report the gdn prefill backend and WHY, from vllm's own predicate rather than a guess."""

    out: dict[str, Any] = {}
    try:
        # the predicate lives in the gdn module that uses it; there is no vllm.utils.flashinfer_utils
        # in 0.23.0. strip the lru_cache so this cannot answer from an earlier call's cache.
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _is_libs_cu13_install_intact,
        )

        check = getattr(_is_libs_cu13_install_intact, "__wrapped__", _is_libs_cu13_install_intact)
        out["libs_cu13_intact"] = bool(check())
    except Exception as exc:
        out["libs_cu13_intact"] = f"unavailable: {type(exc).__name__}: {exc}"

    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        out["sm"] = f"{major}.{minor}"
        out["device_name"] = torch.cuda.get_device_name(0)
        out["device_count"] = torch.cuda.device_count()
        out["total_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception as exc:
        out["sm"] = f"unavailable: {exc}"
    return out


@app.function(gpu="B200", timeout=_PROBE_TIMEOUT_SECONDS, secrets=_HF_SECRET)
def probe_b200(base_model: str, max_num_seqs: int, plan: list[dict]) -> dict[str, Any]:
    return _probe(base_model, max_num_seqs, "B200", plan)


@app.function(gpu="B300", timeout=_PROBE_TIMEOUT_SECONDS, secrets=_HF_SECRET)
def probe_b300(base_model: str, max_num_seqs: int, plan: list[dict]) -> dict[str, Any]:
    return _probe(base_model, max_num_seqs, "B300", plan)


def _probe(base_model: str, max_num_seqs: int, gpu: str, plan: list[dict]) -> dict[str, Any]:
    """boot the real engine once, then run every load cell in ``plan`` against that one engine.

    the boot dominates the bill (~960s measured for the 9b), so a plan of load cells amortizes it.
    the engine is built INSIDE the event loop that later drives it: v1's ``AsyncLLM`` builds an
    asyncio mp client in ``__init__``, and a client bound to one loop cannot be driven from a
    loop created later by a separate ``asyncio.run``.
    """

    import asyncio

    return asyncio.run(_probe_async(base_model, max_num_seqs, gpu, plan))


async def _probe_async(
    base_model: str, max_num_seqs: int, gpu: str, plan: list[dict]
) -> dict[str, Any]:
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    from flash.serving.src.engine.boot import engine_args_for
    from flash.serving.src.engine.model_config import engine_overrides_for
    from flash.serving.src.store import settings as cfg

    overrides = dict(engine_overrides_for(base_model))
    overrides["max_num_seqs"] = max_num_seqs

    started = time.time()
    # cfg is the SETTINGS module (MAX_LORAS, DTYPE, vllm_engine_kwargs), not model_config -- the
    # per-model overrides come in separately. boot.py:214 uses the same pairing.
    kwargs = engine_args_for(base_model, overrides, cfg)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
    boot_seconds = time.time() - started

    result: dict[str, Any] = {
        "gpu": gpu,
        "base_model": base_model,
        "max_num_seqs": max_num_seqs,
        "boot_seconds": round(boot_seconds, 1),
        "engine_args": {k: str(v) for k, v in sorted(kwargs.items())},
        **_gdn_backend_decision(),
    }

    try:
        _read_engine_state(engine, result)
        if result.get("gdn_active_backend") == "triton":
            # publishing capacity for a kernel we would not ship is worse than publishing nothing.
            result["aborted"] = "gdn resolved to triton; cutlass repair did not take on this card"
            return result
        result["loads"] = await _run_plan(engine, plan)
    finally:
        try:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as exc:  # teardown must never mask a real reading
            result["shutdown_error"] = f"{type(exc).__name__}: {exc}"

    return result


def _read_engine_state(engine: Any, result: dict[str, Any]) -> None:
    """record the allocator sizing and the resolved gdn backend, naming the path that answered.

    ``AsyncLLM`` exposes neither ``.engine`` nor ``.cache_config`` on its class surface, so the
    lookup tries the known shapes and RECORDS which one worked. a wrong guess then costs a dict
    entry instead of another cold boot.
    """

    vllm_config = None
    for path in ("vllm_config", "engine_core.vllm_config", "engine.vllm_config"):
        probe: Any = engine
        try:
            for part in path.split("."):
                probe = getattr(probe, part)
        except AttributeError:
            continue
        vllm_config = probe
        result["vllm_config_path"] = path
        break

    if vllm_config is None:
        result["allocator_read_error"] = "no vllm_config attribute path resolved on AsyncLLM"
        return

    cache_config = getattr(vllm_config, "cache_config", None)
    blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    result["num_gpu_blocks"] = blocks
    result["block_size"] = block_size
    if blocks and block_size:
        result["kv_tokens"] = int(blocks) * int(block_size)

    sched = getattr(vllm_config, "scheduler_config", None)
    result["resolved_max_num_seqs"] = getattr(sched, "max_num_seqs", None)
    result["resolved_max_num_batched_tokens"] = getattr(sched, "max_num_batched_tokens", None)

    # residual headroom after weights + kv are allocated. the selection gate wants >= 10 GiB free;
    # a value read from the card beats one derived from gpu_memory_utilization arithmetic.
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        result["free_vram_gib_after_boot"] = round(free / 1024**3, 2)
        result["used_vram_gib_after_boot"] = round((total - free) / 1024**3, 2)
    except Exception as exc:
        result["vram_read_error"] = f"{type(exc).__name__}: {exc}"

    # the backend the engine ACTUALLY resolved, from vllm's own resolver rather than a log scrape.
    # a triton answer means the cutlass repair did not take, and no capacity number from this boot
    # describes a kernel we would ship.
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        requested, active = _resolve_gdn_prefill_backend(vllm_config)
        result["gdn_requested_backend"] = requested
        result["gdn_active_backend"] = active
    except Exception as exc:
        result["gdn_backend_error"] = f"{type(exc).__name__}: {exc}"


async def _run_plan(engine: Any, plan: list[dict]) -> list[dict[str, Any]]:
    return [
        await _measure_load(
            engine,
            concurrency=int(spec["concurrency"]),
            n=int(spec.get("n", 1)),
            prompt_tokens=int(spec.get("prompt_tokens", 512)),
            max_tokens=int(spec.get("max_tokens", 128)),
            tag=str(spec.get("tag", "")),
        )
        for spec in plan
    ]


async def _measure_load(
    engine: Any,
    concurrency: int,
    n: int,
    prompt_tokens: int,
    max_tokens: int,
    tag: str,
) -> dict[str, Any]:
    """drive ``concurrency`` simultaneous requests and report per-request and aggregate rates.

    the point of the sweep: if the engine is cap-bound, per-request throughput falls roughly in
    proportion to offered load while AGGREGATE throughput stays flat. that pair of numbers is what
    distinguishes a real capacity knee from a gpu that is simply busy.

    ``n`` is the reason this campaign exists: one logical request fans out to ``n`` child sequences,
    so ``concurrency * n`` -- not ``concurrency`` -- is what ``max_num_seqs`` actually bounds.
    """

    import asyncio

    from vllm import SamplingParams

    # a word repeated to roughly ``prompt_tokens``; prefill cost should be real, not a two-token
    # request that measures only decode.
    prompt = "capacity " * max(1, prompt_tokens)
    # n > 1 at temperature 0 is REJECTED by the mirror parity contract, so benchmarking it would
    # measure a request shape we never serve. fan-out cells sample; the n=1 cells stay greedy.
    temperature = 0.0 if n == 1 else 0.7
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True, n=n)

    errors: list[str] = []

    async def one(index: int) -> dict[str, Any] | None:
        started = time.time()
        ttft = None
        final = None
        try:
            async for out in engine.generate(prompt, params, request_id=f"bench-{tag}-{index}"):
                if ttft is None and out.outputs and out.outputs[0].token_ids:
                    ttft = time.time() - started
                final = out
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            return None
        tokens = sum(len(o.token_ids) for o in final.outputs) if final is not None else 0
        elapsed = time.time() - started
        return {
            "seconds": elapsed,
            "tokens": tokens,
            "ttft": ttft,
            "tok_s": tokens / elapsed if elapsed else 0.0,
        }

    wall_started = time.time()
    gathered = await asyncio.gather(*(one(i) for i in range(concurrency)))
    wall = time.time() - wall_started

    per_request = [r for r in gathered if r is not None]
    total_tokens = sum(r["tokens"] for r in per_request)
    rates = sorted(r["tok_s"] for r in per_request)
    lat = sorted(r["seconds"] for r in per_request)
    ttfts = sorted(r["ttft"] for r in per_request if r["ttft"] is not None)

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return round(values[min(len(values) - 1, int(q * len(values)))], 3)

    return {
        "tag": tag,
        "concurrency": concurrency,
        "n": n,
        # the number max_num_seqs actually bounds, stated so it is never confused with concurrency.
        "child_sequences": concurrency * n,
        "prompt_tokens_requested": prompt_tokens,
        "max_tokens": max_tokens,
        "completed": len(per_request),
        "errors": errors[:5],
        "error_count": len(errors),
        "wall_seconds": round(wall, 2),
        "total_output_tokens": total_tokens,
        # aggregate: what the CONTAINER delivers. per-request median: what one caller feels.
        "container_tok_s": round(total_tokens / wall, 1) if wall else 0.0,
        "requests_per_s": round(len(per_request) / wall, 3) if wall else 0.0,
        "per_request_median_tok_s": round(rates[len(rates) // 2], 1) if rates else 0.0,
        "per_request_min_tok_s": round(rates[0], 1) if rates else 0.0,
        "latency_p50": pct(lat, 0.50),
        "latency_p95": pct(lat, 0.95),
        "latency_p99": pct(lat, 0.99),
        "ttft_p50": pct(ttfts, 0.50),
        "ttft_p95": pct(ttfts, 0.95),
    }


def _default_plan(max_num_seqs: int) -> list[dict]:
    """load cells spanning below, at, and above the sequence cap, at both n=1 and n=4.

    above-cap cells are the point: they are where engine-internal queueing shows up as flat
    aggregate throughput with falling per-request rate.
    """

    plan: list[dict] = [
        {"concurrency": c, "n": 1, "tag": f"n1c{c}"}
        for c in (1, max(2, max_num_seqs // 2), max_num_seqs, max_num_seqs * 2)
    ]
    # n=4 is the api's documented maximum fan-out; at concurrency == cap//4 the child count exactly
    # fills the engine, and at cap//2 it deliberately overruns it.
    plan.extend(
        {"concurrency": c, "n": 4, "tag": f"n4c{c}"}
        for c in (max(1, max_num_seqs // 4), max(2, max_num_seqs // 2))
    )
    return plan


@app.local_entrypoint()
def probe(
    gpu: str = "B200",
    base_model: str = "Qwen/Qwen3.5-9B",
    max_num_seqs: int = 16,
    plan: str = "",
) -> None:
    """boot exactly one engine, run the load plan against it, print json.

    ``plan`` is optional json (a list of load cells); omitted, a default sweep spanning the
    sequence cap at n=1 and n=4 is used.
    """

    cells = json.loads(plan) if plan else _default_plan(max_num_seqs)
    fn = {"B200": probe_b200, "B300": probe_b300}[gpu.upper()]
    out = fn.remote(base_model, max_num_seqs, cells)
    print("BENCH_JSON_START")
    print(json.dumps(out, indent=2, default=str))
    print("BENCH_JSON_END")
