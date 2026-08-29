"""Isolated Modal app measuring the hosted per-model capacity envelope.

Run:
    modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode canary
    modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode sweep

Each model is measured on ITS OWN production tier (L40S / H100 / H200), not on one shared card, so
the envelope describes the capacity a customer actually gets rather than a hypothetical uniform
fleet.

ISOLATION CONTRACT. This app is deliberately NOT ``flash/serving/app/modal_app.py``:

* Its own ``APP_NAME``. That app hardcodes ``freesolo-lora-serving``, the live production deployment;
  deploying it for a benchmark would replace production.
* No router, no ASGI front door, no custom domain. The production router mandates a durable usage
  outbox that settles billing against the real backend, so driving load through it would write
  synthetic usage into production billing.
* No Supabase, no backend URL, no internal key. Only ``HF_TOKEN`` is forwarded, and only because
  downloading gated weights requires it.
* ``max_containers=1`` and ``min_containers=0``. The envelope is per-card capacity, so autoscaling
  must not silently add a second GPU mid-measurement and inflate the numbers.
* Base models only. No adapter is ever registered.

It DOES reuse the production engine, image, and generation path, so what is measured is the shipped
serving stack rather than a benchmark-only reimplementation.
"""

# NO `from __future__ import annotations` here, deliberately. It makes every annotation a string, and
# Modal's class-parameter validator resolves `base_model: str` through the real type object; under
# postponed evaluation it receives the STRING "str" and dies with
# `AttributeError: 'str' object has no attribute '__name__'` at decoration time. flash/serving/app/
# modal_app.py omits the import for the same reason.

import json
import os
from pathlib import Path
from typing import Any

import modal

REPO_DIR = Path(__file__).resolve().parent.parent

# Distinct from APP_NAME in flash/serving/app/modal_app.py. If these ever collide, deploying the
# benchmark would overwrite production serving.
APP_NAME = "flash-bench-hosted-capacity"
# Separate from the production HF cache volume so a benchmark run cannot evict production's warm
# weights or compile artifacts.
CACHE_VOLUME_NAME = "flash-bench-hosted-capacity-cache"
CACHE_MOUNT = "/vol/bench-cache"

# The 35B needs ~17 min of engine init before it serves a token; the ceiling bounds a stuck boot.
STARTUP_TIMEOUT_SECONDS = 2700
# A near-32k bucket runs its whole concurrency grid on one boot, so the container must outlive the
# sum of the grid's cells plus teardown, not just one cell.
TIMEOUT_SECONDS = 7200
# Short: the container is torn down as soon as its lane finishes, and an idle benchmark card is pure
# waste. Production's window is sized to amortize cold boots across real traffic; there is none here.
SCALEDOWN_WINDOW_SECONDS = 120

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _HF_TOKEN})] if _HF_TOKEN else []

image = (
    modal.Image.from_registry(
        # Digest-pinned, matching flash/serving/app/modal_app.py: the measured stack must be the
        # shipped one, so the image is copied rather than independently chosen.
        "nvidia/cuda:13.0.0-devel-ubuntu22.04@sha256:1470d2d7904fac4e5cb3bdfd4993305c46d3ee76deb0213eaaf248e5cf9c7400",
        add_python="3.12",
    )
    .apt_install("build-essential", "git", "ninja-build")
    .pip_install_from_pyproject(
        str(REPO_DIR / "pyproject.toml"),
        optional_dependencies=["serve-runtime", "serving"],
    )
    .add_local_file(
        str(REPO_DIR / "docker" / "patch_vllm_moe_lora.py"),
        remote_path="/root/patch_vllm_moe_lora.py",
        copy=True,
    )
    .run_commands(
        "python /root/patch_vllm_moe_lora.py && "
        "python /root/patch_vllm_moe_lora.py --verify && "
        "rm /root/patch_vllm_moe_lora.py"
    )
    .env(
        {
            "HF_HOME": CACHE_MOUNT,
            "HF_HUB_CACHE": f"{CACHE_MOUNT}/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_MOUNT}/transformers",
            "VLLM_CACHE_ROOT": f"{CACHE_MOUNT}/vllm",
            "HF_HUB_DISABLE_XET": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_MOE_USE_DEEP_GEMM": "0",
            # Marks this container as the benchmark so any shared code path can refuse to treat it
            # as a production serving replica.
            "FLASH_BENCH_CAPACITY": "1",
        }
    )
    .add_local_python_source("flash")
)

app = modal.App(APP_NAME, image=image)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

from flash.serving.bench.catalog import (  # noqa: E402
    BENCH_MODELS,
    bench_engine_overrides_for,
    bench_gpu_for,
)
from flash.serving.src.engine.lora_engine import _LoraEngineImpl  # noqa: E402


class _BenchEngineImpl(_LoraEngineImpl):
    """Production engine, with self-healing disabled.

    ``_load`` is NOT overridden. The bench catalog delegates to ``model_config`` and returns
    production's exact overrides, so a copied loader would only be a second implementation free to
    drift from the one being measured -- which is precisely what it did: an earlier copy still used
    the pre-rename ``dict[str, Lock]`` adapter-lock keys.

    The inherited loader's adapter hydration is a no-op here rather than a failure: this app carries
    no platform credentials, ``_load_adapters_for_base`` swallows the resulting error and returns an
    empty list, and the cached-LoRA preload then iterates an empty registry. No adapter is registered
    at any point, so nothing on the LoRA path is measured or needed.
    """

    def _self_heal_if_dead(self, _context: str) -> None:
        """Never drain the container mid-measurement.

        A dead engine must surface as failed requests in the evidence, because a self-heal here would
        silently discard the very failures the error rate is meant to report.
        """
        return


def _bench_class_name(gpu: str) -> str:
    """Deterministic, Modal-safe class name for one tier -- 'H100!' -> 'BenchEngine_H100_'.

    The GPU alone is the identity here (unlike production, which also keys on max_inputs) because
    EVERY bench class is max_inputs=1: concurrency is generated inside the container.
    """
    return "BenchEngine_" + "".join(ch if ch.isalnum() else "_" for ch in gpu)


def _build_bench_engine(gpu: str, class_name: str) -> Any:
    """Register one Modal ``@app.cls`` bench engine pinned to ``gpu``.

    Modal fixes a class's GPU at decoration time, so the three hosted tiers need three classes. As in
    flash/serving/app/modal_app.py, class identity is fixed BEFORE the decorators run: ``modal.
    concurrent`` returns a wrapper holding the user class separately, so renaming afterwards would
    rename the wrapper and leave every tier registered under a ``<locals>`` qualname Modal rejects.
    """

    class _Engine(_BenchEngineImpl):
        base_model: str = modal.parameter()

        @modal.enter()
        async def load(self) -> None:
            await self._load()

        @modal.method()
        async def probe(self) -> dict[str, Any]:
            """Hardware, kernel-path, and KV-cache provenance for this container."""
            from flash.serving.bench.probe import probe_all

            return probe_all(self.base_model, self)

        @modal.method()
        async def warmup(self, requests: int = 5) -> dict[str, Any]:
            """Sequential short requests so compile and graph capture finish before measuring.

            Their timings are reported but never merged into the envelope: the first request after
            boot pays one-time costs that would distort every percentile it entered.
            """
            return await _run_warmup(self, requests)

        @modal.method()
        async def run_bucket(
            self,
            bucket_name: str,
            concurrency_points: list[int],
            block: int = 0,
        ) -> dict[str, Any]:
            """Measure one bucket across its concurrency grid on this container."""
            return await _run_bucket(self, bucket_name, concurrency_points, block)

    _Engine.pinned_gpu = gpu
    _Engine.__name__ = class_name
    _Engine.__qualname__ = class_name
    globals()[class_name] = _Engine
    engine = app.cls(
        gpu=gpu,
        secrets=_secrets,
        volumes={CACHE_MOUNT: cache_volume},
        timeout=TIMEOUT_SECONDS,
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        # Per-card capacity: autoscaling would add a second GPU mid-measurement and inflate it.
        min_containers=0,
        max_containers=1,
        # Every Modal call lands on ONE container. Concurrency is generated inside the container by
        # the driver, so Modal must never spread the offered load across replicas.
    )(modal.concurrent(max_inputs=1, target_inputs=1)(_Engine))
    globals()[class_name] = engine
    return engine


def _distinct_bench_gpus() -> list[str]:
    """Distinct tiers across the bench catalog, order-stable."""
    tiers: dict[str, None] = {}
    for base_model in BENCH_MODELS:
        tiers.setdefault(bench_gpu_for(base_model), None)
    return list(tiers)


ENGINE_BY_GPU: dict[str, Any] = {
    gpu: _build_bench_engine(gpu, _bench_class_name(gpu)) for gpu in _distinct_bench_gpus()
}


def _engine_for(base_model: str) -> Any:
    """The Modal bench class to run ``base_model`` on (its tier's class)."""
    return ENGINE_BY_GPU[bench_gpu_for(base_model)]


async def _run_warmup(engine: Any, requests: int) -> dict[str, Any]:
    """Sequential warmups on ``engine``, reported separately from the envelope."""
    import time

    from flash.serving.bench.driver import run_request
    from flash.serving.bench.workload import BUCKETS_BY_NAME, fit_prompt_to_tokens

    bucket = BUCKETS_BY_NAME["short_interactive"]
    origin = time.monotonic()
    out = []
    exact = 0
    for index in range(requests):
        uid = f"warmup-{index}"
        messages, exact = fit_prompt_to_tokens(engine.tokenizer, uid, bucket.target_input_tokens)
        record = await run_request(
            engine,
            engine.base_model,
            messages,
            bucket.max_output_tokens,
            uid,
            bucket="warmup",
            concurrency=1,
            block=0,
            origin=origin,
        )
        out.append(record.to_json())
    return {"warmups": out, "assembled_prompt_tokens": exact}


async def _run_bucket(
    engine: Any,
    bucket_name: str,
    concurrency_points: list[int],
    block: int,
) -> dict[str, Any]:
    """One bucket's whole concurrency grid on one already-booted engine.

    The boot dominates cost (~960s of ~1000s per cell in the prior campaign), so the grid runs
    against a single engine rather than paying a boot per point.
    """
    from flash.serving.bench.driver import run_cell
    from flash.serving.bench.metrics import summarize_curve
    from flash.serving.bench.probe import probe_all
    from flash.serving.bench.workload import BUCKETS_BY_NAME

    bucket = BUCKETS_BY_NAME[bucket_name]
    cells = []
    records = []
    for concurrency in concurrency_points:
        # Depth floors come from the bucket, so a near-32k cell is not asked for the same attempt
        # count as a short turn.
        result, cell_records = await run_cell(
            engine,
            engine.tokenizer,
            engine.base_model,
            bucket,
            concurrency,
            block,
        )
        cells.append(result)
        records.extend(cell_records)
        print(
            f"[bench] {engine.base_model} {bucket_name} c={concurrency} "
            f"rps={result.successful_rps:.3f} tok/s={result.output_tokens_per_second:.1f} "
            f"err={result.error_rate:.3f} bound={result.error_rate_upper_bound:.3f} "
            f"resolved={result.error_bound_resolved} degraded={result.degraded}",
            flush=True,
        )
        # Stop climbing once the engine is failing outright: further points would spend GPU time
        # measuring progressively deeper failure, which the envelope does not need.
        if result.succeeded == 0:
            print(f"[bench] halting {bucket_name}: no successes at c={concurrency}", flush=True)
            break
    return {
        "base_model": engine.base_model,
        "bucket": bucket_name,
        "block": block,
        "cells": [cell.to_json() for cell in cells],
        "curve": summarize_curve(cells),
        "records": [record.to_json() for record in records],
        "provenance": probe_all(engine.base_model, engine),
    }


def _write_artifact(payload: dict[str, Any], name: str) -> Path:
    out_dir = Path(os.environ.get("BENCH_OUT_DIR", "/tmp/flash-bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] wrote {path}", flush=True)
    return path


def _run_canary(base_model: str, engine: Any, expected_gpu: str) -> dict[str, Any]:
    """Boot, verify the card and kernel path, and warm up. The cheap gate before any sweep."""
    from flash.serving.bench.probe import gpu_matches

    probe = engine.probe.remote()
    print(json.dumps(probe, indent=2), flush=True)
    if not gpu_matches(probe, expected_gpu):
        raise RuntimeError(f"expected {expected_gpu}, got {(probe.get('gpu') or {}).get('name')!r}")
    cutlass = (probe.get("gdn_prefill") or {}).get("cutlass") or {}
    if cutlass.get("checked") and not cutlass.get("intact"):
        # A warning, not a failure: the campaign measures the shipped dev runtime as it is. The
        # report carries this label so nobody reads the numbers as the FlashInfer fast path.
        print(
            "[bench] WARNING: cutlass cu13 install is NOT intact -> GDN prefill falls back to "
            "Triton on Blackwell. Results describe the SLOW prefill path.",
            flush=True,
        )
    warm = engine.warmup.remote(5)
    print(json.dumps(warm, indent=2), flush=True)
    return {"probe": probe, "warmup": warm}


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-9B",
    mode: str = "canary",
    bucket: str = "",
    block: int = 0,
) -> None:
    """Drive one model's benchmark lane on its own production tier.

    ``canary`` boots the engine, verifies the card, reports the GDN prefill kernel, and runs a
    handful of warmups. It is the cheap gate that must pass before any sweep spends money.
    """
    from flash.serving.bench.workload import BUCKETS, concurrency_grid, workload_checksum

    expected_gpu = bench_gpu_for(base_model)
    engine = _engine_for(base_model)(base_model=base_model)
    gate = _run_canary(base_model, engine, expected_gpu)

    if mode == "canary":
        _write_artifact(
            {
                "base_model": base_model,
                "gpu": expected_gpu,
                "mode": "canary",
                "probe": gate["probe"],
                "warmup": gate["warmup"],
                "workload_checksum": workload_checksum(),
            },
            f"canary-{base_model.replace('/', '_')}.json",
        )
        return

    overrides = bench_engine_overrides_for(base_model)
    grid = list(concurrency_grid(int(overrides.get("max_num_seqs", 8))))
    buckets = [bucket] if bucket else [b.name for b in BUCKETS]
    results = []
    for name in buckets:
        payload = engine.run_bucket.remote(name, grid, block)
        results.append(payload)
        _write_artifact(payload, f"sweep-{base_model.replace('/', '_')}-{name}-b{block}.json")
    _write_artifact(
        {
            "base_model": base_model,
            "gpu": expected_gpu,
            "mode": mode,
            "grid": grid,
            "workload_checksum": workload_checksum(),
            "buckets": [payload["curve"] for payload in results],
        },
        f"summary-{base_model.replace('/', '_')}-b{block}.json",
    )
