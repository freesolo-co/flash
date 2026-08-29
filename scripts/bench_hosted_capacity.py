"""Isolated Modal app measuring the hosted per-model capacity envelope.

Run (``--ceiling-usd`` is REQUIRED; ``main`` exits on the zero default, so a command without it
cannot run. The figures below are worst-case RESERVATIONS, deliberately far above expected spend,
and each also clears the 80% submission stop ``reserve`` enforces -- see
docs/serving-capacity-envelope.md):
    modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode canary \
        --ceiling-usd 7
    modal run scripts/bench_hosted_capacity.py --base-model Qwen/Qwen3.5-9B --mode sweep \
        --bucket short_interactive --ceiling-usd 16

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
import time
import uuid
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
# A bucket runs its whole concurrency grid on one boot, so the container must outlive the sum of the
# grid's cells AND their drains, not just the measured windows. The widest preregistered lane is
# near_32k: 6 concurrency points x (600s window + 900s drain) = 9000s, and a 420s bucket still needs
# 7920s. A 7200s ceiling terminated `run_bucket` before the most expensive bucket could persist its
# artifact, so the run paid for the whole grid and published nothing. Derived below from the
# preregistered bounds rather than typed, so widening a bucket cannot silently reintroduce the gap.
TIMEOUT_HEADROOM_SECONDS = 900
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
from flash.serving.bench.driver import (  # noqa: E402
    REQUEST_TIMEOUT_SECONDS,
    prompt_fit_seconds_bound,
)
from flash.serving.bench.workload import BUCKETS, concurrency_grid  # noqa: E402
from flash.serving.src.engine.lora_engine import _LoraEngineImpl  # noqa: E402

# How many warmup requests a cold container issues. Shared by `_run_canary`, `_ensure_warm`, the
# method timeout and both budget estimators, so the bound reserved and the bound enforced cannot
# drift from the number actually issued. Defined here because `_worst_case_bucket_seconds` runs at
# import time to derive TIMEOUT_SECONDS.
CANARY_WARMUP_REQUESTS = 5


def _worst_case_bucket_seconds() -> float:
    """Longest a single ``run_bucket`` call can legitimately take, from the preregistered bounds.

    Every concurrency point pays its bucket's ``max_seconds`` window plus a drain bounded by
    ``REQUEST_TIMEOUT_SECONDS``. The widest grid across all bench models decides the ceiling, since
    one ``timeout`` covers every tier's class.
    """
    points = max(
        len(concurrency_grid(int(bench_engine_overrides_for(base_model).get("max_num_seqs", 8))))
        for base_model in BENCH_MODELS
    )
    cells = points * (max(bucket.max_seconds for bucket in BUCKETS) + REQUEST_TIMEOUT_SECONDS)
    # A bucket landing on a COLD replacement container warms it first -- the path `_ensure_warm`
    # exists to handle -- and those warmups are SEQUENTIAL, each bounded by REQUEST_TIMEOUT_SECONDS.
    # Timing only the cells would kill that legitimate path mid-flight, and because the timeout fires
    # before `run_bucket` returns, the bucket's artifact is never written: the run loses the
    # measurement it already paid for.
    return cells + REQUEST_TIMEOUT_SECONDS * CANARY_WARMUP_REQUESTS


# Derived, never typed: widening a bucket or the concurrency grid raises this automatically instead
# of silently reintroducing a ceiling below the work the grid is allowed to do.
TIMEOUT_SECONDS = int(_worst_case_bucket_seconds() + TIMEOUT_HEADROOM_SECONDS)


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
            invocation: str = "",
        ) -> dict[str, Any]:
            """Measure one bucket across its concurrency grid on this container."""
            return await _run_bucket(self, bucket_name, concurrency_points, block, invocation)

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


def _require_healthy_warmup(warm: Any, label: str) -> None:
    """Raise unless every warmup record succeeded.

    `run_request` converts exceptions, timeouts, malformed streams and cache-verification failures
    into records with `ok=False` rather than raising, so a warmup call returns NORMALLY even when
    the generation path is entirely broken. Without this check a warmup would wave through paid work
    already known to be invalid -- and that is true of a replacement container's self-warmup just as
    much as of the canary gate, so both paths share this one predicate.
    """
    records = warm.get("warmups") if isinstance(warm, dict) else None
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{label} warmup returned no records: {warm!r}")
    failed = [record for record in records if not record.get("ok")]
    if failed:
        reasons = sorted({str(record.get("error")) for record in failed})
        raise RuntimeError(
            f"{label} warmup failed {len(failed)}/{len(records)} requests ({', '.join(reasons)}); "
            "refusing to start paid work against a broken generation path"
        )


async def _ensure_warm(engine: Any) -> dict[str, Any] | None:
    """Warm THIS container if it has not been warmed yet, and report whether it had to.

    `max_containers=1` caps simultaneous replicas; it does NOT pin successive remote calls to one
    container. Modal may replace or preempt the container between the canary and a bucket, so a
    sweep whose gate passed on container A can measure its first cells on a freshly booted
    container B -- paying compile and lazy-workspace costs inside the measured window while the
    report says the warmup gate passed.

    The flag lives on the container instance, so it is FALSE exactly when this process has not run
    a warmup, which is the condition that matters. A returned dict means this container was cold
    and warmed itself; None means it was already warm.
    """
    if getattr(engine, "_bench_warmed", False):
        return None
    warm = await _run_warmup(engine, CANARY_WARMUP_REQUESTS)
    # A cold replacement container is exactly where an unhealthy engine surfaces, so its warmup gets
    # the same check the canary gate applies rather than being trusted because the canary passed on
    # a DIFFERENT container.
    _require_healthy_warmup(warm, "replacement-container")
    engine._bench_warmed = True
    return warm


async def _run_warmup(engine: Any, requests: int) -> dict[str, Any]:
    """Sequential warmups on ``engine``, reported separately from the envelope."""
    import time

    from flash.serving.bench.driver import run_request
    from flash.serving.bench.workload import BUCKETS_BY_NAME, fit_prompt_to_tokens

    bucket = BUCKETS_BY_NAME["short_interactive"]
    # This container has now paid its one-time costs, so a bucket landing here need not repeat them.
    engine._bench_warmed = True
    origin = time.monotonic()
    out = []
    exact = 0
    # Prompts are derived from the UID, so a fixed `warmup-{i}` reissued the SAME five prompts on
    # every invocation. Within the 120s scaledown window the container survives, and the second
    # canary hits a retained prefix cache -- which the driver correctly scores as
    # ERROR_CACHE_CONTAMINATED, refusing the sweep even though generation was healthy. A nonce per
    # invocation makes each warmup prompt request-unique from its first token, like every other
    # prompt the harness issues.
    nonce = uuid.uuid4().hex[:12]
    for index in range(requests):
        uid = f"warmup-{nonce}-{index}"
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
            # The warmup is the gate that runs before any sweep, so it is the cheapest place to
            # discover that the engine sizes prompts differently than the fitter does.
            expected_prompt_tokens=exact,
        )
        out.append(record.to_json())
    return {"warmups": out, "assembled_prompt_tokens": exact}


async def _run_bucket(
    engine: Any,
    bucket_name: str,
    concurrency_points: list[int],
    block: int,
    invocation: str = "",
) -> dict[str, Any]:
    """One bucket's whole concurrency grid on one already-booted engine.

    The boot dominates cost (~960s of ~1000s per cell in the prior campaign), so the grid runs
    against a single engine rather than paying a boot per point.
    """
    from flash.serving.bench.catalog import bench_catalog_summary
    from flash.serving.bench.driver import run_cell
    from flash.serving.bench.metrics import summarize_curve
    from flash.serving.bench.probe import probe_all
    from flash.serving.bench.workload import BUCKETS_BY_NAME

    bucket = BUCKETS_BY_NAME[bucket_name]
    # A replacement container measures cold otherwise; see `_ensure_warm`. Recorded in the payload
    # so a reader can tell a bucket that inherited the canary's warm container from one that had to
    # warm itself, rather than having to assume every bucket ran on the gated container.
    cold_start_warmup = await _ensure_warm(engine)
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
            invocation=invocation,
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
        #
        # Gated on the IN-WINDOW count, not `succeeded`. A cell whose requests all complete during
        # the drain has zero steady-state throughput -- it is already classified degraded and its
        # published rates are 0 -- yet `succeeded` stays positive, so climbing continued and bought
        # another full window-and-drain tail per remaining point.
        if result.succeeded_in_window == 0:
            print(
                f"[bench] halting {bucket_name}: no steady-state successes at c={concurrency}",
                flush=True,
            )
            break
    return {
        "base_model": engine.base_model,
        "bucket": bucket_name,
        "block": block,
        # Recorded so a rerun's artifact is distinguishable from the run it replaced, and so the
        # prompts it sent can be reconstructed exactly.
        "invocation": invocation,
        "cells": [cell.to_json() for cell in cells],
        "curve": summarize_curve(cells),
        "records": [record.to_json() for record in records],
        "provenance": probe_all(engine.base_model, engine),
        # Non-None when THIS container was cold and warmed itself, i.e. it is not the container the
        # canary gated. None means it inherited the gated warm container.
        "cold_start_warmup": cold_start_warmup,
        # The resolved engine shape, so a cell's numbers stay interpretable after catalog drift.
        "engine_catalog": next(
            (row for row in bench_catalog_summary() if row["base_model"] == engine.base_model),
            None,
        ),
    }


def _write_artifact(payload: dict[str, Any], name: str) -> Path:
    out_dir = Path(os.environ.get("BENCH_OUT_DIR", "/tmp/flash-bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] wrote {path}", flush=True)
    return path


def _require_resolved_gdn_backend(probe: dict[str, Any]) -> None:
    """Raise unless the probe established WHICH GDN prefill backend the engine chose.

    The harness documents the kernel path as a publication gate, but the gate only checked card
    identity and Cutlass integrity. `probe_gdn_backend` records `resolved=None` whenever the
    resolver is absent, its signature moved, or the served config would not load -- and none of
    those stop the engine from booting, serving, and billing. So a healthy warmup waved through a
    sweep whose kernel path was never established, and the envelope would be published without the
    one label that makes a Blackwell number interpretable.

    Unknown is not a backend. Refuse here, where the cost is one canary, rather than after a paid
    sweep whose numbers cannot be attributed to a kernel.
    """
    gdn = probe.get("gdn_prefill") or {}
    if gdn.get("resolved"):
        return
    detail = str(gdn.get("reason") or "probe recorded no reason")
    if gdn.get("resolver_signature_mismatch"):
        detail = f"resolver signature mismatch: {detail}"
    raise RuntimeError(
        "GDN prefill backend is unresolved "
        f"({detail}); refusing to start paid work whose kernel path cannot be labelled"
    )


def _run_canary(base_model: str, engine: Any, expected_gpu: str) -> dict[str, Any]:
    """Boot, verify the card and kernel path, and warm up. The cheap gate before any sweep."""
    from flash.serving.bench.probe import gpu_matches

    probe = engine.probe.remote()
    print(json.dumps(probe, indent=2), flush=True)
    if not gpu_matches(probe, expected_gpu):
        raise RuntimeError(f"expected {expected_gpu}, got {(probe.get('gpu') or {}).get('name')!r}")
    _require_resolved_gdn_backend(probe)
    cutlass = (probe.get("gdn_prefill") or {}).get("cutlass") or {}
    if cutlass.get("checked") and not cutlass.get("intact"):
        # A warning, not a failure: the campaign measures the shipped dev runtime as it is. The
        # report carries this label so nobody reads the numbers as the FlashInfer fast path.
        print(
            "[bench] WARNING: cutlass cu13 install is NOT intact -> GDN prefill falls back to "
            "Triton on Blackwell. Results describe the SLOW prefill path.",
            flush=True,
        )
    warm = engine.warmup.remote(CANARY_WARMUP_REQUESTS)
    print(json.dumps(warm, indent=2), flush=True)
    # `run_request` converts exceptions, timeouts, malformed streams and cache-verification failures
    # into records with ok=False rather than raising, so `warmup.remote` returns normally even when
    # the generation path is entirely broken. Without this check the cheap gate would wave through an
    # expensive sweep already known to be invalid.
    _require_healthy_warmup(warm, "canary")
    return {"probe": probe, "warmup": warm}


# Conservative per-lane GPU-second estimates, used to reserve against the ceiling BEFORE any
# allocation. Deliberately generous: a reservation that overestimates stops a lane early, while one
# that underestimates lets a lane overspend, and only the second failure costs money.
MODES = ("canary", "sweep")


def _canary_gpu_seconds_estimate() -> float:
    """Worst-case billed GPU-seconds for a canary lane.

    A flat constant here was the same defect the sweep estimator had, in a lane the sweep fix did
    not touch: the canary bills for a boot Modal allows to run to `STARTUP_TIMEOUT_SECONDS`, plus
    `CANARY_WARMUP_REQUESTS` SEQUENTIAL warmups each of which the driver allows to run to
    `REQUEST_TIMEOUT_SECONDS`. Reserving less accepts a ceiling the lane's own bounds permit it to
    exceed, which is precisely what `BudgetLedger` exists to prevent.

    The canary itself makes TWO remote calls -- `probe.remote()` then `warmup.remote()` -- and
    `max_containers=1` caps simultaneous replicas without binding successive calls to one container.
    So the warmup can land on a replacement that pays its own cold boot, exactly the case
    `_ensure_warm` exists to handle on the sweep side. Both boots are reserved here; the warmup
    requests are counted once because a replacement runs the same five, not a second set.
    """
    calls = 2
    return float(STARTUP_TIMEOUT_SECONDS) * calls + REQUEST_TIMEOUT_SECONDS * CANARY_WARMUP_REQUESTS


def _sweep_gpu_seconds_estimate(base_model: str, selected: list[Any]) -> float:
    """Worst-case GPU-seconds for the selected buckets, from their own preregistered bounds.

    Deliberately an UPPER bound. A reservation is a spending authorization, so it must be wrong in
    the direction that refuses a run rather than the direction that overspends: every cell is priced
    at its bucket's `max_seconds`, even though a cell that meets its floors early exits sooner.

    EVERY bounded paid phase is reserved, not just the measured windows. A sweep bills for four
    things, and pricing only the third would let an accepted run exceed its own ceiling:

    1. the cold boot;
    2. the canary, which always runs before the sweep and whose 5 warmup requests are SEQUENTIAL,
       so worst case each one consumes its own `REQUEST_TIMEOUT_SECONDS`;
    3. each cell's measured window, bounded by the bucket's `max_seconds`, PLUS the prompt
       fitting that precedes it -- excluded from the window on purpose so tokenization cannot
       distort the measurement, but still executed on the rented GPU and therefore billed;
    4. each cell's DRAIN, which waits up to `REQUEST_TIMEOUT_SECONDS` for requests still in flight
       when the window closed. This is the largest omission: at 900s per cell it can exceed the
       measured time it follows, and it happens after every cell, not once per sweep.

    The grid width comes from the engine's real `max_num_seqs`, so it tracks the catalog rather than
    a hardcoded six.
    """
    overrides = bench_engine_overrides_for(base_model)
    points = len(list(concurrency_grid(int(overrides.get("max_num_seqs", 8)))))
    cells = points * len(selected)
    measured = sum(float(bucket.max_seconds) * points for bucket in selected)
    # Prompt fitting runs inside the container BEFORE each cell's window opens, so it is
    # billed but appears in no `max_seconds`. Per cell, not per bucket: every concurrency
    # point rebuilds its own pool.
    fitting = sum(prompt_fit_seconds_bound(bucket) * points for bucket in selected)
    drains = REQUEST_TIMEOUT_SECONDS * cells
    canary = REQUEST_TIMEOUT_SECONDS * CANARY_WARMUP_REQUESTS
    # The boot is reserved at the ceiling Modal actually allows a stuck boot to reach, not at a
    # typical observed boot. Same reasoning as the canary lane.
    boot = float(STARTUP_TIMEOUT_SECONDS)
    # Every bucket is a SEPARATE remote call, and `max_containers=1` caps simultaneous replicas
    # without pinning successive calls to one container -- which is exactly why `_ensure_warm`
    # exists. So each call can land on a replacement container and bill another cold boot plus
    # another sequential warmup. Reserving one boot for the whole campaign left that bounded,
    # handled, and entirely foreseeable path unfunded, so a sweep accepted under its ceiling could
    # bill past it once per selected bucket. Priced per call, since that is where the exposure is.
    replacements = (boot + canary) * len(selected)
    # The canary is TWO separately bootable remote calls -- `probe.remote()` then `warmup.remote()`
    # -- so a sweep makes `len(selected) + 2` of them, not `len(selected) + 1`. The canary lane
    # already prices both; this lane priced only the initial boot, under-reserving every sweep by a
    # whole `STARTUP_TIMEOUT_SECONDS` whenever the warmup landed on its own cold replacement. The
    # warmup requests are NOT doubled: a replacement runs the same five, not a second set.
    canary_replacement_boot = boot
    return boot + canary_replacement_boot + canary + measured + fitting + drains + replacements


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-9B",
    mode: str = "canary",
    bucket: str = "",
    block: int = 0,
    ceiling_usd: float = 0.0,
) -> None:
    """Drive one model's benchmark lane on its own production tier.

    ``canary`` boots the engine, verifies the card, reports the GDN prefill kernel, and runs a
    handful of warmups. It is the cheap gate that must pass before any sweep spends money.

    ``ceiling_usd`` is REQUIRED and has no usable default: every path below allocates a GPU, so the
    budget is reserved here, before the first remote call, rather than advertised and never
    consulted.
    """
    from flash.serving.bench.budget import BudgetLedger
    from flash.serving.bench.catalog import bench_catalog_summary
    from flash.serving.bench.workload import BUCKETS, concurrency_grid, workload_checksum

    # Checked before anything else: an unknown mode must not reach a remote call, because every one
    # of them allocates the model's GPU.
    if mode not in MODES:
        raise SystemExit(f"unknown --mode {mode!r}; expected one of {', '.join(MODES)}")
    if ceiling_usd <= 0:
        raise SystemExit(
            "--ceiling-usd is required and must be positive: this entrypoint allocates a GPU, so "
            "it refuses to run against an unauthorized budget"
        )
    # Submit-time, not on a rented GPU. `--bucket` is only indexed inside `run_bucket` on the
    # remote side, so a typo used to pay for a full cold boot and warmup before failing with no
    # measurement. Every argument that can be checked without a card is checked here.
    selected = [b for b in BUCKETS if not bucket or b.name == bucket]
    if bucket and not selected:
        raise SystemExit(
            f"unknown --bucket {bucket!r}; expected one of {', '.join(b.name for b in BUCKETS)}"
        )

    expected_gpu = bench_gpu_for(base_model)
    ledger = BudgetLedger(ceiling_usd=ceiling_usd)
    # Reserve the WORST CASE of the work actually selected, not a flat per-mode constant. A sweep
    # without `--bucket` runs every bucket, and each bucket's grid can burn `max_seconds` at every
    # concurrency point; a single 3600s reservation let a campaign whose own preregistered bounds
    # permit far more be accepted under a ceiling it cannot honour. The ceiling is only real if the
    # reservation covers the whole campaign before the first remote call.
    estimate = (
        _canary_gpu_seconds_estimate()
        if mode == "canary"
        else _sweep_gpu_seconds_estimate(base_model, selected)
    )
    # Raises BudgetExceeded rather than allocating. This is the line that makes the ceiling real.
    entry = ledger.reserve(f"{mode}:{base_model}", estimate, expected_gpu)
    # The reservation is the WORST CASE, deliberately generous. Publishing it as the cost would
    # overstate every campaign by the margin that makes the ceiling safe, so the lane is settled
    # against measured wall time before the artifact is written and `settled_usd` replaces the
    # reservation in `committed_usd`. Billing starts at the first remote call, so the clock starts
    # here rather than at process start, which would bill local argument checking.
    lane_started = time.monotonic()

    catalog = {row["base_model"]: row for row in bench_catalog_summary()}
    # The resolved engine shape travels WITH the numbers. Without it a published capacity figure
    # cannot be tied to the checkpoint, context limit, sequence cap, or LoRA allocation that
    # produced it, and catalog drift would silently reinterpret old results.
    provenance = catalog.get(base_model)

    engine = _engine_for(base_model)(base_model=base_model)
    gate = _run_canary(base_model, engine, expected_gpu)

    if mode == "canary":
        ledger.settle(entry, time.monotonic() - lane_started, note="measured canary wall")
        _write_artifact(
            {
                "base_model": base_model,
                "gpu": expected_gpu,
                "mode": "canary",
                "engine_catalog": provenance,
                "probe": gate["probe"],
                "warmup": gate["warmup"],
                "workload_checksum": workload_checksum(),
                "budget": ledger.to_json(),
            },
            f"canary-{base_model.replace('/', '_')}.json",
        )
        return

    overrides = bench_engine_overrides_for(base_model)
    grid = list(concurrency_grid(int(overrides.get("max_num_seqs", 8))))
    # One nonce per sweep invocation, keying every measured prompt HEADER. A retry at the same block
    # would otherwise re-send byte-identical prompts, and inside Modal's 120s scaledown the previous
    # container and its prefix cache are still alive: the driver would score those hits
    # ERROR_CACHE_CONTAMINATED and throw away a paid rerun whose engine was healthy. The filler body
    # is keyed separately and does not move, so the workload stays reproducible.
    invocation = uuid.uuid4().hex[:12]
    print(f"[bench] invocation nonce {invocation}", flush=True)
    results = []
    for name in [b.name for b in selected]:
        payload = engine.run_bucket.remote(name, grid, block, invocation)
        results.append(payload)
        _write_artifact(payload, f"sweep-{base_model.replace('/', '_')}-{name}-b{block}.json")
    ledger.settle(entry, time.monotonic() - lane_started, note="measured sweep wall")
    _write_artifact(
        {
            "base_model": base_model,
            "gpu": expected_gpu,
            "mode": mode,
            "grid": grid,
            "invocation": invocation,
            "engine_catalog": provenance,
            "workload_checksum": workload_checksum(),
            "buckets": [
                {"bucket": payload["bucket"], "curve": payload["curve"]} for payload in results
            ],
            "budget": ledger.to_json(),
        },
        f"summary-{base_model.replace('/', '_')}-b{block}.json",
    )
