"""Catalog facts and provider-aware pricing accessors for the cost model."""

from __future__ import annotations

from flash.catalog import MODELS, ModelInfo
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    # A10: 125 TFLOPS dense bf16 tensor (NVIDIA spec); Lambda-only 24 GB class, else defaults to 100.
    "A10": 125.0,
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    "A100 PCIe": 312.0,
    "A100 SXM": 312.0,
    # A100 SXM 40GB: same SMs/tensor cores as the 80GB A100 SXM, less HBM only.
    # Without this, 33-40 GB Lambda/Vast quotes fall back to _DEFAULT_TFLOPS.
    "A100 SXM 40GB": 312.0,
    "H100": 990.0,
    # H200: same SMs/tensor cores as H100, more HBM only.
    "H200": 990.0,
    "RTX Pro 6000": 250.0,
    # B200: 2.25 PFLOPS bf16 dense (NVIDIA spec); prevents ~10x cost over-estimate vs _DEFAULT_TFLOPS.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def gpu_hourly_usd(
    name: str,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    min_vram_gb: int = 0,
    exact_type: str = "",
) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    When ``provider`` is ``lambda`` or ``vast`` and the class is offered there, price it through that
    provider's pricing module (live with a static fallback); otherwise use the RunPod static rate.

    ``max_wall_seconds`` (>0) is threaded into the Vast live market so a duration-bound quote prices
    against offers that outlast the run, not a short-lived one filtered out at launch.

    ``min_vram_gb`` (>0) floors the Vast market search at the job's required VRAM — the SAME floor
    ``pick_gpu`` selected under — so a high-VRAM class isn't crowded off the price-sorted page and
    misquoted on the static fallback (selection/quote parity).
    """
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers import get_provider

        return get_provider("lambda").hourly_rate(name)
    if p == "vast" and info.vast_name:
        # Vast is a live market whose rates differ materially from RunPod's static ones, so price a
        # provider="vast" quote through the Vast pricing module (live + static fallback).
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(
            name,
            max_wall_seconds=max_wall_seconds,
            min_vram_gb=min_vram_gb,
            exact_type=exact_type,
        )
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int, *, provider: str | None = None, max_wall_seconds: float = 0.0
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``, ranked by static $/hr.

    No pin; every fitting class is eligible, validated or not. NOTE this is intentionally
    gate-free: the submit-time allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than the one priced here. ``provider`` restricts
    candidates to what it can provision. ``max_wall_seconds`` (>0) prices the Vast market against
    offers that outlast the run, so a long-run quote doesn't SELECT a class on the strength of a
    short-lived offer that won't survive to launch.
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    # Rank by the rate on the REQUESTED provider, not the RunPod nominal. For Vast, fetch the live offer
    # map ONCE (a duration-bound query bypasses the per-call cache, so pricing per candidate would fire N
    # identical market fetches) and, when reachable, restrict to classes that actually have a rentable
    # offer under the wall cap and rank by LIVE price — else a cheaper class with no surviving offer gets
    # selected/quoted on a static rate the launch path would never rent. Static fallback when offline.
    if (provider or "").strip().lower() == "vast":
        from flash.providers.vast.pricing import live_offer_rates

        live = live_offer_rates(max_wall_seconds=max_wall_seconds, min_vram_gb=required_vram_gb)
        rentable = [g for g in candidates if g.name in live] if live else []
        if rentable:
            candidates = rentable

            def _rate(g: GpuClass) -> float:
                return live[g.name]
        else:

            def _rate(g: GpuClass) -> float:
                return g.hourly_usd
    else:

        def _rate(g: GpuClass) -> float:
            return gpu_hourly_usd(g.name, provider=provider, max_wall_seconds=max_wall_seconds)

    best = min(candidates, key=lambda g: (_rate(g), g.vram_gb, g.name))
    return best.name


def _catalog_model_info(model_id: str) -> ModelInfo:
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info


def total_params_b(model_id: str, revision: str = "") -> float:
    """Total parameter count (billions) for a catalog model.

    when a revision is pinned, size the pinned commit (validated against the catalog, fail-closed)
    so setup/save cost tracks the weights the worker actually loads, not the default-revision count.
    """
    info = _catalog_model_info(model_id)
    if revision:
        from flash.engine.vram import _validated_revision_geometry

        params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
        return params_b
    return info.params_b


def active_params_b(model_id: str) -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM."""
    info = _catalog_model_info(model_id)
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, revision: str = "") -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id, revision) * 2.0


# ~1s mid-range default across grader types (regex ~0.01s to LLM judge ~3s).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION


# Fireworks echo-scoring round-trip per completion (wall time, concurrency-bound like reward grading).
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model.

    Routes through resolve_teacher, the single OPD-teacher resolver, whose recipe.TEACHER_MODELS is
    the one source of teacher prices. ``teacher_model`` is the Fireworks model id chosen via ``[train]
    teacher_model``, or "" for the default GLM 5.2 teacher. Teacher pricing is static, offline, and
    credential-free. OPD echo-scores completions (max_tokens=0), so only the input column is billed,
    but both are returned. An unsupported value falls back defensively to the default rate."""
    from flash.engine.recipe import resolve_teacher

    try:
        return resolve_teacher(teacher_model).usd_per_1m
    except ValueError:
        return resolve_teacher("").usd_per_1m


def teacher_token_cost_usd(
    input_tokens: float, output_tokens: float = 0.0, teacher_model: str = ""
) -> float:
    """External teacher-API dollar cost for a token count. Deterministic; no network."""
    inp, outp = teacher_price_per_1m(teacher_model)
    return (max(0.0, input_tokens) * inp + max(0.0, output_tokens) * outp) / 1_000_000.0


def teacher_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion teacher-scoring latency (s): the explicit override, else the average."""
    if override is not None:
        return max(0.0, override)
    return AVG_TEACHER_SECONDS_PER_COMPLETION
