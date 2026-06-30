"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
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


def gpu_hourly_usd(name: str, provider: str | None = None, max_wall_seconds: float = 0.0) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    When ``provider`` is ``lambda`` or ``vast`` and the class is offered there, price it through that
    provider's pricing module (live with a static fallback); otherwise use the RunPod static rate.

    ``max_wall_seconds`` (>0) is threaded into the Vast live market so a duration-bound quote prices
    against offers that outlast the run, not a short-lived one filtered out at launch (Codex MtzrI).
    """
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers.lambdalabs.pricing import hourly_rate

        return hourly_rate(name)
    if p == "vast" and info.vast_name:
        # Vast is a live verified-datacenter market — its rates differ materially from RunPod's static
        # ones, so a provider="vast" quote must price through the Vast pricing module (live + static
        # fallback), not fall back to GpuClass.hourly_usd (the RunPod rate).
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(name, max_wall_seconds=max_wall_seconds)
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
    short-lived offer that won't survive to launch (Codex MtzrI).
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    # Rank by the rate on the REQUESTED provider so a provider-specific quote picks that provider's
    # cheapest fit (not the cheapest by the RunPod nominal rate). For Vast, fetch the live offer map
    # ONCE (a duration-bound Vast query bypasses the per-call cache per Codex MtzrI, so pricing each
    # candidate via gpu_hourly_usd() inside the key would fire one identical full market fetch per
    # fitting class — N redundant queries; Copilot). When the market is reachable, restrict the
    # candidates to classes that ACTUALLY have a rentable offer under the wall cap and rank by their
    # LIVE price: a cheaper class with no surviving offer must not be selected (and quoted) on its
    # static (RunPod) rate when the launch-time usable_offers path would never rent it (Codex). Fall
    # back to static across all fitting classes only when the market is unreachable (offline / no key /
    # fetch failure) or no fitting class has an offer, so the estimate stays offline-safe.
    if (provider or "").strip().lower() == "vast":
        from flash.providers.vast.pricing import live_offer_rates

        live = live_offer_rates(max_wall_seconds=max_wall_seconds)
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


def total_params_b(model_id: str) -> float:
    """Total parameter count (billions) for a catalog model."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.params_b


def active_params_b(model_id: str) -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str) -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id) * 2.0


# ~1s mid-range default across grader types (regex ~0.01s to LLM judge ~3s).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION
