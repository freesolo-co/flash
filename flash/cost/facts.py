"""Catalog facts and provider-aware pricing accessors for the cost model."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    # A10: 125 TFLOPS dense bf16 tensor (NVIDIA spec); Lambda-only 24 GB class, else defaults to 100.
    "A10": 125.0,
    # MEASURED on a rented RunPod RTX 4090 (sm89, torch 2.10+cu128): 162.2-162.7 TFLOPS sustained
    # on large square bf16 matmuls, against the 165 vendor spec -- within 1.5%, so the spec number
    # is a fair proxy for this class and is kept.
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    # MEASURED on a rented RunPod A100 80GB PCIe (sm80, torch 2.10+cu128): 242-259 TFLOPS sustained
    # across repeated trials at full clocks (1410 MHz) and 31C, i.e. neither thermal- nor
    # power-limited. The 312 figure is the vendor dense-bf16 spec, which real matmul kernels do not
    # reach; using it overstated A100 throughput by ~20% and made the class look faster (and so
    # cheaper per job) than it is. Set to the measured sustained rate.
    "A100 PCIe": 250.0,
    # MEASURED on a rented RunPod A100-SXM4-80GB (sm80, torch 2.10+cu128): 263-265 TFLOPS across
    # repeated trials. Same 108 SMs as the PCIe part but a 400 W envelope against 300 W, which is
    # what the ~6% gain over the measured PCIe rate reflects. Both are far below the shared 312
    # vendor spec.
    "A100 SXM": 264.0,
    # A100 SXM 40GB: same SMs/tensor cores as the 80GB A100 SXM, less HBM only, so it inherits the
    # measured SXM rate. Without an entry, 33-40 GB Lambda/Vast quotes fall back to _DEFAULT_TFLOPS.
    "A100 SXM 40GB": 264.0,
    # MEASURED on a rented RunPod H100 PCIe (sm90, torch 2.10+cu128): 490-504 TFLOPS sustained
    # across repeated trials at full clocks (1755 MHz), 310 W board. The 990 figure is the SXM
    # part's spec; the PCIe board this class actually provisions delivers about half of it. Using
    # 990 made H100 look twice as fast as it is and let it win nearly every ranking decision on
    # throughput it cannot deliver. Set to the measured sustained rate.
    "H100": 500.0,
    # H200: same SM/tensor-core configuration as H100 with more HBM and a higher power envelope.
    # Scaled from the H100 measurement rather than the SXM spec. Not directly measured.
    "H200": 550.0,
    "RTX Pro 6000": 250.0,
    # B200: 2.25 PFLOPS bf16 dense (NVIDIA spec); prevents ~10x cost over-estimate vs _DEFAULT_TFLOPS.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0

# multi-card scaling depends on interconnect: RunPod's pinned SXM classes use NVLink; PCIe and
# GeForce classes do not. unknowns are conservatively PCIe. Vast can mix board forms, so
# `_PCIE_AMBIGUOUS_PROVIDERS` overrides membership; Lambda has no `gpu_count` path.
_NVLINK_CLASSES: frozenset[str] = frozenset(
    {
        # MEASURED: 2x A100-SXM4-80GB on RunPod reached 1.7675x (see MULTI_CARD_SCALING_NVLINK).
        "A100 SXM",
        # same sxm4 board and nvlink fabric as the 80gb part, less hbm only.
        "A100 SXM 40GB",
        # runpod pins H100 to NVIDIA_H100_80GB_HBM3 (providers/base.py) -- the sxm part, with
        # nvlink. the pcie and NVL boards in runpod's ADA_80_PRO pool are explicitly NEGATED by
        # providers/runpod/gpus.py, so a multi-card H100 combination lands on sxm silicon. the
        # measured-tflops comment on the H100 entry above refers to a single-gpu PCIe rental and
        # says nothing about which board a multi-card run gets.
        "H100",
        # sxm parts with nvlink/nvswitch by form factor. INFERRED, not measured -- no multi-card
        # benchmark has been run on either class.
        "H200",
        "B200",
    }
)


# providers that sell a canonical class as a market rather than a pinned board, so an nvlink
# class membership cannot be trusted for a multi-card combination there. vast lists pcie parts under
# aliases that normalize into nvlink-classed entries and its offers carry no interconnect field.
# runpod is absent because it pins an exact gpu id per class; lambda is absent because it has no
# multi-card path at all, so no scaling question arises.
_PCIE_AMBIGUOUS_PROVIDERS: frozenset[str] = frozenset({"vast"})


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def has_nvlink(name: str, provider: str = "") -> bool:
    """Whether cards of class ``name`` are interconnected by nvlink rather than pcie.

    Unknown classes are conservatively PCIe. Vast markets can include PCIe aliases, so report PCIe
    for ambiguous providers rather than over-crediting topology.
    """
    if name not in _NVLINK_CLASSES:
        return False
    return provider.strip().lower() not in _PCIE_AMBIGUOUS_PROVIDERS


# cap B200 training throughput at H200 until workload-specific B200 measurements exist; current
# portable kernels do not realize its peak. serving and VRAM keep the true `gpu_tflops`.
_TRAIN_TFLOPS_CAP: dict[str, float] = {
    # h200-class realized training throughput, not the 2250 dense-bf16 peak. tracks the H200 entry,
    # which is itself now anchored to a measured H100 PCIe rate rather than a vendor spec.
    "B200": 550.0,
}


def effective_train_tflops(name: str) -> float:
    """Realized bf16 TRAINING throughput for the analytical cost model.

    Equal to ``gpu_tflops`` (peak) except where a class's training kernels fall short of peak: b200
    training falls back to portable kernels, so it is capped at h200-class throughput rather than
    its raw 2.25 pflops peak."""
    peak = gpu_tflops(name)
    cap = _TRAIN_TFLOPS_CAP.get(name)
    return min(peak, cap) if cap is not None else peak


def gpu_hourly_usd(
    name: str,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    min_vram_gb: int = 0,
    gpu_type: str = "",
) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    Lambda and Vast use provider pricing with static fallback; others use RunPod rates. Vast gets
    the run-duration and VRAM floors so quote and allocator search the same market.
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
            gpu_type=gpu_type,
        )
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int,
    *,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    cost_key=None,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``.

    This is intentionally gate-free; provider and wall-time filters constrain candidates.
    `cost_key` ranks whole-job cost and is injected to keep the dependency on `analytical` one-way.
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

    if cost_key is not None:
        # rank on job cost, tie-breaking on rate then the same vram/name order as the $/hr path so
        # the two bases stay comparable when a run is unpriceable in only one of them.
        best = min(
            candidates, key=lambda g: (cost_key(g.name, _rate(g)), _rate(g), g.vram_gb, g.name)
        )
    else:
        best = min(candidates, key=lambda g: (_rate(g), g.vram_gb, g.name))
    return best.name


def total_params_b(model_id: str, revision: str = "") -> float:
    """Total parameter count (billions), curated entry first and HF safetensors otherwise.

    Pinned revisions use their pinned size. Authorized uncataloged models share `resolve_params_b`
    with VRAM sizing; if neither source can size the model, fail closed.
    """
    info = MODELS.get(model_id)
    if info is not None and info.params_b > 0 and not revision:
        # The managed path answers here and never touches the network.
        return info.params_b

    params_b = _sized_from_hf(model_id, revision)
    if not params_b:
        raise ValueError(
            f"could not size model {model_id!r}: it is not in the catalog "
            f"({', '.join(MODELS)}) and HuggingFace returned no safetensors parameter metadata "
            f"for it, so the run cannot be priced"
        )
    return params_b


# cache successful open-model size lookups by normalized `(model_id, revision)` to avoid repeated HF
# calls per quote. misses remain retryable; cache above `fetch_hf_params_b` so test stubs do not
# leak.
_SIZE_MEMO: dict[tuple[str, str], float] = {}


def _sized_from_hf(model_id: str, revision: str) -> float | None:
    """Resolve an uncataloged model's size, caching only a real answer.

    Do not cache misses: hub errors, rate limits, and missing access are transient.
    """
    key = (model_id, revision)
    if key not in _SIZE_MEMO:
        from flash.engine.vram import resolve_params_b

        params_b = resolve_params_b(model_id, revision=revision)
        if not params_b:
            return None
        _SIZE_MEMO[key] = params_b
    return _SIZE_MEMO[key]


def active_params_b(model_id: str, revision: str = "") -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM.

    Price uncataloged models as dense because they lack curated routing metadata; use the pinned
    total the worker loads.
    """
    info = MODELS.get(model_id)
    if info is None:
        return total_params_b(model_id, revision)
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


# parasail supplied-token scoring round-trip per completion.
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model.

    Routes through the closed managed-teacher catalog. Parasail bills every supplied-token score as
    all prompt tokens plus the one generated output token required by its completions endpoint."""
    from flash.engine.recipe import resolve_teacher

    return resolve_teacher(teacher_model).usd_per_1m


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
