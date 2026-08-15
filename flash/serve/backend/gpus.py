"""Modal GPU classes and a serving fit estimate for the self-hosted reference backend.

Separate from ``flash/providers/base.py`` on purpose. That table describes the TRAINING fleet
(RunPod/Lambda/Vast, RTX 4090s, per-provider offer names); this one describes the cards Modal
rents for SERVING, priced per second and scaled to zero. The two overlap in name only.

Nothing here is measured. Flash has no serving throughput dataset -- the only serving timing in
the tree is a single un-normalized ``verify_latency_s`` smoke elapsed. So the speed column is a
RELATIVE rank derived from memory bandwidth, never an absolute tok/s, and every caller that
renders it must label it an estimate. Decode is bandwidth-bound, which is why bandwidth is the
input; deliberately NOT ``flash/cost/facts.py``'s GPU_COMPUTE_TFLOPS, which is calibrated for
compute-bound training and would rank these cards wrongly.

The per-model default is the catalog's own ``serving.gpu``. That is Freesolo's production
choice, validated on real hardware, so it outranks anything computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from flash.core.catalog import ModelInfo

ServingDtype = Literal["fp8", "bf16"]
Headroom = Literal["ample", "good", "tight", "no"]
Speed = Literal["fastest", "fast", "ok", "slow"]

# vLLM serves fp8 by ONLINE-quantizing an ordinary bf16 checkpoint at load when the model has no
# pre-quantized repo, so a self-hoster gets fp8 economics from the public weights.
_BYTES_PER_PARAM = {"fp8": 1.0, "bf16": 2.0}

# vLLM claims this fraction of the card for the whole engine -- weights, LoRA pool, activations and
# KV cache all come out of it, and the remainder is left to the driver. Sizing against raw VRAM
# would overstate every card by ~10%. Matches the production serving default.
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

# Native fp8 tensor cores start at compute capability 8.9 (L4/L40S/H100/H200/B200). sm80 (A100)
# still serves fp8 through vLLM's Marlin weight-only fallback -- correct, but not the same speed,
# so it is reported rather than hidden.
_FP8_NATIVE_MIN_SM = 89

# vLLM's V1 engine requires sm80+. Below that a request for bf16 is silently DOWNGRADED to fp16
# rather than refused (vllm/config/model.py), so a T4 would serve at quietly degraded quality.
# Excluded by capability, not by name, so a future pre-Ampere card cannot slip back in.
_MIN_SM = 80

# Engine overhead beyond weights + KV + LoRA: CUDA context, graphs, activations, allocator slack.
_RUNTIME_OVERHEAD_GB = 2.5

# Headroom classes, in GB free after everything else is accounted for.
_AMPLE_GB = 16.0
_GOOD_GB = 6.0
_TIGHT_GB = 1.0


@dataclass(frozen=True)
class ModalGpu:
    """One Modal GPU class as the serving backend requests it."""

    name: str  # the exact string Modal's gpu= parameter accepts
    vram_gb: int
    sm: int  # compute capability x10 (89 == sm89), matching _MIN_SM comparisons
    usd_hr: float  # derived from Modal's per-second rate
    bandwidth_gbs: int  # vendor-spec HBM/GDDR bandwidth, the decode-speed proxy

    @property
    def fp8_native(self) -> bool:
        return self.sm >= _FP8_NATIVE_MIN_SM


# Ordered cheapest first, which is also the order the recommendation table renders.
# Prices are Modal's published per-second rates x3600, rounded to the cent; they move, so they are
# labeled approximate wherever they are shown. T4 is deliberately absent (see _MIN_SM).
MODAL_GPUS: tuple[ModalGpu, ...] = (
    ModalGpu("L4", 24, 89, 0.80, 300),
    ModalGpu("A10", 24, 86, 1.10, 600),
    ModalGpu("L40S", 48, 89, 1.95, 864),
    ModalGpu("A100-40GB", 40, 80, 2.10, 1555),
    ModalGpu("A100-80GB", 80, 80, 2.50, 2039),
    ModalGpu("H100", 80, 90, 3.95, 3350),
    ModalGpu("H200", 141, 90, 4.95, 4800),
    ModalGpu("B200", 180, 100, 6.25, 8000),
)

MODAL_GPUS_BY_NAME: dict[str, ModalGpu] = {g.name: g for g in MODAL_GPUS}


@dataclass(frozen=True)
class Fit:
    """One card's estimated fit for one model, as rendered in the recommendation table."""

    gpu: ModalGpu
    dtype: ServingDtype
    weights_gb: float
    kv_gb: float
    lora_gb: float
    total_gb: float
    budget_gb: float  # what vLLM will actually claim: vram x gpu_memory_utilization
    free_gb: float
    headroom: Headroom
    speed: Speed
    is_catalog_default: bool
    fp8_native: bool

    @property
    def fits(self) -> bool:
        return self.headroom != "no"


def serving_dtype(info: ModelInfo) -> ServingDtype:
    """Return the catalog's explicit serving weight dtype for ``info``."""
    serving = info.serving
    return "fp8" if serving is not None and serving.quantization == "fp8" else "bf16"


def _headroom(free_gb: float) -> Headroom:
    if free_gb < _TIGHT_GB:
        return "no"
    if free_gb < _GOOD_GB:
        return "tight"
    if free_gb < _AMPLE_GB:
        return "good"
    return "ample"


def _kv_bytes_per_token(info: ModelInfo, kv_dtype: ServingDtype) -> float:
    """Per-token KV bytes from catalog geometry, or 0.0 when the entry lacks it.

    Mirrors the attention term of ``flash/engine/plan/vram.py``'s architecture sizing: 2 (K and V)
    x kv_heads x head_dim x bytes, over the full-attention layers. The GDN-hybrid models also carry
    a recurrent state, but that is per-SEQUENCE rather than per-token, so it is counted in
    ``_recurrent_state_gb`` instead of inflating the per-token rate.
    """
    kv_heads = int(getattr(info, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(info, "head_dim", 0) or 0)
    layers = int(getattr(info, "num_attention_layers", 0) or 0)
    if not (kv_heads and head_dim and layers):
        return 0.0
    return 2 * kv_heads * head_dim * layers * _BYTES_PER_PARAM.get(kv_dtype, 1.0)


def _recurrent_state_gb(info: ModelInfo, sequences: int) -> float:
    """Recurrent + convolution state for GDN-hybrid layers, which scales per sequence not per token."""
    linear_layers = int(getattr(info, "num_linear_attention_layers", 0) or 0)
    key_heads = int(getattr(info, "linear_num_key_heads", 0) or 0)
    value_heads = int(getattr(info, "linear_num_value_heads", 0) or 0)
    key_dim = int(getattr(info, "linear_key_head_dim", 0) or 0)
    value_dim = int(getattr(info, "linear_value_head_dim", 0) or 0)
    conv = int(getattr(info, "linear_conv_kernel_dim", 0) or 0)
    if not all((linear_layers, key_heads, value_heads, key_dim, value_dim, conv)):
        return 0.0
    # vLLM keeps gated-deltanet recurrent and conv state in bf16 pages regardless of kv dtype.
    state_elements = value_heads * key_dim * value_dim
    state_elements += (key_heads * key_dim + value_heads * value_dim) * conv
    return linear_layers * sequences * state_elements * 2 / 1e9


def _lora_pool_gb(max_loras: int, max_lora_rank: int, info: ModelInfo) -> float:
    """GPU LoRA buffers, which vLLM PRE-ALLOCATES at max_loras x max_lora_rank.

    Sized from the model's real LoRA target shapes when the catalog has them, so this tracks the
    adapter that would actually be served rather than a flat guess. Not by adapter count: vLLM
    reserves the full pool at engine init whether or not the adapters exist yet.
    """
    shapes = getattr(info, "lora_target_shapes", ()) or ()
    params_per_rank = sum((fan_in + fan_out) * count for fan_in, fan_out, count in shapes)
    if not params_per_rank:
        return 0.0
    # bf16 A and B factors; the pool holds max_loras of them at the configured rank.
    return max_loras * max_lora_rank * params_per_rank * 2 / 1e9


def _speed_rank(gpu: ModalGpu, fastest_bandwidth: int) -> Speed:
    """Relative decode-speed band, spaced so the CHEAP cards are still distinguishable.

    Anchoring linearly on the fastest card is useless in practice: B200 has ~27x L4's bandwidth,
    so every affordable card collapses into one bucket and the column stops informing the choice
    it exists to inform. Banding on log2 of the ratio keeps L4 / A10 / L40S apart, which is the
    comparison a self-hoster is actually making.
    """
    if not fastest_bandwidth or gpu.bandwidth_gbs <= 0:
        return "slow"
    # doublings behind the fastest card: 0 -> fastest, ~1 -> fast, ~2-3 -> ok, beyond -> slow.
    doublings = math.log2(fastest_bandwidth / gpu.bandwidth_gbs)
    if doublings <= 0.75:
        return "fastest"
    if doublings <= 1.75:
        return "fast"
    if doublings <= 3.25:
        return "ok"
    return "slow"


def estimate_fit(
    info: ModelInfo,
    gpu: ModalGpu,
    *,
    dtype: ServingDtype | Literal[""] = "",
    kv_dtype: ServingDtype = "fp8",
    context_len: int = 0,
    max_seqs: int = 0,
    max_loras: int = 0,
    max_lora_rank: int = 0,
    fastest_bandwidth: int = 0,
) -> Fit:
    """Estimate whether ``info`` serves on ``gpu``, and with how much room to spare.

    Defaults come from the model's own ``serving`` entry so the estimate describes the
    configuration the generated app would actually deploy -- including its dtype, which the
    catalog decides per model rather than uniformly (see ``serving_dtype``).
    """
    serving = getattr(info, "serving", None)
    dtype = dtype or serving_dtype(info)
    context_len = context_len or (getattr(serving, "max_model_len", 0) or 8192)
    max_seqs = max_seqs or (getattr(serving, "max_num_seqs", 0) or 8)
    max_loras = max_loras or (getattr(serving, "max_loras", 0) or 16)
    max_lora_rank = max_lora_rank or (getattr(serving, "max_lora_rank", 0) or 32)
    fastest_bandwidth = fastest_bandwidth or max(g.bandwidth_gbs for g in MODAL_GPUS)

    params_b = float(getattr(info, "params_b", 0.0) or 0.0)
    weights_gb = params_b * _BYTES_PER_PARAM.get(dtype, 2.0)

    kv_gb = _kv_bytes_per_token(info, kv_dtype) * context_len * max_seqs / 1e9
    kv_gb += _recurrent_state_gb(info, max_seqs)
    lora_gb = _lora_pool_gb(max_loras, max_lora_rank, info)

    total_gb = weights_gb + kv_gb + lora_gb + _RUNTIME_OVERHEAD_GB
    # measure against what vLLM will claim, not the card's nameplate: the engine never gets the
    # last ~10%, so a model that "fits in 141 GB" can still fail to start on a 141 GB card.
    util = float(getattr(serving, "gpu_memory_utilization", 0.0) or 0.0)
    budget_gb = gpu.vram_gb * (util or _DEFAULT_GPU_MEMORY_UTILIZATION)
    free_gb = budget_gb - total_gb
    catalog_gpu = (getattr(serving, "gpu", "") or "").strip()

    return Fit(
        gpu=gpu,
        dtype=dtype,
        weights_gb=weights_gb,
        kv_gb=kv_gb,
        lora_gb=lora_gb,
        total_gb=total_gb,
        budget_gb=budget_gb,
        free_gb=free_gb,
        headroom=_headroom(free_gb),
        speed=_speed_rank(gpu, fastest_bandwidth),
        is_catalog_default=bool(catalog_gpu) and catalog_gpu == gpu.name,
        fp8_native=gpu.fp8_native or dtype != "fp8",
    )


def recommend(
    info: ModelInfo,
    *,
    dtype: ServingDtype | Literal[""] = "",
    kv_dtype: ServingDtype = "fp8",
    context_len: int = 0,
) -> list[Fit]:
    """Every Modal card's estimated fit for ``info``, cheapest first."""
    fastest = max(g.bandwidth_gbs for g in MODAL_GPUS)
    return [
        estimate_fit(
            info,
            gpu,
            dtype=dtype,
            kv_dtype=kv_dtype,
            context_len=context_len,
            fastest_bandwidth=fastest,
        )
        for gpu in MODAL_GPUS
        if gpu.sm >= _MIN_SM
    ]


def default_gpu(info: ModelInfo) -> ModalGpu | None:
    """The catalog's production-validated serving card for ``info``, when it names one.

    Returns None rather than guessing: a model with no serving entry has no validated choice, and
    the caller should fall back to the cheapest fitting card and say so.
    """
    name = (getattr(getattr(info, "serving", None), "gpu", "") or "").strip()
    return MODAL_GPUS_BY_NAME.get(name)


def cheapest_fitting(fits: list[Fit]) -> Fit | None:
    """The cheapest card that fits, for a model the catalog has no validated choice for."""
    return next((fit for fit in fits if fit.fits), None)
