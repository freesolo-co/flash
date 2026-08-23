"""Catalog facts and provider-aware pricing accessors for the cost model."""

from __future__ import annotations

from flash.core.catalog import MODELS, ModelInfo
from flash.providers.core.base import GPU_INFO

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


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def _catalog_model_info(model_id: str) -> ModelInfo:
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info


# One quote asks "how big is this model" several times over (setup download, required-save
# serialization, the MoE check). For an unpinned catalog model those are dict reads; for a PINNED
# one each is an `HfApi.model_info` round trip, so an ordinary quote repeats the same lookup and
# becomes hostage to hub latency -- and a transient failure on a later call can reject a run the
# earlier calls already validated.
#
# Memoized per process on the normalized (id, revision) pair: a revision names immutable weights, so
# a SUCCESSFUL answer cannot change under us. A MISS is deliberately not cached -- a failed lookup is
# a hub blip, a rate limit, or an HF_TOKEN not yet granted access, not a fact about the model, and
# caching it would make the first failure permanent for the life of a long-lived plane.
#
# Not `functools.cache` on total_params_b: it keys on the literal argument tuple, so a caller that
# omits `revision` and one that passes "" become two entries. Not on `_validated_revision_geometry`
# either, which tests monkeypatch -- caching there would leak a stubbed size across tests.
_PINNED_SIZE_MEMO: dict[tuple[str, str], float] = {}


def total_params_b(model_id: str, revision: str = "") -> float:
    """Total parameter count (billions) for a catalog model.

    a pinned revision sizes that commit (validated against the catalog, fail-closed) so setup/save
    cost tracks the weights the worker actually loads, not the default-revision count.
    """
    info = _catalog_model_info(model_id)
    if not revision:
        return info.params_b
    key = (model_id, revision)
    if key not in _PINNED_SIZE_MEMO:
        from flash.engine.plan.vram import _validated_revision_geometry

        params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
        _PINNED_SIZE_MEMO[key] = params_b
    return _PINNED_SIZE_MEMO[key]


def active_params_b(model_id: str, revision: str = "") -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM.

    ``revision`` is accepted and IGNORED: an active-parameter count is architecture metadata, which
    a pinned commit of the same catalog entry does not change. it stays in the signature because
    callers sizing a pinned run pass it positionally alongside ``total_params_b``, where it does
    matter. uncataloged models are rejected, so an unknown id raises rather than pricing as dense.
    """
    _ = revision
    info = _catalog_model_info(model_id)
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, revision: str = "") -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id, revision) * 2.0


# parasail supplied-token scoring round-trip per completion.
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model.

    Routes through the closed managed-teacher catalog. Parasail bills every supplied-token score as
    all prompt tokens plus the one generated output token required by its completions endpoint."""
    from flash.engine.plan.recipe import resolve_teacher

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
