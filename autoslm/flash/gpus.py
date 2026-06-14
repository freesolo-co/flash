"""AutoSLM's managed GPU table: friendly names -> per-provider identity + metadata.

AutoSLM is a managed, single-GPU-per-run fine-tuning package. Every GPU class AutoSLM
can provision — on RunPod Flash (``enum_member``) and/or Vast.ai verified datacenters
(``vast_name``) — is listed here with the metadata selection needs (VRAM, fallback
hourly rate, min host CUDA on the modern stack), but only classes that passed AutoSLM's
live train+eval smoke on a provider are validated there (``validated_on``) — configs
get those by default. Unvalidated classes are usable with an explicit opt-in
(``gpu.allow_unvalidated = true`` or ``AUTOSLM_GPU_ALLOW_UNVALIDATED=1``), so configs
can't *silently* land on unproven hardware, and ``gpu.type = "cheapest"`` always has a
maximal validated pool to pick from (the cheapest card that fits the model — resolved
cross-provider at submit time by ``autoslm.providers.allocator``; ``resolve_gpu_policy``
is the parse-time RunPod-static provisional).

Architecture floor: Ampere (sm80). bf16 training, vLLM 0.19 kernels, and bitsandbytes
all need sm80+; Turing and older are deliberately absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Lazy import so unit tests that only exercise the mapping don't pull the whole SDK
# graph unless needed. ``runpod_flash`` is a hard dependency, so this import is safe.
def _gpu_enum():
    from runpod_flash import GpuType

    return GpuType


PROVIDERS = ("runpod", "vast")


@dataclass(frozen=True)
class GpuInfo:
    name: str  # canonical friendly name used in configs / the catalog
    enum_member: str | None  # runpod_flash GpuType member name; None -> not on RunPod
    vram_gb: int
    short: str  # endpoint-name-safe token (e.g. "4090", "a5000")
    sm: str  # CUDA arch (informational; sm80+ only)
    hourly_usd: float  # static fallback rate; live pricing overrides (flash/pricing.py)
    # Providers where this class passed AutoSLM's live train+eval smoke. Validation is
    # per-provider: the same silicon behind a different provisioning path (Flash deps
    # install vs a Vast docker image) is a different failure surface.
    validated_on: tuple[str, ...] = ()
    # Min host CUDA (driver) on the modern stack. None -> 12.8. Blackwell (sm120/sm100)
    # needs CUDA-13 drivers to JIT the wheels' PTX (no SASS shipped) — same root cause
    # as the documented RTX 5090 pin.
    min_cuda_modern: str | None = None
    # Vast.ai offer ``gpu_name`` for this class; None -> not provisionable on Vast.
    # A100 SXM4 boards exist in 40 GB and 80 GB variants under ONE Vast name — offers
    # are disambiguated by ``gpu_ram`` (see ``vast_gpu_for_offer``).
    vast_name: str | None = None

    @property
    def validated(self) -> bool:  # back-compat: validated on ANY provider
        return bool(self.validated_on)


# Fallback hourly rates are RunPod secure-cloud on-demand (snapshot 2026-06-11); the
# live rates from flash/pricing.py override them for cost projection + "cheapest".
# Vast-only classes (enum_member=None) carry a Vast verified-datacenter snapshot
# instead — they never enter RunPod-static selection, so the rate is display-only.
GPU_INFO: dict[str, GpuInfo] = {
    g.name: g
    for g in (
        # ---- validated: passed the full train+eval matrix (bench/results/phase1) ----
        GpuInfo(
            "RTX 4090",
            "NVIDIA_GEFORCE_RTX_4090",
            24,
            "4090",
            "sm89",
            0.69,
            validated_on=("runpod",),
            vast_name="RTX 4090",
        ),
        # Vast-validated 2026-06-12: Qwen3-0.6B SFT train+eval smoke on a verified
        # datacenter ($0.60/hr South Korea), incl. vLLM eval on a CUDA-13 driver
        # with the cu128 stack image.
        GpuInfo(
            "RTX 5090",
            "NVIDIA_GEFORCE_RTX_5090",
            32,
            "5090",
            "sm120",
            0.99,
            validated_on=("runpod", "vast"),
            min_cuda_modern="13.0",
            vast_name="RTX 5090",
        ),
        # ---- Ampere/Ada workstation + datacenter cards (cheap capacity pools) ----
        GpuInfo("RTX A4000", "NVIDIA_RTX_A4000", 16, "a4000", "sm86", 0.25, vast_name="RTX A4000"),
        GpuInfo(
            "RTX 2000 Ada",
            "NVIDIA_RTX_2000_ADA_GENERATION",
            16,
            "2000ada",
            "sm89",
            0.24,
            vast_name="RTX 2000Ada",
        ),
        GpuInfo("RTX A4500", "NVIDIA_RTX_A4500", 20, "a4500", "sm86", 0.25, vast_name="RTX A4500"),
        GpuInfo(
            "RTX 4000 Ada",
            "NVIDIA_RTX_4000_ADA_GENERATION",
            20,
            "4000ada",
            "sm89",
            0.26,
            vast_name="RTX 4000Ada",
        ),
        # Validated 2026-06-11: Qwen3-0.6B SFT + GRPO smokes passed (train+eval,
        # bench/results/phase6/gpu_matrix/) — the cheapest 24 GB class by far.
        GpuInfo(
            "RTX A5000",
            "NVIDIA_RTX_A5000",
            24,
            "a5000",
            "sm86",
            0.27,
            validated_on=("runpod",),
            vast_name="RTX A5000",
        ),
        # Vast-validated 2026-06-12: Qwen3-0.6B SFT train+eval smoke on a verified
        # datacenter (tests/live/test_vast_live_smoke.py; $0.25/hr Czechia).
        GpuInfo(
            "RTX 3090",
            "NVIDIA_GEFORCE_RTX_3090",
            24,
            "3090",
            "sm86",
            0.46,
            validated_on=("vast",),
            vast_name="RTX 3090",
        ),
        GpuInfo("L4", "NVIDIA_L4", 24, "l4", "sm89", 0.39, vast_name="L4"),
        # Blackwell workstation card; cheap verified-datacenter capacity on Vast.
        # Vast-validated 2026-06-12: Qwen3-0.6B SFT train+eval smoke incl. vLLM eval
        # on a CUDA-13 driver with the cu128 stack image ($0.34/hr Hungary).
        GpuInfo(
            "RTX Pro 4000",
            None,
            24,
            "pro4000",
            "sm120",
            0.34,
            validated_on=("vast",),
            min_cuda_modern="13.0",
            vast_name="RTX PRO 4000",
        ),
        GpuInfo("RTX A6000", "NVIDIA_RTX_A6000", 48, "a6000", "sm86", 0.49, vast_name="RTX A6000"),
        GpuInfo("A40", "NVIDIA_A40", 48, "a40", "sm86", 0.44, vast_name="A40"),
        GpuInfo(
            "RTX 6000 Ada",
            "NVIDIA_RTX_6000_ADA_GENERATION",
            48,
            "6000ada",
            "sm89",
            0.77,
            vast_name="RTX 6000Ada",
        ),
        # L40S exists at RunPod but not in the Flash SDK's GpuType enum -> Vast-only.
        GpuInfo("L40S", None, 48, "l40s", "sm89", 0.87, vast_name="L40S"),
        # ---- big-VRAM tier (large-MoE QLoRA, OPD teachers, future >9B bf16) ----
        # 40 GB SXM4 boards share Vast's "A100 SXM4" name with the 80 GB variant;
        # offers are split by gpu_ram (vast_gpu_for_offer). Not a RunPod Flash class.
        GpuInfo("A100 SXM 40GB", None, 40, "a100sxm40", "sm80", 0.89, vast_name="A100 SXM4"),
        # Validated 2026-06-11: 0.6B SFT smoke + full Qwen3.6-35B-A3B QLoRA SFT
        # (72 GB checkpoint, 160 GB disk) train+eval end-to-end (phase6).
        GpuInfo(
            "A100 PCIe",
            "NVIDIA_A100_80GB_PCIe",
            80,
            "a100pcie",
            "sm80",
            1.39,
            validated_on=("runpod",),
            vast_name="A100 PCIE",
        ),
        GpuInfo(
            "A100 SXM", "NVIDIA_A100_SXM4_80GB", 80, "a100sxm", "sm80", 1.49, vast_name="A100 SXM4"
        ),
        GpuInfo("H100", "NVIDIA_H100_80GB_HBM3", 80, "h100", "sm90", 3.29, vast_name="H100 SXM"),
        GpuInfo("H200", "NVIDIA_H200", 141, "h200", "sm90", 4.39, vast_name="H200"),
        GpuInfo(
            "RTX Pro 6000",
            "NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION",
            96,
            "pro6000",
            "sm120",
            2.09,
            min_cuda_modern="13.0",
        ),
        GpuInfo("B200", "NVIDIA_B200", 180, "b200", "sm100", 5.89, min_cuda_modern="13.0"),
    )
}

# Canonical friendly names AutoSLM exposes in configs / the catalog.
KNOWN = tuple(GPU_INFO)
# Classes proven by a live train+eval smoke (the default selection pool).
SUPPORTED = tuple(g.name for g in GPU_INFO.values() if g.validated)

# GPU-policy keywords accepted in ``gpu.type`` (resolved to a concrete class at parse
# time by ``resolve_gpu_policy``; the submit-time allocator re-resolves them live).
POLICY_NAMES = ("cheapest", "auto")


def _alias_keys(name: str) -> set[str]:
    """All accepted spellings of a friendly name (lowercased)."""
    base = name.lower()
    keys = {base, base.replace(" ", ""), base.replace(" ", "_"), base.replace(" ", "-")}
    if base.startswith("rtx "):
        tail = base[4:]
        keys |= {tail, tail.replace(" ", ""), tail.replace(" ", "_")}
    keys.add(f"nvidia {base}")
    return keys


_ALIASES: dict[str, str] = {}
for _info in GPU_INFO.values():
    for _k in _alias_keys(_info.name):
        _ALIASES[_k] = _info.name
# Spellings that don't fall out of the generic rules: full marketing names (what
# nvidia-smi / the RunPod API print) and historical AutoSLM aliases.
_ALIASES.update(
    {
        "nvidia geforce rtx 4090": "RTX 4090",
        "nvidia geforce rtx 5090": "RTX 5090",
        "nvidia geforce rtx 3090": "RTX 3090",
        "nvidia l4": "L4",
        "nvidia a40": "A40",
        "nvidia rtx 6000 ada generation": "RTX 6000 Ada",
        "rtx 6000 ada generation": "RTX 6000 Ada",
        "nvidia rtx 4000 ada generation": "RTX 4000 Ada",
        "nvidia rtx 2000 ada generation": "RTX 2000 Ada",
        "nvidia a100 80gb pcie": "A100 PCIe",
        "a100 80gb pcie": "A100 PCIe",
        "a100-80g-pcie": "A100 PCIe",
        "nvidia a100-sxm4-80gb": "A100 SXM",
        "a100-sxm4-80gb": "A100 SXM",
        "a100": "A100 PCIe",
        "nvidia h100 80gb hbm3": "H100",
        "h100 80gb hbm3": "H100",
        "nvidia h200": "H200",
        "nvidia b200": "B200",
        "rtx pro 6000 blackwell": "RTX Pro 6000",
        "nvidia rtx pro 6000 blackwell server edition": "RTX Pro 6000",
    }
)


class UnsupportedGpuError(ValueError):
    pass


def canonical_gpu(name: str) -> str:
    """Normalize a friendly GPU name to one of ``KNOWN``; raise otherwise."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnsupportedGpuError(
        f'unsupported gpu {name!r}; AutoSLM manages {", ".join(KNOWN)} (or gpu.type = "cheapest")'
    )


def get_gpu_info(name: str) -> GpuInfo:
    return GPU_INFO[canonical_gpu(name)]


def is_validated(name: str, provider: str | None = None) -> bool:
    """Validated on ``provider`` (when given) or on any provider (provider=None)."""
    info = get_gpu_info(name)
    if provider is None or provider == "auto":
        return info.validated
    return provider in info.validated_on


def providers_for(name: str) -> tuple[str, ...]:
    """Providers that can provision this GPU class."""
    info = get_gpu_info(name)
    out = []
    if info.enum_member:
        out.append("runpod")
    if info.vast_name:
        out.append("vast")
    return tuple(out)


# Boards under-report usable VRAM vs the class's nominal size (measured live: L4
# offers carry 23034 MB for the 24 GB class, A40 offers 46068 MB for the 48 GB
# class — ~3 GB under), so class matching gets a tolerance. Safe at 3.5 GB: names
# shared across VRAM variants differ by >= 40 GB (A100 SXM4 40/80).
_VRAM_MATCH_TOLERANCE_GB = 3.5


def vast_gpu_for_offer(gpu_name: str, gpu_ram_mb: float) -> str | None:
    """Map a Vast offer (``gpu_name`` + ``gpu_ram`` MB) to a canonical GPU class.

    Returns None for anything not in the managed table — that's the hard Ampere+
    floor (T4/2080 Ti/Quadro RTX offers never match). Names shared across VRAM
    variants (A100 SXM4 40/80 GB) resolve to the largest class the board's actual
    RAM covers.
    """
    fitting = [
        g
        for g in GPU_INFO.values()
        if g.vast_name == gpu_name and g.vram_gb <= gpu_ram_mb / 1024 + _VRAM_MATCH_TOLERANCE_GB
    ]
    if not fitting:
        return None
    return max(fitting, key=lambda g: g.vram_gb).name


def unvalidated_allowed(explicit: bool | None = None) -> bool:
    """Whether configs may target a non-``validated`` GPU class."""
    if explicit is not None:
        return explicit
    return os.environ.get("AUTOSLM_GPU_ALLOW_UNVALIDATED", "") not in ("", "0", "false")


def flash_gpu(name: str):
    """Return the RunPod Flash ``GpuType`` for a friendly GPU name."""
    info = get_gpu_info(name)
    if not info.enum_member:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    return getattr(_gpu_enum(), info.enum_member)


def gpu_api_id(name: str) -> str:
    """RunPod API GPU id (the ``GpuType`` enum value, e.g. 'NVIDIA GeForce RTX 4090')."""
    return flash_gpu(name).value


def gpu_for_model(model_id: str) -> str:
    """Default GPU for a model when a config omits ``gpu.type``.

    Models needing >= 32 GB VRAM go to the RTX 5090; everything else to the 4090.
    (Unchanged default: cheaper classes are opt-in via an explicit type or "cheapest".)
    """
    from autoslm.catalog import get_model

    try:
        info = get_model(model_id)
    except Exception:
        return "RTX 5090"
    return "RTX 5090" if info.min_vram_gb >= 32 else "RTX 4090"


def cheapest_gpu(min_vram_gb: int, include_unvalidated: bool = False) -> str:
    """Cheapest RunPod GPU class with at least ``min_vram_gb`` VRAM (live rates, cached).

    Rates come from flash/pricing.py (RunPod live pricing with the static
    ``GpuInfo.hourly_usd`` fallback). Note: cheapest per *hour*, not per run — slower
    cards trade wall time for rate; the validated pool keeps the trade sane.
    RunPod-static by design (the cross-provider equivalent lives in
    ``autoslm.providers.allocator``): Vast-only classes are excluded so the result is
    always deployable via Flash, and offline resolution stays deterministic.
    """
    pool = [
        g
        for g in GPU_INFO.values()
        if g.enum_member
        and g.vram_gb >= min_vram_gb
        and (include_unvalidated or "runpod" in g.validated_on)
    ]
    if not pool:
        raise UnsupportedGpuError(
            f"no {'known' if include_unvalidated else 'validated'} GPU has >= {min_vram_gb} GB VRAM"
        )
    from autoslm.flash.pricing import hourly_rate

    return min(pool, key=lambda g: (hourly_rate(g.name), g.vram_gb)).name


def resolve_gpu_policy(
    requested: str,
    model_id: str,
    allow_unvalidated: bool | None = None,
    algorithm: str = "sft",
) -> str:
    """Resolve ``gpu.type`` (a concrete class or a policy word) to a friendly name.

    Parse-time, RunPod-static provisional: "cheapest"/"auto" pick the cheapest
    RunPod-validated class whose VRAM covers the model; concrete names are
    canonicalized. The submit-time allocator (``autoslm.providers.allocator``)
    re-resolves policy words live across providers. Catalog models use their
    ``min_vram_gb``; unlisted (open-policy) models are sized from the HF safetensors
    estimate so a large model picks a fitting A100/H100 class instead of defaulting
    to 24 GB and being rejected after.
    """
    key = (requested or "").strip().lower()
    if key not in POLICY_NAMES:
        return canonical_gpu(requested)
    from autoslm.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        min_vram = info.min_vram_gb
    else:
        from autoslm.engine.vram import estimate_vram_gb, fetch_hf_params_b

        params_b = fetch_hf_params_b(model_id)
        # No metadata (offline / private) -> keep the conservative 24 GB floor.
        min_vram = int(estimate_vram_gb(params_b, algorithm)) if params_b else 24
    return cheapest_gpu(min_vram, include_unvalidated=unvalidated_allowed(allow_unvalidated))


def gpu_short(name: str) -> str:
    """Short, endpoint-name-safe token for a GPU (e.g. '4090')."""
    return get_gpu_info(name).short


def min_cuda_modern(name: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the modern stack."""
    return get_gpu_info(name).min_cuda_modern or "12.8"
