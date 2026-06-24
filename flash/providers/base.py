"""Shared GPU-provider interface + GPU registry.

RunPod is the managed GPU substrate. This module owns the parts that are not specific to the
RunPod transport:

  * ``GpuClass`` — one managed GPU class with its RunPod Flash identity.
  * ``JobHandle`` / ``PollResult`` — the persisted-handle + poll-outcome shapes the
    orchestrator round-trips through the provider.
  * ``Candidate`` / ``Allocation`` — the allocation result.
  * The canonicalization / alias / policy helpers every call site already used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flash.spec import JobSpec


# ---------------------------------------------------------------------------
# GPU class registry (provider-agnostic identity)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuClass:
    """One managed RunPod GPU class: a friendly name + RunPod Flash identity/metadata."""

    name: str  # canonical friendly name used in configs / the catalog
    enum_member: str | None  # runpod_flash GpuType member name; None -> not on RunPod
    vram_gb: int
    short: str  # endpoint-name-safe token (e.g. "4090", "a5000")
    sm: str  # CUDA arch (informational; sm80+ only)
    hourly_usd: float  # static rate used by pricing, cost projection, and ranking
    # Min host CUDA (driver) on the modern stack. None -> 12.8. Blackwell (sm120/sm100)
    # needs CUDA-13 drivers to JIT the wheels' PTX (no SASS shipped).
    min_cuda_modern: str | None = None
    # Whether this class has passed Flash's LIVE validation smoke (a real train+eval run on the
    # card). The deployed control plane REJECTS a submit for a non-validated class ("gpu type 'X'
    # has not passed Flash's live validation smoke"), so client-side allocation restricts to the
    # validated pool by default (see ``validated_classes`` / allocator) — otherwise a default
    # `flash train` could pick the absolute-cheapest fitting class (e.g. "L4") that the
    # server then refuses, and the run never submits. Exactly the smoke-validated members below
    # are marked True.
    validated: bool = False


# Static hourly rates are RunPod secure-cloud on-demand snapshots.
GPU_CLASSES: tuple[GpuClass, ...] = (
    GpuClass(
        "RTX 4090",
        "NVIDIA_GEFORCE_RTX_4090",
        24,
        "4090",
        "sm89",
        0.69,
        validated=True,
    ),
    GpuClass(
        "RTX 5090",
        "NVIDIA_GEFORCE_RTX_5090",
        32,
        "5090",
        "sm120",
        0.99,
        min_cuda_modern="13.0",
        validated=True,
    ),
    # ---- Ampere/Ada workstation + datacenter cards (cheap capacity pools) ----
    # 24 GB is the floor: the sub-24 GB tiers (16 GB RTX A4000 / RTX 2000 Ada, 20 GB RTX A4500 /
    # RTX 4000 Ada) were dropped — the 24 GB classes below are the smallest managed cards.
    GpuClass(
        "RTX 3090",
        "NVIDIA_GEFORCE_RTX_3090",
        24,
        "3090",
        "sm86",
        0.46,
        validated=True,
    ),
    GpuClass("L4", "NVIDIA_L4", 24, "l4", "sm89", 0.39),
    # Live-validated 2026-06-22: Qwen3.5-0.8B/9B SFT+GRPO train smokes (RunPod). The 48 GB tier that
    # fills the 32->80 GB gap (e.g. 4B GRPO @ 35 GB) ~55% cheaper than the A100.
    GpuClass(
        "RTX A6000", "NVIDIA_RTX_A6000", 48, "a6000", "sm86", 0.49,
        validated=True,
    ),
    GpuClass("A40", "NVIDIA_A40", 48, "a40", "sm86", 0.44),
    GpuClass(
        "RTX 6000 Ada",
        "NVIDIA_RTX_6000_ADA_GENERATION",
        48,
        "6000ada",
        "sm89",
        0.77,
    ),
    # ---- big-VRAM tier (9B bf16 GRPO, future >9B bf16) ----
    # Validated 2026-06-11: 0.6B SFT smoke (phase6).
    GpuClass(
        "A100 PCIe",
        "NVIDIA_A100_80GB_PCIe",
        80,
        "a100pcie",
        "sm80",
        1.39,
        validated=True,
    ),
    # Live-validated 2026-06-22: Qwen3.5 0.8B/MiniCPM/2B/9B SFT+GRPO train smokes (RunPod).
    GpuClass(
        "A100 SXM", "NVIDIA_A100_SXM4_80GB", 80, "a100sxm", "sm80", 1.49,
        validated=True,
    ),
    # Live-validated 2026-06-22: MiniCPM/2B/4B SFT+GRPO train smokes (RunPod).
    GpuClass(
        "H100", "NVIDIA_H100_80GB_HBM3", 80, "h100", "sm90", 3.29,
        validated=True,
    ),
    # Live-validated 2026-06-22: MiniCPM/2B/4B SFT+GRPO train smokes (RunPod, sm120/CUDA-13).
    GpuClass(
        "RTX Pro 6000",
        "NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION",
        96,
        "pro6000",
        "sm120",
        2.09,
        min_cuda_modern="13.0",
        validated=True,
    ),
)

GPU_INFO: dict[str, GpuClass] = {g.name: g for g in GPU_CLASSES}

# Canonical friendly names Flash exposes in configs / the catalog.
KNOWN = tuple(GPU_INFO)

# The names that have passed Flash's live validation smoke. Client-side allocation restricts to
# these by default because the deployed control plane REJECTS a submit for any class outside the
# pool ("gpu type 'X' has not passed Flash's live validation smoke"); allocating an unvalidated
# (e.g. absolute-cheapest) class would just make the server refuse the run. Kept in sync with the
# ``validated=True`` flags above.
VALIDATED = tuple(g.name for g in GPU_CLASSES if g.validated)


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
# nvidia-smi / the RunPod API print) and historical Flash aliases.
_ALIASES.update(
    {
        "nvidia geforce rtx 4090": "RTX 4090",
        "nvidia geforce rtx 5090": "RTX 5090",
        "nvidia geforce rtx 3090": "RTX 3090",
        "nvidia l4": "L4",
        "nvidia a40": "A40",
        "nvidia rtx 6000 ada generation": "RTX 6000 Ada",
        "rtx 6000 ada generation": "RTX 6000 Ada",
        "nvidia a100 80gb pcie": "A100 PCIe",
        "a100 80gb pcie": "A100 PCIe",
        "a100-80g-pcie": "A100 PCIe",
        "nvidia a100-sxm4-80gb": "A100 SXM",
        "a100-sxm4-80gb": "A100 SXM",
        "a100": "A100 PCIe",
        "nvidia h100 80gb hbm3": "H100",
        "h100 80gb hbm3": "H100",
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
        f'unsupported gpu {name!r}; Flash manages {", ".join(KNOWN)}'
    )


def get_gpu_info(name: str) -> GpuClass:
    return GPU_INFO[canonical_gpu(name)]


def providers_for(name: str) -> tuple[str, ...]:
    """Providers that can provision this GPU class."""
    info = get_gpu_info(name)
    return ("runpod",) if info.enum_member else ()


def gpu_short(name: str) -> str:
    """Short, endpoint-name-safe token for a GPU (e.g. '4090')."""
    return get_gpu_info(name).short


def min_cuda_modern(name: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the modern stack."""
    return get_gpu_info(name).min_cuda_modern or "12.8"


def cheapest_gpu(min_vram_gb: int) -> str:
    """Cheapest validated RunPod GPU class with at least ``min_vram_gb`` VRAM.

    RunPod-static by design so the result is always deployable via Flash, and offline resolution
    stays deterministic. Restricted to the validated pool so the picked class matches what the
    deployed control plane will actually accept — a non-validated class submits then gets rejected.
    """
    pool = [
        g
        for g in GPU_INFO.values()
        if g.enum_member and g.vram_gb >= min_vram_gb and g.validated
    ]
    if not pool:
        raise UnsupportedGpuError(
            f"no validated RunPod-provisionable GPU class has >= {min_vram_gb} GB VRAM"
        )
    from flash.providers.runpod.pricing import hourly_rate

    return min(pool, key=lambda g: (hourly_rate(g.name), g.vram_gb)).name


def provisional_gpu(
    model_id: str,
    algorithm: str = "sft",
    *,
    train=None,
    thinking: bool = False,
) -> str:
    """The cheapest VALIDATED GPU class whose VRAM covers the model -- a parse-time provisional.

    GPU pinning is gone: this picks the cheapest RunPod-provisionable class whose VRAM covers the
    model, restricted to the validated pool (``cheapest_gpu``'s default) so the provisional
    matches what the deployed control plane will accept. The submit-time allocator
    (``flash.providers.allocator``) always re-resolves the cheapest fitting validated class; this
    is the RunPod-static, offline-deterministic equivalent the schema uses for sizing/display.
    """
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.allocator import vram_headroom

    # Honor FLASH_VRAM_HEADROOM so parse-time sizing matches the submit-time allocator exactly.
    min_vram = model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )
    return cheapest_gpu(min_vram)


# ---------------------------------------------------------------------------
# Handles + poll outcomes (round-tripped through any provider)
# ---------------------------------------------------------------------------
@dataclass
class JobHandle:
    """Provider-tagged, persisted handle: enough to reattach/cancel from any process.

    The provider owns the rest of its handle shape (RunPod: endpoint_id/job_id). ``provider`` is
    the routing key the orchestrator uses to dispatch poll/cancel/destroy through the registry.
    """

    provider: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"provider": self.provider, **self.data}

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        d = dict(d)
        provider = d.pop("provider", "runpod")
        return cls(provider=provider, data=d)


@dataclass
class PollResult:
    ok: bool
    metrics: dict | None = None
    # "job_failed"    : genuine worker/job code error (NOT retried)
    # "job_preempted" : provider killed the worker (platform termination) -> infra-shaped, retried
    # "no_capacity"   : NEVER scheduled — no provider capacity for the pinned GPU class (job sat
    #                   IN_QUEUE / the only worker stayed THROTTLED) -> infra-shaped, retried on the
    #                   next-best GPU. Distinct from "stalled" (a worker WAS scheduled then stopped
    #                   making progress) so the terminal message points at capacity, not worker health.
    # "stalled"       : a scheduled worker made no progress within the budget -> infra-shaped, retried
    # "poll_error"    : client-side polling / deploy breakdown -> infra-shaped, retried
    failure: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# Allocation result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    provider: str
    gpu: str
    hourly_usd: float
    vram_gb: int


@dataclass(frozen=True)
class Allocation:
    provider: str
    gpu: str
    hourly_usd: float
    min_vram_gb: int
    candidates: tuple[Candidate, ...]  # full ranked list (retry walks this)


# ---------------------------------------------------------------------------
# The provider interface (FIXED method set both providers implement)
# ---------------------------------------------------------------------------
@runtime_checkable
class Provider(Protocol):
    """The GPU-substrate interface implemented by ``providers/runpod``."""

    name: str

    def is_configured(self) -> bool:
        """Whether this provider is usable right now (creds present, net reachable)."""
        ...

    def preflight(self, require_hf: bool = True) -> list[str]:
        """Missing-config problems (empty list == ready). The control plane aggregates
        these into one fail-fast error at startup."""
        ...

    def gpu_classes(self) -> list[GpuClass]:
        """The GPU classes this provider can provision (its rows of the shared table)."""
        ...

    def hourly_rate(self, gpu: str) -> float:
        """Static $/hr for one friendly GPU name."""
        ...

    def submit_run(
        self,
        spec: JobSpec,
        seed: int,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
    ) -> PollResult:
        """Deploy/rent -> submit -> persist handle (via ``on_handle``) -> poll."""
        ...

    def poll(self, handle: JobHandle, spec: JobSpec, seed: int, *, log: Any = None) -> PollResult:
        """Reattach to a persisted handle and poll it to a terminal state."""
        ...

    def cancel(self, handle: JobHandle) -> None:
        """Stop the exact remote worker for this handle (cross-process)."""
        ...

    def destroy(self, handle: JobHandle) -> None:
        """Tear down the billable resource this handle owns (idempotent)."""
        ...

    def gc(self, spec: JobSpec) -> None:
        """Best-effort: reap any resource this run may have left registered."""
        ...

    def sweep_orphans(self, active_labels: set[str] | None = None) -> list[int]:
        """Destroy any billable resource this provider owns that no live run claims.

        Crash recovery: run at server startup (and after runs). ``active_labels`` is the
        set of instance-label PREFIXES still owned by recoverable runs — anything this
        provider rented that matches none of them is an orphan. Returns the destroyed
        resource ids. Providers without a standing-billing substrate (RunPod's
        serverless endpoints self-reap) implement this as a no-op."""
        ...
