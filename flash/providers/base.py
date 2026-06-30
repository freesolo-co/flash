"""Shared GPU-provider interface + GPU registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from flash.spec import JobSpec


@dataclass(frozen=True)
class GpuClass:
    """One managed GPU class with its provider identity/metadata."""

    name: str
    enum_member: str | None  # runpod_flash GpuType member name; None -> not on RunPod
    vram_gb: int
    short: str  # endpoint-name-safe token
    sm: str
    hourly_usd: float
    # None -> 12.8. Blackwell (sm120/sm100) needs CUDA-13 to JIT PTX (no SASS shipped).
    min_cuda_modern: str | None = None
    # Server REJECTS non-validated classes; client restricts to this pool by default.
    validated: bool = False
    lambda_name: str | None = None  # None -> not on Lambda; priced from Lambda live rates


# Hourly rates are RunPod secure-cloud on-demand snapshots.
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
    # Lambda-only; RunPod has no A10. Allocator reaches it only after cheaper RunPod classes exhaust.
    GpuClass("A10", None, 24, "a10", "sm86", 1.29, lambda_name="gpu_1x_a10", validated=True),
    # Lambda-only 40 GB A100; fills the 32->80 GB gap on Lambda.
    GpuClass(
        "A100 SXM 40GB",
        None,
        40,
        "a100sxm40",
        "sm80",
        1.99,
        lambda_name="gpu_1x_a100_sxm4",
        validated=True,
    ),
    GpuClass(
        "A100 PCIe",
        "NVIDIA_A100_80GB_PCIe",
        80,
        "a100pcie",
        "sm80",
        1.39,
        validated=True,
    ),
    GpuClass(
        "A100 SXM",
        "NVIDIA_A100_SXM4_80GB",
        80,
        "a100sxm",
        "sm80",
        1.49,
        validated=True,
    ),
    GpuClass(
        "H100",
        "NVIDIA_H100_80GB_HBM3",
        80,
        "h100",
        "sm90",
        3.29,
        validated=True,
        lambda_name="gpu_1x_h100_pcie",
    ),
    GpuClass(
        "H200",
        "NVIDIA_H200",
        141,
        "h200",
        "sm90",
        4.39,
        validated=True,
    ),
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
    # 180 GB usable (NVIDIA advertises 192 GB; size to the safe 180 per RunPod/Lambda listings).
    GpuClass(
        "B200",
        "NVIDIA_B200",
        180,
        "b200",
        "sm100",
        5.89,
        min_cuda_modern="13.0",
        validated=True,
        lambda_name="gpu_1x_b200_sxm6",
    ),
)

GPU_INFO: dict[str, GpuClass] = {g.name: g for g in GPU_CLASSES}
KNOWN = tuple(GPU_INFO)
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
# Full marketing names (nvidia-smi / RunPod API) and historical aliases not covered by generic rules.
_ALIASES.update(
    {
        "nvidia geforce rtx 4090": "RTX 4090",
        "nvidia geforce rtx 5090": "RTX 5090",
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
        "nvidia b200": "B200",
        "b200 sxm6": "B200",
        "nvidia b200 180gb": "B200",
        "nvidia b200 sxm6": "B200",
    }
)


class UnsupportedGpuError(ValueError):
    pass


def canonical_gpu(name: str) -> str:
    """Normalize a friendly GPU name to a managed class; raise otherwise."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnsupportedGpuError(f"unsupported gpu {name!r}; Flash manages {', '.join(KNOWN)}")


def get_gpu_info(name: str) -> GpuClass:
    return GPU_INFO[canonical_gpu(name)]


def providers_for(name: str) -> tuple[str, ...]:
    """Providers that can provision this GPU class."""
    info = get_gpu_info(name)
    out: list[str] = []
    if info.enum_member:
        out.append("runpod")
    if info.lambda_name:
        out.append("lambda")
    return tuple(out)


def gpu_short(name: str) -> str:
    """Short, endpoint-name-safe token for a GPU (e.g. '4090')."""
    return get_gpu_info(name).short


def min_cuda_modern(name: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the modern stack."""
    return get_gpu_info(name).min_cuda_modern or "12.8"


def cheapest_gpu(min_vram_gb: int) -> str:
    """Cheapest validated RunPod GPU class with at least ``min_vram_gb`` VRAM."""
    pool = [
        g for g in GPU_INFO.values() if g.enum_member and g.vram_gb >= min_vram_gb and g.validated
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
    """Cheapest validated GPU for this model: parse-time provisional used by the schema for sizing/display."""
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.allocator import vram_headroom

    min_vram = model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )
    return cheapest_gpu(min_vram)


@dataclass
class JobHandle:
    """Provider-tagged persisted handle: enough to reattach/cancel from any process."""

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
    # failure: job_failed, oom, job_preempted, no_capacity, stalled, or poll_error.
    failure: str | None = None
    detail: str | None = None


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
    candidates: tuple[Candidate, ...]  # full ranked list; retry walks this


@runtime_checkable
class Provider(Protocol):
    """GPU-substrate interface implemented by each provider."""

    name: str

    def is_configured(self) -> bool:
        """Whether this provider is usable right now (creds present, net reachable)."""
        ...

    def preflight(self, require_hf: bool = True) -> list[str]:
        """Missing-config problems; empty list == ready."""
        ...

    def gpu_classes(self) -> list[GpuClass]:
        """GPU classes this provider can provision."""
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
        on_last_gpu: bool = False,
        code_prefix: str | None = None,
    ) -> PollResult:
        """Deploy/rent -> submit -> persist handle (via ``on_handle``) -> poll to terminal.

        ``on_last_gpu``: no further GPU attempt follows, so capacity backstops may wait longer.
        """
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

    def sweep_orphans(
        self,
        active_labels: set[str] | Callable[[], set[str]] | None = None,
        known_labels: set[str] | Callable[[], set[str]] | None = None,
    ) -> list[int | str]:
        """Destroy billable resources this provider owns that no live run claims.

        ``active_labels``: raw run ids of live runs (may be a callable resolved after listing, to
        close the launch race). ``known_labels``: universe of run ids this plane may own — reap only
        resources in this set and not in active_labels (multi-plane safety: two planes sharing one
        account only reap their own orphans). ``None`` = unscoped single-plane behavior (correct for
        single-plane prod). Returns destroyed resource ids.
        """
        ...
