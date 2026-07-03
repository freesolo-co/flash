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
    vast_name: str | None = None  # None -> not on Vast; priced from Vast live/static rates
    vast_aliases: tuple[str, ...] = ()


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
        vast_name="RTX 4090",
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
        vast_name="RTX 5090",
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
        vast_name="A100 SXM4",
        vast_aliases=("A100 PCIE",),
    ),
    GpuClass(
        "A100 PCIe",
        "NVIDIA_A100_80GB_PCIe",
        80,
        "a100pcie",
        "sm80",
        1.39,
        validated=True,
        vast_name="A100 PCIE",
    ),
    GpuClass(
        "A100 SXM",
        "NVIDIA_A100_SXM4_80GB",
        80,
        "a100sxm",
        "sm80",
        1.49,
        validated=True,
        vast_name="A100 SXM4",
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
        vast_name="H100 SXM",
        # Vast lists PCIe H100s separately as "H100 PCIE"; accept them as H100 capacity (same sm90 / 80
        # GB; for our single-GPU runs the SXM-vs-PCIe interconnect gap doesn't apply). Priced live off
        # the actual offer, so a cheaper PCIe board just makes the class more available.
        vast_aliases=("H100 PCIE",),
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
LEGACY_GPU_CLASSES: tuple[GpuClass, ...] = (
    GpuClass(
        "RTX A6000",
        "NVIDIA_RTX_A6000",
        48,
        "a6000",
        "sm86",
        0.49,
        lambda_name="gpu_1x_a6000",
        vast_name="RTX A6000",
    ),
)
_GPU_INFO_ALL: dict[str, GpuClass] = {**GPU_INFO, **{g.name: g for g in LEGACY_GPU_CLASSES}}
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
for _info in _GPU_INFO_ALL.values():
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


class CapacityLookupError(RuntimeError):
    """A provider's LIVE capacity/offer lookup failed transiently (network / API blip / rate limit) —
    distinct from ``UnsupportedGpuError`` ("no GPU class fits this job"). A per-provider failure degrades
    to the other providers; only when it was the SOLE reason NO candidate was found does ``allocate``
    re-raise it. Because it is NOT an ``UnsupportedGpuError``, the runner treats it as infra-retryable
    (poll_error) rather than terminal — so a run whose only fitting capacity a transient outage hid is
    retried on its infra budget instead of being killed."""


class UnreconciledCreateError(RuntimeError):
    """A non-idempotent provider create (e.g. Vast's ``PUT /asks``) failed AMBIGUOUSLY and could NOT be
    reconciled: the possibly-created resource is not visible yet (object-store / API eventual
    consistency), so we cannot adopt it and we cannot prove it does not exist. Retrying the run would
    rent a SECOND instance while a phantom from this attempt may still materialize and bill under the
    still-active run (where ``sweep_orphans`` shields it). The orchestrator must therefore FAIL THE RUN
    TERMINALLY rather than consume a retry — the run's teardown plus a later sweep (the run is now
    inactive, so no longer shielded) reclaim any late-materializing instance, preserving the
    cost-safety invariant that a rented box is always destroyed."""


def canonical_gpu(name: str) -> str:
    """Normalize a friendly GPU name to an active or retired class; raise otherwise.

    Retired classes resolve for teardown/pricing compatibility but stay absent from ``GPU_INFO`` so
    allocation, display, and provider offer matching do not select them.
    """
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnsupportedGpuError(f"unsupported gpu {name!r}; Flash manages {', '.join(KNOWN)}")


def get_gpu_info(name: str) -> GpuClass:
    return _GPU_INFO_ALL[canonical_gpu(name)]


def gpu_classes_for(identity_attr: str) -> list[GpuClass]:
    """Managed GPU classes a provider can provision: those whose per-provider identity field is set
    (``enum_member`` for RunPod, ``lambda_name`` for Lambda, ``vast_name`` for Vast). One catalog query
    shared by every provider's ``gpu_classes()`` so the "which classes do I offer" rule can't drift."""
    return [g for g in GPU_INFO.values() if getattr(g, identity_attr)]


def static_rates_for(identity_attr: str) -> dict[str, float]:
    """Friendly GPU name -> static ``GpuClass.hourly_usd`` for the classes a provider offers (keyed by
    the same per-provider identity field). The offline/fallback rate snapshot for RunPod and Vast;
    Lambda keeps its own list-price map because its prices differ from this RunPod-snapshot field."""
    return {name: info.hourly_usd for name, info in GPU_INFO.items() if getattr(info, identity_attr)}


# Slack between a board's REPORTED VRAM and its class nominal (boards under-report: an A100 SXM4 40 GB
# reports ~40960 MB, an A40 ~46068 MB / 48 GB). vast_gpu_for_offer allows a class whose nominal is at
# most this far ABOVE the offer's reported RAM, so a real board still matches its class.
_VRAM_MATCH_TOLERANCE_GB = 3.5


def vast_gpu_for_offer(gpu_name: str, gpu_ram_mb: float) -> str | None:
    """Map a Vast offer (``gpu_name`` + ``gpu_ram`` MB) to a canonical managed GPU class.

    Returns None for anything not in the managed table — the hard Ampere+ floor (T4 / 2080 Ti /
    Quadro RTX offers never match). Names shared across VRAM variants ("A100 SXM4" = 40/80 GB) resolve
    to the LARGEST class the board's actual RAM covers.
    """
    fitting = [
        g
        for g in GPU_INFO.values()
        if (g.vast_name == gpu_name or gpu_name in g.vast_aliases)
        and g.vram_gb <= gpu_ram_mb / 1024 + _VRAM_MATCH_TOLERANCE_GB
    ]
    if not fitting:
        return None
    return max(fitting, key=lambda g: g.vram_gb).name


def providers_for(name: str) -> tuple[str, ...]:
    """Providers that can provision this GPU class."""
    info = get_gpu_info(name)
    out: list[str] = []
    if info.enum_member:
        out.append("runpod")
    if info.lambda_name:
        out.append("lambda")
    if info.vast_name:
        out.append("vast")
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


@dataclass(frozen=True)
class AllocationConstraints:
    """Run-scoped extras a capacity/market-aware provider's ``live_candidates`` needs (Vast prices
    against the run's disk/duration floors) — carried here so they don't leak into ``allocate``'s
    signature per-provider. RunPod/Lambda ignore them."""

    disk_gb: float = 0.0
    max_wall_seconds: float = 0.0


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

    def live_candidates(
        self, need_vram_gb: int, constraints: AllocationConstraints
    ) -> list[Candidate]:
        """GPU-class candidates this provider can actually provision right now for a run needing >=
        need_vram_gb VRAM. RunPod filters its static table and never raises; capacity/market-aware
        providers (Lambda/Vast) query live availability and raise CapacityLookupError on a transient
        lookup blip so allocate() can degrade to the others yet still tell 'no fit' from 'outage'."""
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

    # NOTE: ``supports_weight_cache: bool`` is an OPTIONAL capability attr (read via
    # ``getattr(prov, "supports_weight_cache", False)``, off this Protocol for the same isinstance
    # reason as below): True only for the provider that offers the shared weight-cache network volume
    # (RunPod). The runner gates its one-shot cache-less retry fallback on it; every other provider
    # defaults False.

    # NOTE: ``run_instances_remaining(run_id) -> list[int]`` is an OPTIONAL capability, intentionally
    # NOT declared on this ``@runtime_checkable`` Protocol — adding it would make it a REQUIRED member
    # for ``isinstance(prov, Provider)``, which RunPod (serverless, self-reaping — nothing to enumerate)
    # and Lambda do not implement. Instance providers that CAN enumerate billable resources by run
    # label (Vast) implement it so the handle-less recovery resubmit can require a CONFIRMED reap before
    # launching a second worker (a best-effort ``gc`` returns no error on an unconfirmed teardown).
    # Callers detect it via ``getattr(prov, "run_instances_remaining", None)`` (see server/_runtime.py).
    # Contract: ``[]`` == CONFIRMED no resource for the run remains; non-empty == a possibly-live one
    # survives; RAISES on an incomplete enumeration so a caller can't mistake "couldn't list" for "clear".

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
