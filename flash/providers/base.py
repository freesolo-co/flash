"""Shared GPU-provider interface + the provider-agnostic GPU registry.

Both substrates (RunPod Flash, Vast.ai verified datacenters) implement the SAME
``Provider`` protocol and expose the SAME module set under ``providers/<name>/`` so a
provider is pluggable/swappable. This module owns the parts that are NOT provider
specific:

  * ``GpuClass`` — one managed GPU class with its per-provider identity
    (``enum_member`` for RunPod, ``vast_name`` for Vast) and per-provider
    ``validated_on``. Each provider owns *which* classes it lists (its ``gpus.py``
    carves its rows out of ``GPU_CLASSES``), but the class table itself is shared so a
    friendly name canonicalizes to one identity everywhere (catalog, config, serving).
  * ``JobHandle`` / ``PollResult`` — the persisted-handle + poll-outcome shapes the
    orchestrator round-trips through any provider.
  * ``Candidate`` / ``Allocation`` — the cross-provider allocation result.
  * The canonicalization / alias / policy helpers every call site already used.

The ``Provider`` protocol is the FIXED method set both providers implement; the
orchestrator dispatches cancel/poll/destroy generically through the persisted
handle's ``provider`` key. The post-run GC backstop is the deliberate exception:
RunPod's ``gc`` runs unconditionally (a name-reconstruction backstop for rN-suffixed
endpoints the persisted handle can't name) and Vast's ``gc`` is called by name only
when Vast is available (its billing-leak reap), so that path branches per provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flash.spec import JobSpec


# ---------------------------------------------------------------------------
# GPU class registry (provider-agnostic identity + per-provider validation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuClass:
    """One managed GPU class: a friendly name + per-provider identity/metadata.

    Provider-agnostic by design — the identity columns (``enum_member`` for RunPod's
    Flash ``GpuType``; ``vast_name`` for the Vast offer ``gpu_name``) and
    ``validated_on`` carry the per-provider facts, but a class is a single canonical
    row so the catalog / config / serving all agree on what e.g. "RTX 5090" is.
    """

    name: str  # canonical friendly name used in configs / the catalog
    enum_member: str | None  # runpod_flash GpuType member name; None -> not on RunPod
    vram_gb: int
    short: str  # endpoint-name-safe token (e.g. "4090", "a5000")
    sm: str  # CUDA arch (informational; sm80+ only)
    hourly_usd: float  # static fallback rate; live pricing overrides (pricing.py)
    # Providers where this class passed Flash's live train+eval smoke. Validation is
    # per-provider: the same silicon behind a different provisioning path (Flash deps
    # install vs a Vast docker image) is a different failure surface.
    validated_on: tuple[str, ...] = ()
    # Min host CUDA (driver) on the modern stack. None -> 12.8. Blackwell (sm120/sm100)
    # needs CUDA-13 drivers to JIT the wheels' PTX (no SASS shipped).
    min_cuda_modern: str | None = None
    # Vast.ai offer ``gpu_name`` for this class; None -> not provisionable on Vast.
    # A100 SXM4 boards exist in 40 GB and 80 GB variants under ONE Vast name — offers
    # are disambiguated by ``gpu_ram`` (see ``vast_gpu_for_offer``).
    vast_name: str | None = None

    @property
    def validated(self) -> bool:  # validated on ANY provider
        return bool(self.validated_on)


# Fallback hourly rates are RunPod secure-cloud on-demand (snapshot 2026-06-11); live
# rates from the provider pricing module override them. Vast-only classes
# (enum_member=None) carry a Vast verified-datacenter snapshot instead.
GPU_CLASSES: tuple[GpuClass, ...] = (
    # ---- validated: passed the full train+eval matrix (bench/results/phase1) ----
    GpuClass(
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
    # datacenter ($0.60/hr South Korea), incl. vLLM eval on a CUDA-13 driver.
    GpuClass(
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
    GpuClass("RTX A4000", "NVIDIA_RTX_A4000", 16, "a4000", "sm86", 0.25, vast_name="RTX A4000"),
    GpuClass(
        "RTX 2000 Ada",
        "NVIDIA_RTX_2000_ADA_GENERATION",
        16,
        "2000ada",
        "sm89",
        0.24,
        vast_name="RTX 2000Ada",
    ),
    GpuClass("RTX A4500", "NVIDIA_RTX_A4500", 20, "a4500", "sm86", 0.25, vast_name="RTX A4500"),
    GpuClass(
        "RTX 4000 Ada",
        "NVIDIA_RTX_4000_ADA_GENERATION",
        20,
        "4000ada",
        "sm89",
        0.26,
        vast_name="RTX 4000Ada",
    ),
    # Validated 2026-06-11: Qwen3-0.6B SFT + GRPO smokes passed — cheapest 24 GB class.
    GpuClass(
        "RTX A5000",
        "NVIDIA_RTX_A5000",
        24,
        "a5000",
        "sm86",
        0.27,
        validated_on=("runpod",),
        vast_name="RTX A5000",
    ),
    # Vast-validated 2026-06-12: Qwen3-0.6B SFT train+eval smoke ($0.25/hr Czechia).
    GpuClass(
        "RTX 3090",
        "NVIDIA_GEFORCE_RTX_3090",
        24,
        "3090",
        "sm86",
        0.46,
        validated_on=("vast",),
        vast_name="RTX 3090",
    ),
    GpuClass("L4", "NVIDIA_L4", 24, "l4", "sm89", 0.39, vast_name="L4"),
    # Blackwell workstation card; cheap verified-datacenter capacity on Vast.
    # Vast-validated 2026-06-12: Qwen3-0.6B SFT train+eval smoke incl. vLLM eval on a
    # CUDA-13 driver with the cu128 stack image ($0.34/hr Hungary). Vast-only.
    GpuClass(
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
    GpuClass("RTX A6000", "NVIDIA_RTX_A6000", 48, "a6000", "sm86", 0.49, vast_name="RTX A6000"),
    GpuClass("A40", "NVIDIA_A40", 48, "a40", "sm86", 0.44, vast_name="A40"),
    GpuClass(
        "RTX 6000 Ada",
        "NVIDIA_RTX_6000_ADA_GENERATION",
        48,
        "6000ada",
        "sm89",
        0.77,
        vast_name="RTX 6000Ada",
    ),
    # L40S exists at RunPod but not in the Flash SDK's GpuType enum -> Vast-only.
    GpuClass("L40S", None, 48, "l40s", "sm89", 0.87, vast_name="L40S"),
    # ---- big-VRAM tier (large-MoE QLoRA, future >9B bf16) ----
    # 40 GB SXM4 boards share Vast's "A100 SXM4" name with the 80 GB variant; offers
    # are split by gpu_ram (vast_gpu_for_offer). Not a RunPod Flash class -> Vast-only.
    GpuClass("A100 SXM 40GB", None, 40, "a100sxm40", "sm80", 0.89, vast_name="A100 SXM4"),
    # Validated 2026-06-11: 0.6B SFT smoke (phase6).
    GpuClass(
        "A100 PCIe",
        "NVIDIA_A100_80GB_PCIe",
        80,
        "a100pcie",
        "sm80",
        1.39,
        validated_on=("runpod",),
        vast_name="A100 PCIE",
    ),
    GpuClass(
        "A100 SXM", "NVIDIA_A100_SXM4_80GB", 80, "a100sxm", "sm80", 1.49, vast_name="A100 SXM4"
    ),
    GpuClass("H100", "NVIDIA_H100_80GB_HBM3", 80, "h100", "sm90", 3.29, vast_name="H100 SXM"),
    # H100 NVL (94 GB) has no RunPod Flash GpuType member -> Vast-only. Cheaper than the
    # 80 GB SXM H100 on the live market and carries 14 GB more VRAM, so it's a strong
    # cost/VRAM pick for big-context GRPO tiers.
    GpuClass(
        "H100 NVL", None, 94, "h100nvl", "sm90", 2.39, validated_on=("vast",), vast_name="H100 NVL"
    ),
    GpuClass(
        "RTX Pro 6000",
        "NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION",
        96,
        "pro6000",
        "sm120",
        2.09,
        min_cuda_modern="13.0",
    ),
    # RTX Pro 6000 Blackwell Workstation Edition: same 96 GB die as the Server Edition,
    # a distinct RunPod GpuType, typically a touch cheaper. Also offered on Vast. The
    # single biggest non-datacenter 96 GB option -> cheapest 80 GB-floor GRPO host.
    GpuClass(
        "RTX Pro 6000 WK",
        "NVIDIA_RTX_PRO_6000_BLACKWELL_WORKSTATION_EDITION",
        96,
        "pro6000wk",
        "sm120",
        1.79,
        validated_on=("runpod", "vast"),
        min_cuda_modern="13.0",
        vast_name="RTX PRO 6000",
    ),
)

GPU_INFO: dict[str, GpuClass] = {g.name: g for g in GPU_CLASSES}

# Canonical friendly names Flash exposes in configs / the catalog.
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
        f'unsupported gpu {name!r}; Flash manages {", ".join(KNOWN)} (or gpu.type = "cheapest")'
    )


def get_gpu_info(name: str) -> GpuClass:
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
    # Truthy allowlist (not a falsey denylist): only an explicit truthy value opts in, so
    # "false"/"False"/"no"/"off"/"0"/"" all correctly leave unvalidated GPUs disabled.
    return os.environ.get("AUTOSLM_GPU_ALLOW_UNVALIDATED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gpu_short(name: str) -> str:
    """Short, endpoint-name-safe token for a GPU (e.g. '4090')."""
    return get_gpu_info(name).short


def min_cuda_modern(name: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the modern stack."""
    return get_gpu_info(name).min_cuda_modern or "12.8"


def cheapest_gpu(min_vram_gb: int, include_unvalidated: bool = False) -> str:
    """Cheapest RunPod GPU class with at least ``min_vram_gb`` VRAM (live rates, cached).

    RunPod-static by design (the cross-provider equivalent lives in
    ``flash.providers.allocator``): Vast-only classes are excluded so the result is
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
    from flash.providers.runpod.pricing import hourly_rate

    return min(pool, key=lambda g: (hourly_rate(g.name), g.vram_gb)).name


def resolve_gpu_policy(
    requested: str,
    model_id: str,
    allow_unvalidated: bool | None = None,
    algorithm: str = "sft",
    *,
    train=None,
    thinking: bool = False,
) -> str:
    """Resolve ``gpu.type`` (a concrete class or a policy word) to a friendly name.

    Parse-time, RunPod-static provisional: "cheapest"/"auto" pick the cheapest
    RunPod-validated class whose VRAM covers the model; concrete names are
    canonicalized. The submit-time allocator (``flash.providers.allocator``)
    re-resolves policy words live across providers.
    """
    key = (requested or "").strip().lower()
    if key not in POLICY_NAMES:
        return canonical_gpu(requested)
    # Use the allocator's disaggregated-AWARE sizing (which itself honors AUTOSLM_VRAM_HEADROOM
    # via vram_headroom, so parse-time matches submit-time exactly — PR #176). For a
    # disaggregated GRPO run ([train].inference_gpus>0) the rollout server is tensor-parallel
    # across the inference GPUs and the trainer is on separate cards, so the binding PER-GPU need
    # is far below the colocate total. Sizing the parse-time provisional with the colocate total
    # would raise here (e.g. "no validated GPU has >= 103 GB") and block the advertised
    # disaggregated-only model BEFORE the submit-time allocator can apply the smaller per-GPU
    # estimate. required_vram_gb falls back to the plain colocate figure when inference_gpus==0.
    from flash.providers.allocator import required_vram_gb

    min_vram = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    return cheapest_gpu(min_vram, include_unvalidated=unvalidated_allowed(allow_unvalidated))


# ---------------------------------------------------------------------------
# Handles + poll outcomes (round-tripped through any provider)
# ---------------------------------------------------------------------------
@dataclass
class JobHandle:
    """Provider-tagged, persisted handle: enough to reattach/cancel from any process.

    Each provider owns the rest of its handle shape (RunPod: endpoint_id/job_id; Vast:
    instance_id/offer_id/...). ``provider`` is the routing key the orchestrator uses to
    dispatch poll/cancel/destroy generically through the registry.
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
    failure: str | None = None  # "job_failed" | "stalled" | "poll_error"
    detail: str | None = None


# ---------------------------------------------------------------------------
# Allocation result (cross-provider)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    provider: str
    gpu: str
    hourly_usd: float
    vram_gb: int
    validated: bool
    # Opaque per-provider provisioning hint (e.g. the chosen Vast offer). The
    # allocator stays provider-agnostic; the provider interprets it at submit time.
    offer: Any = None


@dataclass(frozen=True)
class Allocation:
    provider: str
    gpu: str
    hourly_usd: float
    min_vram_gb: int
    candidates: tuple[Candidate, ...]  # full ranked list (retry walks this)
    offer: Any = None  # the chosen provider's provisioning hint (vast offer | None)
    # Per-provider book of provisioning hints for the live-market walk (vast offers).
    provider_offers: tuple[Any, ...] = ()


# ---------------------------------------------------------------------------
# The provider interface (FIXED method set both providers implement)
# ---------------------------------------------------------------------------
@runtime_checkable
class Provider(Protocol):
    """The pluggable GPU-substrate interface.

    Both ``providers/runpod`` and ``providers/vast`` expose ``PROVIDER`` implementing
    this protocol with an identical module layout (api/auth/pricing/gpus/jobs/
    train/preflight). The orchestrator/allocator only ever talk to these methods, so a
    provider is swappable without touching the control plane.
    """

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
        """$/hr for one friendly GPU name (live if available, else static)."""
        ...

    def submit_run(
        self,
        spec: JobSpec,
        seed: int,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        offers: Any = None,
        exclude_machine_ids: Any = frozenset(),
    ) -> PollResult:
        """Deploy/rent -> submit -> persist handle (via ``on_handle``) -> poll.

        ``exclude_machine_ids`` is the run's blacklist (machines that already failed
        this run); a provider that re-searches the live market mid-submit (Vast) must
        keep them excluded so a stalled/sick machine is never re-picked. RunPod ignores
        it (no in-provider market re-search)."""
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
