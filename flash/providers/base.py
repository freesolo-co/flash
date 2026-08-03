"""Shared GPU-provider interface + GPU registry."""

from __future__ import annotations

import math
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
KNOWN = tuple(GPU_INFO)
VALIDATED = tuple(g.name for g in GPU_CLASSES if g.validated)

# fp8 KV cache is an Ada/Hopper/Blackwell feature; the training workers enable it exactly when the
# device compute capability is >= (8, 9) (see engine/worker/rl_train.py and opd_train.py). Sizing has
# no live device, so it infers fp8 support from a candidate class's ``sm`` string.
_FP8_KV_MIN_CAPABILITY = (8, 9)


def _sm_capability(sm: str) -> tuple[int, int]:
    """(major, minor) compute capability from a GpuClass ``sm`` string.

    ``'sm80' -> (8, 0)``, ``'sm89' -> (8, 9)``, ``'sm90' -> (9, 0)``, ``'sm100' -> (10, 0)``,
    ``'sm120' -> (12, 0)``. Unparseable -> ``(0, 0)`` (treated as pre-fp8)."""
    digits = sm[2:] if sm.startswith("sm") else sm
    if not digits.isdigit():
        return (0, 0)
    n = int(digits)
    return (n // 10, n % 10)


def supports_fp8_kv(gpu_name: str) -> bool:
    """Whether this GPU class's compute capability supports an fp8 KV cache (cc >= 8.9)."""
    return _sm_capability(get_gpu_info(gpu_name).sm) >= _FP8_KV_MIN_CAPABILITY


def max_non_fp8_kv_vram_gb() -> int:
    """Largest VRAM (GB) among validated GPU classes that do NOT support fp8 KV cache.

    A run whose requirement exceeds this can only be placed on a modern (cc >= 8.9) card, so a
    colocated vLLM rollout on it is guaranteed to use an fp8 KV cache — half the bytes a bf16
    estimate reserves. Derived from the catalog so it tracks any GPU-class changes."""
    return max(
        (
            g.vram_gb
            for g in GPU_CLASSES
            if g.validated and _sm_capability(g.sm) < _FP8_KV_MIN_CAPABILITY
        ),
        default=0,
    )


def _alias_keys(name: str) -> set[str]:
    """All accepted spellings of a friendly name (lowercased)."""
    base = name.lower()
    keys = {base, base.replace(" ", ""), base.replace(" ", "_"), base.replace(" ", "-")}
    if base.startswith("rtx "):
        tail = base[4:]
        keys |= {tail, tail.replace(" ", ""), tail.replace(" ", "_")}
    keys.add(f"nvidia {base}")
    return keys


_ALIAS_TARGETS: dict[str, set[str]] = {}
for _info in GPU_INFO.values():
    for _provider_name in (_info.name, _info.vast_name, *_info.vast_aliases):
        if not _provider_name:
            continue
        for _k in _alias_keys(_provider_name):
            _ALIAS_TARGETS.setdefault(_k, set()).add(_info.name)
_ALIASES: dict[str, str] = {
    key: next(iter(targets)) for key, targets in _ALIAS_TARGETS.items() if len(targets) == 1
}
# exact canonical names always resolve to themselves even if a provider spelling collides with them.
_ALIASES.update({_info.name.lower(): _info.name for _info in GPU_INFO.values()})
# full marketing names (nvidia-smi / runpod api) and historical aliases not covered by generic rules.
_ALIASES.update(
    {
        "nvidia geforce rtx 4090": "RTX 4090",
        "nvidia geforce rtx 5090": "RTX 5090",
        "nvidia a100 80gb pcie": "A100 PCIe",
        "a100 80gb pcie": "A100 PCIe",
        "a100-80g-pcie": "A100 PCIe",
        "nvidia a100-sxm4-40gb": "A100 SXM 40GB",
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
    """Normalize a friendly GPU name to a managed GPU class; raise otherwise."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnsupportedGpuError(f"unsupported gpu {name!r}; Flash manages {', '.join(KNOWN)}")


def get_gpu_info(name: str) -> GpuClass:
    return GPU_INFO[canonical_gpu(name)]


def gpu_classes_for(identity_attr: str) -> list[GpuClass]:
    """Managed GPU classes a provider can provision: those whose per-provider identity field is set
    (``enum_member`` for RunPod, ``lambda_name`` for Lambda, ``vast_name`` for Vast). One catalog query
    shared by every provider's ``gpu_classes()`` so the "which classes do I offer" rule can't drift."""
    return [g for g in GPU_INFO.values() if getattr(g, identity_attr)]


def static_rates_for(identity_attr: str) -> dict[str, float]:
    """Friendly GPU name -> static ``GpuClass.hourly_usd`` for the classes a provider offers (keyed by
    the same per-provider identity field). The offline/fallback rate snapshot for RunPod and Vast;
    Lambda keeps its own list-price map because its prices differ from this RunPod-snapshot field."""
    return {
        name: info.hourly_usd for name, info in GPU_INFO.items() if getattr(info, identity_attr)
    }


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


# Assumed run length when a spec sets no max_steps, used only to amortize the once-per-run block
# across a run whose true horizon is not knowable during selection (it derives from dataset row
# counts, which the allocator does not have). It is the MEDIAN step count of the measured corpus
# (88 rollout arms: min 3, median 8, max 48), so it is the length a typical run actually has.
#
# Its blast radius was measured rather than assumed. Sweeping the assumed horizon over 1..200 across
# a grid of completion and context shapes, the winning card is stable across wide ranges and flips
# at only a few points, so the worst-case overpay from assuming wrong is bounded:
#   assumed   1: 1.367x   <- the old steps=1 behaviour, the worst end of the range
#   assumed   8: 1.269x   <- this value, the best of the fixed choices tested
#   assumed  16: 1.324x
#   assumed  32: 1.395x
#   assumed  64: 1.395x
# A spec WITH max_steps never uses this: it ranks on its real horizon and none of the above applies.
_RANKING_STEPS_FALLBACK = 8


def run_config_for_ranking(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
):
    """Raw train knobs -> a ``RunConfig`` to rank hardware against.

    The single place spec knobs become a priced run, so every selection path (parse-time
    provisional, submit-time allocator, cost estimate) ranks on identical inputs. Imported lazily
    because the cost model imports this module.

    The step count is a real input, not a placeholder. It used to be pinned at 1 on the reasoning
    that a per-step key never sees the run's length; that stopped being true once a once-per-run
    block entered the model, because ``steps=1`` gives that block the weight of a single step (see
    ``cost.analytical.step_cost_key``). ``max_steps`` is read through the same ``knob`` helper as
    every other spec value, which reaches both call sites unchanged -- the schema path passes a dict
    and the lifecycle path an object, and ``knob`` already handles both.

    ``_RANKING_STEPS_FALLBACK`` covers a spec with no ``max_steps``, where the true horizon comes
    from dataset row counts the allocator does not have and must not fetch mid-selection.
    """
    from flash.cost.types import RunConfig

    def knob(key):
        """A positive whole number from the spec, or None to take the recipe default.

        Bools are rejected before the numeric coercion because ``bool`` is an ``int`` subclass, so
        a stray ``True`` would otherwise read as the number 1.
        """
        if train is None:
            return None
        value = train.get(key) if isinstance(train, dict) else getattr(train, key, None)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 1:
            return None
        return int(number)

    return RunConfig(
        model_id=model_id,
        method=algorithm,
        steps=knob("max_steps") or _RANKING_STEPS_FALLBACK,
        seq_len=knob("max_context_tokens"),
        completion_len=knob("max_completion_tokens"),
        batch_size=knob("batch_size"),
        group_size=knob("group_size"),
        lora_rank=knob("lora_rank"),
        thinking=thinking,
        model_revision=model_revision,
    )


def _run_cost_key(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
):
    """``(gpu, hourly_rate) -> dollars per step`` for this run, or None if it can't be priced."""
    try:
        from flash.cost.analytical import step_cost_key

        return step_cost_key(
            run_config_for_ranking(
                model_id,
                algorithm,
                train=train,
                thinking=thinking,
                model_revision=model_revision,
            )
        )
    except Exception:  # unpriceable run -- rank on $/hr, never fail selection
        return None


# FSDP shards parameters/optimizer across cards but replicates activations and adds collective
# buffers; a combination only counts as fitting when the combined VRAM clears the need with this
# margin. conservative on purpose: a too-optimistic shard model OOMs a paid run.
SHARD_VRAM_EFFICIENCY = 0.85
# activations/buffers replicate PER CARD regardless of sharding; every card in a combination must
# individually hold this floor on top of its parameter shard, so tiny cards cannot fake a fit by
# count alone (e.g. 4 x 12 GB is not 40 GB of usable capacity for a 24 GB-activation job).
REPLICATED_PER_CARD_GB = 8
# combinations larger than this are never proposed: shard-efficiency loss and inter-card overhead
# grow with count, and cost estimates get less reliable.
MAX_COMBINATION_CARDS = 4


def combined_vram_gb(vram_gb: int, gpu_count: int) -> float:
    """Run-usable VRAM across ``gpu_count`` cards of ``vram_gb`` each.

    THE single fit model: parse-time sizing, the submit-time allocator's candidate filter, and its
    pinned-class check all call this. They used to carry three copies of the arithmetic, which is
    how parse-time drifted into rejecting shapes submit-time would have accepted.

    Returns 0.0 for a multi-card box whose cards cannot individually clear the replicated floor --
    such a box has no usable capacity at all, rather than a small amount.
    """
    if gpu_count <= 1:
        return float(vram_gb)
    if vram_gb <= REPLICATED_PER_CARD_GB:
        return 0.0
    return gpu_count * (vram_gb - REPLICATED_PER_CARD_GB) * SHARD_VRAM_EFFICIENCY + (
        REPLICATED_PER_CARD_GB
    )


def cheapest_gpu(min_vram_gb: int, *, gpu_count: int = 1, cost_key=None) -> str:
    """Cheapest validated GPU class whose ``gpu_count``-card shape holds ``min_vram_gb``.

    ``gpu_count`` is the run's card ceiling. Sizing a multi-card run against one card is what made
    ``--gpus 4`` inert: the spec was rejected here before the allocator ever got to shard it.

    ``cost_key`` is ``(gpu_name, hourly_rate) -> comparable``; when given, classes are ranked by
    what the JOB costs on each rather than by rental rate, matching the submit-time allocator.
    Omitted, this ranks on $/hr for callers with no run to price.
    """
    # the allocator never proposes a wider combination, so sizing against one would admit a spec
    # here only for submit to reject it -- the same defect this parameter exists to fix, inverted.
    # size against the largest count providers actually RENT (powers of two): a ceiling of 3 buys
    # 2 cards at submit, so sizing on 3 would admit a shape that never gets provisioned.
    cards = largest_rentable_count(gpu_count)
    pool = [
        g
        for g in GPU_INFO.values()
        if g.enum_member and g.validated and combined_vram_gb(g.vram_gb, cards) >= min_vram_gb
    ]
    if not pool:
        shape = f" even as a {cards}-card combination" if cards > 1 else ""
        raise UnsupportedGpuError(f"no validated GPU class has >= {min_vram_gb} GB VRAM{shape}")
    from flash.providers.runpod.pricing import hourly_rate

    if cost_key is not None:
        return min(
            pool,
            key=lambda g: (cost_key(g.name, hourly_rate(g.name)), hourly_rate(g.name), g.vram_gb),
        ).name
    return min(pool, key=lambda g: (hourly_rate(g.name), g.vram_gb)).name


def provisional_gpu(
    model_id: str,
    algorithm: str = "sft",
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
    gpu_count: int = 1,
) -> str:
    """Cheapest validated GPU for this model: parse-time provisional used by the schema for sizing/display.

    ``gpu_count`` is the run's card ceiling, so a run allowed to shard is sized against the shape it
    will actually be allocated. The returned class name is per-card either way.
    """
    from flash.engine.vram import model_required_vram_gb
    from flash.providers.allocator import vram_headroom

    min_vram = model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
        model_revision=model_revision,
    )
    try:
        # same job-cost basis the submit-time allocator ranks on, so the class previewed here is
        # the class that actually gets provisioned; falls back to $/hr for an unpriceable run.
        return cheapest_gpu(
            min_vram,
            gpu_count=gpu_count,
            cost_key=_run_cost_key(
                model_id, algorithm, train=train, thinking=thinking, model_revision=model_revision
            ),
        )
    except UnsupportedGpuError as exc:
        if (algorithm or "").lower() == "opd":
            cards = max(1, min(int(gpu_count), MAX_COMBINATION_CARDS))
            biggest = max(
                (
                    combined_vram_gb(g.vram_gb, cards)
                    for g in GPU_CLASSES
                    if g.enum_member and g.validated
                ),
                default=0,
            )
            # OPD is resident-only: the HF/PEFT trainer AND the colocated vLLM student rollout engine
            # both stay resident, so the card must hold two model-weight copies plus the rollout KV
            # cache (which scales with batch_size x group_size) at the loss backward peak. Point the
            # user at the knobs that shrink it instead of the opaque "no GPU that big" message.
            shape = (
                f"any {cards}-card validated GPU combination"
                if cards > 1
                else ("any single validated GPU")
            )
            raise UnsupportedGpuError(
                f"opd needs >= {min_vram} GB VRAM, more than {shape} "
                f"({biggest:g} GB max). opd is resident-only: the trainer and the colocated vLLM student "
                f"rollout engine hold two model-weight copies plus the rollout KV cache at once. "
                f"Lower [train].group_size and/or [train].batch_size (rollout concurrency = "
                f"batch_size x group_size; distillation needs no group variance, so group_size=1 is "
                f"fine) and/or [train].max_completion_tokens / [train].max_context_tokens to fit."
            ) from exc
        raise


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
        provider = d.pop("provider", None)
        if not isinstance(provider, str) or not provider:
            raise ValueError("persisted provider identity is missing or invalid")
        return cls(provider=provider, data=d)


@dataclass
class PollResult:
    ok: bool
    metrics: dict | None = None
    # failure: job_failed, oom, job_preempted, no_capacity, stalled, or poll_error.
    failure: str | None = None
    detail: str | None = None


def rentable_gpu_counts(max_gpu_count: int) -> tuple[int, ...]:
    """Card counts worth asking a provider about, largest first, up to ``max_gpu_count``.

    Powers of two only. Every provider sells multi-card boxes in powers of two, and the trainer
    shards over exactly the cards it rents: verl asserts ``num_attention_heads % sp_size == 0``, so
    a count that does not divide the head count aborts at step 0 rather than training slowly.
    Catalog head counts are 8, 16, or 24 -- every one divisible by 1, 2, 4, and 8, and none by an
    odd count above 1 (24 % 3 == 0 but 8 % 3 != 0). Powers of two are therefore the counts that are
    safe for every model, so no unrunnable shape is ever allocated.
    """
    cap = max(1, int(max_gpu_count))
    counts, count = [], 1
    while count <= cap:
        counts.append(count)
        count *= 2
    return tuple(reversed(counts))


def largest_rentable_count(max_gpu_count: int) -> int:
    """Widest shape a ``max_gpu_count`` ceiling can actually be provisioned as.

    Sizing gates must use this rather than the raw ceiling: only powers of two up to
    ``MAX_COMBINATION_CARDS`` are ever offered (see ``rentable_gpu_counts``), so a ceiling of 3
    buys 2 cards. Sizing on 3 would admit a spec that submit can never rent.
    """
    return rentable_gpu_counts(min(max(1, int(max_gpu_count)), MAX_COMBINATION_CARDS))[0]


@dataclass(frozen=True)
class Candidate:
    provider: str
    gpu: str
    hourly_usd: float
    vram_gb: int
    # cards in this candidate combination; hourly_usd and vram_gb stay PER-CARD so existing
    # single-gpu consumers are unchanged. total cost = gpu_count * hourly_usd.
    gpu_count: int = 1

    @property
    def total_hourly_usd(self) -> float:
        return self.gpu_count * self.hourly_usd

    @property
    def total_vram_gb(self) -> int:
        return self.gpu_count * self.vram_gb


@dataclass(frozen=True)
class Allocation:
    provider: str
    gpu: str
    hourly_usd: float
    min_vram_gb: int
    candidates: tuple[Candidate, ...]  # full ranked list; retry walks this
    # cards in the chosen combination (1 = classic single-gpu allocation; hourly_usd is per-card)
    gpu_count: int = 1


@dataclass(frozen=True)
class AllocationConstraints:
    """Run-scoped extras a capacity/market-aware provider's ``live_candidates`` needs (Vast prices
    against the run's disk/duration floors) — carried here so they don't leak into ``allocate``'s
    signature per-provider. A provider ignores the fields it has no use for."""

    disk_gb: float = 0.0
    max_wall_seconds: float = 0.0
    gpu_type: str = ""
    # Ceiling on cards per machine the caller can use (1 = single-card only). Each provider reports
    # the counts it can ACTUALLY rent on one box, which differ in kind: RunPod takes a count
    # parameter, Lambda names the count in the instance type, Vast has it baked into the offer. The
    # allocator owns the fit/cost policy and never assumes a count is rentable just because the
    # single-card class exists.
    max_gpu_count: int = 1


@runtime_checkable
class Provider(Protocol):
    """GPU-substrate interface implemented by each provider."""

    name: str

    def is_configured(self) -> bool:
        """Whether this provider is enabled on this control plane; does not probe network reachability."""
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
        _deadline_at: float | None = None,
    ) -> PollResult:
        """Deploy/rent -> submit -> persist handle (via ``on_handle``) -> poll to terminal.

        ``on_last_gpu``: no further GPU attempt follows, so capacity backstops may wait longer.
        """
        ...

    def poll(
        self,
        handle: JobHandle,
        spec: JobSpec,
        seed: int,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
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
