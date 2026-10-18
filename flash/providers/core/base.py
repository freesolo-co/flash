"""Shared GPU-provider interface + GPU registry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from flash.providers.core.sharding import MAX_COMBINATION_CARDS, combined_vram_gb

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from flash.core.spec import JobSpec


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
    """A transient live-capacity lookup failure, distinct from no fitting GPU.

    Allocation can degrade to other providers, but re-raises when an outage alone hid all fitting
    capacity so the runner uses its infrastructure retry budget.
    """


class CapacityUnavailableError(CapacityLookupError):
    """A live provider answered successfully but currently offers no fitting capacity."""


class UnreconciledCreateError(RuntimeError):
    """An ambiguous non-idempotent create could not be reconciled.

    Retrying could double-provision while the first resource materializes later. Fail terminally so
    teardown and the later unshielded orphan sweep can reclaim it.
    """


class RunExhaustedProviderPoolError(RuntimeError):
    """Every fitting offer for this class is one THIS run already rented and lost.

    Distinct from an empty pool: the operator fix is another class or provider, not waiting. The
    message is authored by Flash rather than quoted from a provider response, which is what lets
    supervision surface it verbatim -- generic provider text is withheld from the run record
    because it can quote a request that carried a credential.
    """


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
    """Map a Vast name and reported VRAM to a managed GPU class.

    Unmanaged and pre-Ampere offers return None. Shared names resolve to the largest class whose
    nominal VRAM the board covers.
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


def run_config_for_ranking(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
    overrides: dict | None = None,
):
    """Build the one-step RunConfig shared by every hardware-ranking path.

    Ranking is per step, so run length is irrelevant. Import lazily because the cost model imports
    this module.

    ``overrides`` carries knobs that come from the frozen workload profile rather than the authored
    ``train`` table -- see ``flash.cost.spec.sft_ranking_overrides``, which is the single place they
    are derived. They are applied last because the profile measured what will actually execute and
    the authored value is only the request.
    """
    from flash.core.catalog import samples_on_policy
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

    def rollout_batch():
        """Prompts a rollout step really trains on: the authored batch, capped by the pool.

        Both workers retain at most ``max_examples`` rows and then clamp the batch to what is left
        (`resolve_grpo_prompts_per_step`, and opd's `min(knobs.prompts_per_step, len(prompts))`), so
        an authored 128 against `max_examples = 2` trains on 2. Ranking the raw 128 would size
        hardware for a step that cannot happen. `flash.cost.spec._on_policy_prompts_per_step` already
        takes this same minimum, so skipping it here left ranking and the persisted quote disagreeing.

        The recipe default is resolved BEFORE the minimum, because the workers clamp it too: they
        cap whatever the batch resolved to, authored or defaulted. Returning None for an unauthored
        batch would let `normalized()` fill 64 (grpo) or 8 (opd) uncapped, so a run whose whole pool
        is 2 rows would still rank for 64.
        """
        from flash.engine.plan.recipe import RECIPE

        authored = knob("prompts_per_step")
        retained = knob("max_examples")
        if retained is None:
            return authored
        recipe = RECIPE.opd if algorithm == "opd" else RECIPE.rl
        return min(authored if authored is not None else int(recipe.prompts_per_step), retained)

    fields = {
        "model_id": model_id,
        "method": algorithm,
        "steps": 1,
        "seq_len": knob("max_context_tokens"),
        "completion_len": knob("max_completion_tokens"),
        # RunConfig.batch_size is "examples per optimizer update", which each algorithm authors
        # under a different key: sft as `batch_size`, grpo/opd as `prompts_per_step` (the schema
        # rejects `batch_size` outright for them). Reading only the sft name left every authored
        # rollout batch as None here, so ranking priced the recipe default -- 64 against an
        # authored 32 -- and could select a costlier shape than the run actually needs.
        # `flash.cost.spec.estimate_for_spec` splits the same way, with a third branch this has no
        # input for: it prefers a measured sft workload profile, which ranking runs before --
        # `overrides` below is how that measurement reaches this function.
        "batch_size": (rollout_batch() if samples_on_policy(algorithm) else knob("batch_size")),
        "group_size": knob("group_size"),
        "lora_rank": knob("lora_rank"),
        "thinking": thinking,
        "model_revision": model_revision,
    }
    fields.update(overrides or {})
    return RunConfig(**fields)


def _run_cost_key(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
    overrides: dict | None = None,
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
                overrides=overrides,
            )
        )
    except Exception:  # unpriceable run -- rank on $/hr, never fail selection
        return None


def authored_gpu_ceiling(gpu_type: str, gpu_count: int | None) -> int | None:
    """Return the author's card ceiling, or ``None`` when the run may be auto-sized.

    THE single definition of "did the author choose a shape?". Flash decides GPU shape at four
    boundaries that never call each other -- the parse gate (``flash/schema``), the offline
    ``--cost`` quote (``flash/cost/analytical.py``) and ``allocate()`` (which rents the hardware) --
    and each one previously re-derived this rule in its own shape. That drifted three times in
    review: a pinned 24 GB RTX 4090 with no authored count quoted EIGHT cards for an 80 GB run, and
    ``allocate()`` resolved the same pin to a ceiling of 8. A test at one boundary cannot fail for a
    bug at another, so the suite stayed green while half the boundaries were wrong.

    A pinned class with no count is a ONE-CARD pin, not an invitation to widen: escalating it would
    bill hardware the author never asked for. Auto-sizing applies only when NEITHER is authored.
    ``gpu.count`` is parsed with ``minimum=1``, so a falsy authored count is unreachable rather than
    silently meaning "one".
    """
    if gpu_count is not None:
        return gpu_count
    return 1 if gpu_type else None


def _eligible_gpu_infos(gpu_names: tuple[str, ...] | None = None) -> tuple[GpuClass, ...]:
    """Return the validated structural pool used by offline sizing."""
    if gpu_names is None:
        return tuple(g for g in GPU_CLASSES if g.enum_member and g.validated)
    names = set(gpu_names)
    return tuple(g for g in GPU_CLASSES if g.name in names and g.validated)


def gpu_capacity_shape(
    gpu_count: int,
    *,
    min_vram_gb: float | None = None,
    gpu_names: tuple[str, ...] | None = None,
    executed_width=None,
) -> tuple[GpuClass, int, float] | None:
    """Return the largest shape, or the smallest shape clearing ``min_vram_gb``.

    The count in the returned shape is the RENTED count -- it labels what the user buys and what
    `--gpus N` asks for. The capacity is valued at ``executed_width(count)``, the ranks that join.
    For sft those differ, and a message that mixes them contradicts itself: a 2-card ceiling that
    launches one rank was reported as providing 234.1 GB against a 230 GB need (so the rejection
    read as a fit), then pointed at a 4-card shape credited 252.8 GB that really delivers 191.6.
    """
    count = largest_rentable_count(gpu_count)
    launched = executed_width(count) if executed_width else count
    shapes = [
        (gpu, count, combined_vram_gb(gpu.vram_gb, launched))
        for gpu in _eligible_gpu_infos(gpu_names)
    ]
    if min_vram_gb is not None:
        shapes = [shape for shape in shapes if shape[2] >= min_vram_gb]
        return (
            min(shapes, key=lambda shape: (shape[2], shape[0].vram_gb, shape[0].name))
            if shapes
            else None
        )
    return (
        max(shapes, key=lambda shape: (shape[2], shape[0].vram_gb, shape[0].name))
        if shapes
        else None
    )


def smallest_fitting_gpu_count(
    min_vram_gb: float,
    *,
    max_gpu_count: int,
    gpu_names: tuple[str, ...] | None = None,
    executed_width=None,
) -> int | None:
    """Return the smallest rentable count whose structural pool can hold the run.

    ``executed_width`` maps a rented count to the ranks that actually JOIN the run, defaulting to
    "all of them"; only sft narrows it (``allocator._executed_width``). Two consequences shape this
    search. The rank count is valued directly rather than through ``gpu_capacity_shape``, which
    snaps its argument to a RENTABLE count -- right for cards you buy, wrong for ranks that launch,
    since sft widths are not powers of two (batch 3 over 3 rows launches 3, and snapping to 2
    under-credits a shape that fits). And every rentable count is visited rather than stopping at
    the first miss, because the executed width is not monotonic in the rented count: that same run
    launches 1 rank on 2 cards but 3 on 4. Returning the SMALLEST fitting count keeps the cheapest
    shape winning.
    """
    launched = executed_width or (lambda count: count)
    fitting = [
        count
        for count in rentable_gpu_counts(max_gpu_count)
        if any(
            combined_vram_gb(gpu.vram_gb, launched(count)) >= min_vram_gb
            for gpu in _eligible_gpu_infos(gpu_names)
        )
    ]
    return min(fitting) if fitting else None


def cheapest_gpu(
    min_vram_gb: int, *, gpu_count: int = 1, cost_key=None, executed_width=None
) -> str:
    """Return the cheapest fitting validated GPU class for the card ceiling.

    Size against the rentable multi-card shape so ``--gpus`` is not rejected as single-card, but
    credit only ``executed_width`` when the run launches fewer ranks than it rents. A ``cost_key``
    ranks job cost; without one, rank hourly rate.
    """
    # the allocator never proposes a wider combination, so sizing against one would admit a spec
    # here only for submit to reject it -- the same defect this parameter exists to fix, inverted.
    # size against the largest count providers actually RENT (powers of two): a ceiling of 3 buys
    # 2 cards at submit, so sizing on 3 would admit a shape that never gets provisioned.
    cards = largest_rentable_count(gpu_count)
    launched = executed_width(cards) if executed_width else cards
    pool = [
        g for g in _eligible_gpu_infos() if combined_vram_gb(g.vram_gb, launched) >= min_vram_gb
    ]
    if not pool:
        shape = f" even as a {cards}-card combination" if cards > 1 else ""
        raise UnsupportedGpuError(f"no validated GPU class has >= {min_vram_gb} GB VRAM{shape}")
    from flash.providers.runpod.client.pricing import hourly_rate

    if cost_key is not None:
        return min(
            pool,
            key=lambda g: (cost_key(g.name, hourly_rate(g.name)), hourly_rate(g.name), g.vram_gb),
        ).name
    return min(pool, key=lambda g: (hourly_rate(g.name), g.vram_gb)).name


def provisional_gpu_count(
    model_id: str,
    algorithm: str = "sft",
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
    geometry_model_revision: str | None = None,
    gpu_count: int | None = None,
) -> int:
    """Resolve an authored ceiling or the smallest geometry-safe auto-sized count."""
    from flash.providers.core.allocator import (
        _executed_width,
        geometry_safe_gpu_cap,
        required_vram_gb,
    )

    ceiling = MAX_COMBINATION_CARDS if gpu_count is None else gpu_count
    geometry_revision = (
        model_revision if geometry_model_revision is None else geometry_model_revision
    )
    safe_ceiling = geometry_safe_gpu_cap(model_id, ceiling, model_revision=geometry_revision)
    if gpu_count is not None:
        return safe_ceiling
    need = required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        model_revision=model_revision,
    )
    # parse has no workload profile yet, so retained rows are deliberately unmeasured. the shared
    # allocator rule treats that as no row narrowing while still bounding sft by its known batch.
    executed_width = _executed_width(algorithm, train, None)
    fitting_count = smallest_fitting_gpu_count(
        need,
        max_gpu_count=safe_ceiling,
        executed_width=executed_width,
    )
    if fitting_count is not None:
        return fitting_count
    # the caller still needs a shape to preview for its rejection. use the smallest rental that
    # reaches the widest executable rank count, rather than idle cards that add no capacity.
    return min(
        rentable_gpu_counts(safe_ceiling),
        key=lambda count: (-executed_width(count), count),
    )


def provisional_gpu(
    model_id: str,
    algorithm: str = "sft",
    *,
    train=None,
    thinking: bool = False,
    model_revision: str = "",
    geometry_model_revision: str | None = None,
    gpu_count: int | None = None,
    authored_gpu_ceiling: int | None = None,
) -> str:
    """Return the offline class preview for an authored ceiling or auto-sized run.

    Two DIFFERENT numbers, deliberately not folded together:

    ``gpu_count`` is the width to preview a class AT, already through the geometry cap. The class
    is chosen for the width, so previewing an authored eight on a model the cap holds to four names
    a class that run never gets.

    ``authored_gpu_ceiling`` is what the author wrote (``None`` when auto-sized) and only steers the
    rejection's remedy: a ceiling the author chose can be raised, whereas an auto-sized run has
    already tried every width and must be told to lower its knobs instead. Deriving it from the
    capped width would report the cap as the author's choice. Callers that pass the author's value
    straight through as ``gpu_count`` -- for whom the two ARE the same number -- may leave it unset.
    """
    if authored_gpu_ceiling is None and gpu_count is not None:
        authored_gpu_ceiling = gpu_count
    from flash.engine.plan.vram import model_required_vram_gb
    from flash.providers.core.allocator import _executed_width, geometry_safe_gpu_cap, vram_headroom

    min_vram = model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
        model_revision=model_revision,
    )
    effective_count = provisional_gpu_count(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        model_revision=model_revision,
        geometry_model_revision=geometry_model_revision,
        gpu_count=gpu_count,
    )
    geometry_revision = (
        model_revision if geometry_model_revision is None else geometry_model_revision
    )
    auto_cap = geometry_safe_gpu_cap(
        model_id, MAX_COMBINATION_CARDS, model_revision=geometry_revision
    )
    # no profile exists at parse time, so rows stay unmeasured while the known sft batch still bounds
    # the ranks that can join. this is the same rule allocation applies once a profile is available.
    executed_width = _executed_width(algorithm, train, None)
    try:
        # same job-cost basis the submit-time allocator ranks on, so the class previewed here is
        # the class that actually gets provisioned; falls back to $/hr for an unpriceable run.
        return cheapest_gpu(
            min_vram,
            gpu_count=effective_count,
            cost_key=_run_cost_key(
                model_id, algorithm, train=train, thinking=thinking, model_revision=model_revision
            ),
            executed_width=executed_width,
        )
    except UnsupportedGpuError as exc:
        # deferred: fit_errors imports from here, so a module-level import would be circular.
        from flash.providers.core.fit_errors import vram_fit_error_message

        raise UnsupportedGpuError(
            vram_fit_error_message(
                algorithm,
                min_vram,
                requested_gpu_count=authored_gpu_ceiling,
                effective_gpu_count=effective_count,
                max_gpu_count=auto_cap,
                executed_width=executed_width,
            )
        ) from exc


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
    """Return rentable card counts, largest first, up to ``max_gpu_count``.

    Use powers of two: providers sell those shapes, and vLLM requires ``num_attention_heads %
    tp_size == 0`` wherever the rollout engine runs tensor-parallel (grpo and opd hand it the rented
    card count; sft has no rollout engine, and its width is bounded by the batch instead -- see
    ``sft_data_parallel_cards``). Every current catalog head count (8, 8, 16, 16, 24, 16) divides 1,
    2, 4, and 8, so today every rentable shape is legal for every row. That is a property of today's
    catalog, not an invariant: ``allocator.geometry_safe_gpu_cap`` checks each row's own recorded head
    count so a future row with, say, 20 heads is capped rather than rented and failed at rollout
    engine init. This function only enumerates the shapes providers rent.
    """
    cap = max(1, int(max_gpu_count))
    counts, count = [], 1
    while count <= cap:
        counts.append(count)
        count *= 2
    return tuple(reversed(counts))


def largest_rentable_count(max_gpu_count: int) -> int:
    """Return the widest shape the card ceiling can actually rent.

    Only powers of two up to ``MAX_COMBINATION_CARDS`` are offered, so a ceiling of 3 buys 2 cards.
    """
    return rentable_gpu_counts(min(max(1, int(max_gpu_count)), MAX_COMBINATION_CARDS))[0]


def wider_shape_remedy(
    vram_options: Iterable[int], need: float, *, ceiling: int, above: int = 0, executed_width=None
) -> str:
    """The ``--gpus N`` clause a fit failure carries, or ``""`` when no wider shape would fit.

    A fit failure is only worth reporting as unsatisfiable when NO shape the user can ask for
    would fix it. `[gpu] count` is a user-authored ceiling, so a run that fails at the authored
    count but fits at a wider one is a one-flag fix, not a dead end -- and the error is the only
    place the user learns which flag.

    The width is SEARCHED with ``combined_vram_gb``, the same fit model that rejected the run, so
    a suggested N is one this function proved rather than one inferred from total VRAM. ``ceiling``
    must be the caller's ``geometry_safe_gpu_cap`` so the suggestion is never a width the rollout
    engine rejects at init after the box is rented, and ``above`` excludes the counts already tried.
    The smallest fitting count wins: the cheapest shape that works, not the widest on offer.

    ``vram_options`` carries only classes a provider in play will actually rent at a wider count
    (see ``rents_arbitrary_card_counts``); passing none leaves the failure a bare dead end, which
    is the honest answer when no offline check can confirm the wider SKU exists.

    ``executed_width`` (``allocator._executed_width``) has to bound the suggestion too: `--gpus 2`
    on a run whose batch launches one rank buys a second card that contributes no memory, so the
    user pays twice to fail identically. Advice is searched with the rule that will judge the retry.

    Every fit-rejection message routes through here so the remedy cannot drift in wording or in
    the rule that produces it.
    """
    launched = executed_width or (lambda count: count)
    best = 0
    for vram_gb in vram_options:
        for count in sorted(rentable_gpu_counts(max(1, int(ceiling)))):
            if count > above and combined_vram_gb(vram_gb, launched(count)) >= need:
                best = count if best == 0 else min(best, count)
                break
    if best == 0:
        return ""
    return f"; it fits on {best} cards -- raise the card ceiling with `--gpus {best}`"


@dataclass(frozen=True)
class Candidate:
    provider: str
    gpu: str
    hourly_usd: float
    vram_gb: int
    # cards in this candidate combination; hourly_usd and vram_gb stay PER-CARD so existing
    # single-gpu consumers are unchanged. total cost = gpu_count * hourly_usd.
    gpu_count: int = 1
    # ranks that join the run after allocation; none preserves all-rented-card behavior for candidates
    # constructed by providers or tests before the allocator stamps the run-specific width.
    executed_gpu_count: int | None = None

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
    # Whole-run fit floor before the allocator reduces it to a per-card market query. Capacity-aware
    # providers use this to distinguish a missing rentable shape from a sold-out one.
    required_vram_gb: int = 0
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

    def submit_attempt(
        self,
        spec: JobSpec,
        *,
        log: Any = None,
        on_handle: Any = None,
        attempt: int = 0,
        runtime_secrets: dict[str, str] | None = None,
        on_last_gpu: bool = False,
        source_snapshot: dict | None = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        """Deploy/rent -> submit -> persist handle (via ``on_handle``) -> poll one attempt to terminal.

        One run may execute several attempts, each on its own host or gpu class; this runs exactly
        one of them. ``on_last_gpu``: no further gpu attempt follows, so capacity backstops may wait
        longer.
        """
        ...

    def poll_attempt(
        self,
        handle: JobHandle,
        spec: JobSpec,
        *,
        log: Any = None,
        _deadline_at: float | None = None,
    ) -> PollResult:
        """Reattach to one attempt's persisted handle and poll it to a terminal state."""
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
    # reason as below): True for providers that attach the managed shared weight cache. The runner
    # gates its one-shot cache-less retry fallback on it; every other provider defaults False.

    # ``run_instances_remaining(run_id)`` is optional and must stay off this runtime-checkable
    # protocol. callers use getattr in server/_runtime.py. ``[]`` confirms clear; non-empty means a
    # survivor. incomplete enumeration raises so recovery cannot mistake lookup failure for clear.

    def sweep_orphans(
        self,
        active_labels: set[str] | Callable[[], set[str]] | None = None,
        known_labels: set[str] | Callable[[], set[str]] | None = None,
    ) -> list[int | str]:
        """Destroy provider resources unclaimed by live runs.

        Resolve callable ``active_labels`` after listing to close the launch race. ``known_labels``
        limits cleanup to this plane in shared accounts; None keeps single-plane behavior.
        """
        ...
