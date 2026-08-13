"""Structured job specification shared by CLI/API/runner and GPU workers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal
from uuid import UUID

from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
from flash.core.catalog import DEFAULT_MODEL, normalize_algorithm, samples_on_policy
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV

_FALSE_STRINGS = {"", "0", "false", "no", "off", "none"}

# default for old payloads and callers that do not select a per-run seed.
FIXED_SEED = 42
_MAX_SEED = 2**63 - 1

CreditAssignment = Literal["per_episode", "per_turn"]
DEFAULT_CREDIT_ASSIGNMENT: CreditAssignment = "per_episode"
PER_TURN_CREDIT_ASSIGNMENT: CreditAssignment = "per_turn"
CREDIT_ASSIGNMENTS: tuple[CreditAssignment, ...] = (
    DEFAULT_CREDIT_ASSIGNMENT,
    PER_TURN_CREDIT_ASSIGNMENT,
)


def _coerce_credit_assignment(value: Any) -> CreditAssignment:
    """coerce credit assignment to a known mode and reject malformed payloads.

    missing or blank uses the default; unknown internal/persisted values must not silently downgrade.
    """
    if value is None:
        return DEFAULT_CREDIT_ASSIGNMENT
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return DEFAULT_CREDIT_ASSIGNMENT
        for mode in CREDIT_ASSIGNMENTS:
            if normalized == mode:
                return mode
    raise ValueError(f"credit_assignment must be one of {CREDIT_ASSIGNMENTS}; got {value!r}")


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob to a tuple of strings; a bare string is one element, not iterated."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def coerce_bool(value: Any) -> bool:
    """Parse a bool from loosely-typed sources; treats "false"/"0"/"no"/"off" as False."""
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


def require_project_id(value: Any) -> str:
    """Return one explicit canonical Freesolo project UUID or reject it."""
    if value is None:
        raise ValueError("project is required and must be nonblank")
    if not isinstance(value, str):
        raise TypeError("project must be a string")
    project_id = value.strip()
    if not project_id:
        raise ValueError("project is required and must be nonblank")
    try:
        return str(UUID(project_id))
    except ValueError as exc:
        raise ValueError("project must be a valid UUID") from exc


def _coerce_wandb(value: Any) -> WandbSpec:
    """Coerce to WandbSpec; non-dict input returns default."""
    if not isinstance(value, dict):
        return WandbSpec()

    def _label(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return WandbSpec(project=_label(value.get("project")), run_name=_label(value.get("run_name")))


def _volume_gb(value: Any, default: int = 100) -> int:
    """Parse volume size in GB; non-positive / non-numeric / missing values return default."""
    if isinstance(value, bool):
        # bool is an int subclass; reject to avoid int(True)==1 becoming a 1 GB volume.
        return default
    try:
        gb = int(value)
    except (TypeError, ValueError):
        return default
    return gb if gb > 0 else default


def _validated_gpu_type(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    from flash.providers.base import GPU_INFO, UnsupportedGpuError, canonical_gpu

    try:
        canonical = canonical_gpu(value)
    except UnsupportedGpuError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc
    info = GPU_INFO.get(canonical)
    if info is None or not info.validated:
        raise ValueError(f"{field_name} {canonical!r} must name an active validated GPU class")
    return canonical


def _validated_gpu_type_fallbacks(value: Any, *, head: str) -> tuple[str, ...]:
    """Canonicalize the alternatives after `gpu.type`, preserving authored order.

    Duplicates are dropped rather than rejected: the same class named twice asks for nothing the
    first entry did not already ask for, and allocation walks a set. Order is the author's stated
    preference, so the FIRST occurrence is the one kept.
    """
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("gpu.type_fallbacks must be a list of strings")
    seen = {head} if head else set()
    fallbacks: list[str] = []
    for entry in value:
        canonical = _validated_gpu_type(entry, field_name="gpu.type_fallbacks entry")
        if canonical in seen:
            continue
        seen.add(canonical)
        fallbacks.append(canonical)
    return tuple(fallbacks)


def _opt_int(value: Any) -> int | None:
    """Parse optional int; rejects bools (bool is int subclass — int(True)==1 is a footgun)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def _opt_float(value: Any) -> float | None:
    """Parse optional float; rejects bools (mirrors _opt_int)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)


def _migrated_optimizer_batch(train: dict, algorithm: str) -> tuple[int | None, int | None]:
    """``(batch_size, prompts_per_step)`` with the pre-1.1.43 rollout spelling migrated.

    grpo/opd authored the optimizer batch as ``batch_size`` until it was split into
    ``prompts_per_step``. The schema rejects the old name on SUBMISSION, but `from_dict` also
    reparses PERSISTED specs, and the deployed release (1.1.40) has no ``prompts_per_step`` at all
    -- so every rollout run in flight across this upgrade carries only the old key. The workers read
    ``prompts_per_step`` alone, so without this a recovered run silently resumes on the recipe
    default: 64 instead of an authored 32 on grpo, which OOMs a card rented for 32, and 8 on opd.

    The old key is MOVED, not copied, and is dropped whenever the new one is present. Leaving it
    populated would re-emit both names from ``to_dict()``, and a spec carrying both is rejected by
    the schema -- breaking the resubmit that recovery and ``flash runs get`` perform, and leaving
    `vram.py::_optimizer_batch_value` (which takes the larger of the two) free to size a card off
    the stale value that ranking ignores. That applies to a payload written mid-upgrade carrying
    BOTH spellings too: ``prompts_per_step`` wins, so the superseded key has to go with it.

    A non-positive legacy value is discarded rather than migrated: ``minimum=1`` would have
    rejected it at submission, and carrying it forward only fails later on a rented GPU. It is
    discarded from BOTH names for the same round-trip reason -- retaining it under the old one
    would re-emit a key the schema rejects for this algorithm.
    """
    batch_size = _opt_int(train.get("batch_size"))
    prompts_per_step = _opt_int(train.get("prompts_per_step"))
    if not samples_on_policy(algorithm):
        return batch_size, prompts_per_step
    if prompts_per_step is not None:
        return None, prompts_per_step
    return None, batch_size if batch_size is not None and batch_size >= 1 else None


_MAX_GPU_COUNT = 8


def _gpu_count(value: Any, *, field_name: str = "gpu.count") -> int:
    """Parse the per-job gpu count (1..8); rejects bools and out-of-range values.

    one job occupies ``count`` cards of the chosen class on a single worker; count == 1 is the
    historical single-gpu behavior. sharded-fit sizing lands with the multi-gpu training paths.
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got bool {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
    if count < 1 or count > _MAX_GPU_COUNT:
        raise ValueError(f"{field_name} must be between 1 and {_MAX_GPU_COUNT}, got {count}")
    return count


def gpu_count_of(spec: Any) -> int:
    """Best-effort per-job gpu count from a JobSpec-like value; defaults to 1 when absent/None."""
    count = getattr(getattr(spec, "gpu", None), "count", 1)
    return count if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else 1


def parse_seed(value: Any = FIXED_SEED) -> int:
    """Parse one bounded nonnegative per-run seed without coercing floats or bools."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an integer")
    if value < 0 or value > _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    return value


def parse_max_steps(value: Any) -> int | None:
    """parse the optional exact optimizer-update horizon without coercion.

    positive values are authoritative; absent or non-positive values canonicalize to none so every
    parser, serializer, resolver, and save-step path shares one sentinel.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("train.max_steps must be an integer or null")
    return value if value > 0 else None


# this plane's own public origin: the address a gpu worker dials back on to reach the teacher
# broker. distinct from FLASH_API_URL, which is where an operator's cli dials in from and may be a
# private address the worker cannot resolve. control-plane-owned, so a caller cannot declare it as
# a worker env override or an [environment] secret.
PUBLIC_URL_ENV = "FLASH_PUBLIC_URL"
TEACHER_CAPABILITY_ENV = "FLASH_TEACHER_CAPABILITY"
MANAGED_TEACHER_CREDENTIAL_ENV_KEYS = frozenset({"PARASAIL_API_KEY"})
CONTROL_PLANE_OWNED_ENV_KEYS = frozenset(
    {
        "RUN_ID",
        "HF_REPO",
        "FLASH_ARM",
        "SEED",
        OPD_RESUME_REVISION_ENV,
        PUBLIC_URL_ENV,
        TEACHER_CAPABILITY_ENV,
        # the redactors' own transport: build_worker_env overwrites this with the generated list of
        # declared secret names, so a job declaring it would have its credential silently replaced
        # by that list and fail at runtime. control-plane-owned, hence rejected at declaration.
        SECRET_ENV_KEYS_ENV,
        *MANAGED_TEACHER_CREDENTIAL_ENV_KEYS,
    }
)


# the trainer every run reports. run_sft, run_opd and run_rl all delegate straight to verl, so this
# is a constant rather than a resolution: no job-spec field can select anything else. recorded on
# effective_preparation so a stored run says which trainer produced it.
TRAINER_BACKEND = "verl"


def require_matching_seed(spec: JobSpec, seed: Any) -> int:
    """Require the retained provider seed argument to match the authoritative JobSpec seed."""
    provided = parse_seed(seed)
    canonical = parse_seed(spec.seed)
    if provided != canonical:
        raise ValueError(
            f"provider seed {provided} does not match JobSpec.seed {canonical}; "
            "use spec.seed as the provider seed"
        )
    return canonical


def _model_revision(value: Any) -> str:
    """parse an optional exact model-repository revision."""
    if not isinstance(value, str):
        raise TypeError("model_revision must be a string")
    return value.strip()


def parse_positive_int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    """Parse a strictly increasing list or tuple of positive integers."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of integers")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} entries must be integers")
        if item <= 0:
            raise ValueError(f"{name} entries must be positive")
        out.append(item)
    if out != sorted(set(out)):
        raise ValueError(f"{name} must be strictly increasing with no duplicates")
    return tuple(out)


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    # Third-party requirements this environment's scorer imports, appended to Flash's own worker
    # requirement at submit (worker_pip_with_extras). Empty means the scorer needs nothing beyond
    # the worker's baseline; entries here never displace it.
    pip: tuple[str, ...] = ()
    # Names only — values sent out-of-band via runtime_secrets, never stored in spec.
    secrets: tuple[str, ...] = ()
    # Resolved once in control plane to avoid GitHub rate-limits on cold spawn waves.
    resolved_sha: str = ""


# internal persisted-record compatibility only, never public config parsing. #968 removed
# advantage_clip, but already-provisioned strict records still carry the unused key and otherwise fail
# attach/recovery. do not generalize this allowlist: removed ``seeds`` must still raise (#536).
REMOVED_PERSISTED_TRAIN_KEYS = frozenset({"advantage_clip"})


@dataclass(frozen=True)
class TrainSpec:
    epochs: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    lora_rank: int = field(default=32, metadata={"introduced_in": "0.2.0"})
    # Optional user knob. 0 is the "unset" sentinel: __post_init__ derives 2 x lora_rank from it, so
    # a directly constructed TrainSpec(lora_rank=8) gets alpha 16 like the parsed path rather than a
    # stale scalar. Authoring it is rejected alongside init_from_adapter, where the source adapter's
    # alpha is authoritative (see spec_from_dict).
    lora_alpha: int = field(default=0, metadata={"introduced_in": "1.1.35"})
    # artifact-store adapter ref output by `flash runs status`:
    # ``<hf_repo>:<phase>/<run_id>``.
    init_from_adapter: str = field(default="", metadata={"introduced_in": "0.2.0"})
    # internal: immutable source dataset commit used for a prepared warm-start. parsed only from
    # control-plane jobspec payloads; the public config schema does not accept this key.
    init_from_adapter_revision: str = ""
    # PLATFORM-MANAGED: control-plane-assigned HF artifact repo. Not a user config key: it has no
    # ``introduced_in`` so it is absent from TRAIN_SCHEMA_KEYS and ``[train] hf_repo`` is rejected.
    # JobSpec.from_dict still round-trips the control-plane-assigned value; to_dict() omits it.
    hf_repo: str = ""
    # None -> worker's tuned recipe default.
    learning_rate: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    # sft only. the packaged-dataset estimate resolves this against the selected row count into
    # examples_per_update (packed) or pins the optimizer batch to 1 (unpacked). grpo/opd have no
    # profile, so they take prompts_per_step instead and reject this key: the two are not the same
    # quantity, and accepting one name for both let the standard sft memory workaround
    # (batch_size = 1) silently mean one prompt per optimizer update on rl.
    batch_size: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    # grpo/opd ONLY: the optimizer batch itself, straight through to verl's data.train_batch_size
    # and ppo_mini_batch_size. no workload profile sits in between.
    prompts_per_step: int | None = field(default=None, metadata={"introduced_in": "1.1.43"})
    max_context_tokens: int | None = field(default=None, metadata={"introduced_in": "0.2.49"})
    save_every: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_steps: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    save_at_steps: tuple[int, ...] = field(default=(), metadata={"introduced_in": "0.2.57"})
    max_examples: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    group_size: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    temperature: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_completion_tokens: int | None = field(default=None, metadata={"introduced_in": "0.2.49"})
    kl_penalty_coef: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    entropy_quantile: float | None = field(default=None, metadata={"introduced_in": "1.0.15"})
    thinking_length_penalty_coef: float | None = field(
        default=None, metadata={"introduced_in": "0.2.0"}
    )
    # opd only: the managed teacher's canonical friendly alias. provider and repository ids are not
    # accepted as user input. an empty value selects the recipe default, glm-5.2.
    teacher_model: str = field(default="", metadata={"introduced_in": "0.2.56"})
    stop_sequences: tuple[str, ...] = field(default=(), metadata={"introduced_in": "0.2.0"})
    # canonical json of vllm structured-output kwargs ("" = unconstrained). normalized once
    # at parse time (schema/fields.py) so worker/hub/api hops carry one stable string form.
    structured_outputs: str = field(default="", metadata={"introduced_in": "0.2.56"})
    credit_assignment: CreditAssignment = field(
        default=DEFAULT_CREDIT_ASSIGNMENT, metadata={"introduced_in": "1.0.2"}
    )

    def __post_init__(self) -> None:
        if not self.lora_alpha:
            object.__setattr__(self, "lora_alpha", 2 * self.lora_rank)
        max_steps = parse_max_steps(self.max_steps)
        save_at_steps = parse_positive_int_tuple(self.save_at_steps, name="train.save_at_steps")
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "save_at_steps", save_at_steps)
        effective_max_steps = max_steps or 0
        if save_at_steps and effective_max_steps <= 0:
            raise ValueError("train.save_at_steps requires positive train.max_steps")
        if save_at_steps and save_at_steps[-1] > effective_max_steps:
            raise ValueError("train.save_at_steps cannot contain a step beyond train.max_steps")


@dataclass(frozen=True)
class GpuSpec:
    # empty selects managed auto-allocation; a set value restricts allocation to `acceptable_types`
    # (this class plus `type_fallbacks`), preferred in authored order. a lone `type` is therefore a
    # hard pin to exactly that class, unchanged.
    type: str = ""
    disk_gb: int = 60
    max_wall_seconds: int = 24 * 3600
    max_retries: int = 5
    # PLATFORM-MANAGED: runner assigns weight-cache volume; None = cold download.
    network_volume: str | None = None
    network_volume_gb: int = 100
    provider: str = ""
    # number of cards of `type` a single training worker occupies (1..8). count > 1 provisions a
    # multi-gpu pod; the training loop shards across them in the sft/opd multi-gpu paths.
    count: int = 1
    # the authored alternatives AFTER `type`, when `[gpu] type` was written as an ordered list.
    # `type` stays a single concrete class because every worker, provider submit path, and endpoint
    # name reads it as one; only allocation looks at the wider set. empty for a scalar pin.
    type_fallbacks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # coerce/validate here so every path (from_dict and direct construction) is guarded.
        object.__setattr__(self, "count", _gpu_count(self.count))
        object.__setattr__(self, "type_fallbacks", tuple(self.type_fallbacks))
        if self.type_fallbacks and not self.type:
            raise ValueError("gpu.type_fallbacks requires gpu.type")

    @property
    def acceptable_types(self) -> tuple[str, ...]:
        """Classes allocation may rent, in authored preference order; empty means auto-allocate.

        The single source of "what did the author pin", so a caller never has to remember to union
        `type` with `type_fallbacks` and accidentally narrow an ordered list back to its head.
        """
        return (self.type, *self.type_fallbacks) if self.type else ()


# platform-managed [gpu] fields: the runner assigns disk sizing, the shared weight-cache volume, and
# retry/wall-clock lifecycle policy; the user never authors them. single-sourced here so the public
# serializer (JobSpec.to_dict), the user-facing parser (flash.schema), and the runner's effective-spec
# validator strip, reject, and exclude exactly the same set. divergence would leak a managed field
# into the public surface or fail the submit round trip.
MANAGED_GPU_KEYS = frozenset(
    {"disk_gb", "network_volume", "network_volume_gb", "max_retries", "max_wall_seconds"}
)

# Removed public top-level fields that a PERSISTED record can still carry. Stored run records are
# never rewritten, and effective_preparation.worker_spec is written with to_internal_dict() (asdict),
# which emitted every field including the defaulted ones -- so every record written before a field
# was dropped still names it. This registry also drives historical public-digest replay for
# model_revision, which remains an internal field. Without it a still-running job can lose its
# recovery, deploy, and serving paths after the upgrade.
#
# Ignored on READ only. Most keys here no longer exist as JobSpec fields; model_revision remains an
# internal runner field, but the public schema rejects it and to_dict omits it. Keeping it in this set
# lets digest recovery identify the historical public key without weakening JobSpec.from_dict.
_DROPPED_TOP_LEVEL_KEYS = frozenset(
    {"model_policy", "model_revision", "worker_env", "workload_profile_kind"}
)

# Tolerating a dropped key keeps a pre-upgrade run's recovery path, but the values behind it stop
# being applied -- so a run that authored them now trains on managed defaults instead of what was
# submitted. That is announced rather than silent. Only user-authorable keys qualify: model_policy
# was platform-managed (to_dict popped it), so its loss cannot change what a submitted config asked
# for. The values are deliberately NOT forwarded -- doing so would reinstate the override table this
# removal exists to close, and would deliver it without the validation the deleted parser performed.
_ANNOUNCED_DROPPED_KEYS = frozenset({"worker_env"})


def _announce_dropped_keys(data: dict[str, Any]) -> None:
    """Log the removed user-authored keys a persisted record still carries and no longer applies.

    Only announced when the payload names the run. A public spec is ``to_dict()`` output, which pops
    the server-assigned run_id, and the same stored run is read through both shapes -- so warning
    without an id would emit an unactionable "run <unknown>" line and duplicate the identified one
    the worker-spec read already produced. Every provisioned run records an internal worker spec
    (asdict, run_id included), and that is the read this fires on.
    """
    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        return
    for key in sorted(_ANNOUNCED_DROPPED_KEYS):
        value = data.get(key)
        if not value:
            continue
        names = ", ".join(sorted(str(k) for k in value)) if isinstance(value, dict) else str(value)
        logging.getLogger("flash.spec").warning(
            "run %s was submitted with [%s] (%s); that table was removed and its values are NOT "
            "applied -- this run uses managed defaults instead. resubmit without it if the run "
            "depends on those values.",
            run_id,
            key,
            names,
        )


@dataclass(frozen=True)
class WandbSpec:
    project: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class JobSpec:
    model: str = DEFAULT_MODEL
    algorithm: str = "sft"
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    gpu: GpuSpec = field(default_factory=GpuSpec)
    run_id: str = "local"
    seed: int = FIXED_SEED
    thinking: bool = False
    wandb: WandbSpec = field(default_factory=WandbSpec)
    # internal base-model revision resolved by the runner. the public schema rejects this key, while
    # persisted and worker specs keep it for exact model loading, profiling, geometry validation, and
    # warm-start equality checks.
    model_revision: str = ""
    # platform-managed marker: True when the runner resolved model_revision for a spec whose public
    # input carried no pin (SFT, where `_resolve_model_revision(required=True)` pins the base so
    # workload profiling keys on an immutable commit). An AUTHORED pin can still exist on a persisted
    # pre-removal run and stays rejected at deploy; rejecting the auto-assigned one made every SFT run
    # and every adapter warm-started from one permanently undeployable, unservable, and unscoreable by
    # `flash env eval`.
    #
    # stripped by to_dict() like the other platform-managed carriers. Deploy reads the provenance
    # from the internal worker spec under `effective_preparation` instead (see
    # `_internal_spec_from_status`), which carries it verbatim.
    model_revision_auto: bool = False
    # platform-managed marker: true when the author omitted both gpu.type and gpu.count and the stored
    # integer 1 is only the digest-stable public placeholder. allocation reads this marker as
    # "auto-size"; a type pin or authored count=1 leaves it false and remains a hard ceiling.
    gpu_count_auto: bool = False
    # platform-managed workload-profile carrier. public configs never author these fields.
    workload_profile_input_digest: str = ""
    # the flash version that keyed the digest above. it has to travel with the spec because the
    # worker cannot re-derive it: `flash.__version__` reads distribution metadata, and the worker
    # instance never installs a flash distribution -- it runs the plane's own source snapshot off
    # PYTHONPATH (Dockerfile.worker ships deps only). there `__version__` is the "0+unknown"
    # fallback, so a worker deriving this locally keys a digest the plane can never produce.
    workload_profile_producer_version: str = ""
    workload_profile: dict[str, Any] = field(default_factory=dict)
    # canonical freesolo project uuid. every config, control-plane record, and worker round trip
    # carries the same explicit identity; there is no name/default/sole-project resolution.
    project: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", parse_seed(self.seed))
        object.__setattr__(self, "model_revision", _model_revision(self.model_revision))
        # the marker qualifies a pin; it cannot outlive one. a spec carrying it with no revision
        # would let a later edit that clears model_revision leave a True marker behind, and the
        # deploy guard reads the pair.
        if self.model_revision_auto and not self.model_revision:
            object.__setattr__(self, "model_revision_auto", False)
        # NOT cleared when gpu.count != 1. the marker records PROVENANCE -- that the author omitted
        # gpu.count -- which stays true after allocation resolves a shape onto the spec. clearing it
        # there destroyed the only record distinguishing an auto-sized run from an authored
        # single-card pin: the public halves are byte-identical (to_dict strips the marker and keeps
        # the placeholder count=1 for digest stability), so a recovered auto-sized run came back
        # hard-pinned to one card and could never be re-offered its multi-card shape. consumers that
        # mean "auto-size NOW" must additionally check that no shape has been resolved yet.
        profile_digest = str(self.workload_profile_input_digest or "")
        if profile_digest and (
            len(profile_digest) != 64 or any(c not in "0123456789abcdef" for c in profile_digest)
        ):
            raise ValueError("workload profile input digest must be lowercase sha256 hex")
        if not isinstance(self.workload_profile, dict):
            raise TypeError("workload_profile must be an object")
        if profile_digest and not str(self.workload_profile_producer_version or ""):
            raise ValueError("workload profile digest requires its producer version")

    @property
    def phase(self) -> str:
        return "rl" if self.algorithm == "grpo" else self.algorithm

    @property
    def authored_gpu_count(self) -> int | None:
        """The author's card ceiling: ``None`` when they omitted ``gpu.count``.

        THE single reading of "is the stored count real, or the placeholder?". `gpu_count_auto` and
        `gpu.count` are one value -- an optional ceiling -- split across two fields only because
        `to_dict` must keep a digest-stable integer in the public spec. Every consumer that wants
        the ceiling had to rejoin them by hand, and the hand-written guards disagreed: one checked
        `count == 1`, another `count == 1 and not type`, two checked neither. Rejoin them here so a
        consumer cannot invent a fifth spelling.
        """
        return None if self.gpu_count_auto else gpu_count_of(self)

    def to_dict(self) -> dict[str, Any]:
        """return the public user-authorable job specification.

        omit platform-managed fields because the control plane/runner assigns them and the public
        parser rejects them. internal callers use ``to_internal_dict()``.
        """
        data = asdict(self)
        # server-assigned identity — never authored in a config.
        data.pop("run_id", None)
        # model_revision is runner-managed and no longer part of the public config or status spec.
        # Internal round trips keep the value and marker through to_internal_dict(). Historical public
        # specs that emitted this key are replayed only while verifying their stored preparation
        # digest, using the exact value from the persisted bytes.
        data.pop("model_revision", None)
        data.pop("model_revision_auto", None)
        # keep gpu.count=1 in the public gpu object for preparation-digest stability. only the
        # platform-managed provenance marker is stripped; internal round trips carry it verbatim.
        data.pop("gpu_count_auto", None)
        data.pop("workload_profile_input_digest", None)
        data.pop("workload_profile_producer_version", None)
        data.pop("workload_profile", None)
        train = data["train"]
        train.pop("init_from_adapter_revision", None)
        train.pop("hf_repo", None)  # control-plane-assigned artifact repo
        if train.get("init_from_adapter"):
            # the source adapter's topology is authoritative for a warm start, so the parser rejects
            # both keys alongside init_from_adapter. the runner writes the inherited rank/alpha onto
            # the public spec, so they must be stripped here or the submit round trip re-validates
            # a combination the parser refuses.
            train.pop("lora_rank", None)
            train.pop("lora_alpha", None)
        # runner-assigned disk sizing, shared weight-cache volume, and retry/wall-clock lifecycle.
        gpu = data["gpu"]
        for managed in MANAGED_GPU_KEYS:
            gpu.pop(managed, None)
        # an ordered pin round-trips in the AUTHORED spelling -- `type` is the list, and the derived
        # head/fallbacks split does not appear. the public parser owns that split and rejects
        # `type_fallbacks` as unauthorable, so emitting it would fail the resubmit round trip.
        fallbacks = gpu.pop("type_fallbacks", ())
        if fallbacks:
            gpu["type"] = [gpu["type"], *fallbacks]
        data["environment"].pop("resolved_sha", None)  # resolve-once env ref pin
        # [environment] pip stays in the payload: it is the author's own scorer dependencies, which
        # only they can declare. The submit paths append it to worker_pip_for_env, so what travels
        # here is the extra requirements, not the worker's own.
        return data

    def to_internal_dict(self) -> dict[str, Any]:
        """Return the complete control-plane and worker representation."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_internal_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        if not isinstance(data, dict):
            raise TypeError("job spec must be an object")
        allowed_top_level = {item.name for item in fields(cls)}
        unknown_top_level = sorted(set(data) - allowed_top_level - _DROPPED_TOP_LEVEL_KEYS)
        if unknown_top_level:
            raise ValueError(f"job spec has unknown key(s): {', '.join(unknown_top_level)}")
        _announce_dropped_keys(data)
        env = data.get("environment") or {}
        # Reject stale payloads carrying a local `path`; worker only runs published env ids.
        if isinstance(env, dict) and env.get("path"):
            raise ValueError(
                "local environment paths are no longer supported; the worker only runs "
                "published Freesolo environment ids"
            )
        train = data.get("train", {})
        if train is None:
            train = {}
        if not isinstance(train, dict):
            raise TypeError("train must be an object")
        train = {
            key: value for key, value in train.items() if key not in REMOVED_PERSISTED_TRAIN_KEYS
        }
        unknown_train = sorted(set(train) - {item.name for item in fields(TrainSpec)})
        if unknown_train:
            raise ValueError(f"train has unknown key(s): {', '.join(unknown_train)}")
        gpu = data.get("gpu")
        if gpu is None:
            gpu = {}
        if not isinstance(gpu, dict):
            raise TypeError("gpu must be an object")
        unknown_gpu = sorted(set(gpu) - {item.name for item in fields(GpuSpec)})
        if unknown_gpu:
            raise ValueError(f"gpu has unknown key(s): {', '.join(unknown_gpu)}")
        gpu_type_raw = gpu.get("type", "")
        if not isinstance(gpu_type_raw, str):
            raise TypeError("gpu.type must be a string")
        gpu_type = (
            _validated_gpu_type(gpu_type_raw, field_name="gpu.type") if gpu_type_raw.strip() else ""
        )
        # an ordered `[gpu] type` list is split by the public parser into a concrete head plus these
        # fallbacks, so this internal boundary still sees one class per field. every entry is
        # canonicalized and validated exactly like the head: a persisted record is re-read here on
        # every recovery hop, and an unvalidated fallback would only fail once allocation reached it.
        gpu_type_fallbacks = _validated_gpu_type_fallbacks(
            gpu.get("type_fallbacks", ()), head=gpu_type
        )
        provider = gpu.get("provider", "")
        if not isinstance(provider, str):
            raise TypeError("gpu.provider must be a string")
        provider = provider.strip().lower()
        if provider or gpu_type:
            from flash.providers import PROVIDER_NAMES
            from flash.providers.base import providers_for

            if provider and provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {provider!r}")
            # every acceptable class has to be provisionable by the pinned provider, not just the
            # head: a fallback the provider cannot carry is dead weight that would silently narrow
            # the list back to one class at allocation time.
            for candidate in (gpu_type, *gpu_type_fallbacks):
                if candidate and provider and provider not in providers_for(candidate):
                    raise ValueError(
                        f"gpu.provider {provider!r} cannot provision gpu.type {candidate!r}"
                    )
        project_raw = data.get("project", "")
        if not isinstance(project_raw, str):
            raise TypeError("project must be a string")
        project = require_project_id(project_raw) if project_raw.strip() else ""
        algorithm = normalize_algorithm(data.get("algorithm", cls.algorithm))
        # one reading of the optimizer batch for both keys: the rollout spelling changed in 1.1.43
        # and a persisted spec can still carry the old one.
        batch_size, prompts_per_step = _migrated_optimizer_batch(train, algorithm)
        return cls(
            model=data.get("model", cls.model),
            model_revision=_model_revision(data.get("model_revision", cls.model_revision)),
            algorithm=algorithm,
            environment=EnvironmentSpec(
                id=env.get("id", ""),
                params=dict(env.get("params") or {}),
                pip=tuple(str(p) for p in env.get("pip") or ()),
                secrets=_str_tuple(env.get("secrets")),
                resolved_sha=str(env.get("resolved_sha") or ""),
            ),
            train=TrainSpec(
                epochs=_opt_int(train.get("epochs")),
                lora_rank=int(train.get("lora_rank", 32)),
                # round-trip a stored alpha (authored value, internal carrier, or a warm-start's
                # inherited parent alpha); fall back to 2 x rank when absent.
                lora_alpha=(
                    int(train["lora_alpha"])
                    if "lora_alpha" in train
                    else 2 * int(train.get("lora_rank", 32))
                ),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
                init_from_adapter_revision=str(train.get("init_from_adapter_revision") or ""),
                hf_repo=str(train.get("hf_repo") or ""),
                learning_rate=_opt_float(train.get("learning_rate")),
                batch_size=batch_size,
                prompts_per_step=prompts_per_step,
                max_context_tokens=_opt_int(train.get("max_context_tokens")),
                save_every=_opt_int(train.get("save_every")),
                max_steps=parse_max_steps(train.get("max_steps")),
                save_at_steps=train.get("save_at_steps"),
                max_examples=_opt_int(train.get("max_examples")),
                group_size=_opt_int(train.get("group_size")),
                temperature=_opt_float(train.get("temperature")),
                max_completion_tokens=_opt_int(train.get("max_completion_tokens")),
                kl_penalty_coef=_opt_float(train.get("kl_penalty_coef")),
                entropy_quantile=_opt_float(train.get("entropy_quantile")),
                thinking_length_penalty_coef=_opt_float(train.get("thinking_length_penalty_coef")),
                teacher_model=str(train.get("teacher_model") or ""),
                stop_sequences=_str_tuple(train.get("stop_sequences")),
                structured_outputs=str(train.get("structured_outputs") or ""),
                credit_assignment=_coerce_credit_assignment(train.get("credit_assignment")),
            ),
            gpu=GpuSpec(
                type=gpu_type,
                provider=provider,
                disk_gb=int(gpu.get("disk_gb", 60)),
                max_wall_seconds=int(gpu.get("max_wall_seconds", 24 * 3600)),
                max_retries=int(gpu.get("max_retries", 5)),
                # network_volume/network_volume_gb round-trip so the runner-assigned weight cache
                # survives the to_dict()->from_dict() hops in _with_model_disk / _spec_with_gpu /
                # _assign_managed_hf_repo before deploy.
                network_volume=gpu.get("network_volume"),
                network_volume_gb=_volume_gb(gpu.get("network_volume_gb")),
                count=gpu.get("count", 1),
                type_fallbacks=gpu_type_fallbacks,
            ),
            run_id=data.get("run_id", "local"),
            thinking=coerce_bool(data.get("thinking", False)),
            wandb=_coerce_wandb(data.get("wandb")),
            seed=parse_seed(data.get("seed", FIXED_SEED)),
            model_revision_auto=coerce_bool(data.get("model_revision_auto", False)),
            gpu_count_auto=coerce_bool(data.get("gpu_count_auto", False)),
            workload_profile_input_digest=str(data.get("workload_profile_input_digest") or ""),
            workload_profile_producer_version=str(
                data.get("workload_profile_producer_version") or ""
            ),
            workload_profile=dict(data.get("workload_profile") or {}),
            project=project,
        )

    @classmethod
    def from_json(cls, raw: str) -> JobSpec:
        return cls.from_dict(json.loads(raw))


def load_job_spec_from_env() -> JobSpec | None:
    """Load FLASH_JOB_SPEC_JSON or FLASH_JOB_SPEC_PATH if present on a worker node."""
    raw = os.environ.get("FLASH_JOB_SPEC_JSON")
    if raw:
        return JobSpec.from_json(raw)
    path = os.environ.get("FLASH_JOB_SPEC_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return JobSpec.from_json(f.read())
    return None
