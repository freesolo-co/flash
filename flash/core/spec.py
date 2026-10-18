"""Structured job specification shared by CLI/API/runner and GPU workers.

One `JobSpec` value is read through four different contracts, and the serializers below are how
they are told apart:

- authored configuration -- what a user writes. validated by `flash.schema.spec_from_dict`, NOT by
  `JobSpec.from_dict`; the two are not interchangeable.
- public representation -- `to_dict()`, deliberately lossy: it strips every platform-managed field
  so the result re-validates through the authored parser.
- persisted recovery record -- `from_dict()`, strict about the current internal shape. its decoding
  helpers live in `flash.core.spec_persistence`.
- resolved worker payload -- `to_internal_dict()` / `to_json()`, complete, including every managed
  and resolved field the GPU worker needs.

`flash/runner/lifecycle/preparation.py::_preparation_digest` hashes the canonical JSON of `to_dict()` and
`to_internal_dict()`, so the BYTES those two emit are a recovery contract, not an implementation
detail. See their docstrings before changing either.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal
from uuid import UUID

from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
from flash.core.catalog import DEFAULT_MODEL, normalize_algorithm
from flash.core.spec_persistence import (
    opt_float,
    opt_int,
    str_tuple,
    validated_persisted_providers,
    validated_section,
    volume_gb,
)
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
    """coerce persisted credit assignment to a known explicit mode."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        for mode in CREDIT_ASSIGNMENTS:
            if normalized == mode:
                return mode
    raise ValueError(f"credit_assignment must be one of {CREDIT_ASSIGNMENTS}; got {value!r}")


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


def _validated_gpu_type(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    from flash.providers.core.base import GPU_INFO, UnsupportedGpuError, canonical_gpu

    try:
        canonical = canonical_gpu(value)
    except UnsupportedGpuError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc
    info = GPU_INFO.get(canonical)
    if info is None or not info.validated:
        raise ValueError(f"{field_name} {canonical!r} must name an active validated GPU class")
    return canonical


def _validated_gpu_type_fallbacks(value: Any, *, head: str) -> tuple[str, ...]:
    """Canonicalize and deduplicate fallback classes in authored order."""
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


def _parse_persisted_gpu_types(gpu: dict) -> tuple[str, tuple[str, ...]]:
    """Read either public list form or internal head-plus-fallback form."""
    gpu_type_raw = gpu.get("type", "")
    extra_types: tuple = ()
    if isinstance(gpu_type_raw, (list, tuple)):
        authored = list(gpu_type_raw)
        if not authored:
            raise ValueError("gpu.type list must name at least one gpu")
        gpu_type_raw, extra_types = authored[0], tuple(authored[1:])
        if "type_fallbacks" in gpu:
            raise ValueError("gpu.type list and gpu.type_fallbacks cannot both be set")
    if not isinstance(gpu_type_raw, str):
        raise TypeError("gpu.type must be a string")
    gpu_type = (
        _validated_gpu_type(gpu_type_raw, field_name="gpu.type") if gpu_type_raw.strip() else ""
    )
    return gpu_type, _validated_gpu_type_fallbacks(
        extra_types or gpu.get("type_fallbacks", ()), head=gpu_type
    )


def persisted_gpu_types(spec: Any) -> tuple[str, ...]:
    """Read acceptable GPU classes from a raw persisted spec without raising."""
    if not isinstance(spec, dict):
        return ()
    gpu = spec.get("gpu")
    raw = gpu.get("type") if isinstance(gpu, dict) else None
    if isinstance(raw, str):
        raw = (raw,)
    elif not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(entry for entry in raw if isinstance(entry, str) and entry))


def persisted_gpu_head(spec: Any) -> str:
    """First acceptable class from a raw persisted spec, or an empty string."""
    types = persisted_gpu_types(spec)
    return types[0] if types else ""


def attributed_gpu_type(status: Any) -> str:
    """GPU class actually selected for a raw run status, or its authored head."""

    def field(name: str) -> Any:
        return status.get(name) if isinstance(status, dict) else getattr(status, name, None)

    remote = field("remote") or field("realized_cost_remote")
    allocated = remote.get("allocated_gpu") if isinstance(remote, dict) else None
    if isinstance(allocated, str) and allocated:
        return allocated
    preparation = field("effective_preparation")
    worker_spec = preparation.get("worker_spec") if isinstance(preparation, dict) else None
    selected = persisted_gpu_head(worker_spec)
    return selected or persisted_gpu_head(field("spec"))


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
        "GITHUB_TOKEN",
        "GIT_ASKPASS",
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
class EnvironmentPackageSpec:
    artifact_revision: str
    archive_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        from flash.envs.meta.identity import is_commit_sha

        if (
            not is_commit_sha(self.artifact_revision)
            or self.artifact_revision.lower() != self.artifact_revision
        ):
            raise ValueError(
                "staged environment artifact revision must be lowercase immutable commit hex"
            )
        for name, digest in (
            ("archive", self.archive_sha256),
            ("manifest", self.manifest_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"staged environment {name} digest must be lowercase sha256 hex")


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    # third-party requirements this environment's scorer imports, appended to flash's own worker
    # requirement at submit (worker_pip_with_extras). empty means the scorer needs nothing beyond
    # the worker's baseline; entries here never displace it.
    pip: tuple[str, ...] = ()
    # names only, with values sent out-of-band through runtime_secrets and never stored in spec.
    secrets: tuple[str, ...] = ()
    # resolved once in the control plane to avoid github rate limits on cold spawn waves.
    resolved_sha: str = ""
    # controller-staged immutable package transport. absent only before run preparation completes.
    package: EnvironmentPackageSpec | None = None

    def __post_init__(self) -> None:
        if self.package is None:
            return
        from flash.envs.meta.identity import canonical_environment_id, is_commit_sha

        canonical_environment_id(self.id)
        if not is_commit_sha(self.resolved_sha) or self.resolved_sha.lower() != self.resolved_sha:
            raise ValueError(
                "staged environment resolved sha must be lowercase immutable commit hex"
            )


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
        if self.save_every is not None:
            if isinstance(self.save_every, bool) or not isinstance(self.save_every, int):
                raise TypeError("train.save_every must be an integer or null")
            if self.save_every <= 0:
                raise ValueError("train.save_every must be positive")
        effective_max_steps = max_steps or 0
        if save_at_steps and effective_max_steps <= 0:
            raise ValueError("train.save_at_steps requires positive train.max_steps")
        if save_at_steps and save_at_steps[-1] > effective_max_steps:
            raise ValueError("train.save_at_steps cannot contain a step beyond train.max_steps")


@dataclass(frozen=True)
class GpuSpec:
    # empty selects managed allocation; acceptable types compete on cost.
    type: str = ""
    disk_gb: int = 60
    max_wall_seconds: int = 24 * 3600
    max_retries: int = 5
    # PLATFORM-MANAGED: runner assigns weight-cache volume; None = cold download.
    network_volume: str | None = None
    network_volume_gb: int = 100
    provider: str = ""
    providers: tuple[str, ...] = ()
    # number of cards of `type` a single training worker occupies (1..8). count > 1 provisions a
    # multi-gpu pod; the training loop shards across them in the sft/opd multi-gpu paths.
    count: int = 1
    # authored alternatives after the concrete head class.
    type_fallbacks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # coerce/validate here so every path (from_dict and direct construction) is guarded.
        from flash.providers.core.registry import validated_provider_preferences

        providers = validated_provider_preferences(
            self.providers, allow_empty=isinstance(self.providers, tuple)
        )
        if self.provider and providers:
            raise ValueError("gpu.provider and gpu.providers cannot both be set")
        if not isinstance(self.type, str):
            raise TypeError("gpu.type must be a string")
        gpu_type = (
            _validated_gpu_type(self.type, field_name="gpu.type") if self.type.strip() else ""
        )
        fallbacks = _validated_gpu_type_fallbacks(self.type_fallbacks, head=gpu_type)
        if fallbacks and not gpu_type:
            raise ValueError("gpu.type_fallbacks requires gpu.type")
        object.__setattr__(self, "type", gpu_type)
        object.__setattr__(self, "type_fallbacks", fallbacks)
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "count", _gpu_count(self.count))

    @property
    def acceptable_types(self) -> tuple[str, ...]:
        """Classes allocation may rent, in authored order."""
        return (self.type, *self.type_fallbacks) if self.type else ()


# platform-managed [gpu] fields: the runner assigns disk sizing, the shared weight-cache volume, and
# retry/wall-clock lifecycle policy; the user never authors them. single-sourced here so the public
# serializer (JobSpec.to_dict), the user-facing parser (flash.schema), and the runner's effective-spec
# validator strip, reject, and exclude exactly the same set. divergence would leak a managed field
# into the public surface or fail the submit round trip.
MANAGED_GPU_KEYS = frozenset(
    {"disk_gb", "network_volume", "network_volume_gb", "max_retries", "max_wall_seconds"}
)

# the other two halves of the same boundary. together with the managed gpu registry they name every field
# that separates the worker contract from the public one -- which is the whole difference between
# them, and is why `to_dict` can be a projection of `to_internal_dict` rather than a second
# serializer. they were bare `data.pop(...)` calls, so the boundary existed only as a sequence of
# statements: nothing could enumerate it, and a field added to one serializer but not the other was
# invisible until a submit round trip or a recovery digest failed.
#
# order is irrelevant (the payload is canonicalized with sort_keys), but presence is a recovery
# contract: `_preparation_digest` hashes the public payload, so moving a name in or out of this set
# invalidates the stored digest of every warm-start and workload-profile run in flight.
#
MANAGED_TOP_LEVEL_KEYS = frozenset(
    {
        # server-assigned identity -- never authored in a config.
        "run_id",
        # runner-managed and absent from the public config and status spec. internal round trips
        # keep the value and markers through to_internal_dict().
        "model_revision",
        "model_revision_auto",
        "model_revision_force_pin",
        # the provenance marker only. gpu.count=1 stays in the public [gpu] object for digest
        # stability; internal round trips carry the marker verbatim.
        "gpu_count_auto",
        "workload_profile",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
    }
)

# platform-managed [train] fields: the control-plane-assigned artifact repo and the resolve-once
# revision pin of a warm-start source.
MANAGED_TRAIN_KEYS = frozenset({"hf_repo", "init_from_adapter_revision"})

# platform-managed [environment] field: the env ref is resolved once and pinned by the control plane.
# `pip` deliberately stays public -- it is the author's own scorer dependencies, which only they can
# declare.
MANAGED_ENVIRONMENT_KEYS = frozenset({"resolved_sha", "package"})

# every managed section, so the public/worker boundary can be walked rather than restated. the two
# things not here are the ones that are not removals: `train.lora_rank`/`lora_alpha` are stripped
# only for a warm start, and `gpu.type_fallbacks` is respelled into `gpu.type` rather than dropped.
MANAGED_SECTION_KEYS = (
    ("train", MANAGED_TRAIN_KEYS),
    ("gpu", MANAGED_GPU_KEYS),
    ("environment", MANAGED_ENVIRONMENT_KEYS),
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
    # platform-managed marker for a runner-resolved immutable model revision. stripped by to_dict()
    # like the other platform-managed carriers and retained in the internal worker spec.
    model_revision_auto: bool = False
    # transient internal request for the runner to verify an exact auto-managed immutable pin instead
    # of resolving the model's current default head. preparation clears it after successful
    # verification, so persisted worker specs carry false rather than a reusable verification request.
    model_revision_force_pin: bool = False
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
        for field_name in ("model_revision_auto", "model_revision_force_pin"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.model_revision_force_pin:
            revision = self.model_revision
            if not self.model_revision_auto:
                raise ValueError("model_revision_force_pin requires model_revision_auto=True")
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError(
                    "model_revision_force_pin requires a full lowercase immutable model_revision"
                )
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

        This is the PUBLIC contract and it is deliberately lossy: what it drops is what the authored
        parser would reject, so the result re-validates through ``flash.schema.spec_from_dict``. Do
        not hand its output to a worker or provider path -- those need ``to_internal_dict()``.

        Built as a PROJECTION of the worker payload, because that is the actual relationship: the
        public contract is the worker contract minus the platform-managed fields, which
        MANAGED_TOP_LEVEL_KEYS and MANAGED_SECTION_KEYS name. This is still a blacklist -- a newly
        added field is public until something strips it, exactly as before -- but the strips are now
        enumerable rather than a run of statements, so a test can assert the boundary instead of
        trusting that two independent `asdict(self)` bodies were kept in sync by hand. What the
        registries deliberately do NOT cover is the two adjustments below that are not removals.

        Its bytes are a recovery contract. ``_preparation_digest`` hashes this output as canonical
        JSON, so key PRESENCE is load-bearing even though key order is not: adding or dropping one
        key here invalidates the stored digest of every warm-start and workload-profile run in
        flight, which fails integrity validation on recovery.
        """
        data = self.to_internal_dict()
        for managed in MANAGED_TOP_LEVEL_KEYS:
            data.pop(managed, None)
        for section, managed_keys in MANAGED_SECTION_KEYS:
            for managed in managed_keys:
                data[section].pop(managed, None)
        train = data["train"]
        if train.get("init_from_adapter"):
            # the source adapter's topology is authoritative for a warm start, so the parser rejects
            # both keys alongside init_from_adapter. the runner writes the inherited rank/alpha onto
            # the public spec, so they must be stripped here or the submit round trip re-validates
            # a combination the parser refuses. conditional, so it cannot be a registry entry.
            train.pop("lora_rank", None)
            train.pop("lora_alpha", None)
        # restore the public list spelling for an ordered pin -- a reshape rather than a removal,
        # which is the other thing no registry can express. the default stands in for the key the
        # worker payload omits when there are no fallbacks: unlike `providers`, this one is read
        # back, so it cannot rely on the omission alone.
        gpu = data["gpu"]
        fallbacks = gpu.pop("type_fallbacks", ())
        if fallbacks:
            gpu["type"] = [gpu["type"], *fallbacks]
        # an empty `providers` needs no strip: the worker payload this projects from already omits
        # it, for the same reason the public contract wants it gone -- an internal default must not
        # read back as an authored empty list, which the parser rejects as a likely config bug.
        return data

    def to_internal_dict(self) -> dict[str, Any]:
        """Return the complete control-plane and worker representation.

        This is the RESOLVED WORKER contract: every managed and resolved field the GPU worker needs,
        including the ones ``to_dict()`` strips. It is what gets persisted as
        ``effective_preparation.worker_spec`` and what ``to_json()`` ships to a provider. The
        authored parser rejects this shape, so it must never be offered as a public payload.

        Its bytes are a recovery contract for the same reason as ``to_dict()`` -- and note the two
        empty-value pops below are part of it: an omitted empty field and an explicit empty one hash
        differently, so removing either pop would break existing digests.
        """
        data = asdict(self)
        # missing and unset are the same internal state. omitting the empty value preserves that state
        # without serializing it into an explicit empty preference, which every parser rejects.
        if not data["gpu"].get("providers"):
            data["gpu"].pop("providers", None)
        # omit an empty field to preserve existing preparation digests.
        if not data["gpu"].get("type_fallbacks"):
            data["gpu"].pop("type_fallbacks", None)
        # omit an unstaged package so historical worker payloads and preparation digests stay stable.
        if not data["environment"].get("package"):
            data["environment"].pop("package", None)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_internal_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Decode a current persisted or internal job spec.

        Authored configuration goes through ``flash.schema.spec_from_dict``. This parser accepts
        platform-managed fields but rejects keys outside the current internal schema.
        """
        if not isinstance(data, dict):
            raise TypeError("job spec must be an object")
        allowed_top_level = {item.name for item in fields(cls)}
        unknown_top_level = sorted(set(data) - allowed_top_level)
        if unknown_top_level:
            raise ValueError(f"job spec has unknown key(s): {', '.join(unknown_top_level)}")
        env = validated_section(
            data, "environment", {item.name for item in fields(EnvironmentSpec)}
        )
        raw_package = env.get("package")
        if raw_package is not None and not isinstance(raw_package, dict):
            raise TypeError("environment.package must be an object")
        if isinstance(raw_package, dict):
            package_keys = {item.name for item in fields(EnvironmentPackageSpec)}
            unknown_package = sorted(set(raw_package) - package_keys)
            if unknown_package:
                raise ValueError(
                    "environment.package has unknown key(s): " + ", ".join(unknown_package)
                )
        package = (
            EnvironmentPackageSpec(
                artifact_revision=str(raw_package.get("artifact_revision") or ""),
                archive_sha256=str(raw_package.get("archive_sha256") or ""),
                manifest_sha256=str(raw_package.get("manifest_sha256") or ""),
            )
            if raw_package is not None
            else None
        )
        train = validated_section(data, "train", {item.name for item in fields(TrainSpec)})
        credit_assignment = (
            _coerce_credit_assignment(train["credit_assignment"])
            if "credit_assignment" in train
            else DEFAULT_CREDIT_ASSIGNMENT
        )
        gpu = validated_section(data, "gpu", {item.name for item in fields(GpuSpec)})
        gpu_type, gpu_type_fallbacks = _parse_persisted_gpu_types(gpu)
        provider, providers = validated_persisted_providers(gpu, gpu_type, gpu_type_fallbacks)
        project_raw = data.get("project", "")
        if not isinstance(project_raw, str):
            raise TypeError("project must be a string")
        project = require_project_id(project_raw) if project_raw.strip() else ""
        model_revision = _model_revision(data.get("model_revision", cls.model_revision))
        model_revision_auto = coerce_bool(data.get("model_revision_auto", False))
        if model_revision and not model_revision_auto:
            raise ValueError("model_revision requires model_revision_auto=True")
        algorithm = normalize_algorithm(data.get("algorithm", cls.algorithm))
        if algorithm in {"grpo", "opd"} and train.get("batch_size") is not None:
            raise ValueError(
                f"train.batch_size does not apply to {algorithm}; use train.prompts_per_step"
            )
        if algorithm == "grpo":
            for name, value in (
                ("prompts_per_step", train.get("prompts_per_step")),
                ("group_size", train.get("group_size")),
            ):
                if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                    raise TypeError(f"train.{name} must be an integer or omitted for GRPO")
        return cls(
            model=data.get("model", cls.model),
            model_revision=model_revision,
            algorithm=algorithm,
            environment=EnvironmentSpec(
                id=env.get("id", ""),
                params=dict(env.get("params") or {}),
                pip=tuple(str(p) for p in env.get("pip") or ()),
                secrets=str_tuple(env.get("secrets")),
                resolved_sha=str(env.get("resolved_sha") or ""),
                package=package,
            ),
            train=TrainSpec(
                epochs=opt_int(train.get("epochs")),
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
                learning_rate=opt_float(train.get("learning_rate")),
                batch_size=opt_int(train.get("batch_size")),
                prompts_per_step=opt_int(train.get("prompts_per_step")),
                max_context_tokens=opt_int(train.get("max_context_tokens")),
                save_every=opt_int(train.get("save_every")),
                max_steps=parse_max_steps(train.get("max_steps")),
                save_at_steps=train.get("save_at_steps"),
                max_examples=opt_int(train.get("max_examples")),
                group_size=opt_int(train.get("group_size")),
                temperature=opt_float(train.get("temperature")),
                max_completion_tokens=opt_int(train.get("max_completion_tokens")),
                kl_penalty_coef=opt_float(train.get("kl_penalty_coef")),
                entropy_quantile=opt_float(train.get("entropy_quantile")),
                thinking_length_penalty_coef=opt_float(train.get("thinking_length_penalty_coef")),
                teacher_model=str(train.get("teacher_model") or ""),
                stop_sequences=str_tuple(train.get("stop_sequences")),
                structured_outputs=str(train.get("structured_outputs") or ""),
                credit_assignment=credit_assignment,
            ),
            gpu=GpuSpec(
                type=gpu_type,
                provider=provider,
                providers=providers,
                disk_gb=int(gpu.get("disk_gb", 60)),
                max_wall_seconds=int(gpu.get("max_wall_seconds", 24 * 3600)),
                max_retries=int(gpu.get("max_retries", 5)),
                # network_volume/network_volume_gb round-trip so the runner-assigned weight cache
                # survives the to_dict()->from_dict() hops in _with_model_disk / _spec_with_gpu /
                # _assign_managed_hf_repo before deploy.
                network_volume=gpu.get("network_volume"),
                network_volume_gb=volume_gb(gpu.get("network_volume_gb")),
                count=gpu.get("count", 1),
                type_fallbacks=gpu_type_fallbacks,
            ),
            run_id=data.get("run_id", "local"),
            thinking=coerce_bool(data.get("thinking", False)),
            wandb=_coerce_wandb(data.get("wandb")),
            seed=parse_seed(data.get("seed", FIXED_SEED)),
            model_revision_auto=model_revision_auto,
            model_revision_force_pin=coerce_bool(data.get("model_revision_force_pin", False)),
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
