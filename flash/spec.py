"""Structured job specification shared by CLI/API/runner and GPU workers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal
from uuid import UUID

from .catalog import DEFAULT_MODEL, normalize_algorithm
from .opd_retry_contract import OPD_RESUME_REVISION_ENV

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


CONTROL_PANEL_URL_ENV = "FLASH_CONTROL_PANEL_URL"
TEACHER_CAPABILITY_ENV = "FLASH_TEACHER_CAPABILITY"
MANAGED_TEACHER_CREDENTIAL_ENV_KEYS = frozenset({"PARASAIL_API_KEY"})
CONTROL_PLANE_OWNED_ENV_KEYS = frozenset(
    {
        "RUN_ID",
        "HF_REPO",
        "FLASH_ARM",
        "SEED",
        OPD_RESUME_REVISION_ENV,
        CONTROL_PANEL_URL_ENV,
        TEACHER_CAPABILITY_ENV,
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
    # Pip requirements the GPU worker needs for this environment; empty means "use defaults"
    # (resolved via worker_pip_for_env in spec_payload / provider submit). An explicit
    # [environment] pip is the escape hatch.
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
    # Derived, not a user knob: always 2 x lora_rank. No ``introduced_in`` so it is absent from
    # TRAIN_SCHEMA_KEYS and ``[train] lora_alpha`` is rejected; spec_from_dict recomputes it from
    # lora_rank and to_dict() omits it. The internal from_dict still round-trips the stored value so
    # a warm-start's inherited parent alpha survives control-plane -> worker serialization.
    lora_alpha: int = 64
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
    batch_size: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
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
    # empty selects managed auto-allocation; a set value hard-pins that validated gpu class.
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

    def __post_init__(self) -> None:
        # coerce/validate here so every path (from_dict and direct construction) is guarded.
        object.__setattr__(self, "count", _gpu_count(self.count))


# platform-managed [gpu] fields: the runner assigns disk sizing, the shared weight-cache volume, and
# retry/wall-clock lifecycle policy; the user never authors them. single-sourced here so the public
# serializer (JobSpec.to_dict), the user-facing parser (flash.schema), and the runner's effective-spec
# validator strip, reject, and exclude exactly the same set. divergence would leak a managed field
# into the public surface or fail the submit round trip.
MANAGED_GPU_KEYS = frozenset(
    {"disk_gb", "network_volume", "network_volume_gb", "max_retries", "max_wall_seconds"}
)

# Removed top-level fields that a PERSISTED record can still carry. Stored run records are never
# rewritten, and effective_preparation.worker_spec is written with to_internal_dict() (asdict), which
# emitted every field including the defaulted ones -- so every record written before a field was
# dropped still names it. from_dict is strict, so without this the first reload after the upgrade
# raises and a still-running job loses its recovery, deploy, and serving paths.
#
# Ignored on READ only: nothing here is a JobSpec field, so an authored config naming one is still
# rejected as unknown by the schema layer's own key check (see schema._TOP_LEVEL_KEYS).
_DROPPED_TOP_LEVEL_KEYS = frozenset({"model_policy", "worker_env"})

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
    model_revision: str = ""
    # platform-managed workload-profile carrier. public configs never author these fields.
    workload_profile_kind: str = ""
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
        profile_kind = str(self.workload_profile_kind or "")
        if profile_kind not in {"", "sft"}:
            raise ValueError("unsupported workload profile kind")
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
        if self.workload_profile_kind:
            return "profile"
        return "rl" if self.algorithm == "grpo" else self.algorithm

    def to_dict(self) -> dict[str, Any]:
        """return the public user-authorable job specification.

        omit platform-managed fields because the control plane/runner assigns them and the public
        parser rejects them. internal callers use ``to_internal_dict()``.
        """
        data = asdict(self)
        # server-assigned identity — never authored in a config.
        data.pop("run_id", None)
        data.pop("workload_profile_kind", None)
        data.pop("workload_profile_input_digest", None)
        data.pop("workload_profile_producer_version", None)
        data.pop("workload_profile", None)
        train = data["train"]
        train.pop("init_from_adapter_revision", None)
        train.pop("hf_repo", None)  # control-plane-assigned artifact repo
        train.pop("lora_alpha", None)  # derived (2 x lora_rank), recomputed on parse
        if train.get("init_from_adapter"):
            train.pop("lora_rank", None)
        # runner-assigned disk sizing, shared weight-cache volume, and retry/wall-clock lifecycle.
        gpu = data["gpu"]
        for managed in MANAGED_GPU_KEYS:
            gpu.pop(managed, None)
        data["environment"].pop("resolved_sha", None)  # resolve-once env ref pin
        # platform-managed worker requirement list, resolved by worker_pip_for_env at submit.
        data["environment"].pop("pip", None)
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
        provider = gpu.get("provider", "")
        if not isinstance(provider, str):
            raise TypeError("gpu.provider must be a string")
        provider = provider.strip().lower()
        if provider or gpu_type:
            from flash.providers import PROVIDER_NAMES
            from flash.providers.base import providers_for

            if provider and provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {provider!r}")
            if gpu_type and provider and provider not in providers_for(gpu_type):
                raise ValueError(
                    f"gpu.provider {provider!r} cannot provision gpu.type {gpu_type!r}"
                )
        project_raw = data.get("project", "")
        if not isinstance(project_raw, str):
            raise TypeError("project must be a string")
        project = require_project_id(project_raw) if project_raw.strip() else ""
        return cls(
            model=data.get("model", cls.model),
            model_revision=_model_revision(data.get("model_revision", cls.model_revision)),
            algorithm=normalize_algorithm(data.get("algorithm", cls.algorithm)),
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
                # round-trip a stored alpha (internal carrier + warm-start's inherited parent alpha);
                # derive 2 x rank when absent (a stripped public to_dict() being reconstructed).
                lora_alpha=(
                    int(train["lora_alpha"])
                    if "lora_alpha" in train
                    else 2 * int(train.get("lora_rank", 32))
                ),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
                init_from_adapter_revision=str(train.get("init_from_adapter_revision") or ""),
                hf_repo=str(train.get("hf_repo") or ""),
                learning_rate=_opt_float(train.get("learning_rate")),
                batch_size=_opt_int(train.get("batch_size")),
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
            ),
            run_id=data.get("run_id", "local"),
            thinking=coerce_bool(data.get("thinking", False)),
            wandb=_coerce_wandb(data.get("wandb")),
            seed=parse_seed(data.get("seed", FIXED_SEED)),
            workload_profile_kind=str(data.get("workload_profile_kind") or ""),
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
