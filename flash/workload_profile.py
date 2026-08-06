"""Immutable workload profiles used to freeze evidence-backed cost quotes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any

SFT_PROFILE_KIND = "sft"
ROLLOUT_PROFILE_KINDS = ("grpo", "opd")
WORKLOAD_PROFILE_SCHEMA_VERSION = 1
SFT_PACKING_POLICY_VERSION = 1
ROLLOUT_SAMPLE_POLICY_VERSION = 1
_PROFILE_RUN_PREFIX = "profile-sft-"
_ROLLOUT_PROFILE_RUN_PREFIX = "profile-rollout-"

# measured generation latency ages out; the shape it was measured from does not. a provider that
# slows down, or a card whose neighbours change, invalidates the seconds without invalidating the
# token counts, so the two carry different lifetimes and only one of them is keyed.
ROLLOUT_LATENCY_MAX_AGE_S = 24 * 60 * 60

# which rows the profile measured. sft is never sampled: it either measures every source row or the
# deterministic max_examples prefix training will consume, so both policies are exact.
SFT_SAMPLE_POLICY_FULL = "exact-full"
SFT_SAMPLE_POLICY_PREFIX = "exact-prefix"


def sft_sample_policy(max_examples: object) -> str:
    """The row-selection rule this workload uses, derived from the authored ``max_examples``."""
    try:
        cap = int(max_examples or 0)
    except (TypeError, ValueError):
        cap = 0
    return SFT_SAMPLE_POLICY_PREFIX if cap > 0 else SFT_SAMPLE_POLICY_FULL


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_mapping(value: object) -> str:
    return _sha256(value if isinstance(value, dict) else {})


def sft_profile_input_payload(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> dict[str, object]:
    """Return the non-secret immutable inputs that determine one sft workload profile."""
    train = spec.train
    environment = spec.environment
    return {
        "schema_version": WORKLOAD_PROFILE_SCHEMA_VERSION,
        "kind": SFT_PROFILE_KIND,
        "packing_policy_version": SFT_PACKING_POLICY_VERSION,
        "producer_version": str(producer_version),
        "environment": {
            "id": str(environment.id),
            "resolved_sha": str(environment.resolved_sha or ""),
            "params_sha256": _digest_mapping(environment.params),
        },
        "model": {
            "id": str(spec.model),
            "revision": str(spec.model_revision or ""),
            "tokenizer_revision": str(tokenizer_revision),
        },
        "seed": int(spec.seed),
        "thinking": bool(spec.thinking),
        "worker_env_sha256": _digest_mapping(spec.worker_env),
        "train": {
            "epochs": train.epochs,
            "batch_size": train.batch_size,
            "max_context_tokens": train.max_context_tokens,
            "max_steps": train.max_steps,
            "max_examples": train.max_examples,
        },
    }


def sft_profile_input_digest(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> str:
    return _sha256(
        sft_profile_input_payload(
            spec,
            tokenizer_revision=tokenizer_revision,
            producer_version=producer_version,
        )
    )


def sft_profile_run_id(input_digest: str) -> str:
    if len(input_digest) != 64 or any(c not in "0123456789abcdef" for c in input_digest):
        raise ValueError("input_digest must be a lowercase sha256 hex digest")
    return f"{_PROFILE_RUN_PREFIX}{input_digest}"


@dataclass(frozen=True)
class SftWorkloadProfile:
    """Aggregate-only description of the exact sft workload consumed by training.

    Two kinds of field live here and they are not interchangeable. Everything down to
    ``sample_policy`` is *measurement*: derived only from the immutable inputs the input digest
    keys, so re-running the same preprocessing reproduces it byte for byte. ``created_at`` is
    *provenance*: it records which profile run emitted this artifact and is deliberately outside
    ``_content()``, because the training worker re-derives the measurement to check that the
    workload did not move under the frozen quote, and a timestamp in the digest would fail that
    check on every run for the one reason that is not a workload change.

    There is no trust flag. For sft the measurement is exact or the worker raises, so an artifact
    that exists is one the quote may use; a profile that could not be produced is a failed profile
    *run*, and its state and error live on the run record. PR2's sampled rollout evidence is where
    a trust verdict has content, because a sampled profile can complete and still be too noisy to
    quote from.
    """

    input_digest: str
    producer_version: str
    tokenizer_revision: str
    environment_id: str
    environment_revision: str
    source_examples: int
    selected_examples: int
    retained_examples: int
    dropped_examples: int
    epochs: int
    max_length: int
    packing_mode: str
    architecture_mode: str
    packed_blocks: int
    real_tokens_per_epoch: int
    supervised_tokens_per_epoch: int
    padded_compute_tokens_per_epoch: int
    authoritative_real_tokens: int
    authoritative_supervised_tokens: int
    authoritative_compute_tokens: int
    realized_max_length: int
    examples_per_update: int
    derived_steps: int
    authoritative_steps: int
    packing_efficiency: float
    sample_policy: str
    schema_version: int = WORKLOAD_PROFILE_SCHEMA_VERSION
    kind: str = SFT_PROFILE_KIND
    # unix seconds stamped by the profile run that produced this artifact. 0.0 on a recomputation
    # (the training worker re-derives the measurement to check parity and is not a producer).
    created_at: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported workload profile schema version")
        if self.kind != SFT_PROFILE_KIND:
            raise ValueError("sft workload profile kind must be 'sft'")
        if len(self.input_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.input_digest
        ):
            raise ValueError("input_digest must be a lowercase sha256 hex digest")
        if not self.producer_version:
            raise ValueError("producer_version is required")
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision is required")
        if not self.environment_id:
            raise ValueError("environment_id is required")
        if not self.environment_revision:
            raise ValueError("environment_revision is required")
        counts = {
            "source_examples": self.source_examples,
            "selected_examples": self.selected_examples,
            "retained_examples": self.retained_examples,
            "dropped_examples": self.dropped_examples,
            "packed_blocks": self.packed_blocks,
            "real_tokens_per_epoch": self.real_tokens_per_epoch,
            "supervised_tokens_per_epoch": self.supervised_tokens_per_epoch,
            "padded_compute_tokens_per_epoch": self.padded_compute_tokens_per_epoch,
            "authoritative_real_tokens": self.authoritative_real_tokens,
            "authoritative_supervised_tokens": self.authoritative_supervised_tokens,
            "authoritative_compute_tokens": self.authoritative_compute_tokens,
            "realized_max_length": self.realized_max_length,
            "examples_per_update": self.examples_per_update,
            "derived_steps": self.derived_steps,
            "authoritative_steps": self.authoritative_steps,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.selected_examples > self.source_examples:
            raise ValueError("selected examples cannot exceed source examples")
        if self.retained_examples + self.dropped_examples != self.selected_examples:
            raise ValueError("retained plus dropped examples must equal selected examples")
        if self.retained_examples < 1:
            raise ValueError("at least one sft example must be retained")
        if self.packed_blocks < 1:
            raise ValueError("at least one packed block is required")
        if self.real_tokens_per_epoch < self.retained_examples:
            raise ValueError("real token count cannot be smaller than retained example count")
        if self.supervised_tokens_per_epoch > self.real_tokens_per_epoch:
            raise ValueError("supervised tokens cannot exceed real tokens")
        if self.padded_compute_tokens_per_epoch < self.real_tokens_per_epoch:
            raise ValueError("padded compute tokens cannot be smaller than real tokens")
        if self.authoritative_real_tokens < 1:
            raise ValueError("authoritative real tokens must be positive")
        if self.authoritative_supervised_tokens > self.authoritative_real_tokens:
            raise ValueError("authoritative supervised tokens cannot exceed real tokens")
        if self.authoritative_compute_tokens < self.authoritative_real_tokens:
            raise ValueError("authoritative compute tokens cannot be smaller than real tokens")
        if not 1 <= self.realized_max_length <= self.max_length:
            raise ValueError("realized max length must be within the configured maximum")
        if self.examples_per_update < 1:
            raise ValueError("examples per update must be positive")
        if self.derived_steps < 1 or self.authoritative_steps < 1:
            raise ValueError("step counts must be positive")
        if self.packing_mode not in {"packed", "exact-unpacked"}:
            raise ValueError("packing_mode must be 'packed' or 'exact-unpacked'")
        if not self.architecture_mode:
            raise ValueError("architecture_mode is required")
        if not math.isfinite(self.packing_efficiency) or not 0 < self.packing_efficiency <= 1:
            raise ValueError("packing_efficiency must be finite and in (0, 1]")
        expected_efficiency = self.real_tokens_per_epoch / self.padded_compute_tokens_per_epoch
        if not math.isclose(self.packing_efficiency, expected_efficiency, rel_tol=1e-12):
            raise ValueError("packing_efficiency does not match token aggregates")
        if self.sample_policy not in {SFT_SAMPLE_POLICY_FULL, SFT_SAMPLE_POLICY_PREFIX}:
            raise ValueError("sample_policy must be 'exact-full' or 'exact-prefix'")
        if self.sample_policy == SFT_SAMPLE_POLICY_FULL and (
            self.selected_examples != self.source_examples
        ):
            raise ValueError("an exact-full profile must select every source example")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, (int, float)):
            raise ValueError("created_at must be a number")
        if not math.isfinite(self.created_at) or self.created_at < 0:
            raise ValueError("created_at must be a non-negative unix timestamp")

    def _content(self) -> dict[str, object]:
        """The measurement alone: what the content digest and worker parity are taken over."""
        return {name: getattr(self, name) for name in _measurement_field_names()}

    @property
    def content_digest(self) -> str:
        """Digest of the measurement, so the same workload digests the same on any profile run."""
        return _sha256(self._content())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content(),
            "created_at": float(self.created_at),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, raw: object) -> SftWorkloadProfile:
        if not isinstance(raw, dict):
            raise ValueError("workload profile must be an object")
        data = dict(raw)
        digest = data.pop("content_digest", None)
        if not isinstance(digest, str):
            raise ValueError("workload profile has no content digest")
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("workload profile fields do not match the schema")
        profile = cls(**data)
        if profile.content_digest != digest:
            raise ValueError("workload profile content digest does not match")
        return profile


def _measurement_field_names(cls: type | None = None) -> tuple[str, ...]:
    """Every field except provenance, taken from ``compare`` so the two can never drift apart.

    ``compare=False`` is what excludes a field from dataclass equality, which is exactly the
    training worker's parity check. Deriving the digest from the same flag keeps one definition of
    "this is measurement" instead of a hand-maintained list that a new field could silently miss.
    """
    return tuple(f.name for f in fields(cls or SftWorkloadProfile) if f.compare)


def rollout_profile_run_id(input_digest: str) -> str:
    if len(input_digest) != 64 or any(c not in "0123456789abcdef" for c in input_digest):
        raise ValueError("input_digest must be a lowercase sha256 hex digest")
    return f"{_ROLLOUT_PROFILE_RUN_PREFIX}{input_digest}"


def rollout_profile_input_payload(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> dict[str, object]:
    """Return the non-secret immutable inputs that determine one rollout workload profile.

    Deliberately excludes the training horizon (``epochs``, ``max_steps``, ``max_examples``): a
    short run and a long run over the same environment, model and generation settings draw
    completions from the same distribution, so they share one profile and one charge. Anything that
    changes what the model is asked or how much it may say is keyed, because that moves the
    distribution rather than how many times it is sampled.
    """
    train = spec.train
    environment = spec.environment
    return {
        "schema_version": WORKLOAD_PROFILE_SCHEMA_VERSION,
        "kind": str(spec.algorithm),
        "sample_policy_version": ROLLOUT_SAMPLE_POLICY_VERSION,
        "producer_version": str(producer_version),
        "environment": {
            "id": str(environment.id),
            "resolved_sha": str(environment.resolved_sha or ""),
            "params_sha256": _digest_mapping(environment.params),
        },
        "model": {
            "id": str(spec.model),
            "revision": str(spec.model_revision or ""),
            "tokenizer_revision": str(tokenizer_revision),
        },
        "seed": int(spec.seed),
        "thinking": bool(spec.thinking),
        "worker_env_sha256": _digest_mapping(spec.worker_env),
        "generation": {
            "max_completion_tokens": train.max_completion_tokens,
            "max_context_tokens": train.max_context_tokens,
            "group_size": train.group_size,
            "teacher_model": str(getattr(train, "teacher_model", "") or ""),
            # temperature moves the length distribution this profile exists to measure, so a
            # reading taken at one setting must never be reused for a run at another. None is
            # keyed distinctly from any float: it means "whatever the backend defaults to",
            # which is not knowably the same distribution as an explicit value.
            "temperature": (
                None if train.temperature is None else float(train.temperature)
            ),
        },
    }


def rollout_profile_input_digest(
    spec: Any,
    *,
    tokenizer_revision: str,
    producer_version: str,
) -> str:
    return _sha256(
        rollout_profile_input_payload(
            spec,
            tokenizer_revision=tokenizer_revision,
            producer_version=producer_version,
        )
    )


@dataclass(frozen=True)
class RolloutWorkloadProfile:
    """Sampled description of what one grpo/opd step actually generates and grades.

    Unlike the sft profile this is a SAMPLE, not a census, and that difference is the whole reason
    the type carries a trust verdict. An sft profile either measures the exact rows training will
    consume or the run fails; a rollout profile can finish, report numbers, and still be too thin,
    too failure-ridden or too stale to move a quote. ``trustworthy`` is what the quote consults, and
    it is deliberately not a single "did it run" flag.

    Two clocks. The token distribution is structural: the same environment, model and generation
    settings produce it again, so it is keyed by the input digest and never expires. The measured
    seconds are not: they belong to one provider on one card at one moment, so they carry
    ``measured_at`` and age out after ``ROLLOUT_LATENCY_MAX_AGE_S``. Mixing the two lifetimes is how
    a profile silently starts quoting last week's throughput.

    ``completion_tokens_*`` is the field this type exists for. The quote previously billed
    ``max_completion_tokens``, a capacity knob, as though it were expected work; measured realized
    generation runs 0.077-0.392x of it and the resulting bias tracks the knob rather than the
    workload. Aggregates and provenance only: no prompts, no completions, no token ids, no
    credentials.
    """

    input_digest: str
    producer_version: str
    tokenizer_revision: str
    environment_id: str
    environment_revision: str
    kind: str
    # what was sampled, and how much of it survived. a profile whose successes are mostly failures
    # measured the failure path, not the workload.
    sampled_prompts: int
    completed_rollouts: int
    failed_rollouts: int
    # realized generation. the distribution, not a point estimate: a mean alone cannot tell a
    # uniformly-short workload from a bimodal one whose long tail sets the step time.
    completion_tokens_mean: float
    completion_tokens_p50: int
    completion_tokens_p90: int
    completion_tokens_max: int
    prompt_tokens_mean: float
    # how many completions ended because the model stopped rather than because it ran out of room.
    # a high truncation rate means the cap IS binding and the realized distribution is censored,
    # which is the one case where billing the cap is not wrong.
    truncated_rollouts: int
    eos_rollouts: int
    # measured seconds. these age.
    generation_seconds_per_completion: float
    reward_seconds_per_completion: float
    reward_samples: int
    reward_failures: int
    reference_gpu: str
    reference_provider: str
    sample_policy: str
    sample_policy_version: int = ROLLOUT_SAMPLE_POLICY_VERSION
    schema_version: int = WORKLOAD_PROFILE_SCHEMA_VERSION
    # provenance, outside the content digest for the same reason sft's created_at is: a worker
    # re-deriving the measurement must reproduce the digest, and a timestamp never would.
    created_at: float = field(default=0.0, compare=False)
    measured_at: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported workload profile schema version")
        if self.sample_policy_version != ROLLOUT_SAMPLE_POLICY_VERSION:
            raise ValueError("unsupported rollout sample policy version")
        if self.kind not in ROLLOUT_PROFILE_KINDS:
            raise ValueError("rollout workload profile kind must be 'grpo' or 'opd'")
        if len(self.input_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.input_digest
        ):
            raise ValueError("input_digest must be a lowercase sha256 hex digest")
        for name in ("producer_version", "tokenizer_revision", "environment_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if not self.environment_revision:
            raise ValueError("environment_revision is required")
        if not self.sample_policy:
            raise ValueError("sample_policy is required")
        counts = {
            "sampled_prompts": self.sampled_prompts,
            "completed_rollouts": self.completed_rollouts,
            "failed_rollouts": self.failed_rollouts,
            "completion_tokens_p50": self.completion_tokens_p50,
            "completion_tokens_p90": self.completion_tokens_p90,
            "completion_tokens_max": self.completion_tokens_max,
            "truncated_rollouts": self.truncated_rollouts,
            "eos_rollouts": self.eos_rollouts,
            "reward_samples": self.reward_samples,
            "reward_failures": self.reward_failures,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        rates = {
            "completion_tokens_mean": self.completion_tokens_mean,
            "prompt_tokens_mean": self.prompt_tokens_mean,
            "generation_seconds_per_completion": self.generation_seconds_per_completion,
            "reward_seconds_per_completion": self.reward_seconds_per_completion,
        }
        for name, value in rates.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.completion_tokens_p90 < self.completion_tokens_p50:
            raise ValueError("p90 completion tokens cannot be below p50")
        if self.completion_tokens_max < self.completion_tokens_p90:
            raise ValueError("max completion tokens cannot be below p90")
        if self.truncated_rollouts + self.eos_rollouts > self.completed_rollouts:
            raise ValueError("truncated plus eos rollouts cannot exceed completed rollouts")
        if self.reward_failures > self.reward_samples:
            raise ValueError("reward failures cannot exceed reward samples")
        for name in ("created_at", "measured_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative unix timestamp")

    @property
    def truncation_rate(self) -> float:
        """Share of completions that hit the cap instead of stopping on their own."""
        if self.completed_rollouts <= 0:
            return 0.0
        return self.truncated_rollouts / self.completed_rollouts

    def trustworthy(self, *, now: float, min_rollouts: int = 8) -> tuple[bool, str]:
        """Whether this reading may move a quote, and if not, why not.

        Returns a reason rather than a bare bool because every caller that refuses a profile has to
        tell a user what to do about it, and re-deriving the reason from the fields at each call
        site is how two call sites end up disagreeing about what "untrustworthy" meant.

        The bar is deliberately about *evidence*, not success: a profile that ran cleanly on two
        prompts is not more quotable than one that failed, it is just quieter about it.
        """
        if self.completed_rollouts < min_rollouts:
            return False, (
                f"only {self.completed_rollouts} rollout(s) completed, "
                f"below the {min_rollouts} needed to quote from"
            )
        if self.failed_rollouts >= self.completed_rollouts:
            return False, (
                f"{self.failed_rollouts} rollout(s) failed against "
                f"{self.completed_rollouts} completed; the sample describes the failure path"
            )
        if self.completion_tokens_max <= 0:
            return False, "every sampled completion was empty, so no generation was measured"
        age = now - self.measured_at
        if self.measured_at <= 0 or age > ROLLOUT_LATENCY_MAX_AGE_S:
            return False, (
                "measured generation latency is older than "
                f"{ROLLOUT_LATENCY_MAX_AGE_S // 3600}h and must be re-measured"
            )
        return True, ""

    def _content(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _measurement_field_names(type(self))}

    @property
    def content_digest(self) -> str:
        return _sha256(self._content())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content(),
            "created_at": float(self.created_at),
            "measured_at": float(self.measured_at),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, raw: object) -> RolloutWorkloadProfile:
        if not isinstance(raw, dict):
            raise ValueError("workload profile must be an object")
        data = dict(raw)
        digest = data.pop("content_digest", None)
        if not isinstance(digest, str):
            raise ValueError("workload profile has no content digest")
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("workload profile fields do not match the schema")
        profile = cls(**data)
        if profile.content_digest != digest:
            raise ValueError("workload profile content digest does not match")
        return profile


class WorkloadProfileMismatch(ValueError):
    """The attached profile is absent, malformed, or does not describe this exact spec.

    Distinct from the transient failures a quote can also hit (a hub lookup for revision-aware
    sizing, say): this identity is re-derived from the spec itself, so it resolves the same way
    every time. Callers that retry infrastructure use the type to fail fast instead.
    """


def require_matching_sft_profile(
    raw: object,
    *,
    input_digest: str,
    producer_version: str,
    tokenizer_revision: str,
) -> SftWorkloadProfile:
    try:
        profile = SftWorkloadProfile.from_dict(raw)
    except ValueError as exc:
        raise WorkloadProfileMismatch(str(exc)) from exc
    if profile.input_digest != input_digest:
        raise WorkloadProfileMismatch("workload profile input digest does not match")
    if profile.producer_version != producer_version:
        raise WorkloadProfileMismatch("workload profile producer version does not match")
    if profile.tokenizer_revision != tokenizer_revision:
        raise WorkloadProfileMismatch("workload profile tokenizer revision does not match")
    return profile


def require_matching_rollout_profile(
    raw: object,
    *,
    input_digest: str,
    producer_version: str,
    tokenizer_revision: str,
    now: float,
) -> RolloutWorkloadProfile:
    """The rollout counterpart of ``require_matching_sft_profile``, plus the trust gate.

    Identity and trust are checked in one place on purpose. A caller that verified the digest and
    then forgot the verdict would quote from a profile that matches this spec exactly and measured
    almost nothing, which is the failure this type was given a verdict to prevent.
    """
    try:
        profile = RolloutWorkloadProfile.from_dict(raw)
    except ValueError as exc:
        raise WorkloadProfileMismatch(str(exc)) from exc
    if profile.input_digest != input_digest:
        raise WorkloadProfileMismatch("workload profile input digest does not match")
    if profile.producer_version != producer_version:
        raise WorkloadProfileMismatch("workload profile producer version does not match")
    if profile.tokenizer_revision != tokenizer_revision:
        raise WorkloadProfileMismatch("workload profile tokenizer revision does not match")
    ok, reason = profile.trustworthy(now=now)
    if not ok:
        raise WorkloadProfileMismatch(f"rollout workload profile is not trustworthy: {reason}")
    return profile
