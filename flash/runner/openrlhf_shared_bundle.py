"""sealed control-plane bundles for compatible shared OpenRLHF runs."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum

from flash.catalog import ModelInfo, get_model, supports_image_training
from flash.engine.vram import grpo_rollout_seq_len, opd_rollout_seq_len
from flash.providers.base import get_gpu_info, supports_fp8_kv
from flash.spec import JobSpec

_TARGET_MODULES = ("all-linear",)
_SUPPORTED_ALGORITHMS = frozenset({"grpo", "opd"})
_SUPPORTED_GPU_CAPS = {"H100": 4, "H200": 8}
_MAX_SHARED_MODEL_PARAMS_B = 10.0
_SHARED_RUNTIME_RESERVE_GIB = {"H100": 24.0, "H200": 28.0}
_LORA_PERSISTENT_BYTES_PER_PARAM = 9.0
_LORA_ACTIVE_BYTES_PER_PARAM = 4.0
_REFERENCE_ADAPTER_BYTES_PER_PARAM = 2.0


class BundleAdmissionOutcome(StrEnum):
    """result of offering one logical run to a bundle."""

    ADMITTED = "admitted"
    QUEUED = "queued"
    REJECTED = "rejected"


class LogicalRunStatus(StrEnum):
    """independent public lifecycle for one admitted logical run."""

    QUEUED = "queued"
    ACTIVE = "active"
    FINISHING = "finishing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_RUN_TRANSITIONS = {
    LogicalRunStatus.QUEUED: frozenset(
        {LogicalRunStatus.ACTIVE, LogicalRunStatus.FAILED, LogicalRunStatus.CANCELLED}
    ),
    LogicalRunStatus.ACTIVE: frozenset(
        {LogicalRunStatus.FINISHING, LogicalRunStatus.FAILED, LogicalRunStatus.CANCELLED}
    ),
    LogicalRunStatus.FINISHING: frozenset(
        {LogicalRunStatus.DONE, LogicalRunStatus.FAILED, LogicalRunStatus.CANCELLED}
    ),
    LogicalRunStatus.DONE: frozenset(),
    LogicalRunStatus.FAILED: frozenset(),
    LogicalRunStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class BundleCompatibilityKey:
    """immutable engine geometry required by one prepared job specification."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    processor_id: str
    engine_mode: str
    tensor_parallel_size: int
    lora_rank: int
    lora_target_modules: tuple[str, ...]
    max_model_length: int
    parameter_dtype: str
    kv_cache_dtype: str
    attention_backend: str
    kernel_requirements: tuple[str, ...]
    structured_output_capable: bool
    gpu_type: str
    gpu_count: int

    @classmethod
    def from_job_spec(cls, spec: JobSpec) -> BundleCompatibilityKey:
        """derive a deterministic compatibility key from a prepared worker spec."""

        info = _catalog_model_info(spec)
        gpu_type = _gpu_type(spec)
        gpu = get_gpu_info(gpu_type)
        max_model_length = _rollout_context(spec)
        engine_mode = "language-model-only" if supports_image_training(info) else "single-turn-text"
        attention_backend = "flashinfer-or-triton" if gpu.sm == "sm120" else "vllm-default"
        architecture = (
            "hybrid-linear-attention"
            if int(info.num_linear_attention_layers or 0) > 0
            else "dense-attention"
        )
        kernel_requirements = tuple(sorted((architecture, f"compute-capability:{gpu.sm}")))
        return cls(
            model_id=spec.model,
            model_revision=spec.model_revision,
            tokenizer_id=spec.model,
            processor_id=spec.model,
            engine_mode=engine_mode,
            tensor_parallel_size=1,
            lora_rank=int(spec.train.lora_rank),
            lora_target_modules=_TARGET_MODULES,
            max_model_length=max_model_length,
            parameter_dtype="bf16",
            kv_cache_dtype="fp8" if supports_fp8_kv(gpu_type) else "bf16",
            attention_backend=attention_backend,
            kernel_requirements=kernel_requirements,
            structured_output_capable=bool(spec.train.structured_outputs),
            gpu_type=gpu_type,
            gpu_count=int(spec.gpu.count),
        )

    @property
    def digest(self) -> str:
        """return a stable content digest suitable for internal bundle identities."""

        canonical = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def incompatibility_reason(self, other: BundleCompatibilityKey) -> str | None:
        """describe the first deterministic geometry mismatch, if any."""

        for field_name in self.__dataclass_fields__:
            expected = getattr(self, field_name)
            actual = getattr(other, field_name)
            if expected != actual:
                return (
                    f"incompatible shared-engine {field_name}: "
                    f"bundle requires {expected!r}, run requires {actual!r}"
                )
        return None


@dataclass(frozen=True, slots=True)
class BundleAdmissionEstimate:
    """conservative pre-allocation capacity estimate for one compatibility key."""

    estimated_n: int
    gpu_type: str
    gpu_vram_gib: float
    shared_base_gib: float
    shared_runtime_reserve_gib: float
    active_adapter_workspace_gib: float
    per_run_persistent_gib: float
    usable_per_run_gib: float
    safety_cap: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BundleAdmissionDecision:
    """one surfaced admission result including capacity and a concrete reason."""

    outcome: BundleAdmissionOutcome
    run_id: str
    bundle_id: str | None
    estimated_n: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BundleRunSnapshot:
    """immutable view of one logical run's independent bundle status."""

    run_id: str
    status: LogicalRunStatus
    error: str | None
    spec_json: str

    @property
    def spec(self) -> JobSpec:
        """reconstruct the sealed prepared job specification."""

        return JobSpec.from_json(self.spec_json)


@dataclass(slots=True)
class _BundleRunRecord:
    run_id: str
    spec_json: str
    status: LogicalRunStatus = LogicalRunStatus.QUEUED
    error: str | None = None

    def snapshot(self) -> BundleRunSnapshot:
        return BundleRunSnapshot(
            run_id=self.run_id,
            status=self.status,
            error=self.error,
            spec_json=self.spec_json,
        )


def _gpu_type(spec: JobSpec) -> str:
    return str(spec.gpu.exact_type or spec.gpu.type).strip()


def _rollout_context(spec: JobSpec) -> int:
    max_context = int(spec.train.max_context_tokens or 0)
    max_completion = spec.train.max_completion_tokens
    if spec.algorithm == "opd":
        return opd_rollout_seq_len(max_context, max_completion, spec.thinking)
    return grpo_rollout_seq_len(max_context, max_completion, spec.thinking)


def _catalog_model_info(spec: JobSpec) -> ModelInfo:
    try:
        return get_model(spec.model)
    except ValueError as exc:
        raise ValueError(
            f"shared bundle admission requires cataloged model geometry for {spec.model!r}"
        ) from exc


def _lora_parameter_count(info: ModelInfo, rank: int) -> int:
    target_shapes = tuple(info.lora_target_shapes or ())
    if not target_shapes:
        return 0
    target_dimensions = sum(
        (int(in_features) + int(out_features)) * int(count)
        for in_features, out_features, count in target_shapes
    )
    return max(1, int(rank)) * target_dimensions


def _gib(byte_count: float) -> float:
    return float(byte_count) / float(1024**3)


def estimate_bundle_admission(spec: JobSpec) -> BundleAdmissionEstimate:
    """estimate safe max-N before provider allocation or live GPU measurement."""

    gpu_type = _gpu_type(spec)
    try:
        gpu = get_gpu_info(gpu_type)
    except ValueError as exc:
        return BundleAdmissionEstimate(
            estimated_n=0,
            gpu_type=gpu_type,
            gpu_vram_gib=0.0,
            shared_base_gib=0.0,
            shared_runtime_reserve_gib=0.0,
            active_adapter_workspace_gib=0.0,
            per_run_persistent_gib=0.0,
            usable_per_run_gib=0.0,
            safety_cap=0,
            reason=str(exc),
        )

    if spec.algorithm not in _SUPPORTED_ALGORITHMS:
        return _unsupported_estimate(spec, gpu.vram_gb, "shared bundles support only GRPO and OPD")
    if int(spec.gpu.count) != 1:
        return _unsupported_estimate(
            spec,
            gpu.vram_gb,
            "shared bundles require exactly one GPU before the multi-GPU controller exists",
        )
    if gpu_type not in _SUPPORTED_GPU_CAPS:
        return _unsupported_estimate(
            spec,
            gpu.vram_gb,
            f"shared bundle admission is not profiled for GPU class {gpu_type!r}",
        )

    try:
        info = _catalog_model_info(spec)
    except ValueError as exc:
        return _unsupported_estimate(spec, gpu.vram_gb, str(exc))
    if info.params_b <= 0:
        return _unsupported_estimate(
            spec,
            gpu.vram_gb,
            "shared bundle admission requires a known base-model parameter count",
        )
    if info.params_b > _MAX_SHARED_MODEL_PARAMS_B:
        return _unsupported_estimate(
            spec,
            gpu.vram_gb,
            "shared-resident one-GPU admission is limited to models at or below 10B before PR9",
        )

    lora_parameters = _lora_parameter_count(info, spec.train.lora_rank)
    if lora_parameters <= 0:
        return _unsupported_estimate(
            spec,
            gpu.vram_gb,
            "shared bundle admission requires cataloged LoRA target-module geometry",
        )

    shared_base_gib = _gib(2.0 * info.params_b * 1_000_000_000 * 2.0)
    shared_runtime_reserve_gib = _SHARED_RUNTIME_RESERVE_GIB[gpu_type]
    active_adapter_workspace_gib = _gib(lora_parameters * _LORA_ACTIVE_BYTES_PER_PARAM)
    per_run_bytes = lora_parameters * _LORA_PERSISTENT_BYTES_PER_PARAM
    if float(spec.train.kl_penalty_coef or 0.0) > 0:
        per_run_bytes += lora_parameters * _REFERENCE_ADAPTER_BYTES_PER_PARAM
    per_run_persistent_gib = _gib(per_run_bytes)
    usable_per_run_gib = max(
        0.0,
        float(gpu.vram_gb)
        - shared_base_gib
        - shared_runtime_reserve_gib
        - active_adapter_workspace_gib,
    )
    memory_cap = math.floor(usable_per_run_gib / per_run_persistent_gib)

    rank = max(1, int(spec.train.lora_rank))
    context_scale = max(1, math.ceil(_rollout_context(spec) / 8192))
    base_cap = _SUPPORTED_GPU_CAPS[gpu_type]
    rank_adjusted_cap = max(1, math.floor(base_cap * 64 / rank))
    safety_cap = max(1, rank_adjusted_cap // context_scale)
    estimated_n = max(0, min(memory_cap, safety_cap))
    reason = None
    if estimated_n < 1:
        reason = "estimated shared base, runtime reserve, and one run do not fit safely"
    return BundleAdmissionEstimate(
        estimated_n=estimated_n,
        gpu_type=gpu_type,
        gpu_vram_gib=float(gpu.vram_gb),
        shared_base_gib=shared_base_gib,
        shared_runtime_reserve_gib=shared_runtime_reserve_gib,
        active_adapter_workspace_gib=active_adapter_workspace_gib,
        per_run_persistent_gib=per_run_persistent_gib,
        usable_per_run_gib=usable_per_run_gib,
        safety_cap=safety_cap,
        reason=reason,
    )


def _unsupported_estimate(
    spec: JobSpec,
    gpu_vram_gib: float,
    reason: str,
) -> BundleAdmissionEstimate:
    return BundleAdmissionEstimate(
        estimated_n=0,
        gpu_type=_gpu_type(spec),
        gpu_vram_gib=float(gpu_vram_gib),
        shared_base_gib=0.0,
        shared_runtime_reserve_gib=0.0,
        active_adapter_workspace_gib=0.0,
        per_run_persistent_gib=0.0,
        usable_per_run_gib=0.0,
        safety_cap=0,
        reason=reason,
    )


class SharedEngineBundle:
    """one packing window whose sealed members share one future allocation."""

    def __init__(
        self,
        bundle_id: str,
        seed_spec: JobSpec,
        *,
        packing_delay_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized_bundle_id = str(bundle_id).strip()
        if not normalized_bundle_id:
            raise ValueError("bundle_id must not be empty")
        if packing_delay_s < 0 or not math.isfinite(packing_delay_s):
            raise ValueError("packing_delay_s must be non-negative and finite")
        self._bundle_id = normalized_bundle_id
        self._compatibility_key = BundleCompatibilityKey.from_job_spec(seed_spec)
        self._admission_estimate = estimate_bundle_admission(seed_spec)
        self._clock = clock
        self._opened_at = float(clock())
        self._packing_deadline = self._opened_at + float(packing_delay_s)
        self._runs: dict[str, _BundleRunRecord] = {}
        self._sealed_members: tuple[str, ...] | None = None

    @property
    def bundle_id(self) -> str:
        return self._bundle_id

    @property
    def compatibility_key(self) -> BundleCompatibilityKey:
        return self._compatibility_key

    @property
    def admission_estimate(self) -> BundleAdmissionEstimate:
        return self._admission_estimate

    @property
    def estimated_n(self) -> int:
        return self._admission_estimate.estimated_n

    @property
    def packing_deadline(self) -> float:
        return self._packing_deadline

    @property
    def sealed(self) -> bool:
        return self._sealed_members is not None

    @property
    def member_run_ids(self) -> tuple[str, ...]:
        if self._sealed_members is not None:
            return self._sealed_members
        return tuple(self._runs)

    @property
    def full(self) -> bool:
        return self.estimated_n > 0 and len(self._runs) >= self.estimated_n

    def should_seal(self, now: float | None = None) -> bool:
        """return whether capacity or the bounded packing deadline was reached."""

        if self.sealed:
            return False
        current = self._clock() if now is None else float(now)
        return self.full or current >= self._packing_deadline

    def try_admit(self, spec: JobSpec) -> BundleAdmissionDecision:
        """admit one compatible run or surface why it must wait elsewhere."""

        run_id = str(spec.run_id).strip()
        if not run_id:
            return self._decision(
                BundleAdmissionOutcome.REJECTED,
                run_id,
                "run_id must not be empty",
            )
        if run_id in self._runs:
            return self._decision(
                BundleAdmissionOutcome.REJECTED,
                run_id,
                f"run {run_id!r} is already a member of bundle {self.bundle_id}",
            )
        try:
            candidate_key = BundleCompatibilityKey.from_job_spec(spec)
        except (TypeError, ValueError) as exc:
            return self._decision(BundleAdmissionOutcome.REJECTED, run_id, str(exc))
        incompatibility = self.compatibility_key.incompatibility_reason(candidate_key)
        if incompatibility is not None:
            return self._decision(BundleAdmissionOutcome.REJECTED, run_id, incompatibility)
        if self.admission_estimate.reason is not None or self.estimated_n < 1:
            return self._decision(
                BundleAdmissionOutcome.REJECTED,
                run_id,
                self.admission_estimate.reason or "bundle has no safe admission capacity",
            )
        if self.sealed:
            return self._decision(
                BundleAdmissionOutcome.QUEUED,
                run_id,
                f"bundle {self.bundle_id} is sealed; create a new compatible bundle",
            )
        if self.full:
            return self._decision(
                BundleAdmissionOutcome.QUEUED,
                run_id,
                f"bundle capacity reached: estimated_n={self.estimated_n}",
            )

        self._runs[run_id] = _BundleRunRecord(run_id=run_id, spec_json=spec.to_json())
        return self._decision(BundleAdmissionOutcome.ADMITTED, run_id, None)

    def seal(self) -> tuple[str, ...]:
        """freeze the member set for one future shared allocation."""

        if self._sealed_members is None:
            if not self._runs:
                raise ValueError("cannot seal an empty shared-engine bundle")
            self._sealed_members = tuple(self._runs)
        return self._sealed_members

    def run_snapshot(self, run_id: str) -> BundleRunSnapshot:
        return self._require_run(run_id).snapshot()

    def run_snapshots(self) -> tuple[BundleRunSnapshot, ...]:
        return tuple(record.snapshot() for record in self._runs.values())

    def transition_run(
        self,
        run_id: str,
        status: LogicalRunStatus,
        *,
        error: str | None = None,
    ) -> BundleRunSnapshot:
        """transition exactly one logical run without mutating sibling state."""

        record = self._require_run(run_id)
        target = LogicalRunStatus(status)
        if target not in _ALLOWED_RUN_TRANSITIONS[record.status]:
            raise ValueError(
                f"invalid bundle run transition for {record.run_id}: {record.status} -> {target}"
            )
        if target is LogicalRunStatus.FAILED:
            normalized_error = str(error or "").strip()
            if not normalized_error:
                raise ValueError("failed run status requires an error reason")
            record.error = normalized_error
        elif error is not None:
            raise ValueError("error is valid only when transitioning a run to failed")
        record.status = target
        return record.snapshot()

    def _decision(
        self,
        outcome: BundleAdmissionOutcome,
        run_id: str,
        reason: str | None,
    ) -> BundleAdmissionDecision:
        return BundleAdmissionDecision(
            outcome=outcome,
            run_id=run_id,
            bundle_id=self.bundle_id,
            estimated_n=self.estimated_n,
            reason=reason,
        )

    def _require_run(self, run_id: str) -> _BundleRunRecord:
        normalized_run_id = str(run_id).strip()
        try:
            return self._runs[normalized_run_id]
        except KeyError as exc:
            raise KeyError(f"unknown run {normalized_run_id!r} in bundle {self.bundle_id}") from exc


class SharedEngineBundlePacker:
    """group prepared runs into bounded, exact-compatibility packing windows."""

    def __init__(
        self,
        *,
        packing_delay_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if packing_delay_s < 0 or not math.isfinite(packing_delay_s):
            raise ValueError("packing_delay_s must be non-negative and finite")
        self._packing_delay_s = float(packing_delay_s)
        self._clock = clock
        self._bundles: list[SharedEngineBundle] = []
        self._bundle_counter = 0

    @property
    def bundles(self) -> tuple[SharedEngineBundle, ...]:
        return tuple(self._bundles)

    def offer(self, spec: JobSpec) -> BundleAdmissionDecision:
        """admit to an open compatible bundle or create the next packing window."""

        estimate = estimate_bundle_admission(spec)
        if estimate.estimated_n < 1:
            return BundleAdmissionDecision(
                outcome=BundleAdmissionOutcome.REJECTED,
                run_id=str(spec.run_id).strip(),
                bundle_id=None,
                estimated_n=0,
                reason=estimate.reason or "bundle has no safe admission capacity",
            )
        candidate_key = BundleCompatibilityKey.from_job_spec(spec)
        for bundle in self._bundles:
            if bundle.sealed or bundle.compatibility_key != candidate_key:
                continue
            decision = bundle.try_admit(spec)
            if decision.outcome is BundleAdmissionOutcome.ADMITTED:
                if bundle.full:
                    bundle.seal()
                return decision
            if decision.outcome is BundleAdmissionOutcome.REJECTED:
                return decision

        bundle = self._new_bundle(spec, candidate_key)
        decision = bundle.try_admit(spec)
        if decision.outcome is BundleAdmissionOutcome.ADMITTED:
            self._bundles.append(bundle)
            if bundle.full:
                bundle.seal()
        return decision

    def seal_ready(self, now: float | None = None) -> tuple[SharedEngineBundle, ...]:
        """seal every nonempty bundle whose capacity or deadline was reached."""

        sealed: list[SharedEngineBundle] = []
        for bundle in self._bundles:
            if bundle.should_seal(now):
                bundle.seal()
                sealed.append(bundle)
        return tuple(sealed)

    def seal_all(self) -> tuple[SharedEngineBundle, ...]:
        """seal all nonempty packing windows at a caller-controlled launch boundary."""

        sealed: list[SharedEngineBundle] = []
        for bundle in self._bundles:
            if not bundle.sealed and bundle.member_run_ids:
                bundle.seal()
                sealed.append(bundle)
        return tuple(sealed)

    def _new_bundle(
        self,
        spec: JobSpec,
        compatibility_key: BundleCompatibilityKey,
    ) -> SharedEngineBundle:
        self._bundle_counter += 1
        bundle_id = f"shared-{compatibility_key.digest[:12]}-{self._bundle_counter}"
        return SharedEngineBundle(
            bundle_id,
            spec,
            packing_delay_s=self._packing_delay_s,
            clock=self._clock,
        )
