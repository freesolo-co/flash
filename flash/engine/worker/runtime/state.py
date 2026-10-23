"""explicit per-attempt configuration and worker resource ownership."""

from __future__ import annotations

import math
import os
import threading
import time
from _thread import LockType
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from flash.core.spec import FIXED_SEED, JobSpec
from flash.engine.worker.runtime.phases import PhaseTimings
from flash.envs.loading.staged import (
    StagedEnvironmentMaterialization,
    load_staged_freesolo_environment,
)
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV

if TYPE_CHECKING:
    from flash.engine.worker.io.heartbeat import HeartbeatState


def _resolve_worker_seed(job_spec, env_seed: str | None) -> int:
    if job_spec is not None:
        return int(job_spec.seed)
    try:
        seed = int(env_seed) if env_seed is not None else FIXED_SEED
    except (TypeError, ValueError):
        return FIXED_SEED
    return seed if 0 <= seed <= 2**63 - 1 else FIXED_SEED


@dataclass(frozen=True)
class WorkerConfig:
    spec: JobSpec | None = None
    hf_repo: str = ""
    run_id: str = "local"
    run_mode: str = "sft"
    attempt: int = 0
    seed: int = FIXED_SEED
    phase: str = "sft"
    thinking: bool = False
    opd_resume_revision: str = ""
    deadline_at: float | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> WorkerConfig:
        env = dict(environ)
        raw = env.get("FLASH_JOB_SPEC_JSON")
        if not raw and env.get("FLASH_JOB_SPEC_PATH"):
            from pathlib import Path

            raw = Path(env["FLASH_JOB_SPEC_PATH"]).read_text()
        spec = JobSpec.from_json(raw) if raw else None
        attempt = env.get("ATTEMPT", "0")
        if not attempt or any(char < "0" or char > "9" for char in attempt):
            raise RuntimeError("managed worker ATTEMPT must be an unsigned decimal integer")
        raw_deadline = env.get("FLASH_RUN_DEADLINE_AT")
        try:
            deadline = float(raw_deadline) if raw_deadline is not None else None
            if deadline is not None and (deadline <= 0 or not math.isfinite(deadline)):
                raise ValueError
        except (ValueError, TypeError):
            raise RuntimeError("worker run wall deadline is invalid") from None
        mode = env.get("RUN_MODE", "sft")
        return cls(
            spec=spec,
            hf_repo=env.get("HF_REPO", ""),
            run_id=env.get("RUN_ID", "local"),
            run_mode=mode,
            attempt=int(attempt),
            seed=_resolve_worker_seed(spec, env.get("SEED")),
            phase=env.get("PHASE", spec.phase if spec else mode),
            thinking=bool(spec and spec.thinking),
            opd_resume_revision=env.get(OPD_RESUME_REVISION_ENV, "").strip(),
            deadline_at=deadline,
        )


@dataclass
class WorkerContext:
    config: WorkerConfig
    active_env: Any = None
    active_env_package: StagedEnvironmentMaterialization | None = None
    started_at: float = field(default_factory=time.time)
    phase_timings: PhaseTimings = field(default_factory=PhaseTimings)
    resume_upload_lock: LockType = field(default_factory=threading.Lock)
    heartbeat_state: HeartbeatState = field(init=False)

    def __post_init__(self):
        from flash.engine.worker.io.heartbeat import HeartbeatState

        self.heartbeat_state = HeartbeatState()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkerContext:
        return cls(WorkerConfig.from_env(os.environ if environ is None else environ))

    def remaining_wall_seconds(self) -> float | None:
        if self.config.deadline_at is None:
            return None
        return max(0.0, self.config.deadline_at - time.time())

    def require_active_env(self):
        if self.active_env is not None:
            return self.active_env
        spec = self.config.spec
        if spec is None or not spec.environment.id:
            raise RuntimeError("worker requires a JobSpec naming [environment] id")
        env, package = load_staged_freesolo_environment(
            spec.environment, spec.environment.params, hf_repo=spec.train.hf_repo
        )
        self.active_env_package = package
        from flash.content.multimodal import validate_image_observation_environment

        try:
            validate_image_observation_environment(env, spec)
            if hasattr(env, "thinking"):
                env.thinking = bool(spec.thinking)
        except BaseException:
            self.close()
            raise
        self.active_env = env
        return env

    def close(self) -> None:
        package, self.active_env_package = self.active_env_package, None
        self.active_env = None
        if package is not None:
            package.cleanup()
