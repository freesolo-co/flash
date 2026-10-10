"""run-scoped worker state and active environment ownership."""

from __future__ import annotations

import math
import os
import time

from flash.core.spec import FIXED_SEED, load_job_spec_from_env
from flash.envs.loading.staged import (
    StagedEnvironmentMaterialization,
    load_staged_freesolo_environment,
)
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV


def _resolve_worker_seed(job_spec, env_seed: str | None) -> int:
    if job_spec is not None:
        return int(job_spec.seed)
    try:
        seed = int(env_seed) if env_seed is not None else FIXED_SEED
    except (TypeError, ValueError):
        return FIXED_SEED
    return seed if 0 <= seed <= 2**63 - 1 else FIXED_SEED


def _parse_attempt_env() -> int:
    raw = os.environ.get("ATTEMPT")
    if raw is None:
        return 0
    if not raw or any(char < "0" or char > "9" for char in raw):
        raise RuntimeError("managed worker ATTEMPT must be an unsigned decimal integer")
    return int(raw)


HF_REPO = os.environ.get("HF_REPO", "")
RUN_ID = os.environ.get("RUN_ID", "local")
RUN_MODE = os.environ.get("RUN_MODE", "sft")
ATTEMPT = _parse_attempt_env()
# captured at import, which is before this attempt can have started ray. any ray session older than
# this belongs to a PREVIOUS attempt on the same reused pod (/tmp survives a retry), and reporting
# one as this attempt's evidence would send the next diagnosis after a failure that never happened.
WORKER_START_TIME = time.time()
JOB_SPEC = load_job_spec_from_env()
SEED = _resolve_worker_seed(JOB_SPEC, os.environ.get("SEED"))
PHASE = os.environ.get(
    "PHASE",
    JOB_SPEC.phase if JOB_SPEC else (RUN_MODE if RUN_MODE in ("sft", "rl", "opd") else "sft"),
)
OPD_RESUME_REVISION = os.environ.get(OPD_RESUME_REVISION_ENV, "").strip()
THINKING = JOB_SPEC.thinking if JOB_SPEC else False


def _remaining_worker_wall_seconds() -> float | None:
    raw_deadline = os.environ.get("FLASH_RUN_DEADLINE_AT")
    if raw_deadline is None:
        return None
    try:
        deadline = float(raw_deadline)
    except (TypeError, ValueError):
        raise RuntimeError("worker run wall deadline is invalid") from None
    now = time.time()
    if deadline <= 0 or now <= 0 or not math.isfinite(deadline) or not math.isfinite(now):
        raise RuntimeError("worker run wall deadline is invalid")
    return max(0.0, deadline - now)


def _load_active_env():
    """Load the run's Freesolo environment from the JobSpec; require an explicit env."""
    if JOB_SPEC is None:
        return None
    env_id = JOB_SPEC.environment.id
    if not env_id:
        raise RuntimeError(
            "JobSpec sets no environment: provide [environment] id "
            "(a Freesolo environment id like 'your-org/your-project/your-env', returned by "
            "`flash env push --project <project-uuid> --name <name>`)."
        )
    global ACTIVE_ENV_PACKAGE
    env, ACTIVE_ENV_PACKAGE = load_staged_freesolo_environment(
        JOB_SPEC.environment,
        JOB_SPEC.environment.params,
        hf_repo=JOB_SPEC.train.hf_repo,
    )
    from flash.content.multimodal import validate_image_observation_environment

    # this capability classifies a future dynamic-image environment only. full per-turn media wiring
    # remains separate; the guard makes an incompatible model or opd teacher fail before model work.
    validate_image_observation_environment(env, JOB_SPEC)
    # tell the env whether this run samples <think> blocks, so the multi-turn scoring path strips
    # reasoning exactly like the single-turn path does (see FreesoloEnvironment.record_model_turn).
    if hasattr(env, "thinking"):
        env.thinking = bool(JOB_SPEC.thinking)
    return env


ACTIVE_ENV = None
ACTIVE_ENV_PACKAGE: StagedEnvironmentMaterialization | None = None


def _cleanup_active_env_package() -> None:
    global ACTIVE_ENV_PACKAGE
    if ACTIVE_ENV_PACKAGE is None:
        return
    ACTIVE_ENV_PACKAGE.cleanup()
    ACTIVE_ENV_PACKAGE = None


def require_active_env():
    """Return the run's loaded environment, raising a clear error if none is loaded."""
    global ACTIVE_ENV
    if ACTIVE_ENV is None:
        ACTIVE_ENV = _load_active_env()
    if ACTIVE_ENV is None:
        raise RuntimeError(
            "no environment is loaded: this worker was started without a JobSpec "
            "(FLASH_JOB_SPEC_JSON / FLASH_JOB_SPEC_PATH is unset). A train/eval run must "
            "carry a JobSpec naming [environment] id "
            "(a Freesolo environment id like 'your-org/your-project/your-env', returned by "
            "`flash env push --project <project-uuid> --name <name>`)."
        )
    return ACTIVE_ENV
