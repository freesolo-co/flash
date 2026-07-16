"""Platform runner: drives managed RunPod GPUs, one allocation per run."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

try:
    import fcntl
except ImportError:  # pragma: no cover - linux production fails closed below
    fcntl = None

from flash.catalog import ModelInfo, resolve_model
from flash.opd_retry_contract import (
    OPD_RETRY_CONTRACT_STATUS_KEY,
    OPD_RETRY_CONTRACT_VERSION,
    require_opd_retry_contract_version,
)
from flash.providers._poll import _MAX_ATTEMPT_ID, _attempt_int
from flash.spec import JobSpec

_STATE_DIR = os.path.join(os.path.expanduser("~"), ".flash")
RUNS_DIR = os.path.join(_STATE_DIR, "runs")
RESULTS_DIR = os.path.join(_STATE_DIR, "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# `done` IS deployable, so excluded; cancelled/failed/dry_run must never flip to `deployed`.
_UNDEPLOYABLE_STATES = TERMINAL_STATES - {"done"}
# serialize local writers before taking each run's interprocess lock.
_STATUS_LOCK = threading.Lock()
_RUN_DEADLINE_AT_KEY = "run_deadline_at"
_NEXT_ATTEMPT_KEY = "next_attempt"
_CLEANUP_REMOTES_KEY = "cleanup_remotes"
_OPD_RETRY_CONTRACT_KEY = OPD_RETRY_CONTRACT_STATUS_KEY
_PRIVATE_STATUS_KEYS = frozenset(
    {
        _RUN_DEADLINE_AT_KEY,
        _NEXT_ATTEMPT_KEY,
        _CLEANUP_REMOTES_KEY,
        _OPD_RETRY_CONTRACT_KEY,
    }
)
_PRIVATE_VALUE_UNSET = object()
MIN_PROVIDER_WALL_SECONDS = 60


def artifacts_dir(spec: JobSpec) -> str:
    """Run-scoped artifact root: results/runpod/<phase>/<run_id>."""
    return os.path.join(RESULTS_DIR, "runpod", spec.phase, spec.run_id)


def adapter_prefix(spec: JobSpec) -> str:
    """A run's adapter location on the HF artifact store: ``<phase>/<run_id>``."""
    return f"{spec.phase}/{spec.run_id}"


def adapter_ref(spec: JobSpec) -> str | None:
    """INTERNAL storage reference for a run's trained adapter (artifact registration only)."""
    if not spec.train.hf_repo:
        return None
    return f"{spec.train.hf_repo}:{adapter_prefix(spec)}"


def _adapter_ref_from_status_spec(raw: dict) -> str | None:
    """The public short adapter reference (`<run_id>`) shown by `flash status` — exactly what
    users paste into train.init_from_adapter; `<run_id>/step-N` targets a saved checkpoint."""
    try:
        spec = JobSpec.from_dict(raw)
    except Exception:
        return None
    if not spec.train.hf_repo:
        return None
    return spec.run_id


def _gpu_rate(gpu_type: str) -> float:
    """Static representative $/hr for cost projection."""
    try:
        from flash.providers import get_provider

        return get_provider("runpod").hourly_rate(gpu_type)
    except Exception:
        return 0.80


def charge_usd_for_spec(spec, *, steps: int | None = None, fallback: float = 0.0) -> float:
    """Return the estimated customer charge, prorated by completed steps when requested."""
    try:
        from flash.cost.analytical import estimate_cost
        from flash.cost.spec import estimate_for_spec, runconfig_from_spec

        if steps is None:
            return float(estimate_for_spec(spec).total_usd)
        n = max(0, int(steps))
        if n == 0:
            return 0.0
        cfg = runconfig_from_spec(spec)
        planned = int(cfg.steps or 0)
        if planned > 0:
            n = min(n, planned)
        # a partial (cancelled) reprice only counts required saves that could already have landed by
        # the completed step; keeping a save beyond the reduced horizon would also trip the run
        # config's save_at_steps <= steps guard and drop the whole estimate to the fallback.
        reached_saves = tuple(s for s in cfg.save_at_steps if s <= n)
        if not cfg.is_grpo and cfg.train_tokens and planned > 0:
            scaled_tokens = max(1, int(cfg.train_tokens * n / planned))
            cfg = replace(cfg, steps=n, train_tokens=scaled_tokens, save_at_steps=reached_saves)
        else:
            cfg = replace(cfg, steps=n, save_at_steps=reached_saves)
        return float(estimate_cost(cfg).total_usd)
    except Exception:
        return float(fallback)


def _require_priced_sft_examples(spec: JobSpec) -> None:
    if spec.algorithm == "sft" and int(spec.train.max_examples or 0) <= 0:
        raise ValueError(
            "train.max_examples must be set to a positive row count for SFT "
            "(use the full dataset row count for an uncapped run)"
        )


def _status_estimated_charge(status: RunStatus, spec, *, fallback: float = 0.0) -> float:
    quote = getattr(status, "estimated_cost_usd", None)
    if quote is not None:
        return float(quote)
    return charge_usd_for_spec(spec, fallback=fallback)


# Heartbeat stages that mean the worker has entered training (GPU work underway). The per-step
# `step` field is 1-indexed and only appears once a step COMPLETES, so the expensive first step (a
# GRPO rollout can be ~17 min, an opd step waits on the teacher round-trips) streams one of these
# stages with NO step yet -- still real GPU time.
_TRAINING_STAGES = frozenset({"rl_step", "sft_step", "opd_step"})


def actual_steps_run(status: RunStatus) -> int:
    """How many optimizer steps to bill a (cancelled) run for.

    The worker streams a per-step heartbeat whose ``step`` field is the last COMPLETED optimizer step
    (1-indexed; the last one we persisted is the furthest it reached). Cancelled after N steps -> N.
    The first step reports no ``step`` until it completes, so a cancel mid-first-step would look like
    0 steps despite real GPU time -- we floor to 1 whenever a training-stage heartbeat is present.
    Returns 0 only when no training heartbeat was seen (cancelled during cold-start/setup) -> $0."""
    hb = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else {}
    step = hb.get("step")
    if isinstance(step, (int, float)) and step > 0:
        return int(step)
    # Training started (rl_step/sft_step/opd_step) but no completed step yet -> mid-first-step -> 1.
    if hb.get("stage") in _TRAINING_STAGES:
        return 1
    return 0


def _require_valid_deadline(value: object) -> float:
    """Return a finite positive unix deadline or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0:
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    return deadline


def _canonical_run_deadline(raw: dict) -> tuple[RunStatus, float]:
    status = _runstatus_from_json(raw)
    spec = JobSpec.from_dict(status.spec)
    created_at = _require_valid_deadline(status.created_at)
    max_wall_seconds = _require_valid_deadline(spec.gpu.max_wall_seconds)
    return status, _require_valid_deadline(created_at + max_wall_seconds)


def _checked_stored_run_deadline(stored: object, canonical: float) -> float:
    deadline = _require_valid_deadline(stored)
    if not math.isclose(deadline, canonical, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            "persisted run wall deadline does not match canonical submission deadline; "
            "no further provisioning is allowed"
        )
    return deadline


def _load_run_deadline_at(run_id: str) -> float:
    """Return the persisted canonical submission-to-terminal deadline."""
    raw = _load_status_json(run_id)
    _status, canonical = _canonical_run_deadline(raw)
    if _RUN_DEADLINE_AT_KEY not in raw:
        raise RuntimeError(
            "persisted run wall deadline is missing; no further provisioning is allowed"
        )
    return _checked_stored_run_deadline(raw[_RUN_DEADLINE_AT_KEY], canonical)


def _remaining_run_wall_seconds(run_id: str, *, now: float | None = None) -> float:
    """Return non-negative wall allowance remaining on the run-global deadline."""
    current = time.time() if now is None else now
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or not math.isfinite(current)
        or current <= 0
    ):
        raise ValueError("current clock is invalid")
    return max(0.0, _load_run_deadline_at(run_id) - float(current))


def _spec_with_remaining_wall(
    spec: JobSpec,
    *,
    require_provider_minimum: bool,
    now: float | None = None,
) -> JobSpec:
    """Copy a spec with only the run-global wall allowance still available."""
    remaining = _remaining_run_wall_seconds(spec.run_id, now=now)
    if remaining <= 0:
        raise RuntimeError("run wall deadline exhausted; no further provisioning is allowed")
    if require_provider_minimum and remaining < MIN_PROVIDER_WALL_SECONDS:
        raise RuntimeError(
            "run wall deadline has less than the 60-second minimum provider allowance remaining; "
            "no further provisioning is allowed"
        )
    allowance = max(1, int(remaining))
    return replace(spec, gpu=replace(spec.gpu, max_wall_seconds=allowance))


def _infer_next_attempt(raw: dict) -> int:
    if _NEXT_ATTEMPT_KEY not in raw:
        raise RuntimeError("stored next attempt identity is missing")
    stored = raw[_NEXT_ATTEMPT_KEY]
    if _attempt_int(stored) is None:
        raise RuntimeError("stored next attempt identity is invalid")
    return stored


def _verified_opd_retry_state(run_id: str) -> tuple[int, str | None]:
    """Verify one locked opd retry snapshot and return its attempt plus resume revision."""
    with _status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        spec = JobSpec.from_dict(status.spec)
        if spec.algorithm != "opd":
            raise RuntimeError("opd retry verification requires an opd run")
        try:
            contract_version = require_opd_retry_contract_version(
                raw.get(_OPD_RETRY_CONTRACT_KEY)
            )
        except ValueError as exc:
            raise RuntimeError("opd retry contract is missing or invalid; replacement is blocked") from exc
        next_attempt = _infer_next_attempt(raw)
        hf_repo = spec.train.hf_repo
        # phase is the hf-prefix component the worker uploads under ({phase}/{run_id}/...), so it locates
        # both the markers and any full-state resume checkpoint the replacement can continue from.
        phase = spec.phase
        seed = spec.seed
    from flash.providers._hf_artifacts import verify_opd_replacement_safe

    resume_revision = verify_opd_replacement_safe(
        hf_repo=hf_repo,
        run_id=run_id,
        seed=seed,
        next_attempt=next_attempt,
        contract_version=contract_version,
        phase=phase,
    )
    return next_attempt, resume_revision


def _verified_opd_next_attempt(run_id: str) -> int:
    """Return just the verified next attempt, discarding the resume revision."""
    return _verified_opd_retry_state(run_id)[0]


def _reserve_attempt(
    run_id: str,
    *,
    minimum_attempt: int = 0,
    expected_next_attempt: int | None = None,
) -> int:
    """Durably consume one run-global attempt identity before provider creation."""
    minimum = _attempt_int(minimum_attempt)
    if minimum is None:
        raise RuntimeError("minimum attempt identity is invalid")
    expected = None
    if expected_next_attempt is not None:
        expected = _attempt_int(expected_next_attempt)
        if expected is None:
            raise RuntimeError("expected next attempt identity is invalid")
    with _status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        current = _infer_next_attempt(raw)
        if expected is not None and current != expected:
            raise RuntimeError("stored next attempt identity changed after retry verification")
        spec = JobSpec.from_dict(status.spec)
        if spec.algorithm == "opd":
            try:
                require_opd_retry_contract_version(raw.get(_OPD_RETRY_CONTRACT_KEY))
            except ValueError as exc:
                raise RuntimeError(
                    "opd retry contract is missing or invalid; replacement is blocked"
                ) from exc
            if expected is None:
                raise RuntimeError("opd attempt reservation requires verified retry evidence")
            if minimum > expected:
                raise RuntimeError("minimum opd attempt exceeds the verified retry snapshot")
            attempt = expected
        else:
            attempt = max(current, minimum)
        if attempt >= _MAX_ATTEMPT_ID:
            raise RuntimeError("run attempt identity is exhausted")
        _save_status_unlocked(status, _next_attempt=attempt + 1)
        return attempt


@dataclass
class RunStatus:
    run_id: str
    state: str
    spec: dict
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    # Submit-time flash.cost quote. Successful runs copy this into cost_usd at completion so the
    # customer is charged exactly what was estimated before paid work started.
    estimated_cost_usd: float | None = None
    error: str | None = None
    artifacts_dir: str | None = None
    adapter_ref: str | None = None
    deployment: dict | None = None
    remote: dict | None = None
    # Instance providers (lambda/vast) configured WHEN THIS RUN WAS SUBMITTED — the set that could have
    # owned a pre-handle non-idempotent create. Recovery's phantom guard (_confirm_run_clear) fails
    # closed for any of these that is no longer configurable (so it can't ENUMERATE to prove clear),
    # scoped here so a plane that never configured Vast never blocks a handle-less recovery on it. None
    # for runs created outside submit() / pre-feature records.
    submitted_instance_providers: list[str] | None = None
    # Realized provider cost (COGS), pulled from the provider's billing API after the run
    # finishes by the reconciliation job (flash/server/reconcile.py) and reported to the
    # freesolo backend for estimator accuracy. Distinct from ``cost_usd`` (the flash.cost ESTIMATE
    # we charge the customer); ``reconciled_at`` marks that the realized pull has happened so it
    # isn't re-pulled. Both stay None for un-reconciled / pre-instrumentation runs.
    realized_cost_usd: float | None = None
    reconciled_at: float | None = None
    # Stamped ONCE on first terminal transition; survives later updated_at bumps from deploy/reconcile.
    finished_at: float | None = None
    billing_context: dict | None = None
    billing_state: str | None = None
    billing_error: str | None = None
    billing_charge: dict | None = None
    platform_context: dict | None = None
    last_heartbeat: dict | None = None
    gpu_status: dict | None = None
    effective_preparation: dict | None = None

    def to_dict(self) -> dict:
        """Return the public run status representation."""
        from flash.serve.urls import public_deployment

        data = _status_storage_dict(self)
        data["spec"] = _public_status_spec(data.get("spec"))
        # internal warm-start preparation (storage locators, digests) never leaves the server
        data.pop("effective_preparation", None)
        if isinstance(self.deployment, dict):
            data["deployment"] = public_deployment(self.deployment)
        return data


def _public_status_spec(raw):
    """Canonicalize valid specs and safely redact malformed legacy shapes."""
    if not isinstance(raw, dict):
        return raw
    try:
        data = JobSpec.from_dict(raw).to_dict()
    except Exception:
        data = dict(raw)
        train = data.get("train")
        if isinstance(train, dict):
            train = dict(train)
            train.pop("init_from_adapter_revision", None)
            init_ref = train.get("init_from_adapter")
            if init_ref is not None and (not isinstance(init_ref, str) or init_ref.strip()):
                train.pop("lora_rank", None)
            data["train"] = train
    _redact_internal_adapter_ref(data)
    return data


def _redact_internal_adapter_ref(data: dict) -> None:
    """Never surface an internal storage locator in the public spec.

    A worker/effective or legacy record can persist ``train.init_from_adapter`` as the internal
    storage ref ``<hf_repo>:<phase>/<run_id>[/checkpoints/step-N]``, which embeds the private HF
    repo. Rewrite it back to the user-facing checkpoint ref (``<run_id>[/step-N]``); a public ref
    (``parse_adapter_storage_ref`` returns ``None``) is left untouched.
    """
    train = data.get("train")
    if not isinstance(train, dict):
        return
    ref = train.get("init_from_adapter")
    if not isinstance(ref, str) or not ref.strip():
        return
    from flash.schema import format_checkpoint_ref, parse_adapter_storage_ref

    resolved = parse_adapter_storage_ref(ref)
    if resolved is None:
        return  # already a user-facing ref, not an internal storage locator
    _repo, prefix = resolved
    match = re.fullmatch(
        r"(?:sft|rl|opd)/(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,127})"
        r"(?:/checkpoints/step-(?P<step>\d+))?",
        prefix,
    )
    if match is None:
        # Parseable storage ref with an unexpected prefix shape; drop rather than leak the repo.
        train.pop("init_from_adapter", None)
        return
    step = match.group("step")
    train["init_from_adapter"] = format_checkpoint_ref(
        match.group("run"), int(step) if step is not None else None
    )


def _status_storage_dict(status: RunStatus) -> dict:
    """Serialize status for persistence without filtering internal deployment state."""
    data = asdict(status)
    data["adapter_ref"] = (
        _adapter_ref_from_status_spec(status.spec) if status.state in {"done", "deployed"} else None
    )
    return data


class _RunCancelled(RuntimeError):
    """User cancellation observed mid-run; terminal, never retried/overwritten."""


class _TerminalHandleRace(_RunCancelled):
    """A provider handle was created after the run became terminal."""


def new_run_id() -> str:
    return f"flash-{int(time.time())}-{uuid.uuid4().hex[:8]}"


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def require_safe_run_id(run_id: str) -> str:
    """Reject run ids that could traverse outside the runs directory."""
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def runs_file_path(run_id: str, suffix: str) -> str:
    """Containment-checked path for a run's file under RUNS_DIR."""
    base = os.path.abspath(RUNS_DIR)
    path = os.path.normpath(os.path.join(base, f"{require_safe_run_id(run_id)}{suffix}"))
    if not path.startswith(base + os.sep):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return path


@contextlib.contextmanager
def _status_guard(run_id: str):
    """Serialize one run's status mutations across threads and Linux processes."""
    if fcntl is None:
        raise RuntimeError("interprocess run-status locking is unavailable")
    os.makedirs(RUNS_DIR, exist_ok=True)
    lock_path = runs_file_path(run_id, ".lock")
    with _STATUS_LOCK:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def _with_model_disk(spec: JobSpec, info: ModelInfo) -> dict:
    """Spec dict with gpu.disk_gb raised to the model's catalog min_disk_gb."""
    d = spec.to_internal_dict()
    need = int(getattr(info, "min_disk_gb", 0) or 0)
    if need > int(d["gpu"].get("disk_gb") or 0):
        d["gpu"] = {**d["gpu"], "disk_gb": need}
    return d


_ARTIFACT_NAMESPACE = "Freesolo-Co"
_ARTIFACT_REPO_PREFIX = "flashrun-"
_ARTIFACT_REPO_NAME_MAX = 96


def _environment_artifact_repo_name(env_id: str) -> str:
    """Stable HF dataset repo name for all runs of one environment."""
    raw = (env_id or "default-environment").strip() or "default-environment"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw.lower()).strip("-") or "environment"
    budget = _ARTIFACT_REPO_NAME_MAX - len(_ARTIFACT_REPO_PREFIX) - len(digest) - 1
    slug = slug[:budget].rstrip("-") or "environment"
    return f"{_ARTIFACT_REPO_PREFIX}{slug}-{digest}"


def managed_hf_repo_for_environment(env_id: str) -> str:
    """Private HF dataset repo shared by runs that use the same environment id."""
    return f"{_ARTIFACT_NAMESPACE}/{_environment_artifact_repo_name(env_id)}"


def _file_digest(path: str, digest) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def flash_code_prefix() -> str:
    """Content-addressed HF path for the current ``flash`` package snapshot."""
    import flash

    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    digest = hashlib.sha1()
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, pkg_dir).replace(os.sep, "/")
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            _file_digest(path, digest)
            digest.update(b"\0")
    return f"code/{digest.hexdigest()[:32]}/flash"


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the environment-scoped HF artifact repo (platform-managed, never user-set)."""
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the artifact repo")
    repo = managed_hf_repo_for_environment(spec.environment.id)
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return JobSpec.from_dict(d)


def _assign_resolved_env_sha(spec: JobSpec) -> JobSpec:
    """Pin env ref->SHA once so N workers don't fan-out N GitHub API calls (secondary rate-limit). Best-effort."""
    import logging

    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return spec
    try:
        from flash.envs.loader import (
            _parse_github_environment_ref,
            _resolve_ref_sha,
            is_managed_environment_slug,
            managed_slug_to_github_ref,
        )

        ref_str = (
            managed_slug_to_github_ref(env_id) if is_managed_environment_slug(env_id) else env_id
        )
        parsed = _parse_github_environment_ref(ref_str)
        if parsed is None:
            return spec  # local/path or non-GitHub env: nothing to pin
        sha = _resolve_ref_sha(parsed, timeout=10.0, max_rate_limit_retries=0)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "resolve-once: could not pin env ref->sha for %r (%s); worker will resolve", env_id, e
        )
        return spec
    if not sha:
        return spec
    d = spec.to_internal_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return JobSpec.from_dict(d)


WEIGHT_CACHE_VOLUME_NAME = "flash-weights"
WEIGHT_CACHE_VOLUME_GB = 100
# Peak footprint ~= 2x bf16 download (checkpoint + Xet temp); must fit the fixed cache volume.
_WEIGHT_CACHE_PEAK_FACTOR = 2.0


def _fits_weight_cache(info: ModelInfo) -> bool:
    """Whether the model's peak download footprint fits the shared weight-cache volume."""
    if not info.params_b:
        return (
            True  # unknown size -> keep the (attach) default; curated catalog models always set it
        )
    download_gb = info.params_b * 2.0  # bf16: 2 bytes/param (mirrors cost.facts.download_weight_gb)
    return _WEIGHT_CACHE_PEAK_FACTOR * download_gb <= WEIGHT_CACHE_VOLUME_GB


def _assign_weight_cache_volume(spec: JobSpec, info: ModelInfo | None = None) -> JobSpec:
    """Attach the shared weight-cache volume for PUBLIC catalog models only.

    Open-model ("allow") runs are never given the shared cache — private weights must not reach the
    shared cross-tenant mount. A pre-set non-shared volume is always honored as-is.
    """
    is_catalog = getattr(spec, "model_policy", "catalog") == "catalog"
    existing = getattr(spec.gpu, "network_volume", None)
    if existing and existing != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    attach = is_catalog and (info is None or _fits_weight_cache(info))
    if attach == (existing == WEIGHT_CACHE_VOLUME_NAME):
        return spec
    d = spec.to_internal_dict()
    if attach:
        d["gpu"] = {
            **d["gpu"],
            "network_volume": WEIGHT_CACHE_VOLUME_NAME,
            "network_volume_gb": WEIGHT_CACHE_VOLUME_GB,
        }
    else:
        d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)


def _run_job_background(
    spec: JobSpec,
    runtime_secrets: dict[str, str] | None = None,
    *,
    resolve_env_sha: bool = False,
) -> None:
    """Daemon-thread entrypoint: swallows exceptions to suppress noisy thread tracebacks."""
    import logging

    try:
        if resolve_env_sha:
            with contextlib.suppress(Exception):
                spec = _assign_resolved_env_sha(spec)
        if runtime_secrets:
            _run_job(spec, runtime_secrets=runtime_secrets)
        else:
            _run_job(spec)
    except Exception as exc:
        detail = f"{type(exc).__name__}: background run failed"
        with contextlib.suppress(Exception):
            if get_status(spec.run_id).state not in TERMINAL_STATES:
                _update(spec.run_id, "failed", error=detail)
        logging.getLogger(__name__).warning(
            "background run %s ended in error: %s", spec.run_id, detail
        )


def _context_org_id(context: dict | None) -> str:
    if not isinstance(context, dict):
        return ""
    return str(context.get("org_id") or "").strip()


def _status_org_id(status: RunStatus) -> str:
    return _context_org_id(status.billing_context) or _context_org_id(status.platform_context)


def _source_owned_by_key(src_run_id: str, owner_key_id: int | None) -> bool:
    if owner_key_id is None:
        return False
    try:
        from flash.server import db

        return db.run_owner(src_run_id) == owner_key_id
    except Exception:
        return False


def _require_supported_adapter_continuation(spec: JobSpec) -> None:
    if spec.algorithm == "sft" and spec.train.init_from_adapter:
        raise ValueError(
            "train.init_from_adapter is supported only for GRPO and OPD continue-in-place runs; "
            "SFT adapter continuation is not supported"
        )


def _prepare_init_from_adapter(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None]:
    """prepare public and worker specs with source-authoritative adapter metadata."""
    _require_supported_adapter_continuation(spec)
    ref = spec.train.init_from_adapter
    if not ref:
        return spec, spec, None
    from flash.lora_rank import (
        adapter_artifact_identity,
        load_hf_adapter_config,
        preflight_init_adapter_lora_rank,
        resolve_hf_dataset_revision,
    )
    from flash.runner.checkpoints import CheckpointListingError, adapter_artifact_exists
    from flash.schema import checkpoint_storage_ref, parse_checkpoint_ref

    parsed = parse_checkpoint_ref(ref)
    if parsed is None:
        raise ValueError(
            "train.init_from_adapter must be `<run_id>` or `<run_id>/step-N` "
            f"(a checkpoint listed by `flash checkpoints`); got {ref!r}"
        )
    src_run_id, step = parsed
    try:
        src_status = get_status(src_run_id)
    except FileNotFoundError:
        raise ValueError(f"train.init_from_adapter references unknown run {src_run_id!r}") from None
    owner_org_id = owner_org_id.strip()
    if owner_org_id:
        src_org_id = _status_org_id(src_status)
        owner_ok = (
            src_org_id == owner_org_id
            if src_org_id
            else _source_owned_by_key(src_run_id, owner_key_id)
        )
        if not owner_ok:
            raise ValueError(
                "train.init_from_adapter source run must belong to the same Freesolo org"
            )
    src_spec = JobSpec.from_dict(src_status.spec)
    if src_spec.model != spec.model:
        raise ValueError(
            f"train.init_from_adapter source model {src_spec.model!r} does not match target model "
            f"{spec.model!r}"
        )
    if src_spec.model_revision != spec.model_revision:
        raise ValueError(
            "train.init_from_adapter source model_revision "
            f"{src_spec.model_revision!r} does not match target model_revision "
            f"{spec.model_revision!r}"
        )
    if not src_spec.train.hf_repo:
        raise ValueError(
            f"train.init_from_adapter run {src_run_id!r} has no stored adapter artifacts"
        )
    if step is None and src_status.state not in {"done", "deployed"}:
        raise ValueError(
            f"train.init_from_adapter references run {src_run_id!r}, but that run is "
            f"{src_status.state!r}; use a completed source run or a concrete "
            f"{src_run_id}/step-N checkpoint"
        )
    storage = checkpoint_storage_ref(src_spec.train.hf_repo, src_spec.phase, src_run_id, step)
    revision = resolve_hf_dataset_revision(src_spec.train.hf_repo, token)
    try:
        exists = adapter_artifact_exists(src_spec, step=step, revision=revision)
    except CheckpointListingError as exc:
        raise ValueError(str(exc)) from exc
    if not exists:
        target = f"{src_run_id}/step-{step}" if step is not None else src_run_id
        raise ValueError(
            f"train.init_from_adapter references {target!r}, but its complete adapter artifact "
            "was not found"
        )
    worker_spec = replace(
        spec,
        train=replace(
            spec.train,
            init_from_adapter=storage,
            init_from_adapter_revision=revision,
        ),
    )
    config = load_hf_adapter_config(storage, token, revision)
    metadata = preflight_init_adapter_lora_rank(
        worker_spec, token=token, config_loader=lambda _ref, _token, _revision: config
    )
    assert metadata is not None
    identity = adapter_artifact_identity(storage, config, token, revision).to_dict()
    public_spec = replace(spec, train=replace(spec.train, lora_alpha=metadata.alpha))
    worker_spec = replace(
        worker_spec,
        train=replace(worker_spec.train, lora_rank=metadata.rank, lora_alpha=metadata.alpha),
    )
    return public_spec, worker_spec, identity


def _resolve_init_from_adapter(
    spec: JobSpec, *, owner_org_id: str = "", owner_key_id: int | None = None
) -> JobSpec:
    return _prepare_init_from_adapter(
        spec,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
        token=os.environ.get("HF_TOKEN"),
    )[1]


def _mark_warmstart_source(worker_spec: JobSpec, child_run_id: str) -> None:
    """Drop a 0-byte ``referenced_by/<child_run_id>`` marker into the warm-start SOURCE run's HF repo.

    The always-on artifact GC (``flash.server.repo_cleanup``) treats a source repo carrying a RECENT
    such marker as still-referenced and spares its artifacts for the GC age window — so a child that
    warm-starts (``init_from_adapter``) off an aged, undeployed source is not reaped out from under it.
    ``worker_spec`` is post-resolution, so its ``init_from_adapter`` is the internal
    ``<repo>:<phase>/<run_id>...`` storage ref whose repo is the source. Best-effort: a failed marker
    never blocks submission — it only forfeits the GC grace (the source can still be spared by being
    deployed or recently written). Emitted on submit AND re-emitted on recovery (``_runtime``), so a
    child recovered across restarts keeps its source's marker fresh past the age window."""
    import io

    ref = worker_spec.train.init_from_adapter
    if not ref or ":" not in ref or not child_run_id or child_run_id == "local":
        return
    source_repo = ref.split(":", 1)[0].strip()
    if not source_repo:
        return
    with contextlib.suppress(Exception):
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=io.BytesIO(b""),
            path_in_repo=f"referenced_by/{child_run_id}",
            repo_id=source_repo,
            repo_type="dataset",
        )


def _preparation_digest(
    public_spec: JobSpec, worker_spec: JobSpec, adapter_identity: dict | None
) -> str:
    payload = {
        "version": 1,
        "public_spec": public_spec.to_dict(),
        "worker_spec": worker_spec.to_internal_dict(),
        "adapter_identity": adapter_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_effective_spec(public_spec: JobSpec, worker_spec: JobSpec) -> None:
    public = public_spec.to_internal_dict()
    effective = worker_spec.to_internal_dict()
    public_train = dict(public["train"])
    effective_train = dict(effective["train"])
    public_ref = public_train.get("init_from_adapter") or ""
    internal_ref = effective_train.get("init_from_adapter") or ""
    for train_field in (
        "init_from_adapter",
        "init_from_adapter_revision",
        "lora_rank",
        "lora_alpha",
    ):
        effective_train[train_field] = public_train.get(train_field)
    effective["train"] = effective_train
    public_environment = dict(public["environment"])
    effective_environment = dict(effective["environment"])
    public_sha = public_environment.get("resolved_sha")
    effective_sha = effective_environment.get("resolved_sha")
    if not public_sha and isinstance(effective_sha, str):
        from flash.envs.loader import _is_commit_sha

        if _is_commit_sha(effective_sha):
            effective_environment["resolved_sha"] = ""
    effective["environment"] = effective_environment
    public_gpu = dict(public["gpu"])
    effective_gpu = {**effective["gpu"], "type": public_gpu["type"]}
    if (
        public_gpu.get("network_volume") == WEIGHT_CACHE_VOLUME_NAME
        and effective_gpu.get("network_volume") is None
    ):
        effective_gpu["network_volume"] = WEIGHT_CACHE_VOLUME_NAME
    effective["gpu"] = effective_gpu
    if effective != public:
        raise ValueError("persisted effective preparation does not match the public run")
    if not public_ref:
        if internal_ref or worker_spec.train.init_from_adapter_revision:
            raise ValueError("persisted effective preparation has an unexpected source adapter")
        return
    from flash.schema import parse_adapter_storage_ref, parse_checkpoint_ref

    public_target = parse_checkpoint_ref(public_ref)
    resolved = parse_adapter_storage_ref(internal_ref)
    if public_target is None or resolved is None:
        raise ValueError("persisted effective preparation has an invalid source adapter")
    _repo, prefix = resolved
    match = re.fullmatch(
        r"(?:sft|rl|opd)/(?P<run>[A-Za-z0-9][A-Za-z0-9._-]{0,127})"
        r"(?:/checkpoints/step-(?P<step>\d+))?",
        prefix,
    )
    if match is None:
        raise ValueError("persisted effective preparation has an invalid source adapter")
    source_run, source_step = public_target
    internal_step = int(match.group("step")) if match.group("step") is not None else None
    if match.group("run") != source_run or internal_step != source_step:
        raise ValueError("persisted effective preparation source does not match the public run")
    if not worker_spec.train.init_from_adapter_revision:
        raise ValueError("persisted effective preparation has no pinned source revision")


def _resolve_model_revision(spec: JobSpec) -> JobSpec:
    authored = spec.model_revision
    if not authored:
        return spec
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            spec.model,
            revision=authored,
        )
        resolved = str(getattr(info, "sha", "") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
            raise ValueError("resolved revision is not an immutable commit")
    except Exception as exc:
        raise ValueError(
            f"could not resolve model_revision for model {spec.model!r}; "
            "verify that the revision exists and the operator token can access it"
        ) from exc
    return replace(spec, model_revision=resolved)


@dataclass(frozen=True)
class PreparedJob:
    public_spec: JobSpec
    worker_spec: JobSpec
    estimated_cost_usd: float
    adapter_identity: dict | None = None


def prepare_job(
    spec: JobSpec,
    *,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
) -> PreparedJob:
    """Prepare all read-only submission inputs before persistence or allocation."""
    spec = _resolve_model_revision(spec)
    _require_priced_sft_examples(spec)
    _require_supported_adapter_continuation(spec)
    if spec.gpu.provider or spec.gpu.exact_type:
        from flash.providers import PROVIDER_NAMES, available_providers
        from flash.providers.base import providers_for

        configured = available_providers()
        provider = spec.gpu.provider.strip().lower()
        if provider:
            if provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {spec.gpu.provider!r}")
            if provider not in configured:
                raise ValueError(f"requested gpu.provider {provider!r} is not configured")
        elif not any(name in configured for name in providers_for(spec.gpu.exact_type)):
            raise ValueError(
                f"no configured provider can provision gpu.exact_type {spec.gpu.exact_type!r}"
            )
    info = resolve_model(
        spec.model,
        spec.algorithm,
        policy=spec.model_policy,
        gpu=spec.gpu.exact_type or spec.gpu.type,
        model_revision=spec.model_revision,
    )
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else new_run_id()
    spec = JobSpec.from_dict({**_with_model_disk(spec, info), "run_id": run_id})
    spec = _assign_managed_hf_repo(spec)
    spec = _assign_weight_cache_volume(spec, info)
    owner_org_id = _context_org_id(billing_context) or _context_org_id(platform_context)
    public_spec, worker_spec, adapter_identity = _prepare_init_from_adapter(
        spec,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
        token=os.environ.get("HF_TOKEN"),
    )
    from flash.cost.spec import estimate_for_spec
    from flash.lora_rank import preflight_train_context_within_serving

    preflight_train_context_within_serving(worker_spec)
    estimated_cost_usd = float(estimate_for_spec(worker_spec).total_usd)
    return PreparedJob(
        public_spec=public_spec,
        worker_spec=worker_spec,
        estimated_cost_usd=estimated_cost_usd,
        adapter_identity=adapter_identity,
    )


def _persist_effective_worker_spec(worker_spec: JobSpec) -> bool:
    """Persist the selected worker spec before provider provisioning starts."""
    status = get_status(worker_spec.run_id)
    if status.state in TERMINAL_STATES:
        return False
    snapshot = status.effective_preparation
    public_spec = JobSpec.from_dict(status.spec)
    if public_spec.train.init_from_adapter:
        if not isinstance(snapshot, dict):
            raise ValueError("persisted effective preparation is malformed")
        effective_spec_from_status(status)
        adapter_identity = snapshot.get("adapter_identity")
    else:
        adapter_identity = None
    _validate_effective_spec(public_spec, worker_spec)
    effective_preparation = {
        "worker_spec": worker_spec.to_internal_dict(),
        "adapter_identity": adapter_identity,
        "preparation_digest": _preparation_digest(public_spec, worker_spec, adapter_identity),
    }
    return _update(
        worker_spec.run_id,
        status.state,
        effective_preparation=effective_preparation,
    )


def submit_job(
    spec: JobSpec,
    dry_run: bool = False,
    background: bool = False,
    runtime_secrets: dict[str, str] | None = None,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
    prepared_job: PreparedJob | None = None,
) -> RunStatus:
    """Submit a prepared job, allocating resources only outside dry-run mode."""
    prepared = prepared_job or prepare_job(
        spec,
        billing_context=billing_context,
        platform_context=platform_context,
        owner_key_id=owner_key_id,
    )
    public_spec = prepared.public_spec
    worker_spec = prepared.worker_spec
    estimated_cost_usd = prepared.estimated_cost_usd
    from flash.providers import INSTANCE_PROVIDERS, available_providers

    if not dry_run:
        # Record the warm-start dependency on the SOURCE repo so the artifact GC spares it while this
        # child is around (best-effort; never blocks submission). A dry-run preview must not mutate
        # the source repo, so this HF write stays real-submit-only (unlike the read-only preflights
        # above, which now run in both modes).
        _mark_warmstart_source(worker_spec, public_spec.run_id)
    # env ref->sha pin is deferred (background) or after status save (sync) — never on creation path.
    status = RunStatus(
        run_id=public_spec.run_id,
        state="queued",
        spec=public_spec.to_dict(),
        estimated_cost_usd=estimated_cost_usd,
        billing_context=billing_context,
        billing_state="pending" if billing_context else None,
        platform_context=platform_context,
        effective_preparation={
            "worker_spec": worker_spec.to_internal_dict(),
            "adapter_identity": prepared.adapter_identity,
            "preparation_digest": _preparation_digest(
                public_spec, worker_spec, prepared.adapter_identity
            ),
        },
        # Snapshot the instance providers available at submit so a later handle-less recovery can fail
        # closed for any phantom-capable one whose creds were since dropped (see _confirm_run_clear).
        # Creds-only check (available_providers -> is_configured), no network on the create path.
        submitted_instance_providers=[n for n in available_providers() if n in INSTANCE_PROVIDERS],
    )
    _save_status(
        status,
        _run_deadline_at=status.created_at + float(public_spec.gpu.max_wall_seconds),
        _next_attempt=0,
        _opd_retry_contract_version=(
            OPD_RETRY_CONTRACT_VERSION
            if public_spec.algorithm == "opd"
            else _PRIVATE_VALUE_UNSET
        ),
    )
    _report_status(status)
    if dry_run:
        # A dry-run persists a state=dry_run record (retrievable, listable, and stageable for a
        # deploy dry-run) — same contract as a real submit minus GPU allocation, provisioning, and
        # billing. Everything above already validated the spec; just flip the state and return.
        status.state = "dry_run"
        _save_status(status)
        _report_status(status)
        return status
    if background:
        threading.Thread(
            target=_run_job_background,
            args=(worker_spec, runtime_secrets or {}),
            kwargs={"resolve_env_sha": True},
            daemon=True,
        ).start()
        return get_status(public_spec.run_id)
    worker_spec = _assign_resolved_env_sha(worker_spec)
    if runtime_secrets:
        _run_job(worker_spec, runtime_secrets=runtime_secrets)
    else:
        _run_job(worker_spec)
    return get_status(public_spec.run_id)


def _runstatus_from_json(d: dict) -> RunStatus:
    # Tolerant load: drop unknown keys before constructing RunStatus. A status JSON written by an
    # OLDER control plane can carry a since-removed field (e.g. ``resume_seed_index`` from the
    # pre-#317 multi-seed era) -- and `~/.flash/runs/*.json` is never GC'd, so those files exist in
    # prod RIGHT NOW. A strict ``RunStatus(**d)`` raises TypeError on such a key; the read sites
    # (get_status callers, recover/reconcile) catch only FileNotFoundError, so it would escape and
    # 500 runs-list / poll / recover / reconcile. This is operational tolerance for data already on
    # disk, NOT feature back-compat -- the removed field itself stays gone (it's simply ignored).
    return RunStatus(**{k: v for k, v in d.items() if k in RunStatus.__dataclass_fields__})


def _load_status_json(run_id: str) -> dict:
    path = runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"invalid stored run status for {run_id}")
    return value


def get_status(run_id: str) -> RunStatus:
    return _runstatus_from_json(_load_status_json(run_id))


def effective_spec_from_status(status: RunStatus, *, verify_source: bool = False) -> JobSpec:
    """Load the private prepared worker spec, optionally revalidating its source artifact."""
    public_spec = JobSpec.from_dict(status.spec)
    snapshot = status.effective_preparation
    if not isinstance(snapshot, dict):
        if public_spec.train.init_from_adapter:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} cannot be recovered "
                "because its original preparation snapshot is unavailable"
            )
        return public_spec
    raw_worker = snapshot.get("worker_spec")
    if not isinstance(raw_worker, dict):
        raise ValueError("persisted effective preparation is malformed")
    worker_spec = JobSpec.from_dict(raw_worker)
    _validate_effective_spec(public_spec, worker_spec)
    expected = snapshot.get("adapter_identity")
    stored_digest = snapshot.get("preparation_digest")
    if public_spec.train.init_from_adapter:
        if not isinstance(expected, dict) or not expected.get("digest"):
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} cannot be recovered "
                "because its original artifact identity is unavailable"
            )
        if not isinstance(stored_digest, str) or stored_digest != _preparation_digest(
            public_spec, worker_spec, expected
        ):
            raise ValueError("persisted effective preparation failed integrity validation")
    if verify_source and public_spec.train.init_from_adapter:
        try:
            from flash.lora_rank import (
                adapter_artifact_identity,
                inspect_adapter_config,
                load_hf_adapter_config,
            )

            revision = worker_spec.train.init_from_adapter_revision
            config = load_hf_adapter_config(
                worker_spec.train.init_from_adapter,
                os.environ.get("HF_TOKEN"),
                revision,
            )
            metadata = inspect_adapter_config(
                config,
                source="pinned warm-start adapter",
                target_model=worker_spec.model,
            )
            if (
                metadata.rank != worker_spec.train.lora_rank
                or metadata.alpha != worker_spec.train.lora_alpha
            ):
                raise ValueError("prepared adapter topology changed")
            current = adapter_artifact_identity(
                worker_spec.train.init_from_adapter,
                config,
                os.environ.get("HF_TOKEN"),
                revision,
            ).to_dict()
        except Exception as exc:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} could not be revalidated"
            ) from exc
        if current != expected:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} changed after submission; "
                "recovery was refused"
            )
    return worker_spec


def list_runs() -> list[RunStatus]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(RUNS_DIR, name)) as f:
                runs.append(_runstatus_from_json(json.load(f)))
    return runs


def list_run_ids() -> list[str]:
    """Run ids by filename only (no JSON parse) so a corrupt record can't break the listing."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    return [
        name[: -len(".json")] for name in sorted(os.listdir(RUNS_DIR)) if name.endswith(".json")
    ]


def get_logs(run_id: str) -> str:
    log_path = runs_file_path(run_id, ".log")
    if not os.path.exists(log_path):
        return ""
    with open(log_path) as f:
        return f.read()


def _sanitize_status_value(value, *, depth: int = 0):
    """Bound a heartbeat payload before persisting it in run status JSON."""
    if depth > 5:
        return str(value)[:200]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        return [_sanitize_status_value(v, depth=depth + 1) for v in value[:16]]
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 64:
                out["truncated"] = True
                break
            out[str(k)[:120]] = _sanitize_status_value(v, depth=depth + 1)
        return out
    return str(value)[:500]


def record_heartbeat(run_id: str, heartbeat: dict) -> None:
    """Persist the latest worker heartbeat/GPU snapshot without changing run state."""
    if not run_id or not isinstance(heartbeat, dict):
        return
    if not os.path.exists(runs_file_path(run_id, ".json")):
        return
    hb = _sanitize_status_value(heartbeat)
    gpu = (hb.get("gpu") or hb.get("diag")) if isinstance(hb, dict) else None
    with _status_guard(run_id):
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        status.last_heartbeat = hb
        status.gpu_status = gpu if isinstance(gpu, dict) else None
        status.updated_at = time.time()
        _save_status_unlocked(status)
    _report_status(status)


def _persist_metrics(spec: JobSpec, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id> and return the customer training cost.

    The run id keeps concurrent/sequential runs of the same phase from
    overwriting each other's artifacts. ``metrics["wall_seconds"]`` is the worker's training-loop
    wall time; setup/cold-start is reported separately and is not included here."""
    from flash.engine.accounting import sanitize_worker_metrics

    metrics = sanitize_worker_metrics(metrics)
    dest = artifacts_dir(spec)
    os.makedirs(dest, exist_ok=True)
    # Use allocated_gpu (worker-stamped) not spec.gpu.type; policy GPUs can be reallocated.
    gpu_type = metrics.get("allocated_gpu") or spec.gpu.type
    rate = _gpu_rate(gpu_type)
    cost = metrics.get("cost_usd")
    if cost:
        cost = float(cost or 0.0)
    else:
        wall = float(metrics.get("wall_seconds") or 0.0)
        cost = wall / 3600.0 * rate
        metrics = {**metrics, "cost_usd": cost}
        metrics.setdefault("notes", {})
        if isinstance(metrics["notes"], dict):
            metrics["notes"]["provider"] = "runpod"
            metrics["notes"]["runpod_rate_usd_hr"] = rate
            metrics["notes"]["runpod_gpu"] = gpu_type
    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with contextlib.suppress(Exception):
        from flash.server.run_registry import record_training_checkpoint

        record_training_checkpoint(spec=spec, metrics=metrics, artifact_path=dest)
    return float(cost)


def _remote_resource_identity(remote: object) -> tuple | None:
    """Return the exact strict provider resource identity used for compare-and-clear."""
    if not isinstance(remote, dict):
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

            handle = RunpodJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.endpoint_id,
                handle.job_id,
                handle.key_fingerprint,
            )
        if provider == "lambda":
            from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

            handle = LambdaJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.instance_id,
                handle.instance_type,
                handle.region,
                handle.name,
            )
        if provider == "vast":
            from flash.providers.vast.jobs.builders import VastJobHandle

            handle = VastJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.instance_id,
                handle.offer_id,
                handle.machine_id,
                handle.label,
            )
    except (TypeError, ValueError):
        return None
    return None


def _compare_and_clear_remote(run_id: str, expected_remote: dict) -> bool:
    """Clear only the nonterminal remote that still names the destroyed resource."""
    expected_identity = _remote_resource_identity(expected_remote)
    if expected_identity is None:
        return False
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if _remote_resource_identity(status.remote) != expected_identity:
            return False
        status.remote = None
        status.updated_at = time.time()
        _save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def _canonical_cleanup_remote(remote: object) -> dict | None:
    """Return the complete strict teardown handle for one exact resource."""
    if not isinstance(remote, dict) or _remote_resource_identity(remote) is None:
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

            return RunpodJobHandle.from_dict(remote).to_dict()
        if provider == "lambda":
            from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

            return LambdaJobHandle.from_dict(remote).to_dict()
        if provider == "vast":
            from flash.providers.vast.jobs.builders import VastJobHandle

            return VastJobHandle.from_dict(remote).to_dict()
    except (TypeError, ValueError):
        return None
    return None


def _cleanup_remote_key(remote: object) -> tuple | None:
    record = _canonical_cleanup_remote(remote)
    if record is None:
        return None
    return _remote_resource_identity(record), record["attempt"]


def _cleanup_remotes_from_raw(raw: dict) -> list[dict]:
    value = raw.get(_CLEANUP_REMOTES_KEY, [])
    if not isinstance(value, list):
        raise RuntimeError("stored cleanup remotes are invalid")
    records = []
    seen = set()
    for item in value:
        record = _canonical_cleanup_remote(item)
        key = _cleanup_remote_key(record)
        if record is None or key is None:
            raise RuntimeError("stored cleanup remote is invalid")
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _snapshot_cleanup_remotes(run_id: str) -> list[dict]:
    with _status_guard(run_id):
        return _cleanup_remotes_from_raw(_load_status_json(run_id))


def _compare_and_remove_cleanup_remote(run_id: str, expected_remote: dict) -> bool:
    expected_key = _cleanup_remote_key(expected_remote)
    if expected_key is None:
        return False
    with _status_guard(run_id):
        raw = _load_status_json(run_id)
        records = _cleanup_remotes_from_raw(raw)
        remaining = [record for record in records if _cleanup_remote_key(record) != expected_key]
        if len(remaining) == len(records):
            return False
        _save_status_unlocked(
            _runstatus_from_json(raw),
            _cleanup_remotes=remaining or None,
        )
    return True


def _drain_cleanup_remotes(run_id: str) -> set[tuple]:
    """Teardown every tracked resource independently, removing only confirmed exact records."""
    records = _snapshot_cleanup_remotes(run_id)
    attempted = set()
    if not records:
        return attempted
    from flash.providers.base import JobHandle
    from flash.runner.lifecycle import _strict_teardown_handle

    for record in records:
        identity = _remote_resource_identity(record)
        if identity is None:
            continue
        attempted.add(identity)
        try:
            _strict_teardown_handle(JobHandle.from_dict(record))
        except Exception:
            continue
        with contextlib.suppress(Exception):
            _compare_and_remove_cleanup_remote(run_id, record)
    return attempted


def _preserve_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist cleanup identity without changing a terminal lifecycle state."""
    record = _canonical_cleanup_remote(remote)
    key = _cleanup_remote_key(record)
    if record is None or key is None:
        return False
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        records = _cleanup_remotes_from_raw(raw)
        if all(_cleanup_remote_key(existing) != key for existing in records):
            records.append(record)
        current_identity = _remote_resource_identity(status.remote)
        identity = _remote_resource_identity(record)
        if current_identity is None or current_identity == identity:
            status.remote = dict(remote)
        status.updated_at = time.time()
        _save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def _update(run_id: str, state: str, *, allow_from_terminal: bool = False, **updates) -> bool:
    """Atomically transition run state with terminal-stickiness. Returns False if rejected.

    Returns ``True`` if the transition was applied, ``False`` if it was rejected because
    the run was already in a terminal state (the sticky compare-and-set below). Callers
    that gate PAID work on a transition (e.g. the recovery path resuming ``_run_training``)
    must check this return so a run concurrently flipped terminal does not get resumed.
    """
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES and state != status.state and not allow_from_terminal:
            return False
        was_terminal = status.state in TERMINAL_STATES
        prev_updated_at = status.updated_at
        status.state = state
        status.updated_at = time.time()
        if state in TERMINAL_STATES and status.finished_at is None:
            # legacy run already terminal: backfill from prior updated_at, not now.
            status.finished_at = prev_updated_at if was_terminal else status.updated_at
        for key, value in updates.items():
            setattr(status, key, value)
        _save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def record_realized_cost(run_id: str, *, realized_cost_usd: float, reconciled_at: float) -> None:
    """Persist reconciliation COGS without touching run state. No-ops if run vanished."""
    with _status_guard(run_id):
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        status.realized_cost_usd = realized_cost_usd
        status.reconciled_at = reconciled_at
        status.updated_at = time.time()
        _save_status_unlocked(status)
    _report_status(status)


_BILLING_FIELDS = frozenset({"billing_state", "billing_error", "billing_charge"})
# deployed is non-terminal but reconciled; its finished_at must survive billing field-only writes.
_FINISHED_AT_PRESERVED_STATES = TERMINAL_STATES | {"deployed"}


def record_billing_state(run_id: str, **fields) -> None:
    """Persist billing fields without touching run state. Never downgrades a charged run."""
    bad = set(fields) - _BILLING_FIELDS
    if bad:
        raise ValueError(f"record_billing_state only writes billing fields, got: {sorted(bad)}")
    with _status_guard(run_id):
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        new_billing_state = fields.get("billing_state")
        if (
            status.billing_state == "charged"
            and "billing_state" in fields
            and new_billing_state != "charged"
        ):
            return
        # Backfill finished_at before bumping updated_at so reconcile._terminal_ts isn't skewed.
        if (
            status.state in _FINISHED_AT_PRESERVED_STATES
            and status.finished_at is None
            and not status.reconciled_at
        ):
            status.finished_at = status.updated_at
        for key, value in fields.items():
            setattr(status, key, value)
        status.updated_at = time.time()
        _save_status_unlocked(status)
    _report_status(status)


def _report_status(status: RunStatus) -> None:
    with contextlib.suppress(Exception):
        from flash.server.run_registry import record_training_run

        record_training_run(status=status)


def _save_status(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | None | object = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
) -> None:
    with _status_guard(status.run_id):
        if _opd_retry_contract_version is not _PRIVATE_VALUE_UNSET:
            require_opd_retry_contract_version(_opd_retry_contract_version)
            if JobSpec.from_dict(status.spec).algorithm != "opd":
                raise ValueError("opd retry contract cannot be stored for a non-opd run")
        if not os.path.exists(runs_file_path(status.run_id, ".json")):
            if _run_deadline_at is _PRIVATE_VALUE_UNSET:
                spec = JobSpec.from_dict(status.spec)
                _run_deadline_at = _require_valid_deadline(
                    _require_valid_deadline(status.created_at)
                    + _require_valid_deadline(spec.gpu.max_wall_seconds)
                )
            if _next_attempt is _PRIVATE_VALUE_UNSET:
                _next_attempt = 0
        _save_status_unlocked(
            status,
            _run_deadline_at=_run_deadline_at,
            _next_attempt=_next_attempt,
            _cleanup_remotes=_cleanup_remotes,
            _opd_retry_contract_version=_opd_retry_contract_version,
        )


def _save_status_unlocked(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | None | object = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    # write-then-rename so concurrent readers never see a half-written file.
    path = runs_file_path(status.run_id, ".json")
    existing = _load_status_json(status.run_id) if os.path.exists(path) else {}
    private_values = {
        _RUN_DEADLINE_AT_KEY: _run_deadline_at,
        _NEXT_ATTEMPT_KEY: _next_attempt,
        _CLEANUP_REMOTES_KEY: _cleanup_remotes,
        _OPD_RETRY_CONTRACT_KEY: _opd_retry_contract_version,
    }
    data = _status_storage_dict(status)
    for key in _PRIVATE_STATUS_KEYS:
        value = private_values[key]
        if value is _PRIVATE_VALUE_UNSET:
            value = existing.get(key, _PRIVATE_VALUE_UNSET)
        if value is not _PRIVATE_VALUE_UNSET and value is not None:
            data[key] = value
    fd, tmp = tempfile.mkstemp(dir=RUNS_DIR, prefix=f"{status.run_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(RUNS_DIR, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


from flash.runner.deploy import (  # noqa: E402,F401
    DeploymentRevocationError,
    DeploymentStatePersistenceError,
    attach_run,
    cancel_run,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_failed,
    mark_deployment_pending,
    mark_deployment_revocation_failed,
    mark_deployment_undeployed,
    mark_undeployed,
)
from flash.runner.lifecycle import (  # noqa: E402,F401
    _gc_run_endpoints,
    _run_job,
    _run_job_inner,
    _run_training,
    _spec_with_gpu,
    _submit_seed_supervised,
)
from flash.runner.verified_revisions import (  # noqa: E402,F401
    add_verified_adapter_revision,
    clear_verified_adapter_revisions,
    read_verified_adapter_revisions,
    verified_adapter_revision_generation,
)
