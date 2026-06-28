"""Platform runner: drives managed RunPod GPUs, one allocation per seed."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

from flash.catalog import ModelInfo, resolve_model
from flash.spec import FIXED_SEED, JobSpec  # noqa: F401  (re-exported for lifecycle/deploy)

_STATE_DIR = os.path.join(os.path.expanduser("~"), ".flash")
RUNS_DIR = os.path.join(_STATE_DIR, "runs")
RESULTS_DIR = os.path.join(_STATE_DIR, "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# `done` IS deployable, so excluded; cancelled/failed/dry_run must never flip to `deployed`.
_UNDEPLOYABLE_STATES = TERMINAL_STATES - {"done"}
_STATUS_LOCK = threading.Lock()


def artifacts_dir(spec: JobSpec) -> str:
    """Run-scoped artifact root: results/runpod/<phase>/<run_id>."""
    return os.path.join(RESULTS_DIR, "runpod", spec.phase, spec.run_id)


def adapter_prefix(spec: JobSpec) -> str:
    """A run's adapter location on the HF artifact store: ``<phase>/<run_id>``."""
    return f"{spec.phase}/{spec.run_id}"


def adapter_ref(spec: JobSpec) -> str | None:
    """Full init_from_adapter reference for a run's trained adapter."""
    if not spec.train.hf_repo:
        return None
    return f"{spec.train.hf_repo}:{adapter_prefix(spec)}"


def _adapter_ref_from_status_spec(raw: dict) -> str | None:
    try:
        return adapter_ref(JobSpec.from_dict(raw))
    except Exception:
        return None


def _gpu_rate(gpu_type: str) -> float:
    """Static representative $/hr for cost projection."""
    try:
        from flash.providers.runpod.pricing import hourly_rate

        return hourly_rate(gpu_type)
    except Exception:
        return 0.80


@dataclass
class RunStatus:
    run_id: str
    state: str
    spec: dict
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    error: str | None = None
    artifacts_dir: str | None = None
    adapter_ref: str | None = None
    deployment: dict | None = None
    remote: dict | None = None
    # Realized provider cost (COGS), pulled from the provider's billing API after the run
    # finishes by the reconciliation job (flash/server/reconcile.py) and reported to the
    # freesolo backend for estimator accuracy. Distinct from ``cost_usd`` (the wall x $/hr
    # PROJECTION); ``reconciled_at`` marks that the realized pull has happened so it isn't
    # re-pulled. Both stay None for un-reconciled / pre-instrumentation runs.
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

    def to_dict(self) -> dict:
        data = asdict(self)
        data["adapter_ref"] = (
            _adapter_ref_from_status_spec(self.spec) if self.state in {"done", "deployed"} else None
        )
        return data

    @classmethod
    def from_persisted(cls, data: dict) -> RunStatus:
        """Build a RunStatus from persisted status JSON, ignoring unknown keys.

        Run-status JSON is forward/backward compatible across control-plane upgrades: a release
        that REMOVES a field (e.g. the old multi-seed ``resume_seed_index``) must still load runs
        written by the prior version. A plain ``RunStatus(**data)`` would raise ``TypeError`` on
        any extra key, breaking startup ``recover_runs`` / ``flash status`` / ``list_runs`` for
        every in-flight run carried across the upgrade. Drop keys not on the current dataclass."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class _RunCancelled(RuntimeError):
    """User cancellation observed mid-run; terminal, never retried/overwritten."""


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


def _with_model_disk(spec: JobSpec, info: ModelInfo) -> dict:
    """Spec dict with gpu.disk_gb raised to the model's catalog min_disk_gb."""
    d = spec.to_dict()
    need = int(getattr(info, "min_disk_gb", 0) or 0)
    if need > int(d["gpu"].get("disk_gb") or 0):
        d["gpu"] = {**d["gpu"], "disk_gb": need}
    return d


_ARTIFACT_NAMESPACE = "Freesolo-Co"


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the per-run HF artifact repo (platform-managed, never user-set); run_id must be finalized first."""
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the per-run artifact repo")
    repo = f"{_ARTIFACT_NAMESPACE}/flashrun-{spec.run_id}"
    d = spec.to_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return JobSpec.from_dict(d)


def _assign_resolved_env_sha(spec: JobSpec) -> JobSpec:
    """Pin env ref->SHA once so N workers don't fan-out N GitHub API calls (secondary rate-limit). Best-effort."""
    import logging

    env_id = spec.environment.id
    if not env_id or spec.environment.resolved_sha:
        return spec
    try:
        from flash.envs.adapter import (
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
    d = spec.to_dict()
    d["environment"] = {**d["environment"], "resolved_sha": sha}
    return JobSpec.from_dict(d)


WEIGHT_CACHE_VOLUME_NAME = "flash-weights"
WEIGHT_CACHE_VOLUME_GB = 100
# Peak footprint ~= 2x bf16 download (checkpoint + Xet temp); must fit the fixed cache volume.
_WEIGHT_CACHE_PEAK_FACTOR = 2.0


def _fits_weight_cache(info: ModelInfo) -> bool:
    """Whether the model's peak download footprint fits the shared weight-cache volume."""
    if not info.params_b:
        return True  # unknown size -> keep the (attach) default; curated catalog models always set it
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
    d = spec.to_dict()
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
    except Exception as e:
        with contextlib.suppress(Exception):
            if get_status(spec.run_id).state not in TERMINAL_STATES:
                _update(spec.run_id, "failed", error=str(e))
        logging.getLogger(__name__).warning("background run %s ended in error: %s", spec.run_id, e)


def submit_job(
    spec: JobSpec,
    dry_run: bool = False,
    background: bool = False,
    runtime_secrets: dict[str, str] | None = None,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
) -> RunStatus:
    """Submit a job. In real mode this allocates and provisions the cheapest validated GPU class
    that fits the run; dry-run only records state."""
    info = resolve_model(spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type)
    # "local" is the JobSpec placeholder; treat it as unset so programmatic callers get unique ids.
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else new_run_id()
    spec = JobSpec.from_dict({**_with_model_disk(spec, info), "run_id": run_id})
    spec = _assign_managed_hf_repo(spec)
    spec = _assign_weight_cache_volume(spec, info)
    # env ref->sha pin is deferred (background) or after status save (sync) — never on creation path.
    status = RunStatus(
        run_id=spec.run_id,
        state="queued",
        spec=spec.to_dict(),
        billing_context=billing_context,
        billing_state="pending" if billing_context else None,
        platform_context=platform_context,
    )
    _save_status(status)
    _report_status(status)
    if dry_run:
        status.state = "dry_run"
        _save_status(status)
        _report_status(status)
        return status
    if background:
        threading.Thread(
            target=_run_job_background,
            args=(spec, runtime_secrets or {}),
            kwargs={"resolve_env_sha": True},
            daemon=True,
        ).start()
        return get_status(spec.run_id)
    spec = _assign_resolved_env_sha(spec)
    if runtime_secrets:
        _run_job(spec, runtime_secrets=runtime_secrets)
    else:
        _run_job(spec)
    return get_status(spec.run_id)


def get_status(run_id: str) -> RunStatus:
    path = runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        return RunStatus.from_persisted(json.load(f))


def list_runs() -> list[RunStatus]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(RUNS_DIR, name)) as f:
                runs.append(RunStatus.from_persisted(json.load(f)))
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
    with _STATUS_LOCK:
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        status.last_heartbeat = hb
        status.gpu_status = gpu if isinstance(gpu, dict) else None
        status.updated_at = time.time()
        _save_status(status)
    _report_status(status)


def _persist_metrics(spec: JobSpec, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id> and return the cost.

    The run id keeps concurrent/sequential runs of the same phase from
    overwriting each other's artifacts."""
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


def _update(run_id: str, state: str, *, allow_from_terminal: bool = False, **updates) -> bool:
    """Atomically transition run state with terminal-stickiness. Returns False if rejected.

    Returns ``True`` if the transition was applied, ``False`` if it was rejected because
    the run was already in a terminal state (the sticky compare-and-set below). Callers
    that gate PAID work on a transition (e.g. the recovery path resuming ``_run_training``)
    must check this return so a run concurrently flipped terminal does not get resumed.

    ``allow_from_terminal`` is the NARROW escape hatch used ONLY by cancel_run's final
    ``cancelled`` write over a racing ``done`` from mark_undeployed — never for a genuine
    training-completion race.
    """
    report_status: RunStatus | None = None
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in TERMINAL_STATES and state != status.state and not allow_from_terminal:
            return False
        was_terminal = status.state in TERMINAL_STATES
        prev_updated_at = status.updated_at
        status.state = state
        status.updated_at = time.time()
        if state in TERMINAL_STATES and status.finished_at is None:
            # Legacy run already terminal: backfill from prior updated_at, not now.
            status.finished_at = prev_updated_at if was_terminal else status.updated_at
        for key, value in updates.items():
            setattr(status, key, value)
        _save_status(status)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def record_realized_cost(run_id: str, *, realized_cost_usd: float, reconciled_at: float) -> None:
    """Persist reconciliation COGS without touching run state. No-ops if run vanished."""
    with _STATUS_LOCK:
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        status.realized_cost_usd = realized_cost_usd
        status.reconciled_at = reconciled_at
        status.updated_at = time.time()
        _save_status(status)
    _report_status(status)


_BILLING_FIELDS = frozenset({"billing_state", "billing_error", "billing_charge"})
# deployed is non-terminal but reconciled; its finished_at must survive billing field-only writes.
_FINISHED_AT_PRESERVED_STATES = TERMINAL_STATES | {"deployed"}


def record_billing_state(run_id: str, **fields) -> None:
    """Persist billing fields without touching run state. Never downgrades a charged run."""
    bad = set(fields) - _BILLING_FIELDS
    if bad:
        raise ValueError(f"record_billing_state only writes billing fields, got: {sorted(bad)}")
    with _STATUS_LOCK:
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
        _save_status(status)
    _report_status(status)


def _report_status(status: RunStatus) -> None:
    with contextlib.suppress(Exception):
        from flash.server.run_registry import record_training_run

        record_training_run(status=status)


def _save_status(status: RunStatus) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    # Write-then-rename so concurrent readers never see a half-written file.
    path = runs_file_path(status.run_id, ".json")
    fd, tmp = tempfile.mkstemp(dir=RUNS_DIR, prefix=f"{status.run_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


from flash.runner.deploy import (  # noqa: E402,F401
    attach_checkpoint_deployment,
    attach_run,
    cancel_run,
    mark_deployed,
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
