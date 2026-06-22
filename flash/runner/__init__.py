"""Platform runner: drives managed GPUs across providers (RunPod Flash + Vast), one allocation per seed."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from flash.catalog import ModelInfo, resolve_model
from flash.spec import JobSpec

# Fixed local storage roots (not operator-configurable): run-state JSON + result artifacts,
# both under the ~/.flash state dir (same root as server/db.py's DB_PATH) so a single
# directory holds all control-plane state — mount one volume at ~/.flash to persist it.
# Tests redirect them via monkeypatch.setattr(runner, "RUNS_DIR"/"RESULTS_DIR").
_STATE_DIR = os.path.join(os.path.expanduser("~"), ".flash")
RUNS_DIR = os.path.join(_STATE_DIR, "runs")
RESULTS_DIR = os.path.join(_STATE_DIR, "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# Terminal states a deploy must NOT overwrite. `done` is terminal but IS deployable
# (deploying a finished run is the whole point), so it's excluded here; cancelled/failed/
# dry_run must never be flipped to `deployed`.
_UNDEPLOYABLE_STATES = TERMINAL_STATES - {"done"}
# Serializes the read-check-write in _update so a status transition is an atomic
# compare-and-set (the control plane is single-instance with per-run threads).
_STATUS_LOCK = threading.Lock()


def artifacts_dir(spec: JobSpec) -> str:
    """Run-scoped artifact root: results/runpod/<phase>/<run_id>."""
    return os.path.join(RESULTS_DIR, "runpod", spec.phase, spec.run_id)


def adapter_prefix(spec: JobSpec, seed: int | None = None) -> str:
    """A run's adapter location on the HF artifact store."""
    chosen = spec.train.seeds[0] if seed is None else seed
    return f"{spec.phase}/{spec.run_id}/seed{chosen}"


def _gpu_rate(gpu_type: str) -> float:
    """Representative $/hr for cost projection (live RunPod pricing, static fallback);
    the worker also records wall time so cost = wall_hours * rate."""
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
    deployment: dict | None = None
    # Durable job handle {endpoint_id, endpoint_name, job_id} — lets any process
    # reattach to / cancel the remote job (see `flash attach`).
    remote: dict | None = None
    # Index of the next seed to run for a multi-seed job, set while the remote handle
    # is cleared in the gap between seeds. Lets recover_runs resume the remaining seeds
    # after an inter-seed restart instead of failing the run (losing completed work).
    resume_seed_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _RunCancelled(RuntimeError):
    """User cancellation observed mid-run; terminal, never retried/overwritten."""


def new_run_id() -> str:
    return f"flash-{int(time.time())}-{uuid.uuid4().hex[:8]}"


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def require_safe_run_id(run_id: str) -> str:
    """Reject run ids that could traverse outside the runs directory.

    Run ids flow from API path params into filesystem paths (status json,
    log files); restrict them to a conservative filename alphabet.
    """
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def runs_file_path(run_id: str, suffix: str) -> str:
    """Containment-checked path for a run's file under RUNS_DIR.

    Belt and braces with require_safe_run_id: the resolved path must stay
    inside the runs directory even if the alphabet check ever regresses.
    """
    base = os.path.abspath(RUNS_DIR)
    path = os.path.normpath(os.path.join(base, f"{require_safe_run_id(run_id)}{suffix}"))
    if not path.startswith(base + os.sep):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return path


def _with_model_disk(spec: JobSpec, info: ModelInfo) -> dict:
    """Spec dict with gpu.disk_gb raised to the model's min_disk_gb (catalog).

    Big-checkpoint models (whose weights alone exceed the default) need more container
    disk than the platform's 64 GB default; this makes them work without users having
    to know the right ``gpu.disk_gb``.
    """
    d = spec.to_dict()
    need = int(getattr(info, "min_disk_gb", 0) or 0)
    if need > int(d["gpu"].get("disk_gb") or 0):
        d["gpu"] = {**d["gpu"], "disk_gb": need}
    return d


# The HF namespace the control plane creates per-run artifact repos under: the operator org whose
# HF_TOKEN the control plane runs with. An operator-infra constant, not a user/env knob.
_ARTIFACT_NAMESPACE = "Freesolo-Co"


def _assign_managed_hf_repo(spec: JobSpec) -> JobSpec:
    """Assign the run's HF artifact repo server-side — it is platform-managed, never user-set.

    Each run gets its own private dataset repo ``Freesolo-Co/flashrun-<run_id>``. The control-plane
    HF_TOKEN creates and writes it (code, adapters, checkpoints, telemetry); a user-chosen namespace
    would 403 that token at ``upload_code``. Any inbound ``train.hf_repo`` is overwritten. The
    run_id must be finalized first: a per-run repo keyed on the placeholder ``"local"`` would
    collide across runs and overwrite each other's code/adapters/state.
    """
    if not spec.run_id or spec.run_id == "local":
        raise ValueError("run_id must be finalized before assigning the per-run artifact repo")
    repo = f"{_ARTIFACT_NAMESPACE}/flashrun-{spec.run_id}"
    d = spec.to_dict()
    d["train"] = {**d["train"], "hf_repo": repo}
    return JobSpec.from_dict(d)


def submit_job(spec: JobSpec, dry_run: bool = False, background: bool = False) -> RunStatus:
    """Submit a job. In real mode this allocates and provisions the cheapest validated GPU class
    across the configured providers (RunPod Flash or Vast); dry-run only records state."""
    info = resolve_model(spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type)
    # Finalize the run_id BEFORE assigning the per-run artifact repo. The JobSpec default run_id is
    # the placeholder "local" (truthy), so `or new_run_id()` alone would keep it; treat "local" as
    # unset so programmatic/test callers also get a unique id and per-run repos never collide.
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else new_run_id()
    spec = JobSpec.from_dict({**_with_model_disk(spec, info), "run_id": run_id})
    # The artifact repo is assigned here, after the run_id is finalized: per-run, operator-owned.
    spec = _assign_managed_hf_repo(spec)
    status = RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    _save_status(status)
    if dry_run:
        status.state = "dry_run"
        _save_status(status)
        return status
    if background:
        threading.Thread(target=_run_job, args=(spec,), daemon=True).start()
        return get_status(spec.run_id)
    _run_job(spec)
    return get_status(spec.run_id)


def get_status(run_id: str) -> RunStatus:
    path = runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        return RunStatus(**json.load(f))


def list_runs() -> list[RunStatus]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(RUNS_DIR, name)) as f:
                runs.append(RunStatus(**json.load(f)))
    return runs


def get_logs(run_id: str) -> str:
    log_path = runs_file_path(run_id, ".log")
    if not os.path.exists(log_path):
        return ""
    with open(log_path) as f:
        return f.read()


def _persist_metrics(spec: JobSpec, seed: int, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id>/seedN and return the cost.

    The run id keeps concurrent/sequential runs of the same phase+seed from
    overwriting each other's artifacts. Vast runs arrive with ``cost_usd`` already
    stamped from the offer's real $/hr (plus provider notes) and short-circuit the
    rate fallback below (the RunPod projection)."""
    dest = os.path.join(artifacts_dir(spec), f"seed{seed}")
    os.makedirs(dest, exist_ok=True)
    # Rate the actually-allocated class, not the parse-time provisional spec.gpu.type:
    # a policy GPU can be re-allocated to a different RunPod class at submit time, so
    # the worker stamps "allocated_gpu" into metrics for the cost fallback below.
    gpu_type = metrics.get("allocated_gpu") or spec.gpu.type
    rate = _gpu_rate(gpu_type)
    # A non-runpod provider (e.g. Vast) stamps the real cost_usd from its offer's $/hr
    # AND tags notes["provider"] with its own name — and a near-zero-duration run can
    # legitimately stamp cost_usd == 0.0. The RunPod arm, by contrast, never stamps a real
    # cost: it arrives with cost_usd absent (or a 0.0 placeholder) and no provider note, so
    # the wall-based projection below must run. A bare `cost or 0.0` would treat the Vast
    # 0.0 as "absent" and re-rate it against RunPod pricing while overwriting the provider
    # notes, mis-attributing the run to 'runpod'. So fall back only when the cost is
    # missing/zero AND it has NOT already been attributed to a non-runpod provider.
    _notes = metrics.get("notes")
    _stamped_provider = _notes.get("provider") if isinstance(_notes, dict) else None
    _non_runpod = bool(_stamped_provider) and _stamped_provider != "runpod"
    cost = metrics.get("cost_usd")
    if cost or _non_runpod:
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
    return float(cost)


def _update(run_id: str, state: str, *, allow_from_terminal: bool = False, **updates) -> bool:
    """Atomically transition a run's status, honoring terminal-stickiness.

    Returns ``True`` if the transition was applied, ``False`` if it was rejected because
    the run was already in a terminal state (the sticky compare-and-set below). Callers
    that gate PAID work on a transition (e.g. the recovery path resuming ``_run_seed_loop``)
    must check this return so a run concurrently flipped terminal does not get resumed.
    """
    # The read-check-write below must be atomic: a concurrent `flash cancel` (also via
    # _update) landing between the get_status read and the _save_status write could
    # otherwise be clobbered by this stale background update, resurrecting a cancelled
    # run. The control plane is single-instance with per-run threads, so a process-wide
    # lock serializes all status transitions into a compare-and-set.
    with _STATUS_LOCK:
        status = get_status(run_id)
        # Terminal states are STICKY: once a run is done/failed/cancelled/dry_run, no
        # other state may overwrite it. This closes the whole cancel-race class at the
        # source — a cancel landing between a caller's check and a later write
        # (provisioning/running, or even a late terminal done/failed from a worker that
        # finished as the cancel arrived) can no longer resurrect the run. Same-state
        # writes still pass so terminal field updates (cost_usd, error, artifacts_dir)
        # are preserved.
        #
        # allow_from_terminal is the NARROW escape hatch used ONLY by cancel_run's final
        # `cancelled` transition, and ONLY when the run was `deployed` at cancel entry (see
        # cancel_run). In that case an explicit user cancel must WIN over a racing
        # mark_undeployed() that flipped the `deployed` run to terminal `done` mid-teardown —
        # that `done` is an undeploy artifact (restoring the pre-deploy completion marker while
        # retiring serving), not a fresh result. Without the override the `cancelled` write
        # no-ops against the freshly-written `done` and the run wrongly ends `done` despite the
        # user asking to cancel. cancel_run passes allow_from_terminal=False for a non-deployed
        # run, so a GENUINE training-completion `done` racing in from the run's own training
        # thread is protected by the CAS below — cancel correctly loses to a real finish.
        if status.state in TERMINAL_STATES and state != status.state and not allow_from_terminal:
            return False
        status.state = state
        status.updated_at = time.time()
        for key, value in updates.items():
            setattr(status, key, value)
        _save_status(status)
        return True


def _save_status(status: RunStatus) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    # Write-then-rename: a concurrent reader (poll on /v1/runs or /logs) must
    # never observe a half-written/truncated file and 500 on JSONDecodeError.
    # The temp name is UNIQUE per write (mkstemp) so two threads updating the same
    # run (e.g. a cancel racing the background seed update) can't clobber each
    # other's temp file mid-dump — each os.replace is atomic and independent.
    path = runs_file_path(status.run_id, ".json")
    fd, tmp = tempfile.mkstemp(dir=RUNS_DIR, prefix=f"{status.run_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


# Re-export the run-execution and deploy/recover transitions as package-level attributes
# so external `from flash.runner import X` keeps working AND the test monkeypatches
# (flash.runner._run_job / ._gc_run_endpoints / .cancel_run ...) resolve here. These imports
# run AFTER the store layer above is fully defined; lifecycle/deploy import the store via
# FUNCTION-LOCAL lazy `from flash.runner import ...` to avoid a partially-initialized cycle.
from flash.runner.deploy import (  # noqa: E402,F401
    attach_run,
    cancel_run,
    mark_deployed,
    mark_deployment_undeployed,
    mark_undeployed,
    resume_run,
)
from flash.runner.lifecycle import (  # noqa: E402,F401
    _gc_run_endpoints,
    _run_job,
    _run_job_inner,
    _run_seed_loop,
    _spec_with_gpu,
    _submit_seed_supervised,
)
