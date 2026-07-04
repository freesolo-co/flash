"""Platform runner: drives managed RunPod GPUs, one allocation per run."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

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
    """The customer charge for a run: the flash.cost estimate (training-only steps x sec/step x $/hr).

    This is the price the run was QUOTED at submit. ``steps=None`` prices the spec's planned steps
    (a completed run is charged exactly its quote); pass the actual steps that ran to re-price a
    CANCELLED run at how far it got. Returns ``fallback`` if the spec can't be priced (so a charge is
    never blocked by a pricing failure)."""
    try:
        from flash.cost.analytical import estimate_cost
        from flash.cost.spec import estimate_for_spec, runconfig_from_spec

        if steps is None:
            return float(estimate_for_spec(spec).total_usd)
        n = max(0, int(steps))
        if n == 0:
            return 0.0  # cancelled before any training step -> nothing to charge
        from dataclasses import replace

        cfg = runconfig_from_spec(spec)
        planned = int(cfg.steps or 0)
        if planned > 0:
            n = min(n, planned)
        # SFT is priced from train_tokens (not steps), so lowering steps ALONE wouldn't prorate a
        # cancel -- estimate_cost would still charge the full-run token estimate. Scale the token
        # count to the fraction of steps that ran so a mid-training SFT cancel is charged its share,
        # mirroring GRPO's steps-based proration.
        if not cfg.is_grpo and cfg.train_tokens and planned > 0:
            scaled_tokens = max(1, int(cfg.train_tokens * n / planned))
            cfg = replace(cfg, steps=n, train_tokens=scaled_tokens)
        else:
            cfg = replace(cfg, steps=n)
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
# GRPO rollout can be ~17 min) streams one of these stages with NO step yet -- still real GPU time.
_TRAINING_STAGES = frozenset({"rl_step", "sft_step"})


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
    # Training started (rl_step/sft_step) but no completed step yet -> mid-first-step -> bill 1.
    if hb.get("stage") in _TRAINING_STAGES:
        return 1
    return 0


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

    def to_dict(self) -> dict:
        data = asdict(self)
        data["adapter_ref"] = (
            _adapter_ref_from_status_spec(self.spec) if self.state in {"done", "deployed"} else None
        )
        return data


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


def _resolve_init_from_adapter(
    spec: JobSpec, *, owner_org_id: str = "", owner_key_id: int | None = None
) -> JobSpec:
    """Resolve the public `<run_id>[/step-N]` warm-start ref into the internal storage reference.

    The control plane owns run metadata, so the short ref is resolved HERE (never on the worker):
    the source run's hf_repo + phase key the artifact location the worker downloads from.
    """
    ref = spec.train.init_from_adapter
    if not ref:
        return spec
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
        if src_org_id:
            owner_ok = src_org_id == owner_org_id
        else:
            owner_ok = _source_owned_by_key(src_run_id, owner_key_id)
        if not owner_ok:
            raise ValueError(
                "train.init_from_adapter source run must belong to the same Freesolo org"
            )
    src_spec = JobSpec.from_dict(src_status.spec)
    if not src_spec.train.hf_repo:
        raise ValueError(
            f"train.init_from_adapter run {src_run_id!r} has no stored adapter artifacts"
        )
    if step is not None:
        from flash.runner.checkpoints import CheckpointListingError, checkpoint_step_exists

        try:
            exists = checkpoint_step_exists(src_spec, step)
        except CheckpointListingError as exc:
            raise ValueError(str(exc)) from exc
        if not exists:
            raise ValueError(
                f"train.init_from_adapter references {src_run_id}/step-{step}, but that "
                "deployable checkpoint was not found"
            )
    else:
        if src_status.state not in {"done", "deployed"}:
            raise ValueError(
                f"train.init_from_adapter references run {src_run_id!r}, but that run is "
                f"{src_status.state!r}; use a completed source run or a concrete "
                f"{src_run_id}/step-N checkpoint"
            )
        from flash.runner.checkpoints import CheckpointListingError, final_adapter_exists

        try:
            exists = final_adapter_exists(src_spec)
        except CheckpointListingError as exc:
            raise ValueError(str(exc)) from exc
        if not exists:
            raise ValueError(
                f"train.init_from_adapter references run {src_run_id!r}, but its final "
                "adapter was not found; use a concrete checkpoint ref like "
                f"{src_run_id}/step-N if one exists"
            )
    storage = checkpoint_storage_ref(src_spec.train.hf_repo, src_spec.phase, src_run_id, step)
    return replace(spec, train=replace(spec.train, init_from_adapter=storage))


def submit_job(
    spec: JobSpec,
    dry_run: bool = False,
    background: bool = False,
    runtime_secrets: dict[str, str] | None = None,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
) -> RunStatus:
    """Submit a job. In real mode this allocates and provisions the cheapest validated GPU class
    that fits the run; dry-run only records state."""
    _require_priced_sft_examples(spec)
    info = resolve_model(spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type)
    # "local" is the JobSpec placeholder; treat it as unset so programmatic callers get unique ids.
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else new_run_id()
    spec = JobSpec.from_dict({**_with_model_disk(spec, info), "run_id": run_id})
    spec = _assign_managed_hf_repo(spec)
    spec = _assign_weight_cache_volume(spec, info)
    from flash.providers import INSTANCE_PROVIDERS, available_providers

    public_spec = spec
    estimated_cost_usd: float | None = None
    if not dry_run:
        from flash.cost.spec import estimate_for_spec

        estimated_cost_usd = float(estimate_for_spec(public_spec).total_usd)
    owner_org_id = _context_org_id(billing_context) or _context_org_id(platform_context)
    worker_spec = _resolve_init_from_adapter(
        public_spec,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
    )
    if not dry_run:
        from flash.lora_rank import preflight_init_adapter_lora_rank

        preflight_init_adapter_lora_rank(worker_spec, token=os.environ.get("HF_TOKEN"))
    # env ref->sha pin is deferred (background) or after status save (sync) — never on creation path.
    status = RunStatus(
        run_id=public_spec.run_id,
        state="queued",
        spec=public_spec.to_dict(),
        estimated_cost_usd=estimated_cost_usd,
        billing_context=billing_context,
        billing_state="pending" if billing_context else None,
        platform_context=platform_context,
        # Snapshot the instance providers available at submit so a later handle-less recovery can fail
        # closed for any phantom-capable one whose creds were since dropped (see _confirm_run_clear).
        # Creds-only check (available_providers -> is_configured), no network on the create path.
        submitted_instance_providers=[
            n for n in available_providers() if n in INSTANCE_PROVIDERS
        ],
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


def get_status(run_id: str) -> RunStatus:
    path = runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        return _runstatus_from_json(json.load(f))


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
    """Write metrics to results/runpod/<phase>/<run_id> and return the customer training cost.

    The run id keeps concurrent/sequential runs of the same phase from
    overwriting each other's artifacts. ``metrics["wall_seconds"]`` is the worker's training-loop
    wall time; setup/cold-start is reported separately and is not included here."""
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
    attach_run,
    cancel_run,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_failed,
    mark_deployment_pending,
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
