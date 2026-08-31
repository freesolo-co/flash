"""run status records, persistence, and run-scoped paths."""

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

try:
    import fcntl
except ImportError:  # pragma: no cover - linux production fails closed below
    fcntl = None

from flash._internal.paths import data_dir
from flash.core.spec import JobSpec
from flash.engine.plan.prompt_budget import PromptBudget
from flash.teacher.retry_contract import (
    OPD_RETRY_CONTRACT_STATUS_KEY,
    require_opd_retry_contract_version,
)

_STATE_DIR = str(data_dir())
RUNS_DIR = os.path.join(_STATE_DIR, "runs")
RESULTS_DIR = os.path.join(_STATE_DIR, "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# `done` IS deployable, so excluded; cancelled/failed/dry_run must never flip to `deployed`.
_UNDEPLOYABLE_STATES = TERMINAL_STATES - {"done"}
# serialize local writers before taking each run's interprocess lock.
_RUN_DEADLINE_AT_KEY = "run_deadline_at"
_NEXT_ATTEMPT_KEY = "next_attempt"
_CLEANUP_REMOTES_KEY = "cleanup_remotes"
_OPD_RETRY_CONTRACT_KEY = OPD_RETRY_CONTRACT_STATUS_KEY
_RETRY_STATE_KEY = "retry_state"
_ACTIVE_LAUNCH_CLAIM_KEY = "active_launch_claim"
_PRIVATE_STATUS_KEYS = frozenset(
    {
        _RUN_DEADLINE_AT_KEY,
        _NEXT_ATTEMPT_KEY,
        _CLEANUP_REMOTES_KEY,
        _OPD_RETRY_CONTRACT_KEY,
        _RETRY_STATE_KEY,
        _ACTIVE_LAUNCH_CLAIM_KEY,
    }
)
_PRIVATE_VALUE_UNSET = object()
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


def _internal_spec_from_status(status: RunStatus) -> JobSpec:
    """Reconstruct the run's complete internal job spec for the runner's lifecycle logic.

    status.spec is the public representation and omits platform-managed fields (hf_repo,
    max_wall_seconds, run_id, ...); their authoritative values are persisted verbatim in the
    internal worker spec under effective_preparation (recorded for every provisioned run). Prefer
    that carrier; fall back to the public spec for runs recorded before an effective worker spec
    exists, where those fields carry their managed defaults.
    """
    snapshot = status.effective_preparation
    raw_worker = snapshot.get("worker_spec") if isinstance(snapshot, dict) else None
    if isinstance(raw_worker, dict):
        try:
            return JobSpec.from_dict(raw_worker)
        except Exception:
            pass
    return JobSpec.from_dict(status.spec)


def _adapter_ref_for_status(status: RunStatus) -> str | None:
    """The public short adapter reference (`<run_id>`) shown by `flash runs status` once a run's trained

    adapter is registered; exactly what users paste into train.init_from_adapter (`<run_id>/step-N`
    targets a saved checkpoint). hf_repo, the control-plane-assigned artifact repo that signals the
    adapter exists, is platform-managed and read from the internal worker spec (see
    _internal_spec_from_status); run_id comes from the RunStatus itself.
    """
    raw_worker = (status.effective_preparation or {}).get("worker_spec")
    if not raw_worker:
        return None
    try:
        spec = _internal_spec_from_status(status)
    except Exception:
        # a status json written by an older plane can carry since-removed spec keys (e.g.
        # ``gpu.exact_type`` pre-#670), and stored run records are never rewritten. jobspec.from_dict
        # is strict, so parsing raises -- and one such record would 500 the whole runs list. same
        # operational tolerance as _runstatus_from_json: the record stays readable, it just shows no
        # adapter ref (its spec cannot name one we could resolve).
        return None
    if not spec.train.hf_repo:
        return None
    from flash.schema import format_checkpoint_ref

    return format_checkpoint_ref(status.run_id, None)


# Heartbeat stages that mean the worker has entered training (GPU work underway). The per-step
# `step` field is 1-indexed and only appears once a step COMPLETES, so the expensive first step (a
# GRPO rollout can be ~17 min, an opd step waits on the teacher round-trips) streams one of these
# stages with NO step yet -- still real GPU time.
_TRAINING_STAGES = frozenset({"rl_step", "sft_step", "opd_step"})


@dataclass
class RunStatus:
    run_id: str
    state: str
    spec: dict
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    report_sequence: int = 0
    cost_usd: float = 0.0
    # Submit-time flash.cost quote. Successful runs copy this into cost_usd at completion so the
    # customer is charged exactly what was estimated before paid work started.
    estimated_cost_usd: float | None = None
    error: str | None = None
    artifacts_dir: str | None = None
    adapter_ref: str | None = None
    deployment: dict | None = None
    remote: dict | None = None
    # private canonical attempt proofs stamped once when each lifecycle milestone is first observed.
    lifecycle_started_attempt: int | None = None
    lifecycle_progressed_attempt: int | None = None
    # exact provider handle whose teardown was confirmed independently of retained billing identity.
    cleanup_confirmed_remote: dict | None = None
    # exact torn-down provider handle retained until both delayed realized-cost reconciliation and any
    # pending customer charge complete. it is private and is not an active resource.
    realized_cost_remote: dict | None = None
    # Instance providers (lambda/vast) configured WHEN THIS RUN WAS SUBMITTED — the set that could have
    # owned a pre-handle non-idempotent create. Recovery's phantom guard (_confirm_run_clear) fails
    # closed for any of these that is no longer configurable (so it can't ENUMERATE to prove clear),
    # scoped here so a plane that never configured Vast never blocks a handle-less recovery on it. None
    # for runs created outside submit() / pre-feature records.
    submitted_instance_providers: list[str] | None = None
    # Realized provider cost (COGS), pulled from the provider's billing API after the run
    # finishes by the reconciliation job (flash/server/domain/ops/reconcile.py) and reported to the
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
    workload_profile_input_digest: str | None = None
    workload_profile: dict | None = None
    # submit-time derived grpo/opd prompt budget from flash.engine.plan.prompt_budget. workers drop
    # over-budget prompts rather than truncating them, so record the value before gpu allocation.
    # none for sft, which reports truncation through workload_profile, and for older records.
    prompt_budget: PromptBudget | None = None
    effective_preparation: dict | None = None
    # full managed source descriptor, kept private because it carries the repository path.
    source_snapshot: dict | None = None
    source_verified_attempt: int | None = None

    def to_dict(self) -> dict:
        """Return the public run status representation."""
        from flash.serve.contract.urls import public_deployment

        data = _status_storage_dict(self)
        data["spec"] = _public_status_spec(data.get("spec"))
        data.pop("report_sequence", None)
        data.pop("lifecycle_started_attempt", None)
        data.pop("lifecycle_progressed_attempt", None)
        data.pop("cleanup_confirmed_remote", None)
        data.pop("realized_cost_remote", None)
        # internal warm-start preparation (storage locators, digests) never leaves the server
        data.pop("effective_preparation", None)
        heartbeat = data.get("last_heartbeat")
        if isinstance(heartbeat, dict):
            heartbeat.pop("source_provenance", None)
        source_snapshot = data.pop("source_snapshot", None)
        data.pop("source_verified_attempt", None)
        if source_snapshot is not None:
            from flash.snapshot.archive import safe_public_projection

            data["source_provenance"] = safe_public_projection(
                source_snapshot,
                verified_attempt=self.source_verified_attempt,
            )
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
                # mirror to_dict()'s warm-start strip: the parser rejects both keys alongside
                # init_from_adapter, so leaving either here yields a public status spec that
                # cannot be re-submitted. alpha only reaches this branch now that it is
                # authorable and no longer stripped unconditionally.
                train.pop("lora_rank", None)
                train.pop("lora_alpha", None)
            data["train"] = train
    _redact_internal_adapter_ref(data)
    return data


def _redact_internal_adapter_ref(data: dict) -> None:
    """Never surface an internal storage locator in the public spec.

    A ref is published only when it is PROVEN user-facing, never merely because this build failed to
    parse it as internal. Those are different claims: a persisted locator whose phase this build no
    longer knows (``opsd``, removed in #784) stops parsing as internal, and inferring "public" from
    that published the private repo verbatim.
    """
    train = data.get("train")
    if not isinstance(train, dict):
        return
    ref = train.get("init_from_adapter")
    if not isinstance(ref, str) or not ref.strip():
        return
    from flash.schema import format_checkpoint_ref, parse_adapter_storage_ref, parse_checkpoint_ref

    if parse_checkpoint_ref(ref) is not None:
        return  # the user-facing grammar, and the only one a submit accepts
    resolved = parse_adapter_storage_ref(ref)
    if resolved is None:
        # Neither grammar: cannot show it is free of a private repo, so do not publish it.
        train.pop("init_from_adapter", None)
        return
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
    if data.get("source_snapshot") is None:
        data.pop("source_snapshot", None)
    data["adapter_ref"] = (
        _adapter_ref_for_status(status) if status.state in {"done", "deployed"} else None
    )
    return data


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


def _save_status(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | object | None = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
    _retry_state: dict | object | None = _PRIVATE_VALUE_UNSET,
    _active_launch_claim: dict | object | None = _PRIVATE_VALUE_UNSET,
) -> None:
    from flash.runner.lifecycle import deadlines

    with _status_guard(status.run_id):
        if _opd_retry_contract_version is not _PRIVATE_VALUE_UNSET:
            require_opd_retry_contract_version(_opd_retry_contract_version)
            if JobSpec.from_dict(status.spec).algorithm != "opd":
                raise ValueError("opd retry contract cannot be stored for a non-opd run")
        if not os.path.exists(runs_file_path(status.run_id, ".json")):
            # both defaults below are derived from the internal spec, so parse it once. a record
            # whose spec no longer parses (an older writer's shape, reached by the billing sweep)
            # still has to persist, so an unreadable spec yields no spec-derived default rather
            # than failing the save.
            try:
                spec = _internal_spec_from_status(status)
            except (ValueError, TypeError):
                spec = None
            if _run_deadline_at is _PRIVATE_VALUE_UNSET and spec is not None:
                # max_wall_seconds is managed and stripped from the public status.spec; source the
                # run-global wall budget from the internal worker spec so the auto-computed deadline
                # reloads consistently (see _canonical_run_deadline).
                base = deadlines._require_valid_deadline(status.created_at)
                _run_deadline_at = deadlines._require_valid_deadline(
                    base + deadlines._require_valid_deadline(spec.gpu.max_wall_seconds)
                )
            if _next_attempt is _PRIVATE_VALUE_UNSET:
                _next_attempt = 0
            if _retry_state is _PRIVATE_VALUE_UNSET and spec is not None:
                from flash.runner.supervise.retry_decision import RetryState

                # retry policy is derived from the spec, so a record without one gets no snapshot.
                # every reader requires one and fails closed, which is the right answer here: a run
                # whose spec cannot be read cannot be relaunched either.
                _retry_state = RetryState.initial_for_spec(spec).to_snapshot()
        _save_status_unlocked(
            status,
            _run_deadline_at=_run_deadline_at,
            _next_attempt=_next_attempt,
            _cleanup_remotes=_cleanup_remotes,
            _opd_retry_contract_version=_opd_retry_contract_version,
            _retry_state=_retry_state,
            _active_launch_claim=_active_launch_claim,
        )


def _save_status_unlocked(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | object | None = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
    _retry_state: dict | object | None = _PRIVATE_VALUE_UNSET,
    _active_launch_claim: dict | object | None = _PRIVATE_VALUE_UNSET,
) -> None:
    from flash.runner.lifecycle import reporting
    from flash.runner.lifecycle.status import _load_status_json

    os.makedirs(RUNS_DIR, exist_ok=True)
    # write-then-rename so concurrent readers never see a half-written file.
    path = runs_file_path(status.run_id, ".json")
    existing = _load_status_json(status.run_id) if os.path.exists(path) else {}
    existing_sequence = reporting._valid_status_report_sequence(existing.get("report_sequence", 0))
    current_sequence = reporting._valid_status_report_sequence(status.report_sequence)
    with reporting._STATUS_REPORT_LOCK:
        local_sequence = max(
            reporting._STATUS_REPORT_LAST_QUEUED.get(status.run_id, 0),
            reporting._STATUS_REPORT_LAST_SENT.get(status.run_id, 0),
        )
    status.report_sequence = max(current_sequence, existing_sequence, local_sequence) + 1
    private_values = {
        _RUN_DEADLINE_AT_KEY: _run_deadline_at,
        _NEXT_ATTEMPT_KEY: _next_attempt,
        _CLEANUP_REMOTES_KEY: _cleanup_remotes,
        _OPD_RETRY_CONTRACT_KEY: _opd_retry_contract_version,
        _RETRY_STATE_KEY: _retry_state,
        _ACTIVE_LAUNCH_CLAIM_KEY: _active_launch_claim,
    }
    data = _status_storage_dict(status)
    for key in _PRIVATE_STATUS_KEYS:
        value = private_values[key]
        if value is _PRIVATE_VALUE_UNSET:
            value = existing.get(key, _PRIVATE_VALUE_UNSET)
        # an explicit None drops the key (it skips the carry-forward above, then this write);
        # _PRIVATE_VALUE_UNSET means "keep whatever is on disk".
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
