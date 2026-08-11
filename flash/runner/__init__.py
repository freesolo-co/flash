"""Platform runner: drives managed RunPod GPUs, one allocation per run."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

try:
    import fcntl
except ImportError:  # pragma: no cover - linux production fails closed below
    fcntl = None

from flash._internal.paths import data_dir
from flash.adapters.artifacts import MAX_ATTEMPT_ID as MAX_ATTEMPT_ID

# `resolve_model` and `TRAINER_BACKEND` have no call site left here since submission moved to
# `.submit`, but tests patch the first and read the second as attributes of THIS package, and the
# moved code reads them back through it. an autofix that drops them breaks those tests.
from flash.core.catalog import ModelInfo, resolve_model  # noqa: F401
from flash.core.spec import (  # noqa: F401
    _DROPPED_TOP_LEVEL_KEYS,
    MANAGED_GPU_KEYS,
    TRAINER_BACKEND,
    GpuSpec,
    JobSpec,
)
from flash.providers._lifecycle.poll import _attempt_int as _attempt_int
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
_STATUS_LOCK = threading.Lock()
_STATUS_REPORT_LOCK = threading.RLock()
_STATUS_REPORT_CONDITION = threading.Condition(_STATUS_REPORT_LOCK)
_STATUS_REPORT_EXECUTOR: ThreadPoolExecutor | None = None
_STATUS_REPORT_QUEUES: dict[str, deque[tuple[RunStatus, int, threading.Event, int]]] = {}
_STATUS_REPORT_WORKERS: dict[str, Future[None]] = {}
_STATUS_REPORT_ACTIVE: set[str] = set()
_STATUS_REPORT_DRAINING: set[str] = set()
_STATUS_REPORT_PENDING = 0
_STATUS_REPORT_ACCEPTING = True
_STATUS_REPORT_LAST_SENT: dict[str, int] = {}
_STATUS_REPORT_LAST_ATTEMPTED: dict[str, int] = {}
_STATUS_REPORT_LAST_QUEUED: dict[str, int] = {}
_RUN_DEADLINE_AT_KEY = "run_deadline_at"
_NEXT_ATTEMPT_KEY = "next_attempt"
_CLEANUP_REMOTES_KEY = "cleanup_remotes"
_OPD_RETRY_CONTRACT_KEY = OPD_RETRY_CONTRACT_STATUS_KEY
# when the plane first heard from a profile's worker. a profile's wall bounds the WORK it does, not
# the wait for a machine to do it on, so its deadline runs from here rather than from submission --
# see _canonical_run_deadline.
_PROFILE_WALL_ARMED_AT_KEY = "profile_wall_armed_at"
# the lowest attempt id belonging to THIS profile lifecycle. a relaunch reuses the run id and
# carries the attempt counter, so without a floor `next_attempt - 1` still names the spent
# lifecycle's attempt while the fresh one queues -- see _persist_profile_submission.
_PROFILE_ATTEMPT_FLOOR_KEY = "profile_attempt_floor"
_PRIVATE_STATUS_KEYS = frozenset(
    {
        _RUN_DEADLINE_AT_KEY,
        _NEXT_ATTEMPT_KEY,
        _CLEANUP_REMOTES_KEY,
        _OPD_RETRY_CONTRACT_KEY,
        _PROFILE_WALL_ARMED_AT_KEY,
        _PROFILE_ATTEMPT_FLOOR_KEY,
    }
)
_PRIVATE_VALUE_UNSET = object()
MIN_PROVIDER_WALL_SECONDS = 60
_WORKLOAD_PROFILE_WALL_SECONDS = 10 * 60
_WORKLOAD_PROFILE_MAX_RETRIES = 1
# a profile's wall bounds the WORK it does, not the wait for a machine to do it on. each provider
# attempt gets its own IN_QUEUE grace (300s) and the infra retry floor allows several of them, so
# a 600s deadline measured from submission cannot survive even two capacity cycles: the run dies
# "run wall deadline exceeded" having profiled nothing, on hardware it never got. queue time gets
# this separate explicit allowance and the wall itself starts at the first heartbeat, so the quote
# stays wall x hourly (see estimate_profile_cost) rather than paying for the queue.
_WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS = 30 * 60
# the plane arms work time at the first heartbeat, but the box enforces a deadline minted when the
# machine was RENTED. between those two clocks sits everything cloud-init does before the worker
# can speak -- waiting
# for docker+gpu (itself budgeted up to ~600s), image pull with retries, extra_pip install and the
# HF code fetch. The box timer is only a backstop against a silent box: the plane re-reads the run
# deadline every poll, tightens it the moment arming happens, and tears the box down when it gives
# up.
_WORKLOAD_PROFILE_PROVISION_ALLOWANCE_SECONDS = 20 * 60


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
    if not (status.effective_preparation or {}).get("worker_spec"):
        return None
    try:
        spec = _internal_spec_from_status(status)
    except Exception:
        # a status json written by an OLDER plane can carry since-removed spec keys (e.g.
        # ``gpu.exact_type`` pre-#670), and stored run records are never rewritten. JobSpec.from_dict
        # is strict, so parsing raises -- and one such record would 500 the whole runs list. same
        # operational tolerance as _runstatus_from_json: the record stays readable, it just shows no
        # adapter ref (its spec cannot name one we could resolve).
        return None
    if spec.workload_profile_kind or not spec.train.hf_repo:
        return None
    return status.run_id


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
    # Instance providers (lambda/vast) configured WHEN THIS RUN WAS SUBMITTED — the set that could have
    # owned a pre-handle non-idempotent create. Recovery's phantom guard (_confirm_run_clear) fails
    # closed for any of these that is no longer configurable (so it can't ENUMERATE to prove clear),
    # scoped here so a plane that never configured Vast never blocks a handle-less recovery on it. None
    # for runs created outside submit() / pre-feature records.
    submitted_instance_providers: list[str] | None = None
    # Realized provider cost (COGS), pulled from the provider's billing API after the run
    # finishes by the reconciliation job (flash/server/domain/reconcile.py) and reported to the
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
    workload_profile_kind: str | None = None
    workload_profile_input_digest: str | None = None
    workload_profile: dict | None = None
    effective_preparation: dict | None = None

    def to_dict(self) -> dict:
        """Return the public run status representation."""
        from flash.serve.urls import public_deployment

        data = _status_storage_dict(self)
        data["spec"] = _public_status_spec(data.get("spec"))
        data.pop("report_sequence", None)
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
    data["adapter_ref"] = (
        _adapter_ref_for_status(status) if status.state in {"done", "deployed"} else None
    )
    return data


class WarmStartPreparationError(ValueError):
    """A submit failed while preparing ``train.init_from_adapter``'s source adapter.

    Lets the submit route blame the adapter only for failures that really came from resolving it.
    """


class WorkloadProfilePending(RuntimeError):
    """Training preparation is blocked on a separately billed workload-profile run."""

    def __init__(
        self,
        profile_run_id: str,
        state: str,
        *,
        prepared_job: object | None = None,
        spent_at: float | None = None,
    ) -> None:
        self.profile_run_id = profile_run_id
        self.state = state
        self.prepared_job = prepared_job
        # set when the previous profile under this id is spent (failed/cancelled/dry_run) and its
        # claim has to be taken over before a replacement can run. the value is that run's own
        # created_at, which is what makes the takeover decidable: a claim stamp re-read at takeover
        # time would already be the winner's, so every later caller would think it won too.
        # None means no spent run was observed and the claim is taken by insert instead.
        self.spent_at = spent_at
        super().__init__(
            f"workload profile {profile_run_id} is {state}; retry training preparation after it succeeds"
        )


class WorkloadProfileUnavailable(ValueError):
    """The exact workload profile failed or cannot be trusted."""


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


_DEFAULT_ARTIFACT_NAMESPACE = "Freesolo-Co"
_ARTIFACT_REPO_PREFIX = "flashrun-"
_ARTIFACT_REPO_NAME_MAX = 96


WEIGHT_CACHE_VOLUME_NAME = "flash-weights"
# Must stay >= weight_cache_catalog_peak_gb(), which is what the whole catalog actually needs on one
# shared volume; test_volume_holds_whole_catalog_with_largest_model_in_transit fails if it does not.
# Grow this (and the provisioned volumes) rather than dropping a model from the cache.
WEIGHT_CACHE_VOLUME_GB = 250
# A download needs the checkpoint plus roughly as much again for Xet reconstruction scratch.
_WEIGHT_CACHE_PEAK_FACTOR = 2.0


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
        from flash.server.platform import db

        return db.run_owner(src_run_id) == owner_key_id
    except Exception:
        return False


@dataclass(frozen=True)
class PreparedJob:
    public_spec: JobSpec
    worker_spec: JobSpec
    estimated_cost_usd: float
    adapter_identity: dict | None = None


_BILLING_FIELDS = frozenset({"billing_state", "billing_error", "billing_charge"})
# deployed is non-terminal but reconciled; its finished_at must survive billing field-only writes.
_FINISHED_AT_PRESERVED_STATES = TERMINAL_STATES | {"deployed"}


def _send_status_report(status: RunStatus) -> bool:
    from flash.server.domain.run_registry import record_training_run

    return record_training_run(status=status)


def _valid_status_report_sequence(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _status_report_sequence_unlocked(status: RunStatus) -> int:
    run_id = status.run_id
    persisted = _valid_status_report_sequence(getattr(status, "report_sequence", 0))
    if persisted > 0:
        return persisted
    return _STATUS_REPORT_LAST_QUEUED.get(run_id, 0) + 1


def _deliver_status_report(status: RunStatus, sequence: int, attempt_budget: int) -> bool:
    with _STATUS_REPORT_LOCK:
        if sequence <= _STATUS_REPORT_LAST_SENT.get(status.run_id, 0):
            return True
        if sequence < _STATUS_REPORT_LAST_ATTEMPTED.get(status.run_id, 0):
            return True
        _STATUS_REPORT_LAST_ATTEMPTED[status.run_id] = sequence
    for attempt in range(attempt_budget):
        try:
            if _send_status_report(status) is not False:
                with _STATUS_REPORT_LOCK:
                    _STATUS_REPORT_LAST_SENT[status.run_id] = sequence
                return True
        except Exception:
            pass
        if attempt + 1 < attempt_budget:
            with _STATUS_REPORT_LOCK:
                if not _STATUS_REPORT_ACCEPTING:
                    return False
    return False


def _finish_status_report(done: threading.Event) -> None:
    global _STATUS_REPORT_PENDING
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_PENDING -= 1
        done.set()
        _STATUS_REPORT_CONDITION.notify_all()


def _drain_status_report_run(run_id: str) -> None:
    with _STATUS_REPORT_CONDITION:
        if run_id in _STATUS_REPORT_DRAINING:
            return
        _STATUS_REPORT_DRAINING.add(run_id)
    while True:
        with _STATUS_REPORT_CONDITION:
            queue = _STATUS_REPORT_QUEUES.get(run_id)
            if not queue:
                _STATUS_REPORT_QUEUES.pop(run_id, None)
                _STATUS_REPORT_ACTIVE.discard(run_id)
                _STATUS_REPORT_DRAINING.discard(run_id)
                _STATUS_REPORT_CONDITION.notify_all()
                return
            status, sequence, done, attempt_budget = queue.popleft()
        delivered = False
        try:
            delivered = _deliver_status_report(status, sequence, attempt_budget)
        finally:
            if not delivered:
                with _STATUS_REPORT_CONDITION:
                    if _STATUS_REPORT_LAST_QUEUED.get(run_id) == sequence:
                        sent = _STATUS_REPORT_LAST_SENT.get(run_id, 0)
                        if sent:
                            _STATUS_REPORT_LAST_QUEUED[run_id] = sent
                        else:
                            _STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
            _finish_status_report(done)


def _discard_status_report_worker(run_id: str, future: Future[None]) -> None:
    fallback = False
    with _STATUS_REPORT_CONDITION:
        if _STATUS_REPORT_WORKERS.get(run_id) is not future:
            return
        _STATUS_REPORT_WORKERS.pop(run_id, None)
        if run_id not in _STATUS_REPORT_DRAINING:
            _STATUS_REPORT_ACTIVE.discard(run_id)
            if _STATUS_REPORT_QUEUES.get(run_id):
                fallback = not _start_status_report_worker_unlocked(run_id)
        _STATUS_REPORT_CONDITION.notify_all()
    if fallback:
        _drain_status_report_run(run_id)


def _status_report_executor_unlocked() -> ThreadPoolExecutor:
    global _STATUS_REPORT_EXECUTOR
    if _STATUS_REPORT_EXECUTOR is None:
        _STATUS_REPORT_EXECUTOR = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="flash-status-reporter",
        )
    return _STATUS_REPORT_EXECUTOR


def _start_status_report_worker_unlocked(run_id: str) -> bool:
    global _STATUS_REPORT_EXECUTOR
    _STATUS_REPORT_ACTIVE.add(run_id)
    executor = _status_report_executor_unlocked()
    try:
        future = executor.submit(_drain_status_report_run, run_id)
    except RuntimeError:
        can_drain_queued_work = (
            not getattr(executor, "_shutdown", False)
            and not getattr(executor, "_broken", False)
            and any(thread.is_alive() for thread in getattr(executor, "_threads", ()))
        )
        if can_drain_queued_work:
            return True
        if _STATUS_REPORT_EXECUTOR is executor:
            _STATUS_REPORT_EXECUTOR = None
        return False
    _STATUS_REPORT_WORKERS[run_id] = future
    future.add_done_callback(
        lambda completed, current_run_id=run_id: _discard_status_report_worker(
            current_run_id, completed
        )
    )
    return True


def _cancel_status_report_run_unlocked(run_id: str, *, forget_sequence: bool) -> None:
    global _STATUS_REPORT_PENDING
    queue = _STATUS_REPORT_QUEUES.pop(run_id, None)
    if queue is not None:
        while queue:
            _, _, done, _attempt_budget = queue.popleft()
            _STATUS_REPORT_PENDING -= 1
            done.set()
    worker = _STATUS_REPORT_WORKERS.get(run_id)
    if (worker is None or worker.done()) and run_id not in _STATUS_REPORT_DRAINING:
        _STATUS_REPORT_ACTIVE.discard(run_id)
    if forget_sequence:
        attempted = _STATUS_REPORT_LAST_ATTEMPTED.get(run_id, 0)
        if attempted:
            _STATUS_REPORT_LAST_QUEUED[run_id] = attempted
        else:
            _STATUS_REPORT_LAST_QUEUED.pop(run_id, None)
    _STATUS_REPORT_CONDITION.notify_all()


def _open_status_reporter() -> None:
    global _STATUS_REPORT_ACCEPTING
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_ACCEPTING = True


def _queue_status_report(status: RunStatus, *, wait: bool) -> None:
    global _STATUS_REPORT_PENDING
    snapshot = copy.deepcopy(status)
    run_id = snapshot.run_id
    done = threading.Event()
    fallback = False
    attempt_budget = 1 if wait else 2
    with _STATUS_REPORT_CONDITION:
        if not _STATUS_REPORT_ACCEPTING:
            return
        sequence = _status_report_sequence_unlocked(snapshot)
        while (
            wait
            and sequence == _STATUS_REPORT_LAST_QUEUED.get(run_id, 0)
            and sequence > _STATUS_REPORT_LAST_SENT.get(run_id, 0)
        ):
            _STATUS_REPORT_CONDITION.wait()
            if not _STATUS_REPORT_ACCEPTING:
                return
        if sequence <= _STATUS_REPORT_LAST_QUEUED.get(run_id, 0):
            return
        _STATUS_REPORT_LAST_QUEUED[run_id] = sequence
        _STATUS_REPORT_QUEUES.setdefault(run_id, deque()).append(
            (snapshot, sequence, done, attempt_budget)
        )
        _STATUS_REPORT_PENDING += 1
        if run_id not in _STATUS_REPORT_ACTIVE:
            fallback = not _start_status_report_worker_unlocked(run_id)
        if fallback and not wait:
            _cancel_status_report_run_unlocked(run_id, forget_sequence=True)
    if fallback and wait:
        _drain_status_report_run(run_id)
    if wait:
        done.wait()


def _report_status(status: RunStatus) -> None:
    _queue_status_report(status, wait=True)


def _report_status_async(status: RunStatus) -> None:
    _queue_status_report(status, wait=False)


def _wait_for_status_reports(timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    with _STATUS_REPORT_CONDITION:
        while _STATUS_REPORT_PENDING:
            if deadline is None:
                _STATUS_REPORT_CONDITION.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _STATUS_REPORT_CONDITION.wait(remaining)
        return True


def _cancel_queued_status_reports_unlocked(*, forget_sequences: bool) -> None:
    for run_id in tuple(_STATUS_REPORT_QUEUES):
        _cancel_status_report_run_unlocked(run_id, forget_sequence=forget_sequences)


def _shutdown_status_reporter(timeout: float = 15.0, *, close: bool = False) -> bool:
    global _STATUS_REPORT_ACCEPTING, _STATUS_REPORT_EXECUTOR
    with _STATUS_REPORT_CONDITION:
        _STATUS_REPORT_ACCEPTING = not close
    flushed = _wait_for_status_reports(timeout)
    with _STATUS_REPORT_CONDITION:
        executor = _STATUS_REPORT_EXECUTOR
        _STATUS_REPORT_EXECUTOR = None
        if not flushed:
            _cancel_queued_status_reports_unlocked(forget_sequences=close)
            for future in tuple(_STATUS_REPORT_WORKERS.values()):
                future.cancel()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=not flushed)
    return flushed


def _save_status(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | object | None = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
    _profile_wall_armed_at: float | object = _PRIVATE_VALUE_UNSET,
    _profile_attempt_floor: int | object = _PRIVATE_VALUE_UNSET,
) -> None:
    with _status_guard(status.run_id):
        if _opd_retry_contract_version is not _PRIVATE_VALUE_UNSET:
            require_opd_retry_contract_version(_opd_retry_contract_version)
            if JobSpec.from_dict(status.spec).algorithm != "opd":
                raise ValueError("opd retry contract cannot be stored for a non-opd run")
        if not os.path.exists(runs_file_path(status.run_id, ".json")):
            if _run_deadline_at is _PRIVATE_VALUE_UNSET:
                # max_wall_seconds is managed and stripped from the public status.spec; source the
                # run-global wall budget from the internal worker spec so the auto-computed deadline
                # reloads consistently (see _canonical_run_deadline).
                spec = _internal_spec_from_status(status)
                base = _require_valid_deadline(status.created_at)
                if spec.workload_profile_kind:
                    # a fresh profile has not been armed yet, so it holds the queue allowance on top
                    # of its work budget -- same basis _canonical_run_deadline reconstructs on read.
                    base += _WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS
                _run_deadline_at = _require_valid_deadline(
                    base + _require_valid_deadline(spec.gpu.max_wall_seconds)
                )
            if _next_attempt is _PRIVATE_VALUE_UNSET:
                _next_attempt = 0
        _save_status_unlocked(
            status,
            _run_deadline_at=_run_deadline_at,
            _next_attempt=_next_attempt,
            _cleanup_remotes=_cleanup_remotes,
            _opd_retry_contract_version=_opd_retry_contract_version,
            _profile_wall_armed_at=_profile_wall_armed_at,
            _profile_attempt_floor=_profile_attempt_floor,
        )


def _save_status_unlocked(
    status: RunStatus,
    *,
    _run_deadline_at: float | object = _PRIVATE_VALUE_UNSET,
    _next_attempt: int | object = _PRIVATE_VALUE_UNSET,
    _cleanup_remotes: list[dict] | object | None = _PRIVATE_VALUE_UNSET,
    _opd_retry_contract_version: int | object = _PRIVATE_VALUE_UNSET,
    _profile_wall_armed_at: float | object = _PRIVATE_VALUE_UNSET,
    _profile_attempt_floor: int | object = _PRIVATE_VALUE_UNSET,
) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    # write-then-rename so concurrent readers never see a half-written file.
    path = runs_file_path(status.run_id, ".json")
    existing = _load_status_json(status.run_id) if os.path.exists(path) else {}
    existing_sequence = _valid_status_report_sequence(existing.get("report_sequence", 0))
    current_sequence = _valid_status_report_sequence(status.report_sequence)
    with _STATUS_REPORT_LOCK:
        local_sequence = max(
            _STATUS_REPORT_LAST_QUEUED.get(status.run_id, 0),
            _STATUS_REPORT_LAST_SENT.get(status.run_id, 0),
        )
    status.report_sequence = max(current_sequence, existing_sequence, local_sequence) + 1
    private_values = {
        _RUN_DEADLINE_AT_KEY: _run_deadline_at,
        _NEXT_ATTEMPT_KEY: _next_attempt,
        _CLEANUP_REMOTES_KEY: _cleanup_remotes,
        _OPD_RETRY_CONTRACT_KEY: _opd_retry_contract_version,
        _PROFILE_WALL_ARMED_AT_KEY: _profile_wall_armed_at,
        _PROFILE_ATTEMPT_FLOOR_KEY: _profile_attempt_floor,
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


from flash.runner.artifacts import (  # noqa: E402,F401
    _assign_managed_hf_repo,
    _assign_resolved_env_sha,
    _environment_artifact_repo_name,
    _file_digest,
    artifact_namespace,
    flash_code_prefix,
    managed_hf_repo_for_environment,
)
from flash.runner.attempts import (  # noqa: E402,F401
    _heartbeat_attempt_is_current,
    _infer_next_attempt,
    _latest_reserved_attempt,
    _reserve_attempt,
    _verified_opd_next_attempt,
    _verified_opd_retry_state,
)
from flash.runner.costs import (  # noqa: E402,F401
    _gpu_rate,
    _status_estimated_charge,
    actual_steps_run,
    cancelled_charge_usd,
    charge_usd_for_spec,
    profile_steps_run,
    record_billing_state,
    record_realized_cost,
)
from flash.runner.deadlines import (  # noqa: E402,F401
    _canonical_run_deadline,
    _checked_stored_run_deadline,
    _load_run_deadline_at,
    _remaining_run_wall_seconds,
    _require_valid_deadline,
    _spec_with_remaining_wall,
    _worker_deadline_at,
)

# redundant-alias form: this is a pure re-export for the sibling modules that resolve the name as
# `runner._profile_wall_armed_at`, and `_save_status` below takes a keyword argument of the same
# name, so a plain import would read as a redefinition of it.
from flash.runner.deadlines import _profile_wall_armed_at as _profile_wall_armed_at  # noqa: E402
from flash.runner.preparation import (  # noqa: E402,F401
    _adopted_warmstart_revision,
    _assign_profile_environment_content_sha,
    _inherit_warmstart_revision,
    _mark_warmstart_source,
    _preparation_digest,
    _prepare_init_from_adapter,
    _prepare_init_from_adapter_inner,
    _prepared_before_environment_content_sha,
    _prepared_before_public_alpha,
    _prepared_sft_profile_job,
    _profile_producer_version,
    _require_pinned_profile_environment,
    _require_sft_workload_profile,
    _require_supported_adapter_continuation,
    _resolve_model_revision,
    _validate_effective_spec,
    _warmstart_source_is_authorized,
)
from flash.runner.reconciliation import (  # noqa: E402,F401
    _canonical_cleanup_remote,
    _cleanup_remote_key,
    _cleanup_remotes_from_raw,
    _compare_and_clear_remote,
    _compare_and_complete_remote,
    _compare_and_fail_remote,
    _compare_and_prepare_resubmit,
    _compare_and_remove_cleanup_remote,
    _drain_cleanup_remotes,
    _expected_remote_matches,
    _preserve_cleanup_remote,
    _record_cleanup_remote,
    _remote_resource_identity,
    _snapshot_cleanup_remotes,
)
from flash.runner.results.verified_revisions import (  # noqa: E402,F401
    add_verified_adapter_revision,
    invalidate_verified_adapter_revisions,
    read_verified_adapter_revisions,
    verified_adapter_revision_generation,
)
from flash.runner.status import (  # noqa: E402,F401
    _STATUS_LIST_LIMIT,
    _STATUS_METRICS_HISTORY_LIMIT,
    _load_status_json,
    _persist_metrics,
    _runstatus_from_json,
    _sanitize_status_value,
    _update,
    effective_spec_from_status,
    get_logs,
    get_status,
    list_run_ids,
    list_runs,
    reallocation_spec_from_status,
    record_heartbeat,
)
from flash.runner.submit import (  # noqa: E402,F401
    _persist_effective_worker_spec,
    _persist_profile_submission,
    _reject_managed_volume_removal,
    prepare_job,
    submit_job,
)
from flash.runner.supervise.deploy import (  # noqa: E402,F401
    DeploymentRevocationError,
    DeploymentStatePersistenceError,
    attach_run,
    cancel_run,
)
from flash.runner.supervise.lifecycle import (  # noqa: E402,F401
    _run_job,
    _run_job_inner,
    _run_training,
    _spec_with_gpu,
    _submit_seed_supervised,
)
from flash.runner.supervise.recovery import _gc_run_endpoints  # noqa: E402,F401
from flash.runner.supervise.transitions import (  # noqa: E402,F401
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_failed,
    mark_deployment_pending,
    mark_deployment_revocation_failed,
    mark_deployment_undeployed,
    mark_undeployed,
)
from flash.runner.weight_cache import (  # noqa: E402,F401
    _assign_weight_cache_volume,
    _download_gb,
    _fits_weight_cache,
    _peak_gb,
    weight_cache_catalog_peak_gb,
)
