"""Platform runner: drives managed RunPod GPUs, one allocation per run."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

try:
    import fcntl
except ImportError:  # pragma: no cover - linux production fails closed below
    fcntl = None

from flash._paths import data_dir
from flash.catalog import ModelInfo, resolve_model
from flash.opd_retry_contract import (
    OPD_RETRY_CONTRACT_STATUS_KEY,
    OPD_RETRY_CONTRACT_VERSION,
    require_opd_retry_contract_version,
)
from flash.providers._poll import _MAX_ATTEMPT_ID, _attempt_int
from flash.spec import MANAGED_GPU_KEYS, TRAINER_BACKEND, GpuSpec, JobSpec, gpu_count_of

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
# attempt gets its own IN_QUEUE grace (300s) and the infra retry floor allows several of them, so a
# 600s deadline measured from submission cannot survive even two capacity cycles: the run dies "run
# wall deadline exceeded" having profiled nothing, on hardware it never got. queue time gets this
# separate explicit allowance and the wall itself starts at the first heartbeat, so the quote stays
# wall x hourly (see estimate_profile_cost) rather than paying for the queue.
_WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS = 30 * 60


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
    max_wall_seconds, run_id, ...); their authoritative values are persisted verbatim in the internal
    worker spec under effective_preparation (recorded for every provisioned run). Prefer that
    carrier; fall back to the public spec for runs recorded before an effective worker spec exists,
    where those fields carry their managed defaults.
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
    targets a saved checkpoint).

    hf_repo, the control-plane-assigned artifact repo that signals the adapter exists, is
    platform-managed and read from the internal worker spec (see _internal_spec_from_status); run_id
    comes from the RunStatus itself.
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


def _gpu_rate(gpu_type: str, provider: str = "") -> float:
    """Static representative $/hr for cost projection.

    Prices on the provider that actually ran the job when it is known; provider rates for the
    same class differ (a RunPod-priced table misreports a Lambda or Vast run). Falls back to any
    configured provider that offers the class, so a plane without RunPod still prices its runs.

    Never raises: this feeds cost ANNOTATION on an already-finished run, so a provider-registry
    problem must degrade to the flat estimate rather than fail the metrics write.
    """
    try:
        from flash.providers import available_providers, get_provider

        # the billing substrate first when known, then any other configured provider that offers
        # the class -- so a plane without RunPod still prices its runs.
        names = [provider.strip().lower()] if provider.strip() else []
        names += [n for n in available_providers() if n not in names]
    except Exception:
        return 0.80
    for name in names:
        try:
            rate = get_provider(name).hourly_rate(gpu_type)
        except Exception:
            continue
        if rate:
            return float(rate)
    return 0.80


def charge_usd_for_spec(spec, *, steps: int | None = None, fallback: float = 0.0) -> float:
    """Return the estimated customer charge, prorated by completed steps when requested."""
    try:
        from flash.cost.analytical import estimate_cost
        from flash.cost.spec import estimate_for_spec, runconfig_from_spec

        if getattr(spec, "workload_profile_kind", ""):
            # a profile job has no optimizer steps to prorate, so its charge is all-or-nothing: the
            # bounded wall it rented, or zero if it never started. the caller passes steps=0 for the
            # latter (see profile_steps_run), and honouring it matters because the id is derived
            # from the workload rather than the account -- a profile cancelled before launch would
            # otherwise bill the full wall cap to whichever submitter happened to win the claim.
            if steps is not None and int(steps) <= 0:
                return 0.0
            return float(estimate_for_spec(spec).total_usd)
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


def profile_steps_run(status: RunStatus) -> int:
    """Whether a cancelled profile job rented anything: 1 if it started, 0 if it never did.

    A profile has no optimizer steps, so ``actual_steps_run`` reads 0 for every one of them -- it
    looks for training-stage heartbeats a profile never emits. Billing a cancel on that number
    would hand back the rented wall of a profile that ran to completion. Billing the quote
    unconditionally has the opposite failure: a profile cancelled while still queued would charge
    the whole wall cap for work no gpu did, and because the id is derived from the workload rather
    than the account, that charge lands on whichever submitter won the claim.

    The distinguishing signal is that a worker spoke, but on a RELAUNCH the stored word may not be
    this lifecycle's: a profile's run id is derived from the workload, so a relaunch reuses it, and
    ``record_heartbeat`` keeps whatever arrives under it for visibility while refusing to arm the
    wall from a heartbeat whose provenance it rejected. Billing the stored stage there charges a
    relaunch cancelled in the queue for a machine it never rented. So a relaunch -- and only a
    relaunch, marked by the attempt floor its takeover records -- is billed on the arm, which is
    written only for a heartbeat that passed ``_heartbeat_attempt_is_current``. A first lifecycle
    has no earlier worker to be confused with and bills on the stored word as before.
    The charge is all-or-nothing rather than prorated because a profile is quoted as a wall cap,
    not a per-step price -- see ``charge_usd_for_spec``."""
    hb = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else {}
    if not hb.get("stage"):
        return 0
    try:
        raw = _load_status_json(status.run_id)
    except (FileNotFoundError, ValueError):
        return 1
    if _PROFILE_ATTEMPT_FLOOR_KEY not in raw:
        return 1
    return 1 if _profile_wall_armed_at(raw) is not None else 0


def _require_valid_deadline(value: object) -> float:
    """Return a finite positive unix deadline or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0:
        raise RuntimeError("run wall deadline is invalid; no further provisioning is allowed")
    return deadline


def _profile_wall_armed_at(raw: dict) -> float | None:
    """Return when this profile's work budget started, or None if it has not started yet.

    Absent means no worker has spoken for this profile yet, which is the normal state while it
    queues. A stored value is validated like any other deadline input: a corrupt one fails closed
    rather than silently reverting to the submission basis and shortening the budget."""
    if _PROFILE_WALL_ARMED_AT_KEY not in raw:
        return None
    return _require_valid_deadline(raw[_PROFILE_WALL_ARMED_AT_KEY])


def _canonical_run_deadline(raw: dict) -> tuple[RunStatus, float]:
    status = _runstatus_from_json(raw)
    # max_wall_seconds is platform-managed and stripped from the public status.spec, so source the
    # run-global wall budget from the internal worker spec (the same value submit recorded).
    spec = _internal_spec_from_status(status)
    created_at = _require_valid_deadline(status.created_at)
    max_wall_seconds = _require_valid_deadline(spec.gpu.max_wall_seconds)
    if spec.workload_profile_kind:
        # a profile's wall budget bounds its WORK. before a worker speaks, the run is still waiting
        # on capacity, so it holds the queue allowance ON TOP of its untouched work budget; once one
        # speaks, the work budget runs from that moment and the remaining queue allowance is
        # dropped. the basis is recomputed from persisted state (never from the wall clock), so this
        # stays a pure function of the record and _checked_stored_run_deadline still validates it.
        armed_at = _profile_wall_armed_at(raw)
        basis = (
            created_at + _WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS if armed_at is None else armed_at
        )
        return status, _require_valid_deadline(basis + max_wall_seconds)
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


def _worker_deadline_at(run_id: str, spec: JobSpec, *, now: float | None = None) -> float:
    """Return the absolute deadline the worker may enforce for this launch.

    The persisted run deadline is submission-to-terminal and, for an unarmed profile, still holds
    the queue allowance on top of the work budget. The bootstrap enforces whatever absolute
    deadline it is handed (see ``_worker_execution_deadline``) independently of max_wall_seconds,
    so passing the run-global one lets a profile that got capacity immediately work through the
    queue window on a job priced for its wall alone. Bound it to the work budget from launch, so
    the deadline the worker enforces matches the wall ``_spec_with_remaining_wall`` grants.

    Once armed, the persisted deadline is already work-budget-from-arm, and it is the authority:
    taking the min keeps a relaunched or slow-to-speak worker from extending past it.

    While UNARMED the stored deadline is not a ceiling on the work, because it still carries the
    queue allowance that arming discards: it runs to created_at + queue + work, so once the wait
    passes the allowance the remainder is SHORTER than the work budget. Taking the min there hands
    the worker whatever is left of a window measured from submission -- at a 2100s wait, 300s of a
    600s budget -- while `_spec_with_remaining_wall` grants the provider a full one and the first
    heartbeat expands the plane's own deadline to armed_at + work. The worker never learns of that
    expansion, so it would die mid-measurement on exactly the slow-capacity day the allowance
    exists to survive. Unarmed, the work budget from launch is the authority.
    """
    stored = _load_run_deadline_at(run_id)
    if not spec.workload_profile_kind:
        return stored
    current = time.time() if now is None else now
    work_budget_at = float(current) + float(_WORKLOAD_PROFILE_WALL_SECONDS)
    if _profile_wall_armed_at(_load_status_json(run_id)) is None:
        return work_budget_at
    return min(stored, work_budget_at)


def _spec_with_remaining_wall(
    spec: JobSpec,
    *,
    require_provider_minimum: bool,
    now: float | None = None,
) -> JobSpec:
    """Copy a spec with only the run-global wall allowance still available."""
    remaining = _remaining_run_wall_seconds(spec.run_id, now=now)
    # exhaustion is judged on the REAL remaining allowance, before any profile substitution below.
    # a profile's grant replaces `remaining` outright, so deferring this check past that assignment
    # would make it unreachable for profiles and let a run provision after its own deadline had
    # passed -- and the first heartbeat would then arm a fresh work window from that moment,
    # turning the bounded queue allowance into an unbounded one.
    if remaining <= 0:
        raise RuntimeError("run wall deadline exhausted; no further provisioning is allowed")
    if spec.workload_profile_kind:
        # an unarmed profile's remaining allowance still holds the queue budget, which exists to
        # outlast capacity waits -- not to be spent working. handing it to the worker would let a
        # profile that got capacity immediately run for the whole queue budget too, on a job billed
        # for its wall alone (estimate_profile_cost prices wall x hourly).
        #
        # grant the WORK budget flat rather than min(remaining, work): `remaining` is measured
        # against a deadline that still contains the unspent queue allowance, so once the queue wait
        # passes that allowance the min() starts truncating. at a 1900s wait the provider would get
        # 500s while the plane grants a full 600s the moment a heartbeat arms -- the shorter number
        # goes to the side actually doing the work, killing the profile mid-measurement on exactly
        # the slow-capacity days the queue allowance exists to survive.
        #
        # the run-global deadline still bounds the work: _worker_deadline_at hands the worker
        # min(stored, now + work_budget), so this grant sets the wall, not a licence to outlive it.
        remaining = float(_WORKLOAD_PROFILE_WALL_SECONDS)
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


def _heartbeat_attempt_is_current(hb: object, raw: dict) -> bool:
    """True when a heartbeat carries the attempt identity this run most recently reserved.

    The plane-side half of ``_heartbeat_matches_attempt``. That one runs provider-side where the
    launch timestamp is in hand; here the equivalent identity is the reserved attempt, which the
    worker stamps on every heartbeat and ``_save_status`` already persists as ``next_attempt``
    (the NEXT id to hand out, so the live attempt is one below it -- same arithmetic as
    ``_latest_reserved_attempt``, computed from the caller's already-loaded record because this runs
    inside the status guard and must not re-read it).
    """
    if not isinstance(hb, dict):
        return False
    try:
        next_attempt = _attempt_int(_infer_next_attempt(raw))
    except RuntimeError:
        return False
    if next_attempt is None:
        return False
    # `_reserve_attempt` runs before the provider launch (lifecycle.py), so a live worker's
    # heartbeat always sits one below the stored counter. Zero means nothing has been reserved yet;
    # accept attempt 0 there rather than rejecting, because the launch path writes the counter and
    # the worker's first heartbeat can be read back in either order, and refusing to arm would hand
    # the run a budget measured from a moment before it started working.
    expected = next_attempt - 1 if next_attempt > 0 else 0
    if _attempt_int(hb.get("attempt")) != expected:
        return False
    # ...and it must belong to THIS lifecycle. a relaunch reuses the run id and carries the counter,
    # so until it reserves an attempt of its own, `expected` still names the SPENT lifecycle's --
    # and a prior worker that outlived its record stamps exactly that, recently enough to pass every
    # other check. the floor is that carried counter, so a heartbeat below it predates this run.
    floor = _attempt_int(raw.get(_PROFILE_ATTEMPT_FLOOR_KEY))
    return floor is None or expected >= floor


def _verified_opd_retry_state(run_id: str) -> tuple[int, str | None]:
    """Verify one locked opd retry snapshot and return its attempt plus resume revision."""
    with _status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        # hf_repo is platform-managed and stripped from the public status.spec; the opd replacement
        # locates its resume checkpoint by hf_repo, so source the complete internal worker spec.
        spec = _internal_spec_from_status(status)
        if spec.algorithm != "opd":
            raise RuntimeError("opd retry verification requires an opd run")
        try:
            contract_version = require_opd_retry_contract_version(raw.get(_OPD_RETRY_CONTRACT_KEY))
        except ValueError as exc:
            raise RuntimeError(
                "opd retry contract is missing or invalid; replacement is blocked"
            ) from exc
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


def _latest_reserved_attempt(run_id: str) -> int | None:
    """Return the newest durably reserved attempt, or none before any reservation."""
    try:
        raw = _load_status_json(run_id)
        next_attempt = _infer_next_attempt(raw)
    except Exception:
        return None
    return next_attempt - 1 if next_attempt > 0 else None


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
                train.pop("lora_rank", None)
            data["train"] = train
    _redact_internal_adapter_ref(data)
    return data


def _redact_internal_adapter_ref(data: dict) -> None:
    """Never surface an internal storage locator in the public spec.

    A worker/effective or legacy record can persist ``train.init_from_adapter`` as the internal
    storage ref ``<hf_repo>:<phase>/<run_id>[/checkpoints/step-N]``, which embeds the private HF
    repo. Rewrite it back to the user-facing checkpoint ref (``<run_id>[/step-N]``).

    A ref is published only when it is PROVEN user-facing, never merely because this build failed
    to parse it as internal. Those are different claims: a persisted locator whose phase this build
    no longer knows (``opsd``, removed in #784) stops parsing as internal, and inferring "public"
    from that published the private repo verbatim (chatgpt-codex-connector). Unrecognized shapes are
    dropped, which is what the malformed-prefix branch below has always done.
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


def artifact_namespace() -> str:
    """The HuggingFace namespace run artifacts are created under.

    Flash streams code, checkpoints and adapters through HF dataset repos that the control plane
    CREATES, so the namespace has to be one the operator's ``HF_TOKEN`` can write to. Hardcoding
    Freesolo's made self-hosting impossible: ``_assign_managed_hf_repo`` runs on every submit, and
    a self-hoster's token cannot create ``Freesolo-Co/flashrun-*``, so the run failed at upload
    before any training started.

    ``FLASH_HF_NAMESPACE`` overrides it (a user or an org). Defaults to the managed namespace, so
    the hosted deployment is unaffected.
    """
    return (os.environ.get("FLASH_HF_NAMESPACE") or "").strip() or _DEFAULT_ARTIFACT_NAMESPACE


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
    return f"{artifact_namespace()}/{_environment_artifact_repo_name(env_id)}"


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
# Must stay >= weight_cache_catalog_peak_gb(), which is what the whole catalog actually needs on one
# shared volume; test_volume_holds_whole_catalog_with_largest_model_in_transit fails if it does not.
# Grow this (and the provisioned volumes) rather than dropping a model from the cache.
WEIGHT_CACHE_VOLUME_GB = 250
# A download needs the checkpoint plus roughly as much again for Xet reconstruction scratch.
_WEIGHT_CACHE_PEAK_FACTOR = 2.0


def _download_gb(info: ModelInfo) -> float:
    """Full bf16 checkpoint size in GB, from catalog geometry (2 bytes/param).

    Same rule as ``cost.facts.download_weight_gb``, which cannot be reused here: it resolves a model
    *id* and fail-closes on anything off-catalog, while cache sizing runs on a ``ModelInfo`` that may
    legitimately carry no ``params_b``.
    """
    return (info.params_b or 0.0) * 2.0


def _peak_gb(info: ModelInfo) -> float:
    """GB the volume must have free for this model to finish downloading."""
    return _WEIGHT_CACHE_PEAK_FACTOR * _download_gb(info)


def _fits_weight_cache(info: ModelInfo) -> bool:
    """Whether the model's peak download footprint fits the shared weight-cache volume.

    Sizes the model against an EMPTY volume. That is the right question for "may this model use
    the cache at all", but it is NOT sufficient to prove the whole catalog can be warmed onto one
    volume -- the cache is shared and never evicted, so a later model meets a volume already
    holding every earlier one. ``weight_cache_catalog_peak_gb`` answers that cumulative question;
    keep both in agreement when adding a large model.
    """
    if not info.params_b:
        return (
            True  # unknown size -> keep the (attach) default; curated catalog models always set it
        )
    return _peak_gb(info) <= WEIGHT_CACHE_VOLUME_GB


def weight_cache_catalog_peak_gb() -> float:
    """Peak GB the volume must hold to warm the WHOLE catalog onto one shared volume.

    Every catalog model ends up resident together and nothing is ever evicted, so the worst moment
    is the largest model downloading -- needing room for its scratch -- while all the others are
    already on disk. That is strictly more than any single model's own peak, which is why sizing the
    volume per-model let the 35B fail with "Disk quota exceeded" on every datacenter at 200 GB.

    Derived from ``params_b``, so it slightly understates a repo that also ships tokenizer/config/
    index files; the measured-bytes figure for today's catalog is ~5 GB higher. Keep real headroom
    over this number rather than sizing to it exactly.
    """
    from flash.catalog import MODELS

    cached = [info for info in MODELS.values() if _fits_weight_cache(info)]
    if not cached:
        return 0.0
    largest = max(cached, key=_download_gb)
    resident_others = sum(_download_gb(info) for info in cached if info is not largest)
    return resident_others + _peak_gb(largest)


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
    pinned = existing == WEIGHT_CACHE_VOLUME_NAME
    # An already-pinned spec is only "correct" if it also carries the CURRENT managed size. A stale
    # or internally-round-tripped spec can hold the shared name at a previous, smaller size; taking
    # the no-op return there would deploy an undersized volume for models this size now admits.
    sized = getattr(spec.gpu, "network_volume_gb", None) == WEIGHT_CACHE_VOLUME_GB
    if attach == pinned and (sized or not attach):
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
    """prepare public and worker specs with source-authoritative adapter metadata.

    Failures here are genuinely about the warm-start source, so they are tagged
    ``WarmStartPreparationError`` for the submit route. Everything else in ``prepare_job`` (gpu
    sizing, budget, environment resolution) must keep its own message rather than be reported as a
    bad adapter, since those run for non-warm-start runs too and have nothing to do with the adapter.
    """
    try:
        return _prepare_init_from_adapter_inner(
            spec, owner_org_id=owner_org_id, owner_key_id=owner_key_id, token=token
        )
    except WarmStartPreparationError:
        raise
    except Exception as exc:
        raise WarmStartPreparationError(str(exc)) from exc


def _prepare_init_from_adapter_inner(
    spec: JobSpec,
    *,
    owner_org_id: str = "",
    owner_key_id: int | None = None,
    token: str | None = None,
) -> tuple[JobSpec, JobSpec, dict | None]:
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
            f"(a checkpoint listed by `flash runs checkpoint`); got {ref!r}"
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
    # hf_repo is platform-managed and stripped from the source run's public spec; its authoritative
    # value lives in that run's internal worker spec (see _internal_spec_from_status), which the
    # warm-start needs to locate the source adapter artifacts.
    src_spec = _internal_spec_from_status(src_status)
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
    worker_payload = worker_spec.to_internal_dict()
    # omit empty fields so existing version-1 snapshots keep their historical digest.
    for key in (
        "workload_profile_kind",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not worker_payload.get(key):
            worker_payload.pop(key, None)
    payload = {
        "version": 1,
        "public_spec": public_spec.to_dict(),
        "worker_spec": worker_payload,
        "adapter_identity": adapter_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_effective_spec(public_spec: JobSpec, worker_spec: JobSpec) -> None:
    public = public_spec.to_internal_dict()
    effective = worker_spec.to_internal_dict()
    # run_id and model_policy are platform-managed top-level fields stripped from the public spec, so
    # the reconstructed public spec carries only their defaults ("local"/"catalog"). exclude them from
    # the structural check; their integrity is covered by the sha256 preparation digest, and the
    # worker spec is already keyed by run_id at the persist boundary.
    for managed_top in (
        "run_id",
        "model_policy",
        "workload_profile_kind",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        effective[managed_top] = public.get(managed_top)
    public_train = dict(public["train"])
    effective_train = dict(effective["train"])
    public_ref = public_train.get("init_from_adapter") or ""
    internal_ref = effective_train.get("init_from_adapter") or ""
    for train_field in (
        "init_from_adapter",
        "init_from_adapter_revision",
        "lora_rank",
        "lora_alpha",
        # platform-managed artifact repo: stripped from the public spec (digest-protected), so the
        # reconstructed public spec carries only the default. exclude it from the structural check.
        "hf_repo",
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
    # gpu.count is a CEILING, and the allocator may satisfy it with fewer cards (2x of a class when
    # 4 was allowed). _spec_with_gpu writes the SELECTED count onto the worker spec -- the worker
    # sizes its rank count from it and the provider payload rents it -- so comparing it against the
    # authored ceiling would fail every narrowed run here, before any provider is reached. narrowing
    # only: a worker spec claiming MORE cards than the run authorized is a real integrity failure and
    # still raises. the exact selected count is digest-protected and persisted for launch.
    effective_count = int(effective_gpu.get("count", 1) or 1)
    public_count = int(public_gpu.get("count", 1) or 1)
    if 1 <= effective_count <= public_count:
        effective_gpu["count"] = public_gpu.get("count")
    # disk sizing, the weight-cache volume, and retry/wall-clock lifecycle policy are platform-managed
    # (MANAGED_GPU_KEYS) and stripped from the public spec, so the reconstructed public spec carries
    # only defaults for them. exclude them from the structural comparison; their integrity is covered
    # by the sha256 preparation digest, and the committed weight-cache volume is guarded against
    # illegitimate removal at the persist boundary (see _reject_managed_volume_removal).
    for managed_gpu in MANAGED_GPU_KEYS:
        effective_gpu[managed_gpu] = public_gpu.get(managed_gpu)
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


def _resolve_model_revision(spec: JobSpec, *, required: bool = False) -> JobSpec:
    authored = spec.model_revision
    if not authored and not required:
        return spec
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
            spec.model,
            revision=authored or None,
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


def _profile_producer_version() -> str:
    from flash import __version__

    return str(__version__)


def _require_pinned_profile_environment(spec: JobSpec) -> JobSpec:
    pinned = _assign_resolved_env_sha(spec)
    if not pinned.environment.id:
        raise WorkloadProfileUnavailable("sft workload profiling requires an environment id")
    if not pinned.environment.resolved_sha:
        raise WorkloadProfileUnavailable(
            "sft workload profiling requires an immutable resolved environment revision"
        )
    return pinned


def _prepared_sft_profile_job(spec: JobSpec, *, input_digest: str) -> PreparedJob:
    """Prepare the cpu-only profile job that measures ``spec``'s exact sft workload.

    Deliberately NOT ``prepare_job()``: that path prepares a *training* run, and every step of it
    is about weights this job never loads. Running it here would resolve revision-specific model
    geometry, reserve model-sized disk, attach the shared weight cache, prepare warm-start adapter
    continuation, and finally price the training shape -- which needs the very profile this job
    exists to produce. The profile loads a tokenizer and the pinned environment, then exits.

    What it does keep is the small set of things the worker genuinely needs: the deterministic
    run id, the immutable model/env revisions the caller already pinned, a bounded wall and retry
    policy, and the managed artifact repo the worker downloads its code prefix from and uploads
    metrics.json/DONE/console to. GPU type and provider pins are dropped: they describe where the
    training run must land, and inheriting them would rent an H100 to tokenize on cpu.
    """
    from flash.workload_profile import SFT_PROFILE_KIND, sft_profile_run_id

    profile_spec = replace(
        spec,
        run_id=sft_profile_run_id(input_digest),
        gpu=replace(
            spec.gpu,
            type="",
            provider="",
            count=1,
            disk_gb=GpuSpec.disk_gb,
            network_volume=None,
            max_wall_seconds=_WORKLOAD_PROFILE_WALL_SECONDS,
            max_retries=_WORKLOAD_PROFILE_MAX_RETRIES,
        ),
        workload_profile_kind=SFT_PROFILE_KIND,
        workload_profile_input_digest=input_digest,
        workload_profile_producer_version=_profile_producer_version(),
        workload_profile={},
    )
    profile_spec = _assign_managed_hf_repo(profile_spec)
    from flash.cost.spec import estimate_for_spec

    return PreparedJob(
        public_spec=profile_spec,
        worker_spec=profile_spec,
        estimated_cost_usd=float(estimate_for_spec(profile_spec).total_usd),
    )


def _require_sft_workload_profile(spec: JobSpec) -> JobSpec:
    """Attach the exact sft workload profile for ``spec``, or fail closed until one exists."""
    from flash.workload_profile import (
        SFT_PROFILE_KIND,
        require_matching_sft_profile,
        sft_profile_input_digest,
        sft_profile_run_id,
    )

    producer_version = _profile_producer_version()
    tokenizer_revision = spec.model_revision
    input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=tokenizer_revision,
        producer_version=producer_version,
    )
    if spec.workload_profile:
        profile = require_matching_sft_profile(
            spec.workload_profile,
            input_digest=input_digest,
            producer_version=producer_version,
            tokenizer_revision=tokenizer_revision,
        )
        return replace(
            spec,
            workload_profile_kind="",
            workload_profile_input_digest=input_digest,
            workload_profile_producer_version=producer_version,
            workload_profile=profile.to_dict(),
        )

    profile_run_id = sft_profile_run_id(input_digest)
    try:
        status = get_status(profile_run_id)
    except FileNotFoundError as exc:
        prepared = _prepared_sft_profile_job(spec, input_digest=input_digest)
        raise WorkloadProfilePending(
            profile_run_id,
            "required",
            prepared_job=prepared,
        ) from exc

    if (
        status.workload_profile_kind != SFT_PROFILE_KIND
        or status.workload_profile_input_digest != input_digest
    ):
        raise WorkloadProfileUnavailable("stored workload profile identity does not match")
    if status.state == "done":
        try:
            profile = require_matching_sft_profile(
                status.workload_profile,
                input_digest=input_digest,
                producer_version=producer_version,
                tokenizer_revision=tokenizer_revision,
            )
        except ValueError as exc:
            raise WorkloadProfileUnavailable(
                "stored sft workload profile failed integrity validation"
            ) from exc
        return replace(
            spec,
            workload_profile_input_digest=input_digest,
            workload_profile_producer_version=producer_version,
            workload_profile=profile.to_dict(),
        )
    if status.state in {"failed", "cancelled", "dry_run"}:
        # a spent profile is not a verdict on the workload. the id is derived from the workload
        # alone, so a preempted pod or a cancel would otherwise make this exact config unquotable
        # for everyone, forever, with nothing in the system that could ever clear it. offer a fresh
        # profile job the same way a missing one is offered; the claim decides who actually runs it.
        prepared = _prepared_sft_profile_job(spec, input_digest=input_digest)
        raise WorkloadProfilePending(
            profile_run_id,
            status.state,
            prepared_job=prepared,
            spent_at=status.created_at,
        )
    raise WorkloadProfilePending(profile_run_id, status.state)


def prepare_job(
    spec: JobSpec,
    *,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
) -> PreparedJob:
    """Prepare all read-only submission inputs before persistence or allocation."""
    spec = _resolve_model_revision(spec, required=spec.algorithm == "sft")
    _require_supported_adapter_continuation(spec)
    if spec.algorithm == "sft":
        spec = _require_pinned_profile_environment(spec)
        spec = _require_sft_workload_profile(spec)
    if spec.train.structured_outputs:
        from flash.serve.preflight import preflight_serving_path

        preflight_serving_path(spec)
    else:
        from flash.lora_rank import (
            ServingPreflightError,
            preflight_train_context_within_serving,
        )

        # mirror preflight_serving_path: surface the specific context error as a
        # ServingPreflightError so create_run re-raises it unchanged instead of the
        # warm-start path masking it with a generic preparation message
        try:
            preflight_train_context_within_serving(spec)
        except ValueError as exc:
            raise ServingPreflightError(str(exc)) from exc
    if spec.gpu.provider or spec.gpu.type:
        from flash.providers import PROVIDER_NAMES, available_providers
        from flash.providers.base import providers_for

        configured = available_providers()
        provider = spec.gpu.provider.strip().lower()
        if provider:
            if provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {spec.gpu.provider!r}")
            if provider not in configured:
                raise ValueError(f"requested gpu.provider {provider!r} is not configured")
        elif not any(name in configured for name in providers_for(spec.gpu.type)):
            raise ValueError(f"no configured provider can provision gpu.type {spec.gpu.type!r}")
    from flash.providers.allocator import geometry_safe_gpu_cap

    preflight_gpu_count = geometry_safe_gpu_cap(
        spec.model, gpu_count_of(spec), model_revision=spec.model_revision
    )
    preflight_gpu = spec.gpu.type
    if not preflight_gpu and spec.model_policy == "allow":
        # open-model auto runs size this fit preflight against the provisional class the schema
        # already validated against, not the empty public gpu.type: resolve_model ->
        # _resolve_open_model falls back to DEFAULT_GPU on empty, which would reject an uncatalogued
        # model larger than the default but fitting a managed class -- after it passed schema.
        from flash.providers.base import provisional_gpu

        preflight_gpu = provisional_gpu(
            spec.model,
            spec.algorithm,
            train=spec.train,
            thinking=spec.thinking,
            model_revision=spec.model_revision,
            # same card ceiling the allocator will honour, so this preflight cannot reject a shape
            # allocation would have accepted.
            gpu_count=preflight_gpu_count,
        )
    info = resolve_model(
        spec.model,
        spec.algorithm,
        policy=spec.model_policy,
        gpu=preflight_gpu,
        model_revision=spec.model_revision,
        # same ceiling the preflight class was chosen under, so this cannot reject a shape
        # allocation would have accepted.
        gpu_count=preflight_gpu_count,
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

    # profile jobs route to their bounded-wall charge inside estimate_for_spec; they cannot be priced
    # through the training estimator, which requires the profile this job produces.
    estimated_cost_usd = float(estimate_for_spec(worker_spec).total_usd)
    return PreparedJob(
        public_spec=public_spec,
        worker_spec=worker_spec,
        estimated_cost_usd=estimated_cost_usd,
        adapter_identity=adapter_identity,
    )


def _reject_managed_volume_removal(snapshot: object, worker_spec: JobSpec) -> None:
    """Fail closed if a re-prepared worker spec drops a non-shared weight-cache volume.

    network_volume is platform-managed and no longer travels in the public spec, so the committed
    volume lives only in the prior preparation snapshot. The SHARED platform cache
    (WEIGHT_CACHE_VOLUME_NAME) may be dropped on a capacity fallback, but a per-org escape-hatch
    volume an open-model run opted into must never be silently removed or swapped.
    """
    if not isinstance(snapshot, dict):
        return
    committed = ((snapshot.get("worker_spec") or {}).get("gpu") or {}).get("network_volume")
    if not committed or committed == WEIGHT_CACHE_VOLUME_NAME:
        return
    if worker_spec.gpu.network_volume != committed:
        raise ValueError("persisted effective preparation drops a non-shared weight-cache volume")


def _persist_effective_worker_spec(
    worker_spec: JobSpec, *, estimated_cost_usd: float | None = None
) -> bool:
    """Persist the selected worker spec and exact quote before provider provisioning starts."""
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
    _reject_managed_volume_removal(snapshot, worker_spec)
    _validate_effective_spec(public_spec, worker_spec)
    effective_preparation = {
        "worker_spec": worker_spec.to_internal_dict(),
        "workload_profile": worker_spec.workload_profile or None,
        "adapter_identity": adapter_identity,
        "preparation_digest": _preparation_digest(public_spec, worker_spec, adapter_identity),
        "backend": TRAINER_BACKEND,
    }
    fields = {"effective_preparation": effective_preparation}
    if estimated_cost_usd is not None:
        fields["estimated_cost_usd"] = float(estimated_cost_usd)
    return _update(worker_spec.run_id, status.state, **fields)


def _persist_profile_submission(status: RunStatus, save_kwargs: dict) -> RunStatus | None:
    """Write a profile's submission record, returning a live run to join instead of restarting.

    A profile's run id is derived from the workload rather than the account, so this id is reused
    by design and the record it writes may not be the first under it.
    """
    with _status_guard(status.run_id):
        raw_existing = (
            _load_status_json(status.run_id)
            if os.path.exists(runs_file_path(status.run_id, ".json"))
            else None
        )
        existing = _runstatus_from_json(raw_existing) if raw_existing is not None else None
        # a live profile under this id is joined, never restarted: a concurrent submitter of the
        # same config lands here and must wait on the running one rather than launch a second
        # billed copy of identical work.
        if existing is not None and existing.state not in _UNDEPLOYABLE_STATES:
            return existing
        # a spent one is replaced. the caller only reaches this after winning the takeover on that
        # exact spent record, so overwriting it is the relaunch, not a lost update.
        if raw_existing is not None:
            # the RECORD is replaced but the ARTIFACTS are not: the reused id means this lifecycle
            # uploads to the HF prefix ({phase}/{run_id}) the spent one left behind, so two private
            # keys have to carry across the overwrite rather than restart with it.
            #
            # attempt identity stays globally monotonic. error_<phase>_attempt<N>.txt is
            # attempt-scoped, and _instance_poll treats a present one as THIS handle's crash
            # ("error files are attempt-scoped, so a present file already belongs to this exact
            # handle") -- sound only while an id never repeats. restarting at 0 hands the fresh run
            # the spent one's attempt-0 error file, and it dies job_failed seconds after launch,
            # deterministically, on hardware it never used.
            carried_attempt = _infer_next_attempt(raw_existing)
            save_kwargs["_next_attempt"] = carried_attempt
            # carrying the counter keeps the ids monotonic, but it also means that until THIS
            # lifecycle reserves one, `next_attempt - 1` still names the SPENT lifecycle's attempt.
            # a prior worker that outlived its record stamps exactly that id, and its heartbeats are
            # genuinely recent, so the provenance check would accept one and arm this run's work
            # budget while it is still queuing for a machine. record the carried counter as this
            # lifecycle's floor: every attempt below it belongs to the run that already ended.
            save_kwargs["_profile_attempt_floor"] = carried_attempt
            # the wall, by contrast, must NOT carry: an arm records that a worker spoke, and that
            # worker was the previous lifecycle's. inheriting it dates this run's budget to a
            # heartbeat predating its own submission -- and since _canonical_run_deadline rebuilds
            # the deadline from that basis, the stored pair stops matching and every read fails the
            # tamper check, wedging this workload's profile id for every submitter. None drops the
            # stored key rather than carrying it forward.
            save_kwargs["_profile_wall_armed_at"] = None
        _save_status_unlocked(status, **save_kwargs)
    return None


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
    """Submit a prepared job, allocating resources only outside dry-run mode.

    A missing sft workload profile propagates as ``WorkloadProfilePending`` rather than being
    launched from here. Launching a profile requires claiming its deterministic id FIRST
    (``db.claim_profile_run`` / ``db.reclaim_spent_profile_run``), because the id is derived from
    the workload rather than the account: without the claim two submitters of the same config both
    launch, the work is profiled and billed twice, and the takeover that unwedges a spent profile
    loses the ordering it compares against. That claim lives in the server db, which this module
    deliberately does not depend on, so the caller that owns the key performs it -- see
    ``flash/server/routes/runs.py``, which claims and only then submits.
    """
    if prepared_job is not None:
        prepared = prepared_job
    else:
        prepared = prepare_job(
            spec,
            billing_context=billing_context,
            platform_context=platform_context,
            owner_key_id=owner_key_id,
        )
    public_spec = prepared.public_spec
    worker_spec = prepared.worker_spec
    estimated_cost_usd = prepared.estimated_cost_usd
    from flash.multimodal import preflight_validate_image_opd

    preflight_validate_image_opd(worker_spec)
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
        workload_profile_kind=worker_spec.workload_profile_kind or None,
        workload_profile_input_digest=worker_spec.workload_profile_input_digest or None,
        workload_profile=worker_spec.workload_profile or None,
        effective_preparation={
            "worker_spec": worker_spec.to_internal_dict(),
            "workload_profile": worker_spec.workload_profile or None,
            "adapter_identity": prepared.adapter_identity,
            "preparation_digest": _preparation_digest(
                public_spec, worker_spec, prepared.adapter_identity
            ),
            "backend": TRAINER_BACKEND,
        },
        # Snapshot the instance providers available at submit so a later handle-less recovery can fail
        # closed for any phantom-capable one whose creds were since dropped (see _confirm_run_clear).
        # Creds-only check (available_providers -> is_configured), no network on the create path.
        submitted_instance_providers=[n for n in available_providers() if n in INSTANCE_PROVIDERS],
    )
    save_kwargs = {
        "_run_deadline_at": (
            status.created_at
            + float(public_spec.gpu.max_wall_seconds)
            + (
                _WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS
                if worker_spec.workload_profile_kind
                else 0.0
            )
        ),
        "_next_attempt": 0,
        "_opd_retry_contract_version": (
            OPD_RETRY_CONTRACT_VERSION if public_spec.algorithm == "opd" else _PRIVATE_VALUE_UNSET
        ),
    }
    if worker_spec.workload_profile_kind:
        joined = _persist_profile_submission(status, save_kwargs)
        if joined is not None:
            return joined
    else:
        _save_status(status, **save_kwargs)
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
    has_workload_profile = bool(
        worker_spec.workload_profile_kind
        or worker_spec.workload_profile_input_digest
        or worker_spec.workload_profile
    )
    if has_workload_profile:
        if snapshot.get("workload_profile") != (worker_spec.workload_profile or None):
            raise ValueError("persisted workload profile does not match the worker spec")
        if not isinstance(stored_digest, str) or stored_digest != _preparation_digest(
            public_spec, worker_spec, expected
        ):
            raise ValueError("persisted effective preparation failed integrity validation")
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


def reallocation_spec_from_status(status: RunStatus, *, verify_source: bool = False) -> JobSpec:
    """Effective worker spec for RE-ALLOCATING a recovered run.

    Identical to effective_spec_from_status, except gpu.type is restored to the run's original
    public value: empty for an auto run, the pinned class for a pinned run. The persisted effective
    snapshot bakes the *allocated* class into gpu.type via _spec_with_gpu, so feeding it straight
    back to allocate() would hard-pin an originally-unpinned run to the prior attempt's class after a
    control-plane restart or attach -- blocking OOM escalation and retries on other providers/classes.
    Use this only where recovery re-enters allocate(); polling a live attempt and endpoint cleanup
    keep the concrete effective spec.
    """
    worker_spec = effective_spec_from_status(status, verify_source=verify_source)
    public_type = JobSpec.from_dict(status.spec).gpu.type
    if worker_spec.gpu.type == public_type:
        return worker_spec
    restored = worker_spec.to_internal_dict()
    restored["gpu"] = {**restored["gpu"], "type": public_type}
    return JobSpec.from_dict(restored)


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


_STATUS_LIST_LIMIT = 16
_STATUS_METRICS_HISTORY_LIMIT = 1024


def _sanitize_status_value(value, *, depth: int = 0, field: str = ""):
    """Bound a heartbeat payload before persisting it in run status JSON."""
    if depth > 5:
        return str(value)[:200]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        if field == "metrics_last":
            values = value[-_STATUS_METRICS_HISTORY_LIMIT:]
        else:
            values = value[:_STATUS_LIST_LIMIT]
        return [_sanitize_status_value(v, depth=depth + 1) for v in values]
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 64:
                out["truncated"] = True
                break
            sanitized_key = str(k)[:120]
            out[sanitized_key] = _sanitize_status_value(v, depth=depth + 1, field=sanitized_key)
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
        # first word from a profile's worker starts its work budget. the arm and the deadline it
        # implies are written together under this guard so a reader never sees one without the
        # other -- _checked_stored_run_deadline rejects that pair as tampering and halts the run.
        #
        # only a heartbeat from THIS run may arm it. a profile's run id is derived from the
        # workload, so a relaunch reuses the id and its artifact prefix, and the first heartbeat
        # read back can be the previous lifecycle's leftover (observed: a 2.8-hour-old one arming a
        # 5-second-old run). the same provenance requirement _heartbeat_matches_attempt enforces
        # provider-side, applied to the boundary available here.
        arm_kwargs: dict[str, float] = {}
        raw = _load_status_json(run_id)
        armed_spec = _internal_spec_from_status(status)
        if armed_spec.workload_profile_kind and _profile_wall_armed_at(raw) is None:
            hb_ts = hb.get("ts") if isinstance(hb, dict) else None
            fresh = (
                not isinstance(hb_ts, bool)
                and isinstance(hb_ts, (int, float))
                and math.isfinite(float(hb_ts))
                and float(hb_ts) >= _require_valid_deadline(status.created_at)
                # ...and it must be THIS attempt. a timestamp test alone admits a still-live worker
                # from a cancelled earlier lifecycle: it writes to the same workload-derived prefix
                # and its heartbeats are genuinely recent, so they pass `>= created_at` and arm the
                # replacement's work budget while it is still queuing for capacity.
                and _heartbeat_attempt_is_current(hb, raw)
            )
            if fresh:
                armed_at = time.time()
                arm_kwargs = {
                    "_profile_wall_armed_at": armed_at,
                    "_run_deadline_at": _require_valid_deadline(
                        armed_at + _require_valid_deadline(armed_spec.gpu.max_wall_seconds)
                    ),
                }
        # Checkpoint-stage heartbeats (checkpoint_uploading/deployable/uploaded) omit metrics_last; carry
        # the existing per-step backlog forward so `flash runs log -f` doesn't drop it mid-save until the next
        # metrics-bearing heartbeat lands.
        if isinstance(hb, dict) and not hb.get("metrics_last"):
            prev = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else None
            prev_metrics = prev.get("metrics_last") if isinstance(prev, dict) else None
            # only carry the backlog forward within the same attempt; a boot/retry heartbeat for a
            # new attempt must not inherit the prior attempt's stale per-step metrics.
            same_attempt = prev is not None and prev.get("attempt") == hb.get("attempt")
            if same_attempt and isinstance(prev_metrics, list) and prev_metrics:
                hb["metrics_last"] = prev_metrics
        status.last_heartbeat = hb
        status.gpu_status = gpu if isinstance(gpu, dict) else None
        status.updated_at = time.time()
        _save_status_unlocked(status, **arm_kwargs)
    _report_status(status)


def _persist_metrics(spec: JobSpec, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id> and return the customer training cost.

    The run id keeps concurrent/sequential runs of the same phase from
    overwriting each other's artifacts. ``metrics["wall_seconds"]`` is the worker's training-loop
    wall time; setup/cold-start is reported separately and is not included here."""
    from flash.engine.accounting import sanitize_worker_metrics

    metrics = sanitize_worker_metrics(metrics)
    if spec.workload_profile_kind:
        from flash.workload_profile import require_matching_sft_profile

        profile = require_matching_sft_profile(
            metrics.get("workload_profile"),
            input_digest=spec.workload_profile_input_digest,
            producer_version=spec.workload_profile_producer_version,
            tokenizer_revision=spec.model_revision,
        )
        metrics = {**metrics, "workload_profile": profile.to_dict()}
        current = get_status(spec.run_id)
        if current.state in TERMINAL_STATES:
            raise ValueError("workload profile metrics arrived after the run became terminal")
        _update(
            spec.run_id,
            current.state,
            workload_profile=profile.to_dict(),
        )
    dest = artifacts_dir(spec)
    os.makedirs(dest, exist_ok=True)
    # Use allocated_gpu (worker-stamped) not spec.gpu.type; policy GPUs can be reallocated.
    gpu_type = metrics.get("allocated_gpu") or spec.gpu.type
    # the substrate that actually billed the run; empty on a record predating the stamp, in which
    # case _gpu_rate prices off whichever configured provider offers the class.
    provider = str(metrics.get("allocated_provider") or "")
    rate = _gpu_rate(gpu_type, provider)
    # `hourly_rate` is per CARD, so a sharded run costs the wall times the rate times the number of
    # cards it actually occupied. `allocated_gpu_count` is worker/lifecycle-stamped for the same
    # reason `allocated_gpu` is: the spec's gpu.count is only a ceiling and allocation may pick
    # fewer. Absent on records predating the stamp, where one card is the correct reading.
    gpu_count = max(1, int(metrics.get("allocated_gpu_count") or 1))
    cost = metrics.get("cost_usd")
    if cost:
        cost = float(cost or 0.0)
    else:
        wall = float(metrics.get("wall_seconds") or 0.0)
        cost = wall / 3600.0 * rate * gpu_count
        metrics = {**metrics, "cost_usd": cost}
        metrics.setdefault("notes", {})
        if isinstance(metrics["notes"], dict):
            metrics["notes"]["provider"] = provider or "unknown"
            metrics["notes"]["gpu_rate_usd_hr"] = rate
            metrics["notes"]["gpu"] = gpu_type
            metrics["notes"]["gpu_count"] = gpu_count
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


def _expected_remote_matches(current: object, expected: dict | None) -> bool:
    if expected is None:
        return current is None
    expected_identity = _remote_resource_identity(expected)
    return expected_identity is not None and _remote_resource_identity(current) == expected_identity


def _compare_and_clear_remote(run_id: str, expected_remote: dict) -> bool:
    """Clear only the nonterminal remote that still names the destroyed resource."""
    if _remote_resource_identity(expected_remote) is None:
        return False
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.remote = None
        status.updated_at = time.time()
        _save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def _compare_and_prepare_resubmit(
    run_id: str,
    expected_remote: dict | None,
    *,
    expected_state: str | None = None,
) -> bool:
    """Claim a nonterminal recovery launch only while its expected remote still owns the run."""
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if expected_state is not None and status.state != expected_state:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.state = "provisioning"
        status.updated_at = time.time()
        _save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


def _compare_and_fail_remote(
    run_id: str,
    expected_remote: dict | None,
    error: str,
) -> bool:
    """CAS a nonterminal expected remote to failed and confirm the durable write."""
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        previous_updated_at = status.updated_at
        status.state = "failed"
        status.error = error
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at or previous_updated_at
        _save_status_unlocked(status)
        report_status = status
    confirmed = get_status(run_id)
    expected_after = expected_remote
    if (
        confirmed.state != "failed"
        or not _expected_remote_matches(confirmed.remote, expected_after)
        or confirmed.error != error
    ):
        raise RuntimeError("terminal recovery failure was not durably confirmed")
    if report_status is not None:
        _report_status(report_status)
    return True


def _compare_and_complete_remote(
    run_id: str,
    expected_remote: dict | None,
    spec: JobSpec,
    metrics: dict,
) -> bool:
    """Adopt strict completed artifacts only while the captured remote still owns the run."""
    report_status: RunStatus | None = None
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
    if expected_remote is not None and not _record_cleanup_remote(run_id, expected_remote):
        return False
    recovered_cost = _persist_metrics(spec, metrics)
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        measured = float(status.cost_usd or 0.0) + recovered_cost
        charge_usd = _status_estimated_charge(status, spec, fallback=measured)
        status.state = "done"
        status.cost_usd = charge_usd
        status.artifacts_dir = artifacts_dir(spec)
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at
        _save_status_unlocked(status)
        report_status = status
    confirmed = get_status(run_id)
    if confirmed.state != "done" or not _expected_remote_matches(confirmed.remote, expected_remote):
        raise RuntimeError("terminal recovery completion was not durably confirmed")
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
            resource_deleted = _strict_teardown_handle(JobHandle.from_dict(record), run_id)
        except Exception:
            continue
        if resource_deleted:
            with contextlib.suppress(Exception):
                _compare_and_remove_cleanup_remote(run_id, record)
    return attempted


def _record_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist one exact cleanup identity without changing the active remote."""
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
        status.updated_at = time.time()
        _save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        _report_status(report_status)
    return True


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


def _send_status_report(status: RunStatus) -> bool:
    from flash.server.run_registry import record_training_run

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
    _cleanup_remotes: list[dict] | None | object = _PRIVATE_VALUE_UNSET,
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
    _cleanup_remotes: list[dict] | None | object = _PRIVATE_VALUE_UNSET,
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
    invalidate_verified_adapter_revisions,
    read_verified_adapter_revisions,
    verified_adapter_revision_generation,
)
