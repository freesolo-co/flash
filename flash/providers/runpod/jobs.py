"""Durable run primitives: deploy -> submit -> poll with a persisted job handle."""

from __future__ import annotations

import asyncio  # noqa: F401
import base64
import contextlib
import json
import math
import time
from dataclasses import dataclass

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.providers._lifecycle.deadline import (
    deadline_kwargs,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.poll import (
    _attempt_int,
    heartbeat_oom_for_attempt,
    surface_heartbeat,
)
from flash.providers.artifacts.hf import (
    make_hf_failure_detail_reader,
    make_hf_heartbeat_reader,
    worker_flagged_retriable,
)
from flash.providers.base import (  # noqa: F401
    PollResult,
    UnreconciledCreateError,
    canonical_gpu,
)
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.gpus import flash_gpu  # noqa: F401
from flash.providers.runpod.serverless import (  # noqa: F401
    DEFAULT_EXECUTION_TIMEOUT_MS,
    FLASH_SDK_LOCK,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    _patch_runpod_backoff,
    _train_body,
    isolate_flash_state,
    min_cuda_for,
    resolve_worker_deps,
    worker_image_for_gpu,
)
from flash.providers.runpod.serverless import (
    endpoint_name as _endpoint_name,
)

endpoint_name = _endpoint_name
logger = get_logger(__name__)

# Per-part cap for sanitized terminal-failure text. The parts are already provider-bounded; the
# limit exists so redaction can never silently truncate a tail we chose to surface. It is applied
# after sanitizing the complete text, keeping the newest bytes.
FAILURE_TEXT_LIMIT = 64_000

# Re-export for callers that import PollResult from here.
__all__ = [
    "JobHandle",
    "PollResult",
    "apply_disk_gb",
    "build_function_input",
    "capacity_escalation_note",
    "decode_output",
    "deploy_train_endpoint",
    "grow_weight_cache_volumes",
    "make_hf_failure_detail_reader",
    "make_hf_heartbeat_reader",
    "poll_job",
    "queue_wait_note",
    "submit_run",
    "weight_cache_datacenters",
    "weight_cache_endpoint_kwargs",
    "weight_cache_grow_headroom_s",
    "weight_cache_volume_name",
    "weight_cache_volumes",
]


# capacity grace on the LAST candidate class: there is nowhere left to walk, so wait longer before
# giving up. purely a timing knob -- it says nothing about whether a retry follows, because
# on_last_gpu (runner/lifecycle.py) is also true when the infra retry budget is exhausted.
LAST_GPU_CAPACITY_GRACE_S = 900.0

# how long ONE "a worker is coming up" health reading keeps suppressing the capacity timer. probes
# run every 90s, so this absorbs a couple of missed or failed probes and no more: if health stops
# being readable entirely, the capacity timer re-arms rather than waiting out the run on an
# observation nobody can still confirm.
WORKER_COMING_UP_TTL_S = 300.0

# how often _classify_queue_state re-reads endpoint health, and therefore how often the
# unhealthy/throttled timers can fire at all. `queue_wait_note` rounds those graces up to this
# cadence when deciding which deadline to report: capacity, no-progress and the wall deadline are
# checked EVERY poll, so a health grace that runs out mid-gap is not acted on until the next probe
# and must not be quoted as if it were. Kept equal to the probe gate in _classify_queue_state.
HEALTH_PROBE_CADENCE_S = 90.0

TERMINAL_OK = {"COMPLETED"}
# CANCELLED/TIMED_OUT = provider-killed (retriable); FAILED = worker died on its own (fails fast).
PLATFORM_TERMINATIONS = {"CANCELLED", "TIMED_OUT"}
TERMINAL_FAIL = {"FAILED"} | PLATFORM_TERMINATIONS


def stall_kwargs(on_last_gpu: bool = False) -> dict:
    """poll_job stall-window kwargs. queue/throttled grace is ~5 min normally, ~15 min on last GPU (nowhere left to walk)."""
    grace = LAST_GPU_CAPACITY_GRACE_S if on_last_gpu else 300.0
    return {
        "stall_after_s": 1500.0,
        "setup_grace_s": 3000.0,
        "queue_grace_s": grace,
        "throttled_grace_s": grace,
    }


def apply_disk_gb(config, disk_gb: int | None) -> None:
    """Raise the worker's container disk on a built endpoint config. Raise-only (SDK default is 64 GB)."""
    if not disk_gb:
        return
    template = getattr(config, "template", None)
    if template is None:
        logger.warning("disk_gb=%s requested but endpoint config has no template", disk_gb)
        return

    template.containerDiskInGb = max(int(disk_gb), int(template.containerDiskInGb or 0))


@dataclass
class JobHandle:
    endpoint_id: str
    endpoint_name: str
    key_fingerprint: str
    job_id: str | None
    attempt: int
    started_ts: float

    def to_dict(self) -> dict:
        data = {
            "provider": "runpod",
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "key_fingerprint": self.key_fingerprint,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }
        if self.job_id is not None:
            data["job_id"] = self.job_id
        return data

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        if d.get("provider") != "runpod":
            raise ValueError("persisted RunPod provider identity is invalid")
        attempt = _attempt_int(d.get("attempt"))
        if attempt is None:
            raise ValueError("persisted RunPod attempt identity is invalid")
        started_raw = d.get("started_ts")
        if isinstance(started_raw, bool) or not isinstance(started_raw, (int, float)):
            raise ValueError("persisted RunPod launch timestamp is invalid")
        started_ts = float(started_raw)
        if not math.isfinite(started_ts) or started_ts <= 0:
            raise ValueError("persisted RunPod launch timestamp is invalid")
        endpoint_id = d.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError("persisted RunPod endpoint identity is invalid")
        endpoint_name = d.get("endpoint_name")
        if not isinstance(endpoint_name, str) or not endpoint_name:
            raise ValueError("persisted RunPod endpoint name is invalid")
        fingerprint = d.get("key_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 16
            or not fingerprint.startswith("rpk-")
            or any(char not in "0123456789abcdef" for char in fingerprint[4:])
        ):
            raise ValueError("persisted RunPod key fingerprint is invalid")
        job_id = d.get("job_id")
        if job_id is not None and (not isinstance(job_id, str) or not job_id):
            raise ValueError("persisted RunPod job identity is invalid")
        return cls(endpoint_id, endpoint_name, fingerprint, job_id, attempt, started_ts)


def _is_workers_quota_error(exc: Exception) -> bool:
    """True when a RunPod exception signals the account worker quota is exhausted."""
    msg = str(exc).lower()
    return "max workers across all endpoints" in msg


def _is_balance_error(exc: Exception) -> bool:
    """True when RunPod refuses to create an endpoint because the account is out of balance.

    Fail over immediately: another account may have funds, while sweeping cannot repair balance.
    """
    return "account balance" in str(exc).lower()


def build_function_input(payload: dict) -> dict:
    """The FunctionRequest dict a Flash queue worker expects for `_train_body(payload)`."""
    if WORKER_IMAGE:
        return payload
    from runpod_flash.runtime.serialization import serialize_args
    from runpod_flash.stubs.live_serverless import get_function_source

    source, _src_hash = get_function_source(_train_body)
    return {
        "function_name": "_train_body",
        "function_code": source,
        "args": serialize_args((payload,)),
        "accelerate_downloads": True,
        "dependencies": resolve_worker_deps(),
        "system_dependencies": WORKER_SYSTEM_DEPS,
    }


def _safe_failure_text(value: object, limit: int = FAILURE_TEXT_LIMIT) -> str:
    """Redact credentials out of one part of a user-visible RunPod failure detail.

    Provider errors and worker stdout tails reach the run log verbatim, so a control-plane secret
    the worker echoed would be printed. The instance providers sanitize every part of their failure
    detail; this keeps RunPod symmetric with them. The complete text is sanitized before the bound
    is applied, and the bound keeps the newest bytes: slicing first could cut a credential at the
    boundary so its surviving part no longer value-matches.
    """
    sanitized = sanitize_diagnostic(value, limit=1 << 30)
    return sanitized[-max(0, int(limit)) :]


def decode_output(output) -> dict:
    """Decode a queue-job output into the worker's metrics dict (handles live-function and baked-image shapes)."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"unexpected job output tail: {_safe_failure_text(output, 200)}"
            ) from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
    if "success" in output or "result" in output:
        if output.get("success") and output.get("result") is not None:
            import cloudpickle

            result = cloudpickle.loads(base64.b64decode(output["result"]))
            if not isinstance(result, dict):
                raise RuntimeError(f"flash job returned no metrics: {result!r}")
            return result
        err = output.get("error") or "unknown worker error"
        stdout_tail = _safe_failure_text(output.get("stdout") or "")
        raise RuntimeError(
            f"Remote execution failed: {_safe_failure_text(err)}\n"
            f"--- worker stdout tail ---\n{stdout_tail}"
        )
    if output.get("error"):
        stdout_tail = _safe_failure_text(output.get("stdout") or "")
        msg = f"Remote execution failed: {_safe_failure_text(output['error'])}"
        if stdout_tail:
            msg += f"\n--- worker stdout tail ---\n{stdout_tail}"
        raise RuntimeError(msg)
    return output


def _append_failure_artifacts(detail: str, failure_detail_reader) -> str:
    """Append worker-uploaded failure artifacts to a RunPod terminal-status detail."""
    if failure_detail_reader is None:
        return detail
    extra = failure_detail_reader(force=True)
    if not extra:
        return detail
    if detail:
        return f"{detail}\n{extra}"
    return extra


@dataclass
class GraceTimer:
    """Grace timer: arms on first active poll, expires after continuous active state.

    `since` excludes unobservable intervals; clearing it fully resets the timer.
    """

    since: float | None = None
    seen: float | None = None

    def expired(self, active: bool, now: float, grace: float) -> bool:
        self.seen = now
        if not active:
            self.since = None
            return False
        if self.since is None:
            self.since = now  # first poll the state held -> arm, but never fail on the same poll
            return False
        return now - self.since > grace

    def unknown(self, now: float) -> None:
        """A reading that proves nothing either way: hold confirmed duration, drop the blind gap.

        Resetting hides persistent failure; charging the gap can expire a newly restarted state.
        """
        if self.since is not None and self.seen is not None:
            self.since += now - self.seen
        self.seen = now


def capacity_escalation_note(on_last_gpu: bool) -> str:
    """The escalation clause a capacity failure detail carries. States the escalation fact ONLY.

    `on_last_gpu` means no further class escalation follows, not that classes are exhausted or a
    retry occurs. Keep both branches neutral; the supervisor owns target selection and budget.
    """
    return (
        "no further GPU-class escalation follows"
        if on_last_gpu
        else "GPU-class escalation may follow"
    )


def _health_deadline_in(
    raw_remaining: float, *, armed: bool, now: float, probe_at: float | None
) -> float:
    """When a health grace can actually FIRE, not when it nominally runs out.

    `_classify_queue_state` evaluates the unhealthy/throttled timers only on the ~90s probe cadence,
    so a grace that runs out mid-gap is not acted on until the next probe. Capacity, no-progress and
    the wall deadline are checked every poll and need no such adjustment, which is why quoting a raw
    health remainder against them overstates how soon the health timer bites.

    Eligibility is measured from `probe_at`, the clock the CURRENT probe was actually stamped with,
    not from `now`. Those differ whenever the health request itself was slow: `now` is read after
    the request returns, so assuming a full cadence from it would push the next probe further out
    than it really is and could hide a health grace that fires first. A request that outran a whole
    cadence leaves the next probe due immediately.

    The checks that can fire this timer, as offsets from `now`: the `expired()` call a few lines
    below this note in the SAME pass, then one per health probe. An unarmed timer is not charged an
    extra cadence -- `GraceTimer.expired` arms `since = now` on that same call ("first poll the state
    held -> arm"), so its grace starts running now; it simply cannot also EXPIRE on the pass that
    arms it, which is what dropping that first check from its candidate list expresses.

    `expired()` fires on `now - since > grace`, strictly, so a probe landing exactly ON the deadline
    does not fire it and the one after does.
    """
    # when the next probe is due, relative to `now`; never negative, and 0 when already due.
    next_probe_in = 0.0 if probe_at is None else max(probe_at + HEALTH_PROBE_CADENCE_S - now, 0.0)
    if armed and raw_remaining < 0.0:
        return 0.0  # the call below this note fires it
    if next_probe_in > raw_remaining:
        return next_probe_in
    # the first probe strictly beyond the grace's own deadline
    extra = math.floor((raw_remaining - next_probe_in) / HEALTH_PROBE_CADENCE_S) + 1
    return next_probe_in + extra * HEALTH_PROBE_CADENCE_S


def queue_wait_note(
    now: float,
    *,
    unhealthy: bool,
    unhealthy_since: float | None,
    unhealthy_grace_s: float,
    throttled: bool,
    throttled_since: float | None,
    throttled_grace_s: float,
    queued_since: float | None,
    queue_grace_s: float,
    capacity_applies: bool = True,
    on_last_gpu: bool,
    absolute_deadline: float | None,
    stall_since: float,
    stall_limit_s: float,
    probe_at: float | None = None,
) -> str:
    """The budget clause on a "still queued" line: how long we have waited of how long we will.

    Without it the queued line repeats unchanged every 90s and nothing separates a wait still
    within budget from a wedged one, so the operator's natural reaction is to cancel and relaunch,
    throwing away queue position.

    Report the deadline that will actually FIRE, which means the one with the LEAST TIME LEFT.
    FIVE independent deadlines can end a queued wait -- the capacity, unhealthy and throttled
    graces, the no-progress stall limit, and the run's absolute wall deadline -- they run
    CONCURRENTLY, and poll_job takes each as its own argument, so none may be given fixed priority.
    Any of the shorter ones can expire first: an unhealthy worker is neither usable nor
    initializing, so the capacity timer keeps counting underneath it, and quoting a 240s unhealthy
    grace on a job already 800s into a 900s capacity grace would promise 240s when 100s remain.
    Whichever is nearest is the honest number, because that is the one the operator will hit.

    The no-progress limit always applies, so there is always at least one deadline to report.
    """
    # (fires_in, check_order, waited, budget, label, clause). The health timers cannot fire at their
    # raw deadline: _classify_queue_state only re-checks them on the ~90s probe cadence, while
    # capacity, no-progress and the wall deadline are checked every poll. So a health timer actually
    # expires at its NEXT eligible probe. Charging that lateness is what makes "nearest" honest -- an
    # unhealthy timer with 10s of raw grace left does not beat a capacity timer with 20s, because
    # capacity fires inside the probe gap.
    #
    # `check_order` breaks ties in the order the remaining checks of THIS iteration actually run,
    # counted from this line: the health timers a few lines below, then _classify_stall, then the
    # next iteration's wall check and the capacity timer inside the next _classify_queue_state.
    # It decides between deadlines that are ALREADY SPENT, where raw overdue-ness says nothing about
    # which one bites: capacity and the wall were checked before this note and did not fire, so if a
    # slow health request pushed both them and the no-progress limit past due, _classify_stall is the
    # next check to run and `stalled` is what the operator will see. Ranking "most overdue" first
    # would name a deadline this iteration has already declined to act on.
    health_order, stall_order, wall_order, capacity_order = 0, 1, 2, 3
    candidates = []
    if unhealthy:
        waited = 0.0 if unhealthy_since is None else now - unhealthy_since
        remaining = _health_deadline_in(
            unhealthy_grace_s - waited,
            armed=unhealthy_since is not None,
            now=now,
            probe_at=probe_at,
        )
        candidates.append((remaining, health_order, waited, unhealthy_grace_s, "unhealthy", ""))
    if throttled:
        waited = 0.0 if throttled_since is None else now - throttled_since
        remaining = _health_deadline_in(
            throttled_grace_s - waited,
            armed=throttled_since is not None,
            now=now,
            probe_at=probe_at,
        )
        candidates.append((remaining, health_order, waited, throttled_grace_s, "throttled", ""))
    # the caller only asks for a note when it has just confirmed no usable or recovering worker, so
    # the capacity timer is either running or about to re-arm on the next poll. a cleared `since`
    # therefore means zero elapsed, not "not applicable": _classify_queue_state clears it via the
    # PREVIOUS probe's still-recent worker sighting, one step before the fresh probe below finds no
    # worker at all. dropping the candidate there would quote the 3000s no-progress limit while the
    # 300s capacity grace is what actually terminates the run.
    if capacity_applies:
        waited = 0.0 if queued_since is None else now - queued_since
        # the escalation fact ONLY, like `capacity_escalation_note`: on_last_gpu is also true when
        # the infra retry budget is exhausted with GPU classes still untried, so it must never be
        # reported as "this GPU class is pinned". derive "is given longer" from the grace ACTUALLY
        # in force, not from the flag: poll_job exposes on_last_gpu and queue_grace_s independently
        # and never ties them, so a caller can set the flag while overriding the grace down.
        clause = (
            " (no further GPU-class escalation follows, so capacity is given longer)"
            if on_last_gpu and queue_grace_s > 300.0
            else " (no further GPU-class escalation follows)"
            if on_last_gpu
            else ""
        )
        # clamped, like every candidate below: a deadline already past fires at 0, and how far past
        # it is says nothing about which one bites -- capacity was checked BEFORE this note with the
        # pre-request clock and declined to fire, so on a slow health request it queues behind the
        # checks that have not run yet. `check_order` is what resolves that, not overdue-ness.
        candidates.append(
            (
                max(queue_grace_s - waited, 0.0),
                capacity_order,
                waited,
                queue_grace_s,
                "capacity",
                clause,
            )
        )
    # the no-progress limit, checked one call after the queue classifier in the same iteration.
    # setup_grace_s normally dwarfs the queue graces, but poll_job accepts both independently, so a
    # short stall_after_s can be the real deadline while the capacity grace is still wide open.
    stall_waited = now - stall_since
    candidates.append(
        (
            max(stall_limit_s - stall_waited, 0.0),
            stall_order,
            stall_waited,
            stall_limit_s,
            "no-progress",
            "",
        )
    )
    if absolute_deadline is not None:
        # the run-global wall clock, which poll_job checks before every status read. on a late
        # retry or reattachment it can be shorter than any grace, and it is not a grace at all --
        # nothing "arms" it -- so it is reported as the remaining budget rather than time served.
        wall_left = absolute_deadline - now
        candidates.append((max(wall_left, 0.0), wall_order, 0.0, wall_left, "run wall", ""))
    _, _, waited, budget, label, clause = min(candidates, key=lambda c: (c[0], c[1]))
    if label == "run wall":
        return f"{int(max(budget, 0.0))}s left of the run wall deadline"
    unit = "limit" if label == "no-progress" else "grace"
    # clamped at both ends: never more elapsed than the budget (this line prints before the timer
    # is checked, so it would otherwise read "waited 360s of 240s"), and never negative -- a timer
    # stamped a moment after `now` was read would print "waited -20s", which reads as a bug.
    return f"waited {int(min(max(waited, 0.0), budget))}s of {int(budget)}s {label} {unit}{clause}"


def submit_run(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict[str, str] | None = None,
    on_last_gpu: bool = False,
    code_prefix: str | None = None,
    deadline_at: float | None = None,
) -> PollResult:
    """Deploy, submit, persist handle via ``on_handle``, and poll to completion."""
    from flash.envs.base import worker_pip_with_extras
    from flash.providers.runpod.serverless import _run_suffix, build_worker_env
    from flash.runner import flash_code_prefix

    deadline_at = require_deadline_at(deadline_at)
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("RunPod attempt identity is invalid")
    timeout_s = int(require_create_allowance(deadline_at))
    # Per-attempt suffix so a retry lands on a fresh endpoint, not the same throttled/sick host.
    suffix = _run_suffix(spec.run_id)
    if attempt_id:
        suffix = f"{suffix}r{attempt_id}"
    # the author's [environment] pip is appended to the worker requirement, never substituted for it.
    extra_pip = worker_pip_with_extras(spec.environment.id, spec.environment.pip)
    worker_env = build_worker_env(
        spec,
        seed,
        runtime_secrets=runtime_secrets,
    )
    worker_env["ATTEMPT"] = str(attempt_id)
    endpoint_id, name, key_fingerprint = deploy_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=suffix,
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
        **deadline_kwargs(deploy_train_endpoint, deadline_at),
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": spec.seed,
        "env": worker_env,
        "extra_pip": extra_pip,
        "code_prefix": code_prefix or flash_code_prefix(),
        "deadline_at": deadline_at,
    }
    try:
        require_create_allowance(deadline_at)
        submitted_ts = time.time()
        job_id = runpod_api.submit_job(
            endpoint_id,
            build_function_input(payload),
            key_fingerprint=key_fingerprint,
            **deadline_kwargs(runpod_api.submit_job, deadline_at),
        )
    except Exception as exc:
        # the queue post is non-idempotent: only a positively confirmed endpoint deletion makes
        # the original failure safe to retry. otherwise persist exact cleanup identity and stop.
        deletion_confirmed = False
        with contextlib.suppress(Exception):
            deletion_confirmed = (
                runpod_api.delete_endpoint_for_fingerprint(endpoint_id, key_fingerprint) is True
            )
        if deletion_confirmed:
            raise
        if on_handle is not None:
            on_handle(
                JobHandle(
                    endpoint_id,
                    name,
                    key_fingerprint,
                    None,
                    attempt_id,
                    time.time(),
                ).to_dict()
            )
        raise UnreconciledCreateError(
            "RunPod queue submission could not be reconciled and endpoint deletion was unconfirmed"
        ) from exc
    handle = JobHandle(
        endpoint_id,
        name,
        key_fingerprint,
        job_id,
        attempt_id,
        submitted_ts,
    )
    if log is not None:
        print(
            f"submitted job: endpoint={name} ({endpoint_id}) job={job_id} "
            f"attempt={attempt} gpu={spec.gpu.type} phase={spec.phase} seed={seed}",
            file=log,
            flush=True,
        )
    if on_handle is not None:
        on_handle(handle.to_dict())
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    reader = (
        make_hf_heartbeat_reader(
            hf_repo,
            prefix,
            **deadline_kwargs(make_hf_heartbeat_reader, deadline_at),
        )
        if hf_repo
        else None
    )
    failure_reader = (
        make_hf_failure_detail_reader(
            hf_repo,
            prefix,
            spec.phase,
            attempt=attempt_id,
            **deadline_kwargs(make_hf_failure_detail_reader, deadline_at),
        )
        if hf_repo
        else None
    )
    return poll_job(
        handle,
        log=log,
        heartbeat_reader=reader,
        failure_detail_reader=failure_reader,
        current_attempt=attempt_id,
        **deadline_kwargs(poll_job, deadline_at),
        on_last_gpu=on_last_gpu,
        **stall_kwargs(on_last_gpu=on_last_gpu),
    )


# make_hf_heartbeat_reader / make_hf_failure_detail_reader and the heartbeat-provenance predicate
# worker_flagged_retriable are provider-neutral and live in flash.providers.artifacts.hf; they are
# re-exported here so the runpod package and tests reference them as runpod.jobs.<name>.


def surfaced_worker_flags(
    heartbeat_reader,
    last_hb_key,
    say,
    current_attempt: int | None = None,
    *,
    launch_ts: float | None = None,
) -> tuple:
    """Read once for heartbeat surfacing plus structured retriable/OOM flags."""
    hb = heartbeat_reader(force=True) if heartbeat_reader is not None else None
    last_hb_key, _ = surface_heartbeat(lambda: hb, last_hb_key, say)
    retriable = worker_flagged_retriable(
        lambda force=False: hb, launch_ts=launch_ts, current_attempt=current_attempt
    )
    return last_hb_key, retriable, heartbeat_oom_for_attempt(hb, current_attempt)


# re-exported so the volume and sweep helpers stay reachable as `jobs.<name>`: the tests patch
# `_sweep_idle_flash_endpoints` and reach into `_idle_since` here, and `deploy_train_endpoint`
# below calls all of them.
from flash.providers.runpod.resources import (  # noqa: E402,F401,I001
    _VOLUME_INCAPABLE_DATACENTERS,
    WEIGHT_CACHE_GROW_BUDGET_S,
    _idle_since,
    _idle_since_lock,
    _is_flash_endpoint,
    _sweep_idle_flash_endpoints,
    canonical_endpoint_name,
    grow_weight_cache_volumes,
    weight_cache_datacenters,
    weight_cache_endpoint_kwargs,
    weight_cache_grow_headroom_s,
    weight_cache_volume_name,
    weight_cache_volumes,
)


from flash.providers.runpod.job_execution import (  # noqa: E402
    deploy_train_endpoint,
    poll_job,
)
