"""Durable run primitives: deploy -> submit -> poll with a persisted job handle."""

from __future__ import annotations

import asyncio  # noqa: F401
import contextlib
import json
import math
import time
from dataclasses import dataclass

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.core.spec import gpu_count_of
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.instances.poll import (
    _attempt_int,
    heartbeat_oom_for_attempt,
    surface_heartbeat,
)
from flash.providers.artifacts.hf import (
    make_hf_failure_detail_reader,
    make_hf_heartbeat_reader,
    worker_flagged_retriable,
)
from flash.providers.core.base import (  # noqa: F401
    PollResult,
    UnreconciledCreateError,
    canonical_gpu,
)
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client.gpus import flash_gpu  # noqa: F401
from flash.providers.runpod.serverless import (  # noqa: F401
    DEFAULT_EXECUTION_TIMEOUT_MS,
    FLASH_SDK_LOCK,
    WORKER_IMAGE,
    _patch_runpod_backoff,
    isolate_flash_state,
    min_cuda_for,
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
    "capacity_escalation_note",
    "capacity_grace_multiplier",
    "decode_output",
    "deploy_train_endpoint",
    "grow_weight_cache_volumes",
    "make_hf_failure_detail_reader",
    "make_hf_heartbeat_reader",
    "poll_job",
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

# multi-card shapes are rarer than single cards, so a grace sized for 1x expires on a 4x wait that
# was merely slow rather than starved. expiring it does not find capacity faster: the supervisor
# tears the endpoint down and re-requests THE SAME class (`_select_candidate` re-picks the only
# fitting shape once nothing is untried), so the run pays a fresh cold start to rejoin the same
# queue it just left. observed: 3-5 attempts and ~55 min of queueing per arm before a single
# optimizer step, with the multi-GPU arms churning most. scale the wait with the card count
# instead, so scarcity is waited out on one queue position rather than re-requested.
CAPACITY_GRACE_PER_GPU_CAP = 4


def capacity_grace_multiplier(gpu_count: int) -> int:
    """Scarcity multiplier on the capacity grace for a ``gpu_count``-card shape.

    Linear in the card count and capped: a 4x shape waits 4x as long as a 1x one, and anything
    wider than the cap waits the cap rather than growing without bound. Single-card runs multiply
    by 1, so their timing is exactly what it was.
    """
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        return 1
    return min(gpu_count, CAPACITY_GRACE_PER_GPU_CAP)


# how long ONE "a worker is coming up" health reading keeps suppressing the capacity timer. probes
# run every 90s, so this absorbs a couple of missed or failed probes and no more: if health stops
# being readable entirely, the capacity timer re-arms rather than waiting out the run on an
# observation nobody can still confirm.
WORKER_COMING_UP_TTL_S = 300.0

TERMINAL_OK = {"COMPLETED"}
# CANCELLED/TIMED_OUT = provider-killed (retriable); FAILED = worker died on its own (fails fast).
PLATFORM_TERMINATIONS = {"CANCELLED", "TIMED_OUT"}
TERMINAL_FAIL = {"FAILED"} | PLATFORM_TERMINATIONS


def stall_kwargs(on_last_gpu: bool = False, gpu_count: int = 1) -> dict:
    """poll_job stall-window kwargs. queue/throttled grace is ~5 min normally, ~15 min on last GPU
    (nowhere left to walk), then scaled by the card count because multi-card shapes are scarcer."""
    grace = (LAST_GPU_CAPACITY_GRACE_S if on_last_gpu else 300.0) * capacity_grace_multiplier(
        gpu_count
    )
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
        if not runpod_api._is_valid_key_fingerprint(fingerprint):
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
    """Decode a queue-job output into the worker's metrics dict."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"unexpected job output tail: {_safe_failure_text(output, 200)}"
            ) from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
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


def submit_run(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict[str, str] | None = None,
    on_last_gpu: bool = False,
    *,
    source_snapshot: dict,
    deadline_at: float | None = None,
) -> PollResult:
    """Deploy, submit, persist handle via ``on_handle``, and poll to completion."""
    from flash.envs.loading.base import worker_pip_with_extras
    from flash.providers.runpod.serverless import _run_suffix, build_worker_env
    from flash.source_snapshot import parse_descriptor

    deadline_at = require_deadline_at(deadline_at)
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("RunPod attempt identity is invalid")
    source_descriptor = parse_descriptor(source_snapshot)
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
        "run_id": spec.run_id,
        "attempt": attempt_id,
        "seed": spec.seed,
        "env": worker_env,
        "extra_pip": extra_pip,
        "source_snapshot": source_descriptor.to_dict(),
        "deadline_at": deadline_at,
    }
    try:
        require_create_allowance(deadline_at)
        submitted_ts = time.time()
        job_id = runpod_api.submit_job(
            endpoint_id,
            payload,
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
        # the count actually rented for this attempt, which allocation may have resolved to fewer
        # cards than the spec's ceiling named -- so read the effective spec, not the run's request.
        **stall_kwargs(on_last_gpu=on_last_gpu, gpu_count=gpu_count_of(spec)),
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
from flash.providers.runpod.execution.resources import (  # noqa: E402,F401,I001
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


from flash.providers.runpod.execution.job_execution import (  # noqa: E402
    deploy_train_endpoint,
    poll_job,
)
