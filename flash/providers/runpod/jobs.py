"""Durable run primitives: deploy -> submit -> poll with a persisted job handle."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flash._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
from flash.providers._poll import (
    PollErrorTracker,
    _attempt_int,
    heartbeat_oom_for_attempt,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers.base import PollResult, canonical_gpu
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.gpus import flash_gpu
from flash.providers.runpod.train import (
    DEFAULT_EXECUTION_TIMEOUT_MS,
    FLASH_SDK_LOCK,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    _patch_runpod_backoff,
    _train_body,
    endpoint_name,
    isolate_flash_state,
    min_cuda_for,
    resolve_worker_deps,
    worker_image_for_gpu,
)

logger = get_logger(__name__)

# Re-export for callers that import PollResult from here.
__all__ = [
    "JobHandle",
    "PollResult",
    "apply_disk_gb",
    "build_function_input",
    "decode_output",
    "deploy_train_endpoint",
    "make_hf_failure_detail_reader",
    "make_hf_heartbeat_reader",
    "make_hf_text_reader",
    "poll_job",
    "submit_run",
    "weight_cache_datacenters",
    "weight_cache_endpoint_kwargs",
    "weight_cache_volume_name",
    "weight_cache_volumes",
]

TERMINAL_OK = {"COMPLETED"}
# CANCELLED/TIMED_OUT = provider-killed (retriable); FAILED = worker died on its own (fails fast).
PLATFORM_TERMINATIONS = {"CANCELLED", "TIMED_OUT"}
TERMINAL_FAIL = {"FAILED"} | PLATFORM_TERMINATIONS

def stall_kwargs(on_last_gpu: bool = False) -> dict:
    """poll_job stall-window kwargs. queue/throttled grace is ~5 min normally, ~15 min on last GPU (nowhere left to walk)."""
    grace = 900.0 if on_last_gpu else 300.0
    return {
        "stall_after_s": 1500.0,
        "setup_grace_s": 3000.0,
        "queue_grace_s": grace,
        "throttled_grace_s": grace,
    }


# DCs that do NOT support network volumes — creating one there 500s the whole deploy.
# SDK exposes no capability flag; stale list degrades gracefully (falls back to cold cross-region run).
_VOLUME_INCAPABLE_DATACENTERS = frozenset({"US-MO-1"})


def weight_cache_datacenters() -> list:
    """Every volume-capable RunPod DC (DataCenter.all() minus _VOLUME_INCAPABLE_DATACENTERS)."""
    from runpod_flash.core.resources.datacenter import DataCenter

    return [dc for dc in DataCenter.all() if dc.value not in _VOLUME_INCAPABLE_DATACENTERS]


def weight_cache_volume_name(base: str, dc) -> str:
    """Physical volume name for ``base`` in datacenter ``dc``.

    DC MUST be in the name: the SDK keys resource tracking on name alone (no datacenter), so same-named
    volumes across DCs collide and the 2nd deploy crashes (unimplemented undeploy).
    """
    return f"{base}-{dc.value.lower()}"


def weight_cache_volumes(spec) -> list:
    """One NetworkVolume per storage datacenter; empty when the cache is off."""
    base = getattr(spec.gpu, "network_volume", None) if spec is not None else None
    if not base:
        return []
    dcs = weight_cache_datacenters()
    if not dcs:
        return []
    from runpod_flash import NetworkVolume

    from flash.spec import _volume_gb

    size = _volume_gb(getattr(spec.gpu, "network_volume_gb", 100))
    return [
        NetworkVolume(name=weight_cache_volume_name(str(base), dc), size=size, datacenter=dc)
        for dc in dcs
    ]


def weight_cache_endpoint_kwargs(spec) -> dict:
    """Endpoint kwargs that attach the weight-cache fleet, or ``{}`` (best-effort; cache off = no volumes)."""
    try:
        vols = weight_cache_volumes(spec)
        if not vols:
            return {}
        return {"volume": vols, "datacenter": weight_cache_datacenters()}
    except Exception as exc:
        logger.warning("weight cache disabled for this run (%s); deploying with no volume", exc)
        return {}


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
    job_id: str
    # Attempts share the seed's HF heartbeat path, so poll_job needs this to reject prior-attempt leftovers.
    attempt: int = 0

    def to_dict(self) -> dict:
        return {
            "provider": "runpod",
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "job_id": self.job_id,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        return cls(
            d["endpoint_id"],
            d.get("endpoint_name", ""),
            d["job_id"],
            _attempt_int(d.get("attempt")) or 0,
        )


def _is_workers_quota_error(exc: Exception) -> bool:
    """True when a RunPod exception signals the account worker quota is exhausted."""
    msg = str(exc).lower()
    return "max workers across all endpoints" in msg


# {endpoint_id: (first_observed_idle_ts, owning_key_fingerprint)} — grace timer per endpoint.
# Serialized by _idle_since_lock; two threads (periodic reaper + deploy-time quota sweep) can race.
_idle_since: dict[str, tuple[float, str]] = {}
_idle_since_lock = threading.Lock()


def canonical_endpoint_name(name: str) -> str:
    """Strip the SDK's ``live-`` prefix; the SDK registers ``live-flash-...`` but we track ``flash-...``."""
    return (name or "").removeprefix("live-")


def _is_flash_endpoint(name: str) -> bool:
    """True for a flash training endpoint (in either bare or live- registered form)."""
    return canonical_endpoint_name(name).startswith("flash-")


def _sweep_idle_flash_endpoints(
    protected: set[str],
    min_idle_s: float = 0.0,
    reap_warm: bool = True,
    known: set[str] | None = None,
) -> int:
    """Delete idle, ORPHANED flash training endpoints — workers doing nothing that still hold
    RunPod worker quota (runs that finished/crashed without tearing their endpoint down). Returns
    the count deleted.

    Safe by construction:

    - ``known`` — when supplied, the endpoint names for EVERY run THIS control plane has a record
      of. Only endpoints in this set are reapable; one whose name this plane has never issued
      belongs to ANOTHER control plane sharing the account and is left alone (multi-plane safety).
      ``None`` keeps unscoped behavior. Unlike ``protected`` this guards a second plane's
      *idle/between-jobs* endpoint — a busy one is already safe (it never reads as idle).
    - ``protected`` — endpoint names tied to a LIVE run (both the bare ``flash-...`` and the SDK's
      ``live-flash-...`` form). Never deleted, even if momentarily idle (e.g. between jobs).
    - ``reap_warm`` — when True (the run-aware periodic reaper, which protects EVERY live run),
      a merely *warm* ``idle``/``ready`` worker left over after a job counts as doing nothing and
      is reclaimable; that warm-idle state is the dominant leak, since RunPod keeps a worker warm
      after each job. When False (the deploy-time reactive sweep, which only protects the current
      run), a warm worker is treated as busy so the sweep reaps only endpoints that have FULLY
      scaled to zero — it must not delete another live run's between-jobs warm endpoint.
    - ``min_idle_s`` requires the idle reading to PERSIST across sweeps, so a single transient
      zero (cold start / between jobs) never triggers a delete.

    Resilient to a partial pool: ``RUNPOD_API_KEY`` may be a multi-account pool, and we list each
    account independently (``list_endpoints_by_key``). An account that fails to list this cycle
    (rejected/expired key, credit/quota, transient blip) is WARNed and SKIPPED — the accounts that
    DID respond are still reaped, and the skipped account's grace timers are preserved for the next
    sweep. (Before, the sweep listed via the all-or-nothing ``list_endpoints``: one unhealthy pool
    key aborted the WHOLE sweep, so idle orphans on healthy accounts piled up indefinitely while the
    failure was logged only at DEBUG — the bug this guards against.)
    """
    deleted = 0
    try:
        by_fp, failed_fps = runpod_api.list_endpoints_by_key()
    except Exception:
        logger.warning("idle-sweep: could not list any RunPod pool account; skipping sweep", exc_info=True)
        return 0
    if failed_fps:
        logger.warning(
            "idle-sweep: %d of %d RunPod pool account(s) failed to list this cycle; reaping the %d "
            "that responded and retrying the rest next sweep",
            len(failed_fps),
            len(by_fp) + len(failed_fps),
            len(by_fp),
        )
    responded_fps = set(by_fp)
    now = time.time()
    still_idle: set[str] = set()
    with _idle_since_lock:
        for fp, endpoints in by_fp.items():
            for ep in endpoints:
                ep_name = ep.get("name") or ""
                eid = ep.get("id")
                if not (eid and _is_flash_endpoint(ep_name)):
                    continue
                canon = canonical_endpoint_name(ep_name)
                if canon in protected:
                    continue
                if known is not None and canon not in known:
                    continue
                try:
                    health = runpod_api.endpoint_health_for_fingerprint(eid, fp) or {}
                    workers = health.get("workers")
                    jobs_info = health.get("jobs")
                    if not isinstance(workers, dict) or not workers or not isinstance(jobs_info, dict):
                        continue
                    busy_workers = (workers.get("running") or 0) + (workers.get("initializing") or 0)
                    if not reap_warm:
                        busy_workers += (workers.get("ready") or 0) + (workers.get("idle") or 0)
                    in_flight = (jobs_info.get("inQueue") or 0) + (jobs_info.get("inProgress") or 0)
                    if busy_workers != 0 or in_flight != 0:
                        _idle_since.pop(eid, None)
                        continue
                    still_idle.add(eid)
                    first_idle, _owner = _idle_since.setdefault(eid, (now, fp))
                    if now - first_idle < min_idle_s:
                        continue
                    if runpod_api.delete_endpoint_for_fingerprint(eid, fp):
                        deleted += 1
                        _idle_since.pop(eid, None)
                        logger.info("idle-sweep: deleted idle endpoint %s (%s)", ep_name, eid)
                except Exception:
                    logger.debug(
                        "idle-sweep: error processing endpoint %s (%s)", ep_name, eid, exc_info=True
                    )
                    continue
        # Prune stale timers only for accounts that responded this cycle (authoritative view).
        # Timers owned by failed accounts are kept — resetting them would restart the grace on each flake.
        prunable = {eid for eid, (_ts, owner_fp) in _idle_since.items() if owner_fp in responded_fps}
        for stale in prunable - still_idle:
            _idle_since.pop(stale, None)
    return deleted


def deploy_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
    endpoint_kwargs: dict | Callable[[], dict] | None = None,
) -> tuple[str, str]:
    """Deploy (or reuse) the run's uniquely-named worker endpoint; return (id, name).

    ``endpoint_kwargs`` may be a callable factory — re-invoked per account on quota failover so the
    SDK doesn't reuse a volume id stamped for the previous account.
    """
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.resource_manager import ResourceManager

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod.auth import ensure_auth

    _patch_runpod_backoff()
    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    image = worker_image_for_gpu(friendly, allow_default=True)

    def _deploy_once():
        """One get_or_deploy on the currently-active account."""
        with FLASH_SDK_LOCK:
            isolate_flash_state(name_suffix)
            kwargs = {
                "name": name,
                "gpu": flash_gpu(friendly),
                "gpu_count": 1,
                "min_cuda_version": min_cuda_for(friendly),
                "execution_timeout_ms": execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
                "workers": (0, 1),
            }
            if image:
                kwargs["image"] = image
            else:
                kwargs["dependencies"] = resolve_worker_deps()
                kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
            # Re-invoke factory per account (avoids reusing a volume id stamped for the prior account).
            override = endpoint_kwargs() if callable(endpoint_kwargs) else endpoint_kwargs
            kwargs.update(override if override is not None else weight_cache_endpoint_kwargs(spec))
            ep = Endpoint(**kwargs)
            ep._qb_target = _train_body
            config = ep._build_resource_config()
            apply_disk_gb(config, disk_gb)
            rm = ResourceManager()
            return asyncio.run(rm.get_or_deploy_resource(config))

    _QUOTA_MAX_RETRIES = 3
    resource = None
    # Bound by count, not advance_key() return value — advance_key() always wraps so can't signal exhaustion.
    failovers_left = max(0, rp_keys.key_count() - 1)
    while resource is None:
        ensure_auth()  # collapse RUNPOD_API_KEY to the (possibly failed-over) active account key
        quota_exc: Exception | None = None
        for quota_attempt in range(_QUOTA_MAX_RETRIES):
            if quota_attempt > 0:
                # Under acute quota pressure, sweep idle orphaned flash training endpoints on THIS
                # account NOW (min_idle_s=0) to free a slot. This only protects THIS run's endpoint,
                # so it stays conservative (reap_warm=False): it reaps only endpoints fully scaled
                # to zero, never another live run's between-jobs WARM endpoint. The control-plane
                # periodic reaper does the run-aware, graced warm-idle sweep across all live runs.
                swept = _sweep_idle_flash_endpoints(
                    protected={canonical_endpoint_name(name)}, min_idle_s=0.0, reap_warm=False
                )
                wait_s = 30 * quota_attempt
                logger.warning(
                    "RunPod worker quota hit (attempt %d/%d): swept %d idle flash-* endpoint(s); "
                    "retrying in %ds",
                    quota_attempt + 1, _QUOTA_MAX_RETRIES, swept, wait_s,
                )
                time.sleep(wait_s)
            try:
                resource = _deploy_once()
                break  # success
            except Exception as exc:
                if not _is_workers_quota_error(exc):
                    raise
                quota_exc = exc
        if resource is not None:
            break
        if failovers_left > 0:
            rp_keys.advance_key()
            failovers_left -= 1
            logger.warning(
                "RunPod worker quota exhausted on this account after sweeping; failing over to "
                "the next RUNPOD_API_KEY account (%d configured)",
                rp_keys.key_count(),
            )
            continue
        raise quota_exc or RuntimeError("deploy_train_endpoint: worker quota exhausted")

    endpoint_id = getattr(resource, "id", None)
    if not endpoint_id:
        raise RuntimeError(f"deploy_train_endpoint: no endpoint id on resource {resource!r}")
    return endpoint_id, name


def build_function_input(payload: dict) -> dict:
    """The FunctionRequest dict a Flash queue worker expects for `_train_body(payload)`."""
    if os.environ.get("FLASH_WORKER_IMAGE") or WORKER_IMAGE:
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


def decode_output(output) -> dict:
    """Decode a queue-job output into the worker's metrics dict (handles live-function and baked-image shapes)."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unexpected job output: {output[:200]}") from exc
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
        stdout_tail = (output.get("stdout") or "")[-1500:]
        raise RuntimeError(
            f"Remote execution failed: {err}\n--- worker stdout tail ---\n{stdout_tail}"
        )
    if output.get("error"):
        stdout_tail = (output.get("stdout") or "")[-1500:]
        msg = f"Remote execution failed: {output['error']}"
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
    """Grace timer: arms on first active poll, expires after ``grace`` seconds of continuous active state."""

    since: float | None = None

    def expired(self, active: bool, now: float, grace: float) -> bool:
        if not active:
            self.since = None
            return False
        if self.since is None:
            self.since = now  # first poll the state held -> arm, but never fail on the same poll
            return False
        return now - self.since > grace


def poll_job(
    handle: JobHandle,
    log=None,
    interval_s: float = 10.0,
    heartbeat_reader=None,
    failure_detail_reader=None,
    stall_after_s: float = 1200.0,
    setup_grace_s: float = 3000.0,
    unhealthy_grace_s: float = 240.0,
    throttled_grace_s: float = 300.0,
    queue_grace_s: float = 300.0,
    deadline_s: float | None = None,
    current_attempt: int | None = None,
) -> PollResult:
    """Poll a queue job to completion; resilient to transient API errors.

    Uses setup_grace_s (large) until first training heartbeat, then stall_after_s (tight).
    Fails fast on THROTTLED/UNHEALTHY workers and jobs stuck IN_QUEUE past queue_grace_s.
    """

    say = make_say(log)
    poll_errors = PollErrorTracker(say, interval_s)

    start = time.time()
    last_status = None
    last_hb_key = None
    last_hb_ts = 0.0
    last_hb_attempt = -1  # -1 sentinel < any real attempt; gates out prior-attempt leftover heartbeats
    last_progress = time.time()
    seen_heartbeat = False
    last_health_probe = 0.0
    unhealthy_timer = GraceTimer()
    throttled_timer = GraceTimer()
    queued_timer = GraceTimer()
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            st = runpod_api.job_status(handle.endpoint_id, handle.job_id)
            poll_errors.reset()
        except runpod_api.RunpodApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        status = st.get("status")
        if status != last_status:
            say(f"job {handle.job_id}: {status}")
            last_status = status
            last_progress = time.time()
        if status in TERMINAL_OK:
            try:
                return PollResult(True, metrics=decode_output(st.get("output")))
            except RuntimeError as e:
                last_hb_key, retriable, oom = surfaced_worker_flags(
                    heartbeat_reader, last_hb_key, say, current_attempt
                )
                detail = _append_failure_artifacts(str(e), failure_detail_reader)
                return PollResult(
                    False,
                    failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
                    detail=detail,
                )
        if status in TERMINAL_FAIL:
            detail = str(st.get("error") or "")[:1500]
            out = st.get("output")
            if isinstance(out, dict) and out.get("stdout"):
                detail += "\n--- worker stdout tail ---\n" + str(out["stdout"])[-2000:]
            elif not detail:
                detail = str(out)[:1500]
            if status in PLATFORM_TERMINATIONS:
                return PollResult(False, failure="job_preempted", detail=f"[{status}] {detail}")
            last_hb_key, retriable, oom = surfaced_worker_flags(
                heartbeat_reader, last_hb_key, say, current_attempt
            )
            detail = _append_failure_artifacts(detail, failure_detail_reader)
            return PollResult(
                False,
                failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
                detail=f"[{status}] {detail}",
            )
        now = time.time()
        if queued_timer.expired(status == "IN_QUEUE", now, queue_grace_s):
            return PollResult(
                False,
                failure="no_capacity",
                detail=f"never scheduled: job stuck IN_QUEUE for {int(now - queued_timer.since)}s "
                "(no RunPod capacity for the pinned GPU class); retrying on the next-best GPU",
            )
        if status != "IN_QUEUE":
            # The in-queue grace timers measure CONTINUOUS throttle/unhealthy while queued (like
            # queued_timer, driven every iteration above); reset them whenever the job leaves the
            # queue so a re-queue after an IN_PROGRESS spell doesn't fire on a stale arm time.
            unhealthy_timer.since = None
            throttled_timer.since = None
        elif now - last_health_probe > 90:
            last_health_probe = now
            try:
                h = runpod_api.endpoint_health(handle.endpoint_id)
                workers = h.get("workers") or {}
                usable = workers.get("running") or workers.get("ready") or workers.get("idle")
                recovering = workers.get("initializing")
                if (
                    any(workers.get(k) for k in ("throttled", "unhealthy", "initializing"))
                    or not usable
                ):
                    say(f"queued; workers: {workers}")
                if unhealthy_timer.expired(
                    workers.get("unhealthy") and not usable and not recovering,
                    now,
                    unhealthy_grace_s,
                ):
                    return PollResult(
                        False,
                        failure="stalled",
                        detail=f"worker stuck unhealthy for "
                        f"{int(now - unhealthy_timer.since)}s while IN_QUEUE (likely a failed "
                        f"image pull); retrying on a fresh endpoint",
                    )
                if throttled_timer.expired(
                    workers.get("throttled") and not usable and not recovering,
                    now,
                    throttled_grace_s,
                ):
                    return PollResult(
                        False,
                        failure="no_capacity",
                        detail=f"never scheduled: worker stuck THROTTLED for "
                        f"{int(now - throttled_timer.since)}s while IN_QUEUE (no RunPod "
                        f"capacity for the pinned GPU class); retrying on the next-best GPU",
                    )
            except Exception:
                pass
        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            if stage is not None:
                hb_ts = new_key[2] if new_key else None
                hb_step = new_key[1] if new_key else None
                hb_attempt = _attempt_int(new_key[3]) if new_key else None
                is_training_hb = is_training_heartbeat(stage, hb_step)
                if current_attempt is not None and hb_attempt != current_attempt:
                    # Non-current heartbeat: ignore so stale progress never tightens the stall window.
                    pass
                elif hb_attempt is not None and hb_attempt > last_hb_attempt:
                    # Fresh attempt: reset ts baseline and re-derive seen_heartbeat so cold-start grace rearms.
                    last_hb_attempt = hb_attempt
                    last_hb_ts = hb_ts or 0.0
                    last_progress = time.time()
                    seen_heartbeat = is_training_hb
                elif (hb_attempt is None or hb_attempt == last_hb_attempt) and (
                    hb_ts is None or hb_ts > last_hb_ts
                ):
                    # Gate progress on ts advancing — a stale late upload must not buy a fresh stall window.
                    if hb_ts is not None:
                        last_hb_ts = hb_ts
                    last_progress = time.time()
                    if is_training_hb:
                        seen_heartbeat = True
        in_setup = heartbeat_reader is not None and not seen_heartbeat
        stall_limit = setup_grace_s if in_setup else stall_after_s
        if time.time() - last_progress > stall_limit:
            phase = "setup (pre-training)" if in_setup else "training"
            return PollResult(
                False,
                failure="stalled",
                detail=f"no worker progress for {int(time.time() - last_progress)}s "
                f"during {phase} (job status {status}, limit {int(stall_limit)}s)",
            )
        time.sleep(interval_s)


def submit_run(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict[str, str] | None = None,
    on_last_gpu: bool = False,
    code_prefix: str | None = None,
) -> PollResult:
    """Deploy, submit, persist handle via ``on_handle``, and poll to completion."""
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import _run_suffix, build_worker_env, chalk_extra_pip
    from flash.runner import flash_code_prefix

    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    # Per-attempt suffix so a retry lands on a fresh endpoint, not the same throttled/sick host.
    suffix = _run_suffix(spec.run_id)
    if attempt:
        suffix = f"{suffix}r{attempt}"
    extra_pip = (
        list(spec.environment.pip) or worker_pip_for_env(spec.environment.id)
    ) + chalk_extra_pip(spec)
    worker_env = build_worker_env(spec, seed, runtime_secrets=runtime_secrets)
    worker_env["ATTEMPT"] = str(int(attempt))
    endpoint_id, name = deploy_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=suffix,
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": worker_env,
        "extra_pip": extra_pip,
        "code_prefix": code_prefix or flash_code_prefix(),
    }
    try:
        job_id = runpod_api.submit_job(endpoint_id, build_function_input(payload))
    except Exception:
        # Delete orphaned endpoint — rN suffix can't be reconstructed from run_id later.
        with contextlib.suppress(Exception):
            runpod_api.delete_endpoint(endpoint_id)
        raise
    handle = JobHandle(endpoint_id, name, job_id, int(attempt))
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
    reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
    failure_reader = (
        make_hf_failure_detail_reader(hf_repo, prefix, spec.phase, attempt=int(attempt))
        if hf_repo
        else None
    )
    return poll_job(
        handle,
        log=log,
        heartbeat_reader=reader,
        failure_detail_reader=failure_reader,
        current_attempt=int(attempt),
        **stall_kwargs(on_last_gpu=on_last_gpu),
    )


def make_hf_text_reader(hf_repo: str, path_in_repo: str, min_interval_s: float = 45.0):
    """Rate-limited reader for an HF artifact; returns None until it exists or on any error."""
    state = {"last": 0.0}

    def read(force: bool = False) -> str | None:
        if not hf_repo:
            return None
        if not force and time.time() - state["last"] < min_interval_s:
            return None
        state["last"] = time.time()
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                hf_repo,
                path_in_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return f.read()
        except Exception:
            return None

    return read


def make_hf_heartbeat_reader(hf_repo: str, prefix: str, min_interval_s: float = 30.0):
    """Rate-limited JSON reader for ``{prefix}/heartbeat.json`` on HF."""
    text_reader = make_hf_text_reader(hf_repo, f"{prefix}/heartbeat.json", min_interval_s)

    def read(force: bool = False) -> dict | None:
        raw = text_reader(force=force)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    return read


def make_hf_failure_detail_reader(
    hf_repo: str,
    prefix: str,
    phase: str,
    min_interval_s: float = 45.0,
    attempt: int = 0,
):
    """Reader for worker-uploaded failure artifacts on HF (error/console txt); force-read after terminal failure."""
    # Attempt-scoped to match the worker's error_artifact_name(mode, attempt).
    err_name = f"error_{phase}_attempt{int(attempt or 0)}.txt"
    error_reader = make_hf_text_reader(hf_repo, f"{prefix}/{err_name}", min_interval_s)
    console_reader = make_hf_text_reader(
        hf_repo, f"{prefix}/console_{phase}.txt", min_interval_s
    )

    def read(force: bool = False) -> str | None:
        parts: list[str] = []
        error_text = error_reader(force=force)
        if error_text:
            parts.append(f"--- {err_name} ---\n{error_text[-4000:]}")
        console_text = console_reader(force=force)
        if console_text:
            parts.append(f"--- console_{phase}.txt ---\n{console_text[-4000:]}")
        return "\n".join(parts) if parts else None

    return read


def worker_flagged_retriable(heartbeat_reader) -> bool:
    """True if the worker stamped ``retriable`` in its last heartbeat (forces a fresh read)."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict):
        return False
    return bool(hb.get("retriable"))


def surfaced_worker_flags(heartbeat_reader, last_hb_key, say, current_attempt: int | None = None) -> tuple:
    """Read once for heartbeat surfacing plus structured retriable/OOM flags."""
    hb = heartbeat_reader(force=True) if heartbeat_reader is not None else None
    last_hb_key, _ = surface_heartbeat(lambda: hb, last_hb_key, say)
    retriable = bool(hb.get("retriable")) if isinstance(hb, dict) else False
    return last_hb_key, retriable, heartbeat_oom_for_attempt(hb, current_attempt)
