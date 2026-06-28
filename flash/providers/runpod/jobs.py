"""Durable run primitives: explicit deploy -> submit -> poll with a persisted job handle.

Calling `runpod_flash`'s all-in-one blocking handler directly would tie a run's life to
one client process and one HTTP poll loop: a client crash/network blip orphans an
otherwise-healthy GPU job (no job id is ever persisted), and any SDK polling bug kills
the run. This module owns the lifecycle instead:

  deploy_train_endpoint()  -> endpoint_id (Flash SDK deploy, same worker template)
  build_function_input()   -> the exact FunctionRequest payload Flash workers expect
  submit + poll_job()      -> REST queue API with hardened retries; the job handle
                              {endpoint_id, job_id} is persisted by the runner so
                              any process can re-attach (`flash status --follow`).
"""

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
    is_training_heartbeat,
    make_say,
    surface_forced_heartbeat,
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

# Re-export so callers/tests that did ``from ...jobs import PollResult`` keep working.
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
# The provider killed the worker (reclaim/preempt/time-cap) -> infra-shaped, retried. A worker
# "FAILED" is the run dying on its own (real traceback) -> fails fast.
PLATFORM_TERMINATIONS = {"CANCELLED", "TIMED_OUT"}
TERMINAL_FAIL = {"FAILED"} | PLATFORM_TERMINATIONS

def stall_kwargs(on_last_gpu: bool = False) -> dict:
    """``poll_job`` stall-window kwargs, shared by the submit and reattach paths so a recovered
    run uses the same tuning as the original submit. The original submit's ``on_last_gpu`` is
    PERSISTED in the run handle (the runner's ``on_handle`` writes it into ``remote``), so a
    cross-process reattach (``RunpodProvider.poll``) reads it back and calls this with the same
    value — a last-candidate run keeps its longer no-capacity grace after a control-plane restart
    instead of being judged on the shorter non-last window. ``stall_after_s`` = post-training-heartbeat
    window; ``setup_grace_s`` = the larger cold-start window before the first training heartbeat;
    ``queue_grace_s``/``throttled_grace_s`` = the two no-capacity backstops — how long a job may
    sit IN_QUEUE with no worker (``queue_grace_s``) or wait on a worker stuck THROTTLED
    (``throttled_grace_s``) before we treat the pinned GPU class as out of capacity and walk to
    the next-best one.

    These backstops are tuned to whether a further GPU attempt will follow. While a retry can still
    fall to a next-best class (``on_last_gpu`` False) we wait ~5 min: long enough to ride out a brief
    capacity blip, short enough that a genuinely starved class hands off to the next-best one
    promptly. When no further GPU attempt will be made — the candidate list is exhausted OR the retry
    budget is exhausted (``on_last_gpu`` True) — there is nowhere left to walk, so we wait ~15 min before
    giving up: burning the last attempt on a class with no fallback (and no retry left to spend the
    saved time on) is worse than waiting out a longer queue. Both are no-capacity backstops only:
    once the job leaves IN_QUEUE (a worker picks it
    up), the much larger ``setup_grace_s`` governs cold start and we never walk off an IN_PROGRESS
    job at the capacity grace.
    """
    grace = 900.0 if on_last_gpu else 300.0
    return {
        "stall_after_s": 1500.0,
        "setup_grace_s": 3000.0,
        "queue_grace_s": grace,
        "throttled_grace_s": grace,
    }


# RunPod DCs in the SDK's ``DataCenter.all()`` enum that do NOT support network volumes. LIVE-FOUND:
# the SDK enum is NOT the network-volume DC set (that assumption was wrong) — creating a volume in one
# of these 500s the WHOLE deploy ("data center ... does not support network volumes"), so eager runs
# would always fall back to cold and the cache would never work. The SDK exposes no volume-capability
# flag, so we maintain the exclusion here. (If RunPod drops volume support in another DC, that deploy
# 500s -> the lifecycle no_capacity/poll_error cache-drop falls back to a cold cross-region run, so a
# stale list degrades gracefully rather than wedging — but add the DC here to restore its cache.)
_VOLUME_INCAPABLE_DATACENTERS = frozenset({"US-MO-1"})


def weight_cache_datacenters() -> list:
    """Every VOLUME-CAPABLE RunPod DC — both the set the endpoint is allowed across AND the set we
    attach a per-DC volume in (eager: a volume in every region a run can land in, so any landing is
    warm). ``DataCenter.all()`` minus ``_VOLUME_INCAPABLE_DATACENTERS`` (the enum includes DCs RunPod
    no longer backs with network volumes — see that constant). A SDK upgrade that adds a storage region
    is picked up automatically; one that adds a volume-less region must be excluded above.
    """
    from runpod_flash.core.resources.datacenter import DataCenter

    return [dc for dc in DataCenter.all() if dc.value not in _VOLUME_INCAPABLE_DATACENTERS]


def weight_cache_volume_name(base: str, dc) -> str:
    """Physical volume name for ``base`` in datacenter ``dc`` — DISTINCT per DC.

    The cache is one logical volume (``base`` == ``spec.gpu.network_volume``, e.g. ``flash-weights``)
    realized as one physical volume per datacenter. The DC MUST be in the name: the runpod_flash SDK
    keys its in-memory/persisted resource tracking on ``NetworkVolume:{name}`` WITHOUT the
    datacenter (resources/base.py ``get_resource_key``), so N same-named volumes collide on one key
    and deploying the 2nd triggers a replace -> the SDK's unimplemented ``NetworkVolume.undeploy`` ->
    crash. A per-DC name gives each volume a unique key. The worker is unaffected — every volume
    mounts at the same ``/runpod-volume`` regardless of name.
    """
    return f"{base}-{dc.value.lower()}"


def weight_cache_volumes(spec) -> list:
    """One ``NetworkVolume`` per storage datacenter — the EAGER fleet (``[]`` only if the cache is off).

    Empty unless ``spec.gpu.network_volume`` is set (the runner assigns the logical base name for
    eligible runs). Otherwise one physical volume per ``weight_cache_datacenters()`` entry — every
    storage DC, so the cache exists in whichever region the endpoint lands in. Each physical volume is
    ``<base>-<dc>`` (see ``weight_cache_volume_name``), idempotent by (name, datacenter): runpod_flash
    reuses an existing volume of that name/DC, so this is create-or-attach (the first deploy provisions
    the whole fleet; later deploys just re-attach).

    Multi-account pools: ``deploy_train_endpoint`` re-runs the WHOLE deploy on quota failover, so the
    volumes are re-created on whichever account ends up hosting the endpoint (account-scoped). Orphans
    on the failed-over-FROM account are reclaimed by ``preload --teardown`` (sweeps every pool account).
    """
    base = getattr(spec.gpu, "network_volume", None) if spec is not None else None
    if not base:
        return []
    dcs = weight_cache_datacenters()  # EAGER: a volume in every storage DC
    if not dcs:
        return []
    from runpod_flash import NetworkVolume

    from flash.spec import _volume_gb

    # Reuse the spec's tolerant parser: a stale/hand-edited spec with a non-numeric, "0", or negative
    # network_volume_gb defaults to 100 GB rather than raising (which best-effort would swallow into a
    # no-cache deploy) or creating a nonsensical 0-GB volume — matches _volume_gb's contract/tests.
    size = _volume_gb(getattr(spec.gpu, "network_volume_gb", 100))
    return [
        NetworkVolume(name=weight_cache_volume_name(str(base), dc), size=size, datacenter=dc)
        for dc in dcs
    ]


def weight_cache_endpoint_kwargs(spec) -> dict:
    """Endpoint kwargs that attach the eager weight-cache fleet, or ``{}`` (best-effort).

    ``{"volume": [vol per storage dc...], "datacenter": [ALL storage DCs]}`` — the endpoint is allowed
    across ALL DCs (so it lands wherever there's capacity) AND carries a volume in every one of them, so
    whichever DC it lands in is warm. The SDK's "every volume DC must be in the endpoint datacenter
    list" rule holds exactly (the two lists are the same storage-DC set). The first deploy
    create-or-attaches the whole fleet; later deploys re-attach.

    Returns ``{}`` only when the cache is off (no ``network_volume`` on the spec). Best-effort: ANY
    failure (SDK import, validation) -> ``{}`` so the run deploys with no volume rather than failing;
    the lifecycle still drops the volume on a no_capacity retry to widen onto the non-storage DC pool.
    """
    try:
        vols = weight_cache_volumes(spec)
        if not vols:
            return {}  # cache off -> cold (no volumes, RunPod picks any region)
        return {"volume": vols, "datacenter": weight_cache_datacenters()}
    except Exception as exc:
        # Best-effort: never let the cache break a deploy — fall back to a no-volume run.
        logger.warning("weight cache disabled for this run (%s); deploying with no volume", exc)
        return {}


def apply_disk_gb(config, disk_gb: int | None) -> None:
    """Raise the worker's container disk on a built endpoint config.

    The Flash SDK's ``PodTemplate.containerDiskInGb`` defaults to 64 GB and the
    ``Endpoint`` wrapper exposes no disk knob, which is what blocked models whose
    checkpoint alone exceeds 64 GB. The template
    is already populated by the SDK's validators when the resource config is built, so
    raising the field here is the supported injection point. Raise-only: shrinking
    below the SDK default buys nothing (serverless disk isn't billed separately) and
    would regress runs whose configs carry the historical ``disk_gb = 60`` default.
    """
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

    def to_dict(self) -> dict:
        return {
            "provider": "runpod",
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        # `provider` is routing metadata consumed upstream (runner); handles
        # persisted before it existed default to runpod there.
        return cls(d["endpoint_id"], d.get("endpoint_name", ""), d["job_id"])


def _is_workers_quota_error(exc: Exception) -> bool:
    """True when a RunPod exception signals the account worker quota is exhausted."""
    msg = str(exc).lower()
    return "max workers across all endpoints" in msg


# Per-endpoint grace state: ``{endpoint_id: (first_observed_idle_ts, owning_key_fingerprint)}``, so a
# candidate must STAY idle across sweeps for ``min_idle_s`` before deletion (a cold-starting /
# between-jobs endpoint reports a transient zero we must not act on). The owning account's
# (non-secret) fingerprint is stored alongside the timestamp so the prune can drop a stale timer as
# soon as THAT account responds again — even while OTHER accounts are failing to list — without it
# growing unbounded during a partial outage.
#
# Two threads can run a sweep at once — the periodic control-plane reaper (via asyncio.to_thread)
# and a deploy-time quota sweep — so every read/write of ``_idle_since`` is serialized by this lock
# (a dedicated lock, NOT FLASH_SDK_LOCK, since the sweep uses the REST API, not the Flash SDK).
# Holding it across the sweep also prevents a concurrent sweep's prune from disturbing this one's
# grace timers; contention is negligible (the reaper runs every 10 min, deploy sweeps are rare).
_idle_since: dict[str, tuple[float, str]] = {}
_idle_since_lock = threading.Lock()


def canonical_endpoint_name(name: str) -> str:
    """The bare ``flash-<gpu>-<run>`` endpoint name, with the ``runpod-flash`` SDK's live-provisioned
    ``live-`` prefix stripped.

    One RunPod endpoint has TWO names: flash computes and stores the bare ``flash-...`` form (the
    name it asks the SDK to deploy), but the SDK registers the live resource as ``live-flash-...`` so
    it can't collide with the same-named template — so the RunPod account *lists* it as
    ``live-flash-...``. Canonicalizing on the boundary (every set we build AND every listed name we
    test) means the rest of the code only ever deals with the single bare form, instead of every call
    site re-deriving both. Idempotent; safe on non-flash names. Actual RunPod ops are keyed by
    endpoint *id*, never the name, so this normalization only affects matching."""
    return (name or "").removeprefix("live-")


def _is_flash_endpoint(name: str) -> bool:
    """True for a flash training endpoint this sweep may reap (in either registered form).
    Serving runs on freesolo's Modal app, not RunPod, so the only flash-* RunPod endpoints are
    training endpoints."""
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
      ``None`` keeps the legacy unscoped behavior. Unlike ``protected`` this guards a second plane's
      *idle/between-jobs* endpoint — a busy one is already safe (it never reads as idle).
    - ``protected`` — endpoint names tied to a LIVE run (both the bare ``flash-...`` and the SDK's
      ``live-flash-...`` form). Never deleted, even if momentarily idle (e.g. between seeds).
    - ``reap_warm`` — when True (the run-aware periodic reaper, which protects EVERY live run),
      a merely *warm* ``idle``/``ready`` worker left over after a job counts as doing nothing and
      is reclaimable; that warm-idle state is the dominant leak, since RunPod keeps a worker warm
      after each job. When False (the deploy-time reactive sweep, which only protects the current
      run), a warm worker is treated as busy so the sweep reaps only endpoints that have FULLY
      scaled to zero — it must not delete another live run's between-seeds warm endpoint.
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
        # Only reached when the pool itself is unusable (no key configured) — a per-account
        # failure is captured in failed_fps, not raised. WARN (not DEBUG): a reaper that can't
        # list anything is exactly the silent stall we want visible in the logs.
        logger.warning("idle-sweep: could not list any RunPod pool account; skipping sweep", exc_info=True)
        return 0
    if failed_fps:
        # A flaky/expired account must NOT silently no-op cleanup of the others (the orphan-pile-up
        # bug). Surface it loudly and carry on with whatever accounts responded.
        logger.warning(
            "idle-sweep: %d of %d RunPod pool account(s) failed to list this cycle; reaping the %d "
            "that responded and retrying the rest next sweep",
            len(failed_fps),
            len(by_fp) + len(failed_fps),
            len(by_fp),
        )
    # The (non-secret) fingerprints of the accounts that DID list this cycle. A grace timer is only
    # safe to prune once its OWNING account responded — then we know its endpoint's true state (still
    # idle, or genuinely gone). Timers owned by an account that failed this cycle are left untouched.
    responded_fps = set(by_fp)
    now = time.time()
    still_idle: set[str] = set()
    # Serialize all _idle_since access (see the lock's definition): a concurrent sweep must not
    # mutate the dict mid-iteration (the prune below would raise) or disturb these grace timers.
    with _idle_since_lock:
        for fp, endpoints in by_fp.items():
            for ep in endpoints:
                ep_name = ep.get("name") or ""
                eid = ep.get("id")
                if not (eid and _is_flash_endpoint(ep_name)):
                    continue
                # Compare against the bare form: ``protected``/``known`` are canonical, and RunPod
                # lists endpoints under the SDK's ``live-flash-...`` form (see canonical_endpoint_name).
                canon = canonical_endpoint_name(ep_name)
                if canon in protected:
                    continue
                # Multi-plane scope: only reap endpoints this plane has a record of. One whose name
                # is unknown here belongs to another control plane on the same account — leave it.
                if known is not None and canon not in known:
                    continue
                try:
                    # Account-scoped: we know which pool account owns this endpoint (its non-secret
                    # fingerprint), so query/delete with that exact key (resolved inside runpod_api,
                    # no 404-failover waterfall across the other accounts).
                    health = runpod_api.endpoint_health_for_fingerprint(eid, fp) or {}
                    workers = health.get("workers")
                    jobs_info = health.get("jobs")
                    # Require non-empty dicts: a missing/empty workers section means the health
                    # response is incomplete and we can't confirm the endpoint is idle.
                    if not isinstance(workers, dict) or not workers or not isinstance(jobs_info, dict):
                        continue
                    # "Busy" = a worker actually working or spinning up, OR a job queued/in progress.
                    # With reap_warm, a warm idle/ready worker with no pending work is NOT busy — it
                    # is the leftover we reclaim (the protected set + grace keep it safe). Without it,
                    # a warm worker counts as busy so only fully-scaled-to-zero endpoints are reaped.
                    busy_workers = (workers.get("running") or 0) + (workers.get("initializing") or 0)
                    if not reap_warm:
                        busy_workers += (workers.get("ready") or 0) + (workers.get("idle") or 0)
                    in_flight = (jobs_info.get("inQueue") or 0) + (jobs_info.get("inProgress") or 0)
                    if busy_workers != 0 or in_flight != 0:
                        _idle_since.pop(eid, None)  # busy again -> reset the grace timer
                        continue
                    still_idle.add(eid)
                    first_idle, _owner = _idle_since.setdefault(eid, (now, fp))
                    if now - first_idle < min_idle_s:
                        continue  # idle, but not for long enough yet — wait for the next sweep
                    if runpod_api.delete_endpoint_for_fingerprint(eid, fp):
                        deleted += 1
                        _idle_since.pop(eid, None)
                        logger.info("idle-sweep: deleted idle endpoint %s (%s)", ep_name, eid)
                except Exception:
                    logger.debug(
                        "idle-sweep: error processing endpoint %s (%s)", ep_name, eid, exc_info=True
                    )
                    continue
        # Drop grace timers for endpoints no longer idle/present, but ONLY for timers whose owning
        # account responded this cycle. For those accounts we have an authoritative view: an endpoint
        # that's still listed-and-idle is in ``still_idle``; one that's gone (busy again, or deleted)
        # is not, so its timer is stale and safe to drop. A timer owned by an account that FAILED to
        # list this cycle is left untouched — we can't tell if its endpoint went busy or vanished, so
        # resetting it would restart the grace every time that account flakes and the orphan would
        # never age out. (Full view => every owner responded => every non-still-idle timer is pruned,
        # the original behavior; this only changes the partial-outage case, where keying the prune on
        # the owning account — not on "ids seen this cycle" — lets a vanished endpoint from a HEALTHY
        # account age out instead of leaking its timer until the unrelated failing account recovers.)
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

    On a worker-quota error, sweeps idle flash-* endpoints (from crashed/completed runs
    that skipped GC) and retries up to ``_QUOTA_MAX_RETRIES`` times with backoff. If the
    account's quota stays exhausted after sweeping and ``RUNPOD_API_KEY`` configures more
    than one account, fails over to the next account (``keys.advance_key``) and deploys
    there. A single key => single account, no failover (unchanged behavior).

    ``endpoint_kwargs`` overrides the volume/datacenter attachment (default: the full multi-DC
    weight-cache fleet from ``weight_cache_endpoint_kwargs(spec)``). The preload driver passes a
    SINGLE-DC volume+datacenter so the worker provably lands in that region and warms its volume. It
    may be a dict OR a zero-arg FACTORY: under a multi-key pool the deploy retries on the next account
    after a quota failover, and the SDK can stamp an account-scoped id onto a NetworkVolume object —
    so a callable is re-invoked per account to build a FRESH volume (else the next account reuses the
    first account's stale volume id and the single-DC preload fails).
    """
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.resource_manager import ResourceManager

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod.auth import ensure_auth

    _patch_runpod_backoff()
    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    # deploy a self-contained serverless-worker image directly. by default this is WORKER_IMAGE;
    # when per-sm warmed images are enabled, the selected GPU class picks the matching image tag.
    # FLASH_WORKER_IMAGE remains the absolute hotfix override.
    image = worker_image_for_gpu(friendly, allow_default=True)

    def _deploy_once():
        """One get_or_deploy on the currently-active account (SDK + lock critical section)."""
        # isolate_flash_state mutates runpod_flash's process-wide registry globals for this run's
        # suffix, and ResourceManager + the deploy share the SDK's asyncio singleton. Hold the
        # lock across the whole critical section so a concurrent run can't swap the registry
        # scope or race the event loop mid-deploy.
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
            # Attach the multi-region weight cache (best-effort: {} when no cache / on any error).
            # The endpoint is allowed across every cache DC, so it is NOT pinned to one region.
            # A caller (preload) may override with a single-DC volume+datacenter.
            # Resolve a factory FRESH on each account attempt (see docstring: avoids reusing a
            # NetworkVolume the SDK stamped with the prior account's id across a quota failover).
            override = endpoint_kwargs() if callable(endpoint_kwargs) else endpoint_kwargs
            kwargs.update(override if override is not None else weight_cache_endpoint_kwargs(spec))
            ep = Endpoint(**kwargs)
            ep._qb_target = _train_body
            config = ep._build_resource_config()
            apply_disk_gb(config, disk_gb)
            # Worker image is PUBLIC, so no container-registry credential is needed to pull it.
            rm = ResourceManager()
            return asyncio.run(rm.get_or_deploy_resource(config))

    _QUOTA_MAX_RETRIES = 3
    resource = None
    # One pass over the pool, bounded purely by a COUNT — NOT by advance_key()'s return value.
    # advance_key() wraps (so it can't itself report "all accounts tried" — that depends on where
    # THIS failover started, which it can't know; a prior run may have left _idx mid-pool). Exactly
    # key_count()-1 advances visit every OTHER account once from any starting account, then we raise
    # — the lifecycle retry budget handles waiting for quota to recover and re-enters with a fresh
    # attempt. Relying on advance_key()'s boolean here would skip the wrapped-over accounts when the
    # loop starts on a non-first key.
    failovers_left = max(0, rp_keys.key_count() - 1)
    while resource is None:
        ensure_auth()  # collapse RUNPOD_API_KEY to the (possibly failed-over) active account key
        quota_exc: Exception | None = None
        for quota_attempt in range(_QUOTA_MAX_RETRIES):
            if quota_attempt > 0:
                # Under acute quota pressure, sweep idle orphaned flash training endpoints on THIS
                # account NOW (min_idle_s=0) to free a slot. This only protects THIS run's endpoint,
                # so it stays conservative (reap_warm=False): it reaps only endpoints fully scaled
                # to zero, never another live run's between-seeds WARM endpoint. The control-plane
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
                quota_exc = exc  # freeable: sweep + retry, then fail over to the next account
        if resource is not None:
            break
        # Quota still exhausted after sweeping this account dry — fail over to the next one, but only
        # until every other account has been tried once (the key_count()-based failovers_left bound).
        # The COUNT is the sole exhaustion signal: advance_key() always advances (wrapping) for a
        # multi-key pool, so a failover that started mid-pool still visits each remaining account
        # exactly once before this budget runs out and we raise.
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
        # Baked serverless-worker image (client mode): the image's rp_handler reads job["input"]
        # and calls _train_body, so the job input IS the train payload (submit_job wraps it in
        # {"input": ...}). No live-function source, no boot-install deps.
        return payload
    # Boot-install fallback (Flash default image + live function): ship _train_body's source for the
    # generic worker to run, plus the pinned worker deps to install on first use.
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
    """Decode a queue-job output into the worker's metrics dict. Handles BOTH job shapes:

    - Flash LIVE-function (boot-install path): a FunctionResponse envelope
      ``{"success": True, "result": <base64 cloudpickle of the dict>}``.
    - Client-mode SERVERLESS handler (baked-image path): our baked rp_handler returns
      ``_train_body(...)``'s metrics dict, which RunPod surfaces as ``job["output"]`` directly —
      no envelope. The metrics dict has no ``success``/``result`` keys, so we return it as-is.
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unexpected job output: {output[:200]}") from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
    # Flash live-function envelope (has success/result/error keys).
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
    # Client-mode serverless handler: the metrics dict IS the output (baked rp_handler).
    if output.get("error"):
        # Mirror the Flash path: append the worker stdout tail when present so poll_job's
        # root-cause diagnostics (e.g. a vLLM crash) survive the client-mode failure shape too.
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

    ``current_attempt`` (the attempt this poll is for) is forwarded to ``worker_flagged_oom`` so a
    prior attempt's lingering ``{"oom": true}`` in the shared-prefix heartbeat can't misclassify an
    unrelated startup/platform failure as an OOM and escalate onto a larger card (see that helper).
    ``None`` keeps the historical behavior for callers that don't know the attempt (reattach poll).

    Two stall windows: the cold-start phase (dep install, per-run env pip, model download,
    vLLM init) is slow and only emits *setup* heartbeats (``SETUP_HEARTBEAT_STAGES``).
    Until a *training* heartbeat arrives we apply the larger ``setup_grace_s`` budget so a
    slow cold start isn't misread as a stall; after it we use the tight ``stall_after_s``.
    Needs a ``heartbeat_reader`` to tell the phases apart — without one we keep
    ``stall_after_s`` throughout (no regression).

    ``failure_detail_reader`` force-reads worker-uploaded artifacts (``error_<phase>.txt`` and
    ``console_<phase>.txt``) after a worker terminal failure so a generic RunPod handler wrapper
    does not hide the real traceback.

    ``throttled_grace_s`` bounds how long we wait on a worker stuck THROTTLED (no RunPod
    capacity for the pinned GPU class) before returning a retryable stall so the runner
    walks to the next-best GPU. THROTTLED means there is no capacity for this class right now, so
    once a class with a cheaper fallback has stayed throttled this long, failing over beats
    blocking the run on a host that won't free up. ``stall_kwargs`` sets this to ~5 min while the
    gpu-walk still has a next-best class, and ~15 min on the last candidate (nowhere left to walk).

    ``queue_grace_s`` is the capacity backstop for that same walk when RunPod *doesn't* surface
    a THROTTLED/UNHEALTHY worker: a job can sit IN_QUEUE with zero workers assigned (or one stuck
    INITIALIZING, or while ``endpoint_health`` errors are swallowed below) and the throttled/
    unhealthy fast-fails never arm — so without this it would burn the full ``setup_grace_s``
    (~50 min). Keyed off the authoritative job status (robust to a failing health probe), it
    returns a retryable stall once a job has been IN_QUEUE longer than ``queue_grace_s`` (tuned by
    ``stall_kwargs`` like ``throttled_grace_s``: ~5 min normally, ~15 min on the last GPU class).
    The queue timer applies only while the job status remains IN_QUEUE; once a worker picks the
    job up (status leaves IN_QUEUE), it resets and ``setup_grace_s`` governs cold start.
    """

    say = make_say(log)
    poll_errors = PollErrorTracker(say, interval_s)

    start = time.time()
    last_status = None
    last_hb_key = None
    last_hb_ts = 0.0  # the worker-ts of the last heartbeat we COUNTED as progress (monotonic gate)
    last_hb_attempt = -1  # the worker-stamped attempt of that heartbeat (so an OLDER attempt's late
    #                       heartbeat can't reset progress); -1 sentinel < any real attempt (0,1,...)
    last_progress = time.time()
    seen_heartbeat = False
    last_health_probe = 0.0
    unhealthy_since: float | None = None  # first time the worker was seen stuck UNHEALTHY
    throttled_since: float | None = None  # first time the worker was seen stuck THROTTLED
    queued_since: float | None = None  # first time the job was seen IN_QUEUE with no worker yet
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
                # COMPLETED but the output decodes as an error (a handler exception). Consult the
                # worker flag too: an infra failure can surface here and must still retry. Read the
                # heartbeat ONCE and reuse it for surfacing AND both flag reads — three separate
                # force-reads added avoidable HF IO and could race (flags from different snapshots if
                # the heartbeat updated between reads).
                hb_reader = _once_forced(heartbeat_reader)
                last_hb_key, _ = surface_forced_heartbeat(hb_reader, last_hb_key, say)
                retriable = worker_flagged_retriable(hb_reader)
                oom = worker_flagged_oom(hb_reader, current_attempt)
                # Fallback when the structured oom flag is absent: scan the LIVE handler error only
                # (THIS attempt's output), NOT the HF error artifact appended below — that artifact is
                # shared per-seed across attempts and could be a prior attempt's stale traceback (it
                # would wrongly escalate, or raise the OOM floor on reattach). Gated on ``not oom``
                # only: detail_flags_cuda_oom skips retriable-infra text, so a CURRENT retriable still
                # wins, while a stale ``retriable`` heartbeat no longer blocks a real OOM here.
                if not oom:
                    oom = detail_flags_cuda_oom(str(e))
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
                # Worker stdout tail is the only place the REAL root cause lives for
                # crashes inside subprocesses (e.g. vLLM EngineCore deaths).
                detail += "\n--- worker stdout tail ---\n" + str(out["stdout"])[-2000:]
            elif not detail:
                detail = str(out)[:1500]
            # THIS attempt's live output (job error + worker stdout tail), captured BEFORE the shared
            # per-seed HF artifact is appended below — the attempt-fresh text the OOM fallback scans.
            live_detail = detail
            # Structural classification only ([{status}] prefix is for human-readable logs).
            # A platform termination (CANCELLED/TIMED_OUT) is already retryable — skip the worker
            # heartbeat read entirely (no worker error there, and it may not even exist yet).
            if status in PLATFORM_TERMINATIONS:
                return PollResult(False, failure="job_preempted", detail=f"[{status}] {detail}")
            # A worker FAILED: consult the structured worker flag. Read the heartbeat ONCE and reuse
            # it for surfacing AND both flag reads (avoids redundant HF IO + cross-snapshot races).
            hb_reader = _once_forced(heartbeat_reader)
            last_hb_key, _ = surface_forced_heartbeat(hb_reader, last_hb_key, say)
            retriable = worker_flagged_retriable(hb_reader)
            oom = worker_flagged_oom(hb_reader, current_attempt)
            # Fallback OOM detection on the attempt-fresh LIVE output only (a child-process OOM whose
            # root cause is only in the worker stdout tail — e.g. vLLM EngineCore). NOT the shared
            # per-seed HF artifact appended below (stale-prone across attempts). Gated on ``not oom``;
            # detail_flags_cuda_oom skips retriable-infra text so a current retriable still wins and a
            # stale retriable heartbeat no longer blocks a real OOM.
            if not oom:
                oom = detail_flags_cuda_oom(live_detail)
            detail = _append_failure_artifacts(detail, failure_detail_reader)
            return PollResult(
                False,
                failure="oom" if oom else ("job_preempted" if retriable else "job_failed"),
                detail=f"[{status}] {detail}",
            )
        # Capacity backstop: bound how long the job may sit IN_QUEUE (no worker has accepted it).
        # The throttled/unhealthy fast-fails below only arm when endpoint_health succeeds AND RunPod
        # reports a THROTTLED/UNHEALTHY worker; a queue with zero workers, one stuck INITIALIZING, or
        # a health probe that keeps erroring (its block is wrapped in `except: pass`) bypasses them and
        # would otherwise wait the full setup_grace_s (~50 min). Keyed off the authoritative job status
        # so it holds even when the health probe is blind: once IN_QUEUE exceeds queue_grace_s, return a
        # retryable stall so the runner's gpu-walk re-provisions on the next-best (in-capacity) class.
        now = time.time()
        if status == "IN_QUEUE":
            if queued_since is None:
                queued_since = now
            elif now - queued_since > queue_grace_s:
                return PollResult(
                    False,
                    failure="no_capacity",
                    detail=f"never scheduled: job stuck IN_QUEUE for {int(now - queued_since)}s "
                    "(no RunPod capacity for the pinned GPU class); retrying on the next-best GPU",
                )
        else:
            queued_since = None
        # While queued, surface worker availability (throttled hosts are the common
        # cause of silent multi-minute waits — make them visible in the run log).
        if status == "IN_QUEUE" and now - last_health_probe > 90:
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
                # Fail fast on a worker stuck UNHEALTHY: a dead worker / failed image pull won't
                # self-recover, so don't burn the full setup_grace_s (~50 min) waiting on it — once
                # it has stayed unhealthy with nothing usable or (re)initializing for
                # unhealthy_grace_s, return a (retryable) stall so the runner re-provisions a FRESH
                # endpoint (fresh image pull, likely a different host). Observed: a mutable image
                # tag republished mid-pull corrupts the worker -> unhealthy, and a fresh pull fixes it.
                if workers.get("unhealthy") and not usable and not recovering:
                    if unhealthy_since is None:
                        unhealthy_since = time.time()
                    elif time.time() - unhealthy_since > unhealthy_grace_s:
                        return PollResult(
                            False,
                            failure="stalled",
                            detail=f"worker stuck unhealthy for "
                            f"{int(time.time() - unhealthy_since)}s while IN_QUEUE (likely a failed "
                            f"image pull); retrying on a fresh endpoint",
                        )
                else:
                    unhealthy_since = None  # recovered / usable worker appeared
                # Fail fast on a worker stuck THROTTLED: RunPod has no capacity for the pinned GPU
                # class/pool and a throttled worker won't self-recover, so don't burn the full
                # setup_grace_s (~50 min) waiting on it. Once it has stayed throttled with nothing
                # usable or (re)initializing for throttled_grace_s, return a (retryable) stall so
                # the runner's gpu-walk re-provisions on the NEXT-BEST GPU class — the cheapest fit
                # often has no capacity while the next-best (a few cents/hr more) does.
                if workers.get("throttled") and not usable and not recovering:
                    if throttled_since is None:
                        throttled_since = time.time()
                    elif time.time() - throttled_since > throttled_grace_s:
                        return PollResult(
                            False,
                            failure="no_capacity",
                            detail=f"never scheduled: worker stuck THROTTLED for "
                            f"{int(time.time() - throttled_since)}s while IN_QUEUE (no RunPod "
                            f"capacity for the pinned GPU class); retrying on the next-best GPU",
                        )
                else:
                    throttled_since = None  # capacity appeared / usable worker
            except Exception:
                # Health surfacing is diagnostic only; a probe failure must not stop polling.
                pass
        # heartbeat progress surfacing + stall detection
        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            # Gate progress on the heartbeat's OWN ts (key = (stage, step, ts, attempt)) ADVANCING: a
            # stale heartbeat that lands late — a newer one skipped by the bounded _HB_UPLOAD_LOCK
            # while an older slow upload finishes — carries an OLDER ts and must NOT buy a fresh stall
            # window for a genuinely stuck worker. Unlike Lambda/Hyperstack we keep crediting the
            # control-plane poll time to last_progress (RunPod's stall math stays in one clock, no
            # worker/control-plane skew); only the DECISION to count it as progress is ts-monotonic.
            hb_ts = new_key[2] if new_key else None
            hb_step = new_key[1] if new_key else None
            hb_attempt = _attempt_int(new_key[3]) if new_key else None
            # Cold-start setup OVER -> tighten to the training window. Shared with lambdalabs: a setup
            # stage never tightens; rl_step/sft_step tighten only at a COMPLETED step (step >= 1) so a
            # step=0 gap-fill during the silent cold first step keeps setup grace; every other non-setup
            # stage (POST-training rl_trained / *_train_done / metrics, no step field) tightens so a hung
            # teardown falls under the tight window. See is_training_heartbeat for the full rationale.
            is_training_hb = is_training_heartbeat(stage, hb_step)
            if hb_attempt is not None and hb_attempt > last_hb_attempt:
                # Newer attempt = a fresh worker after a retry/preemption. Attempts SHARE this run's HF
                # heartbeat path, and the new worker restarts from cold setup, so reset the ts baseline
                # and re-derive the latch from ITS heartbeat (regaining setup grace) rather than letting
                # the prior attempt's training latch suppress the new cold start.
                last_hb_attempt = hb_attempt
                last_hb_ts = hb_ts or 0.0
                last_progress = time.time()
                seen_heartbeat = is_training_hb
            elif (hb_attempt is None or hb_attempt == last_hb_attempt) and (
                hb_ts is None or hb_ts > last_hb_ts
            ):
                # Same (or attempt-less) heartbeat stream: gate progress on the heartbeat's OWN ts
                # ADVANCING (key = (stage, step, ts, attempt)). A stale heartbeat that lands late — a
                # newer one skipped by the bounded _HB_UPLOAD_LOCK while an older slow upload finishes —
                # carries an OLDER ts and must NOT buy a fresh stall window for a stuck worker. Unlike
                # Lambda/Hyperstack we keep crediting the control-plane poll time to last_progress
                # (RunPod's stall math stays in one clock, no worker/control-plane skew); only the
                # DECISION to count it as progress is ts-monotonic.
                if hb_ts is not None:
                    last_hb_ts = hb_ts
                last_progress = time.time()
                if is_training_hb:
                    seen_heartbeat = True
            # else: an OLDER attempt's late heartbeat, or a stale ts within the current attempt — ignore
            # it entirely (no progress credit, no latch change).
        # Cold start (before any training-phase heartbeat) gets the larger setup_grace_s,
        # but only when a heartbeat_reader lets us tell setup from training; without one we
        # can't, so stay on stall_after_s (no regression).
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
) -> PollResult:
    """Durable equivalent of ``submit_train``: deploy, submit, persist handle, poll.

    ``on_handle(handle_dict)`` is invoked as soon as the job is queued so the
    runner can persist {endpoint_id, job_id} for cross-process reattach.

    ``on_last_gpu`` tells the no-capacity backstops no further GPU attempt will follow this one
    (candidate list exhausted OR retry budget exhausted), so there is no next-best class to walk to
    and they wait longer before giving up (see ``stall_kwargs``).
    """
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import _run_suffix, build_worker_env, chalk_extra_pip

    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    # Per-attempt endpoint name: a retry must land on a genuinely fresh endpoint —
    # reusing the name lets the SDK/platform pin the job back onto the same
    # (possibly throttled/sick) host.
    suffix = _run_suffix(spec.run_id)
    if attempt:
        suffix = f"{suffix}r{attempt}"
    # Resolve worker pip deps BEFORE provisioning, so deterministic dependency issues surface
    # before the endpoint exists.
    # extra_pip runs for EVERY job here (the durable baked-image path skips resolve_worker_deps
    # in build_function_input, but _train_body always pip-installs extra_pip), so the chalk spec
    # is appended here to reach default runs.
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
    }
    try:
        job_id = runpod_api.submit_job(endpoint_id, build_function_input(payload))
    except Exception:
        # The endpoint is registered but no run handle exists yet, and a
        # retry endpoint's rN-suffixed name can't be reconstructed from the run
        # id later — delete it now so a transient submit failure doesn't leak a
        # serverless endpoint against the account quota.
        with contextlib.suppress(Exception):
            runpod_api.delete_endpoint(endpoint_id)
        raise
    handle = JobHandle(endpoint_id, name, job_id)
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
        make_hf_failure_detail_reader(hf_repo, prefix, spec.phase) if hf_repo else None
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
    """Rate-limited reader for one HF artifact's text content (None until it exists).

    Generic helper for HF-backed worker artifacts and heartbeats. ``read(force=False)``
    re-downloads at most once per
    ``min_interval_s`` (``force=True`` bypasses the gate); it never raises — any HF error
    (artifact absent, network) returns None.
    """
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
    """Reader for the worker's heartbeat.json on HF (rate-limited, never raises).

    Thin JSON-parsing wrapper over :func:`make_hf_text_reader` bound to ``{prefix}/heartbeat.json``.
    """
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
):
    """Reader for worker-uploaded RunPod failure artifacts on HF.

    The RunPod queue often reports only the outer handler error (for example, "produced no
    /tmp/metrics.json"). The worker writes the actual traceback and console tail to HF; this
    reader lets ``poll_job`` force-download those files after a terminal worker failure.
    """
    error_reader = make_hf_text_reader(hf_repo, f"{prefix}/error_{phase}.txt", min_interval_s)
    console_reader = make_hf_text_reader(
        hf_repo, f"{prefix}/console_{phase}.txt", min_interval_s
    )

    def read(force: bool = False) -> str | None:
        parts: list[str] = []
        error_text = error_reader(force=force)
        if error_text:
            parts.append(f"--- error_{phase}.txt ---\n{error_text[-4000:]}")
        console_text = console_reader(force=force)
        if console_text:
            parts.append(f"--- console_{phase}.txt ---\n{console_text[-4000:]}")
        return "\n".join(parts) if parts else None

    return read


def _once_forced(heartbeat_reader):
    """Wrap a heartbeat reader so the underlying ``force=True`` read happens AT MOST ONCE, then every
    call (whatever its ``force`` arg) returns that one snapshot. A terminal path can then surface the
    heartbeat AND extract the ``retriable``/``oom`` flags from a SINGLE read instead of three — fewer
    HF round-trips, and no risk of the flags coming from different snapshots if the worker updates the
    heartbeat mid-sequence. Returns ``None`` when the reader is ``None`` (callers already handle it)."""
    if heartbeat_reader is None:
        return None
    cache: dict = {}

    def reader(force: bool = False):
        if "hb" not in cache:
            cache["hb"] = heartbeat_reader(force=True)
        return cache["hb"]

    return reader


def worker_flagged_retriable(heartbeat_reader) -> bool:
    """True if the worker stamped ``retriable`` (a RetriableInfraError) in its last heartbeat — the
    structured worker<->poller contract that replaces failure-detail parsing: ``retriable`` means
    retry on a fresh worker. Forces a fresh read past the rate limit."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict):
        return False
    return bool(hb.get("retriable"))


def worker_flagged_oom(heartbeat_reader, current_attempt: int | None = None) -> bool:
    """True if the worker stamped ``oom`` (a CUDA out-of-memory crash) in its last heartbeat — the
    structured signal that lets the runner ESCALATE the retry to a strictly larger GPU instead of
    failing fast on a too-small card. Same fresh-read contract as ``worker_flagged_retriable``.

    ``current_attempt`` gates against a STALE flag: the heartbeat file lives at a per-seed prefix
    SHARED across attempts (the prefix has no attempt segment), so after an OOM escalation the prior
    attempt's ``{"oom": true}`` lingers until the new worker overwrites it. If the escalated endpoint
    dies before writing its own heartbeat (a platform/startup failure on the bigger card), this
    forced read would otherwise misclassify that unrelated failure as an OOM and escalate AGAIN onto
    a needlessly expensive card. When the caller passes the attempt it is polling, we trust the flag
    only when the heartbeat was stamped by THAT attempt (the worker stamps ``attempt`` in every
    heartbeat). ``None`` (e.g. the reattach poll, which has no expected attempt) keeps the historical
    trust-the-flag behavior — no regression. OOM is gated but ``retriable`` is not: a stale retriable
    flag only retries on a fresh worker (which a platform failure warrants anyway), whereas a stale
    OOM wrongly burns a larger, costlier GPU class."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict):
        return False
    if not hb.get("oom"):
        return False
    # A prior attempt's lingering OOM flag must not escalate this attempt: trust it only when the
    # heartbeat was stamped by the attempt we're polling (current_attempt None = caller has no
    # expected attempt, e.g. the reattach poll → keep the historical trust-the-flag behavior).
    return current_attempt is None or _attempt_int(hb.get("attempt")) == current_attempt


def worker_flagged_crash(heartbeat_reader, current_attempt: int | None = None) -> bool:
    """True iff the worker stamped a TERMINAL error heartbeat (stage ``error_*``) for
    ``current_attempt`` — i.e. THIS attempt's worker actually ran and crashed, not a prior attempt's
    leftover. The worker always commits its ``error_*`` heartbeat (a terminal stage, never throttled),
    stamped with ``attempt``, so this is an attempt-scoped crash signal.

    Lets a dead-host crash/preempt split key on THIS attempt's own crash rather than the SHARED
    per-seed ``error_<phase>.txt`` artifact: that file is not attempt-scoped, so a prior attempt's
    traceback (e.g. an OOM that ESCALATED to a larger GPU rather than ending the run) lingers, and a
    host loss before this attempt's worker produced its own error would otherwise be read as this
    attempt's deterministic crash and failed fast instead of continuing recovery. ``None``
    current_attempt keeps the historical 'trust any error heartbeat' behavior."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    if not isinstance(hb, dict):
        return False
    if not str(hb.get("stage") or "").startswith("error_"):
        return False
    return current_attempt is None or _attempt_int(hb.get("attempt")) == current_attempt


def _terminal_exception_block(detail: str) -> str:
    """The LAST traceback in a multi-line stdout/error tail — i.e. the terminal exception.

    ``is_cuda_oom`` ANDs an OOM marker with CUDA/Triton context anywhere in the text it is given, so
    feeding it the WHOLE tail lets the two come from UNRELATED lines: a 'cuda' from an early init log
    plus an 'out of memory' from a later host-RAM/DataLoader OOM would falsely escalate. Restricting
    the scan to the final traceback forces both signals to come from the SAME terminal failure. When
    there is no traceback framing (a bare one-line handler error), fall back to the last few lines,
    where the terminal exception lives, rather than the whole tail."""
    marker = "Traceback (most recent call last):"
    idx = detail.rfind(marker)
    if idx != -1:
        return detail[idx:]
    return "\n".join(detail.splitlines()[-15:])


def detail_flags_cuda_oom(detail: str | None) -> bool:
    """Fallback OOM classifier for the cases the structured heartbeat ``oom`` flag misses: the OOM
    happened in a SUBPROCESS the worker didn't classify on its own exception (a vLLM EngineCore child
    whose CUDA OOM only reaches the parent via stdout), or the terminal error heartbeat upload was
    lost. Runs the SAME CUDA-specific predicate the worker uses (``is_cuda_oom``), so a host/library
    OOM in the text does NOT escalate.

    Scans only the TERMINAL exception block (the last traceback) of ``detail``, never the whole tail:
    ``is_cuda_oom`` ANDs an OOM marker with CUDA/Triton context across the text, so an unrelated earlier
    'cuda' line plus a later host-RAM/DataLoader 'out of memory' would otherwise falsely escalate onto
    a larger card (more VRAM can't fix that failure).

    IMPORTANT — callers must pass only ATTEMPT-FRESH text (this attempt's live job output / its
    attempt-scoped marker), NEVER the shared per-seed HF ``error_<phase>.txt`` artifact: that artifact
    is not attempt-scoped, so a prior attempt's lingering traceback could otherwise label an unrelated
    failure as an OOM and wrongly escalate (or raise the OOM floor on reattach). Skips any text
    carrying the retriable-infra marker so a current RetriableInfraError that merely names a CUDA OOM
    still retries on a fresh same-size GPU (the worker's retriable-wins rule); this lets callers gate
    on ``not oom`` alone, so a STALE ``retriable`` heartbeat can't block a real OOM. Fallback ONLY —
    the primary path stays the structured flag."""
    if not detail:
        return False
    from flash.engine.worker.perf.lifecycle import RETRIABLE_INFRA_MARKER, is_cuda_oom

    if RETRIABLE_INFRA_MARKER.lower() in detail.lower():
        return False
    return is_cuda_oom(None, _terminal_exception_block(detail))
