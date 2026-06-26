"""Lambda Cloud run lifecycle: capacity walk -> launch -> HF-artifact poll -> guaranteed terminate.

The Lambda equivalent of ``providers/runpod/jobs.py``. Lambda has no serverless queue: we launch a
single-GPU instance from a region with capacity, ship a self-contained cloud-init ``user_data``
(``builders.build_user_data``) that runs the prebuilt ``WORKER_IMAGE`` via Docker, and detect
completion purely via the worker's HF artifacts (DONE/metrics.json/heartbeat.json) + the instance's
status — no inbound network to the box is ever needed.

Cost-safety invariant: a launched instance is ALWAYS terminated — the runner's ``finally``, the
poll deadline, the cancel path, and ``sweep_orphans`` (server startup / post-run) each independently
guarantee it. Lambda has no instance-scoped key, so (unlike Vast) there is no in-box self-destruct;
``sweep_orphans`` at control-plane startup is the crash backstop.

The pure dataclasses + builders live in ``.builders`` and are re-exported here so the import path
``flash.providers.lambdalabs.jobs`` is unchanged. The lifecycle functions and the constants tests
monkeypatch stay in this ``__init__`` so a ``monkeypatch.setattr(jobs, …)`` still takes effect.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable

from flash._logging import get_logger
from flash.providers._poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    FIRST_LIVENESS_S,
    PollErrorTracker,
    heartbeat_progress_ts,
    make_say,
    preload_box_reap_due,
    surface_heartbeat,
)
from flash.providers.base import GPU_INFO, PollResult
from flash.providers.lambdalabs import api as lambda_api
from flash.providers.lambdalabs.jobs.builders import (
    LambdaInstance,
    LambdaJobHandle,
    build_payload,
    build_user_data,
    instance_label,
    run_label_prefix,
)
from flash.providers.runpod.jobs import make_hf_heartbeat_reader, make_hf_text_reader

logger = get_logger(__name__)

# How long an instance may sit in a non-active state (provisioning) before we give up and retry.
LOAD_TIMEOUT_S = 900.0
# No-progress window once the instance is active. The cold start on Lambda is dominated by the
# Docker image pull on a fresh host + per-run pip install + model download, none of which emits a
# heartbeat — so until a *training* heartbeat arrives we apply the larger ``SETUP_GRACE_S`` budget;
# after it we use the tight ``STALL_AFTER_S``.
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0
# Provision + cold-start grace added on top of the run's wall cap for the client-side poll deadline
# (Lambda has no server-side execution timeout, so the client deadline + the bootstrap's own cap
# bound spend). Larger than RunPod's because of the on-host Docker pull.
PROVISION_GRACE_S = 3000.0

# Heartbeat stages emitted DURING cold start, before the training loop begins. Receiving one proves
# the worker is alive but NOT that setup finished, so they keep the larger setup grace (cf. RunPod).
_SETUP_HEARTBEAT_STAGES = frozenset(
    {"boot", "sft_start", "rl_start", "sft_model_load", "rl_train_start"}
)

# Lambda instance statuses that mean "the box is gone / will not progress".
_DEAD_STATES = {"terminated", "terminating", "preempted", "unhealthy"}


def resolve_ssh_key_names() -> list[str]:
    """The (single) SSH key name to attach at launch.

    Lambda REQUIRES exactly one SSH key on every launch, even though the box is bootstrapped via
    cloud-init ``user_data`` and we never SSH in. Resolve it from ``LAMBDA_SSH_KEY_NAME`` if set,
    else the first key registered on the account. Raises a clear error if the account has none.
    """
    import os

    pinned = os.environ.get("LAMBDA_SSH_KEY_NAME")
    if pinned:
        return [pinned]
    keys = lambda_api.list_ssh_keys()
    names = [k.get("name") for k in keys if k.get("name")]
    if not names:
        raise lambda_api.LambdaApiError(
            "Lambda launch requires an SSH key on the account, but none are registered and "
            "LAMBDA_SSH_KEY_NAME is unset; add one in the Lambda console (the box is bootstrapped "
            "via user_data, so the key is unused — any key works)."
        )
    return [names[0]]


def usable_instances(gpu_class: str, force: bool = False) -> list[LambdaInstance]:
    """Launchable (region) candidates for a managed GPU class, only where capacity exists now.

    Lambda prices per instance type (not per region), so every candidate for a class carries the
    same $/hr; the list is the set of regions currently advertising capacity. Empty == the class
    has no Lambda capacity right now (the allocator skips it; a mid-run vanish is handled by the
    region walk + the runner's retry). ``force`` bypasses the ``/instance-types`` cache — used by the
    in-launch refresh so it can actually discover newly-freed regions rather than re-reading the
    just-populated allocation cache.
    """
    from flash.providers.lambdalabs.gpus import instance_type_for
    from flash.providers.lambdalabs.pricing import hourly_rate

    info = GPU_INFO[gpu_class]
    itype = instance_type_for(gpu_class)
    rate = hourly_rate(gpu_class)
    return [
        LambdaInstance(
            gpu=gpu_class,
            instance_type=itype,
            region=region,
            vram_gb=info.vram_gb,
            price_usd_hr=rate,
        )
        for region in lambda_api.regions_with_capacity(itype, force=force)
    ]


def _launch_rejection_is_clean(err: Exception) -> bool:
    """True when a launch error is a DEFINITIVE rejection that created NO instance (safe to walk to
    the next region). The shared RestClient fast-fails a non-429 4xx as ``... -> HTTP 4xx: ...``
    (the provider rejected the request outright, e.g. no capacity). Anything else — a 429
    (rate-limited), a 5xx / timeout (``failed after N attempts``), or a 2xx whose response lacked an
    id (``returned no instance id``) — is AMBIGUOUS: the provider may have created a billed instance,
    so we must NOT issue another launch."""
    s = str(err)
    return "-> HTTP 4" in s and "HTTP 429" not in s


def launch_and_submit(
    spec,
    seed: int,
    instances: list[LambdaInstance],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
    mode: str | None = None,
    models: list | None = None,
) -> LambdaJobHandle:
    """Launch the first region that accepts the job; walk regions on a capacity rejection.

    Capacity is a live market — between the allocator's capacity check and the launch the only
    region with capacity is often taken. We walk every advertised region, then refresh the capacity
    list once.

    ``mode="preload"`` + ``models`` launches a download-only warm (the bootstrap pulls the models into
    the mounted cache and exits — no worker); the cache user_data carries the preload payload.
    """
    say = make_say(log)
    if not instances:
        raise lambda_api.LambdaApiError(
            f"no Lambda capacity for {spec.gpu.type} (no region advertises the instance type)"
        )
    # Weight cache: when the run wants it (runner-assigned network_volume), HF_HOME points at the
    # Lambda filesystem bind-mounted at /lambda/nfs/<name> (region-independent path -> one user_data
    # serves every region; we just ensure the FS exists per region in the walk). If the FS can't be
    # ensured in a region, fall back to the cold user_data there. ``gpu=`` selects the per-GPU worker
    # image (dev: worker_image_for_gpu via lambda_image).
    cache_name = getattr(spec.gpu, "network_volume", None)
    cold_user_data = build_user_data(
        build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets), gpu=spec.gpu.type
    )

    def _cache_user_data_for(mount_point: str) -> str:
        """Cache user_data whose bind-mount targets THIS region's actual NFS host path."""
        return build_user_data(
            build_payload(
                spec, seed, attempt, runtime_secrets=runtime_secrets,
                cache_host_mount=mount_point, mode=mode, models=models,
            ),
            gpu=spec.gpu.type,
        )

    # Prebuild the cache user_data for the DEFAULT mount path (/lambda/nfs/<name>) once — the common
    # case, so the walk reuses it without re-rendering. A region whose ensure_filesystem returns a
    # DIFFERENT mount_point rebuilds with that real path (see the walk below), so the bootstrap
    # bind-mount never points at a stale host path.
    default_cache_mount = f"/lambda/nfs/{cache_name}" if cache_name else ""
    cache_user_data = _cache_user_data_for(default_cache_mount) if cache_name else None
    name = instance_label(spec.run_id, seed, attempt)
    ssh_keys = resolve_ssh_key_names()

    tried_regions: set[str] = set()
    candidates = list(instances)
    refreshed = False
    last_err: Exception | None = None
    while candidates:
        inst = candidates.pop(0)
        if inst.region in tried_regions:
            continue
        tried_regions.add(inst.region)
        # Ensure the cache filesystem exists in THIS region (create-if-absent) and attach it at
        # launch; on any failure, launch cold here (best-effort cache, never blocks the run).
        user_data, fs_names = cold_user_data, None
        if cache_name:
            try:
                mount_point = lambda_api.ensure_filesystem(cache_name, inst.region)
                # Use the FS's ACTUAL mount_point: Lambda auto-mounts the NFS filesystem on the host
                # there, and the bootstrap bind-mounts that exact path into the container. If it's the
                # default /lambda/nfs/<name> (the usual case) reuse the prebuilt user_data; otherwise
                # rebuild for this region so the bind-mount doesn't point at a stale/wrong host path
                # (which would silently run cold / fail the preload mount check).
                region_user_data = (
                    cache_user_data if mount_point == default_cache_mount
                    else _cache_user_data_for(mount_point)
                )
                user_data, fs_names = region_user_data, [cache_name]
            except Exception as e:
                # A preload run's WHOLE purpose is to warm the cache; the cold user_data carries no
                # mode/models, so a cold fallback would run a full training bootstrap (GPU billing,
                # timeout) and warm nothing. SKIP this region instead — try the next one, and fail the
                # walk if no region can host the cache. Normal runs still degrade to a cold run.
                if mode == "preload":
                    say(f"weight cache unavailable in {inst.region} ({e}); skipping (preload needs it)")
                    last_err = e
                    continue
                say(f"weight cache unavailable in {inst.region} ({e}); launching cold")
        try:
            instance_id = lambda_api.launch_instance(
                region_name=inst.region,
                instance_type_name=inst.instance_type,
                ssh_key_names=ssh_keys,
                name=name,
                user_data=user_data,
                file_system_names=fs_names,
            )
        except lambda_api.LambdaApiError as e:
            last_err = e
            if not _launch_rejection_is_clean(e):
                # Ambiguous failure (timeout / 5xx / 429 / accepted-but-no-id): Lambda may have
                # created a billed instance whose id we never got. Do NOT launch another in this
                # attempt — reconcile any phantom by run-name and stop; the runner's retry (+ gc /
                # sweep_orphans) re-provisions cleanly. This is the non-idempotent-launch cost-safety
                # the region walk would otherwise violate.
                say(f"ambiguous launch failure in {inst.region}: {e}; reconciling + retrying fresh")
                with contextlib.suppress(Exception):
                    terminate_run_instances(spec.run_id)
                raise lambda_api.LambdaApiError(
                    f"ambiguous Lambda launch failure (possible phantom reaped): {e}"
                ) from e
            say(f"region {inst.region} ({inst.gpu} {inst.instance_type}) rejected: {e}")
            # A CLEAN reject of a CACHE-backed launch whose error mentions the FILESYSTEM was likely
            # caused by the attach itself (a just-created FS not yet attachable, an attach quota, an
            # unsupported pairing) — not the GPU class. Best-effort cache must never make a region the
            # cold path could have served fail outright, so retry THIS region once WITHOUT the cache
            # before walking. Gated to filesystem-shaped errors so a plain capacity reject still walks
            # (a cold retry there would just reject again). Skipped in preload mode (a cache-less
            # preload warms nothing). The reject was clean -> no billed instance -> a 2nd launch is safe.
            fs_attach_reject = fs_names and any(
                tok in str(e).lower() for tok in ("file_system", "filesystem", "file-system")
            )
            if mode != "preload" and fs_attach_reject:
                say(f"retrying {inst.region} WITHOUT the weight cache (attach may have caused the reject)")
                try:
                    instance_id = lambda_api.launch_instance(
                        region_name=inst.region, instance_type_name=inst.instance_type,
                        ssh_key_names=ssh_keys, name=name, user_data=cold_user_data,
                        file_system_names=None,
                    )
                except lambda_api.LambdaApiError as e2:
                    last_err = e2
                    if not _launch_rejection_is_clean(e2):
                        with contextlib.suppress(Exception):
                            terminate_run_instances(spec.run_id)
                        raise lambda_api.LambdaApiError(
                            f"ambiguous Lambda launch failure (possible phantom reaped): {e2}"
                        ) from e2
                    say(f"region {inst.region} also rejected cold: {e2}")
                else:
                    say(
                        f"launched lambda instance {instance_id} (cold, cache-less): {inst.gpu} "
                        f"{inst.instance_type} in {inst.region} attempt={attempt} seed={seed}"
                    )
                    return LambdaJobHandle(
                        instance_id=instance_id, instance_type=inst.instance_type, region=inst.region,
                        name=name, gpu=inst.gpu, hourly_usd=inst.price_usd_hr, attempt=attempt,
                        started_ts=time.time(),
                    )
            # NOT in preload mode: warm_instances pins each preload launch to ONE specific target
            # region and reports that exact region as warmed. Refreshing to a DIFFERENT region here
            # would warm region B while the caller reports the target region A as warmed (cache still
            # cold). A preload that can't run in its target region must FAIL it (walk exhausts ->
            # raise), never silently warm another.
            if mode != "preload" and not candidates and not refreshed:
                refreshed = True
                # Force a fresh capacity fetch (the allocation cache is ~45s stale) so the refresh
                # can discover regions that freed up since the walk started.
                candidates = [
                    c for c in usable_instances(inst.gpu, force=True) if c.region not in tried_regions
                ]
            continue
        say(
            f"launched lambda instance {instance_id}: {inst.gpu} {inst.instance_type} "
            f"${inst.price_usd_hr:.2f}/hr in {inst.region} attempt={attempt} seed={seed}"
        )
        return LambdaJobHandle(
            instance_id=instance_id,
            instance_type=inst.instance_type,
            region=inst.region,
            name=name,
            gpu=inst.gpu,
            hourly_usd=inst.price_usd_hr,
            attempt=attempt,
            started_ts=time.time(),
        )
    # Phantom-instance safety: a non-idempotent launch Lambda ACCEPTED but whose response lacked a
    # parseable id raises (caught above as a region rejection), leaving a billed instance under our
    # run name that no handle owns. Best-effort reap any such instance by run-name before giving up.
    with contextlib.suppress(Exception):
        terminate_run_instances(spec.run_id)
    raise lambda_api.LambdaApiError(
        f"all {len(tried_regions)} Lambda region(s) rejected the {spec.gpu.type} launch "
        f"(no capacity): {last_err}"
    )


# Rate-limited reader for one HF artifact's text content (None until it exists). Shared with
# runpod's poller via make_hf_text_reader; kept under this module-local name because tests
# monkeypatch ``lambda.jobs._make_hf_file_reader`` and the poll/failure paths resolve it as a
# module global (so a monkeypatch still takes effect).
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(hf_repo: str, prefix: str, phase: str, marker: dict | None, attempt: int) -> str:
    """Best root-cause detail we can assemble from the HF artifacts.

    Lambda exposes NO instance console/log API, so the box's own attempt-scoped
    ``lambda_attempt<N>_boot.log`` (pushed to HF by the cloud-init host uploader) is the substitute
    for Vast's ``instance_logs`` — the only home of early-bootstrap failures (docker/GPU not ready,
    image-pull failure). It is attempt-scoped so a retry reads ITS OWN boot log, not a prior one.
    """
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if err:
        parts.append(f"--- error_{phase}.txt ---\n{err[-2000:]}")
    boot = _make_hf_file_reader(hf_repo, f"{prefix}/lambda_attempt{attempt}_boot.log")(force=True)
    if boot:
        parts.append(f"--- lambda_attempt{attempt}_boot.log (host) ---\n{boot[-3000:]}")
    return "\n".join(parts) or "lambda worker terminated without a DONE sentinel"


def poll_lambda_job(
    handle: LambdaJobHandle,
    spec,
    seed: int,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    setup_grace_s: float = SETUP_GRACE_S,
    stall_after_s: float = STALL_AFTER_S,
    first_liveness_s: float = FIRST_LIVENESS_S,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (cf. runpod.jobs.poll_job).

    COMPLETED   fresh DONE sentinel on HF -> metrics.json (cost stamped from the instance's $/hr).
    job_failed  attempt marker with ok=false (a real worker error; fails fast unless the worker
                flagged it retriable).
    job_preempted  instance died without DONE/marker (host loss) -> infra-shaped, retried.
    stalled     never became active within LOAD_TIMEOUT_S; OR became active but emitted NO liveness
                (no boot.log/heartbeat) within ``first_liveness_s`` (sick region / worker never
                started); OR heartbeat frozen past the setup/stall window; OR deadline passed.
    """
    say = make_say(log)

    # Single source of truth for "when did this instance launch". started_ts is a non-Optional float
    # that LambdaJobHandle.from_dict coerces to 0.0 when MISSING (old/corrupt handle), so 0.0 means
    # "unknown launch" (a real launch is a large epoch ts, never 0.0). Fall back to now so EVERY use
    # below -- the load/stall clocks AND done_is_fresh / finish_ok's wall+cost stamping -- treats a
    # recovered corrupt handle consistently, instead of billing/comparing from the 1970 epoch.
    launch_ts = handle.started_ts or time.time()

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/lambda_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")
    # Attempt-scoped host boot log: present within ~2 min iff THIS attempt's cloud-init actually ran
    # (the host uploader starts before the image pull). Its ABSENCE while active is the signal that
    # nothing ever started (see the first-liveness check). Throttled — only fetched once the deadline
    # is in question, never on the hot path.
    boot_log_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/lambda_attempt{handle.attempt}_boot.log", min_interval_s=60.0
    )

    def finish_ok(done_content: str | None = None) -> PollResult:
        raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
        # Prefer the worker's DONE timestamp when present and sane; fall back to now. On delayed
        # recovery the control plane may poll hours after the box wrote DONE, so billing to now
        # would over-bill by the downtime.
        end_ts = time.time()
        if done_content:
            try:
                done_ts = float(done_content.strip())
                if launch_ts <= done_ts <= end_ts:
                    end_ts = done_ts
            except ValueError:
                pass
        wall_h = (end_ts - launch_ts) / 3600.0
        metrics["cost_usd"] = round(wall_h * handle.hourly_usd, 6)
        notes = metrics.get("notes") if isinstance(metrics.get("notes"), dict) else {}
        notes.update(
            {
                "provider": "lambda",
                "lambda_rate_usd_hr": handle.hourly_usd,
                "lambda_gpu": handle.gpu,
                "lambda_instance_type": handle.instance_type,
                "lambda_region": handle.region,
            }
        )
        metrics["notes"] = notes
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        # DONE carries the worker's time.time(); 120 s of clock-skew grace. Anything older predates
        # this attempt (leftover from a prior attempt's resume). Uses launch_ts (not handle.started_ts)
        # so an unknown-launch (0.0) handle doesn't accept every leftover DONE as fresh.
        try:
            return float(content.strip()) > launch_ts - 120.0
        except ValueError:
            return False

    def finish_from_ok_marker() -> PollResult:
        # An ok marker means the worker finished (it wrote metrics.json before the marker), even if
        # the DONE sentinel is STALE — a retry that hit the worker's already-complete path restores
        # the prior attempt's metrics but leaves DONE at the old timestamp. Treat ok-marker + metrics
        # as terminal success; pass the DONE only when it's genuinely fresh (so cost bills to it).
        d = done_reader(force=True)
        return finish_ok(d if (d is not None and done_is_fresh(d)) else None)

    def fail_from_marker(marker: dict | None) -> PollResult:
        # A real worker error fails fast UNLESS it is flagged retriable — the host failure marker
        # (docker/GPU never ready) sets retriable=True, and the worker stamps it in heartbeat for a
        # RetriableInfraError; either retries on a fresh host like a platform termination.
        from flash.providers.runpod.jobs import worker_flagged_retriable

        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(heartbeat_reader)
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.attempt),
        )

    def terminal_artifact_result() -> PollResult | None:
        # One forced read of the worker's terminal HF artifacts (DONE / attempt ok-marker). Returns a
        # terminal PollResult when the worker definitively finished or errored, else None. Used both
        # when the host is dead AND before returning a recovered client-side-deadline `stalled`: a
        # control-plane outage longer than max_wall+grace must not discard a seed the worker actually
        # completed during the downtime (the deadline check would otherwise fire before any DONE read).
        d = done_reader(force=True)
        if d is not None and done_is_fresh(d):
            return finish_ok(d)
        raw = marker_reader(force=True)
        if raw:
            with contextlib.suppress(ValueError):
                m = json.loads(raw)
                if m.get("ok"):
                    return finish_from_ok_marker()  # finished (stale DONE ok)
                return fail_from_marker(m)
        return None

    poll_errors = PollErrorTracker(say, interval_s)
    # Seed the load/stall clocks from the instance's LAUNCH (launch_ts), not this poll's start: on a
    # delayed reattach after a control-plane restart the box has been billing since launch, so a
    # still-booting instance that already blew LOAD_TIMEOUT_S must fail over NOW instead of getting
    # another full window. launch_ts already maps an unknown-launch (0.0) handle to now (see above),
    # so a fresh launch is a no-op and a corrupt handle won't peg the clocks to the epoch.
    start = launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_active = False
    # When this instance became active, anchored to launch (like last_progress) so a reattach whose
    # first read is already ACTIVE measures the first-liveness window from the original launch — it
    # does NOT hand a box that has been silent since before a control-plane restart a fresh window.
    # Advanced to now only on a genuine inactive->active transition observed in this poll session.
    active_since = start
    # Wall-clock this poll session FIRST observed the box ``active`` (set once, never reset). Unlike
    # the launch-anchored ``active_since``, this is genuinely "how long WE have watched it active", so
    # a reattach whose first read is already ACTIVE doesn't immediately fast-fail a box that only just
    # came up — see FIRST_LIVENESS_OBSERVED_FLOOR_S.
    observed_active_since = None
    seen_training_hb = False
    # Any FRESH heartbeat from THIS attempt (boot stage included) proves the worker started — clears
    # the first-liveness deadline. Distinct from seen_training_hb (which gates only the tighter
    # training stall window). A leftover prior-attempt heartbeat (ts < launch) is NOT fresh, so it
    # cannot disarm the deadline on the retry into a sick region it must catch.
    seen_fresh_hb = False
    # Latched once the attempt's host boot.log is OBSERVED (even empty ""): its existence proves
    # cloud-init ran, so a later rate-limited reader None can't spuriously fail the box over as stalled.
    boot_log_seen = False
    # Consecutive forced boot.log reads that came back absent at/after the first-liveness deadline. A
    # lone None can be a transient HF/Hub error, so we require the absence to persist (see
    # BOOT_LOG_ABSENT_POLLS) before declaring the region sick.
    boot_log_absent_polls = 0
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            # A recovered run can blow a launch-anchored deadline on the FIRST reattach tick (the
            # outage lasted past max_wall+grace). Read terminal artifacts once before giving up: if
            # the worker finished/errored during the downtime, persist that instead of retrying.
            terminal = terminal_artifact_result()
            if terminal is not None:
                return terminal
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            inst = lambda_api.get_instance(handle.instance_id)
            poll_errors.reset()
        except lambda_api.LambdaApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        missing_streak = missing_streak + 1 if inst is None else 0
        status = (inst or {}).get("status") or ("missing" if inst is None else "unknown")
        if status != last_status:
            say(f"instance {handle.instance_id}: {status}")
            # Treat a status TRANSITION as progress, but NOT the first observation: last_status
            # starts None, so on a reattach the very first read always "changes" — counting it as
            # progress would overwrite the launch-anchored last_progress and hand a silent-since-
            # launch worker a fresh full setup grace after every control-plane restart.
            if last_status is not None:
                last_progress = time.time()
                if status == "active":
                    active_since = time.time()  # genuine inactive->active: start the liveness clock
            last_status = status
        if status == "active":
            became_active = True
            if observed_active_since is None:
                observed_active_since = time.time()

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        dead = missing_streak >= 3 or status in _DEAD_STATES
        if dead:
            # One forced final read: the worker may have finished right before the box was torn
            # down (the normal success order on this substrate).
            terminal = terminal_artifact_result()
            if terminal is not None:
                return terminal
            # Dead host with no ok-marker/DONE. Distinguish a genuine host LOSS (retry on a fresh
            # host/class) from a worker that actually RAN and CRASHED early -- before it could write
            # the attempt marker terminal_artifact_result() reads -- but DID leave error_{phase}.txt
            # (a bad env id, a config/code error, an OOM). That is a DETERMINISTIC worker error, so
            # fail FAST: classifying it job_preempted burns fresh GPUs re-running a crash that will
            # repeat. A crash the worker flagged retriable (RetriableInfraError, stamped in the
            # heartbeat) still retries, exactly like fail_from_marker. error_{phase}.txt is not
            # attempt-scoped, but this can't flip a genuine preemption to job_failed: a prior
            # attempt's NON-retriable crash already ended the run via this same branch, and a prior
            # retriable crash leaves a retriable heartbeat that keeps this path on job_preempted.
            from flash.providers.runpod.jobs import worker_flagged_retriable

            err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{spec.phase}.txt")(force=True)
            worker_crashed = bool(err and err.strip()) and not worker_flagged_retriable(heartbeat_reader)
            return PollResult(
                False,
                failure="job_failed" if worker_crashed else "job_preempted",
                detail=_failure_detail(hf_repo, prefix, spec.phase, None, handle.attempt),
            )

        raw_marker = marker_reader()
        if raw_marker:
            try:
                marker = json.loads(raw_marker)
            except ValueError:
                marker = None
            if marker and not marker.get("ok"):
                return fail_from_marker(marker)
            if marker and marker.get("ok"):
                return finish_from_ok_marker()  # ok marker + metrics == success (DONE may be stale)

        if not became_active and time.time() - start > LOAD_TIMEOUT_S:
            return PollResult(
                False,
                failure="stalled",
                detail=f"instance stuck in '{status}' for {int(time.time() - start)}s "
                f"(never became active; provisioning / host issue)",
            )

        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            # Credit the heartbeat's OWN timestamp, not the poll time: a heartbeat that was
            # already stale before a control-plane restart must not reset the stall clock to now
            # on the first reattach read (last_hb_key starts None, so even an old heartbeat looks
            # "new"). Clamped to [launch, now]. Healthy workers heartbeat well inside the stall
            # window, so their ts ~= now (no behavior change on the normal path). ``fresh`` is False
            # for a LEFTOVER heartbeat from a prior attempt (ts < launch); we then neither advance
            # last_progress nor mark training seen, so a stale training heartbeat can't arm the
            # tighter training stall window before this attempt overwrites the file. Dates against
            # ``launch_ts`` (NOT the raw handle.started_ts) so an unknown-launch (0.0) handle is
            # anchored to the SAME ``now`` reference as done_is_fresh / the load+stall clocks: a
            # leftover heartbeat predating this reattach is then consistently rejected instead of
            # blanket-trusted (which could otherwise arm the tighter training window off a prior
            # attempt's training heartbeat). On a real launch this is exactly handle.started_ts.
            # Pass handle.attempt so a prior attempt's still-shutting-down worker (heartbeating on the
            # shared seed path with ts > this launch) can't satisfy this attempt's first-liveness.
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts, handle.attempt)
            if fresh:
                last_progress = hb_ts
                seen_fresh_hb = True  # worker is alive (boot or later) -> first-liveness satisfied
                if stage not in _SETUP_HEARTBEAT_STAGES:
                    seen_training_hb = True
        # Before the first TRAINING heartbeat the box is still in the long cold start (Docker pull +
        # pip + model download), so use the larger setup grace; tighten only once training begins.
        if became_active:
            # Fast-failover: a box that reached OS 'active' but never started a worker (no fresh hb AND
            # no attempt-scoped boot.log past first_liveness_s) is a wedged host -> 'stalled' (infra-
            # shaped -> runner escapes cross-provider via #241), instead of burning the ~50min setup
            # grace. A healthy box (even mid image-pull) emits its boot.log within ~2 min. The observed-
            # active floor keeps a reattach from fast-failing a box that only just came up (active_since
            # is launch-anchored and may already be past first_liveness_s) — see the constant's note.
            if (
                not seen_fresh_hb
                and not boot_log_seen
                and time.time() - active_since > first_liveness_s
                and time.time() - observed_active_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                # force=True bypasses the rate-limit so absent-vs-present is accurate; ``is None`` keeps
                # an empty "" boot.log as liveness (its existence proves cloud-init ran), then latch.
                if boot_log_reader(force=True) is None:
                    # A lone None can be a transient HF error -> require the absence to persist across
                    # BOOT_LOG_ABSENT_POLLS before failing over, so a Hub blip doesn't burn a retry.
                    boot_log_absent_polls += 1
                    if boot_log_absent_polls >= BOOT_LOG_ABSENT_POLLS:
                        # The worker may have written DONE / its attempt marker right before the non-
                        # forced done/marker reads above were rate-limited; force-read terminals once
                        # so a finished or explicitly-failed attempt is preserved, not torn down.
                        terminal = terminal_artifact_result()
                        if terminal is not None:
                            return terminal
                        return PollResult(
                            False,
                            failure="stalled",
                            detail=f"no worker liveness (boot.log/heartbeat) for "
                            f"{int(time.time() - active_since)}s after instance became active "
                            f"(cloud-init/worker never started; limit {int(first_liveness_s)}s)",
                        )
                else:
                    boot_log_seen = True
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
                # Same terminal-artifact race as above: a run that finished right before the last non-
                # forced done/marker read was rate-limited must be preserved, not stalled + retried.
                terminal = terminal_artifact_result()
                if terminal is not None:
                    return terminal
                phase = "training" if seen_training_hb else "setup (pre-training)"
                return PollResult(
                    False,
                    failure="stalled",
                    detail=f"no worker progress for {int(time.time() - last_progress)}s "
                    f"during {phase} (instance status {status}, limit {int(limit)}s)",
                )
        time.sleep(interval_s)


def submit_run_lambda(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
) -> PollResult:
    """Lambda equivalent of ``runpod.jobs.submit_run``: launch, persist, poll, terminate.

    The ``finally`` terminate is the cost-safety primary: every exit path — success, failure,
    stall, exception, KeyboardInterrupt — tears the paid instance down.
    """
    if spec.gpu.type not in GPU_INFO:
        raise lambda_api.LambdaApiError(
            f"submit_run_lambda needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    instances = usable_instances(spec.gpu.type)
    handle = launch_and_submit(
        spec, seed, instances, attempt=attempt, log=log, runtime_secrets=runtime_secrets
    )
    # The instance is billing the MOMENT launch_and_submit returns; the teardown ``finally`` must
    # guard EVERYTHING after that point — including ``on_handle`` (persisting the handle can itself
    # raise) — so the paid box is terminated even if the handle is never persisted.
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        # Uniform per-GPU wait: poll_lambda_job uses its default FIRST_LIVENESS_S / SETUP_GRACE_S.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        return poll_lambda_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            deadline_s=deadline,
        )
    finally:
        lambda_api.terminate_instances([handle.instance_id])


def terminate_run_instances(run_id: str) -> list[str]:
    """Terminate every instance belonging to ONE run (names start with its run prefix).

    Cancel/GC path: unlike ``sweep_orphans`` this never looks at other runs, so it is safe to call
    while they are in flight. Best-effort: never raises.
    """
    if not run_id:
        return []
    try:
        instances = lambda_api.list_instances()
    except Exception:
        return []
    prefix = run_label_prefix(run_id)
    ids = [
        str(i.get("id"))
        for i in instances
        if i.get("id")
        and (str(i.get("name") or "") == prefix or str(i.get("name") or "").startswith(prefix + "-s"))
    ]
    return lambda_api.terminate_instances(ids) if ids else []


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
    known_labels: set[str] | Callable[[], set[str]] | None = None,
) -> list[str]:
    """Terminate Flash-named instances that no live run owns; return terminated ids.

    Run at server startup (crash recovery) and after runs. Only names carrying the ``flash-`` run
    prefix are ever touched — nothing else on the account is ours to terminate. ``active_labels``
    may be RAW run ids; each is passed through ``run_label_prefix`` so it matches the same forced
    prefix the instance names carry. Best-effort: never raises.

    ``active_labels`` may also be a CALLABLE returning that set — it is then resolved AFTER the
    instance list is fetched. The periodic in-lifetime sweep passes one so the protection set is
    read post-listing: any instance present in the list had its run's status row committed before
    the instance was launched (hence before this list call), so resolving the live set now is
    guaranteed to include it — closing the launch race where a run started after a pre-captured set
    could have its fresh worker reaped as a phantom orphan.

    ``known_labels`` (optional, RAW run ids or a callable) is the universe of runs THIS control
    plane has a record of. When supplied, an instance is reaped only if its name maps to one of
    them — instances whose run id this plane has never seen are left ALONE. This is the multi-plane
    guard: two control planes sharing one Lambda account each carry disjoint run ids, so without it
    each plane's sweep treats the OTHER's live instances as orphans (their run ids are absent from
    this plane's ``active_labels``) and terminates them — they mutually reap each other every sweep.
    ``None`` keeps the legacy unscoped behavior (reap every non-active ``flash-`` box), correct for
    the single-plane production setup. Resolving it (like ``active_labels``) failing skips the sweep.
    """
    try:
        instances = lambda_api.list_instances()
    except Exception as exc:
        logger.warning("lambda orphan sweep skipped: %s", exc)
        return []
    try:
        labels = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        # Resolving a protection/known set failed (e.g. a db/status read error in the callable). SKIP
        # the sweep — never fall through to an empty set, which would treat every live run's instance
        # as an orphan and reap it. Honors the "never raises" contract.
        logger.warning("lambda orphan sweep skipped: could not resolve run sets: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (labels or set())}
    # None => unscoped (legacy reap-all); a set => only instances attributable to one of THIS
    # plane's known runs are reapable (multi-plane safety). An empty known set means this plane
    # owns no runs at all -> it reaps nothing, never another plane's live boxes.
    known_prefixes = None if known_labels is None else {run_label_prefix(a) for a in (known or set())}

    def _matches(prefixes: set[str]) -> bool:
        # Name-boundary match (EQUAL or followed by the ``-s`` seed boundary) so ``flash-100`` can't
        # shield/claim ``flash-1000-...`` (or vice versa).
        return any(name == p or name.startswith(p + "-s") for p in prefixes)

    now = time.time()
    orphans: list[str] = []
    for inst in instances:
        name = str(inst.get("name") or "")
        if not name.startswith("flash-"):
            continue
        # Warm/preload boxes (``flash-preload-...``) are driver-owned: launched by
        # preload.warm_instances (mode="preload"), NEVER persisted in the run DB (so never in
        # ``active``), and self-terminated in _warm_one_instance's ``finally`` (and by startup
        # recover_runs). A catalog warm can outlast this ~10-min sweep, so reaping them by the bare
        # ``flash-`` prefix would kill an in-progress preload mid-download; normally exempt them.
        # EXCEPTION: a box still alive past its embedded wall deadline + grace has lost its driver (the
        # only thing that terminates instance providers — nothing on the box self-terminates the VM), so
        # reap it to bound the leak rather than exempt it forever (see preload_box_reap_due).
        if name.startswith("flash-preload-"):
            if preload_box_reap_due(name, now):
                iid = inst.get("id")
                if iid:
                    orphans.append(str(iid))
                    logger.warning(
                        "reaping orphaned lambda preload box %s (outlived its wall deadline + grace; "
                        "driver lost)", name)
            continue
        if _matches(active):
            continue  # a live run owns this box — protected
        # Multi-plane guard: with a known set, only reap boxes attributable to one of THIS plane's
        # runs. A box whose run id is absent belongs to ANOTHER control plane on the same account
        # (or predates this plane's registry) — never ours to terminate.
        if known_prefixes is not None and not _matches(known_prefixes):
            continue
        iid = inst.get("id")
        if iid:
            orphans.append(str(iid))
    deleted = lambda_api.terminate_instances(orphans) if orphans else []
    for iid in deleted:
        logger.warning("terminated orphaned lambda instance %s", iid)
    return deleted
