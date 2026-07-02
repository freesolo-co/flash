"""Lambda Cloud run lifecycle: capacity walk -> launch -> HF-artifact poll -> guaranteed terminate.

Cost-safety: a launched instance is ALWAYS terminated — runner finally, poll deadline, cancel, and
sweep_orphans each independently guarantee it. No in-box self-destruct (unlike Vast); sweep_orphans
at startup is the crash backstop.

Constants tests monkeypatch stay in this __init__ so monkeypatch.setattr(jobs, …) still takes effect.
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
    is_training_heartbeat,
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

LOAD_TIMEOUT_S = 900.0
# Setup grace covers Docker pull + pip + model download (no heartbeat until training starts).
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0
PROVISION_GRACE_S = 3000.0

_METRICS_READ_RETRIES = 3
_METRICS_READ_BACKOFF_S = 2.0

_DEAD_STATES = {"terminated", "terminating", "preempted", "unhealthy"}


def resolve_ssh_key_names() -> list[str]:
    """Return the SSH key to attach at launch (required by Lambda even though we never SSH in)."""
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
    """Regions currently advertising capacity for the given GPU class. Empty = no Lambda capacity now."""
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
    """True when the launch was definitively rejected with NO instance created (safe to try next region).

    A 429, 5xx, timeout, or missing-id response is AMBIGUOUS — the provider may have billed an
    instance we can't see, so we must NOT issue another launch.
    """
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
    code_prefix: str | None = None,
) -> LambdaJobHandle:
    """Launch the first region that accepts the job; walk regions on a capacity rejection."""
    say = make_say(log)
    if not instances:
        raise lambda_api.LambdaApiError(
            f"no Lambda capacity for {spec.gpu.type} (no region advertises the instance type)"
        )
    cache_name = getattr(spec.gpu, "network_volume", None)
    cold_user_data = build_user_data(
        build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets, code_prefix=code_prefix),
        gpu=spec.gpu.type,
    )

    def _cache_user_data_for(mount_point: str) -> str:
        """Build user_data with this region's actual NFS mount point."""
        return build_user_data(
            build_payload(
                spec, seed, attempt, runtime_secrets=runtime_secrets,
                cache_host_mount=mount_point, mode=mode, models=models, code_prefix=code_prefix,
            ),
            gpu=spec.gpu.type,
        )

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
        user_data, fs_names = cold_user_data, None
        if cache_name:
            try:
                mount_point = lambda_api.ensure_filesystem(cache_name, inst.region)
                # Rebuild user_data when the actual mount_point differs from the default (rare).
                region_user_data = (
                    cache_user_data if mount_point == default_cache_mount
                    else _cache_user_data_for(mount_point)
                )
                user_data, fs_names = region_user_data, [cache_name]
            except Exception as e:
                # Preload must not fall back cold — it would train instead of warming the cache.
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
                # Ambiguous: Lambda may have billed an instance we never got an id for — don't launch
                # again; reap by run-name and let the runner retry fresh.
                say(f"ambiguous launch failure in {inst.region}: {e}; reconciling + retrying fresh")
                with contextlib.suppress(Exception):
                    terminate_run_instances(spec.run_id)
                raise lambda_api.LambdaApiError(
                    f"ambiguous Lambda launch failure (possible phantom reaped): {e}"
                ) from e
            say(f"region {inst.region} ({inst.gpu} {inst.instance_type}) rejected: {e}")
            # Filesystem-attach errors: retry once without the cache before walking (clean reject = safe).
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
            # Preload must not refresh to a different region (would warm the wrong one).
            if mode != "preload" and not candidates and not refreshed:
                refreshed = True
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
    # Reap any phantom instance (accepted but no id returned) before giving up.
    with contextlib.suppress(Exception):
        terminate_run_instances(spec.run_id)
    raise lambda_api.LambdaApiError(
        f"all {len(tried_regions)} Lambda region(s) rejected the {spec.gpu.type} launch "
        f"(no capacity): {last_err}"
    )


# Tests monkeypatch this name so keep it as a module-level alias.
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(hf_repo: str, prefix: str, phase: str, marker: dict | None, attempt: int) -> str:
    """Assemble failure detail from HF artifacts (boot.log is the only early-bootstrap log source)."""
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err_name = f"error_{phase}_attempt{int(attempt or 0)}.txt"  # matches worker error_artifact_name
    err = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
    if err:
        parts.append(f"--- {err_name} ---\n{err[-2000:]}")
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
    """Poll instance status + HF artifacts to a terminal state."""
    say = make_say(log)

    # 0.0 = corrupt/missing handle; fall back to now to avoid billing from the 1970 epoch.
    launch_ts = handle.started_ts or time.time()

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/lambda_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")
    # Absence of boot.log while active = cloud-init never ran (sick region / stuck host).
    boot_log_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/lambda_attempt{handle.attempt}_boot.log", min_interval_s=60.0
    )

    def finish_ok(done_content: str | None = None) -> PollResult:
        # metrics.json None here is a transient HF blip (worker uploads it before DONE); retry rather
        # than treating a successful run as job_failed (which is not infra-retried).
        raw = metrics_reader(force=True)
        for _ in range(_METRICS_READ_RETRIES):
            if raw is not None:
                break
            time.sleep(_METRICS_READ_BACKOFF_S)
            raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(
                False,
                failure="poll_error",
                detail="DONE seen but metrics.json unreadable after retries (transient HF read)",
            )
        metrics = json.loads(raw)
        # Prefer the worker's DONE timestamp when present and sane; fall back to now. On delayed
        # recovery the control plane may poll hours after the box wrote DONE, so billing to now
        # would over-bill by the downtime.
        end_ts = time.time()
        if done_content:
            try:
                done_ts = float(done_content.strip())
                if launch_ts <= done_ts <= end_ts:
                    end_ts = done_ts  # prefer worker's timestamp to avoid over-billing on delayed recovery
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
        # 120s clock-skew grace; rejects leftover DONE from a prior attempt.
        try:
            return float(content.strip()) > launch_ts - 120.0
        except ValueError:
            return False

    def finish_from_ok_marker() -> PollResult:
        d = done_reader(force=True)
        return finish_ok(d if (d is not None and done_is_fresh(d)) else None)

    def fail_from_marker(marker: dict | None) -> PollResult:
        from flash.providers.runpod.jobs import worker_flagged_retriable

        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(heartbeat_reader)
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.attempt),
        )

    def terminal_artifact_result() -> PollResult | None:
        """Force-read terminal artifacts; preserves work done during a control-plane outage."""
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
    # Anchor all clocks to launch (not poll start) so a reattach doesn't hand a timed-out box a fresh window.
    start = launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_active = False
    active_since = start  # launch-anchored; advanced to now only on a genuine inactive->active transition
    observed_active_since = None  # when THIS session first saw active (unset on reattach until first read)
    seen_training_hb = False
    seen_fresh_hb = False  # any fresh hb (boot or training) disarms the first-liveness deadline
    boot_log_seen = False  # latched once present; guards against rate-limited None after first observe
    boot_log_absent_polls = 0
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
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
            # Skip first observation (last_status=None on reattach) so the launch-anchored clock is not reset.
            if last_status is not None:
                last_progress = time.time()
                if status == "active":
                    active_since = time.time()
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
            terminal = terminal_artifact_result()
            if terminal is not None:
                return terminal
            # THIS attempt's error file present = deterministic crash (fail fast); absent = host loss
            # (retry). Attempt-scoped so a prior attempt's stale traceback can't force a false job_failed.
            # A retriable heartbeat keeps the path on job_preempted regardless.
            from flash.providers.runpod.jobs import worker_flagged_retriable

            err_name = f"error_{spec.phase}_attempt{int(handle.attempt or 0)}.txt"
            err = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
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
                return finish_from_ok_marker()

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
            # Use the heartbeat's own timestamp (clamped to [launch, now]); a stale reattach read
            # must not reset the stall clock to now. ``fresh`` is False for prior-attempt leftover hbs.
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts, handle.attempt)
            if fresh:
                seen_fresh_hb = True
                if stage is not None:
                    last_progress = max(last_progress, hb_ts)  # monotonic: never let progress regress
                    if is_training_heartbeat(stage, new_key[1]):
                        seen_training_hb = True
        if became_active:
            if (
                not seen_fresh_hb
                and not boot_log_seen
                and time.time() - active_since > first_liveness_s
                and time.time() - observed_active_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                # Empty "" boot.log counts as liveness (existence = cloud-init ran).
                if boot_log_reader(force=True) is None:
                    # Require absence to persist (BOOT_LOG_ABSENT_POLLS) to tolerate transient HF blips.
                    boot_log_absent_polls += 1
                    if boot_log_absent_polls >= BOOT_LOG_ABSENT_POLLS:
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
    code_prefix: str | None = None,
) -> PollResult:
    """Launch, poll, and always terminate the instance (finally is the cost-safety primary)."""
    if spec.gpu.type not in GPU_INFO:
        raise lambda_api.LambdaApiError(
            f"submit_run_lambda needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    instances = usable_instances(spec.gpu.type)
    handle = launch_and_submit(
        spec,
        seed,
        instances,
        attempt=attempt,
        log=log,
        runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
    )
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
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
    """Terminate every instance belonging to one run. Best-effort, never raises."""
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
    """Terminate flash-prefixed instances not owned by a live run; return terminated ids.

    ``known_labels``: multi-plane guard — only reap instances attributable to THIS plane's runs.
    Without it, two planes sharing one account mutually reap each other's live instances.
    Callables are resolved after listing so the protection set is current. Never raises.
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
        # Never fall through to an empty set — that would reap every live run's instance.
        logger.warning("lambda orphan sweep skipped: could not resolve run sets: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (labels or set())}
    # None = unscoped (single-plane); empty set = this plane owns nothing, reaps nothing.
    known_prefixes = None if known_labels is None else {run_label_prefix(a) for a in (known or set())}

    def _matches(prefixes: set[str]) -> bool:
        # Boundary match: flash-100 must not match flash-1000-...
        return any(name == p or name.startswith(p + "-s") for p in prefixes)

    now = time.time()
    orphans: list[str] = []
    for inst in instances:
        name = str(inst.get("name") or "")
        if not name.startswith("flash-"):
            continue
        # Preload boxes are exempt (driver-owned, not in run DB) UNLESS past their wall deadline + grace.
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
            continue
        if known_prefixes is not None and not _matches(known_prefixes):
            continue
        iid = inst.get("id")
        if iid:
            orphans.append(str(iid))
    deleted = lambda_api.terminate_instances(orphans) if orphans else []
    for iid in deleted:
        logger.warning("terminated orphaned lambda instance %s", iid)
    return deleted
