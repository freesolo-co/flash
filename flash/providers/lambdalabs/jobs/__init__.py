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

from flash._logging import get_logger
from flash.providers._poll import (
    PollErrorTracker,
    heartbeat_progress_ts,
    make_say,
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
) -> LambdaJobHandle:
    """Launch the first region that accepts the job; walk regions on a capacity rejection.

    Capacity is a live market — between the allocator's capacity check and the launch the only
    region with capacity is often taken. We walk every advertised region, then refresh the capacity
    list once.
    """
    say = make_say(log)
    if not instances:
        raise lambda_api.LambdaApiError(
            f"no Lambda capacity for {spec.gpu.type} (no region advertises the instance type)"
        )
    payload = build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets)
    user_data = build_user_data(payload)
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
        try:
            instance_id = lambda_api.launch_instance(
                region_name=inst.region,
                instance_type_name=inst.instance_type,
                ssh_key_names=ssh_keys,
                name=name,
                user_data=user_data,
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
            if not candidates and not refreshed:
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


def _failure_detail(hf_repo: str, prefix: str, phase: str, marker: dict | None) -> str:
    """Best root-cause detail we can assemble from the HF artifacts.

    Lambda exposes NO instance console/log API, so the box's own ``lambda_boot.log`` (pushed to HF
    by the cloud-init host uploader) is the substitute for Vast's ``instance_logs`` — the only home
    of early-bootstrap failures (docker/GPU not ready, image-pull failure).
    """
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if err:
        parts.append(f"--- error_{phase}.txt ---\n{err[-2000:]}")
    boot = _make_hf_file_reader(hf_repo, f"{prefix}/lambda_boot.log")(force=True)
    if boot:
        parts.append(f"--- lambda_boot.log (host) ---\n{boot[-3000:]}")
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
    deadline_s: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (cf. runpod.jobs.poll_job).

    COMPLETED   fresh DONE sentinel on HF -> metrics.json (cost stamped from the instance's $/hr).
    job_failed  attempt marker with ok=false (a real worker error; fails fast unless the worker
                flagged it retriable).
    job_preempted  instance died without DONE/marker (host loss) -> infra-shaped, retried.
    stalled     never became active within LOAD_TIMEOUT_S, heartbeat frozen, or deadline passed.
    """
    say = make_say(log)

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/lambda_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

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
                if handle.started_ts <= done_ts <= end_ts:
                    end_ts = done_ts
            except ValueError:
                pass
        wall_h = (end_ts - handle.started_ts) / 3600.0
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
        # this attempt (leftover from a prior attempt's resume).
        try:
            return float(content.strip()) > handle.started_ts - 120.0
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
            detail=_failure_detail(hf_repo, prefix, spec.phase, marker),
        )

    poll_errors = PollErrorTracker(say, interval_s)
    # Seed the load/stall clocks from the instance's LAUNCH (handle.started_ts), not this poll's
    # start: on a delayed reattach after a control-plane restart the box has been billing since
    # launch, so a still-booting instance that already blew LOAD_TIMEOUT_S must fail over NOW
    # instead of getting another full window. On a fresh launch started_ts ~= now (no-op).
    start = handle.started_ts if handle.started_ts is not None else time.time()
    last_status = None
    last_hb_key = None
    last_progress = start
    became_active = False
    seen_training_hb = False
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
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
            last_status = status
        if status == "active":
            became_active = True

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        dead = missing_streak >= 3 or status in _DEAD_STATES
        if dead:
            # One forced final read: the worker may have finished right before the box was torn
            # down (the normal success order on this substrate).
            done = done_reader(force=True)
            if done is not None and done_is_fresh(done):
                return finish_ok(done)
            raw_marker = marker_reader(force=True)
            marker = None
            if raw_marker:
                with contextlib.suppress(ValueError):
                    marker = json.loads(raw_marker)
            if marker is not None and marker.get("ok"):
                return finish_from_ok_marker()  # finished right before teardown (stale DONE ok)
            if marker is not None and not marker.get("ok"):
                return fail_from_marker(marker)
            # Dead host, no marker, no DONE: a host loss, not a worker code error -> retry on a
            # fresh host/class. Surface whatever the boot log captured.
            return PollResult(
                False,
                failure="job_preempted",
                detail=_failure_detail(hf_repo, prefix, spec.phase, None),
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
            # tighter training stall window before this attempt overwrites the file.
            hb_ts, fresh = heartbeat_progress_ts(new_key, handle.started_ts)
            if fresh:
                last_progress = hb_ts
                if stage not in _SETUP_HEARTBEAT_STAGES:
                    seen_training_hb = True
        # Before the first TRAINING heartbeat the box is still in the long cold start (Docker pull +
        # pip + model download), so use the larger setup grace; tighten only once training begins.
        if became_active:
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
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
    on_last_gpu: bool = False,
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
        # On the last GPU class there is nowhere left to walk, so be more patient before giving up.
        setup_grace = SETUP_GRACE_S * (1.5 if on_last_gpu else 1.0)
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        return poll_lambda_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            setup_grace_s=setup_grace,
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


def sweep_orphans(active_labels: set[str] | None = None) -> list[str]:
    """Terminate Flash-named instances that no live run owns; return terminated ids.

    Run at server startup (crash recovery) and after runs. Only names carrying the ``flash-`` run
    prefix are ever touched — nothing else on the account is ours to terminate. ``active_labels``
    may be RAW run ids; each is passed through ``run_label_prefix`` so it matches the same forced
    prefix the instance names carry. Best-effort: never raises.
    """
    try:
        instances = lambda_api.list_instances()
    except Exception as exc:
        logger.warning("lambda orphan sweep skipped: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (active_labels or set())}
    orphans: list[str] = []
    for inst in instances:
        name = str(inst.get("name") or "")
        if not name.startswith("flash-"):
            continue
        # Match on the name boundary, not a raw string prefix: a live run's prefix must EQUAL the
        # name or be followed by the ``-s`` seed boundary, so ``flash-100`` can't shield
        # ``flash-1000-...`` (or vice versa).
        if any(name == a or name.startswith(a + "-s") for a in active):
            continue
        iid = inst.get("id")
        if iid:
            orphans.append(str(iid))
    deleted = lambda_api.terminate_instances(orphans) if orphans else []
    for iid in deleted:
        logger.warning("terminated orphaned lambda instance %s", iid)
    return deleted
