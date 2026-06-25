"""Hyperstack run lifecycle: stock walk -> launch VM -> HF-artifact poll -> guaranteed delete.

The Hyperstack equivalent of ``providers/lambdalabs/jobs``. Hyperstack rents a single-GPU VM from a
region with stock, ships the shared cloud-init ``user_data`` (runs ``WORKER_IMAGE`` via Docker), and
detects completion from the worker's HF artifacts. Cost-safety: a launched VM is ALWAYS deleted —
the runner ``finally``, the poll deadline, cancel, and ``sweep_orphans`` each guarantee it. Like
Lambda, there is no in-box self-destruct (no instance-scoped key); ``sweep_orphans`` at startup is
the crash backstop.

Pure dataclasses + builders live in ``.builders`` and are re-exported. Lifecycle functions and the
constants tests monkeypatch stay in this ``__init__``.
"""

from __future__ import annotations

import contextlib
import json
import time

from flash._logging import get_logger
from flash.providers._poll import PollErrorTracker, make_say, surface_heartbeat
from flash.providers.base import GPU_INFO, PollResult, min_cuda_modern
from flash.providers.hyperstack import api as hs_api
from flash.providers.hyperstack.jobs.builders import (
    HyperstackInstance,
    HyperstackJobHandle,
    build_payload,
    build_user_data,
    instance_label,
    run_label_prefix,
)
from flash.providers.runpod.jobs import make_hf_heartbeat_reader, make_hf_text_reader

logger = get_logger(__name__)

# How long a VM may sit in a non-active state (provisioning) before we give up and retry.
LOAD_TIMEOUT_S = 900.0
# Cold-start (Docker pull + pip + model download) emits no heartbeat -> larger setup grace until a
# training heartbeat; tighter window after.
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0
PROVISION_GRACE_S = 3000.0

_SETUP_HEARTBEAT_STAGES = frozenset(
    {"boot", "sft_start", "rl_start", "sft_model_load", "rl_train_start"}
)

# Hyperstack VM statuses that mean "the box is gone / will not progress".
_DEAD_STATES = {"ERROR", "FAILED", "DELETING", "DELETED", "TERMINATED"}


def usable_instances(gpu_class: str, force: bool = False) -> list[HyperstackInstance]:
    """Launchable (region) candidates for a managed GPU class, only where the flavor has stock now.
    ``force`` bypasses the ``/core/flavors`` cache (used by the in-launch refresh so it can discover
    newly-restocked regions instead of re-reading the just-populated allocation cache).

    Hyperstack prices per flavor (not per region), so every candidate carries the same $/hr; the
    list is the regions whose flavor currently advertises stock. Empty == no Hyperstack capacity now.
    """
    from flash.providers.hyperstack.gpus import flavor_for
    from flash.providers.hyperstack.pricing import hourly_rate

    info = GPU_INFO[gpu_class]
    flavor = flavor_for(gpu_class)
    rate = hourly_rate(gpu_class)
    return [
        HyperstackInstance(
            gpu=gpu_class,
            flavor=flavor,
            region=region,
            environment=hs_api.environment_for_region(region),
            vram_gb=info.vram_gb,
            price_usd_hr=rate,
        )
        for region in hs_api.regions_with_stock(flavor, force=force)
    ]


def _launch_rejection_is_clean(err: Exception) -> bool:
    """True when a launch error is a DEFINITIVE rejection that created NO VM (safe to walk). The
    shared RestClient fast-fails a non-429 4xx as ``... -> HTTP 4xx: ...`` (request rejected, e.g.
    no stock). Anything else — 429, 5xx/timeout (``failed after N attempts``), or accepted-but-no-id
    (``returned no VM id``) — is AMBIGUOUS: Hyperstack may have created a billed VM, so we must NOT
    issue another launch."""
    s = str(err)
    return "-> HTTP 4" in s and "HTTP 429" not in s


def launch_and_submit(
    spec,
    seed: int,
    instances: list[HyperstackInstance],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
) -> HyperstackJobHandle:
    """Launch the first region that accepts the job; walk regions on a stock rejection, refresh once."""
    say = make_say(log)
    if not instances:
        raise hs_api.HyperstackApiError(
            f"no Hyperstack stock for {spec.gpu.type} (no region advertises the flavor)"
        )
    payload = build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets)
    user_data = build_user_data(payload)
    name = instance_label(spec.run_id, seed, attempt)

    tried_regions: set[str] = set()
    candidates = list(instances)
    refreshed = False
    last_err: Exception | None = None

    def refresh_once(gpu: str) -> None:
        """One forced stock re-fetch when the walk is exhausted (the alloc cache is ~45s stale)."""
        nonlocal refreshed, candidates
        if not candidates and not refreshed:
            refreshed = True
            candidates = [
                c for c in usable_instances(gpu, force=True) if c.region not in tried_regions
            ]

    while candidates:
        inst = candidates.pop(0)
        if inst.region in tried_regions:
            continue
        tried_regions.add(inst.region)
        # Pre-launch resolution: pick a boot image whose host CUDA covers this GPU class's floor
        # (Blackwell needs 13) and the SSH key. These run BEFORE any non-idempotent launch, so a
        # failure here (e.g. the region advertises stock but has no qualifying CUDA image) created NO
        # VM — it is a CLEAN region skip, never an ambiguous phantom. Walk to the next region.
        try:
            image = hs_api.docker_image_for_region(inst.region, min_cuda=min_cuda_modern(inst.gpu))
            key_name = hs_api.resolve_key_name(inst.environment)
        except hs_api.HyperstackApiError as e:
            last_err = e
            say(f"region {inst.region} ({inst.gpu} {inst.flavor}) unusable (no boot image/key): {e}")
            refresh_once(inst.gpu)
            continue
        try:
            vm_id = hs_api.launch_vm(
                name=name,
                environment_name=inst.environment,
                image_name=image,
                flavor_name=inst.flavor,
                key_name=key_name,
                user_data=user_data,
            )
        except hs_api.HyperstackApiError as e:
            last_err = e
            if not _launch_rejection_is_clean(e):
                # Ambiguous failure (timeout / 5xx / 429 / accepted-but-no-id): Hyperstack may have
                # created a billed VM whose id we never got. Do NOT launch another in this attempt —
                # reconcile any phantom by run-name and stop; the runner's retry (+ gc /
                # sweep_orphans) re-provisions cleanly.
                say(f"ambiguous launch failure in {inst.region}: {e}; reconciling + retrying fresh")
                with contextlib.suppress(Exception):
                    terminate_run_instances(spec.run_id)
                raise hs_api.HyperstackApiError(
                    f"ambiguous Hyperstack launch failure (possible phantom reaped): {e}"
                ) from e
            say(f"region {inst.region} ({inst.gpu} {inst.flavor}) rejected: {e}")
            refresh_once(inst.gpu)
            continue
        say(
            f"launched hyperstack vm {vm_id}: {inst.gpu} {inst.flavor} "
            f"${inst.price_usd_hr:.2f}/hr in {inst.region} attempt={attempt} seed={seed}"
        )
        return HyperstackJobHandle(
            vm_id=vm_id,
            flavor=inst.flavor,
            region=inst.region,
            name=name,
            gpu=inst.gpu,
            hourly_usd=inst.price_usd_hr,
            attempt=attempt,
            started_ts=time.time(),
        )
    # Phantom-VM safety: a non-idempotent launch that Hyperstack ACCEPTED but whose response lacked
    # a parseable id raises (caught above as a region rejection), leaving a billed VM under our run
    # name that no handle owns. Best-effort reap any such VM by run-name before giving up (the
    # post-run gc / sweep_orphans are the backstop, but this closes the window now).
    with contextlib.suppress(Exception):
        terminate_run_instances(spec.run_id)
    raise hs_api.HyperstackApiError(
        f"all {len(tried_regions)} Hyperstack region(s) rejected the {spec.gpu.type} launch "
        f"(no stock): {last_err}"
    )


_make_hf_file_reader = make_hf_text_reader


def _failure_detail(hf_repo: str, prefix: str, phase: str, marker: dict | None) -> str:
    """Best root-cause detail from the HF artifacts. Hyperstack exposes no VM console API, so the
    box's ``hyperstack_boot.log`` (pushed to HF by the cloud-init host uploader) is the only window
    into a pre-worker failure (docker/GPU not ready, image-pull failure)."""
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if err:
        parts.append(f"--- error_{phase}.txt ---\n{err[-2000:]}")
    boot = _make_hf_file_reader(hf_repo, f"{prefix}/hyperstack_boot.log")(force=True)
    if boot:
        parts.append(f"--- hyperstack_boot.log (host) ---\n{boot[-3000:]}")
    return "\n".join(parts) or "hyperstack worker terminated without a DONE sentinel"


def poll_hs_job(
    handle: HyperstackJobHandle,
    spec,
    seed: int,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    setup_grace_s: float = SETUP_GRACE_S,
    stall_after_s: float = STALL_AFTER_S,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll VM status + HF artifacts to a terminal state (cf. lambda.jobs.poll_lambda_job)."""
    say = make_say(log)
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/hyperstack_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

    def finish_ok(done_content: str | None = None) -> PollResult:
        raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
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
                "provider": "hyperstack",
                "hyperstack_rate_usd_hr": handle.hourly_usd,
                "hyperstack_gpu": handle.gpu,
                "hyperstack_flavor": handle.flavor,
                "hyperstack_region": handle.region,
            }
        )
        metrics["notes"] = notes
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        try:
            return float(content.strip()) > handle.started_ts - 120.0
        except ValueError:
            return False

    def finish_from_ok_marker() -> PollResult:
        # ok marker => the worker finished (it wrote metrics before the marker) even if DONE is STALE
        # (a retry hit the already-complete path). Treat ok-marker + metrics as terminal success.
        d = done_reader(force=True)
        return finish_ok(d if (d is not None and done_is_fresh(d)) else None)

    def fail_from_marker(marker: dict | None) -> PollResult:
        from flash.providers.runpod.jobs import worker_flagged_retriable

        # Host failure marker sets retriable=True; the worker stamps it for a RetriableInfraError.
        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(heartbeat_reader)
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=_failure_detail(hf_repo, prefix, spec.phase, marker),
        )

    poll_errors = PollErrorTracker(say, interval_s)
    start = time.time()
    last_status = None
    last_hb_key = None
    last_progress = time.time()
    became_active = False
    seen_training_hb = False
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            vm = hs_api.get_vm(handle.vm_id)
            poll_errors.reset()
        except hs_api.HyperstackApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        missing_streak = missing_streak + 1 if vm is None else 0
        status = ((vm or {}).get("status") or ("missing" if vm is None else "unknown")).upper()
        if status != last_status:
            say(f"vm {handle.vm_id}: {status}")
            last_status = status
            last_progress = time.time()
        if status == "ACTIVE":
            became_active = True

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        dead = missing_streak >= 3 or status in _DEAD_STATES
        if dead:
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
                detail=f"vm stuck in '{status}' for {int(time.time() - start)}s "
                f"(never became active; provisioning / host issue)",
            )

        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            last_progress = time.time()
            if stage not in _SETUP_HEARTBEAT_STAGES:
                seen_training_hb = True
        if became_active:
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
                phase = "training" if seen_training_hb else "setup (pre-training)"
                return PollResult(
                    False,
                    failure="stalled",
                    detail=f"no worker progress for {int(time.time() - last_progress)}s "
                    f"during {phase} (vm status {status}, limit {int(limit)}s)",
                )
        time.sleep(interval_s)


def submit_run_hyperstack(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
    on_last_gpu: bool = False,
) -> PollResult:
    """Hyperstack equivalent of ``runpod.jobs.submit_run``: launch, persist, poll, delete.

    The ``finally`` delete is the cost-safety primary: every exit path tears the paid VM down.
    """
    if spec.gpu.type not in GPU_INFO:
        raise hs_api.HyperstackApiError(
            f"submit_run_hyperstack needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    instances = usable_instances(spec.gpu.type)
    handle = launch_and_submit(
        spec, seed, instances, attempt=attempt, log=log, runtime_secrets=runtime_secrets
    )
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        setup_grace = SETUP_GRACE_S * (1.5 if on_last_gpu else 1.0)
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        return poll_hs_job(
            handle, spec, seed, log=log, heartbeat_reader=reader,
            setup_grace_s=setup_grace, deadline_s=deadline,
        )
    finally:
        hs_api.delete_vm(handle.vm_id)


def cancel(remote: dict) -> None:
    """Cross-process cancel: delete the persisted VM (stops billing)."""
    vm_id = remote.get("vm_id")
    if vm_id:
        hs_api.delete_vm(str(vm_id))


def terminate_run_instances(run_id: str) -> list[str]:
    """Delete every VM belonging to ONE run (names start with its run prefix). Best-effort."""
    if not run_id:
        return []
    try:
        vms = hs_api.list_vms()
    except Exception:
        return []
    prefix = run_label_prefix(run_id)
    ids = [
        str(v.get("id"))
        for v in vms
        if v.get("id")
        and (str(v.get("name") or "") == prefix or str(v.get("name") or "").startswith(prefix + "-s"))
    ]
    return hs_api.delete_vms(ids) if ids else []


def sweep_orphans(active_labels: set[str] | None = None) -> list[str]:
    """Delete Flash-named VMs that no live run owns; return deleted ids. Run at startup + post-run.

    Only names with the ``flash-`` run prefix are touched. ``active_labels`` may be RAW run ids;
    each is passed through ``run_label_prefix`` so it matches the forced prefix the names carry.
    """
    try:
        vms = hs_api.list_vms()
    except Exception as exc:
        logger.warning("hyperstack orphan sweep skipped: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (active_labels or set())}
    orphans: list[str] = []
    for v in vms:
        name = str(v.get("name") or "")
        if not name.startswith("flash-"):
            continue
        if any(name == a or name.startswith(a + "-s") for a in active):
            continue
        vid = v.get("id")
        if vid:
            orphans.append(str(vid))
    deleted = hs_api.delete_vms(orphans) if orphans else []
    for vid in deleted:
        logger.warning("deleted orphaned hyperstack vm %s", vid)
    return deleted
