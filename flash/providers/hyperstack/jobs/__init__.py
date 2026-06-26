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
    mode: str | None = None,
    models: list | None = None,
) -> HyperstackJobHandle:
    """Launch the first region that accepts the job; walk regions on a stock rejection, refresh once.

    ``mode="preload"`` + ``models`` launches a download-only warm (the bootstrap pulls the models into
    the mounted cache volume and exits — no worker)."""
    say = make_say(log)
    if not instances:
        raise hs_api.HyperstackApiError(
            f"no Hyperstack stock for {spec.gpu.type} (no region advertises the flavor)"
        )
    # Weight cache: when wanted (runner-assigned network_volume), HF_HOME points at a block volume
    # formatted+mounted at /mnt/flash-weights (region-independent path) and bind-mounted into the
    # container; the volume is created-if-absent per environment and attached AFTER launch. If the
    # volume can't be ensured we launch cold; if attach fails the cloud-init preamble degrades to cold
    # (no device appears). Block volumes are single-attach, so a concurrent same-region run runs cold.
    # ``gpu=`` selects the per-GPU worker image (dev: worker_image_for_gpu via hyperstack_image).
    cache_name = getattr(spec.gpu, "network_volume", None)
    cold_user_data = build_user_data(
        build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets), gpu=spec.gpu.type
    )
    cache_user_data = None
    cache_gb = 0
    if cache_name:
        from flash.runner import WEIGHT_CACHE_VOLUME_GB
        from flash.spec import _volume_gb

        # Honor the runner-assigned size (mirrors RunPod), defaulting to the standard cache size. Parse
        # tolerantly via _volume_gb: this best-effort weight-cache path must never crash the run on a
        # non-int / stale / hand-edited size ("0", "", "abc", bool) — it just falls back to the default.
        cache_gb = _volume_gb(getattr(spec.gpu, "network_volume_gb", None), default=WEIGHT_CACHE_VOLUME_GB)
        cache_user_data = build_user_data(
            build_payload(
                spec, seed, attempt, runtime_secrets=runtime_secrets,
                cache_host_mount="/mnt/flash-weights", cache_block_device=True,
                mode=mode, models=models,
            ),
            gpu=spec.gpu.type,
        )
    name = instance_label(spec.run_id, seed, attempt)

    tried_regions: set[str] = set()
    candidates = list(instances)
    refreshed = False
    last_err: Exception | None = None

    def refresh_once(gpu: str) -> None:
        """One forced stock re-fetch when the walk is exhausted (the alloc cache is ~45s stale).

        NO-OP in preload mode: ``warm_instances`` pins each preload launch to ONE specific target
        region (``instances=[candidate]``) and reports that exact region as warmed. Refreshing to a
        DIFFERENT region here would warm region B while the caller reports the target region A as
        warmed (its cache actually still cold). A preload that can't run in its target region must
        FAIL that region (the walk exhausts -> raise), never silently warm another.
        """
        nonlocal refreshed, candidates
        if mode == "preload":
            return
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
        # Ensure the cache volume exists in this environment (create-if-absent); on success use the
        # cache user_data and attach the volume after launch. Any failure -> launch cold here — EXCEPT
        # in preload mode, where the cold user_data carries no mode/models, so a cold fallback would
        # boot a full training bootstrap (GPU billing, timeout) and warm nothing. There we SKIP the
        # region instead and let the walk try the next one (failing if none can host the cache).
        vol_id, user_data = None, cold_user_data
        cache_unavailable_reason = None
        if cache_name and not hs_api.region_supports_cache(inst.region):
            cache_unavailable_reason = "weight cache not supported in region"
        elif cache_name:
            try:
                # Per-region physical name (Hyperstack volume names are GLOBALLY unique — a bare
                # cache_name can exist in only one environment account-wide).
                vol_name = hs_api.cache_volume_name(cache_name, inst.region)
                vol_id = hs_api.ensure_volume(vol_name, inst.environment, cache_gb) or None
                if vol_id is not None:
                    user_data = cache_user_data
                else:
                    cache_unavailable_reason = "ensure_volume returned no id"
            except Exception as e:
                vol_id = None
                cache_unavailable_reason = str(e)
        if cache_name and cache_unavailable_reason is not None:
            if mode == "preload":
                say(f"weight cache unavailable in {inst.region} ({cache_unavailable_reason}); "
                    "skipping (preload needs it)")
                last_err = hs_api.HyperstackApiError(
                    f"preload: weight cache unavailable in {inst.region} ({cache_unavailable_reason})"
                )
                refresh_once(inst.gpu)
                continue
            say(f"weight cache unavailable in {inst.region} ({cache_unavailable_reason}); launching cold")
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
        # Attach the cache volume to the freshly-launched VM (best-effort: the cloud-init preamble
        # degrades to a cold run if no device appears, e.g. attach failed or the volume is busy on
        # another VM — block volumes are single-attach).
        if vol_id is not None:
            attached = False
            with contextlib.suppress(Exception):
                attached = hs_api.attach_volume(vm_id, vol_id)
            # A training run survives a failed attach (the preamble runs cold), but a PRELOAD box can't:
            # with no device it would refuse to warm ephemeral disk (the sentinel check) and just burn
            # paid GPU until the wall cap. Treat a failed preload attach as a launch failure — tear the
            # VM down and walk to the next region (failing the warm if none can attach the cache).
            if not attached and mode == "preload":
                say(f"preload: cache attach failed in {inst.region} (vol busy/absent); "
                    "terminating box and trying next region")
                with contextlib.suppress(Exception):
                    terminate_run_instances(spec.run_id)
                last_err = hs_api.HyperstackApiError(
                    f"preload: cache volume attach failed in {inst.region}"
                )
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


def _failure_detail(hf_repo: str, prefix: str, phase: str, marker: dict | None, attempt: int) -> str:
    """Best root-cause detail from the HF artifacts. Hyperstack exposes no VM console API, so the
    box's attempt-scoped ``hyperstack_attempt<N>_boot.log`` (pushed to HF by the cloud-init host
    uploader) is the only window into a pre-worker failure (docker/GPU not ready, image-pull
    failure). Attempt-scoped so a retry reads ITS OWN boot log, not a prior attempt's."""
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if err:
        parts.append(f"--- error_{phase}.txt ---\n{err[-2000:]}")
    boot = _make_hf_file_reader(hf_repo, f"{prefix}/hyperstack_attempt{attempt}_boot.log")(force=True)
    if boot:
        parts.append(f"--- hyperstack_attempt{attempt}_boot.log (host) ---\n{boot[-3000:]}")
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
    first_liveness_s: float = FIRST_LIVENESS_S,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll VM status + HF artifacts to a terminal state (cf. lambda.jobs.poll_lambda_job)."""
    say = make_say(log)
    # Single source of truth for "when did this VM launch". started_ts is a non-Optional float that
    # HyperstackJobHandle.from_dict coerces to 0.0 when MISSING (old/corrupt handle), so 0.0 means
    # "unknown launch" (a real launch is a large epoch ts, never 0.0). Fall back to now so EVERY use
    # below -- the load/stall clocks AND done_is_fresh / finish_ok's wall+cost stamping -- treats a
    # recovered corrupt handle consistently, instead of billing/comparing from the 1970 epoch.
    launch_ts = handle.started_ts or time.time()
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/hyperstack_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")
    # Attempt-scoped host boot log: present within ~2 min iff THIS attempt's cloud-init actually ran
    # (the host uploader starts before the image pull). Its ABSENCE while active is the signal that
    # nothing ever started (see the first-liveness check). Throttled — only fetched once the deadline
    # is in question, never on the hot path.
    boot_log_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/hyperstack_attempt{handle.attempt}_boot.log", min_interval_s=60.0
    )

    def finish_ok(done_content: str | None = None) -> PollResult:
        raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
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
        # launch_ts (not handle.started_ts) so an unknown-launch (0.0) handle doesn't accept every
        # leftover DONE as fresh.
        try:
            return float(content.strip()) > launch_ts - 120.0
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
    # Seed the load/stall clocks from the VM's LAUNCH (launch_ts), not this poll's start: on a
    # delayed reattach after a control-plane restart the box has been billing since launch, so a
    # still-booting VM that already blew LOAD_TIMEOUT_S must fail over NOW instead of getting another
    # full window. launch_ts already maps an unknown-launch (0.0) handle to now (see above), so a
    # fresh launch is a no-op and a corrupt handle won't peg the clocks to the epoch.
    start = launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_active = False
    # When this VM became active, anchored to launch (like last_progress) so a reattach whose first
    # read is already ACTIVE measures the first-liveness window from the original launch — it does NOT
    # hand a box silent since before a control-plane restart a fresh window. Advanced to now only on a
    # genuine inactive->active transition observed in this poll session.
    active_since = start
    # Wall-clock this poll session FIRST observed the VM ACTIVE (set once, never reset). Unlike the
    # launch-anchored ``active_since``, this is genuinely "how long WE have watched it active", so a
    # reattach whose first read is already ACTIVE doesn't immediately fast-fail a VM that only just came
    # up — see FIRST_LIVENESS_OBSERVED_FLOOR_S.
    observed_active_since = None
    seen_training_hb = False
    # Any FRESH heartbeat from THIS attempt (boot stage included) proves the worker started — clears
    # the first-liveness deadline. A leftover prior-attempt heartbeat (ts < launch) is NOT fresh, so
    # it cannot disarm the deadline on the retry into a sick region it must catch.
    seen_fresh_hb = False
    # Latched once the attempt's host boot.log is OBSERVED (even empty ""): its existence proves
    # cloud-init ran, so a later rate-limited reader None can't spuriously fail the VM over as stalled.
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
            # Treat a status TRANSITION as progress, but NOT the first observation: last_status
            # starts None, so on a reattach the very first read always "changes" — counting it as
            # progress would overwrite the launch-anchored last_progress and hand a silent-since-
            # launch worker a fresh full setup grace after every control-plane restart.
            if last_status is not None:
                last_progress = time.time()
                if status == "ACTIVE":
                    active_since = time.time()  # genuine inactive->active: start the liveness clock
            last_status = status
        if status == "ACTIVE":
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
            return PollResult(
                False,
                failure="job_preempted",
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
                detail=f"vm stuck in '{status}' for {int(time.time() - start)}s "
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
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts)
            if fresh:
                last_progress = hb_ts
                seen_fresh_hb = True  # worker is alive (boot or later) -> first-liveness satisfied
                if stage not in _SETUP_HEARTBEAT_STAGES:
                    seen_training_hb = True
        if became_active:
            # Fast-failover: a VM that reached 'ACTIVE' but never started a worker (no fresh hb AND no
            # attempt-scoped boot.log past first_liveness_s) is a wedged host -> 'stalled' (infra-shaped
            # -> runner escapes cross-provider via #241), instead of burning the ~50min setup grace. A
            # healthy box (even mid image-pull) emits its boot.log within ~2 min. The observed-active
            # floor keeps a reattach from fast-failing a VM that only just came up (active_since is
            # launch-anchored and may already be past first_liveness_s) — see the constant's note.
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
                            f"{int(time.time() - active_since)}s after vm became active "
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
        # Uniform per-GPU wait: poll_hs_job uses its default FIRST_LIVENESS_S / SETUP_GRACE_S.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        return poll_hs_job(
            handle, spec, seed, log=log, heartbeat_reader=reader, deadline_s=deadline,
        )
    finally:
        hs_api.delete_vm(handle.vm_id)


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


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
) -> list[str]:
    """Delete Flash-named VMs that no live run owns; return deleted ids. Run at startup + post-run.

    Only names with the ``flash-`` run prefix are touched. ``active_labels`` may be RAW run ids;
    each is passed through ``run_label_prefix`` so it matches the forced prefix the names carry.

    ``active_labels`` may also be a CALLABLE returning that set — it is then resolved AFTER the VM
    list is fetched. The periodic in-lifetime sweep passes one so the protection set is read
    post-listing: any VM present in the list had its run's status row committed before the VM was
    launched (hence before this list call), so resolving the live set now is guaranteed to include
    it — closing the launch race where a run started after a pre-captured set could have its fresh
    worker reaped as a phantom orphan.
    """
    try:
        vms = hs_api.list_vms()
    except Exception as exc:
        logger.warning("hyperstack orphan sweep skipped: %s", exc)
        return []
    try:
        labels = active_labels() if callable(active_labels) else active_labels
    except Exception as exc:
        # Resolving the protection set failed (e.g. a db/status read error in the callable). SKIP the
        # sweep — never fall through to an empty set, which would treat every live run's VM as an
        # orphan and reap it. Honors the "never raises" contract.
        logger.warning("hyperstack orphan sweep skipped: could not resolve active set: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (labels or set())}
    now = time.time()
    orphans: list[str] = []
    for v in vms:
        name = str(v.get("name") or "")
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
                vid = v.get("id")
                if vid:
                    orphans.append(str(vid))
                    logger.warning(
                        "reaping orphaned hyperstack preload vm %s (outlived its wall deadline + "
                        "grace; driver lost)", name)
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
