"""Vast.ai run lifecycle: verified offers -> rented container -> HF-artifact poll -> guaranteed destroy.

The Vast equivalent of ``providers/lambdalabs/jobs/__init__.py``. Vast has no serverless queue and no
VM to cloud-init: we rent a single-GPU CONTAINER from a VERIFIED-DATACENTER offer (the worker image
IS the container), ship the shared instance bootstrap as the container command (``build_onstart``),
and detect completion purely via the worker's HF artifacts (DONE/metrics.json/heartbeat.json) + the
instance's status — no inbound network to the box is ever needed.

Cost-safety invariant: a rented instance is ALWAYS destroyed — the runner's ``finally``, the
onstart's self-destroy backstop, the cancel path, and ``sweep_orphans`` (server startup / post-run)
each independently guarantee it.

The pure dataclasses + builders live in ``.builders`` and are re-exported here so the import path
``flash.providers.vast.jobs`` is unchanged. The lifecycle functions and the operator-visible
constants tests monkeypatch (``usable_offers``, ``deploy_and_submit``, ``poll_vast_job``,
``submit_run_vast``, ``_make_hf_file_reader``, ``RELIABILITY_FLOOR``/``LOAD_TIMEOUT_S``/…) stay in
this ``__init__`` so a ``monkeypatch.setattr(jobs, …)`` still takes effect.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable

from flash._logging import get_logger
from flash.providers._poll import (
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    FIRST_LIVENESS_S,
    PollErrorTracker,
    heartbeat_progress_ts,
    is_training_heartbeat,
    make_say,
    surface_heartbeat,
)
from flash.providers.base import GPU_INFO, PollResult, min_cuda_modern, vast_gpu_for_offer
from flash.providers.runpod.jobs import (
    make_hf_heartbeat_reader,
    make_hf_text_reader,
    worker_flagged_retriable,
)
from flash.providers.vast import api as vast_api
from flash.providers.vast.jobs.builders import (
    VastJobHandle,
    VastOffer,
    build_onstart,
    build_payload,
    instance_label,
    run_label_prefix,
    vast_image,
)

logger = get_logger(__name__)

# Offer-quality floors (beyond verified+datacenter, which are non-negotiable). reliability2 is Vast's
# host-uptime/health score: 0.95 let ~1-in-20 long runs die mid-train ("worker terminated without a
# DONE sentinel" when the host went down); 0.995 (~1-in-200) keeps supply usable while nearly
# eliminating mid-run host deaths. Fixed correctness floors, not operator-tunable.
RELIABILITY_FLOOR = 0.995
MIN_INET_MBPS = 200.0
# How long an instance may sit in a non-running state (image pull) before we give up and retry.
LOAD_TIMEOUT_S = 900.0
# No-progress window once the container is running. The cold start is dominated by the per-run pip
# install (env wheel) + the base-model download, none of which emits a TRAINING heartbeat — so until a
# *training* heartbeat arrives we apply the larger ``SETUP_GRACE_S`` budget; after it we use the tight
# ``STALL_AFTER_S``. This staged grace is the fix for the historical "Vast instance dies every ~25-30
# min": the prior provider used a single flat 1500 s window that fired DURING the cold-start setup
# (model download + vLLM init routinely exceed 25 min with no per-step heartbeat), tearing down a
# perfectly healthy box and retrying — over and over.
SETUP_GRACE_S = 3000.0
STALL_AFTER_S = 1500.0
# Provision + cold-start grace added on top of the run's wall cap for the client-side poll deadline
# (Vast has no server-side execution timeout, so the client deadline + the bootstrap's own cap bound
# spend). Matches Lambda's instance grace.
PROVISION_GRACE_S = 3000.0
# Boards under-report VRAM vs the class nominal (measured live: L4 23034 MB / 24 GB, A40 46068 MB /
# 48 GB = 0.938 of nominal); the server-side gpu_ram filter gets this slack, the class gate stays
# exact (vast_gpu_for_offer).
_SEARCH_VRAM_SLACK = 0.92
# Minimum disk Vast instances are provisioned with (the bootstrap + worker stack + weights need
# headroom regardless of the spec's request). The offer search MUST use this same floor so offers with
# < this disk don't pass the search and then get rejected at create time.
MIN_DISK_GB = 60.0

# The setup-vs-training stall boundary is the SHARED canonical helper ``_poll.is_training_heartbeat``
# (same one runpod + lambdalabs use) so the cold-start grace rule can't drift between providers.

# Vast instance states that mean "the container is gone / will not progress".
_DEAD_STATES = {"exited", "stopped", "offline", "deleted"}


def _effective_disk_gb(spec) -> float:
    """The disk size an instance is actually provisioned with (the create-time floor).

    Both the offer search and ``create_instance`` must agree on this, or offers with a disk between
    ``spec.gpu.disk_gb`` and the floor pass the search then fail to rent.
    """
    return max(float(spec.gpu.disk_gb), MIN_DISK_GB)


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    limit: int = 256,
) -> list[VastOffer]:
    """Verified-datacenter offers able to run the job, cheapest first.

    Server-side filters do the heavy lifting; everything load-bearing is re-checked client-side (belt
    and suspenders — the result rows carry the proof fields).

    ``limit`` is the price-sorted search page size. Callers bucket the rows BY GPU CLASS (cheapest per
    class for the allocator / pricing), so the page must be wide enough to span EVERY fitting managed
    class — at the old 64 a flood of cheap offers from one class could fill the page and silently hide
    a larger fitting class that has usable offers just past the limit. 256 comfortably covers the
    verified-datacenter market across the managed GPU classes; a specific-class caller still filters
    down client-side (a wider page only gives it more candidates).
    """
    rows = vast_api.search_offers(
        int(min_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        limit=int(limit),
    )
    out: list[VastOffer] = []
    for r in rows:
        gpu = vast_gpu_for_offer(str(r.get("gpu_name") or ""), float(r.get("gpu_ram") or 0))
        if gpu is None:  # not a managed class (Ampere+ floor)
            continue
        info = GPU_INFO[gpu]
        dph = float(r.get("dph_total") or 0)
        cuda = float(r.get("cuda_max_good") or 0)
        # Host tier: accept ONLY verified DATACENTER hosts (hosting_type==1); community/marketplace
        # (hosting_type==0) is rejected — the onstart ships run secrets (HF_TOKEN, env creds) to the box.
        _bad_host = r.get("hosting_type") != 1
        if (
            _bad_host
            or r.get("verification") != "verified"
            # Exact class gate: guard against a board whose canonical class nominal VRAM is below the
            # request (the server-side gpu_ram filter only carries slack).
            or info.vram_gb < min_vram_gb
            or float(r.get("reliability2") or 0) < RELIABILITY_FLOOR
            or float(r.get("disk_space") or 0) < float(disk_gb)
            or float(r.get("inet_down") or 0) < MIN_INET_MBPS
            or cuda < float(min_cuda_modern(gpu))  # Blackwell needs CUDA-13 drivers
            or dph <= 0
            or int(r.get("machine_id") or 0) in exclude_machine_ids
        ):
            continue
        out.append(
            VastOffer(
                offer_id=int(r["id"]),
                machine_id=int(r.get("machine_id") or 0),
                gpu=gpu,
                vram_gb=info.vram_gb,
                dph_total=dph,
                cuda_max_good=cuda,
                disk_space=float(r.get("disk_space") or 0),
                reliability=float(r.get("reliability2") or 0),
                inet_down=float(r.get("inet_down") or 0),
                geolocation=str(r.get("geolocation") or ""),
            )
        )
    return sorted(out, key=lambda o: (o.dph_total, o.vram_gb))


def _adopt_instance_by_label(label: str) -> dict | None:
    """Best-effort instance DICT carrying this EXACT (unique per run/seed/attempt) label, or
    ``None``. Reclaims a contract a possibly-successful-but-unconfirmed create left behind, so the
    offer walk adopts it instead of renting a duplicate. The full dict (not just the id) is returned
    so the caller can stamp the handle with the instance's REAL launch time. Any lookup failure ->
    ``None`` (the caller falls back; the orphan sweep remains the backstop)."""
    try:
        for inst in vast_api.list_instances():
            if str(inst.get("label") or "") == label and inst.get("id"):
                return inst
    except Exception as exc:  # listing is best-effort; never let it abort the launch
        logger.warning("vast label reconcile failed (%s); proceeding without adoption", exc)
    return None


def deploy_and_submit(
    spec,
    seed: int,
    offers: list[VastOffer],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
) -> VastJobHandle:
    """Rent the cheapest offer that will actually take the job; walk on rejection.

    Offers are a live market — between search and rent the cheapest one is often gone. We walk up to
    5 ranked offers, then refresh the search once (re-excluding the machines we just tried so a fresh
    market re-search doesn't re-select one that just rejected us).
    """
    say = make_say(log)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(spec, seed, attempt, runtime_secrets=runtime_secrets)
    label = instance_label(spec.run_id, seed, attempt)
    onstart = build_onstart(payload)

    tried: list[VastOffer] = []
    candidates = list(offers[:5])
    refreshed = False
    last_err: Exception | None = None
    while candidates:
        offer = candidates.pop(0)
        tried.append(offer)
        try:
            instance_id = vast_api.create_instance(
                offer.offer_id,
                image=vast_image(offer.gpu),
                disk_gb=_effective_disk_gb(spec),
                env={},
                onstart=onstart,
                label=label,
                runtype="args",
            )
        except vast_api.VastApiError as e:
            last_err = e
            say(f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) rejected: {e}")
            # An AMBIGUOUS create failure (5xx / network-timeout on the NON-IDEMPOTENT PUT /asks) may
            # have created a billed contract that never surfaced in the response. Renting the next
            # offer would leave that one untracked and billing until the orphan sweep. Reconcile by our
            # unique per-attempt label first: if the instance materialized, ADOPT it (no leak, no
            # duplicate) and proceed; only a DEFINITIVE rejection (4xx / success=false body — created
            # nothing) safely walks on. Any lookup failure -> None -> fall through to the normal walk.
            if vast_api.create_error_is_ambiguous(e):
                adopted = _adopt_instance_by_label(label)
                if adopted is not None:
                    iid = int(adopted["id"])
                    # Stamp the handle with the box's REAL launch time (Vast ``start_date`` epoch) so
                    # realized cost + liveness/stall/deadline timing align with its actual runtime, not
                    # the later reconciliation moment. Fall back to now if the field is absent.
                    started = float(adopted.get("start_date") or 0.0) or time.time()
                    say(
                        f"adopted vast instance {iid} from an ambiguous create "
                        f"(label={label}, offer {offer.offer_id}, {offer.gpu})"
                    )
                    return VastJobHandle(
                        instance_id=iid,
                        offer_id=offer.offer_id,
                        machine_id=offer.machine_id,
                        label=label,
                        gpu=offer.gpu,
                        hourly_usd=offer.dph_total,
                        attempt=attempt,
                        started_ts=started,
                    )
                # AMBIGUOUS create with NOTHING adopted: a billed contract may still exist but not be
                # visible yet (object-store / API eventual consistency). Renting another offer would
                # double-provision, so ABORT the walk and surface to the orchestrator (which consumes a
                # run retry); the orphan sweep reclaims any instance that later materializes. We do NOT
                # walk on — a duplicate paid instance is the worse failure (see create_instance).
                raise vast_api.VastApiError(
                    f"ambiguous vast create on offer {offer.offer_id} (label={label}); aborting the "
                    f"offer walk to avoid double-provisioning (orphan sweep reclaims any leak): {e}"
                ) from e
            if not candidates and not refreshed:
                refreshed = True
                taken = {o.machine_id for o in tried}
                # Stay within the allocator-approved class pool: the original ``offers`` are already
                # filtered to the allocated class, so the refresh must not widen to any usable offer.
                allowed = {o.gpu for o in offers}
                candidates = [
                    o
                    for o in usable_offers(
                        min(o.vram_gb for o in offers),
                        _effective_disk_gb(spec),
                        exclude_machine_ids=taken,
                    )
                    if o.gpu in allowed
                ][:5]
            continue
        say(
            f"rented vast instance {instance_id}: {offer.gpu} ${offer.dph_total:.2f}/hr "
            f"(offer {offer.offer_id}, {offer.geolocation}, reliability "
            f"{offer.reliability:.3f}) attempt={attempt} seed={seed}"
        )
        return VastJobHandle(
            instance_id=instance_id,
            offer_id=offer.offer_id,
            machine_id=offer.machine_id,
            label=label,
            gpu=offer.gpu,
            hourly_usd=offer.dph_total,
            attempt=attempt,
            started_ts=time.time(),
        )
    raise vast_api.VastApiError(f"all {len(tried)} vast offers rejected the job: {last_err}")


# Rate-limited reader for one HF artifact's text content (None until it exists). Shared with runpod's
# poller via make_hf_text_reader; kept under this module-local name because tests monkeypatch
# ``vast.jobs._make_hf_file_reader`` and the poll/failure paths resolve it as a module global.
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(
    hf_repo: str, prefix: str, phase: str, marker: dict | None, instance_id: int | None = None
) -> str:
    """Best root-cause detail we can assemble from the HF artifacts + the Vast console.

    Unlike Lambda (no console API -> a host boot.log on HF), Vast exposes a container log API, so
    early-bootstrap failures (pip/env errors before the worker can reach HF) are read live from it.
    """
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    content = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if content:
        parts.append(f"--- error_{phase}.txt ---\n{content[-2000:]}")
    if instance_id:
        logs = vast_api.instance_logs(int(instance_id))
        if logs:
            parts.append(f"--- instance log tail ---\n{logs[-3000:]}")
    return "\n".join(parts) or "vast worker terminated without a DONE sentinel"


def poll_vast_job(
    handle: VastJobHandle,
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
    """Poll instance status + HF artifacts to a terminal state (cf. lambdalabs.jobs.poll_lambda_job).

    COMPLETED     fresh DONE sentinel on HF -> metrics.json (cost stamped from the offer's real $/hr).
    job_failed    attempt marker with ok=false (a real worker error; fails fast unless flagged retriable).
    job_preempted instance died without DONE/marker (host loss) -> infra-shaped, retried.
    stalled       never left loading within LOAD_TIMEOUT_S; OR running but emitted NO heartbeat within
                  ``first_liveness_s``; OR heartbeat frozen past the setup/stall window; OR deadline passed.
    """
    say = make_say(log)

    # started_ts is coerced to 0.0 for an old/corrupt handle; 0.0 means "unknown launch" -> fall back
    # to now so the load/stall clocks and the cost stamp treat a recovered handle consistently.
    launch_ts = handle.started_ts or time.time()

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/vast_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

    def finish_ok(done_content: str | None = None) -> PollResult:
        raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
        # Prefer the worker's DONE timestamp when present and sane; fall back to now (a delayed
        # recovery poll hours after the box wrote DONE would otherwise over-bill by the downtime).
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
                "provider": "vast",
                "vast_rate_usd_hr": handle.hourly_usd,
                "vast_gpu": handle.gpu,
                "vast_offer_id": handle.offer_id,
            }
        )
        metrics["notes"] = notes
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        # DONE carries the worker's time.time(); 120 s of clock-skew grace. Anything older predates
        # this attempt (leftover from a prior attempt's resume).
        try:
            return float(content.strip()) > launch_ts - 120.0
        except ValueError:
            return False

    def finish_from_ok_marker() -> PollResult:
        # An ok marker means the worker finished (it wrote metrics.json before the marker), even if the
        # DONE sentinel is STALE — pass DONE only when genuinely fresh (so cost bills to it).
        d = done_reader(force=True)
        return finish_ok(d if (d is not None and done_is_fresh(d)) else None)

    def fail_from_marker(marker: dict | None) -> PollResult:
        # A real worker error fails fast UNLESS flagged retriable (the worker stamps it in heartbeat
        # for a RetriableInfraError; the bootstrap sets retriable=True in the marker for an HF-side
        # failure) — either retries on a fresh host like a platform termination.
        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(
            heartbeat_reader
        )
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.instance_id),
        )

    def terminal_artifact_result() -> PollResult | None:
        # One forced read of the worker's terminal HF artifacts (DONE / attempt marker). Returns a
        # terminal PollResult when the worker definitively finished or errored, else None.
        d = done_reader(force=True)
        if d is not None and done_is_fresh(d):
            return finish_ok(d)
        raw = marker_reader(force=True)
        if raw:
            with contextlib.suppress(ValueError):
                m = json.loads(raw)
                if m.get("ok"):
                    return finish_from_ok_marker()
                return fail_from_marker(m)
        return None

    poll_errors = PollErrorTracker(say, interval_s)
    # Seed the load/stall clocks from the instance's LAUNCH, not this poll's start: a delayed reattach
    # after a control-plane restart has been billing since launch, so a still-loading instance that
    # already blew LOAD_TIMEOUT_S fails over NOW instead of getting another full window.
    start = launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_running = False
    # When this instance became running, anchored to launch so a reattach whose first read is already
    # running measures the first-liveness window from the original launch (not a fresh window).
    running_since = start
    # Wall-clock THIS poll session first observed the box running (set once). Unlike running_since this
    # is "how long WE have watched it running", so a reattach doesn't fast-fail a box that just came up.
    observed_running_since = None
    seen_training_hb = False
    # Any FRESH heartbeat from THIS attempt (boot included) proves the worker started -> clears the
    # first-liveness deadline. Distinct from seen_training_hb (which gates the tighter training window).
    seen_fresh_hb = False
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            # A recovered run can blow a launch-anchored deadline on the first reattach tick (the
            # outage lasted past max_wall+grace). Read terminal artifacts once before giving up.
            terminal = terminal_artifact_result()
            if terminal is not None:
                return terminal
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            inst = vast_api.get_instance(handle.instance_id)
            poll_errors.reset()
        except vast_api.VastApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        # Verified live: the instance-detail route TRANSIENTLY answers {"instances": null} for healthy
        # instances (and brand-new ones before they materialize). One missing read means nothing — only
        # a sustained streak is a real disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get("actual_status") or ("missing" if inst is None else "unknown")
        if status != last_status:
            say(f"instance {handle.instance_id}: {status}")
            # Treat a status TRANSITION as progress, but NOT the first observation (last_status starts
            # None, so the first read always "changes" — counting it would hand a silent-since-launch
            # worker a fresh setup grace after every control-plane restart).
            if last_status is not None:
                last_progress = time.time()
                if status == "running":
                    running_since = time.time()  # genuine ->running: start the liveness clock
            last_status = status
        if status == "running":
            became_running = True
            if observed_running_since is None:
                observed_running_since = time.time()

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        dead = missing_streak >= 4 or status in _DEAD_STATES
        if dead:
            # One forced final read: the worker may have finished right before the box self-destroyed
            # (the normal success order on this substrate).
            terminal = terminal_artifact_result()
            if terminal is not None:
                return terminal
            # Dead host with no ok-marker/DONE. Distinguish a genuine host LOSS (retry on a fresh host)
            # from a worker that RAN and CRASHED early — before it could write the attempt marker — but
            # left error_{phase}.txt (a bad env id, a config/code error, an OOM): that is DETERMINISTIC,
            # so fail FAST. A crash the worker flagged retriable still retries.
            err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{spec.phase}.txt")(force=True)
            worker_crashed = bool(err and err.strip()) and not worker_flagged_retriable(
                heartbeat_reader
            )
            return PollResult(
                False,
                failure="job_failed" if worker_crashed else "job_preempted",
                detail=_failure_detail(hf_repo, prefix, spec.phase, None, handle.instance_id),
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

        if not became_running and time.time() - start > LOAD_TIMEOUT_S:
            return PollResult(
                False,
                failure="stalled",
                detail=f"instance stuck in '{status}' for {int(time.time() - start)}s "
                f"(never started; image pull / host issue)",
            )

        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            # Credit the heartbeat's OWN timestamp (not poll time) so a heartbeat already stale before a
            # control-plane restart doesn't buy a fresh window; clamp to [launch, now]. ``fresh`` is
            # False for a LEFTOVER heartbeat from a prior attempt (ts < launch, or a different attempt),
            # which must not arm the tighter training window before this attempt overwrites the file.
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts, handle.attempt)
            if fresh:
                # MONOTONIC: never let the progress clock regress. An older heartbeat.json upload can
                # land AFTER a newer one was already credited (object-store eventual consistency), and a
                # backwards step would make the setup/training stall timer fire early and tear down a
                # healthy instance under the tight STALL_AFTER_S window.
                last_progress = max(last_progress, hb_ts)
                seen_fresh_hb = True
                # Tighten setup_grace -> stall window only once setup is genuinely OVER; the shared
                # helper keeps cold-start pings (incl. the silent step=0 first rollout) under setup
                # grace (new_key[1] is the heartbeat's step). See _poll.is_training_heartbeat.
                if is_training_heartbeat(stage, new_key[1]):
                    seen_training_hb = True
        # Before the first TRAINING heartbeat the box is still in the long cold start (per-run pip +
        # model download), so use the larger setup grace; tighten only once training begins.
        if became_running:
            # Fast-failover: a container that reached 'running' but never emitted ANY heartbeat past
            # first_liveness_s is a wedged host -> 'stalled' (infra-shaped -> the runner fails over
            # cross-provider), instead of burning the full setup grace. The observed-running floor
            # keeps a reattach from fast-failing a box that only just came up.
            if (
                not seen_fresh_hb
                and time.time() - running_since > first_liveness_s
                and observed_running_since is not None
                and time.time() - observed_running_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                terminal = terminal_artifact_result()
                if terminal is not None:
                    return terminal
                return PollResult(
                    False,
                    failure="stalled",
                    detail=f"no worker heartbeat for {int(time.time() - running_since)}s after the "
                    f"container started (worker never came up; limit {int(first_liveness_s)}s)",
                )
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
                terminal = terminal_artifact_result()
                if terminal is not None:
                    return terminal
                phase = "training" if seen_training_hb else "setup (pre-training)"
                return PollResult(
                    False,
                    failure="stalled",
                    detail=f"no worker progress for {int(time.time() - last_progress)}s during "
                    f"{phase} (instance status {status}, limit {int(limit)}s)",
                )
        time.sleep(interval_s)


def submit_run_vast(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
) -> PollResult:
    """Vast equivalent of ``lambdalabs.jobs.submit_run_lambda``: rent, persist, poll, destroy.

    The ``finally`` destroy is the cost-safety primary: every exit path — success, failure, stall,
    exception, KeyboardInterrupt — tears the paid instance down.
    """
    # GPU_INFO is keyed by concrete GPU class; a policy word ("cheapest"/"auto") would KeyError opaquely.
    # The allocator resolves policy words to a concrete class upstream, so reaching here with one is a
    # caller bug — name it clearly.
    if spec.gpu.type not in GPU_INFO:
        raise vast_api.VastApiError(
            f"submit_run_vast needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    info = GPU_INFO[spec.gpu.type]
    offers = [
        o for o in usable_offers(info.vram_gb, _effective_disk_gb(spec)) if o.gpu == spec.gpu.type
    ]
    handle = deploy_and_submit(
        spec, seed, offers, attempt=attempt, log=log, runtime_secrets=runtime_secrets
    )
    # The instance is billing the MOMENT deploy_and_submit returns; the teardown ``finally`` must guard
    # EVERYTHING after that point — including ``on_handle`` (persisting the handle can itself raise).
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        # Wall cap + provision/cold-start grace; Vast has no server-side execution timeout, so the
        # client deadline (and the bootstrap's own cap) bound spend.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + PROVISION_GRACE_S
        return poll_vast_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            deadline_s=deadline,
        )
    finally:
        vast_api.destroy_instance(handle.instance_id)


def cancel(remote: dict) -> None:
    """Cross-process cancel: destroy the persisted instance (stops billing)."""
    instance_id = remote.get("instance_id")
    if instance_id:
        vast_api.destroy_instance(int(instance_id))


def destroy_run_instances(run_id: str) -> list[int]:
    """Destroy every instance belonging to ONE run (labels start with its run prefix).

    Cancel/GC path: unlike ``sweep_orphans`` this never looks at other runs, so it is safe to call
    while they are in flight. Best-effort: never raises.
    """
    destroyed: list[int] = []
    if not run_id:
        return destroyed
    try:
        instances = vast_api.list_instances()
    except Exception:
        return destroyed
    prefix = run_label_prefix(run_id)
    for inst in instances:
        iid = inst.get("id")
        label = str(inst.get("label") or "")
        # Match on the label boundary, not a raw string prefix: a label is
        # ``f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"``, so a run's prefix must equal the label
        # or be followed by the ``-s`` seed boundary (else ``flash-100`` would also destroy ``flash-1000``).
        if (
            iid
            and (label == prefix or label.startswith(prefix + "-s"))
            and vast_api.destroy_instance(int(iid))
        ):
            destroyed.append(int(iid))
    return destroyed


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
    known_labels: set[str] | Callable[[], set[str]] | None = None,
) -> list[int]:
    """Destroy Flash-labeled instances that no live run owns; return destroyed ids.

    Run at server startup (crash recovery) and after runs. Only labels carrying the ``flash-`` run
    prefix are ever touched — nothing else on the account is ours to destroy. ``active_labels`` may be
    RAW run ids (or a CALLABLE resolved AFTER listing, to close the launch race); each is passed
    through ``run_label_prefix`` so it matches the forced prefix the instance labels carry.

    ``known_labels`` (optional, RAW run ids or callable) is the universe of runs THIS control plane has
    a record of. When supplied, an instance is reaped ONLY if its label maps to one of them — the
    multi-plane safety guard so two control planes sharing one Vast account never reap each other's
    live instances. ``None`` keeps the legacy unscoped behavior (reap every non-active ``flash-`` box).
    Best-effort: never raises.
    """
    try:
        instances = vast_api.list_instances()
    except Exception as exc:
        logger.warning("vast orphan sweep skipped: %s", exc)
        return []
    try:
        labels = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        # Resolving a protection/known set failed — SKIP the sweep rather than fall through to an empty
        # set (which would treat every live run's instance as an orphan). Honors "never raises".
        logger.warning("vast orphan sweep skipped: could not resolve run sets: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (labels or set())}
    known_prefixes = (
        None if known_labels is None else {run_label_prefix(a) for a in (known or set())}
    )

    def _matches(prefixes: set[str], label: str) -> bool:
        # Name-boundary match (EQUAL or followed by the ``-s`` seed boundary) so ``flash-100`` can't
        # shield/claim ``flash-1000-...`` (or vice versa).
        return any(label == p or label.startswith(p + "-s") for p in prefixes)

    destroyed: list[int] = []
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label.startswith("flash-"):
            continue
        if _matches(active, label):
            continue  # a live run owns this box — protected
        # Multi-plane guard: with a known set, only reap boxes attributable to one of THIS plane's runs.
        if known_prefixes is not None and not _matches(known_prefixes, label):
            continue
        iid = inst.get("id")
        if iid and vast_api.destroy_instance(int(iid)):
            destroyed.append(int(iid))
            logger.warning("destroyed orphaned vast instance %s (label %s)", iid, label)
    return destroyed
