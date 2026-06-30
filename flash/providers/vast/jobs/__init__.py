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
import http.client
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
    surface_heartbeat,
)
from flash.providers.base import (
    GPU_INFO,
    PollResult,
    UnreconciledCreateError,
    min_cuda_modern,
    vast_gpu_for_offer,
)
from flash.providers.runpod.jobs import (
    heartbeat_is_stale_prior_attempt,
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

# Vast instance states that mean "the container is gone / will not progress". ``frozen`` is a
# paused container that KEEPS billing GPU charges (per Vast's show-instances status docs) yet can
# emit no DONE/heartbeat, so a worker that freezes would otherwise only fail over after the full
# setup/training stall window (or load timeout) while the box bills — classify it dead for fast
# failover like the other non-progressing states (Codex). Unlike ``unknown`` it is never this
# poller's no-status fallback, so it needs no ``became_running`` gate.
_DEAD_STATES = {"exited", "stopped", "offline", "deleted", "frozen"}

# A fresh DONE can be visible before the separately-uploaded metrics.json (HF read-after-write is
# eventually consistent). Re-read metrics this many times (this far apart) before treating a
# DONE-without-metrics as a real failure, so a successful run isn't failed on a transient read gap.
_METRICS_AFTER_DONE_RETRIES = 6
_METRICS_AFTER_DONE_WAIT_S = 5.0

# A successful Vast container exits / self-destroys the instant it finishes — often BEFORE HF exposes
# the just-written DONE / vast_attempt marker (read-after-write lag). The dead/missing-instance path
# would then see no terminal artifact on a single read and mis-classify a FINISHED seed as host loss,
# renting a retry that races the same seed's artifacts. Re-read the terminal artifacts this many times
# (this far apart) before concluding loss, so a finished-then-self-destroyed seed is recognized.
_TERMINAL_AFTER_DEAD_RETRIES = 6
_TERMINAL_AFTER_DEAD_WAIT_S = 5.0


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
    max_wall_seconds: float = 0,
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

    ``max_wall_seconds`` is the run's wall cap; when set, the search additionally requires offers to be
    available for at least ``max(60, max_wall) + PROVISION_GRACE_S`` — the SAME deadline the poller
    enforces (it floors the wall at 60 s before adding grace, see poll_vast_job), so the search never
    advertises capacity an offer can't outlast. 0 = no duration floor.
    """
    min_duration = (
        max(60.0, float(max_wall_seconds)) + PROVISION_GRACE_S
        if max_wall_seconds and max_wall_seconds > 0
        else 0
    )
    rows = vast_api.search_offers(
        int(min_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        min_duration_seconds=min_duration,
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
    code_prefix: str | None = None,
) -> VastJobHandle:
    """Rent the cheapest offer that will actually take the job; walk on rejection.

    Offers are a live market — between search and rent the cheapest one is often gone. We walk up to
    5 ranked offers, then refresh the search once (re-excluding the machines we just tried so a fresh
    market re-search doesn't re-select one that just rejected us).
    """
    say = make_say(log)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(
        spec,
        seed,
        attempt,
        runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
    )
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
            # This PRE-RECONCILE log must NOT throw: the create just failed AMBIGUOUSLY (timeout / 5xx /
            # unreadable response on the NON-IDEMPOTENT PUT /asks may have left a billed contract). A
            # raising ``say`` (closed log stream / disk error) here would abort BEFORE
            # create_error_is_ambiguous() and the whole adopt/destroy/terminal path below ever run, so
            # the phantom would never be reconciled and the runner would see a generic deploy error and
            # retry/rent another offer while it keeps billing + writing this run's artifacts (Codex).
            with contextlib.suppress(Exception):
                say(f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) rejected: {e}")
            # An AMBIGUOUS create failure (5xx / network-timeout on the NON-IDEMPOTENT PUT /asks) may
            # have created a billed contract that never surfaced in the response. Renting the next
            # offer would leave that one untracked and billing until the orphan sweep. Reconcile by our
            # unique per-attempt label first: if the instance materialized, ADOPT it (no leak, no
            # duplicate) and proceed; only a DEFINITIVE rejection (4xx / success=false body — created
            # nothing) safely walks on. Any lookup failure -> None -> fall through to the normal walk.
            if vast_api.create_error_is_ambiguous(e):
                adopted = _adopt_instance_by_label(label)
                # A matching row whose id is truthy-but-unparseable (unexpected API shape) cannot be
                # cleanly adopted as a handle. A bare ``int(adopted["id"])`` would raise ValueError
                # BEFORE the terminal UnreconciledCreateError below, aborting the reconcile with the
                # WRONG error — a generic flake the orchestrator retries, double-provisioning. Coerce
                # defensively; on an unparseable id FALL THROUGH to the fail-closed abort path (destroy
                # by label + UnreconciledCreateError), treating the row as a phantom we couldn't adopt
                # ("a phantom may exist") rather than crashing past the reconcile (Codex).
                adopted_id = _coerce_instance_id(adopted.get("id")) if adopted is not None else None
                if adopted_id is not None:
                    iid = adopted_id
                    # Stamp the handle with the box's REAL launch time (Vast ``start_date`` epoch) so
                    # realized cost + liveness/stall/deadline timing align with its actual runtime, not
                    # the later reconciliation moment. Fall back to now if the field is absent.
                    started = float(adopted.get("start_date") or 0.0) or time.time()
                    # SUCCESS log must NOT throw: the instance is already rented/billing, so a raising
                    # ``say`` (closed log stream / disk error) before we return the handle would skip
                    # submit_run_vast's teardown finally and leak the box while the run retries (Codex).
                    with contextlib.suppress(Exception):
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
                # AMBIGUOUS create with NOTHING cleanly adopted (no row under our label, OR a matching
                # row whose id was unparseable): a billed contract may still exist but not be cleanly
                # reclaimable yet (object-store / API eventual consistency). Renting another offer would
                # double-provision, so ABORT the walk and surface to the orchestrator (which consumes a
                # run retry). PROACTIVELY destroy this run's instances by label first (mirrors Lambda's
                # ambiguous path calling terminate_run_instances) so a phantom contract that DID
                # materialize is killed now, not left billing through retries while sweep_orphans
                # shields it as a still-active run. Best-effort (destroy_run_instances never raises).
                destroyed = destroy_run_instances(spec.run_id)
                # ABORT log must NOT throw: a raising ``say`` (closed log stream / disk error) here would
                # exit with the LOGGING exception instead of the terminal UnreconciledCreateError below,
                # so the orchestrator would see a generic deploy/submit flake and retry — double-
                # provisioning if the phantom contract materialized. Suppress so the raise always wins.
                if destroyed:
                    with contextlib.suppress(Exception):
                        say(f"destroyed {len(destroyed)} possible phantom instance(s) {destroyed} on abort")
                # TERMINAL, not retriable: ``destroy_run_instances`` is a point-in-time sweep, so a
                # phantom contract that has not surfaced yet (eventual consistency) survives it. A
                # plain VastApiError here is caught by the orchestrator as ``poll_error`` and RETRIED —
                # which rents another instance while that phantom may still appear and bill under the
                # still-active run (``sweep_orphans`` shields active runs). Raise the terminal type so
                # the run fails fast; teardown + a later sweep (run now inactive) reclaim any late box.
                raise UnreconciledCreateError(
                    f"ambiguous vast create on offer {offer.offer_id} (label={label}); aborting the "
                    f"offer walk to avoid double-provisioning (destroyed phantom by label): {e}"
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
                        max_wall_seconds=float(getattr(spec.gpu, "max_wall_seconds", 0) or 0),
                    )
                    if o.gpu in allowed
                ][:5]
            continue
        # SUCCESS log must NOT throw: the box is rented/billing, so a raising ``say`` here (closed log
        # stream / disk error) before the handle is returned would skip submit_run_vast's teardown
        # finally and leak the instance while the runner retries the "deploy error" (Codex).
        with contextlib.suppress(Exception):
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
    # Heartbeat ATTEMPT-DATING (worker_flagged_retriable / heartbeat_is_stale_prior_attempt) must use
    # the TRUE launch, NOT the now() fallback: those helpers read a falsy launch as "unknown -> can't
    # date by ts" (every heartbeat is then fresh / non-prior). Passing now() here instead would make a
    # normal heartbeat's ts (always < now) look like it predates launch, so a same-attempt crash with a
    # shared error_<phase>.txt would be misread as a prior-attempt leftover -> wrong job_preempted
    # (Cursor MtgwT). Keep `launch_ts` (now()-floored) for elapsed-time math (stall clock, DONE
    # freshness, cost) where a concrete anchor is required; use this raw value only for dating.
    _dating_launch = handle.started_ts or 0.0

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/vast_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

    def finish_ok(done_content: str | None = None, fallback_end_ts: float | None = None) -> PollResult:
        # DONE and metrics.json are SEPARATE HF artifacts; the worker writes metrics.json BEFORE the
        # DONE sentinel, but HF read-after-write is eventually consistent, so a fresh DONE can be
        # visible before metrics.json is readable. Don't fail a SUCCESSFUL run on that transient gap:
        # re-read metrics a few times before classifying DONE-without-metrics as a real failure (Codex
        # MtzrL). time.sleep is mocked in tests, so this adds no test wall-time.
        raw = metrics_reader(force=True)
        attempts_left = _METRICS_AFTER_DONE_RETRIES
        while raw is None and attempts_left > 0:
            say("DONE seen but metrics.json not visible yet; waiting for HF read-after-write")
            time.sleep(_METRICS_AFTER_DONE_WAIT_S)
            raw = metrics_reader(force=True)
            attempts_left -= 1
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
        # Prefer the worker's DONE timestamp when present and sane; fall back to now for the
        # provider-instance wall note. Customer-facing cost below uses worker training wall only.
        end_ts = time.time()
        if done_content:
            try:
                done_ts = float(done_content.strip())
                if launch_ts <= done_ts <= end_ts:
                    end_ts = done_ts
            except ValueError:
                pass
        elif fallback_end_ts is not None:
            # No usable DONE, but a terminal ok-marker carries the worker's OWN completion ts. Use it
            # for the provider-instance wall note so delayed recovery does not look like runtime.
            try:
                ts = float(fallback_end_ts)
                if launch_ts <= ts <= end_ts:
                    end_ts = ts
            except (TypeError, ValueError):
                pass
        instance_wall_s = max(0.0, end_ts - launch_ts)
        try:
            train_wall_s = max(0.0, float(metrics.get("wall_seconds") or 0.0))
        except (TypeError, ValueError):
            train_wall_s = 0.0
        metrics["cost_usd"] = round((train_wall_s / 3600.0) * handle.hourly_usd, 6)
        notes = metrics.get("notes") if isinstance(metrics.get("notes"), dict) else {}
        notes.update(
            {
                "provider": "vast",
                "vast_rate_usd_hr": handle.hourly_usd,
                "vast_gpu": handle.gpu,
                "vast_offer_id": handle.offer_id,
                "vast_instance_wall_seconds": round(instance_wall_s, 3),
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

    def finish_from_ok_marker(marker: dict | None = None) -> PollResult:
        # An ok marker means the worker finished (it wrote metrics.json before the marker), even if the
        # DONE sentinel is STALE — pass DONE only when genuinely fresh. When DONE is absent/stale, fall
        # back to the marker's own completion ts for the provider-instance wall note so a recovered
        # success is not measured to the possibly much later poll time (Codex).
        d = done_reader(force=True)
        fresh = d is not None and done_is_fresh(d)
        marker_ts = marker.get("ts") if isinstance(marker, dict) else None
        return finish_ok(d if fresh else None, fallback_end_ts=None if fresh else marker_ts)

    def fail_from_marker(marker: dict | None) -> PollResult:
        # A real worker error fails fast UNLESS flagged retriable (the worker stamps it in heartbeat
        # for a RetriableInfraError; the bootstrap sets retriable=True in the marker for an HF-side
        # failure) — either retries on a fresh host like a platform termination.
        # Gate the heartbeat flag to THIS attempt: a stale retriable=True left by a prior attempt's
        # worker must not override the current attempt's (non-retriable) marker and turn a fast-fail
        # into a GPU-burning retry loop.
        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(
            heartbeat_reader, launch_ts=_dating_launch, current_attempt=handle.attempt
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
                    return finish_from_ok_marker(m)
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
    # Vast-parity for Lambda's boot.log liveness: a non-empty CONTAINER LOG tail proves the bootstrap
    # is alive (pip install / code fetch can legitimately outlast first_liveness_s before the worker's
    # first heartbeat), so we don't fast-fail a healthy cold start. ``console_log_absent_polls`` requires
    # the silence to persist across BOOT_LOG_ABSENT_POLLS so a transient log-API blip can't burn a retry.
    console_log_seen = False
    console_log_absent_polls = 0
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
        except (
            vast_api.VastApiError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            http.client.HTTPException,
        ) as e:
            # A transient MALFORMED 200 body from the instance-detail API (truncated / non-JSON /
            # invalid-UTF8) makes ``RestClient.request`` raise JSONDecodeError/UnicodeDecodeError/
            # http.client.HTTPException, NOT a VastApiError — the _http retry wrapper only catches the
            # OSError-family transients, so a decode/incomplete-read failure escapes get_instance's
            # ``except VastApiError`` raw. Without catching it here it would ESCAPE the poll loop and a
            # recoverable read blip would be misclassified (e.g. as a terminal/gone instance). A malformed
            # status read is a TRANSIENT poll error: count it against the poll-error budget and keep
            # polling, exactly like a VastApiError. (UnicodeDecodeError is a sibling of JSONDecodeError
            # under ValueError that json.loads raises on invalid-UTF8 bytes; the JSONDecodeError clause
            # alone would miss it. http.client.HTTPException covers IncompleteRead from a truncated
            # ``resp.read()`` — also not an OSError, so the _http retry wrapper lets it through raw;
            # the create/ambiguous-create paths in vast.api already treat it the same way.) (Codex/Cursor)
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

        # ``unknown`` is Vast's "host has no recent heartbeat and won't progress" status — a host loss,
        # so treat it as dead for fast failover instead of waiting out the setup/training stall window.
        # But gate it on ``became_running``: ``unknown`` is ALSO the fallback this poller substitutes
        # when a present instance has no ``actual_status`` yet (line above), which happens during normal
        # provisioning — failing a still-booting box on that would be wrong. A box that WAS running and
        # then reports ``unknown`` is the genuine host-loss case (Codex MtrgK).
        dead = (
            missing_streak >= 4
            or status in _DEAD_STATES
            or (became_running and status == "unknown")
        )
        if dead:
            # The worker may have finished right before the box self-destroyed (the normal success order
            # on this substrate), with its DONE/marker not yet visible on HF (read-after-write lag). A
            # SINGLE read can miss it and mis-classify a finished seed as host loss, renting a retry that
            # races the same seed's artifacts — so re-read the terminal artifacts a bounded number of
            # times before concluding loss (Codex). time.sleep is mocked in tests, so this adds no test
            # wall-time; on a genuine host loss (no artifacts ever) it costs a brief bounded wait before
            # the (already non-billing) box fails over.
            terminal = terminal_artifact_result()
            _terminal_tries = _TERMINAL_AFTER_DEAD_RETRIES
            while terminal is None and _terminal_tries > 0:
                say("instance gone; waiting for HF to expose any terminal DONE/marker before failover")
                time.sleep(_TERMINAL_AFTER_DEAD_WAIT_S)
                terminal = terminal_artifact_result()
                _terminal_tries -= 1
            if terminal is not None:
                return terminal
            # Dead host with no ok-marker/DONE. Distinguish a genuine host LOSS (retry on a fresh host)
            # from a worker that RAN and CRASHED early — before it could write the attempt marker — but
            # left error_{phase}.txt (a bad env id, a config/code error, an OOM): that is DETERMINISTIC,
            # so fail FAST. A crash the worker flagged retriable still retries.
            err = _make_hf_file_reader(hf_repo, f"{prefix}/error_{spec.phase}.txt")(force=True)
            # ``error_{phase}.txt`` and the heartbeat are BOTH run-scoped (shared across this run's
            # retries), so a prior attempt can leave either behind. The worker's crash handler uploads
            # the error file AND a heartbeat stamped with THIS attempt + ts together (and error-stage
            # heartbeats are force-uploaded, never throttled), so a genuine current-attempt crash always
            # leaves a fresh attempt-matching heartbeat next to the error. We therefore use heartbeat
            # provenance to attribute the error: treat it as a CURRENT deterministic crash only when
            #  - the latest heartbeat is NOT a leftover from a PRIOR attempt (else the co-located error
            #    is presumed leftover too -> a host LOSS, retry on a fresh host); AND
            #  - the worker did not flag the failure retriable for THIS attempt.
            # Without the first guard, gating only the retriable flag (the 1a28224 fix) flips a genuine
            # retry-after-host-loss into a fail-fast job_failed once the stale flag is ignored. Dating
            # uses _dating_launch (the TRUE launch, 0.0 == unknown) so a now()-fallback can't misjudge a
            # normal heartbeat as prior-attempt (Cursor MtgwT).
            crash_evidence_is_current = not heartbeat_is_stale_prior_attempt(
                heartbeat_reader, launch_ts=_dating_launch, current_attempt=handle.attempt
            )
            worker_crashed = (
                bool(err and err.strip())
                and crash_evidence_is_current
                and not worker_flagged_retriable(
                    heartbeat_reader, launch_ts=_dating_launch, current_attempt=handle.attempt
                )
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
                return finish_from_ok_marker(marker)

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
            # first_liveness_s MIGHT be a wedged host -> 'stalled' (infra-shaped -> the runner fails over
            # cross-provider), instead of burning the full setup grace. The observed-running floor
            # keeps a reattach from fast-failing a box that only just came up. But a healthy box doing a
            # slow per-run pip install + code fetch (the documented cold-start that SETUP_GRACE_S covers)
            # also has no worker heartbeat yet — so before failing over, consult Vast's container-log
            # API: a non-empty tail means the bootstrap is actively producing output (alive), so latch
            # and let setup_grace_s govern instead. Only a genuinely SILENT container (no logs across
            # BOOT_LOG_ABSENT_POLLS) is the wedged host we fast-fail. Mirrors Lambda's boot.log path.
            if (
                not seen_fresh_hb
                and not console_log_seen
                and time.time() - running_since > first_liveness_s
                and observed_running_since is not None
                and time.time() - observed_running_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                if not vast_api.instance_logs(handle.instance_id):
                    # A lone empty read can be a transient log-API error -> require the silence to
                    # persist across BOOT_LOG_ABSENT_POLLS before failing over.
                    console_log_absent_polls += 1
                    if console_log_absent_polls >= BOOT_LOG_ABSENT_POLLS:
                        terminal = terminal_artifact_result()
                        if terminal is not None:
                            return terminal
                        return PollResult(
                            False,
                            failure="stalled",
                            detail=f"no worker heartbeat AND no container-log output for "
                            f"{int(time.time() - running_since)}s after the container started "
                            f"(worker never came up; limit {int(first_liveness_s)}s)",
                        )
                else:
                    # The bootstrap is producing output -> healthy slow cold start, not wedged. Stop
                    # fast-failing; setup_grace_s/stall_after_s below remain the backstop.
                    console_log_seen = True
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
    code_prefix: str | None = None,
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
        o
        for o in usable_offers(
            info.vram_gb,
            _effective_disk_gb(spec),
            max_wall_seconds=float(getattr(spec.gpu, "max_wall_seconds", 0) or 0),
        )
        if o.gpu == spec.gpu.type
    ]
    deploy_kwargs = {
        "attempt": attempt,
        "log": log,
        "runtime_secrets": runtime_secrets,
    }
    if code_prefix is not None:
        deploy_kwargs["code_prefix"] = code_prefix
    handle = deploy_and_submit(spec, seed, offers, **deploy_kwargs)
    # The instance is billing the MOMENT deploy_and_submit returns; the teardown ``finally`` must guard
    # EVERYTHING after that point — including ``on_handle`` (persisting the handle can itself raise).
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}"
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
        # The teardown can't raise here (it would mask the poll result / original exception). But an
        # UNCONFIRMED single-instance destroy (success:false / breakdown) on an abandoned/retriable run is
        # dangerous: while the run stays ``running`` across a retry/recovery the active-run orphan sweep
        # SHIELDS this run's label — so this attempt's possibly-billing box can survive into the next
        # attempt (a fresh box, this handle cleared) with no persisted handle. Escalate to a run-scoped
        # reap by label (destroy_run_instances re-lists + retries and is NOT active-shielded) so this
        # attempt's box is cleared before the next launches; the warning above stays for the operator if
        # even that can't confirm (Codex).
        if not _best_effort_destroy(handle.instance_id, context="submit_run_vast teardown"):
            with contextlib.suppress(Exception):
                destroy_run_instances(spec.run_id)


def _best_effort_destroy(instance_id, *, context: str) -> bool:
    """``destroy_instance`` for the best-effort teardown paths (submit/poll ``finally``, cancel) that
    must NOT raise — a raise in a ``finally`` would mask the poll result or the original exception.
    Returns the confirmation bool and WARNS when teardown is unconfirmed (Vast ``success: false`` /
    breakdown -> the instance may still be billing) so operators get immediate visibility instead of
    waiting for the next ``sweep_orphans`` pass (Copilot). ``VastProvider.destroy`` keeps RAISING for
    its suppress-wrapped callers; this is the variant for contexts where raising is wrong.

    Pass ``instance_id`` THROUGH unconverted: ``destroy_instance`` does the ``int()`` inside its own
    try/except (-> False on a non-numeric/None id), so converting HERE would re-introduce a raise in
    the very ``finally``/``suppress`` paths this helper exists to keep quiet (Cursor MtlVb)."""
    ok = vast_api.destroy_instance(instance_id)
    if not ok:
        logger.warning(
            "vast teardown unconfirmed for instance %s (%s): success:false / breakdown — instance may "
            "still be billing; sweep_orphans is the backstop",
            instance_id,
            context,
        )
    return ok


def _coerce_instance_id(raw) -> int | None:
    """Best-effort ``int()`` for a Vast instance id from a ``list_instances()`` payload: the int, or
    ``None`` for a missing / non-intable id (unexpected API shape / partial response). The best-effort
    cleanup loops (``destroy_run_instances`` / ``sweep_orphans``, both documented "never raises") use
    this to SKIP a bad id instead of letting a bare ``int(iid)`` raise and abort the whole loop —
    leaving the remaining reapable instances billing (Copilot)."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def cancel(remote: dict) -> None:
    """Cross-process cancel: destroy the persisted instance (stops billing)."""
    instance_id = remote.get("instance_id")
    if instance_id:
        _best_effort_destroy(instance_id, context="cancel")


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
        iid = _coerce_instance_id(inst.get("id"))  # skip a non-intable id, don't abort the loop
        label = str(inst.get("label") or "")
        # Match on the label boundary, not a raw string prefix: a label is
        # ``f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"``, so a run's prefix must equal the label
        # or be followed by the ``-s`` seed boundary (else ``flash-100`` would also destroy ``flash-1000``).
        if (
            iid
            and (label == prefix or label.startswith(prefix + "-s"))
            and vast_api.destroy_instance(iid)
        ):
            destroyed.append(iid)
    return destroyed


def run_instances_remaining(run_id: str) -> list[int]:
    """Instance ids that STILL carry ``run_id``'s label right now.

    An empty list is the CONFIRMED-clear signal: no instance for this run remains. A non-empty list
    means a possibly-live instance survives — e.g. ``destroy_run_instances`` hit a ``success:false`` /
    network breakdown (``destroy_instance`` returns False, so the box is not reaped yet) or a phantom
    from a non-idempotent create surfaced via Vast's eventually-consistent instance list. Unlike
    ``destroy_run_instances`` this RAISES on a listing failure (the caller cannot prove the run is
    clear, so it must treat that as not-clear). Used to gate the handle-less recovery resubmit: never
    launch a second worker for a run while an instance for it might still be writing its HF artifacts.

    Uses the STRICT listing (``list_instances(strict=True)``): a truncated/partial page set raises
    rather than returning silently, so an empty result is a COMPLETE enumeration (a real "clear"), not
    an unseen page that could hide a live instance (Cursor).
    """
    if not run_id:
        return []
    # strict: any incomplete enumeration raises -> caller treats as "could not confirm clear" (defers).
    instances = vast_api.list_instances(strict=True)
    prefix = run_label_prefix(run_id)
    remaining: list[int] = []
    for inst in instances:
        label = str(inst.get("label") or "")
        if not (label == prefix or label.startswith(prefix + "-s")):
            continue
        iid = _coerce_instance_id(inst.get("id"))
        if iid is None:
            # A row carrying THIS run's label but a missing/non-numeric id is a possibly-live instance
            # we can neither enumerate as a concrete target nor destroy. Silently skipping it (as the
            # best-effort destroy_run_instances does) would let this confirmation report a FALSE clear
            # and resubmit a handle-less run over a visible box — so RAISE; the caller treats it as
            # not-clear and defers (Codex). (destroy_run_instances stays lenient: it only attempts
            # reaps and the next sweep retries the unparseable row.)
            raise vast_api.VastApiError(
                f"vast instance for run {run_id!r} carries the run label but an unparseable id "
                f"({inst.get('id')!r}); cannot confirm the run is clear"
            )
        remaining.append(iid)
    return remaining


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
        iid = _coerce_instance_id(inst.get("id"))  # skip a non-intable id, don't abort the sweep
        if iid and vast_api.destroy_instance(iid):
            destroyed.append(iid)
            logger.warning("destroyed orphaned vast instance %s (label %s)", iid, label)
    return destroyed
