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
from flash.providers._hf_artifacts import (
    heartbeat_is_stale_prior_attempt,
    make_hf_heartbeat_reader,
    make_hf_text_reader,
    worker_flagged_retriable,
)
from flash.providers._poll import (
    BOOT_LOG_ABSENT_POLLS,
    FIRST_LIVENESS_OBSERVED_FLOOR_S,
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    PROVISION_GRACE_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
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
from flash.providers.vast import api as vast_api
from flash.providers.vast.jobs.builders import (
    VastJobHandle,
    VastOffer,
    build_onstart,
    build_payload,
    instance_label,
    label_matches_run,
    run_label_prefix,
    vast_image,
)

logger = get_logger(__name__)

# Offer-quality floors (on top of the non-negotiable verified+datacenter gate). reliability2 is Vast's
# host-uptime score: 0.995 (~1-in-200) nearly eliminates mid-run host deaths while keeping supply usable.
RELIABILITY_FLOOR = 0.995
MIN_INET_MBPS = 200.0
# LOAD_TIMEOUT_S / SETUP_GRACE_S / STALL_AFTER_S / PROVISION_GRACE_S are the shared instance-poll timing
# defaults imported from ``_poll`` above. The staged setup-vs-training grace is the fix for the historical
# "Vast box dies every ~25-30 min": the old provider used one flat 1500s window that fired mid cold-start
# and tore down healthy boxes. Kept as module globals here so ``monkeypatch.setattr(jobs, …)`` still bites.
# Boards under-report VRAM vs class nominal (L4 23034/24GB, A40 46068/48GB ≈ 0.938). The server-side
# gpu_ram filter gets this slack; the class gate (vast_gpu_for_offer) stays exact.
_SEARCH_VRAM_SLACK = 0.92
# Minimum disk every instance is provisioned with (bootstrap + worker + weights need headroom). The
# offer search MUST use the same floor so a thin-disk offer can't pass search then fail at create.
MIN_DISK_GB = 60.0

# The setup-vs-training stall boundary is the shared ``_poll.is_training_heartbeat`` (same one runpod +
# lambdalabs use) so the cold-start grace rule can't drift between providers.

# Vast states meaning "the container is gone / won't progress". ``frozen`` is paused-but-still-billing
# yet emits no DONE/heartbeat, so classify it dead for fast failover. Unlike ``unknown`` it is never
# this poller's no-status fallback, so it needs no ``became_running`` gate.
_DEAD_STATES = {"exited", "stopped", "offline", "deleted", "frozen"}

# A fresh DONE can precede the separately-uploaded metrics.json (HF read-after-write lag). Re-read
# metrics a few times before falling back to the infra-retryable poll_error, so a DONE-signalled
# success isn't hard-failed on a transient read gap (mirrors Lambda's finish_ok behavior).
_METRICS_AFTER_DONE_RETRIES = 6
_METRICS_AFTER_DONE_WAIT_S = 5.0

# A successful container self-destroys the instant it finishes — often before HF exposes its DONE /
# attempt marker. Re-read terminal artifacts a few times before concluding host loss, so a
# finished-then-gone seed isn't mis-retried against its own artifacts.
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

    Server-side filters do the heavy lifting; everything load-bearing is re-checked client-side.

    ``limit`` is the price-sorted page size. Callers bucket rows BY GPU CLASS, so the page must span
    every fitting managed class — 256 comfortably covers the verified-datacenter market (a specific-class
    caller just filters down further). ``max_wall_seconds`` (>0) also requires offers available for at
    least ``max(60, max_wall) + PROVISION_GRACE_S`` — the same deadline the poller enforces — so the
    search never advertises capacity an offer can't outlast. 0 = no duration floor.
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
        # Accept ONLY verified DATACENTER hosts (hosting_type==1): the onstart ships run secrets to the box.
        _bad_host = r.get("hosting_type") != 1
        if (
            _bad_host
            or r.get("verification") != "verified"
            # Exact class gate: the server-side gpu_ram filter only carries slack, so re-check nominal VRAM.
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
    """Best-effort instance DICT carrying this EXACT (per run/seed/attempt) label, or ``None``. Reclaims
    a contract a possibly-successful-but-unconfirmed create left behind so the walk adopts it instead of
    renting a duplicate; the full dict (not just the id) lets the caller stamp the real launch time. Any
    lookup failure -> ``None`` (caller falls back; the orphan sweep is the backstop)."""
    try:
        for inst in vast_api.list_instances():
            if str(inst.get("label") or "") == label and inst.get("id"):
                return inst
    except Exception as exc:  # listing is best-effort; never let it abort the launch
        logger.warning("vast label reconcile failed (%s); proceeding without adoption", exc)
    return None


def _reconcile_ambiguous_create(spec, offer: VastOffer, label: str, attempt: int, err, say) -> VastJobHandle:
    """Reconcile the contract a possibly-billed AMBIGUOUS create (5xx/429/timeout on the non-idempotent
    PUT /asks) may have left behind. If an instance materialized under our unique label, ADOPT it;
    otherwise destroy this run's instances by label and raise ``UnreconciledCreateError`` to abort the
    offer walk — renting another offer would double-provision. ``say`` is suppressed throughout so a
    logging error can never swallow the terminal raise (which would let the orchestrator retry)."""
    adopted = _adopt_instance_by_label(label)
    # Coerce defensively: a truthy-but-unparseable id can't be adopted -> fall through to the abort.
    adopted_id = _coerce_instance_id(adopted.get("id")) if adopted is not None else None
    if adopted_id is not None:
        # Stamp the box's REAL launch time (start_date epoch) so cost/liveness/stall/deadline timing
        # align with its actual runtime, not this later reconciliation moment.
        started = float(adopted.get("start_date") or 0.0) or time.time()
        with contextlib.suppress(Exception):
            say(
                f"adopted vast instance {adopted_id} from an ambiguous create "
                f"(label={label}, offer {offer.offer_id}, {offer.gpu})"
            )
        return VastJobHandle(
            instance_id=adopted_id,
            offer_id=offer.offer_id,
            machine_id=offer.machine_id,
            label=label,
            gpu=offer.gpu,
            hourly_usd=offer.dph_total,
            attempt=attempt,
            started_ts=started,
        )
    # Nothing cleanly adopted (no row under our label, or an unparseable id). Proactively destroy any
    # phantom by label, then raise TERMINAL (not the retriable VastApiError, which would rent another
    # box while the phantom may still surface and bill under the still-active run).
    destroyed = destroy_run_instances(spec.run_id)
    if destroyed:
        with contextlib.suppress(Exception):
            say(f"destroyed {len(destroyed)} possible phantom instance(s) {destroyed} on abort")
    raise UnreconciledCreateError(
        f"ambiguous vast create on offer {offer.offer_id} (label={label}); aborting the offer walk "
        f"to avoid double-provisioning (destroyed phantom by label): {err}"
    ) from err


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
            )
        except vast_api.VastApiError as e:
            last_err = e
            # suppress: a raising log here must not abort before the ambiguous-create reconcile below.
            with contextlib.suppress(Exception):
                say(f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) rejected: {e}")
            # An AMBIGUOUS failure (5xx/429/timeout/unreadable body on the non-idempotent PUT /asks) may
            # have billed a contract that never surfaced -> reconcile by label (adopt or abort) before
            # renting another offer. A DEFINITIVE 4xx / success=false rejection created nothing: walk on.
            if vast_api.create_error_is_ambiguous(e):
                return _reconcile_ambiguous_create(spec, offer, label, attempt, e, say)
            # Market is live: refresh the search ONCE, re-excluding the machines that just rejected us
            # and staying within the allocator-approved class pool (``offers`` is already class-filtered).
            if not candidates and not refreshed:
                refreshed = True
                allowed = {o.gpu for o in offers}
                candidates = [
                    o
                    for o in usable_offers(
                        min(o.vram_gb for o in offers),
                        _effective_disk_gb(spec),
                        exclude_machine_ids={o.machine_id for o in tried},
                        max_wall_seconds=float(getattr(spec.gpu, "max_wall_seconds", 0) or 0),
                    )
                    if o.gpu in allowed
                ][:5]
            continue
        # suppress: a raising log before we return would skip the teardown finally and leak the box.
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
    hf_repo: str,
    prefix: str,
    phase: str,
    marker: dict | None,
    instance_id: int | None = None,
    attempt: int = 0,
) -> str:
    """Best root-cause detail we can assemble from the HF artifacts + the Vast console.

    Unlike Lambda (no console API -> a host boot.log on HF), Vast exposes a container log API, so
    early-bootstrap failures (pip/env errors before the worker can reach HF) are read live from it.
    """
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    err_name = f"error_{phase}_attempt{int(attempt or 0)}.txt"
    content = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
    if content:
        parts.append(f"--- {err_name} ---\n{content[-2000:]}")
    if instance_id:
        logs = vast_api.instance_logs(int(instance_id))
        if logs:
            parts.append(f"--- instance log tail ---\n{logs[-3000:]}")
    return "\n".join(parts) or "vast worker terminated without a DONE sentinel"


def _read_with_retries(read, *, tries: int, wait_s: float, say, message: str):
    """Call ``read()`` (a zero-arg reader), re-reading up to ``tries`` times ``wait_s`` apart while it
    returns None — for HF read-after-write lag (metrics.json after DONE; terminal artifacts after a box
    self-destructs). ``time.sleep`` is mocked in tests, so this adds no test wall-time."""
    value = read()
    while value is None and tries > 0:
        say(message)
        time.sleep(wait_s)
        value = read()
        tries -= 1
    return value


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

    # started_ts is 0.0 for an old/corrupt handle -> fall back to now for the load/stall clocks + cost.
    launch_ts = handle.started_ts or time.time()
    # Heartbeat attempt-DATING must use the TRUE launch (0.0 == unknown), NOT the now() fallback: a now()
    # anchor makes a normal heartbeat's ts look prior-to-launch, mis-dating a same-attempt crash as a
    # prior-attempt leftover. launch_ts (now-floored) is only for elapsed-time math.
    _dating_launch = handle.started_ts or 0.0

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/vast_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

    def finish_ok(end_ts_hint: float | str | None = None) -> PollResult:
        # metrics.json is written before DONE but HF read-after-write lags: re-read a few times before
        # falling back to the poll_error retry on a DONE-without-metrics.
        raw = _read_with_retries(
            lambda: metrics_reader(force=True),
            tries=_METRICS_AFTER_DONE_RETRIES,
            wait_s=_METRICS_AFTER_DONE_WAIT_S,
            say=say,
            message="DONE seen but metrics.json not visible yet; waiting for HF read-after-write",
        )
        if raw is None:
            # DONE means the worker SIGNALLED SUCCESS; an unreadable metrics.json after the in-line
            # retries is a transient HF read-after-write gap, not a worker error. Mirror Lambda
            # (lambdalabs finish_ok -> poll_error): don't fast-fail a successful run as job_failed —
            # return the infra-retryable poll_error so it gets its bounded infra budget (never a
            # forever-spin: poll_error is capped by infra_retries) instead of a hard terminal failure.
            return PollResult(
                False, failure="poll_error", detail="DONE without metrics.json (transient HF read)"
            )
        try:
            metrics = json.loads(raw)
        except ValueError:
            # A present-but-unparseable metrics.json (truncated read-after-write / corrupt) must NOT
            # escape the poll loop as a raw JSONDecodeError and abort the run past the teardown finally.
            # It is the same transient read-after-write gap on a DONE-signalled success, so classify it
            # exactly like the DONE-without-metrics case above: infra-retryable poll_error, not job_failed.
            return PollResult(
                False, failure="poll_error", detail="DONE with unparseable metrics.json (transient HF read)"
            )
        # Instance wall note anchors to the worker's DONE ts, else the ok-marker's completion ts (so a
        # delayed recovery isn't measured as runtime), else now; adopt only if in [launch, now]. The
        # customer-facing cost below uses the worker training wall only.
        end_ts = time.time()
        # end_ts_hint is the worker's completion time — the DONE sentinel's payload (a str) or an ok
        # marker's ts (a float). float() tolerates surrounding whitespace, so no pre-strip is needed.
        if end_ts_hint is not None:
            with contextlib.suppress(TypeError, ValueError):
                ts = float(end_ts_hint)
                if launch_ts <= ts <= end_ts:
                    end_ts = ts
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
        # An ok marker means the worker finished (metrics.json was written first), even if DONE is stale.
        # Pass DONE only when fresh; else use the marker's own completion ts for the wall note.
        d = done_reader(force=True)
        fresh = d is not None and done_is_fresh(d)
        marker_ts = marker.get("ts") if isinstance(marker, dict) else None
        return finish_ok(d if fresh else marker_ts)

    def fail_from_marker(marker: dict | None) -> PollResult:
        # A real worker error fails fast UNLESS flagged retriable (in the marker, or the worker's
        # heartbeat for a RetriableInfraError). Gate the heartbeat flag to THIS attempt so a stale
        # retriable=True from a prior attempt can't turn a fast-fail into a GPU-burning retry loop.
        retriable = bool(marker and marker.get("retriable")) or worker_flagged_retriable(
            heartbeat_reader, launch_ts=_dating_launch, current_attempt=handle.attempt
        )
        return PollResult(
            False,
            failure="job_preempted" if retriable else "job_failed",
            detail=_failure_detail(
                hf_repo, prefix, spec.phase, marker, handle.instance_id, attempt=handle.attempt
            ),
        )

    def terminal_artifact_result(force: bool = True) -> PollResult | None:
        # The worker's terminal HF artifacts (DONE / attempt marker) -> a terminal PollResult when it
        # definitively finished or errored, else None. The SINGLE terminal detector shared by the poll
        # loop and every give-up path. ``force`` bypasses the read cache: True for the one-shot give-up
        # reads; False for the per-iteration loop poll, which already paces its own reads.
        d = done_reader(force=force)
        if d is not None and done_is_fresh(d):
            return finish_ok(d)
        raw = marker_reader(force=force)
        if raw:
            with contextlib.suppress(ValueError):
                m = json.loads(raw)
                if m.get("ok"):
                    return finish_from_ok_marker(m)
                return fail_from_marker(m)
        return None

    def stalled_unless_terminal(detail: str) -> PollResult:
        # A stall exit still checks for terminal artifacts — the worker may have finished right at the
        # boundary. Use the BOUNDED read (like the deadline / dead-host / poll-error paths) so a fresh
        # DONE/marker not yet visible under HF read-after-write lag isn't missed and mis-classified stalled
        # (which would fail a max_retries=0 run that actually completed, or rent a second box for the seed).
        terminal = _read_with_retries(
            terminal_artifact_result,
            tries=_TERMINAL_AFTER_DEAD_RETRIES,
            wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
            say=say,
            message="stall boundary; waiting for HF to expose any terminal DONE/marker before stalled",
        )
        return terminal if terminal is not None else PollResult(False, failure="stalled", detail=detail)

    poll_errors = PollErrorTracker(say, interval_s)
    # Seed the load/stall clocks from LAUNCH, not this poll's start: a delayed reattach has been billing
    # since launch, so a still-loading box that already blew LOAD_TIMEOUT_S fails over now.
    start = launch_ts
    last_status = None
    last_hb_key = None
    last_progress = start
    became_running = False
    # Anchored to launch so a reattach already-running measures first-liveness from the original launch.
    running_since = start
    # Wall-clock THIS session first saw it running (set once) — "how long WE have watched it", so a
    # reattach doesn't fast-fail a box that just came up.
    observed_running_since = None
    seen_training_hb = False
    # Any FRESH heartbeat from this attempt proves the worker started -> clears the first-liveness
    # deadline (distinct from seen_training_hb, which gates the tighter training window).
    seen_fresh_hb = False
    # Lambda-parity for boot.log liveness: a non-empty CONTAINER LOG tail proves the bootstrap is alive
    # (pip/code fetch can outlast first_liveness_s before the first heartbeat), so we don't fast-fail a
    # healthy cold start. Require silence across BOOT_LOG_ABSENT_POLLS so a log-API blip can't burn a retry.
    console_log_seen = False
    console_log_absent_polls = 0
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            # A recovered run can blow a launch-anchored deadline on the first reattach tick (the outage
            # lasted past max_wall+grace). The worker may also have just finished / self-destructed AT the
            # wall cap with its DONE/marker not yet visible on HF (read-after-write lag), so do the BOUNDED
            # terminal-artifact wait (not a single read) before declaring stalled — else the retry re-rents
            # a GPU for the same seed while the completed attempt's artifacts surface moments later.
            terminal = _read_with_retries(
                terminal_artifact_result,
                tries=_TERMINAL_AFTER_DEAD_RETRIES,
                wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                say=say,
                message="deadline reached; waiting for HF to expose any terminal DONE/marker before giving up",
            )
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
            # A malformed 200 body (truncated / non-JSON / invalid-UTF8) makes RestClient raise a decode
            # / incomplete-read error, NOT a VastApiError — the _http retry wrapper only catches OSError
            # transients, so these escape get_instance raw. Treat them like any transient poll error:
            # count against the budget and keep polling (else a read blip looks like a gone instance).
            if poll_errors.record(e):
                # The status endpoint is down, but the worker may have COMPLETED during the outage and
                # written its terminal DONE/marker to HF (a different endpoint). Do the BOUNDED terminal
                # read (same as the deadline / dead-host paths) before giving up: a prolonged outage can
                # end right as the worker finishes, so a single read can miss the just-written artifact
                # under HF read-after-write lag. Else poll_error tears the box down and the retry relaunches
                # a second worker for an attempt that already finished (duplicate work + double-bill).
                terminal = _read_with_retries(
                    terminal_artifact_result,
                    tries=_TERMINAL_AFTER_DEAD_RETRIES,
                    wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                    say=say,
                    message="status-poll outage; waiting for HF to expose any terminal DONE/marker before poll_error",
                )
                if terminal is not None:
                    return terminal
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        # The instance-detail route transiently answers {"instances": null} for healthy (and brand-new)
        # instances. One missing read means nothing — only a sustained streak is a real disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get("actual_status") or ("missing" if inst is None else "unknown")
        if status != last_status:
            say(f"instance {handle.instance_id}: {status}")
            # Count a status TRANSITION as progress, but NOT the first observation (last_status starts
            # None, so the first read always "changes" — crediting it would hand a silent-since-launch
            # worker a fresh setup grace after every restart).
            if last_status is not None:
                last_progress = time.time()
                if status == "running":
                    running_since = time.time()  # genuine ->running: start the liveness clock
            last_status = status
        if status == "running":
            became_running = True
            if observed_running_since is None:
                observed_running_since = time.time()

        # Per-iteration terminal check: the SAME detector every give-up path uses (force=False — the loop
        # paces its own reads). Folds the DONE and ok/err-marker checks into one call.
        terminal = terminal_artifact_result(force=False)
        if terminal is not None:
            return terminal

        # ``unknown`` = "host has no recent heartbeat and won't progress" (host loss) -> dead for fast
        # failover. Gate on ``became_running`` because ``unknown`` is ALSO this poller's no-status
        # fallback during provisioning; only a box that WAS running then goes unknown is genuine loss.
        dead = (
            missing_streak >= 4
            or status in _DEAD_STATES
            or (became_running and status == "unknown")
        )
        if dead:
            # The worker may have finished just before the box self-destroyed, with its DONE/marker not
            # yet visible on HF (read-after-write lag). Re-read terminal artifacts a bounded number of
            # times before concluding loss.
            terminal = _read_with_retries(
                terminal_artifact_result,
                tries=_TERMINAL_AFTER_DEAD_RETRIES,
                wait_s=_TERMINAL_AFTER_DEAD_WAIT_S,
                say=say,
                message="instance gone; waiting for HF to expose any terminal DONE/marker before failover",
            )
            if terminal is not None:
                return terminal
            # Dead host, no ok-marker/DONE. Distinguish genuine host LOSS (retry on a fresh host) from a
            # worker that RAN and CRASHED early leaving error_{phase}_attempt<N>.txt (bad env/config/OOM):
            # that is DETERMINISTIC -> fail FAST. A crash the worker flagged retriable still retries.
            err_name = f"error_{spec.phase}_attempt{int(handle.attempt or 0)}.txt"
            err = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
            # Error files are attempt-scoped but the heartbeat is run-scoped: gate the crash evidence on
            # heartbeat provenance (dated by _dating_launch, the TRUE launch) so a stale prior-attempt
            # heartbeat can't flip a genuine host-loss retry into a fast-fail job_failed.
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
                detail=_failure_detail(
                    hf_repo, prefix, spec.phase, None, handle.instance_id, attempt=handle.attempt
                ),
            )

        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            # Credit the heartbeat's OWN ts (clamped to [launch, now]) so a pre-restart-stale heartbeat
            # buys no fresh window. ``fresh`` is False for a leftover prior-attempt heartbeat.
            hb_ts, fresh = heartbeat_progress_ts(new_key, launch_ts, handle.attempt)
            if fresh:
                seen_fresh_hb = True  # any fresh hb (incl. a bare liveness ping) disarms first-liveness
                # Advance the stall clock ONLY on a STAGED heartbeat (stage is not None): a bare liveness
                # ping must not let a wedged worker keep resetting the setup/training stall window
                # (mirrors the Lambda poller). MONOTONIC: never regress on an out-of-order upload.
                if stage is not None:
                    last_progress = max(last_progress, hb_ts)
                    # Tighten setup_grace -> stall only once training genuinely begins (the shared helper
                    # keeps cold-start pings, incl. the silent step=0 first rollout, under setup grace).
                    if is_training_heartbeat(stage, new_key[1]):
                        seen_training_hb = True

        # Load timeout: a box that never left loading/unknown within LOAD_TIMEOUT_S never started.
        # But a fresh heartbeat PROVES the worker booted even while the detail API lags in
        # loading/unknown (never flipping to 'running'), so it disarms this — else a healthy,
        # heartbeating box is torn down at 15m on a lagging status feed. deadline_s (+ the finally
        # destroy and periodic sweep) stays the ultimate spend backstop once this is disarmed.
        # ORDER MATTERS: this runs AFTER the heartbeat read above, so THIS tick's first fresh heartbeat
        # (which sets seen_fresh_hb) disarms the timeout on the very iteration it arrives — a heartbeat
        # landing exactly at the 15m mark must not be raced by a stale seen_fresh_hb from the prior tick.
        if not became_running and not seen_fresh_hb and time.time() - start > LOAD_TIMEOUT_S:
            return PollResult(
                False,
                failure="stalled",
                detail=f"instance stuck in '{status}' for {int(time.time() - start)}s "
                f"(never started; image pull / host issue)",
            )

        if became_running:
            # Fast-failover: a box that reached 'running' but emitted NO heartbeat past first_liveness_s
            # might be wedged -> 'stalled'. But a healthy slow cold start (pip install / code fetch) also
            # has no heartbeat yet, so before failing over consult the container-log API: a non-empty tail
            # means the bootstrap is alive (latch, let setup_grace_s govern); only a container SILENT
            # across BOOT_LOG_ABSENT_POLLS is the wedged host. The observed-running floor keeps a reattach
            # from fast-failing a box that just came up. Mirrors Lambda's boot.log path.
            if (
                not seen_fresh_hb
                and not console_log_seen
                and time.time() - running_since > first_liveness_s
                and observed_running_since is not None
                and time.time() - observed_running_since > FIRST_LIVENESS_OBSERVED_FLOOR_S
            ):
                if not vast_api.instance_logs(handle.instance_id):
                    # A lone empty read can be a transient log-API error -> require BOOT_LOG_ABSENT_POLLS.
                    console_log_absent_polls += 1
                    if console_log_absent_polls >= BOOT_LOG_ABSENT_POLLS:
                        return stalled_unless_terminal(
                            f"no worker heartbeat AND no container-log output for "
                            f"{int(time.time() - running_since)}s after the container started "
                            f"(worker never came up; limit {int(first_liveness_s)}s)"
                        )
                else:
                    # Bootstrap is producing output -> healthy slow cold start; setup/stall below backstop.
                    console_log_seen = True
            limit = stall_after_s if seen_training_hb else setup_grace_s
            if time.time() - last_progress > limit:
                phase = "training" if seen_training_hb else "setup (pre-training)"
                return stalled_unless_terminal(
                    f"no worker progress for {int(time.time() - last_progress)}s during "
                    f"{phase} (instance status {status}, limit {int(limit)}s)"
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
    handle = deploy_and_submit(
        spec, seed, offers, attempt=attempt, log=log, runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
    )
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
        # An UNCONFIRMED single-instance destroy (success:false / breakdown) on a retriable run is
        # dangerous: while the run stays ``running`` the active-run sweep SHIELDS its label, so this
        # attempt's possibly-billing box could survive into the next attempt handle-less. Escalate to a
        # run-scoped reap by label (not active-shielded) so it's cleared before the next launch.
        if not _best_effort_destroy(handle.instance_id, context="submit_run_vast teardown"):
            with contextlib.suppress(Exception):
                destroy_run_instances(spec.run_id)


def _best_effort_destroy(instance_id, *, context: str) -> bool:
    """``destroy_instance`` for best-effort teardown paths (submit/poll ``finally``, cancel) that must
    NOT raise. Returns the confirmation bool and WARNS on an unconfirmed teardown (``success: false`` /
    breakdown -> may still be billing) for immediate operator visibility. (``VastProvider.destroy`` wraps
    this and RE-RAISES on failure for its suppress-wrapped callers.)

    Pass ``instance_id`` THROUGH unconverted: ``destroy_instance`` does the ``int()`` internally, so
    converting here would re-introduce a raise in the very ``finally``/``suppress`` paths this quiets."""
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
    """Best-effort ``int()`` for a Vast instance id, or ``None`` for a missing/non-intable id. The
    never-raises cleanup loops use this to SKIP a bad id rather than let ``int()`` abort the whole loop
    and leave the remaining reapable instances billing."""
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
        # Match on the label boundary (not a raw prefix) so ``flash-100`` can't also destroy ``flash-1000``.
        if iid and label_matches_run(label, prefix) and vast_api.destroy_instance(iid):
            destroyed.append(iid)
    return destroyed


def run_instances_remaining(run_id: str) -> list[int]:
    """Instance ids that STILL carry ``run_id``'s label right now.

    An empty list is the CONFIRMED-clear signal; a non-empty list means a possibly-live instance
    survives (an unconfirmed destroy, or a phantom from a non-idempotent create). Unlike
    ``destroy_run_instances`` this RAISES on a listing failure and uses the STRICT listing (a partial
    page set raises), so an empty result is a COMPLETE enumeration — never an unseen page hiding a live
    box. Gates the handle-less recovery resubmit: never launch a second worker while an instance for the
    run might still be writing its HF artifacts.
    """
    if not run_id:
        return []
    # strict: any incomplete enumeration raises -> caller treats as "could not confirm clear" (defers).
    instances = vast_api.list_instances(strict=True)
    prefix = run_label_prefix(run_id)
    remaining: list[int] = []
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label_matches_run(label, prefix):
            continue
        iid = _coerce_instance_id(inst.get("id"))
        if iid is None:
            # A row with THIS run's label but a non-numeric id is possibly-live yet un-targetable.
            # Skipping it (as the lenient destroy_run_instances does) would report a FALSE clear and
            # resubmit over a live box -> RAISE so the caller defers.
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

    Run at server startup (crash recovery) and after runs. Only ``flash-`` prefixed labels are ever
    touched. ``active_labels`` (raw run ids, or a callable resolved AFTER listing to close the launch
    race) are the protected live runs; each is passed through ``run_label_prefix``.

    ``known_labels`` (optional) is the universe of runs THIS plane knows: when supplied, an instance is
    reaped ONLY if its label maps to one — the multi-plane guard so two planes sharing one Vast account
    never reap each other's boxes. ``None`` = legacy unscoped (reap every non-active ``flash-`` box).
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
        return any(label_matches_run(label, p) for p in prefixes)

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
