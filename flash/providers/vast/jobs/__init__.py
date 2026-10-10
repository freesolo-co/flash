"""Run Vast jobs from verified offers through HF-artifact completion and teardown.
The worker image is the rented container and needs no inbound access. Runner finally, self-destroy,
cancel, or orphan sweep must destroy every rental. Keep lifecycle symbols here for test seams.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import math
import sys
import threading
import time
from collections.abc import Callable

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.poll import (
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
    make_say,
)
from flash.providers._lifecycle.instances.poll_instance import (
    InstancePollAdapter,
    poll_instance_job,
)
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
    remaining_seconds,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers.artifacts.hf import (
    error_artifact_name,
    heartbeat_reader_for,
    make_hf_text_reader,
)
from flash.providers.core.base import (
    GPU_INFO,
    PollResult,
    RunExhaustedProviderPoolError,
    UnreconciledCreateError,
    UnsupportedGpuError,
    canonical_gpu,
    min_cuda_modern,
    vast_gpu_for_offer,
)
from flash.providers.vast.client import api as vast_api
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
# setup and training use separate poll graces; LOAD_TIMEOUT_S remains a test seam.
# vast boards under-report VRAM, so search gets slack while vast_gpu_for_offer remains exact.
_SEARCH_VRAM_SLACK = 0.92
# Minimum disk every instance is provisioned with (bootstrap + worker + weights need headroom). The
# offer search MUST use the same floor so a thin-disk offer can't pass search then fail at create.
MIN_DISK_GB = 60.0

# Vast states meaning "the container is gone / won't progress". ``frozen`` is paused-but-still-billing
# yet emits no DONE/heartbeat, so classify it dead for fast failover. Unlike ``unknown`` it is never
# this poller's no-status fallback, so it needs no ``became_running`` gate.
_DEAD_STATES = {"exited", "stopped", "offline", "deleted", "frozen"}

# Machines this run has already rented and lost, keyed by run id.
#
# ``deploy_and_submit``'s own ``tried`` list is rebuilt per call and only covers offers that
# REJECTED the create. A machine that accepts the rental and then never boots is the worse case: it
# is not in ``tried``, it stays in the market at the top of the cheapest-first ranking, and the next
# attempt rents it again. a failed run once rented the same two dead offers eleven and seven times,
# spending its whole retry budget re-renting known-dead boxes.
#
# Process-local and best-effort by design. It is an optimisation over the ranking, not a
# correctness gate: a control plane restart or a second process legitimately starts with an empty
# set and merely re-learns. Persisting it would put market trivia in the run record and still not
# be authoritative, since offer ids churn.
_run_dead_machines: dict[str, set[int]] = {}
# supervision runs on background threads (flash/server/asgi/app.py, supervise/attach.py), so two runs can
# reach the map at once. its mutations are read-modify-write (size check then evict; setdefault then
# add), which is not atomic under the gil.
_dead_machines_lock = threading.Lock()
# bound the per-process footprint of the map above: a long-lived control plane submits unboundedly
# many runs, and nothing else would ever evict a finished run's entry.
_DEAD_MACHINE_RUNS_MAX = 512
# widened row cap for the one re-search that separates "this run burned the cheap page" from "the
# class is gone". only paid when the default page is entirely blacklisted, which needs a run that
# already lost that many hosts.
_EXHAUSTION_RECHECK_LIMIT = 1024
# The tail of vast's own ``load_timeout_detail`` below. It is what identifies the ONE stall that
# indicts the host, and interpolating the same constant into both sides is what keeps them from
# drifting apart.
#
# Keying on ``failure == "stalled"`` alone was wrong: four different conditions report that name,
# and only this one is the pre-boot load timeout (``_classify_load_timeout``, gated on ``not
# state.became_running``). The others -- a mid-TRAINING progress stall, a post-running liveness
# stall, and the client-side wall deadline -- all describe a box that booted and worked. Retiring a
# healthy host for one of those shrinks the pool on every attempt and, with a small pool, makes the
# next resumable attempt hit the "already rented and lost" error instead of reusing the only machine
# available: the same starvation this fix exists to stop, arriving from the other direction.
_NEVER_STARTED_MARKER = "never started; image pull / host issue"


def _is_never_started_stall(result: PollResult) -> bool:
    """True only for the pre-boot load timeout: rented, never ran, never sent a heartbeat."""
    return result.failure == "stalled" and _NEVER_STARTED_MARKER in (result.detail or "")


def _note_dead_machine(run_id: str, machine_id: int | None) -> None:
    """Remember that ``machine_id`` took this run's money and did not deliver a worker."""
    if not run_id or not machine_id or machine_id <= 0:
        return
    with _dead_machines_lock:
        if run_id not in _run_dead_machines and len(_run_dead_machines) >= _DEAD_MACHINE_RUNS_MAX:
            # drop the oldest run's set (dicts preserve insertion order) rather than let the map
            # grow without bound. evicting a live run's set only forfeits the optimisation for it.
            with contextlib.suppress(StopIteration):
                _run_dead_machines.pop(next(iter(_run_dead_machines)), None)
        _run_dead_machines.setdefault(run_id, set()).add(int(machine_id))


def dead_machine_ids(run_id: str) -> frozenset[int]:
    """Machines this run already rented and lost; excluded from further offer searches."""
    with _dead_machines_lock:
        return frozenset(_run_dead_machines.get(run_id, ()))


def forget_dead_machines(run_id: str) -> None:
    """Drop a finished run's blacklist so the map does not grow for the process's lifetime."""
    with _dead_machines_lock:
        _run_dead_machines.pop(run_id, None)


def _effective_disk_gb(spec) -> float:
    """The disk size an instance is actually provisioned with (the create-time floor).

    Both the offer search and ``create_instance`` must agree on this, or offers with a disk between
    ``spec.gpu.disk_gb`` and the floor pass the search then fail to rent.
    """
    return max(float(spec.gpu.disk_gb), MIN_DISK_GB)


def _exact_search_aliases(info) -> tuple[str, ...]:
    """Return Vast aliases safe for an exact-class search.

    Keep only aliases that canonicalize to the pinned class, or ``verify_gpu`` rejects the rented
    board. Drop ambiguous and unknown spellings.
    """
    kept: list[str] = []
    for alias in info.vast_aliases:
        try:
            if canonical_gpu(alias) == info.name:
                kept.append(alias)
        except UnsupportedGpuError:
            pass
    return tuple(kept)


def _rent_duration_floor(spec, deadline_at: float, *, now: float | None = None) -> float:
    """Return the minimum offer duration from rent time to the launch deadline.

    Workload profiles may include provisioning allowance beyond the wall grant; using the actual
    deadline prevents renting a host that expires during boot. Never shorten below the granted wall.
    """
    grant = float(getattr(spec.gpu, "max_wall_seconds", 0) or 0)
    remaining = remaining_seconds(deadline_at, now=now)
    if not math.isfinite(remaining) or remaining <= 0:
        return grant
    return max(grant, remaining)


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    limit: int = 256,
    max_wall_seconds: float = 0,
    gpu_type: str = "",
    num_gpus: int = 1,
    deadline_at: float | None = None,
) -> list[VastOffer]:
    """Return fitting verified-datacenter offers, cheapest first.
    ``num_gpus`` must match one-machine shape because create has no count parameter. Recheck all
    load-bearing filters client-side; positive wall time adds an offer-duration floor.
    """
    min_duration = (
        max(60.0, float(max_wall_seconds)) if max_wall_seconds and max_wall_seconds > 0 else 0
    )
    exact_info = GPU_INFO.get(gpu_type) if gpu_type else None
    if gpu_type and exact_info is None:
        raise ValueError(f"unknown exact Vast GPU class {gpu_type!r}")
    # Seed an exact search only with spellings that will attest as this class on the box (the ambiguous
    # vast_name itself is always kept and disambiguated by the max_vram_mb ceiling below); a cross-
    # architecture capacity alias would rent a board that live-device attestation then rejects.
    gpu_names = (
        (exact_info.vast_name, *_exact_search_aliases(exact_info))
        if exact_info is not None and exact_info.vast_name
        else ()
    )
    search_vram_gb = max(min_vram_gb, exact_info.vram_gb if exact_info is not None else 0)
    search_kwargs = {"gpu_names": gpu_names} if gpu_names else {}
    if exact_info is not None:
        search_kwargs["max_vram_mb"] = int(exact_info.vram_gb * 1024)
    cards = max(1, int(num_gpus))
    rows = vast_api.search_offers(
        int(search_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        min_duration_seconds=min_duration,
        limit=int(limit),
        num_gpus=cards,
        **search_kwargs,
        **deadline_kwargs(vast_api.search_offers, deadline_at),
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
            or (gpu_type and gpu != gpu_type)
            or float(r.get("reliability2") or 0) < RELIABILITY_FLOOR
            or float(r.get("disk_space") or 0) < float(disk_gb)
            or float(r.get("inet_down") or 0) < MIN_INET_MBPS
            or cuda < float(min_cuda_modern(gpu))  # Blackwell needs CUDA-13 drivers
            or dph <= 0
            # the card count is load-bearing twice over (it sizes the rented box AND divides
            # dph_total into the per-card rate), so re-check it rather than trusting the server
            # honoured the num_gpus filter.
            or int(r.get("num_gpus") or 0) != cards
            or int(r.get("machine_id") or 0) in exclude_machine_ids
        ):
            continue
        out.append(
            VastOffer(
                offer_id=int(r["id"]),
                machine_id=int(r.get("machine_id") or 0),
                gpu=gpu,
                vram_gb=info.vram_gb,
                # dph_total prices the WHOLE offer (all `cards` GPUs); every consumer of dph_total
                # treats it as one card's rate, so divide here at the single boundary where the
                # count is known. Skipping this prices an N-card box N times over and the allocator
                # would never choose one.
                dph_total=dph / cards,
                cuda_max_good=cuda,
                disk_space=float(r.get("disk_space") or 0),
                reliability=float(r.get("reliability2") or 0),
                inet_down=float(r.get("inet_down") or 0),
                geolocation=str(r.get("geolocation") or ""),
                gpu_count=cards,
            )
        )
    return sorted(out, key=lambda o: (o.dph_total, o.vram_gb))


def _adopt_instance_by_label(
    label: str,
    *,
    deadline_at: float | None = None,
) -> dict | None:
    """Best-effort instance DICT carrying this EXACT (per run/attempt) label, or ``None``. Reclaims
    a contract a possibly-successful-but-unconfirmed create left behind so the walk adopts it instead of
    renting a duplicate; the full dict (not just the id) lets the caller stamp the real launch time. Any
    lookup failure -> ``None`` (caller falls back; the orphan sweep is the backstop)."""
    try:
        for inst in vast_api.list_instances(
            **deadline_kwargs(vast_api.list_instances, deadline_at),
        ):
            if str(inst.get("label") or "") == label and inst.get("id"):
                return inst
    except Exception as exc:  # listing is best-effort; never let it abort the launch
        logger.warning("vast label reconcile failed: %s", exc)
    return None


def _cleanup_unpublished_instance(run_id: str, instance_id: int, *, context: str) -> bool:
    """clean an exact unpublished instance, falling back to its run label only when unconfirmed."""
    exact_delete_confirmed = False
    with contextlib.suppress(BaseException):
        exact_delete_confirmed = _best_effort_destroy(instance_id, context=context)
    if not exact_delete_confirmed:
        with contextlib.suppress(BaseException):
            destroy_run_instances(run_id)
    return exact_delete_confirmed


def _reconcile_ambiguous_create(
    spec,
    offer: VastOffer,
    label: str,
    attempt: int,
    err,
    say,
    *,
    deadline_at: float,
) -> VastJobHandle:
    """Reconcile the contract a possibly-billed AMBIGUOUS create (5xx/429/timeout on the non-idempotent
    PUT /asks) may have left behind. If an instance materialized under our unique label, ADOPT it;
    otherwise destroy this run's instances by label and raise ``UnreconciledCreateError`` to abort the
    offer walk — renting another offer would double-provision. ``say`` is suppressed throughout so a
    logging error can never swallow the terminal raise (which would let the orchestrator retry)."""
    adopted = _adopt_instance_by_label(
        label,
        **deadline_kwargs(_adopt_instance_by_label, deadline_at),
    )
    # Coerce defensively: a truthy-but-unparseable id can't be adopted -> fall through to the abort.
    adopted_id = _coerce_instance_id(adopted.get("id")) if adopted is not None else None
    if adopted_id is not None:
        # Stamp the box's REAL launch time (start_date epoch) so cost/liveness/stall/deadline timing
        # align with its actual runtime, not this later reconciliation moment.
        started_raw = adopted.get("start_date")
        if isinstance(started_raw, bool) or not isinstance(started_raw, (int, float)):
            _cleanup_unpublished_instance(
                spec.run_id,
                adopted_id,
                context="ambiguous create with invalid launch timestamp",
            )
            raise UnreconciledCreateError(
                "ambiguous vast create returned an invalid launch timestamp"
            )
        started = float(started_raw)
        if not math.isfinite(started) or started <= 0:
            _cleanup_unpublished_instance(
                spec.run_id,
                adopted_id,
                context="ambiguous create with invalid launch timestamp",
            )
            raise UnreconciledCreateError(
                "ambiguous vast create returned an invalid launch timestamp"
            )
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
            hourly_usd=offer.dph_total * offer.gpu_count,
            attempt=attempt,
            started_ts=started,
        )
    # clean exact contract evidence first, then retain the run-label fallback when unconfirmed.
    contract_id = getattr(err, "contract_id", None)
    exact_delete_confirmed = False
    destroyed: list[int] = []
    if contract_id is not None:
        exact_delete_confirmed = _cleanup_unpublished_instance(
            spec.run_id,
            contract_id,
            context="ambiguous create reconciliation",
        )
    else:
        with contextlib.suppress(BaseException):
            destroyed = destroy_run_instances(spec.run_id)
    if destroyed:
        with contextlib.suppress(Exception):
            say(f"destroyed {len(destroyed)} possible phantom instance(s) {destroyed} on abort")
    cleanup = (
        f"confirmed exact deletion of contract {contract_id}"
        if exact_delete_confirmed
        else "attempted phantom cleanup by run label"
    )
    raise UnreconciledCreateError(
        f"ambiguous vast create on offer {offer.offer_id} (label={label}); aborting the offer walk "
        f"to avoid double-provisioning ({cleanup}; {type(err).__name__})"
    ) from err


def deploy_and_submit(
    spec,
    offers: list[VastOffer],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> VastJobHandle:
    """Rent the cheapest accepting offer, walking live-market rejections.

    Try five ranked offers, then refresh once while excluding machines already tried.
    """
    from flash.core.spec import gpu_count_of

    say = make_say(log)
    absolute_deadline = require_deadline_at(deadline_at)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(
        spec,
        attempt,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        **deadline_kwargs(build_payload, absolute_deadline),
    )
    label = instance_label(spec.run_id, attempt)
    onstart = build_onstart(payload)

    tried: list[VastOffer] = []
    candidates = list(offers[:5])
    refreshed = False
    last_err: Exception | None = None
    create_attempted = False
    try:
        while candidates:
            offer = candidates.pop(0)
            tried.append(offer)
            offer_id = offer.offer_id
            image = vast_image(offer.gpu)
            disk_gb = _effective_disk_gb(spec)
            require_create_allowance(absolute_deadline)
            create_attempted = True
            try:
                instance_id = vast_api.create_instance(
                    offer_id,
                    image=image,
                    disk_gb=disk_gb,
                    env={},
                    onstart=onstart,
                    label=label,
                    **deadline_kwargs(vast_api.create_instance, absolute_deadline),
                )
            except vast_api.VastApiError as e:
                last_err = e
                if vast_api.create_error_is_ambiguous(e):
                    try:
                        return _reconcile_ambiguous_create(
                            spec,
                            offer,
                            label,
                            attempt,
                            e,
                            say,
                            **deadline_kwargs(_reconcile_ambiguous_create, absolute_deadline),
                        )
                    except UnreconciledCreateError:
                        create_attempted = False
                        raise
                create_attempted = False
                with contextlib.suppress(Exception):
                    say(
                        f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) "
                        f"rejected: {sanitize_diagnostic(e, limit=1000)}"
                    )
                if not candidates and not refreshed:
                    refreshed = True
                    allowed = {o.gpu for o in offers}
                    candidates = [
                        o
                        for o in usable_offers(
                            min(o.vram_gb for o in offers),
                            _effective_disk_gb(spec),
                            # this call's create-rejections AND the machines earlier attempts of
                            # this run rented and lost. `tried` alone is per-call, so a refresh
                            # would happily re-offer a box a previous attempt already killed.
                            exclude_machine_ids={o.machine_id for o in tried}
                            | dead_machine_ids(spec.run_id),
                            # the exclusion this refresh exists to apply is what makes the default
                            # page too small: `search_offers` caps rows SERVER-side on a price-sorted
                            # prefix, and the machines are dropped client-side afterwards. so the
                            # more boxes this run has burned, the more of the page is already spent
                            # -- and once they fill it the refresh returns empty while dearer usable
                            # capacity sits just past the cap. widen for the same reason, and by the
                            # same amount, as the exhaustion recheck below.
                            limit=_EXHAUSTION_RECHECK_LIMIT,
                            max_wall_seconds=_rent_duration_floor(spec, absolute_deadline),
                            # the transient attempt spec always carries the concrete allocated class.
                            gpu_type=spec.gpu.type,
                            # refresh the SHAPE the allocator chose: dropping the count here would
                            # rent a single-card offer while the worker still starts n ranks.
                            num_gpus=gpu_count_of(spec),
                            **deadline_kwargs(usable_offers, absolute_deadline),
                        )
                        if o.gpu in allowed
                    ][:5]
                continue
            try:
                with contextlib.suppress(Exception):
                    say(
                        f"rented vast instance {instance_id}: {offer.gpu} ${offer.dph_total:.2f}/hr "
                        f"(offer {offer.offer_id}, {offer.geolocation}, reliability "
                        f"{offer.reliability:.3f}) attempt={attempt}"
                    )
                return VastJobHandle(
                    instance_id=instance_id,
                    offer_id=offer.offer_id,
                    machine_id=offer.machine_id,
                    label=label,
                    gpu=offer.gpu,
                    # ``dph_total`` was divided down to a PER-CARD rate for allocator ranking; the
                    # handle's rate is billed against wall-clock once by both the cost stamp and
                    # realized COGS, so restore the whole-offer price or an n-card box
                    # under-reports by exactly n.
                    hourly_usd=offer.dph_total * offer.gpu_count,
                    attempt=attempt,
                    started_ts=time.time(),
                )
            except BaseException:
                create_attempted = False
                _cleanup_unpublished_instance(
                    spec.run_id,
                    instance_id,
                    context="post-create handle acquisition",
                )
                raise
        raise vast_api.VastApiError(
            f"all {len(tried)} vast offers rejected the job: "
            f"{sanitize_diagnostic(last_err, limit=1000)}"
        )
    except BaseException:
        if create_attempted:
            with contextlib.suppress(BaseException):
                destroy_run_instances(spec.run_id)
        raise


# Rate-limited reader for one HF artifact's text content (None until it exists). Shared with runpod's
# poller via make_hf_text_reader; kept under this module-local name because tests monkeypatch
# ``vast.jobs._make_hf_file_reader`` and the poll/failure paths resolve it as a module global.
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(
    hf_repo: str,
    prefix: str,
    phase: str,
    marker: dict | None,
    instance_id: int,
    attempt: int = 0,
) -> str:
    """Assemble bounded failure detail from worker artifacts and the Vast console."""
    parts = []
    if marker and marker.get("error"):
        parts.append(sanitize_diagnostic(marker["error"], limit=4096))
    err_name = error_artifact_name(phase, attempt)
    content = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
    if content:
        parts.append(f"--- {err_name} ---\n{sanitize_diagnostic(content[-4096:], limit=4096)}")
    logs = vast_api.instance_logs(int(instance_id))
    if logs:
        parts.append(f"--- instance log tail ---\n{sanitize_diagnostic(logs[-4096:], limit=4096)}")
    return "\n".join(parts) or "vast worker terminated without a strict terminal marker"


def poll_vast_job(
    handle: VastJobHandle,
    spec,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    setup_grace_s: float = SETUP_GRACE_S,
    stall_after_s: float = STALL_AFTER_S,
    first_liveness_s: float = FIRST_LIVENESS_S,
    deadline_at: float | None = None,
) -> PollResult:
    """Poll Vast status and HF artifacts through the shared instance-poll kernel.

    Stamp cost from worker training wall, use container logs for liveness, and build failures from
    HF artifacts plus the instance log. Read ``LOAD_TIMEOUT_S`` here to preserve the test seam.
    """
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    err_name = error_artifact_name(spec.phase, handle.attempt)

    def stamp_cost_and_notes(metrics, *, end_ts, launch_ts) -> None:
        # Customer cost is the worker TRAIN wall x the offer's live $/hr; the instance-wall note anchors
        # to the worker's DONE / ok-marker ts (already resolved into ``end_ts``), else now.
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

    adapter = InstancePollAdapter(
        instance_id=handle.instance_id,
        run_id=spec.run_id,
        current_attempt=handle.attempt,
        launch_ts=handle.started_ts,
        done_reader=_make_hf_file_reader(
            hf_repo,
            f"{prefix}/DONE",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        marker_reader=_make_hf_file_reader(
            hf_repo,
            f"{prefix}/vast_attempt{handle.attempt}.json",
            min_interval_s=60.0,
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        metrics_reader=_make_hf_file_reader(
            hf_repo,
            f"{prefix}/metrics.json",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        # Resolve get_instance / instance_logs on the api MODULE at call time so a monkeypatch bites.
        fetch_instance=lambda: vast_api.get_instance(
            handle.instance_id,
            **deadline_kwargs(vast_api.get_instance, absolute_deadline),
        ),
        # A malformed 200 body (truncated / non-JSON / invalid-UTF8) makes RestClient raise a decode /
        # incomplete-read error, NOT a VastApiError — the _http retry wrapper only catches OSError
        # transients, so these escape get_instance raw. Treat them like any transient poll error.
        poll_error_exceptions=(
            vast_api.VastApiError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            http.client.HTTPException,
        ),
        status_field="actual_status",
        running_status="running",
        dead_states=_DEAD_STATES,
        missing_dead_threshold=4,
        # A non-empty CONTAINER LOG tail proves the bootstrap is alive during a slow cold start.
        early_liveness_alive=lambda: bool(
            vast_api.instance_logs(
                handle.instance_id,
                **deadline_kwargs(vast_api.instance_logs, absolute_deadline),
            )
        ),
        read_current_error=lambda: _make_hf_file_reader(
            hf_repo,
            f"{prefix}/{err_name}",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        )(force=True),
        stamp_cost_and_notes=stamp_cost_and_notes,
        failure_detail=lambda marker: _failure_detail(
            hf_repo, prefix, spec.phase, marker, handle.instance_id, attempt=handle.attempt
        ),
        load_timeout_detail=lambda status, elapsed: (
            f"instance stuck in '{status}' for {int(elapsed)}s ({_NEVER_STARTED_MARKER})"
        ),
        first_liveness_detail=lambda elapsed, fl: (
            f"no worker heartbeat AND no container-log output for {int(elapsed)}s after the container "
            f"started (worker never came up; limit {int(fl)}s)"
        ),
    )
    return poll_instance_job(
        adapter,
        log=log,
        interval_s=interval_s,
        heartbeat_reader=heartbeat_reader,
        setup_grace_s=setup_grace_s,
        stall_after_s=stall_after_s,
        first_liveness_s=first_liveness_s,
        load_timeout_s=LOAD_TIMEOUT_S,
        **deadline_kwargs(poll_instance_job, absolute_deadline),
    )


def submit_attempt_vast(
    spec,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> PollResult:
    """Vast equivalent of ``lambdalabs.jobs.submit_attempt_lambda``: rent, persist, poll, destroy.

    The ``finally`` destroy is the cost-safety primary: every exit path — success, failure, stall,
    exception, KeyboardInterrupt — tears the paid instance down.
    """
    # GPU_INFO is keyed by concrete GPU class; a policy word ("cheapest"/"auto") would KeyError opaquely.
    # The allocator resolves policy words to a concrete class upstream, so reaching here with one is a
    # caller bug — name it clearly.
    if spec.gpu.type not in GPU_INFO:
        raise vast_api.VastApiError(
            f"submit_attempt_vast needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    from flash.core.spec import gpu_count_of

    absolute_deadline = require_deadline_at(deadline_at)
    info = GPU_INFO[spec.gpu.type]
    # the market for this class BEFORE this run's own dead hosts come out. one search answers both
    # questions -- what to rent, and why the list is empty when it is -- because `usable_offers`
    # applies `exclude_machine_ids` client-side anyway, so filtering here costs no extra call.
    market = [
        o
        for o in usable_offers(
            info.vram_gb,
            _effective_disk_gb(spec),
            max_wall_seconds=_rent_duration_floor(spec, absolute_deadline),
            # the transient attempt spec always carries the concrete allocated class.
            gpu_type=spec.gpu.type,
            # rent the SHAPE the allocator chose: the worker spawns gpu.count ranks, so a
            # single-card offer would oversubscribe one card with n ranks while billing for n.
            num_gpus=gpu_count_of(spec),
            **deadline_kwargs(usable_offers, absolute_deadline),
        )
        if o.gpu == spec.gpu.type
    ]
    excluded = dead_machine_ids(spec.run_id)
    offers = [o for o in market if o.machine_id not in excluded]
    if market and not offers:
        # every row of the cheapest page belongs to a host this run already lost. the page is a
        # price-sorted prefix, not the class: `search_offers` applies its row cap server-side and
        # the machine exclusion is client-side, so dearer usable capacity can sit just past the
        # cap. widen once and re-filter before calling the class exhausted -- concluding it from
        # the first page turns "the cheap boxes are burned" into a false terminal failure.
        market = [
            o
            for o in usable_offers(
                info.vram_gb,
                _effective_disk_gb(spec),
                limit=_EXHAUSTION_RECHECK_LIMIT,
                max_wall_seconds=_rent_duration_floor(spec, absolute_deadline),
                gpu_type=spec.gpu.type,
                num_gpus=gpu_count_of(spec),
                **deadline_kwargs(usable_offers, absolute_deadline),
            )
            if o.gpu == spec.gpu.type
        ]
        offers = [o for o in market if o.machine_id not in excluded]
    if market and not offers:
        # the class HAD offers and this run had already lost every one of them. the generic empty-
        # pool error reads as "vast has no capacity" when the truth is "this run has burned all of
        # it" -- a different operator fix (different class or provider, not waiting).
        #
        # gated on `market`, not on `excluded` being non-empty: the blacklist is keyed by run, so it
        # can hold hosts from a GPU class this attempt already escalated away from, and a dry market
        # would otherwise be blamed on hosts that were never in it. counting the machines actually
        # removed here keeps the number honest for the same reason.
        # a dedicated type, not VastApiError: supervision withholds provider exception text from the
        # run record (it can quote a request that carried a credential), so the one error worth
        # reading would be reduced to its class name like any other. this message is authored here.
        raise RunExhaustedProviderPoolError(
            f"no usable vast offers for {spec.gpu.type} outside the "
            f"{len({o.machine_id for o in market})} machine(s) this run already rented and lost"
        )
    handle = None
    try:
        handle = deploy_and_submit(
            spec,
            offers,
            attempt=attempt,
            log=log,
            runtime_secrets=runtime_secrets,
            source_snapshot=source_snapshot,
            **deadline_kwargs(deploy_and_submit, absolute_deadline),
        )
        if on_handle is not None:
            on_handle(handle.to_dict())
        reader = heartbeat_reader_for(
            spec,
            **deadline_kwargs(heartbeat_reader_for, absolute_deadline),
        )
        result = poll_vast_job(
            handle,
            spec,
            log=log,
            heartbeat_reader=reader,
            **deadline_kwargs(poll_vast_job, absolute_deadline),
        )
        if not result.ok and _is_never_started_stall(result):
            # the machine took the rental and never produced a working worker. blacklist the HOST,
            # not the offer id: vast relists the same box under a fresh offer id, so an offer-keyed
            # entry would be stale before the next attempt searched.
            #
            # only the pre-boot load timeout qualifies (see `_NEVER_STARTED_MARKER`). a box that
            # booted and then stalled mid-training is not indicted: that failure would recur
            # anywhere, and retiring a working host for it starves the pool.
            _note_dead_machine(spec.run_id, getattr(handle, "machine_id", None))
        return result
    finally:
        if handle is not None:
            confirmed = False
            cleanup_exc: BaseException | None = None
            try:
                confirmed = _best_effort_destroy(
                    handle.instance_id,
                    context="submit_attempt_vast teardown",
                )
            except BaseException as exc:
                cleanup_exc = exc
            if not confirmed:
                with contextlib.suppress(BaseException):
                    destroy_run_instances(spec.run_id)
            if cleanup_exc is not None and sys.exc_info()[1] is None:
                raise cleanup_exc


def _best_effort_destroy(instance_id, *, context: str) -> bool:
    """Best-effort destroy for non-raising teardown paths.

    Warn and return False when billing may continue; ``VastProvider.destroy`` escalates separately.
    Pass ``instance_id`` through because conversion here could raise inside ``finally`` cleanup.
    """
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
    """Return one strict positive Vast instance identity, else ``None`` for cleanup skipping."""
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


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
    """Return instance ids still carrying the run label.

    Empty means confirmed clear. Strict listing raises on incomplete enumeration so handle-less
    recovery never launches over a possibly live worker still writing HF artifacts.
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
    """Destroy unclaimed Flash-labeled instances and return their ids.

    Resolve callable active labels after listing to close the launch race. ``known_labels`` limits
    cleanup to this plane; None reaps every inactive ``flash-`` label. Never raises.
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
        # resolving a protection set failed, so skip rather than treat live instances as orphans.
        logger.warning("vast orphan sweep skipped; could not resolve run sets: %s", exc)
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
