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
import math
import sys
import time
from collections.abc import Callable

from flash._logging import get_logger
from flash.diagnostics import sanitize_diagnostic
from flash.providers._deadline import (
    deadline_kwargs,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._hf_artifacts import (
    error_artifact_name,
    heartbeat_reader_for,
    make_hf_text_reader,
)
from flash.providers._instance_poll import InstancePollAdapter, poll_instance_job
from flash.providers._poll import (
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
    make_say,
)
from flash.providers.base import (
    GPU_INFO,
    PollResult,
    UnreconciledCreateError,
    UnsupportedGpuError,
    canonical_gpu,
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
# The shared instance-poll timing defaults imported from ``_poll`` above. The staged setup-vs-training
# grace is the fix for the historical "Vast box dies every ~25-30 min": the old provider used one flat
# 1500s window that fired mid cold-start and tore down healthy boxes. load_timeout_s is read at call
# time so ``monkeypatch.setattr(jobs, …)`` takes effect; setup_grace_s / stall_after_s /
# first_liveness_s are supplied as ``poll_vast_job`` defaults (override by passing the kwarg, not by
# patching the global).
# Boards under-report VRAM vs class nominal (L4 23034/24GB, A40 46068/48GB ≈ 0.938). The server-side
# gpu_ram filter gets this slack; the class gate (vast_gpu_for_offer) stays exact.
_SEARCH_VRAM_SLACK = 0.92
# Minimum disk every instance is provisioned with (bootstrap + worker + weights need headroom). The
# offer search MUST use the same floor so a thin-disk offer can't pass search then fail at create.
MIN_DISK_GB = 60.0

# Vast states meaning "the container is gone / won't progress". ``frozen`` is paused-but-still-billing
# yet emits no DONE/heartbeat, so classify it dead for fast failover. Unlike ``unknown`` it is never
# this poller's no-status fallback, so it needs no ``became_running`` gate.
_DEAD_STATES = {"exited", "stopped", "offline", "deleted", "frozen"}


def _effective_disk_gb(spec) -> float:
    """The disk size an instance is actually provisioned with (the create-time floor).

    Both the offer search and ``create_instance`` must agree on this, or offers with a disk between
    ``spec.gpu.disk_gb`` and the floor pass the search then fail to rent.
    """
    return max(float(spec.gpu.disk_gb), MIN_DISK_GB)


def _exact_search_aliases(info) -> tuple[str, ...]:
    """Vast alias spellings safe to seed an EXACT-class offer search with.

    An exact pin is attested on the box against the live device name (``verify_gpu`` -> ``canonical_gpu``).
    Keep only alias spellings that canonicalize back to THIS class, so the search never pulls a board that
    attestation would then reject: "A100 PCIE" (kept as fungible A100 SXM 40GB capacity for NON-exact runs
    by ``vast_gpu_for_offer``) names a PCIe board that canonicalizes to the distinct "A100 PCIe" class, so
    it would be rented then fail an exact A100-SXM-40GB attestation; H100's "H100 PCIE" canonicalizes to
    "H100" and stays fungible. Ambiguous/unknown spellings (``canonical_gpu`` raises) are dropped.
    """
    kept: list[str] = []
    for alias in info.vast_aliases:
        try:
            if canonical_gpu(alias) == info.name:
                kept.append(alias)
        except UnsupportedGpuError:
            pass
    return tuple(kept)


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    limit: int = 256,
    max_wall_seconds: float = 0,
    exact_type: str = "",
    deadline_at: float | None = None,
) -> list[VastOffer]:
    """Verified-datacenter offers able to run the job, cheapest first.

    Server-side filters do the heavy lifting; everything load-bearing is re-checked client-side.

    ``limit`` is the price-sorted page size. Callers bucket rows BY GPU CLASS, so the page must span
    every fitting managed class — 256 comfortably covers the verified-datacenter market (a specific-class
    caller just filters down further). ``max_wall_seconds`` (>0) also requires offers available for at
    least ``max(60, max_wall)`` so the search never advertises capacity an offer cannot outlast. Provider
    provisioning grace is not added to the run's terminal cutoff. 0 = no duration floor.
    """
    min_duration = (
        max(60.0, float(max_wall_seconds)) if max_wall_seconds and max_wall_seconds > 0 else 0
    )
    exact_info = GPU_INFO.get(exact_type) if exact_type else None
    if exact_type and exact_info is None:
        raise ValueError(f"unknown exact Vast GPU class {exact_type!r}")
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
    rows = vast_api.search_offers(
        int(search_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        min_duration_seconds=min_duration,
        limit=int(limit),
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
            or (exact_type and gpu != exact_type)
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


def _adopt_instance_by_label(
    label: str,
    *,
    deadline_at: float | None = None,
) -> dict | None:
    """Best-effort instance DICT carrying this EXACT (per run/seed/attempt) label, or ``None``. Reclaims
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
            hourly_usd=offer.dph_total,
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
    seed: int,
    offers: list[VastOffer],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
    code_prefix: str | None = None,
    deadline_at: float | None = None,
) -> VastJobHandle:
    """Rent the cheapest offer that will actually take the job; walk on rejection.

    Offers are a live market — between search and rent the cheapest one is often gone. We walk up to
    5 ranked offers, then refresh the search once (re-excluding the machines we just tried so a fresh
    market re-search doesn't re-select one that just rejected us).
    """
    say = make_say(log)
    absolute_deadline = require_deadline_at(deadline_at)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(
        spec,
        seed,
        attempt,
        runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
        **deadline_kwargs(build_payload, absolute_deadline),
    )
    label = instance_label(spec.run_id, seed, attempt)
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
                            exclude_machine_ids={o.machine_id for o in tried},
                            max_wall_seconds=float(getattr(spec.gpu, "max_wall_seconds", 0) or 0),
                            # Mirror the initial submit search: narrow to exact aliases ONLY on the user's
                            # hard pin. Inferring exact_type from ``allowed`` forced an exact refresh for a
                            # non-exact run (its offers are pre-filtered to one canonical class), dropping the
                            # fungible cross-architecture capacity the first broad search had matched.
                            exact_type=spec.gpu.exact_type,
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
        parts.append(
            f"--- {err_name} ---\n{sanitize_diagnostic(content[-4096:], limit=4096)}"
        )
    logs = vast_api.instance_logs(int(instance_id))
    if logs:
        parts.append(
            "--- instance log tail ---\n"
            f"{sanitize_diagnostic(logs[-4096:], limit=4096)}"
        )
    return "\n".join(parts) or "vast worker terminated without a strict terminal marker"


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
    deadline_at: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (cf. lambdalabs.jobs.poll_lambda_job).

    A thin wrapper that builds the Vast :class:`InstancePollAdapter` and defers to the shared
    ``poll_instance_job`` kernel — Vast IS that kernel's baseline, so its behavior is unchanged. Vast
    stamps the customer cost from the worker TRAINING wall (metrics.wall_seconds), notes the provider
    offer + instance wall, reads early-liveness from the container-log API, and — having no host console
    file — assembles failure detail from HF artifacts + the live instance log tail. ``LOAD_TIMEOUT_S`` is
    read here (a module global) so ``monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", ...)`` still bites.
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
            f"instance stuck in '{status}' for {int(elapsed)}s (never started; image pull / host issue)"
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


def submit_run_vast(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
    code_prefix: str | None = None,
    deadline_at: float | None = None,
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
    absolute_deadline = require_deadline_at(deadline_at)
    info = GPU_INFO[spec.gpu.type]
    offers = [
        o
        for o in usable_offers(
            info.vram_gb,
            _effective_disk_gb(spec),
            max_wall_seconds=float(getattr(spec.gpu, "max_wall_seconds", 0) or 0),
            # Narrow to attestation-safe exact aliases ONLY when the user hard-pinned the class.
            # A non-exact run (exact_type == "") keeps the broad fungible search so cross-architecture
            # capacity (e.g. 40GB "A100 PCIE" as A100 SXM 40GB) still counts, matching soft verify_gpu.
            exact_type=spec.gpu.exact_type,
            **deadline_kwargs(usable_offers, absolute_deadline),
        )
        if o.gpu == spec.gpu.type
    ]
    handle = None
    try:
        handle = deploy_and_submit(
            spec,
            seed,
            offers,
            attempt=attempt,
            log=log,
            runtime_secrets=runtime_secrets,
            code_prefix=code_prefix,
            **deadline_kwargs(deploy_and_submit, absolute_deadline),
        )
        if on_handle is not None:
            on_handle(handle.to_dict())
        reader = heartbeat_reader_for(
            spec,
            **deadline_kwargs(heartbeat_reader_for, absolute_deadline),
        )
        return poll_vast_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            **deadline_kwargs(poll_vast_job, absolute_deadline),
        )
    finally:
        if handle is not None:
            confirmed = False
            cleanup_exc: BaseException | None = None
            try:
                confirmed = _best_effort_destroy(
                    handle.instance_id,
                    context="submit_run_vast teardown",
                )
            except BaseException as exc:
                cleanup_exc = exc
            if not confirmed:
                with contextlib.suppress(BaseException):
                    destroy_run_instances(spec.run_id)
            if cleanup_exc is not None and sys.exc_info()[1] is None:
                raise cleanup_exc


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
        if (
            iid
            and label_matches_run(label, prefix)
            and vast_api.destroy_instance(iid)
        ):
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
