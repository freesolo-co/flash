"""Vast.ai run lifecycle: verified offers -> instance -> HF-artifact poll.

The Vast equivalent of ``providers/runpod/jobs.py``. Vast has no serverless queue:
we rent a single-GPU instance from a VERIFIED datacenter offer, ship a self-contained
bootstrap (the private ``_bootstrap`` module) through the onstart script, and detect
completion purely via the worker's HF artifacts (DONE/metrics.json/heartbeat.json) +
the instance's status — no inbound network to the box is ever needed.

The instance bootstrap is an INTERNAL detail of this module (``build_onstart`` reads
``_bootstrap.py``), so the public per-provider module set stays identical to RunPod's.

Cost-safety invariant: a rented instance is ALWAYS destroyed — the runner's
``finally``, the onstart's self-destroy backstop, the cancel path, and
``sweep_orphans`` (server startup / post-run) each independently guarantee it.

This is a package: the monkeypatch-free dataclasses + pure builders (``VastOffer``,
``VastJobHandle``, ``build_payload``, ``build_onstart``, ``vast_image``,
``instance_label``, ``run_label_prefix``) live in ``.builders`` and are re-exported
here so the import path ``flash.providers.vast.jobs`` is unchanged. The lifecycle
functions and the operator-visible constants tests monkeypatch (``usable_offers``,
``deploy_and_submit``, ``poll_vast_job``, ``submit_run_vast``, ``_make_hf_file_reader``,
``time``, ``LOAD_TIMEOUT_S``/``RELIABILITY_FLOOR``/``MIN_INET_MBPS``/``MIN_DISK_GB``)
stay in this ``__init__`` so a ``monkeypatch.setattr(jobs, …)`` still takes effect.
"""

from __future__ import annotations

import contextlib
import json
import time

from flash._logging import get_logger
from flash.providers._poll import PollErrorTracker, make_say, surface_heartbeat
from flash.providers.base import GPU_INFO, PollResult, min_cuda_modern, vast_gpu_for_offer
from flash.providers.runpod.jobs import make_hf_heartbeat_reader, make_hf_text_reader
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

# Offer-quality floors (beyond verified+datacenter, which are non-negotiable). reliability2 is
# Vast's host-uptime/health score: 0.95 let ~1-in-20 long runs die mid-train ("worker terminated
# without a DONE sentinel" when the host went down); 0.995 (~1-in-200) keeps supply usable (the
# >=0.995 datacenter pool measured 67 offers) while nearly eliminating mid-run host deaths. These
# are fixed correctness floors, not operator-tunable.
RELIABILITY_FLOOR = 0.995
MIN_INET_MBPS = 200.0
# How long an instance may sit in a non-running state (image pull) before we give up.
LOAD_TIMEOUT_S = 900.0
# Boards under-report VRAM vs the class nominal (measured live: L4 23034 MB / 24 GB,
# A40 46068 MB / 48 GB = 0.938 of nominal); the server-side gpu_ram filter gets this
# slack, the class gate stays exact (vast_gpu_for_offer).
_SEARCH_VRAM_SLACK = 0.92

# Minimum disk Vast instances are provisioned with (the bootstrap + worker stack +
# weights need headroom regardless of the spec's request). The offer search MUST use
# this same floor so offers with <60 GB disk don't pass the search and then get
# rejected at create time (``create_instance`` enforces the same max).
MIN_DISK_GB = 60.0


def _effective_disk_gb(spec) -> float:
    """The disk size an instance is actually provisioned with (the create-time floor).

    Both the offer search and ``create_instance`` must agree on this, or offers with a
    disk between ``spec.gpu.disk_gb`` and the floor pass the search then fail to rent.
    """
    return max(float(spec.gpu.disk_gb), MIN_DISK_GB)


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    num_gpus: int = 1,
) -> list[VastOffer]:
    """Verified datacenter offers able to run the job, cheapest first.

    ``num_gpus`` (default 1) rents an instance with exactly that many GPUs — the disaggregated
    async GRPO path needs a multi-GPU node ([gpu] count). ``min_vram_gb`` is the PER-GPU floor
    (each card holds a trainer or rollout shard), unchanged by the count.

    Server-side filters do the heavy lifting; everything load-bearing is re-checked
    client-side (belt and suspenders — the result rows carry the proof fields).
    """
    rows = vast_api.search_offers(
        int(min_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
        num_gpus=num_gpus,
    )
    # Host tier: ALWAYS datacenter-only (hosting_type==1). The onstart payload ships run secrets
    # (HF_TOKEN, PRIME_API_KEY, OpenRouter/OpenAI/W&B creds) to the box, so verified-but-lower-trust
    # community/marketplace hosts (hosting_type 0) are never used — even verified + reliability-
    # floored, they're a lower trust tier we don't ship secrets to.
    out: list[VastOffer] = []
    for r in rows:
        gpu = vast_gpu_for_offer(str(r.get("gpu_name") or ""), float(r.get("gpu_ram") or 0))
        if gpu is None:  # not a managed class (Ampere+ floor)
            continue
        info = GPU_INFO[gpu]
        dph = float(r.get("dph_total") or 0)
        cuda = float(r.get("cuda_max_good") or 0)
        # Host tier: accept ONLY verified datacenter hosts (hosting_type==1); community/marketplace
        # (hosting_type==0) and anything else is rejected (we ship run secrets to the box).
        _bad_host = r.get("hosting_type") != 1
        # MULTI-GPU: require a WHOLE-machine offer with an EXPLICIT gpu_frac ~= 1.0. A fractional
        # multi-GPU offer (e.g. 2 of an 8-GPU box, gpu_frac=0.25/0.5) rents 2 GPUs to the INSTANCE
        # but the CONTAINER gets only 1 (Vast sets NVIDIA_VISIBLE_DEVICES=void and mounts the
        # fraction's devices — verified: nvidia-smi -L=1, env void). A MISSING gpu_frac is treated
        # as NOT-whole-machine here (a $0.54 missing-frac offer also yielded a 1-GPU container), so
        # require the field present and >=0.99 — whole-machine offers carry gpu_frac=1.0 explicitly.
        # float() guarded: a present-but-non-numeric gpu_frac (e.g. "" / null-ish string) would
        # raise ValueError and abort the WHOLE offer search; treat unparseable as not-whole-machine
        # (rejected), consistent with requiring an explicit numeric >=0.99.
        try:
            _gf_val = float(r.get("gpu_frac"))
        except (TypeError, ValueError):
            _gf_val = 0.0
        _frac_ok = num_gpus <= 1 or _gf_val >= 0.99
        if (
            _bad_host
            or not _frac_ok
            or r.get("verification") != "verified"
            # Exact class gate: guard against a board whose canonical class nominal VRAM
            # is below the request (e.g. asking for 48 GB but the mapping landed on a
            # 24 GB class) — the server-side gpu_ram filter only carries slack.
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
                num_gpus=int(r.get("num_gpus") or num_gpus),
            )
        )
    return sorted(out, key=lambda o: (o.dph_total, o.vram_gb))


def deploy_and_submit(
    spec,
    seed: int,
    offers: list[VastOffer],
    attempt: int = 0,
    log=None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> VastJobHandle:
    """Rent the cheapest offer that will actually take the job; walk on rejection.

    Offers are a live market — between search and rent the cheapest one is often
    gone. We walk up to 5 ranked offers, then refresh the search once.

    ``exclude_machine_ids`` is the run's blacklist (machines that stalled/failed this
    run earlier). The refresh re-search MUST keep them excluded — otherwise a sick
    machine the orchestrator just blacklisted gets re-selected from the fresh market.
    """

    say = make_say(log)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(spec, seed, attempt)
    label = instance_label(spec.run_id, seed, attempt)
    from flash.providers.runpod.train import WORKER_IMAGE

    install_deps = not WORKER_IMAGE
    tried: list[VastOffer] = []
    candidates = list(offers[:5])
    refreshed = False
    last_err: Exception | None = None
    while candidates:
        offer = candidates.pop(0)
        tried.append(offer)
        onstart = build_onstart(payload, install_deps=install_deps)
        try:
            instance_id = vast_api.create_instance(
                offer.offer_id,
                image=vast_image(),
                disk_gb=_effective_disk_gb(spec),
                # Expose ALL rented GPUs to the container. The worker image is pytorch/pytorch
                # (not nvidia/cuda) and the NVIDIA container runtime only surfaces the GPUs named
                # in NVIDIA_VISIBLE_DEVICES — without this a multi-GPU offer's container can come up
                # with just 1 GPU (observed: a num_gpus=2 5090 offer exposed 1 GPU, breaking the
                # disaggregated split). "all" = all GPUs in the container's cgroup, i.e. exactly the
                # rented ones (not other tenants'); harmless for single-GPU runs.
                env={"NVIDIA_VISIBLE_DEVICES": "all"},
                onstart=onstart,
                label=label,
                runtype="args",
            )
        except vast_api.VastApiError as e:
            last_err = e
            say(f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) rejected: {e}")
            if not candidates and not refreshed:
                refreshed = True
                # Exclude both the machines we just tried this attempt AND the run's
                # standing blacklist (machines that stalled/failed earlier attempts) —
                # otherwise the fresh search can re-select a sick machine the
                # orchestrator deliberately excluded.
                taken = {o.machine_id for o in tried} | set(exclude_machine_ids)
                # Stay within the allocator-approved class pool: the original `offers`
                # are already filtered to the allocated/pinned + validated classes, so
                # the refresh must not widen to any usable offer (which could rent a
                # different or unvalidated GPU than the run spec assumes).
                allowed = {o.gpu for o in offers}
                candidates = [
                    o
                    for o in usable_offers(
                        min(o.vram_gb for o in offers),
                        _effective_disk_gb(spec),
                        exclude_machine_ids=taken,
                        num_gpus=int(getattr(spec.gpu, "count", 1)),
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


# Rate-limited reader for one HF artifact's text content (None until it exists). Shared
# with runpod's poller via make_hf_text_reader; kept under this module-local name because
# tests monkeypatch ``vast.jobs._make_hf_file_reader`` and the poll/failure paths below
# resolve it as a module global (so a monkeypatch still takes effect).
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(
    hf_repo: str, prefix: str, phase: str, marker: dict | None, instance_id: int | None = None
) -> str:
    """Best root-cause detail we can assemble from the HF artifacts."""
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    content = _make_hf_file_reader(hf_repo, f"{prefix}/error_{phase}.txt")(force=True)
    if content:
        parts.append(f"--- error_{phase}.txt ---\n{content[-2000:]}")
    if instance_id:
        # Early-bootstrap failures (pip/env errors before the worker can reach HF)
        # only ever appear on the container console.
        logs = vast_api.instance_logs(int(instance_id))
        if logs:
            parts.append(f"--- instance log tail ---\n{logs[-3000:]}")
    return "\n".join(parts) or "vast worker terminated without a DONE sentinel"


# Vast instance states that mean "the container is gone / will not progress".
_DEAD_STATES = {"exited", "stopped", "offline", "deleted"}


def poll_vast_job(
    handle: VastJobHandle,
    spec,
    seed: int,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    stall_after_s: float = 1500.0,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (cf. jobs.poll_job).

    COMPLETED  fresh DONE sentinel on HF -> metrics.json (cost stamped from the
               offer's real $/hr).
    FAILED     attempt marker with ok=false, or instance dead without DONE.
    STALLED    never left loading within LOAD_TIMEOUT_S, heartbeat frozen for
               stall_after_s, or the client-side deadline passed.
    """

    say = make_say(log)

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
        # Prefer the worker's DONE timestamp when present and sane; fall back to now.
        # On delayed recovery the control plane may call this hours after the instance
        # wrote DONE and self-destroyed, so billing to now would over-bill by the
        # downtime — accepted because a missing/garbled DONE timestamp is rare. DONE
        # carries the worker's time.time().
        end_ts = time.time()
        if done_content:
            try:
                done_ts = float(done_content.strip())
                if handle.started_ts <= done_ts <= end_ts:
                    end_ts = done_ts
            except ValueError:
                # Malformed DONE timestamp: keep end_ts = now rather than trusting garbage.
                pass
        wall_h = (end_ts - handle.started_ts) / 3600.0
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
        # DONE carries the worker's time.time(); 120 s of clock-skew grace. Anything
        # older predates this attempt (leftover from a prior attempt's resume).
        try:
            return float(content.strip()) > handle.started_ts - 120.0
        except ValueError:
            return False

    poll_errors = PollErrorTracker(say, interval_s)

    start = time.time()
    last_status = None
    last_hb_key = None
    last_progress = time.time()
    became_running = False
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            inst = vast_api.get_instance(handle.instance_id)
            poll_errors.reset()
        except vast_api.VastApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        # Verified live: the instance-detail route TRANSIENTLY answers
        # {"instances": null} for perfectly healthy instances (and for brand-new ones
        # before they materialize). A single missing read means nothing — only a
        # sustained streak is a real disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get("actual_status") or ("missing" if inst is None else "unknown")
        if status != last_status:
            say(f"instance {handle.instance_id}: {status}")
            last_status = status
            last_progress = time.time()
        if status == "running":
            became_running = True

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        if missing_streak >= 4 or status in _DEAD_STATES:
            # One forced final read: the worker may have finished right before the
            # instance self-destroyed (the normal success order on this substrate).
            done = done_reader(force=True)
            if done is not None and done_is_fresh(done):
                return finish_ok(done)
            raw_marker = marker_reader(force=True)
            marker = None
            if raw_marker:
                with contextlib.suppress(ValueError):
                    marker = json.loads(raw_marker)
            return PollResult(
                False,
                failure="job_failed",
                detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.instance_id),
            )

        raw_marker = marker_reader()
        if raw_marker:
            try:
                marker = json.loads(raw_marker)
            except ValueError:
                marker = None
            if marker and not marker.get("ok"):
                return PollResult(
                    False,
                    failure="job_failed",
                    detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.instance_id),
                )
            if marker and marker.get("ok"):
                done = done_reader(force=True)
                if done is not None and done_is_fresh(done):
                    return finish_ok(done)

        if not became_running and time.time() - start > LOAD_TIMEOUT_S:
            return PollResult(
                False,
                failure="stalled",
                detail=f"instance stuck in '{status}' for {int(time.time() - start)}s "
                f"(image pull / host issue)",
            )

        new_key, _stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            last_progress = time.time()
        if became_running and time.time() - last_progress > stall_after_s:
            return PollResult(
                False,
                failure="stalled",
                detail=f"no worker progress for {int(time.time() - last_progress)}s "
                f"(instance status {status})",
            )
        time.sleep(interval_s)


def submit_run_vast(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    offers: list[VastOffer] | None = None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> PollResult:
    """Vast equivalent of ``runpod.jobs.submit_run``: rent, persist, poll.

    The ``finally`` destroy is the cost-safety primary: every exit path — success,
    failure, stall, exception, KeyboardInterrupt — tears the paid instance down.
    """
    if offers is None:
        # GPU_INFO is keyed by concrete GPU class; a policy word ("cheapest"/"auto") would
        # KeyError opaquely here. The allocator resolves policy words to a concrete class
        # upstream, so reaching this fallback with one is a caller bug — name it clearly.
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
                exclude_machine_ids=exclude_machine_ids,
                num_gpus=int(getattr(spec.gpu, "count", 1)),
            )
            if o.gpu == spec.gpu.type
        ]
    handle = deploy_and_submit(
        spec, seed, offers, attempt=attempt, log=log, exclude_machine_ids=exclude_machine_ids
    )
    # The instance is rented and billing the MOMENT deploy_and_submit returns; the
    # teardown ``finally`` must guard EVERYTHING after that point — including
    # ``on_handle`` (persisting the remote handle can itself raise). Entering the try
    # before on_handle guarantees the paid instance is destroyed even if the handle is
    # never persisted, closing the rent->persist crash window's billing leak.
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        stall = 1500.0
        # Wall cap + provision/install grace; Vast has no server-side execution
        # timeout, so the client deadline (and the bootstrap's own cap) bound spend.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + 1800
        return poll_vast_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            stall_after_s=stall,
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
    """Destroy every instance belonging to ONE run (labels start with its run id).

    Cancel/GC path: unlike ``sweep_orphans`` this never looks at other runs, so it
    is safe to call while they are in flight. Best-effort: never raises.
    """
    destroyed: list[int] = []
    if not run_id:
        return destroyed
    try:
        instances = vast_api.list_instances()
    except Exception:
        return destroyed
    prefixes = (run_id, f"flash-{run_id}")  # instance_label may force the prefix
    for inst in instances:
        iid = inst.get("id")
        label = str(inst.get("label") or "")
        # Match on the label boundary, not a raw string prefix (see ``sweep_orphans``):
        # an instance label is ``f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"``, so a
        # run's prefix must equal the label or be followed by the ``-s`` seed boundary.
        # A bare ``startswith`` would let run ``100`` also destroy run ``1000``'s instances.
        if (
            iid
            and any(label == p or label.startswith(p + "-s") for p in prefixes)
            and vast_api.destroy_instance(int(iid))
        ):
            destroyed.append(int(iid))
    return destroyed


def sweep_orphans(active_labels: set[str] | None = None) -> list[int]:
    """Destroy Flash-labeled instances that no live run owns; return destroyed ids.

    Run at server startup (crash recovery) and after runs (belt and suspenders).
    Only labels carrying the run-id prefix are ever touched — nothing else on the
    account is ours to destroy. Best-effort: never raises.

    ``active_labels`` may be RAW run ids (what the server tracks) — each is passed
    through ``run_label_prefix`` so it matches the SAME forced-``flash-`` prefix the
    instance labels carry. Passing an already-prefixed label is fine (idempotent), so a
    live run whose id lacks the prefix is still correctly protected.
    """
    destroyed: list[int] = []
    try:
        instances = vast_api.list_instances()
    except Exception as exc:
        logger.warning("vast orphan sweep skipped: %s", exc)
        return destroyed
    active = {run_label_prefix(a) for a in (active_labels or set())}
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label.startswith("flash-"):
            continue
        # Match on the label boundary, not a raw string prefix: an instance label is
        # ``f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"`` (see ``instance_label``),
        # so a live run's prefix must equal the label or be followed by the ``-s`` seed
        # boundary. A bare ``startswith`` would let one run's prefix (e.g. ``flash-100``)
        # shield another run's orphan (``flash-1000-...``) from the sweep.
        if any(label == a or label.startswith(a + "-s") for a in active):
            continue
        iid = inst.get("id")
        if iid and vast_api.destroy_instance(int(iid)):
            destroyed.append(int(iid))
            logger.warning("destroyed orphaned vast instance %s (label %s)", iid, label)
    return destroyed
