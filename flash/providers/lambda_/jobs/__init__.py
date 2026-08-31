"""Lambda Cloud run lifecycle: capacity walk -> launch -> HF-artifact poll -> guaranteed terminate.

Cost-safety: a launched instance is ALWAYS terminated — runner finally, poll deadline, cancel, and
sweep_orphans each independently guarantee it. No in-box self-destruct (unlike Vast); sweep_orphans
at startup is the crash backstop.

Constants tests monkeypatch stay in this __init__ so monkeypatch.setattr(jobs, …) still takes effect.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.poll import (
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
    make_say,
    preload_box_reap_due,
)
from flash.providers._lifecycle.instances.poll_instance import (
    InstancePollAdapter,
    poll_instance_job,
)
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
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
    UnreconciledCreateError,
    UnsupportedGpuError,
)
from flash.providers.lambda_.client import api as lambda_api
from flash.providers.lambda_.jobs.builders import (
    LambdaInstance,
    LambdaJobHandle,
    build_payload,
    build_user_data,
    instance_label,
    label_matches_run,
    run_label_prefix,
)
from flash.providers.lambda_.jobs.reap import (
    _CoarseReapGuard,
    _exact_cleanup_taken,
    _launch_failed_before_the_request,
    _mark_exact_cleanup,
)

logger = get_logger(__name__)

# The shared instance-poll timing defaults imported from ``_poll`` above (setup grace covers Docker pull
# + pip + model download, before any heartbeat). load_timeout_s is read at call time so
# ``monkeypatch.setattr(jobs, …)`` takes effect; setup_grace_s / stall_after_s / first_liveness_s
# are supplied as ``poll_lambda_job`` defaults (override by passing the kwarg, not by patching the global).

_DEAD_STATES = {"terminated", "terminating", "preempted", "unhealthy"}


def resolve_ssh_key_names(*, deadline_at: float | None = None) -> list[str]:
    """Return the SSH key to attach at launch (required by Lambda even though we never SSH in)."""
    keys = lambda_api.list_ssh_keys(
        **deadline_kwargs(lambda_api.list_ssh_keys, deadline_at),
    )
    names = [k.get("name") for k in keys if k.get("name")]
    if not names:
        raise lambda_api.LambdaApiError(
            "Lambda launch requires an SSH key on the account, but none are registered; add one "
            "in the Lambda console (the box is bootstrapped via user_data, so the key is unused "
            "— any key works)."
        )
    return [names[0]]


def usable_instances(
    gpu_class: str,
    force: bool = False,
    *,
    gpu_count: int = 1,
    deadline_at: float | None = None,
) -> list[LambdaInstance]:
    """Regions currently advertising capacity for the given GPU class. Empty = no Lambda capacity now.

    ``gpu_count`` selects the N-card instance type for the class (Lambda names the count in the
    type). A count Lambda does not sell for this class has no catalog entry and therefore no
    capacity, so it drops out here exactly like a sold-out region — the caller never gets an
    unrentable shape back.
    """
    from flash.providers.lambda_.client.gpus import instance_type_disk_gb, instance_type_for
    from flash.providers.lambda_.client.pricing import hourly_rate

    info = GPU_INFO[gpu_class]
    count = max(1, int(gpu_count))
    try:
        # resolve the count against Lambda's own catalog: a multi-card SKU can carry a different
        # suffix than its 1x entry (gpu_1x_h100_pcie vs gpu_8x_h100_sxm5), and a name derived only
        # from the 1x spelling would miss the real type and hide available capacity.
        catalog = lambda_api.list_instance_types(
            force=force,
            **deadline_kwargs(lambda_api.list_instance_types, deadline_at),
        )
    except Exception:
        catalog = None
    try:
        itype = instance_type_for(gpu_class, count, catalog)
    except UnsupportedGpuError:
        return []
    # price_cents_per_hour is per INSTANCE (all N cards); Candidate.hourly_usd is contractually
    # per-card, so divide. Without this an N-card box prices N^2 and the allocator never picks it.
    rate = (
        hourly_rate(
            gpu_class,
            gpu_count=count,
            **deadline_kwargs(hourly_rate, deadline_at),
        )
        / count
    )
    regions = lambda_api.regions_with_capacity(
        itype,
        force=force,
        **deadline_kwargs(lambda_api.regions_with_capacity, deadline_at),
    )
    if catalog is None and regions:
        # the capacity call fetches the same catalog, so a first fetch that exhausted its retries
        # is recovered by the one that just succeeded. Reading it here rather than keeping None
        # keeps a transient blip from reporting an UNMEASURED disk, which the floor below treats as
        # permissive and would let it rent a SKU whose fixed storage is provably too small.
        with contextlib.suppress(Exception):
            catalog = lambda_api.list_instance_types(
                **deadline_kwargs(lambda_api.list_instance_types, deadline_at),
            )
    # carried on the candidate so the launch gate can refuse an undersized SKU without a second
    # catalog fetch; None (unreported storage) stays permissive.
    disk_gb = instance_type_disk_gb(catalog, itype)
    return [
        LambdaInstance(
            gpu=gpu_class,
            instance_type=itype,
            region=region,
            vram_gb=info.vram_gb,
            price_usd_hr=rate,
            gpu_count=count,
            disk_gb=disk_gb,
        )
        for region in regions
    ]


def _launch_rejection_is_clean(err: Exception) -> bool:
    """True when the launch was definitively rejected with NO instance created (safe to try next region).

    A 429, 5xx, timeout, or missing-id response is AMBIGUOUS — the provider may have billed an
    instance we can't see, so we must NOT issue another launch.
    """
    s = str(err)
    return "-> HTTP 4" in s and "HTTP 429" not in s


def _abort_ambiguous_launch(run_id: str, detail: str) -> None:
    """Attempt exact cleanup, then fail closed without issuing another launch."""
    cleanup = "cleanup unconfirmed"
    try:
        instances = lambda_api.list_instances()
        prefix = run_label_prefix(run_id)
        ids = [
            str(instance["id"])
            for instance in instances
            if instance.get("id") and label_matches_run(str(instance.get("name") or ""), prefix)
        ]
        if ids:
            for instance_id in ids:
                lambda_api.terminate_instance_confirmed(instance_id)
            cleanup = f"terminated {len(ids)} matching instance(s)"
        else:
            cleanup = "no matching instance was observable"
    except Exception:
        pass
    raise UnreconciledCreateError(
        f"ambiguous Lambda launch; refusing another create because cleanup is not authoritative "
        f"({cleanup}): {detail}"
    )


def _build_launch_user_data(
    spec,
    attempt: int,
    runtime_secrets: dict | None,
    source_snapshot: dict | None,
    absolute_deadline: float,
    *,
    cache_host_mount: str | None = None,
    mode: str | None = None,
    models: list | None = None,
) -> str:
    """Build user_data with this region's actual NFS mount point."""
    payload_kwargs = {
        "runtime_secrets": runtime_secrets,
        "source_snapshot": source_snapshot,
        "mode": mode,
        "models": models,
        **deadline_kwargs(build_payload, absolute_deadline),
    }
    if cache_host_mount is not None:
        payload_kwargs["cache_host_mount"] = cache_host_mount
    return build_user_data(
        build_payload(spec, attempt, **payload_kwargs),
        gpu=spec.gpu.type,
    )


def _cleanup_unpublished_instance(run_id: str, instance_id: str, *, context: str) -> None:
    """Clean an exact unpublished instance, falling back to its run label only when unconfirmed.

    The window between a successful launch and the returned handle owns a rented box that nothing
    else can see yet, so cleanup here must survive a ``BaseException`` on either step.
    """
    confirmed = False
    with contextlib.suppress(BaseException):
        lambda_api.terminate_instance_confirmed(instance_id)
        confirmed = True
    if not confirmed:
        with contextlib.suppress(BaseException):
            logger.warning(
                "lambda teardown unconfirmed for instance %s (%s); the box may still be billing, "
                "falling back to a run-label reap",
                instance_id,
                context,
            )
        with contextlib.suppress(BaseException):
            terminate_run_instances(run_id)


def _disk_capable_instances(spec, instances: list[LambdaInstance], say) -> list[LambdaInstance]:
    """Drop every Lambda shape whose FIXED disk cannot satisfy the run's ``spec.gpu.disk_gb``.

    Vast sizes the volume at create (``_effective_disk_gb``) and RunPod raises
    ``containerDiskInGb``; Lambda sells storage with the instance type and takes no disk parameter,
    so the only way to honour the same contract is to decline the SKU before renting it. A
    candidate whose catalog entry reported no storage carries ``disk_gb=None`` and is left alone:
    the gate refuses only what it can prove undersized, never what it merely cannot measure.

    Returns the survivors rather than merely asserting one exists. The launch loop pops candidates
    one at a time, so a mixed list that contains a single capable SKU would otherwise still be free
    to rent an undersized one first; filtering makes the floor bind on the box actually launched,
    and applies identically to a refreshed candidate list whose catalog disk only became known on
    the refresh.
    """
    required = float(getattr(spec.gpu, "disk_gb", 0) or 0)
    if required <= 0:
        return list(instances)
    capable: list[LambdaInstance] = []
    undersized: dict[str, float] = {}
    for inst in instances:
        if inst.disk_gb is None or inst.disk_gb >= required:
            capable.append(inst)
            continue
        undersized[inst.instance_type] = inst.disk_gb
    if capable:
        if undersized:
            dropped = ", ".join(f"{i} ({d:g} GB)" for i, d in sorted(undersized.items()))
            say(f"skipping lambda shapes below the run's {required:g} GB disk floor: {dropped}")
        return capable
    shapes = ", ".join(f"{itype} ({disk:g} GB)" for itype, disk in sorted(undersized.items()))
    say(f"refusing lambda launch: {shapes} cannot hold the run's {required:g} GB disk floor")
    raise UnsupportedGpuError(
        f"lambda ships a fixed disk per instance type and has no launch-time disk parameter: "
        f"{shapes} is below the run's required {required:g} GB. Run {spec.gpu.type} on a provider "
        f"that sizes disk at create (vast, runpod) or lower the run's disk floor."
    )


def _lambda_job_handle(instance_id: str, inst: LambdaInstance, name: str, attempt: int):
    return LambdaJobHandle(
        instance_id=instance_id,
        instance_type=inst.instance_type,
        region=inst.region,
        name=name,
        gpu=inst.gpu,
        # ``price_usd_hr`` is PER CARD (the allocator's contract); the handle's rate is billed
        # against wall-clock once by both the cost stamp and realized COGS, so it must price the
        # WHOLE instance or an n-card box under-reports by exactly n.
        hourly_usd=inst.price_usd_hr * inst.gpu_count,
        attempt=attempt,
        started_ts=time.time(),
    )


def _rent_instance(
    plan: _LaunchPlan,
    inst: LambdaInstance,
    say,
    reap,
    *,
    user_data: str,
    file_system_names: list[str] | None,
    describe,
) -> LambdaJobHandle:
    """Rent one box and return its handle, owing nothing rented behind on any exit.

    The whole rent-to-handle window lives here because its ORDER is the correctness argument, and
    every region in the launch walk needs exactly the same order:

    - ``arm`` immediately before the request and only then. A deadline miss in the precheck rents
      nothing, and an armed guard there would reap by run label, killing every other concurrent
      attempt of this run.
    - ``owns`` as the FIRST statement after the create returns. From there the box is rented and
      named, so anything that can raise -- interpolating the message, building the handle -- must
      find the guard already holding the exact id rather than the run label.
    - the guard stays armed through the return: disarming first would leave an interrupt in the
      gap with nothing able to name the box, and the caller's teardown does not exist until it
      holds the handle.

    ``describe`` builds the log line from the id; it is a callable so the message is not
    interpolated before the guard owns the box.
    """
    require_create_allowance(plan.absolute_deadline)
    reap.arm()
    instance_id = lambda_api.launch_instance(
        region_name=inst.region,
        instance_type_name=inst.instance_type,
        ssh_key_names=plan.ssh_keys,
        name=plan.name,
        user_data=user_data,
        file_system_names=file_system_names,
        **deadline_kwargs(lambda_api.launch_instance, plan.absolute_deadline),
    )
    reap.owns(instance_id)
    try:
        # a raising log stream must not stop the handle from being returned; any BaseException
        # (interrupt, SystemExit) tears the still-unpublished box down first.
        with contextlib.suppress(Exception):
            say(describe(instance_id))
        return _lambda_job_handle(instance_id, inst, plan.name, plan.attempt)
    except BaseException as error:
        _cleanup_unpublished_instance(
            plan.spec.run_id, instance_id, context="post-launch handle acquisition"
        )
        # this instance is now terminated by id, so the outer coarse reap must not also fire.
        _mark_exact_cleanup(error)
        raise


def _refresh_launch_candidates(
    inst: LambdaInstance,
    tried_regions: set[str],
    absolute_deadline: float,
) -> list[LambdaInstance]:
    return [
        candidate
        for candidate in usable_instances(
            inst.gpu,
            force=True,
            # refresh the SHAPE already being launched: dropping the count here would
            # fall back to a 1-card type while the worker still starts n ranks.
            gpu_count=inst.gpu_count,
            **deadline_kwargs(usable_instances, absolute_deadline),
        )
        if candidate.region not in tried_regions
    ]


def _raise_all_regions_rejected(spec, tried_regions: set[str], last_err: Exception | None) -> None:
    # Reap any phantom instance (accepted but no id returned) before giving up.
    with contextlib.suppress(Exception):
        terminate_run_instances(spec.run_id)
    raise lambda_api.LambdaApiError(
        f"all {len(tried_regions)} Lambda region(s) rejected the {spec.gpu.type} launch "
        f"(no capacity): {sanitize_diagnostic(last_err, limit=1000)}"
    )


@dataclass(frozen=True)
class _LaunchPlan:
    """One launch walk's fixed inputs: everything every candidate region reuses unchanged."""

    spec: Any
    attempt: int
    runtime_secrets: dict | None
    source_snapshot: dict | None
    absolute_deadline: float
    mode: str | None
    models: list | None
    name: str
    ssh_keys: list[str]
    cache_name: str | None
    cold_user_data: str
    cache_user_data: str | None
    default_cache_mount: str


def _build_launch_plan(
    spec,
    attempt: int,
    runtime_secrets: dict | None,
    source_snapshot: dict | None,
    absolute_deadline: float,
    mode: str | None,
    models: list | None,
) -> _LaunchPlan:
    """Build the label, SSH key, and both user_data variants once, before any region is tried."""
    cache_name = getattr(spec.gpu, "network_volume", None)
    default_cache_mount = f"/lambda/nfs/{cache_name}" if cache_name else ""
    build_kwargs = (spec, attempt, runtime_secrets, source_snapshot, absolute_deadline)
    return _LaunchPlan(
        spec=spec,
        attempt=attempt,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        absolute_deadline=absolute_deadline,
        mode=mode,
        models=models,
        name=instance_label(spec.run_id, attempt),
        ssh_keys=resolve_ssh_key_names(
            **deadline_kwargs(resolve_ssh_key_names, absolute_deadline),
        ),
        cache_name=cache_name,
        cold_user_data=_build_launch_user_data(
            *build_kwargs,
            mode=mode if mode == "preload" else None,
            models=models if mode == "preload" else None,
        ),
        cache_user_data=(
            _build_launch_user_data(
                *build_kwargs,
                cache_host_mount=default_cache_mount,
                mode=mode,
                models=models,
            )
            if cache_name
            else None
        ),
        default_cache_mount=default_cache_mount,
    )


def _region_launch_inputs(
    plan: _LaunchPlan, inst: LambdaInstance, say
) -> tuple[str | None, list[str] | None, Exception | None]:
    """Resolve one region's cached launch inputs or return its attachment failure."""
    if not plan.cache_name:
        return plan.cold_user_data, None, None
    try:
        mount_point = lambda_api.ensure_filesystem(
            plan.cache_name,
            inst.region,
            **deadline_kwargs(lambda_api.ensure_filesystem, plan.absolute_deadline),
        )
        # Rebuild user_data when the actual mount_point differs from the default (rare).
        region_user_data = (
            plan.cache_user_data
            if mount_point == plan.default_cache_mount
            else _build_launch_user_data(
                plan.spec,
                plan.attempt,
                plan.runtime_secrets,
                plan.source_snapshot,
                plan.absolute_deadline,
                cache_host_mount=mount_point,
                mode=plan.mode,
                models=plan.models,
            )
        )
        return region_user_data, [plan.cache_name], None
    except Exception as e:
        detail = sanitize_diagnostic(e, limit=1000)
        suffix = " (preload needs it)" if plan.mode == "preload" else ""
        say(f"weight cache unavailable in {inst.region} ({detail}); skipping{suffix}")
        return None, None, e


def launch_and_submit(
    spec,
    instances: list[LambdaInstance],
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
    mode: str | None = None,
    models: list | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> LambdaJobHandle:
    """Launch the first region that accepts the job; walk regions on a capacity rejection.

    Every exit taken after a launch request is ``BaseException``-guarded (as on Vast): an interrupt
    or a raising log line between a successful launch and the returned handle would otherwise leak
    a rented box that no handle, and therefore no teardown path, can name yet.
    """
    say = make_say(log)
    absolute_deadline = require_deadline_at(deadline_at)
    if not instances:
        raise lambda_api.LambdaApiError(
            f"no Lambda capacity for {spec.gpu.type} (no region advertises the instance type)"
        )
    instances = _disk_capable_instances(spec, instances, say)
    plan = _build_launch_plan(
        spec, attempt, runtime_secrets, source_snapshot, absolute_deadline, mode, models
    )

    tried_regions: set[str] = set()
    candidates = list(instances)
    refreshed = False
    last_err: Exception | None = None
    # armed only while a launch request is in flight and no instance id is in hand yet: the
    # provider may have billed a box nothing can name, so the run label is the only way to reap it.
    reap = _CoarseReapGuard()
    try:
        while candidates:
            inst = candidates.pop(0)
            if inst.region in tried_regions:
                continue
            tried_regions.add(inst.region)
            user_data, fs_names, cache_err = _region_launch_inputs(plan, inst, say)
            last_err = cache_err or last_err
            if user_data is None:
                continue
            try:
                return _rent_instance(
                    plan,
                    inst,
                    say,
                    reap,
                    user_data=user_data,
                    file_system_names=fs_names,
                    # bound rather than closed over: this runs inside the region loop, and a
                    # closure over the loop variable would describe whichever region the walk had
                    # reached by call time rather than the one actually rented.
                    describe=lambda instance_id, inst=inst: (
                        f"launched lambda instance {instance_id}: {inst.gpu} {inst.instance_type} "
                        f"${inst.price_usd_hr:.2f}/hr in {inst.region} "
                        f"attempt={attempt}"
                    ),
                )
            except lambda_api.LambdaApiError as e:
                clean = _launch_rejection_is_clean(e)
                if clean:
                    # rented nothing: stand down on the first statement, before the diagnostic and
                    # the say below, either of which can raise while armed and would then reap by
                    # run label -- killing every other concurrent attempt over a rejected request.
                    reap.disarm()
                last_err = e
                detail = sanitize_diagnostic(e, limit=1000)
                if not clean:
                    # ambiguous creates may have billed an instance, so reconcile before any retry.
                    # The guard stays ARMED across this announcement and the abort: if the say
                    # raises, reconciliation never runs, and only an armed guard can still find a
                    # box that is rented but not yet named.
                    say(
                        f"ambiguous launch failure in {inst.region} ({type(e).__name__}); "
                        "attempting cleanup and failing closed"
                    )
                    _abort_ambiguous_launch(spec.run_id, type(e).__name__)
                say(f"region {inst.region} ({inst.gpu} {inst.instance_type}) rejected: {detail}")
                # Preload must not refresh to a different region (would warm the wrong one).
                if mode != "preload" and not candidates and not refreshed:
                    refreshed = True
                    candidates = _refresh_launch_candidates(inst, tried_regions, absolute_deadline)
                    # an empty refresh has nothing to gate; calling the disk filter on it would fall
                    # through unmatched and raise UnsupportedGpuError with an empty shape list. A
                    # refresh can also report catalog disk the first listing left unknown, so the
                    # filter runs again here rather than trusting the pre-walk pass.
                    if candidates:
                        candidates = _disk_capable_instances(spec, candidates, say)
                continue
        return _raise_all_regions_rejected(spec, tried_regions, last_err)
    except BaseException as error:
        # armed for every window where a box may be rented; stands down only once some inner path
        # proved it already terminated that exact instance. Once an id exists the guard holds it,
        # so this cleans up exactly one instance: the run-label sweep is reserved for the in-flight
        # create that has no id to name, where it is the only thing that can find the box.
        if reap.armed and not _exact_cleanup_taken(error):
            if reap.instance_id is not None:
                _cleanup_unpublished_instance(
                    spec.run_id, reap.instance_id, context="interrupted launch walk"
                )
            elif not _launch_failed_before_the_request(error):
                with contextlib.suppress(BaseException):
                    terminate_run_instances(spec.run_id)
        raise


# Tests monkeypatch this name so keep it as a module-level alias.
_make_hf_file_reader = make_hf_text_reader


def _failure_detail(
    hf_repo: str, prefix: str, phase: str, marker: dict | None, attempt: int
) -> str:
    """Assemble bounded failure detail from the worker and host artifacts."""
    parts = []
    if marker and marker.get("error"):
        parts.append(sanitize_diagnostic(marker["error"], limit=4096))
    err_name = error_artifact_name(phase, attempt)
    err = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
    if err:
        parts.append(f"--- {err_name} ---\n{sanitize_diagnostic(err[-4096:], limit=4096)}")
    boot_name = f"lambda_attempt{attempt}_boot.log"
    boot = _make_hf_file_reader(hf_repo, f"{prefix}/{boot_name}")(force=True)
    if boot:
        parts.append(f"--- {boot_name} (host) ---\n{sanitize_diagnostic(boot[-4096:], limit=4096)}")
    return "\n".join(parts) or "lambda worker terminated without a strict terminal marker"


def poll_lambda_job(
    handle: LambdaJobHandle,
    spec,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    setup_grace_s: float = SETUP_GRACE_S,
    stall_after_s: float = STALL_AFTER_S,
    first_liveness_s: float = FIRST_LIVENESS_S,
    deadline_at: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state.

    A thin wrapper that builds the Lambda :class:`InstancePollAdapter` and defers to the shared
    ``poll_instance_job`` kernel (baselined on Vast). Lambda stamps the customer cost from the INSTANCE
    wall (launch->done), notes the provider instance type + region, and — having no live console API —
    reads early-liveness + failure detail from the host boot.log on HF. ``LOAD_TIMEOUT_S`` is read here
    (a module global) so ``monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", ...)`` still bites.
    """
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    err_name = error_artifact_name(spec.phase, handle.attempt)
    # Absence of boot.log while active = cloud-init never ran (sick region / stuck host).
    boot_log_reader = _make_hf_file_reader(
        hf_repo,
        f"{prefix}/lambda_attempt{handle.attempt}_boot.log",
        min_interval_s=60.0,
        **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
    )

    def stamp_cost_and_notes(metrics, *, end_ts, launch_ts) -> None:
        # Lambda bills the INSTANCE wall (launch -> completion), not the worker's train wall.
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
            f"{prefix}/lambda_attempt{handle.attempt}.json",
            min_interval_s=60.0,
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        metrics_reader=_make_hf_file_reader(
            hf_repo,
            f"{prefix}/metrics.json",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        # Resolve get_instance on the api MODULE at call time so a monkeypatch bites.
        fetch_instance=lambda: lambda_api.get_instance(
            handle.instance_id,
            **deadline_kwargs(lambda_api.get_instance, absolute_deadline),
        ),
        poll_error_exceptions=(lambda_api.LambdaApiError,),
        status_field="status",
        running_status="active",
        dead_states=_DEAD_STATES,
        missing_dead_threshold=3,
        # Empty "" boot.log counts as liveness (existence = cloud-init ran) -> ``is not None``.
        early_liveness_alive=lambda: boot_log_reader(force=True) is not None,
        read_current_error=lambda: _make_hf_file_reader(
            hf_repo,
            f"{prefix}/{err_name}",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        )(force=True),
        stamp_cost_and_notes=stamp_cost_and_notes,
        failure_detail=lambda marker: _failure_detail(
            hf_repo, prefix, spec.phase, marker, handle.attempt
        ),
        load_timeout_detail=lambda status, elapsed: (
            f"instance stuck in '{status}' for {int(elapsed)}s "
            f"(never became active; provisioning / host issue)"
        ),
        first_liveness_detail=lambda elapsed, fl: (
            f"no worker liveness (boot.log/heartbeat) for {int(elapsed)}s after instance became active "
            f"(cloud-init/worker never started; limit {int(fl)}s)"
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


def _teardown_polled_instance(handle: LambdaJobHandle, run_id: str) -> None:
    """Attempt teardown without replacing the already-determined worker outcome."""
    try:
        lambda_api.terminate_instance_confirmed(handle.instance_id)
    except Exception as exc:
        logger.error(
            "lambda teardown unconfirmed for instance %s after poll; the persisted handle remains "
            "available for terminal cleanup and orphan sweeps: %s",
            handle.instance_id,
            sanitize_diagnostic(exc, limit=500),
        )
        with contextlib.suppress(Exception):
            terminate_run_instances(run_id)


def submit_attempt_lambda(
    spec,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> PollResult:
    """Launch, poll, and always terminate the instance (finally is the cost-safety primary)."""
    if spec.gpu.type not in GPU_INFO:
        raise lambda_api.LambdaApiError(
            f"submit_attempt_lambda needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    from flash.core.spec import gpu_count_of

    absolute_deadline = require_deadline_at(deadline_at)
    # rent the SHAPE the allocator chose: the worker spawns gpu.count ranks, so a single-card box
    # here would oversubscribe one card with n ranks while billing for n.
    instances = usable_instances(
        spec.gpu.type,
        gpu_count=gpu_count_of(spec),
        **deadline_kwargs(usable_instances, absolute_deadline),
    )
    handle = launch_and_submit(
        spec,
        instances,
        attempt=attempt,
        log=log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        **deadline_kwargs(launch_and_submit, absolute_deadline),
    )
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        reader = heartbeat_reader_for(
            spec,
            **deadline_kwargs(heartbeat_reader_for, absolute_deadline),
        )
        return poll_lambda_job(
            handle,
            spec,
            log=log,
            heartbeat_reader=reader,
            **deadline_kwargs(poll_lambda_job, absolute_deadline),
        )
    finally:
        _teardown_polled_instance(handle, spec.run_id)


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
        if i.get("id") and label_matches_run(str(i.get("name") or ""), prefix)
    ]
    return lambda_api.terminate_instances(ids) if ids else []


def run_instances_remaining(run_id: str) -> list[str]:
    """Return exact run-labeled instances only after complete enumeration and exact lookup.

    Any malformed row, incomplete listing, or lookup failure raises so handleless recovery cannot
    mistake an unknown fleet state for confirmed cleanup.
    """
    if not run_id:
        return []
    instances = lambda_api.list_instances(strict=True)
    prefix = run_label_prefix(run_id)
    remaining: list[str] = []
    for instance in instances:
        if not label_matches_run(str(instance.get("name") or ""), prefix):
            continue
        raw_id = instance.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise lambda_api.LambdaApiError(
                "lambda instance carries the exact run label but has no usable id"
            )
        instance_id = raw_id.strip()
        if lambda_api.get_instance(instance_id, strict=True) is not None:
            remaining.append(instance_id)
    return remaining


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
        # never fall through to an empty set because that would reap every live run's instance.
        logger.warning("lambda orphan sweep skipped; could not resolve run sets: %s", exc)
        return []
    active = {run_label_prefix(a) for a in (labels or set())}
    # None = unscoped (single-plane); empty set = this plane owns nothing, reaps nothing.
    known_prefixes = (
        None if known_labels is None else {run_label_prefix(a) for a in (known or set())}
    )

    def _matches(prefixes: set[str]) -> bool:
        return any(label_matches_run(name, p) for p in prefixes)

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
                        "driver lost)",
                        name,
                    )
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
