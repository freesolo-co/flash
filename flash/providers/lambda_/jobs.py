"""Lambda Cloud training launch and artifact-backed supervision."""

from __future__ import annotations

import time

from flash._internal.background import TaskStopped
from flash.providers._lifecycle.instances.instance import build_payload, instance_label
from flash.providers._lifecycle.instances.poll import (
    FIRST_LIVENESS_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
)
from flash.providers._lifecycle.instances.poll_instance import (
    InstancePollAdapter,
    poll_instance_job,
)
from flash.providers._lifecycle.net.deadline import require_create_allowance, require_deadline_at
from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.artifacts.channel import (
    error_artifact_name,
    heartbeat_reader_for,
    make_artifact_text_reader,
)
from flash.providers.core.base import CapacityUnavailableError, JobHandle, get_gpu_info
from flash.providers.lambda_ import api
from flash.providers.lambda_.config import scope_fingerprint
from flash.providers.lambda_.handles import LambdaJobHandle, require_scope
from flash.runner.supervise.retry_decision import BOOT_TIMEOUT_FAILURE

# Lambda VMs usually reach "active" in a few minutes, but large multi-GPU types can take far longer.
BOOT_TIMEOUT_S = 1800.0
# Lambda instance states that mean the VM will not make progress.
DEAD_STATES = frozenset({"terminated", "terminating", "unhealthy", "preempted"})


def poll(handle, spec, *, log=None, deadline_at=None):
    h = LambdaJobHandle.from_dict(handle.to_dict())
    require_scope(h)
    if (h.run_id, h.phase, h.hf_repo) != (spec.run_id, spec.phase, spec.train.hf_repo):
        raise ValueError("Lambda attempt does not belong to this run")
    prefix = f"{spec.phase}/{spec.run_id}"

    def reader(filename, **kwargs):
        return make_artifact_text_reader(
            spec.train.hf_repo, f"{prefix}/{filename}", deadline_at=deadline_at, **kwargs
        )

    console = reader(f"console_{spec.phase}_attempt{h.attempt}.txt")
    error = reader(error_artifact_name(spec.phase, h.attempt))

    def stamp(metrics, *, end_ts, launch_ts):
        metrics["cost_usd"] = round(max(0, end_ts - launch_ts) / 3600 * h.hourly_usd, 6)
        notes = metrics.setdefault("notes", {})
        notes.update(
            provider="lambda",
            lambda_instance_usd_hr=h.hourly_usd,
            lambda_instance_id=h.instance_id,
            lambda_instance_type=h.instance_type,
            lambda_region=h.region,
            cost_basis="lambda_on_demand",
        )

    adapter = InstancePollAdapter(
        instance_id=h.instance_id,
        run_id=spec.run_id,
        current_attempt=h.attempt,
        launch_ts=h.started_ts,
        done_reader=reader("DONE"),
        marker_reader=reader(f"lambda_attempt{h.attempt}.json"),
        metrics_reader=reader("metrics.json"),
        fetch_instance=lambda: api.get_instance(
            h.instance_id, name=h.name, deadline_at=deadline_at
        ),
        poll_error_exceptions=(api.LambdaApiError,),
        status_field="status",
        running_status="active",
        dead_states=DEAD_STATES,
        missing_dead_threshold=3,
        early_liveness_alive=lambda: console(force=True) is not None,
        read_current_error=lambda: error(force=True),
        stamp_cost_and_notes=stamp,
        failure_detail=lambda marker: (
            (marker or {}).get("error")
            or "Lambda worker stopped without a successful terminal marker"
        ),
        load_timeout_detail=lambda status, elapsed: (
            f"Lambda instance remained {status} for {int(elapsed)}s"
        ),
        first_liveness_detail=lambda elapsed, limit: (
            f"Lambda worker produced no liveness for {int(elapsed)}s (limit {int(limit)}s)"
        ),
        load_timeout_failure=BOOT_TIMEOUT_FAILURE,
    )
    return poll_instance_job(
        adapter,
        log=log,
        heartbeat_reader=heartbeat_reader_for(spec, deadline_at=deadline_at),
        setup_grace_s=SETUP_GRACE_S,
        stall_after_s=STALL_AFTER_S,
        first_liveness_s=FIRST_LIVENESS_S,
        load_timeout_s=BOOT_TIMEOUT_S,
        deadline_at=deadline_at,
    )


def choose_offer(gpu, count, *, disk_gb, deadline_at):
    """The cheapest live ``count``-card type of ``gpu`` whose fixed disk holds the run.

    Re-reads the catalog (never the cache) right before launch. Raises
    ``CapacityUnavailableError`` when no region has one, so the run requeues.
    """
    fitting = [
        offer
        for offer in api.offers(force=True, deadline_at=deadline_at)
        if offer.gpu == gpu
        and offer.count == count
        and (not disk_gb or offer.disk_gb is None or offer.disk_gb >= disk_gb)
    ]
    if not fitting:
        raise CapacityUnavailableError(
            f"Lambda has no capacity for {count}x {gpu} right now; requeuing",
            cause="no regional capacity",
        )
    offer = min(fitting, key=lambda o: (o.hourly_usd, o.instance_type))
    return offer, offer.regions[0]


def submit_attempt(
    spec,
    *,
    log=None,
    on_handle=None,
    attempt=0,
    runtime_secrets=None,
    source_snapshot=None,
    deadline_at=None,
):
    from flash.core.spec import gpu_count_of
    from flash.providers.lambda_.bootstrap import user_data
    from flash.providers.lambda_.pricing import LAMBDA_GPUS
    from flash.runtime_capsule import build_capsule

    deadline_at = require_deadline_at(deadline_at)
    gpu = get_gpu_info(spec.gpu.type).name
    count = gpu_count_of(spec)
    if gpu not in LAMBDA_GPUS or count not in LAMBDA_GPUS[gpu].counts:
        raise ValueError(f"Lambda does not rent {count}x {gpu}")
    owner_scope = scope_fingerprint()
    offer, region = choose_offer(gpu, count, disk_gb=spec.gpu.disk_gb, deadline_at=deadline_at)
    if log is not None:
        print(
            f"Lambda: launching {offer.instance_type} in {region} at ${offer.hourly_usd:.2f}/hr",
            file=log,
            flush=True,
        )
    payload = build_payload(
        spec,
        attempt,
        arm="lambda",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
    )
    name = instance_label(spec.run_id, attempt)
    capsule, _ = build_capsule("instance-bootstrap")
    script = user_data(payload, capsule, worker_image_for_gpu(gpu), deadline_at=deadline_at)
    require_create_allowance(deadline_at)
    started = time.time()
    instance_id = api.launch(
        name=name,
        run_id=spec.run_id,
        attempt=attempt,
        offer=offer,
        region=region,
        user_data=script,
        deadline_at=deadline_at,
    )
    persisted = detached = False
    try:
        h = LambdaJobHandle(
            instance_id=instance_id,
            gpu=gpu,
            hourly_usd=offer.hourly_usd,
            attempt=attempt,
            started_ts=started,
            instance_type=offer.instance_type,
            region=region,
            name=name,
            run_id=spec.run_id,
            hf_repo=spec.train.hf_repo,
            phase=spec.phase,
            gpu_count=count,
            scope=owner_scope,
        )
        handle = JobHandle.from_dict(h.to_dict())
        if on_handle is not None:
            on_handle(handle)
            persisted = True
        return poll(handle, spec, log=log, deadline_at=deadline_at)
    except TaskStopped:
        # Control-plane shutdown (e.g. a redeploy) detaches supervision only. A durably recorded
        # handle lets the next plane re-attach to the live VM, so leave it running.
        detached = persisted
        raise
    finally:
        if not detached:
            # cleanup errors stay visible; a failed terminate must not be mistaken for a stopped VM.
            api.terminate_confirmed(instance_id, name=name)
