"""Modal training launch and artifact-backed supervision."""

from __future__ import annotations

import time

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
from flash.providers._lifecycle.net.deadline import require_deadline_at
from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.artifacts.hf import (
    error_artifact_name,
    heartbeat_reader_for,
    make_hf_text_reader,
)
from flash.providers.core.base import JobHandle, get_gpu_info
from flash.providers.modal import api
from flash.providers.modal.auth import scope_fingerprint
from flash.providers.modal.handles import ModalJobHandle, require_scope
from flash.providers.modal.pricing import hourly_rate
from flash.providers.modal.startup import start_time_reader


def poll(handle, spec, *, log=None, deadline_at=None):
    h = ModalJobHandle.from_dict(handle.to_dict())
    require_scope(h)
    if (h.run_id, h.phase, h.hf_repo) != (spec.run_id, spec.phase, spec.train.hf_repo):
        raise ValueError("Modal attempt does not belong to this run")
    prefix = f"{spec.phase}/{spec.run_id}"

    def reader(filename, **kwargs):
        return make_hf_text_reader(
            spec.train.hf_repo, f"{prefix}/{filename}", deadline_at=deadline_at, **kwargs
        )

    console = reader(f"console_{spec.phase}_attempt{h.attempt}.txt")
    read_start = start_time_reader(h.to_dict(), deadline_at=deadline_at)
    worker_started = None

    def fetch_instance():
        nonlocal worker_started
        worker_started = worker_started or read_start()
        instance = api.get_instance(h.instance_id, deadline_at=deadline_at)
        if instance is not None and instance["status"] == "pending" and worker_started is not None:
            return {**instance, "status": "running", "started_ts": worker_started}
        return instance

    error = reader(error_artifact_name(spec.phase, h.attempt))

    def stamp(metrics, *, end_ts, launch_ts):
        started = worker_started or read_start()
        if started is None:
            metrics.pop("cost_usd", None)
        else:
            metrics["cost_usd"] = round(max(0, end_ts - started) / 3600 * h.hourly_usd, 6)
        notes = metrics.setdefault("notes", {})
        notes.update(
            provider="modal",
            modal_rate_usd_hr=h.hourly_usd,
            modal_sandbox_id=h.instance_id,
            cost_basis="bootstrap_resource_estimate"
            if started is not None
            else "unknown_start_time",
        )

    adapter = InstancePollAdapter(
        instance_id=h.instance_id,
        run_id=spec.run_id,
        current_attempt=h.attempt,
        launch_ts=h.started_ts,
        done_reader=reader("DONE"),
        marker_reader=reader(f"modal_attempt{h.attempt}.json"),
        metrics_reader=reader("metrics.json"),
        fetch_instance=fetch_instance,
        poll_error_exceptions=(api.ModalApiError,),
        status_field="status",
        running_status="running",
        dead_states=frozenset({"terminated"}),
        missing_dead_threshold=3,
        early_liveness_alive=lambda: console(force=True) is not None,
        read_current_error=lambda: error(force=True),
        stamp_cost_and_notes=stamp,
        failure_detail=lambda marker: (
            (marker or {}).get("error")
            or "Modal worker stopped without a successful terminal marker"
        ),
        load_timeout_failure="no_capacity",
        load_timeout_detail=lambda status, elapsed: (
            f"Modal sandbox remained {status} for {int(elapsed)}s"
        ),
        first_liveness_detail=lambda elapsed, limit: (
            f"Modal worker produced no liveness for {int(elapsed)}s (limit {int(limit)}s)"
        ),
    )
    return poll_instance_job(
        adapter,
        log=log,
        heartbeat_reader=heartbeat_reader_for(spec, deadline_at=deadline_at),
        setup_grace_s=SETUP_GRACE_S,
        stall_after_s=STALL_AFTER_S,
        first_liveness_s=FIRST_LIVENESS_S,
        # Capacity waiting is bounded separately from a running worker's liveness timer.
        load_timeout_s=1800.0,
        deadline_at=deadline_at,
    )


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
    from flash.runtime_capsule import build_capsule

    deadline_at = require_deadline_at(deadline_at)
    if spec.gpu.disk_gb > 512:
        raise ValueError("Modal Sandboxes support at most 512 GiB ephemeral disk")
    gpu = spec.gpu.type
    count = gpu_count_of(spec)
    if count > get_gpu_info(gpu).max_count:
        raise ValueError(f"Modal does not support {count} GPUs of class {gpu}")
    owner_scope = scope_fingerprint()
    payload = build_payload(
        spec,
        attempt,
        arm="modal",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
    )
    import os

    if spec.gpu.network_volume and os.environ.get("FLASH_WEIGHT_CACHE_DISABLED", "").strip() != "1":
        payload["cache_volume"] = spec.gpu.network_volume
        payload["env"]["FLASH_WEIGHT_CACHE_DIR"] = f"{api.CACHE_MOUNT}/hf-cache/hub"
    name = instance_label(spec.run_id, attempt)
    started = time.time()
    capsule, _ = build_capsule("instance-bootstrap")
    sandbox_id = api.create(
        name=name,
        payload=payload,
        image_reference=worker_image_for_gpu(gpu),
        gpu=gpu,
        gpu_count=count,
        capsule=capsule,
        deadline_at=deadline_at,
    )
    try:
        h = ModalJobHandle(
            instance_id=sandbox_id,
            gpu=gpu,
            hourly_usd=hourly_rate(gpu) * count,
            attempt=attempt,
            started_ts=started,
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
        return poll(handle, spec, log=log, deadline_at=deadline_at)
    finally:
        # cleanup errors stay visible; a failed delete must not be mistaken for a stopped worker.
        api.terminate(sandbox_id)
