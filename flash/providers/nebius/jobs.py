"""Nebius Spot training launch and artifact-backed supervision."""

from __future__ import annotations

import time

from flash.providers._lifecycle.instances.instance import build_payload, instance_label
from flash.providers._lifecycle.instances.poll import (
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
)
from flash.providers._lifecycle.instances.poll_instance import (
    InstancePollAdapter,
    poll_instance_job,
)
from flash.providers._lifecycle.net.deadline import require_deadline_at
from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.artifacts.channel import (
    error_artifact_name,
    heartbeat_reader_for,
    make_artifact_text_reader,
)
from flash.providers.core.base import JobHandle, get_gpu_info
from flash.providers.nebius import api
from flash.providers.nebius.config import scope_fingerprint
from flash.providers.nebius.handles import SpotJobHandle, require_scope
from flash.providers.nebius.pricing import hourly_rate


def poll(handle, spec, *, log=None, deadline_at=None):
    h = SpotJobHandle.from_dict(handle.to_dict())
    require_scope(h)
    if (h.run_id, h.phase, h.hf_repo) != (spec.run_id, spec.phase, spec.train.hf_repo):
        raise ValueError("Nebius Spot attempt does not belong to this run")
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
            provider="spot",
            spot_entry_estimate_usd_hr=h.hourly_usd,
            nebius_instance_id=h.instance_id,
            cost_basis="spot_entry_estimate",
        )

    adapter = InstancePollAdapter(
        instance_id=h.instance_id,
        run_id=spec.run_id,
        current_attempt=h.attempt,
        launch_ts=h.started_ts,
        done_reader=reader("DONE"),
        marker_reader=reader(f"spot_attempt{h.attempt}.json"),
        metrics_reader=reader("metrics.json"),
        fetch_instance=lambda: api.get_instance(h.instance_id, deadline_at=deadline_at),
        poll_error_exceptions=(api.NebiusApiError,),
        status_field="status",
        running_status="running",
        dead_states=frozenset({"stopped", "error"}),
        missing_dead_threshold=3,
        early_liveness_alive=lambda: console(force=True) is not None,
        read_current_error=lambda: error(force=True),
        stamp_cost_and_notes=stamp,
        failure_detail=lambda marker: (
            (marker or {}).get("error")
            or "Nebius Spot worker stopped without a successful terminal marker"
        ),
        load_timeout_detail=lambda status, elapsed: (
            f"Nebius Spot sandbox remained {status} for {int(elapsed)}s"
        ),
        first_liveness_detail=lambda elapsed, limit: (
            f"Nebius Spot worker produced no liveness for {int(elapsed)}s (limit {int(limit)}s)"
        ),
    )
    return poll_instance_job(
        adapter,
        log=log,
        heartbeat_reader=heartbeat_reader_for(spec, deadline_at=deadline_at),
        setup_grace_s=SETUP_GRACE_S,
        stall_after_s=STALL_AFTER_S,
        first_liveness_s=FIRST_LIVENESS_S,
        load_timeout_s=LOAD_TIMEOUT_S,
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
    gpu = spec.gpu.type
    count = gpu_count_of(spec)
    if count > get_gpu_info(gpu).max_count:
        raise ValueError(f"Nebius Spot does not support {count} GPUs of class {gpu}")
    owner_scope = scope_fingerprint()
    payload = build_payload(
        spec,
        attempt,
        arm="spot",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
    )
    from flash.providers.nebius.queue import wait_for_price

    wait_for_price(gpu, spec.run_id, deadline_at=deadline_at, log=log)
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
        disk_gb=spec.gpu.disk_gb,
        deadline_at=deadline_at,
    )
    try:
        h = SpotJobHandle(
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
