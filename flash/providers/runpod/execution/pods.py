"""Persistent RunPod Secure Cloud Pod lifecycle for managed training."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import replace

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.instance import _spill_large_spec_to_hf
from flash.providers._lifecycle.instances.instance import (
    build_payload as build_instance_payload,
)
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
from flash.providers._lifecycle.net.deadline import (
    deadline_kwargs,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.artifacts.hf import (
    error_artifact_name,
    heartbeat_reader_for,
    make_hf_text_reader,
)
from flash.providers.core.base import GPU_INFO, PollResult, UnreconciledCreateError
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client import auth as runpod_auth
from flash.providers.runpod.client import pods as runpod_pods
from flash.providers.runpod.client.pricing import hourly_rate
from flash.providers.runpod.execution.identity import (
    EXACT,
    POD_CREATE_PENDING,
    PRE_POD_CREATE,
    SECRET_CREATE_PENDING,
    RunpodCreateAbsent,
    RunpodPodHandle,
    fresh_payload_secret_name,
    pod_attempt_label_base,
    pod_label_from_payload,
)
from flash.providers.runpod.execution.identity import (
    handle_label_digest_is_valid as _handle_label_digest_is_valid,
)
from flash.providers.runpod.execution.identity import (
    payload_for_handle as _payload_for_handle,
)
from flash.providers.runpod.execution.identity import (
    pod_identity_is_incomplete as _pod_identity_is_incomplete,
)
from flash.providers.runpod.execution.identity import (
    pod_matches as _pod_matches,
)

logger = get_logger(__name__)

_CREATE_RECONCILE_POLLS = 3
_CREATE_RECONCILE_WAIT_S = 1.0
_DEAD_STATES = frozenset({"DEAD", "EXITED", "FAILED", "STOPPED", "TERMINATED"})


def _build_instance_payload(spec, seed, attempt, runtime_secrets, source_snapshot, deadline_at):
    payload = build_instance_payload(
        spec,
        seed,
        attempt,
        arm="runpod",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
        preserve_runpod_volume=True,
    )
    return _spill_large_spec_to_hf(payload)


def _new_secret_intent(
    spec,
    seed: int,
    attempt: int,
    fingerprint: str,
    *,
    container_registry_auth_id: str | None,
    started_ts: float,
    deadline_at: float,
) -> RunpodPodHandle:
    from flash.core.spec import gpu_count_of

    for _ in range(_CREATE_RECONCILE_POLLS):
        name = fresh_payload_secret_name()
        account_id, existing = runpod_pods.list_secrets_for_fingerprint(
            fingerprint, name=name, deadline_at=deadline_at
        )
        if existing:
            continue
        label = pod_attempt_label_base(spec.run_id, seed, attempt)
        return RunpodPodHandle(
            instance_id=label,
            gpu=spec.gpu.type,
            hourly_usd=hourly_rate(spec.gpu.type) * gpu_count_of(spec),
            attempt=attempt,
            started_ts=started_ts,
            phase=SECRET_CREATE_PENDING,
            label=label,
            key_fingerprint=fingerprint,
            account_id=account_id,
            payload_secret_id=None,
            payload_secret_name=name,
            data_center_id=None,
            network_volume_id=None,
            container_disk_gb=int(spec.gpu.disk_gb),
            container_registry_auth_id=container_registry_auth_id,
            gpu_count=gpu_count_of(spec),
        )
    raise runpod_api.RunpodApiError("could not allocate a fresh RunPod payload secret identity")


def _create_payload_secret(
    intent: RunpodPodHandle,
    serialized_payload: str,
    *,
    deadline_at: float,
) -> RunpodPodHandle:
    try:
        secret = runpod_pods.create_secret_for_fingerprint(
            intent.key_fingerprint,
            intent.payload_secret_name,
            serialized_payload,
            deadline_at=deadline_at,
        )
    except runpod_pods.RunpodMutationAmbiguous:
        observed = []
        for poll in range(_CREATE_RECONCILE_POLLS):
            account_id, observed = runpod_pods.list_secrets_for_fingerprint(
                intent.key_fingerprint,
                name=intent.payload_secret_name,
                deadline_at=deadline_at,
            )
            if account_id != intent.account_id:
                raise UnreconciledCreateError(
                    "RunPod payload secret account identity changed"
                ) from None
            if observed:
                break
            if poll + 1 < _CREATE_RECONCILE_POLLS:
                time.sleep(_CREATE_RECONCILE_WAIT_S)
        if len(observed) != 1:
            raise UnreconciledCreateError(
                "ambiguous RunPod payload secret creation could not be reconciled"
            ) from None
        secret = observed[0]
    if secret.name != intent.payload_secret_name:
        raise runpod_pods.RunpodMutationAmbiguous("runpod payload secret identity is unknown")
    return replace(intent, phase=PRE_POD_CREATE, payload_secret_id=secret.id)


def _volume_candidates(spec, fingerprint: str, *, deadline_at: float):
    base = getattr(spec.gpu, "network_volume", None)
    if not base:
        return [(None, None)]
    from flash.core.spec_persistence import volume_gb
    from flash.providers.runpod.execution import resources
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        WEIGHT_CACHE_VOLUME_NAME,
    )

    requested = volume_gb(getattr(spec.gpu, "network_volume_gb", WEIGHT_CACHE_VOLUME_GB))
    if str(base) == WEIGHT_CACHE_VOLUME_NAME:
        requested = max(requested, WEIGHT_CACHE_VOLUME_GB)
    data_centers = resources.weight_cache_datacenters(fingerprint, deadline_at=deadline_at)
    candidates = []
    for dc_id in data_centers:
        try:
            volume = resources.ensure_account_volume(
                fingerprint,
                base=str(base),
                data_center_id=dc_id,
                size_gb=requested,
                deadline_at=deadline_at,
            )
        except runpod_pods.RunpodCapacityError:
            continue
        candidates.append((dc_id, volume.id))
    if not candidates:
        raise runpod_pods.RunpodCapacityError(
            "no storage-capable data center was usable in this RunPod account"
        )
    return candidates


def _pod_create_intent(
    secret_handle: RunpodPodHandle,
    spec,
    seed: int,
    *,
    data_center_id: str | None,
    network_volume_id: str | None,
) -> RunpodPodHandle:
    pending = replace(
        secret_handle,
        data_center_id=data_center_id,
        network_volume_id=network_volume_id,
    )
    base = pod_attempt_label_base(spec.run_id, seed, secret_handle.attempt)
    identity = _payload_for_handle(replace(pending, label=base))
    label = pod_label_from_payload(base, secret_handle.payload_secret_name, identity)
    return replace(
        pending,
        instance_id=label,
        phase=PRE_POD_CREATE,
        label=label,
    )


def _exact_handle(pending: RunpodPodHandle, pod: runpod_pods.RunpodPod) -> RunpodPodHandle:
    if (
        pending.image_name is not None
        and pod.image_name is not None
        and pod.image_name != pending.image_name
    ):
        raise UnreconciledCreateError("RunPod realized image conflicts with the persisted request")
    if (
        pending.data_center_id is not None
        and pod.data_center_id is not None
        and pending.data_center_id != pod.data_center_id
    ):
        raise UnreconciledCreateError(
            "RunPod realized data center conflicts with the persisted request"
        )
    return replace(
        pending,
        instance_id=pod.id,
        phase=EXACT,
        hourly_usd=pod.cost_per_hr if pod.cost_per_hr is not None else pending.hourly_usd,
        image_name=pending.image_name or pod.image_name,
        data_center_id=pod.data_center_id or pending.data_center_id,
    )


def _enrich_exact_handle(
    handle: RunpodPodHandle,
    pod: runpod_pods.RunpodPod | None,
) -> RunpodPodHandle:
    if pod is None:
        return handle
    if (
        handle.image_name is not None
        and pod.image_name is not None
        and pod.image_name != handle.image_name
    ):
        raise UnreconciledCreateError("RunPod observed image conflicts with the persisted request")
    if (
        handle.data_center_id is not None
        and pod.data_center_id is not None
        and handle.data_center_id != pod.data_center_id
    ):
        raise UnreconciledCreateError(
            "RunPod observed data center conflicts with the persisted request"
        )
    image_name = handle.image_name or pod.image_name
    data_center_id = handle.data_center_id or pod.data_center_id
    if image_name != handle.image_name or data_center_id != handle.data_center_id:
        return replace(handle, image_name=image_name, data_center_id=data_center_id)
    return handle


def _delete_duplicate_pods(
    fingerprint: str,
    pods: list[runpod_pods.RunpodPod],
    keep_id: str,
    *,
    deadline_at: float,
) -> None:
    for duplicate in pods:
        if duplicate.id == keep_id:
            continue
        runpod_pods.delete_pod_for_fingerprint(duplicate.id, fingerprint, deadline_at=deadline_at)


def _reconcile_ambiguous_create(
    pending: RunpodPodHandle,
    payload: dict,
    *,
    deadline_at: float,
) -> RunpodPodHandle:
    for poll in range(_CREATE_RECONCILE_POLLS):
        observed = runpod_pods.list_pods_for_key(
            runpod_api._key_for_fingerprint(pending.key_fingerprint),
            keep_name=pending.label,
            deadline_at=deadline_at,
        )
        matching = [
            pod
            for pod in observed
            if _pod_matches(
                pod,
                payload,
                network_volume_id=pending.network_volume_id,
                data_center_id=pending.data_center_id,
                allow_preplacement=True,
            )
        ]
        if matching:
            if not _handle_label_digest_is_valid(pending):
                raise UnreconciledCreateError(
                    "ambiguous RunPod Pod label digest does not match its request"
                )
            keep = sorted(matching, key=lambda item: item.id)[0]
            _delete_duplicate_pods(
                pending.key_fingerprint,
                observed,
                keep.id,
                deadline_at=deadline_at,
            )
            return _exact_handle(pending, keep)
        incomplete = [
            pod
            for pod in observed
            if _pod_identity_is_incomplete(
                pod,
                payload,
                network_volume_id=pending.network_volume_id,
                data_center_id=pending.data_center_id,
                allow_preplacement=True,
            )
        ]
        if incomplete:
            raise UnreconciledCreateError(
                "ambiguous RunPod create exposed a Pod with incomplete immutable fields"
            )
        if observed:
            for pod in observed:
                with contextlib.suppress(Exception):
                    runpod_pods.delete_pod_for_fingerprint(
                        pod.id, pending.key_fingerprint, deadline_at=deadline_at
                    )
            raise UnreconciledCreateError(
                "ambiguous RunPod create exposed a Pod with conflicting immutable fields"
            )
        if poll + 1 < _CREATE_RECONCILE_POLLS:
            time.sleep(_CREATE_RECONCILE_WAIT_S)
    raise UnreconciledCreateError(
        "ambiguous RunPod Pod creation remained absent from authoritative account listings"
    )


def create_or_adopt_pod(
    pending: RunpodPodHandle,
    payload: dict,
    *,
    deadline_at: float,
) -> RunpodPodHandle:
    try:
        pod = runpod_pods.create_pod_for_fingerprint(
            pending.key_fingerprint,
            payload,
            deadline_at=deadline_at,
        )
    except runpod_pods.RunpodMutationAmbiguous:
        return _reconcile_ambiguous_create(pending, payload, deadline_at=deadline_at)
    if not _pod_matches(
        pod,
        payload,
        network_volume_id=pending.network_volume_id,
        data_center_id=pending.data_center_id,
        allow_preplacement=True,
    ):
        return _reconcile_ambiguous_create(pending, payload, deadline_at=deadline_at)
    observed = runpod_pods.list_pods_for_key(
        runpod_api._key_for_fingerprint(pending.key_fingerprint),
        keep_name=pending.label,
        deadline_at=deadline_at,
    )
    matching = [
        item
        for item in observed
        if _pod_matches(
            item,
            payload,
            network_volume_id=pending.network_volume_id,
            data_center_id=pending.data_center_id,
            allow_preplacement=True,
        )
    ]
    selected = next((item for item in matching if item.id == pod.id), None)
    if selected is None:
        return _reconcile_ambiguous_create(pending, payload, deadline_at=deadline_at)
    if len(matching) > 1:
        _delete_duplicate_pods(
            pending.key_fingerprint,
            matching,
            selected.id,
            deadline_at=deadline_at,
        )
    return _exact_handle(pending, selected)


def _bounded_secret_observation(
    handle: RunpodPodHandle,
    *,
    deadline_at: float,
) -> list[runpod_pods.RunpodSecret]:
    observed = []
    for poll in range(_CREATE_RECONCILE_POLLS):
        account_id, observed = runpod_pods.list_secrets_for_fingerprint(
            handle.key_fingerprint,
            name=handle.payload_secret_name,
            deadline_at=deadline_at,
        )
        if account_id != handle.account_id:
            raise UnreconciledCreateError("RunPod payload secret account identity changed")
        if observed:
            break
        if poll + 1 < _CREATE_RECONCILE_POLLS:
            time.sleep(_CREATE_RECONCILE_WAIT_S)
    return observed


def resolve_pending_handle(
    handle: RunpodPodHandle,
    spec,
    seed: int,
    *,
    deadline_at: float,
) -> RunpodPodHandle:
    if not handle.pending:
        return handle
    if handle.phase == SECRET_CREATE_PENDING:
        observed = _bounded_secret_observation(handle, deadline_at=deadline_at)
        if not observed:
            raise RunpodCreateAbsent("RunPod payload secret creation was authoritatively absent")
        if len(observed) != 1:
            raise UnreconciledCreateError("RunPod payload secret creation has duplicate identities")
        return replace(handle, phase=PRE_POD_CREATE, payload_secret_id=observed[0].id)
    if handle.phase == PRE_POD_CREATE:
        raise RunpodCreateAbsent("RunPod Pod creation had not begun")
    if handle.phase != POD_CREATE_PENDING:
        raise UnreconciledCreateError("RunPod pending creation phase is invalid")
    key = runpod_api._key_for_fingerprint(handle.key_fingerprint)
    observed = []
    for poll in range(_CREATE_RECONCILE_POLLS):
        observed = runpod_pods.list_pods_for_key(
            key, keep_name=handle.label, deadline_at=deadline_at
        )
        if observed:
            break
        if poll + 1 < _CREATE_RECONCILE_POLLS:
            time.sleep(_CREATE_RECONCILE_WAIT_S)
    if not observed:
        raise RunpodCreateAbsent("RunPod Pod creation was authoritatively absent")
    payload = _payload_for_handle(handle)
    expected_label = pod_label_from_payload(
        pod_attempt_label_base(spec.run_id, seed, handle.attempt),
        handle.payload_secret_name,
        payload,
    )
    if handle.label != expected_label or not _handle_label_digest_is_valid(handle):
        raise UnreconciledCreateError("pending RunPod Pod label digest does not match its request")
    matching = [
        pod
        for pod in observed
        if _pod_matches(
            pod,
            payload,
            network_volume_id=handle.network_volume_id,
            data_center_id=handle.data_center_id,
            allow_preplacement=True,
        )
    ]
    if not matching:
        raise UnreconciledCreateError(
            "pending RunPod Pod identity is incomplete or conflicts with the persisted request"
        )
    keep = sorted(matching, key=lambda item: item.id)[0]
    _delete_duplicate_pods(handle.key_fingerprint, observed, keep.id, deadline_at=deadline_at)
    return _exact_handle(handle, keep)


def _make_hf_file_reader(repo: str, path: str, **kwargs):
    return make_hf_text_reader(repo, path, **kwargs)


def poll_runpod_pod(
    handle: RunpodPodHandle,
    spec,
    seed: int,
    *,
    log=None,
    heartbeat_reader=None,
    on_handle=None,
    interval_s: float = 15.0,
    setup_grace_s: float = SETUP_GRACE_S,
    stall_after_s: float = STALL_AFTER_S,
    first_liveness_s: float = FIRST_LIVENESS_S,
    deadline_at: float | None = None,
) -> PollResult:
    if handle.pending:
        raise ValueError("pending RunPod Pod handles cannot be polled")
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    err_name = error_artifact_name(spec.phase, handle.attempt)

    def reader(path: str, *, min_interval_s: float = 15.0):
        return _make_hf_file_reader(
            hf_repo,
            f"{prefix}/{path}",
            min_interval_s=min_interval_s,
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        )

    error_reader = reader(err_name)

    def fetch_instance():
        nonlocal handle
        pod = runpod_pods.get_pod_for_fingerprint(
            handle.pod_id,
            handle.key_fingerprint,
            **deadline_kwargs(runpod_pods.get_pod_for_fingerprint, absolute_deadline),
        )
        enriched = _enrich_exact_handle(handle, pod)
        if enriched != handle:
            handle = enriched
            if on_handle is not None:
                on_handle(handle.to_dict())
        return None if pod is None else {"desired_status": pod.desired_status}

    def stamp_cost_and_notes(metrics, *, end_ts, launch_ts) -> None:
        wall = max(0.0, end_ts - launch_ts)
        metrics["cost_usd"] = round(wall / 3600.0 * handle.hourly_usd, 6)
        notes = metrics.get("notes") if isinstance(metrics.get("notes"), dict) else {}
        notes.update(
            {
                "provider": "runpod",
                "runpod_rate_usd_hr": handle.hourly_usd,
                "runpod_gpu": handle.gpu,
                "runpod_pod_id": handle.pod_id,
                "runpod_pod_wall_seconds": round(wall, 3),
            }
        )
        metrics["notes"] = notes

    def failure_detail(marker) -> str:
        parts = []
        if marker and marker.get("error"):
            parts.append(sanitize_diagnostic(marker["error"], limit=4096))
        error = error_reader(force=True)
        if error:
            parts.append(f"--- {err_name} ---\n{sanitize_diagnostic(error[-4096:], limit=4096)}")
        return "\n".join(parts) or "runpod worker terminated without a strict terminal marker"

    adapter = InstancePollAdapter(
        instance_id=handle.pod_id,
        run_id=spec.run_id,
        current_attempt=handle.attempt,
        launch_ts=handle.started_ts,
        done_reader=reader("DONE"),
        marker_reader=reader(f"runpod_attempt{handle.attempt}.json", min_interval_s=60.0),
        metrics_reader=reader("metrics.json"),
        fetch_instance=fetch_instance,
        poll_error_exceptions=(runpod_api.RunpodApiError,),
        status_field="desired_status",
        running_status="RUNNING",
        dead_states=_DEAD_STATES,
        missing_dead_threshold=3,
        early_liveness_alive=lambda: bool(
            heartbeat_reader is not None and heartbeat_reader(force=True)
        ),
        read_current_error=lambda: error_reader(force=True),
        stamp_cost_and_notes=stamp_cost_and_notes,
        failure_detail=failure_detail,
        load_timeout_detail=lambda status, elapsed: (
            f"Pod stuck in '{status}' for {int(elapsed)}s (never became running)"
        ),
        first_liveness_detail=lambda elapsed, limit: (
            f"no worker heartbeat for {int(elapsed)}s after Pod became running "
            f"(worker never started; limit {int(limit)}s)"
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


def _delete_payload_secret(handle: RunpodPodHandle, *, deadline_at: float) -> None:
    if handle.payload_secret_id is None:
        observed = _bounded_secret_observation(handle, deadline_at=deadline_at)
        if len(observed) > 1:
            raise runpod_api.RunpodApiError("runpod payload secret cleanup identity is ambiguous")
        if not observed:
            return
        secret = observed[0]
    else:
        secret = runpod_pods.RunpodSecret(handle.payload_secret_id, handle.payload_secret_name)
    runpod_pods.delete_secret_for_fingerprint(
        handle.key_fingerprint,
        secret.id,
        secret.name,
        deadline_at=deadline_at,
    )


def terminate_handle(handle: RunpodPodHandle, *, deadline_at: float) -> None:
    if handle.phase == EXACT:
        runpod_pods.delete_pod_for_fingerprint(
            handle.pod_id, handle.key_fingerprint, deadline_at=deadline_at
        )
    elif handle.phase == POD_CREATE_PENDING:
        key = runpod_api._key_for_fingerprint(handle.key_fingerprint)
        matching = runpod_pods.list_pods_for_key(
            key, keep_name=handle.label, deadline_at=deadline_at
        )
        for pod in matching:
            runpod_pods.delete_pod_for_fingerprint(
                pod.id, handle.key_fingerprint, deadline_at=deadline_at
            )
        if runpod_pods.list_pods_for_key(key, keep_name=handle.label, deadline_at=deadline_at):
            raise runpod_api.RunpodApiError(
                f"runpod pending Pod {handle.label} cleanup is unconfirmed"
            )
    _delete_payload_secret(handle, deadline_at=deadline_at)


def launch_payload_pod(
    spec,
    seed: int,
    *,
    serialized_payload: str,
    fingerprint: str,
    data_center_id: str | None,
    network_volume_id: str | None,
    attempt: int = 0,
    on_handle=None,
    cleanup_guard=None,
    deadline_at: float,
    image_name: str | None = None,
    gpu_type_id_override: str | None = None,
    allowed_cuda_versions: tuple[str, ...] | None = None,
    docker_start_cmd: tuple[str, ...] | None = None,
) -> RunpodPodHandle:
    """Launch one exact phase-aware Pod carrying an opaque payload secret."""
    registry_id = (os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID") or "").strip() or None
    intent = _new_secret_intent(
        spec,
        seed,
        attempt,
        fingerprint,
        container_registry_auth_id=registry_id,
        started_ts=time.time(),
        deadline_at=deadline_at,
    )
    current = replace(
        intent,
        image_name=image_name or worker_image_for_gpu(spec.gpu.type),
        gpu_type_id_override=gpu_type_id_override,
        allowed_cuda_versions=allowed_cuda_versions,
        docker_start_cmd=docker_start_cmd,
    )
    intent = current
    uncertain = False
    resources_started = False
    try:
        if on_handle is not None:
            on_handle(intent.to_dict())
        current = _create_payload_secret(intent, serialized_payload, deadline_at=deadline_at)
        resources_started = True
        if on_handle is not None:
            on_handle(current.to_dict())
        current = _pod_create_intent(
            current,
            spec,
            seed,
            data_center_id=data_center_id,
            network_volume_id=network_volume_id,
        )
        if on_handle is not None:
            on_handle(current.to_dict())
        payload = _payload_for_handle(current)
        current = replace(current, phase=POD_CREATE_PENDING)
        if on_handle is not None:
            on_handle(current.to_dict())
        try:
            exact = create_or_adopt_pod(current, payload, deadline_at=deadline_at)
        except (runpod_pods.RunpodMutationAmbiguous, UnreconciledCreateError):
            uncertain = True
            raise
        if on_handle is not None:
            on_handle(exact.to_dict())
        return exact
    except BaseException as original:
        if resources_started and not uncertain:
            if cleanup_guard is not None:
                try:
                    cleanup_guard()
                except BaseException as guard_error:
                    detail = sanitize_diagnostic(guard_error, limit=512)
                    logger.warning("runpod launch rollback guard failed: %s", detail)
                    original.add_note(f"runpod launch rollback guard also failed: {detail}")
            try:
                terminate_handle(current, deadline_at=max(deadline_at, time.time() + 120.0))
            except BaseException as cleanup_error:
                detail = sanitize_diagnostic(cleanup_error, limit=512)
                logger.warning("runpod launch rollback teardown failed: %s", detail)
                original.add_note(f"runpod launch rollback teardown also failed: {detail}")
        raise


def submit_runpod_pod(
    spec,
    seed: int,
    *,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict[str, str] | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> PollResult:
    if spec.gpu.type not in GPU_INFO:
        raise runpod_api.RunpodApiError(
            f"submit_runpod_pod needs a concrete gpu class, got {spec.gpu.type!r}"
        )
    absolute_deadline = require_deadline_at(deadline_at)
    require_create_allowance(absolute_deadline)
    payload = _build_instance_payload(
        spec, seed, attempt, runtime_secrets, source_snapshot, absolute_deadline
    )
    serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    last_capacity: Exception | None = None
    for key in runpod_auth.ordered_keys():
        fingerprint = runpod_api.key_fingerprint(key)
        try:
            candidates = _volume_candidates(spec, fingerprint, deadline_at=absolute_deadline)
        except runpod_api.RunpodApiError as exc:
            last_capacity = exc
            continue
        for data_center_id, network_volume_id in candidates:
            try:
                handle = launch_payload_pod(
                    spec,
                    seed,
                    serialized_payload=serialized_payload,
                    fingerprint=fingerprint,
                    data_center_id=data_center_id,
                    network_volume_id=network_volume_id,
                    attempt=attempt,
                    on_handle=on_handle,
                    deadline_at=absolute_deadline,
                )
            except (runpod_pods.RunpodMutationAmbiguous, UnreconciledCreateError):
                raise
            except runpod_pods.RunpodCapacityError as exc:
                last_capacity = exc
                continue
            if log is not None:
                print(
                    f"launched runpod Pod {handle.pod_id}: {handle.gpu} x{handle.gpu_count} "
                    f"attempt={attempt} seed={seed}",
                    file=log,
                    flush=True,
                )
            reader = heartbeat_reader_for(
                spec,
                **deadline_kwargs(heartbeat_reader_for, absolute_deadline),
            )
            try:
                return poll_runpod_pod(
                    handle,
                    spec,
                    seed,
                    log=log,
                    heartbeat_reader=reader,
                    on_handle=on_handle,
                    **deadline_kwargs(poll_runpod_pod, absolute_deadline),
                )
            finally:
                terminate_handle(handle, deadline_at=max(absolute_deadline, time.time() + 120.0))
    if last_capacity is not None:
        raise last_capacity
    raise runpod_pods.RunpodCapacityError(
        "no configured RunPod account accepted the exact Pod shape"
    )


from flash.providers.runpod.execution.cleanup import (  # noqa: E402,F401
    destroy_run_pods,
    run_pods_remaining,
    sweep_orphan_pods,
)
