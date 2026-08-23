"""Modal Sandbox training lifecycle through shared HF-artifact polling."""

from __future__ import annotations

import base64
import contextlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from flash._internal.diagnostics import sanitize_diagnostic
from flash._internal.logging import get_logger
from flash.providers._lifecycle.deadline import (
    deadline_kwargs,
    remaining_seconds,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers._lifecycle.instance import (
    InstanceJobHandle,
    _instance_capsule,
    _spill_large_spec_to_hf,
    instance_label,
    run_label_prefix,
)
from flash.providers._lifecycle.instance import build_payload as _shared_build_payload
from flash.providers._lifecycle.poll import (
    FIRST_LIVENESS_S,
    LOAD_TIMEOUT_S,
    SETUP_GRACE_S,
    STALL_AFTER_S,
    make_say,
)
from flash.providers._lifecycle.poll_instance import InstancePollAdapter, poll_instance_job
from flash.providers.artifacts.hf import (
    error_artifact_name,
    heartbeat_reader_for,
    make_hf_text_reader,
)
from flash.providers.base import GPU_INFO, PollResult, UnsupportedGpuError, canonical_gpu
from flash.providers.modal import api as modal_api

logger = get_logger(__name__)

_DEAD_STATES = frozenset({"terminated"})
_BOOTSTRAP_PAYLOAD_ENV = "FLASH_MODAL_BOOTSTRAP_PAYLOAD_B64"
_BOOTSTRAP_CAPSULE_ENV = "FLASH_MODAL_BOOTSTRAP_CAPSULE_B64"
_BOOTSTRAP_CAPSULE_SHA_ENV = "FLASH_MODAL_BOOTSTRAP_CAPSULE_SHA256"
_BOOTSTRAP_LAUNCHER = """import base64, hashlib, os, pathlib, sys
root = pathlib.Path('/root/flash')
root.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode(os.environ.pop('FLASH_MODAL_BOOTSTRAP_PAYLOAD_B64'))
capsule = base64.b64decode(os.environ.pop('FLASH_MODAL_BOOTSTRAP_CAPSULE_B64'))
expected = os.environ.pop('FLASH_MODAL_BOOTSTRAP_CAPSULE_SHA256')
if hashlib.sha256(capsule).hexdigest() != expected:
    raise SystemExit('flash: runtime capsule failed verification')
(root / 'payload.json').write_bytes(payload)
path = root / 'capsule.pyz'
path.write_bytes(capsule)
os.execv(sys.executable, [sys.executable, str(path), 'bootstrap'])
"""


@dataclass
class ModalJobHandle(InstanceJobHandle):
    """Persisted Modal Sandbox identity for reattach, cancellation, and realized cost."""

    label: str
    gpu_request: str

    provider: ClassVar[str] = "modal"

    @staticmethod
    def _coerce_instance_id(raw) -> str:
        if not isinstance(raw, str) or not raw:
            raise ValueError("invalid modal sandbox id")
        return raw

    def _extra_to_dict(self) -> dict:
        return {"label": self.label, "gpu_request": self.gpu_request}

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        label = d.get("label")
        gpu_request = d.get("gpu_request")
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(gpu_request, str)
            or not gpu_request
        ):
            raise ValueError("persisted modal provider identity is incomplete")
        return {"label": label, "gpu_request": gpu_request}


def modal_image(gpu: str) -> str:
    """Return the per-SM Flash worker image consumed directly by Modal."""
    from flash.providers._lifecycle.worker import worker_image_for_gpu

    return worker_image_for_gpu(gpu)


def modal_gpu_request(gpu: str, gpu_count: int) -> str:
    """Return Modal's strictly pinned GPU request for one supported shape."""
    info = GPU_INFO.get(gpu)
    if info is None or not info.modal_name:
        raise UnsupportedGpuError(f"modal does not offer managed gpu class {gpu!r}")
    count = int(gpu_count)
    max_count = 4 if gpu == "A10" else 8
    if count not in (1, 2, 4, 8) or count > max_count:
        raise UnsupportedGpuError(f"modal does not offer {count}x {gpu}")
    # modal silently auto-upgrades an unpinned h100 request to h200; `!` makes the class exact, and
    # the post-create nvidia-smi attestation below independently verifies the realized board.
    pinned = f"{info.modal_name}!"
    return pinned if count == 1 else f"{pinned}:{count}"


def build_payload(
    spec,
    seed: int,
    attempt: int,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> dict:
    """Build the shared instance bootstrap payload with ``arm='modal'``."""
    return _shared_build_payload(
        spec,
        seed,
        attempt,
        arm="modal",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
    )


def bootstrap_environment(payload: dict) -> dict[str, str]:
    """Encode the shared payload and verified capsule into Modal's secret environment."""
    payload = _spill_large_spec_to_hf(payload)
    capsule, capsule_sha256 = _instance_capsule()
    return {
        _BOOTSTRAP_PAYLOAD_ENV: base64.b64encode(json.dumps(payload).encode()).decode(),
        _BOOTSTRAP_CAPSULE_ENV: capsule.replace("\n", ""),
        _BOOTSTRAP_CAPSULE_SHA_ENV: capsule_sha256,
    }


def _realized_gpu_class(product_names: list[str], *, requested: str, count: int) -> str:
    """Canonicalize and verify the boards reported by ``nvidia-smi`` inside the Sandbox."""
    if len(product_names) != count:
        raise modal_api.ModalApiError(
            f"Modal realized {len(product_names)} GPU(s) for a {count}-card request"
        )
    realized: list[str] = []
    for product_name in product_names:
        normalized = product_name.strip().lower()
        if normalized in {"a10g", "nvidia a10g", "a10", "nvidia a10"}:
            realized.append("A10")
        else:
            realized.append(canonical_gpu(product_name))
    unique = set(realized)
    if unique != {requested}:
        raise modal_api.ModalApiError(
            f"Modal realized {sorted(unique)!r} for strictly pinned {requested!r}"
        )
    return requested


def _sandbox_tags(run_id: str, label: str) -> dict[str, str]:
    return {
        modal_api.PROVIDER_TAG: "modal",
        modal_api.RUN_TAG: run_label_prefix(run_id),
        modal_api.LABEL_TAG: label,
    }


def deploy_and_submit(
    spec,
    seed: int,
    attempt: int = 0,
    log=None,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> ModalJobHandle:
    """Create one Sandbox, attest its exact GPU shape, and return its persisted handle."""
    from flash.core.spec import gpu_count_of
    from flash.providers.modal.pricing import hourly_rate

    absolute_deadline = require_deadline_at(deadline_at)
    count = gpu_count_of(spec)
    gpu_request = modal_gpu_request(spec.gpu.type, count)
    label = instance_label(spec.run_id, seed, attempt)
    payload = build_payload(
        spec,
        seed,
        attempt,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        **deadline_kwargs(build_payload, absolute_deadline),
    )
    environment = bootstrap_environment(payload)
    require_create_allowance(absolute_deadline)
    started_ts = time.time()
    instance_id = modal_api.create_sandbox(
        "python",
        "-c",
        _BOOTSTRAP_LAUNCHER,
        image=modal_image(spec.gpu.type),
        gpu=gpu_request,
        env=environment,
        timeout=max(1, math.ceil(remaining_seconds(absolute_deadline))),
        name=label,
        tags=_sandbox_tags(spec.run_id, label),
    )
    try:
        realized = _realized_gpu_class(
            modal_api.sandbox_gpu_names(instance_id),
            requested=spec.gpu.type,
            count=count,
        )
        with contextlib.suppress(Exception):
            make_say(log)(
                f"created Modal Sandbox {instance_id}: {realized} x{count} "
                f"${hourly_rate(realized) * count:.2f}/hr attempt={attempt} seed={seed}"
            )
        return ModalJobHandle(
            instance_id=instance_id,
            label=label,
            gpu_request=gpu_request,
            gpu=realized,
            hourly_usd=hourly_rate(realized) * count,
            attempt=attempt,
            started_ts=started_ts,
        )
    except BaseException:
        with contextlib.suppress(BaseException):
            modal_api.terminate_sandbox(instance_id)
        raise


_make_hf_file_reader = make_hf_text_reader


def _failure_detail(
    hf_repo: str, prefix: str, phase: str, marker: dict | None, attempt: int
) -> str:
    parts: list[str] = []
    if marker and marker.get("error"):
        parts.append(sanitize_diagnostic(marker["error"], limit=4096))
    err_name = error_artifact_name(phase, attempt)
    content = _make_hf_file_reader(hf_repo, f"{prefix}/{err_name}")(force=True)
    if content:
        parts.append(f"--- {err_name} ---\n{sanitize_diagnostic(content[-4096:], limit=4096)}")
    return "\n".join(parts) or "modal worker terminated without a strict terminal marker"


def poll_modal_job(
    handle: ModalJobHandle,
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
    """Poll Modal state and the shared HF artifacts to a terminal result."""
    absolute_deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    err_name = error_artifact_name(spec.phase, handle.attempt)

    def stamp_cost_and_notes(metrics, *, end_ts, launch_ts) -> None:
        instance_wall_s = max(0.0, end_ts - launch_ts)
        metrics["cost_usd"] = round(instance_wall_s / 3600.0 * handle.hourly_usd, 6)
        notes = metrics.get("notes") if isinstance(metrics.get("notes"), dict) else {}
        notes.update(
            {
                "provider": "modal",
                "modal_rate_usd_hr": handle.hourly_usd,
                "modal_gpu": handle.gpu,
                "modal_gpu_request": handle.gpu_request,
                "modal_sandbox_wall_seconds": round(instance_wall_s, 3),
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
            f"{prefix}/modal_attempt{handle.attempt}.json",
            min_interval_s=60.0,
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        metrics_reader=_make_hf_file_reader(
            hf_repo,
            f"{prefix}/metrics.json",
            **deadline_kwargs(_make_hf_file_reader, absolute_deadline),
        ),
        fetch_instance=lambda: modal_api.sandbox_status(handle.instance_id),
        poll_error_exceptions=(modal_api.ModalApiError,),
        status_field="status",
        running_status="running",
        dead_states=_DEAD_STATES,
        missing_dead_threshold=3,
        early_liveness_alive=lambda: modal_api.sandbox_exec_succeeds(handle.instance_id),
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
            f"sandbox stuck in '{status}' for {int(elapsed)}s (never became running)"
        ),
        first_liveness_detail=lambda elapsed, fl: (
            f"no worker heartbeat and no successful sandbox exec for {int(elapsed)}s after startup "
            f"(limit {int(fl)}s)"
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


def submit_run_modal(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> PollResult:
    """Create, persist, poll, and always terminate one Modal training Sandbox."""
    if spec.gpu.type not in GPU_INFO or not GPU_INFO[spec.gpu.type].modal_name:
        raise modal_api.ModalApiError(
            f"submit_run_modal needs a concrete Modal gpu class, got {spec.gpu.type!r}"
        )
    absolute_deadline = require_deadline_at(deadline_at)
    handle = deploy_and_submit(
        spec,
        seed,
        attempt=attempt,
        log=log,
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        **deadline_kwargs(deploy_and_submit, absolute_deadline),
    )
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        reader = heartbeat_reader_for(
            spec,
            **deadline_kwargs(heartbeat_reader_for, absolute_deadline),
        )
        return poll_modal_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            **deadline_kwargs(poll_modal_job, absolute_deadline),
        )
    finally:
        modal_api.terminate_sandbox(handle.instance_id)


def cancel(remote: dict) -> None:
    """Terminate the exact Sandbox in a persisted handle."""
    instance_id = remote.get("instance_id")
    if instance_id:
        modal_api.terminate_sandbox(str(instance_id))


def terminate_run_sandboxes(run_id: str) -> list[str]:
    """Terminate every active Modal Sandbox carrying one run's exact tag."""
    if not run_id:
        return []
    destroyed: list[str] = []
    for sandbox in modal_api.list_sandboxes(
        tags={modal_api.PROVIDER_TAG: "modal", modal_api.RUN_TAG: run_label_prefix(run_id)}
    ):
        instance_id = str(sandbox["id"])
        try:
            modal_api.terminate_sandbox(instance_id)
        except Exception as exc:
            logger.warning(
                "modal sandbox teardown failed for %s: %s", instance_id, type(exc).__name__
            )
            continue
        destroyed.append(instance_id)
    return destroyed


def run_instances_remaining(run_id: str) -> list[str]:
    """Return exact run-tagged Sandbox ids; listing failures raise instead of reporting clear."""
    if not run_id:
        return []
    return [
        str(sandbox["id"])
        for sandbox in modal_api.list_sandboxes(
            tags={modal_api.PROVIDER_TAG: "modal", modal_api.RUN_TAG: run_label_prefix(run_id)}
        )
    ]


def sweep_orphans(
    active_labels: set[str] | Callable[[], set[str]] | None = None,
    known_labels: set[str] | Callable[[], set[str]] | None = None,
) -> list[str]:
    """Terminate inactive Flash Sandboxes, optionally scoped to this control plane's known runs."""
    try:
        sandboxes = modal_api.list_sandboxes(tags={modal_api.PROVIDER_TAG: "modal"})
    except Exception as exc:
        logger.warning("modal orphan sweep skipped: %s", type(exc).__name__)
        return []
    try:
        labels = active_labels() if callable(active_labels) else active_labels
        known = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        logger.warning(
            "modal orphan sweep skipped; could not resolve run sets: %s", type(exc).__name__
        )
        return []
    active = {run_label_prefix(label) for label in (labels or set())}
    known_prefixes = (
        None if known_labels is None else {run_label_prefix(label) for label in (known or set())}
    )
    destroyed: list[str] = []
    for sandbox in sandboxes:
        run_tag = str((sandbox.get("tags") or {}).get(modal_api.RUN_TAG) or "")
        if not run_tag or run_tag in active:
            continue
        if known_prefixes is not None and run_tag not in known_prefixes:
            continue
        instance_id = str(sandbox["id"])
        try:
            modal_api.terminate_sandbox(instance_id)
        except Exception:
            continue
        destroyed.append(instance_id)
        logger.warning("destroyed orphaned modal sandbox %s (run tag %s)", instance_id, run_tag)
    return destroyed
