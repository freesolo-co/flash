"""Reading and writing a run's persisted status record.

`~/.flash/runs/<run_id>.json` is the only durable record a run has, so everything here is written to
survive a partial read: unknown keys are dropped rather than fatal, `list_run_ids` never parses JSON
so one corrupt file cannot hide the rest, and `_update` is a sticky compare-and-set that refuses to
move a run back out of a terminal state. `effective_spec_from_status` is the recovery half -- it
re-derives the private worker spec and refuses it if the stored preparation digest no longer checks
out.

Split out of `flash.runner` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import replace

from flash.core.catalog import validate_model_for_algorithm
from flash.core.spec import JobSpec
from flash.core.spec_persistence import validate_persisted_spec_envelope
from flash.runner.lifecycle import preparation, reporting, state
from flash.runner.lifecycle.state import RunStatus

# every other collaborator is reached through `runner.` rather than bound here. `RUNS_DIR`,
# `get_status`, `_update`, `effective_spec_from_status`, `_gpu_rate`, `_internal_spec_from_status`
# and `_report_status` are all patched as attributes of `flash.runner` by the tests, and binding any
# of them by value would capture the original before the patch lands. the rest go the same way for
# uniformity and to stay independent of the order the parent imports its submodules in.


def _runstatus_from_json(d: dict) -> RunStatus:
    # tolerant load: drop unknown keys before constructing runstatus. source_snapshot is different:
    # malformed persisted identity must fail closed rather than be treated as a legacy absence.
    values = {k: v for k, v in d.items() if k in RunStatus.__dataclass_fields__}
    if d.get("source_snapshot") is not None:
        from flash.snapshot.archive import parse_descriptor

        values["source_snapshot"] = parse_descriptor(d["source_snapshot"]).to_dict()
    return RunStatus(**values)


def _load_status_json(run_id: str) -> dict:
    path = state.runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"invalid stored run status for {run_id}")
    return value


def get_status(run_id: str) -> RunStatus:
    return _runstatus_from_json(_load_status_json(run_id))


def source_snapshot_from_status(status: RunStatus, *, required: bool = False) -> dict | None:
    """Return the strict persisted descriptor without ever reconstructing source identity."""
    raw = status.source_snapshot
    if raw is None:
        if required:
            raise RuntimeError(
                "managed source identity is unavailable; descriptor-less attempts cannot be replaced"
            )
        return None
    from flash.snapshot.archive import parse_descriptor

    return parse_descriptor(raw).to_dict()


def effective_spec_from_status(status: RunStatus, *, verify_source: bool = False) -> JobSpec:
    """Load the private prepared worker spec, optionally revalidating its source artifact."""
    managed_revision_keys = {
        "model_revision",
        "model_revision_auto",
        "model_revision_force_pin",
    }
    leaked_revision_keys = sorted(managed_revision_keys & set(status.spec))
    if leaked_revision_keys:
        raise ValueError(
            "persisted public spec contains platform-managed model revision key(s): "
            + ", ".join(leaked_revision_keys)
        )
    public_spec = JobSpec.from_dict(status.spec)
    validate_model_for_algorithm(public_spec.model, public_spec.algorithm)
    snapshot = status.effective_preparation
    if not isinstance(snapshot, dict):
        if public_spec.train.init_from_adapter:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} cannot be recovered "
                "because its original preparation snapshot is unavailable"
            )
        return public_spec
    raw_worker = snapshot.get("worker_spec")
    if not isinstance(raw_worker, dict):
        raise ValueError("persisted effective preparation is malformed")
    worker_spec = JobSpec.from_dict(raw_worker)
    validate_model_for_algorithm(worker_spec.model, worker_spec.algorithm)
    preparation._validate_effective_spec(public_spec, worker_spec)
    expected = snapshot.get("adapter_identity")
    stored_digest = snapshot.get("preparation_digest")
    if stored_digest is not None:
        validate_persisted_spec_envelope(snapshot)
    has_workload_profile = bool(
        worker_spec.workload_profile_input_digest or worker_spec.workload_profile
    )
    # a runner-managed revision exists only in the worker half, so bind it and its provenance marker
    # to the preparation digest. a plain grpo or opd run otherwise reaches neither digest branch.
    if has_workload_profile and snapshot.get("workload_profile") != (
        worker_spec.workload_profile or None
    ):
        raise ValueError("persisted workload profile does not match the worker spec")
    # `gpu_count_auto` is deliberately NOT a trigger here, unlike `model_revision_auto`. the digest
    # covers the whole public spec including `gpu.type`, which the allocator legitimately rewrites
    # onto the stored status when a run is provisioned -- so gating on the marker made the digest
    # reject ordinary provisioned runs at deploy. Measured: two specs differing only in whether
    # gpu.count was authored deployed differently, the auto-sized one failing integrity validation
    # (tests/test_server_api.py::test_deploy_ignores_stored_training_gpu). Since an omitted count is
    # the DEFAULT, that is nearly every run. The marker's integrity does not need this trigger: it
    # is bounded by `_validate_effective_spec`, which caps an auto-sized count at
    # MAX_COMBINATION_CARDS, and unlike `model_revision_auto` it cannot relax a deploy-time
    # rejection -- a forged marker only widens the allocator's ceiling, which the VRAM fit check and
    # the geometry cap still constrain.
    if (has_workload_profile or worker_spec.model_revision_auto) and (
        not isinstance(stored_digest, str)
        or stored_digest
        != preparation._preparation_digest(
            public_spec, worker_spec, expected, stored_public=status.spec
        )
    ):
        raise ValueError("persisted effective preparation failed integrity validation")
    if public_spec.train.init_from_adapter:
        if not isinstance(expected, dict) or not expected.get("digest"):
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} cannot be recovered "
                "because its original artifact identity is unavailable"
            )
        if not isinstance(stored_digest, str) or stored_digest != preparation._preparation_digest(
            public_spec, worker_spec, expected, stored_public=status.spec
        ):
            raise ValueError("persisted effective preparation failed integrity validation")
    if verify_source and public_spec.train.init_from_adapter:
        try:
            from flash.adapters.lora_rank import (
                adapter_artifact_identity,
                inspect_adapter_config,
                load_hf_adapter_config,
            )

            revision = worker_spec.train.init_from_adapter_revision
            config = load_hf_adapter_config(
                worker_spec.train.init_from_adapter,
                os.environ.get("HF_TOKEN"),
                revision,
            )
            metadata = inspect_adapter_config(
                config,
                source="pinned warm-start adapter",
                target_model=worker_spec.model,
            )
            if (
                metadata.rank != worker_spec.train.lora_rank
                or metadata.alpha != worker_spec.train.lora_alpha
            ):
                raise ValueError("prepared adapter topology changed")
            current = adapter_artifact_identity(
                worker_spec.train.init_from_adapter,
                config,
                os.environ.get("HF_TOKEN"),
                revision,
            ).to_dict()
        except Exception as exc:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} could not be revalidated"
            ) from exc
        if current != expected:
            raise ValueError(
                f"warm-start source {public_spec.train.init_from_adapter!r} changed after submission; "
                "recovery was refused"
            )
    return worker_spec


def reallocation_spec_from_status(status: RunStatus, *, verify_source: bool = False) -> JobSpec:
    """Effective worker spec for RE-ALLOCATING a recovered run.

    Feeding that straight back to allocate() would re-enter recovery with the prior attempt's answer
    as its input -- hard-pinning an originally-unpinned run to one class (blocking OOM escalation
    and retries on other providers/classes), and lowering the ceiling to the one shape that already
    failed, so a run authored for up to 4 cards can never again be offered a 4-card shape. gpu.count
    is a CEILING, so restoring it re-widens the search rather than forcing a size.
    """
    worker_spec = effective_spec_from_status(status, verify_source=verify_source)
    public_gpu = JobSpec.from_dict(status.spec).gpu
    # `gpu_count_auto` needs no restoring here: it is provenance, so the worker half carries it
    # verbatim through allocation. The public half cannot supply it -- to_dict strips the marker and
    # keeps the placeholder count=1, making an auto-sized run's public spec byte-identical to an
    # authored single-card pin -- which is exactly why the worker half must keep it.
    # the fallbacks restore with the class they qualify: an ordered pin whose head was rewritten to
    # the allocated class would otherwise come back from recovery as a bare pin on whichever class
    # the failed attempt happened to rent -- the one shape already known not to be available.
    if (
        worker_spec.gpu.type == public_gpu.type
        and worker_spec.gpu.count == public_gpu.count
        and worker_spec.gpu.type_fallbacks == public_gpu.type_fallbacks
    ):
        return worker_spec
    restored = worker_spec.to_internal_dict()
    restored["gpu"] = {
        **restored["gpu"],
        "type": public_gpu.type,
        "count": public_gpu.count,
        "type_fallbacks": public_gpu.type_fallbacks,
    }
    return JobSpec.from_dict(restored)


def list_runs() -> list[RunStatus]:
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(state.RUNS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(state.RUNS_DIR, name)) as f:
                runs.append(_runstatus_from_json(json.load(f)))
    return runs


def list_run_ids() -> list[str]:
    """Run ids by filename only (no JSON parse) so a corrupt record can't break the listing."""
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    return [
        name[: -len(".json")]
        for name in sorted(os.listdir(state.RUNS_DIR))
        if name.endswith(".json")
    ]


def get_logs(run_id: str) -> str:
    log_path = state.runs_file_path(run_id, ".log")
    if not os.path.exists(log_path):
        return ""
    with open(log_path) as f:
        return f.read()


def _current_attempt(status: RunStatus):
    from flash.runner.lifecycle.protocol import AttemptRecord

    return AttemptRecord.from_dict(status.attempt)


def _record_projection(
    run_id: str,
    field: str,
    value: dict,
    *,
    attempt_id: int,
    fence: int,
    resource_identity: tuple | None = None,
) -> bool:
    """Persist a projection only for the exact current attempt, fence, and resource."""
    from flash.runner.accounting.reconciliation import _remote_resource_identity
    from flash.runner.lifecycle.protocol import bounded_json

    with state._status_guard(run_id):
        status = get_status(run_id)
        attempt = _current_attempt(status)
        if attempt.attempt_id != attempt_id or attempt.fence != fence:
            return False
        if (
            resource_identity is not None
            and _remote_resource_identity(status.remote) != resource_identity
        ):
            return False
        setattr(status, field, bounded_json(value))
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)
    return True


def record_attempt_handle(
    run_id: str,
    handle: dict,
    *,
    attempt_id: int,
    fence: int,
) -> bool:
    """Bind the exact provider resource to the current fenced attempt."""
    from flash.runner.lifecycle.protocol import bounded_json

    with state._status_guard(run_id):
        status = get_status(run_id)
        attempt = _current_attempt(status)
        if attempt.attempt_id != attempt_id or attempt.fence != fence:
            return False
        if status.state in state.TERMINAL_STATES:
            return False
        resource = bounded_json(handle)
        status.attempt = replace(
            attempt,
            state="active",
            provider=handle.get("provider"),
            resource=resource,
        ).to_dict()
        status.remote = resource
        status.state = "running"
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)
    return True


def _same_progress_identity(current: dict, incoming: dict) -> bool:
    current_receipt = current.get("receipt")
    incoming_receipt = incoming.get("receipt")
    if not isinstance(current_receipt, dict) or not isinstance(incoming_receipt, dict):
        return False
    current_path = current_receipt.get("path")
    current_digest = current_receipt.get("digest")
    return (
        isinstance(current_path, str)
        and bool(current_path)
        and isinstance(current_digest, str)
        and len(current_digest) == 64
        and incoming_receipt.get("path") == current_path
        and incoming_receipt.get("digest") == current_digest
    )


def record_progress(run_id: str, value: dict, *, attempt_id: int, fence: int) -> bool:
    """Persist only monotonic progress for the exact current fenced attempt."""
    from flash.runner.lifecycle.protocol import bounded_json

    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = get_status(run_id)
        attempt = _current_attempt(status)
        if attempt.attempt_id != attempt_id or attempt.fence != fence:
            return False
        incoming = bounded_json(value)
        if incoming.get("attempt_id") != attempt_id or incoming.get("fence") != fence:
            return False
        current = status.progress if isinstance(status.progress, dict) else None
        if current is not None:
            current_sequence = current.get("sequence")
            incoming_sequence = incoming.get("sequence")
            if type(current_sequence) is not int or type(incoming_sequence) is not int:
                return False
            if incoming_sequence < current_sequence:
                return False
            if incoming_sequence == current_sequence:
                if not _same_progress_identity(current, incoming):
                    return False
                merged = dict(current)
                if "observed_at" in incoming:
                    merged["observed_at"] = incoming["observed_at"]
                current_receipt = dict(current["receipt"])
                incoming_receipt = incoming["receipt"]
                if "revision" in incoming_receipt:
                    current_receipt["revision"] = incoming_receipt["revision"]
                merged["receipt"] = current_receipt
                incoming = merged
            elif incoming.get("completed_steps", 0) < current.get("completed_steps", 0):
                return False
        status.progress = incoming
        status.updated_at = time.time()
        state._save_status_unlocked(status)
        report_status = status
    reporting._report_status(report_status)
    return True


def record_resource(
    run_id: str,
    value: dict,
    *,
    attempt_id: int,
    fence: int,
    resource_identity: tuple | None = None,
) -> bool:
    return _record_projection(
        run_id,
        "resource",
        value,
        attempt_id=attempt_id,
        fence=fence,
        resource_identity=resource_identity,
    )


def record_result(run_id: str, value: dict, *, attempt_id: int, fence: int) -> bool:
    return _record_projection(run_id, "result", value, attempt_id=attempt_id, fence=fence)


def validate_terminal_source_metrics(
    status: RunStatus,
    metrics: dict,
    *,
    expected_attempt: int | None = None,
    expected_fence: int | None = None,
) -> tuple[dict, int | None]:
    """Require trusted attempt-bound evidence for runs carrying a source descriptor."""
    if not isinstance(metrics, dict):
        raise RuntimeError("terminal metrics are invalid")
    from flash.snapshot.archive import (
        PUBLIC_PROVENANCE_KEY,
        TERMINAL_ATTESTATION_KEY,
        safe_public_projection,
        validate_attestation,
    )

    sanitized = dict(metrics)
    raw_attestation = sanitized.pop(TERMINAL_ATTESTATION_KEY, None)
    sanitized.pop(PUBLIC_PROVENANCE_KEY, None)
    descriptor = source_snapshot_from_status(status)
    if descriptor is None:
        return sanitized, None
    attempt_record = _current_attempt(status)
    if expected_attempt is None:
        expected_attempt = attempt_record.attempt_id
    if expected_fence is None:
        expected_fence = attempt_record.fence
    if expected_attempt != attempt_record.attempt_id or expected_fence != attempt_record.fence:
        raise RuntimeError("terminal evidence does not match the current fenced attempt")
    validate_attestation(
        raw_attestation,
        descriptor,
        run_id=status.run_id,
        attempt=expected_attempt,
        fence=expected_fence,
    )
    sanitized[PUBLIC_PROVENANCE_KEY] = safe_public_projection(
        descriptor,
        verified_attempt=expected_attempt,
    )
    return sanitized, expected_attempt


def _persist_metrics(spec: JobSpec, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id> and return the customer training cost.

    The run id keeps concurrent/sequential runs of the same phase from
    overwriting each other's artifacts. ``metrics["wall_seconds"]`` is the worker's training-loop
    wall time; setup/cold-start is reported separately and is not included here."""
    from flash.engine.result.accounting import sanitize_worker_metrics

    metrics = sanitize_worker_metrics(metrics)
    dest = state.artifacts_dir(spec)
    os.makedirs(dest, exist_ok=True)
    # Use allocated_gpu (worker-stamped) not spec.gpu.type; policy GPUs can be reallocated.
    gpu_type = metrics.get("allocated_gpu") or spec.gpu.type
    # the substrate that actually billed the run; empty on a record predating the stamp, in which
    # case _gpu_rate prices off whichever configured provider offers the class.
    provider = str(metrics.get("allocated_provider") or "")
    from flash.runner.accounting.costs import _gpu_rate

    rate = _gpu_rate(gpu_type, provider)
    # `hourly_rate` is per CARD, so a sharded run costs the wall times the rate times the number of
    # cards it actually occupied. `allocated_gpu_count` is worker/lifecycle-stamped for the same
    # reason `allocated_gpu` is: the spec's gpu.count is only a ceiling and allocation may pick
    # fewer. Absent on records predating the stamp, where one card is the correct reading.
    gpu_count = max(1, int(metrics.get("allocated_gpu_count") or 1))
    cost = metrics.get("cost_usd")
    if cost:
        cost = float(cost or 0.0)
    else:
        wall = float(metrics.get("wall_seconds") or 0.0)
        cost = wall / 3600.0 * rate * gpu_count
        metrics = {**metrics, "cost_usd": cost}
        metrics.setdefault("notes", {})
        if isinstance(metrics["notes"], dict):
            metrics["notes"]["provider"] = provider or "unknown"
            metrics["notes"]["gpu_rate_usd_hr"] = rate
            metrics["notes"]["gpu"] = gpu_type
            metrics["notes"]["gpu_count"] = gpu_count
    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with contextlib.suppress(Exception):
        from flash.server.domain.registry.runs import record_training_checkpoint

        record_training_checkpoint(spec=spec, metrics=metrics, artifact_path=dest)
    return float(cost)


def _update(run_id: str, new_state: str, *, allow_from_terminal: bool = False, **updates) -> bool:
    """Atomically transition run state with terminal-stickiness. Returns False if rejected.

    Returns ``True`` if the transition was applied, ``False`` if it was rejected because the run was
    already in a terminal state (the sticky compare-and-set below). Callers that gate PAID work on a
    transition (e.g. the recovery path resuming ``_run_training``) must check this return so a run
    concurrently flipped terminal does not get resumed.
    """
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = get_status(run_id)
        if (
            status.state in state.TERMINAL_STATES
            and new_state != status.state
            and not allow_from_terminal
        ):
            return False
        was_terminal = status.state in state.TERMINAL_STATES
        status.state = new_state
        status.updated_at = time.time()
        if not was_terminal and new_state in state.TERMINAL_STATES and status.finished_at is None:
            status.finished_at = status.updated_at
        for key, value in updates.items():
            setattr(status, key, value)
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True
