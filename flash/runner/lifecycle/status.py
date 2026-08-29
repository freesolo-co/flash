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
import tempfile
import time
from collections.abc import Sequence

from flash.core.catalog import validate_model_for_algorithm
from flash.core.spec import JobSpec
from flash.core.spec_persistence import validate_persisted_spec_envelope
from flash.runner.lifecycle import attempts, preparation, reporting, state
from flash.runner.lifecycle.salvage import (
    QuarantineWriteFailed,
    _describes_run,
    _durable_teardown_intent,
    _salvage_corrupt_record,
    _stored_reclaim_types,
    has_teardown_address,
    reclaimable_gpu_types,
)
from flash.runner.lifecycle.state import RunStatus
from flash.snapshot.archive import SourceSnapshotError

# `salvage` holds the pure half of quarantine -- what a corrupt record still says and what address
# survives it -- while this module owns the lock, the atomic write, and the terminal transition. the
# names above are used here AND re-exported by that import: `runtime.py` and `attach.py` bind
# `has_teardown_address`, `reclaimable_gpu_types`, and `QuarantineWriteFailed` from this module.

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


def _validate_recovery_status(run_id: str, status: RunStatus) -> RunStatus:
    if status.run_id != run_id:
        raise ValueError(f"stored run status id does not match {run_id}")
    if not isinstance(status.state, str):
        raise TypeError(f"stored run status state must be a string for {run_id}")
    recorded = status.submitted_instance_providers
    if recorded is not None and not isinstance(recorded, list):
        # `_confirm_run_clear` iterates this field as `{str(name) for name in recorded_raw}` with no
        # guard of its own, and its caller at the second `_resubmit_recovered_runs` site is the one
        # that is not wrapped -- so a stored `7` raises `TypeError` out of `recover_runs()`, which
        # the lifespan does not suppress, aborting startup for every other run. Rejecting here
        # routes it to quarantine like any other undecodable field instead of normalizing it: the
        # field is the fail-CLOSED phantom guard, and silently coercing an unreadable value to `[]`
        # would read as "no instance provider could have taken the lost create" and clear the run
        # for resubmit, which is exactly the second billing worker the guard exists to prevent.
        raise TypeError(f"stored submitted instance providers must be a list for {run_id}")
    return status


# a storage blip must not strand a run for the whole process lifetime. `recover_runs()` runs once,
# at lifespan startup, and nothing else enumerates the durable records -- a row skipped for an
# unreadable status is never reconsidered, while its worker keeps billing. so re-read a bounded
# number of times before giving up, and only for the errors that say the STORAGE was unreadable
# (`PermissionError`, `ESTALE`, `EIO`). undecodable BYTES are quarantine's job and re-reading them
# just repeats the same failure, and `FileNotFoundError` -- an `OSError` subclass, so its clause
# comes first -- means the record is gone rather than briefly unavailable.
_RECOVERY_READ_ATTEMPTS = 3
_RECOVERY_READ_BACKOFF_S = 0.2


def _get_recovery_status(run_id: str) -> RunStatus:
    """The validated record, retrying only a transient storage failure."""
    delay = _RECOVERY_READ_BACKOFF_S
    for _ in range(_RECOVERY_READ_ATTEMPTS - 1):
        try:
            return _validate_recovery_status(run_id, get_status(run_id))
        except FileNotFoundError:
            raise
        except OSError:
            time.sleep(delay)
            delay *= 2
    # the last attempt is unguarded so its failure reaches the caller as itself.
    return _validate_recovery_status(run_id, get_status(run_id))


def fail_with_cleanup_intent(run_id: str, detail: str) -> RunStatus | None:
    """Terminalize an unrecoverable run and record what it still owes, in ONE durable write.

    Recovery marks a run `failed` and then has to release whatever it rented. Those are two halves
    of a single fact, and splitting them across writes is what leaves an unreachable record: the
    terminal write is exactly what drops the run from `_RECOVERABLE`, so a crash before the intent
    lands means no later boot revisits it -- `_classify_recoverable_runs` skips it and RunPod's
    `sweep_orphans` is a no-op -- while its worker bills on. Writing intent first only inverts the
    window. So both go into the same `_save_status_unlocked` under the same guard, the way
    quarantine already folds them into its one atomic envelope.

    Which intent applies follows from the record, not from the caller: a persisted handle becomes a
    drainable `cleanup_remotes` record, and a handle-less run whose only address is its derived
    endpoint name gets the reclaim marker instead. Both can be absent, and then this is just the
    terminal write.

    Returns the persisted status so the caller can dispatch teardown from it, or `None` when the
    sticky compare-and-set refused the transition -- an already-settled run keeps its state, and
    nothing is written.
    """
    with state._status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        if not _apply_transition(
            status, "failed", allow_from_terminal=False, updates={"error": detail}
        ):
            return None
        stored = raw.get(state._CLEANUP_REMOTES_KEY)
        existing = list(stored) if isinstance(stored, list) else []
        records = _durable_teardown_intent(status.remote, existing)
        cleanup = (
            records if records is not None and records != existing else state._PRIVATE_VALUE_UNSET
        )
        reclaim: object = state._PRIVATE_VALUE_UNSET
        if not has_teardown_address(status.remote) and not _stored_reclaim_types(raw):
            gpu_types = reclaimable_gpu_types(raw)
            if gpu_types:
                reclaim = list(gpu_types)
        state._save_status_unlocked(
            status, _cleanup_remotes=cleanup, _endpoint_reclaim_pending=reclaim
        )
    reporting._report_status(status)
    return status


def clear_endpoint_reclaim_intent(run_id: str, confirmed: Sequence[str]) -> None:
    """Retire the classes a reclaim just confirmed gone, keeping any the marker has since gained.

    The reclaim reads the marker, calls the provider unlocked -- deletes retry and back off, so this
    is long -- and only then clears. A second quarantine of the same run inside that window extends
    the marker with a class this reclaim never terminated, and dropping the key wholesale would
    erase it unchecked: the endpoint bills with nothing left naming it. Subtracting only what the
    provider confirmed leaves that new class marked, so the next `_startup_cleanup_bg` reclaims it.
    """
    retire = set(confirmed)
    with state._status_guard(run_id):
        raw = _load_status_json(run_id)
        if state._ENDPOINT_RECLAIM_KEY not in raw:
            return
        remaining = [entry for entry in _stored_reclaim_types(raw) if entry not in retire]
        state._save_status_unlocked(
            _runstatus_from_json(raw), _endpoint_reclaim_pending=remaining or None
        )


def endpoint_reclaim_types(run_id: str) -> tuple[str, ...]:
    """Classes this run still owes a name-derived endpoint reclaim under, if any."""
    with contextlib.suppress(Exception):
        return _stored_reclaim_types(_load_status_json(run_id))
    return ()


# what one unreadable record can raise without implicating any other run. `OSError` is here because
# `_load_status_json` only converts a missing file into `FileNotFoundError`; every other read
# failure -- `PermissionError`, `ESTALE`, `EIO` -- propagates from the bare `open()`/`json.load()`
# beneath it, and escaping this tuple would abort the whole classification pass before later
# runs are reattached or resubmitted. `FileNotFoundError` is a subclass, so every handler for
# these must order its `except FileNotFoundError` clause first; both call sites in this module do.
_RECOVERY_RECORD_ERRORS = (
    UnicodeDecodeError,
    ValueError,
    TypeError,
    RecursionError,
    OSError,
    SourceSnapshotError,
)


def _quarantine_corrupt_status(run_id: str, detail: str) -> tuple[RunStatus | None, bool]:
    """Preserve a still-unreadable status record and atomically install a failed envelope."""
    path = state.runs_file_path(run_id, ".json")
    quarantine_path = state.runs_file_path(run_id, f".json.corrupt-{time.time_ns()}")
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    with state._status_guard(run_id):
        try:
            repaired = _get_recovery_status(run_id)
        except FileNotFoundError:
            return None, False
        except _RECOVERY_RECORD_ERRORS:
            pass
        else:
            return repaired, False
        try:
            raw = _load_status_json(run_id)
        except FileNotFoundError:
            return None, False
        except OSError:
            # the bytes were never read, so nothing about this record is known to be malformed.
            # `PermissionError`, `ESTALE` and `EIO` all arrive here, and they say the storage is
            # unreadable right now, not that the content is corrupt. quarantining anyway would
            # publish `spec = {}` and `remote = None` over a record whose handle was never seen and
            # terminalize a possibly-live run, destroying the only address its worker has. leave the
            # durable bytes untouched and report no quarantine: the caller skips this one row for
            # this boot and a later one reads it once the storage answers again.
            return None, False
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
            raw = None
        else:
            # a read that only failed transiently is not a corrupt record. the recheck above reports
            # `ESTALE` and `EIO` through the same `OSError` arm it uses for undecodable bytes, so
            # this third read succeeding proves nothing on its own -- quarantining on it would
            # rewrite a live `running` record to `failed` and tear its remote down because a stale
            # handle cleared. decode and validate the bytes with exactly what the recheck used, and
            # treat only what still fails as corrupt.
            try:
                recovered = _validate_recovery_status(run_id, _runstatus_from_json(raw))
            except _RECOVERY_RECORD_ERRORS:
                pass
            else:
                return recovered, False
        failed, cleanup_remotes = _salvage_corrupt_record(run_id, raw, detail, time.time())
        data = state._status_storage_dict(failed)
        cleanup_remotes = _durable_teardown_intent(failed.remote, cleanup_remotes)
        if cleanup_remotes is not None:
            data[state._CLEANUP_REMOTES_KEY] = cleanup_remotes
        # this write replaces the record wholesale, so the endpoint address it is about to destroy
        # has to be lifted into the envelope now or nothing can ever reach that endpoint again. two
        # sources feed one marker: classes an earlier boot already froze, and classes the corrupt
        # bytes themselves still name. `_teardown_failed_recovery` cannot do this afterwards -- it
        # reads the envelope, whose `spec` is `{}` -- and no later boot revisits a terminal run, so
        # a handle-less endpoint left unaddressed here bills untouched.
        #
        # Folded into the SAME atomic write as the terminal state, not persisted beside it. The two
        # are one fact: this run is failed AND still owes an endpoint. Split across two writes, a
        # crash between them leaves exactly the unreachable terminal record this marker exists to
        # prevent, and quarantine's whole contract is that the envelope replaces the record in one
        # `os.replace`.
        #
        # Skipped when an ADDRESSABLE handle survived salvage: `cleanup_remotes` above already
        # addresses that worker, and a marker cleared only by a confirmed name-derived delete would
        # otherwise outlive the teardown that actually released it. A truthy but unaddressable
        # remnant is not that -- the drain declines it -- so it must not suppress the marker.
        if (
            raw is not None
            and _describes_run(run_id, raw)
            and not has_teardown_address(failed.remote)
        ):
            reclaim = tuple(dict.fromkeys(_stored_reclaim_types(raw) + reclaimable_gpu_types(raw)))
            if reclaim:
                data[state._ENDPOINT_RECLAIM_KEY] = list(reclaim)
        fd, tmp = tempfile.mkstemp(dir=state.RUNS_DIR, prefix=f"{run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(data, file, indent=2, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.link(path, quarantine_path)
            dir_fd = os.open(state.RUNS_DIR, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
                os.replace(tmp, path)
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            return failed, True
        except Exception as exc:
            # the salvage is the only thing that ever knew this record's address, and it dies with
            # this frame unless it escapes: the envelope was not installed, so the classes come from
            # the corrupt bytes, which the caller cannot re-read (the same failing storage) and the
            # envelope no longer holds. Carry it out so the caller can still tear the worker down.
            raise QuarantineWriteFailed(
                failed, data.get(state._ENDPOINT_RECLAIM_KEY) or ()
            ) from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)


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


_STATUS_LIST_LIMIT = 16
_STATUS_METRICS_HISTORY_LIMIT = 1024


def _sanitize_status_value(value, *, depth: int = 0, field: str = ""):
    """Bound a heartbeat payload before persisting it in run status JSON."""
    if depth > 5:
        return str(value)[:200]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, list):
        if field == "metrics_last":
            values = value[-_STATUS_METRICS_HISTORY_LIMIT:]
        else:
            values = value[:_STATUS_LIST_LIMIT]
        return [_sanitize_status_value(v, depth=depth + 1) for v in values]
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 64:
                out["truncated"] = True
                break
            sanitized_key = str(k)[:120]
            out[sanitized_key] = _sanitize_status_value(v, depth=depth + 1, field=sanitized_key)
        return out
    return str(value)[:500]


def record_heartbeat(run_id: str, heartbeat: dict) -> None:
    """Persist the latest worker heartbeat/GPU snapshot without changing run state."""
    if not run_id or not isinstance(heartbeat, dict):
        return
    if not os.path.exists(state.runs_file_path(run_id, ".json")):
        return
    hb = _sanitize_status_value(heartbeat)
    gpu = hb.get("gpu") if isinstance(hb, dict) else None
    with state._status_guard(run_id):
        try:
            status = get_status(run_id)
        except FileNotFoundError:
            return
        prev = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else None
        # a boot/retry heartbeat for a NEW attempt must inherit nothing from the previous one: its
        # metrics are a different run of the steps and its gpu snapshot is a different card.
        same_attempt = prev is not None and prev.get("attempt") == hb.get("attempt")
        # Checkpoint-stage heartbeats (checkpoint_uploading/deployable/uploaded) omit metrics_last; carry
        # the existing per-step backlog forward so `flash runs log -f` doesn't drop it mid-save until the next
        # metrics-bearing heartbeat lands.
        if isinstance(hb, dict) and not hb.get("metrics_last"):
            prev_metrics = prev.get("metrics_last") if isinstance(prev, dict) else None
            if same_attempt and isinstance(prev_metrics, list) and prev_metrics:
                hb["metrics_last"] = prev_metrics
        if isinstance(hb, dict) and status.lifecycle_progressed_attempt is None:
            expected_stage = {
                "sft": "sft_step",
                "grpo": "rl_step",
                "opd": "opd_step",
            }.get(status.spec.get("algorithm"))
            attempt = hb.get("attempt")
            remote_attempt = (
                status.remote.get("attempt") if isinstance(status.remote, dict) else None
            )
            step = hb.get("step")
            if (
                hb.get("stage") in {expected_stage, "done"}
                and expected_stage is not None
                and attempt == remote_attempt
                and isinstance(attempt, int)
                and not isinstance(attempt, bool)
                and attempt >= 0
                and isinstance(step, int)
                and not isinstance(step, bool)
                and step >= 1
            ):
                status.lifecycle_progressed_attempt = attempt
        status.last_heartbeat = hb
        # carried forward on the same rule as the metric backlog above, and for the same reason: most
        # heartbeats carry no `gpu` at all -- only the periodic liveness tick and the terminal ones do
        # -- so assigning unconditionally blanks the snapshot on every checkpoint-stage heartbeat and
        # leaves `flash runs status` and the API reporting no GPU for a running job.
        if isinstance(gpu, dict):
            status.gpu_status = gpu
        elif not same_attempt:
            status.gpu_status = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)


def validate_terminal_source_metrics(
    status: RunStatus,
    metrics: dict,
    *,
    expected_attempt: int | None = None,
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
    if expected_attempt is None:
        remote = status.remote if isinstance(status.remote, dict) else {}
        candidate = remote.get("attempt")
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            expected_attempt = candidate
    if expected_attempt is None:
        expected_attempt = attempts._latest_reserved_attempt(status.run_id)
    if (
        isinstance(expected_attempt, bool)
        or not isinstance(expected_attempt, int)
        or expected_attempt < 0
    ):
        raise RuntimeError("managed attempt identity is unavailable for source attestation")
    validate_attestation(
        raw_attestation,
        descriptor,
        run_id=status.run_id,
        attempt=expected_attempt,
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


def _apply_transition(
    status: RunStatus, new_state: str, *, allow_from_terminal: bool, updates: dict
) -> bool:
    """Apply the sticky terminal compare-and-set in memory. The caller owns the lock and the write.

    Split out so a transition that must also persist cleanup intent can do both in ONE write:
    `fail_with_cleanup_intent` needs exactly this mutation, under the same guard, with private keys
    added to the same `_save_status_unlocked` call.
    """
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
        if (
            key in {"lifecycle_started_attempt", "lifecycle_progressed_attempt"}
            and getattr(status, key) is not None
        ):
            continue
        setattr(status, key, value)
    return True


def _update(run_id: str, new_state: str, *, allow_from_terminal: bool = False, **updates) -> bool:
    """Atomically transition run state with terminal-stickiness. Returns False if rejected.

    Returns ``True`` if the transition was applied, ``False`` if it was rejected because the run was
    already in a terminal state (the sticky compare-and-set below). Callers that gate PAID work on a
    transition (e.g. the recovery path resuming ``_run_training``) must check this return so a run
    concurrently flipped terminal does not get resumed.
    """
    with state._status_guard(run_id):
        status = get_status(run_id)
        if not _apply_transition(
            status, new_state, allow_from_terminal=allow_from_terminal, updates=updates
        ):
            return False
        state._save_status_unlocked(status)
    reporting._report_status(status)
    return True
