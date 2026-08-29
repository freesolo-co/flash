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
import math
import os
import tempfile
import time

from flash.core.catalog import validate_model_for_algorithm
from flash.core.spec import JobSpec
from flash.core.spec_persistence import validate_persisted_spec_envelope
from flash.runner.lifecycle import attempts, preparation, reporting, state
from flash.runner.lifecycle.state import RunStatus
from flash.snapshot.archive import SourceSnapshotError

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
    return status


def _get_recovery_status(run_id: str) -> RunStatus:
    return _validate_recovery_status(run_id, get_status(run_id))


def _salvage_teardown_handle(remote: object) -> dict | None:
    """The strict handle when it parses, else the lenient shape the cleanup drain already accepts.

    Requiring canonical form here would drop the handle for every endpoint a currently deployed
    release created: those writers persist the 16-character `rpk-` fingerprint, which
    `_canonical_cleanup_remote` rejects outright. The record would then reach teardown with nothing
    to delete and a live RunPod endpoint would bill forever. `_drainable_cleanup_remotes` yields
    these verbatim for exactly this reason, and `_strict_teardown_handle` resolves the short
    fingerprint itself.
    """
    from flash.runner.accounting.reconciliation import (
        _canonical_cleanup_remote,
        _uncanonical_teardown_record,
    )

    with contextlib.suppress(Exception):
        return _canonical_cleanup_remote(remote) or _uncanonical_teardown_record(remote)
    return None


def _describes_run(run_id: str, raw: dict | None) -> bool:
    """False when the record names a different run, so none of it is evidence about `run_id`.

    A stored id that does not confirm the filename is the one corruption where salvage is unsafe
    rather than merely lossy: `_strict_teardown_handle` deletes by recorded endpoint id without
    relating it to the run id it is passed, so lifting those handles would let one swapped status
    file terminate another live run's worker. The hard-linked sidecar still preserves the record
    for an operator; the envelope just cannot claim any of it.

    Only an id present and exactly equal to `run_id` is evidence. Everything else is refused --
    a different id, `null`, a number, and an absent key alike. `_status_storage_dict` writes
    `run_id` on every persisted record, so a missing key is itself corruption and not a
    filename-only record: the bytes may still carry another run's live handle, and the filename
    cannot vouch for it. `_validate_recovery_status` already declines to trust these, so lifting
    teardown evidence out of them would destroy exactly what that decoder refused.
    """
    if raw is None:
        return False
    return raw.get("run_id") == run_id


# quarantine owns these; every other readable field is lifted verbatim. `source_snapshot` is
# controlled because it is the one field that fails closed, so it cannot be copied verbatim -- it is
# revalidated by `_salvaged_source_snapshot` and kept only when the descriptor itself still parses.
_QUARANTINE_CONTROLLED = frozenset(
    {"run_id", "state", "error", "finished_at", "updated_at", "remote", "source_snapshot"}
)

# settled top-level outcomes quarantine must not relabel. `deployed` is not in `TERMINAL_STATES`
# but it is a live top-level `status.state` this build writes, and three independent readers treat
# it as settled: `_FINAL_DEPLOYMENT_STATES` in `flash/runner/supervise/deploy.py`,
# `_DEPLOYABLE_STATES` in `flash/server/asgi/app.py`, and `_RECONCILABLE_STATES` in
# `flash/server/domain/ops/reconcile.py`. Rewriting it to `failed` would strand a still-live
# serving deployment: undeployable, no downloadable artifact, and no longer reconcilable.
_QUARANTINE_RETAINED_STATES = state.TERMINAL_STATES | {"deployed"}

# lifted fields that readers dereference as mappings, and the fallback for a stored value that is
# not one. `or {}` does not cover this: every one of these breaks on a *truthy* non-dict, which is
# what `.get()` is called on. Each has a reader that gets no per-record isolation:
#   `spec`/`effective_preparation` -- `_status_storage_dict` reads them through
#     `_adapter_ref_for_status` during the quarantine write itself, and
#     `_quarantine_corrupt_recovery_record` suppresses the `AttributeError`, so the record would
#     silently stay unquarantined and its live remote would never be torn down.
#   `deployment` -- `recover_deployments()` runs synchronously right after `recover_runs()` in the
#     `create_app()` lifespan and opens with `(status.deployment or {}).get("state")`; that raise
#     is not caught per record, so one bad envelope aborts startup before readiness.
# the value is unusable either way; the fallback keeps it from taking a reader down with it.
# `spec` is non-optional on `RunStatus`, so it falls back to an empty mapping rather than `None`.
_QUARANTINE_MAPPING_FALLBACKS: dict[str, dict | None] = {
    "spec": {},
    "effective_preparation": None,
    "deployment": None,
}


def _salvaged_finished_at(raw: dict) -> float | None:
    """The record's own terminal timestamp, which billing treats as the teardown boundary.

    `finished_at` is stamped once on the first terminal transition and is documented on `RunStatus`
    as surviving every later `updated_at` bump, because realized provider cost is billed from the
    handle's `started_ts` through this value (`_terminal_ts` in
    `flash/server/domain/ops/reconcile.py`, `_instance_realized_cost` in
    `flash/providers/core/realized.py`). Restamping a settled record at restart would bill the
    entire outage as wall time.

    Anything that is not a finite positive real is unusable as that boundary -- `_terminal_ts`
    raises on `None` and `float()` raises on a string -- and is refused by the same guards
    `_instance_realized_cost` applies to `started_ts`. Refused means `None`, NOT `now`: for a
    settled record the restart instant is not a worse boundary, it is a fabricated one, and
    substituting it makes `_due()` newly true (it returns `False` while `finished_at is None`), so
    reconciliation would bill from the handle's `started_ts` through the restart -- potentially days
    of usage that never happened. Leaving it unset keeps the record out of that pass entirely, which
    is what the pre-quarantine record already did.
    """
    stored = raw.get("finished_at")
    if isinstance(stored, bool) or not isinstance(stored, (int, float)):
        return None
    if not math.isfinite(stored) or stored <= 0:
        return None
    return float(stored)


def _salvaged_source_snapshot(raw: dict) -> dict | None:
    """The record's own source descriptor when it still parses, else nothing.

    `source_snapshot` is the one lifted field that fails closed, so it cannot be copied verbatim:
    re-lifting a malformed descriptor would rebuild a record `get_status` rejects exactly the way it
    rejected the one being quarantined. But blanket-dropping it is equally wrong, because this is
    rarely the field that failed -- a missing required `spec` raises `TypeError` from the
    `RunStatus` constructor with the descriptor perfectly valid. Erasing it then costs a settled run
    its provenance permanently: `_validated_terminal_source` in
    `flash/server/domain/registry/runs.py` calls `parse_descriptor` on it, and without it the run
    reports `artifactsComplete: false` forever even though its artifacts are complete.

    So parse it here, independently of whatever else failed, and keep it only when it is itself
    valid. `parse_descriptor` is the same validator both the decode path and that reader use, and
    its normalized `to_dict()` is what `_runstatus_from_json` would have stored.
    """
    stored = raw.get("source_snapshot")
    if stored is None:
        return None
    from flash.snapshot.archive import parse_descriptor

    try:
        return parse_descriptor(stored).to_dict()
    except SourceSnapshotError:
        return None


def _durable_teardown_intent(remote: dict | None, cleanup_remotes: list | None) -> list | None:
    """Fold the salvaged active handle into the durable cleanup list before teardown is attempted.

    Quarantine installs the terminal envelope first and dispatches teardown onto a background
    thread afterwards. A process exit between those two points would leave the salvaged handle only
    in `status.remote`, which no startup path deletes: `_classify_recoverable_runs` skips the record
    because it is already terminal, and RunPod's `sweep_orphans` is a no-op. Recording the intent
    inside the same atomic write means the next boot's ordinary cleanup drain finds it.

    The drain tolerates handles the strict reader rejects at every stage it has -- the snapshot
    falls back to `_drainable_cleanup_remotes`, teardown selects by `_teardown_removal_key`, and
    removal preserves unrecognized siblings verbatim -- so the lenient handle
    `_salvage_teardown_handle` keeps is persisted here rather than dropped.
    """
    from flash.runner.accounting.reconciliation import _teardown_removal_key

    key = _teardown_removal_key(remote)
    if key is None:
        return cleanup_remotes
    records = list(cleanup_remotes or [])
    if all(_teardown_removal_key(existing) != key for existing in records):
        records.append(dict(remote or {}))
    return records


def persist_teardown_intent(run_id: str) -> None:
    """Record the run's own active handle as durable cleanup intent, then leave the record alone.

    The counterpart to `_durable_teardown_intent` for a record that is already on disk. Quarantine
    can fold the intent into the one write it is already making; a run terminalized by an ordinary
    `_update` cannot, and dispatching teardown for it without this leaves the same crash window --
    the handle lives only in `status.remote`, which no later boot acts on once the run is terminal.

    Deliberately lenient and quiet. Siblings are read verbatim and matched by
    `_teardown_removal_key`, the same leniency `_durable_teardown_intent` relies on, so a legacy
    short-fingerprint handle is preserved rather than dropped -- `_record_cleanup_remote` refuses
    exactly those, and the strict `_cleanup_remotes_from_raw` raises on them. It does not report the
    write either: this only makes an already-persisted handle drainable, and it runs on the startup
    thread, where a blocking status report is the stall the threaded teardown exists to avoid.
    Already-recorded intent short-circuits before any write.
    """
    with state._status_guard(run_id):
        raw = _load_status_json(run_id)
        status = _runstatus_from_json(raw)
        stored = raw.get(state._CLEANUP_REMOTES_KEY)
        existing = list(stored) if isinstance(stored, list) else []
        records = _durable_teardown_intent(status.remote, existing)
        if records is None or len(records) == len(existing):
            return
        state._save_status_unlocked(status, _cleanup_remotes=records)


def _salvage_corrupt_record(
    run_id: str, raw: dict | None, detail: str, now: float
) -> tuple[RunStatus, list | None]:
    """Build the quarantine envelope, keeping every field the corrupt record still reads back.

    Only the field that actually failed is untrustworthy. `_load_status_json` has already proved
    the record decodes to a dict, and the decode path rejects two things itself -- a run id
    mismatch and a non-string state, both in `_validate_recovery_status` -- plus a malformed
    `source_snapshot`, which `parse_descriptor` raises from inside `get_status`. Rebuilding the
    rest from defaults would drop `billing_context`, `cost_usd`, artifacts and deployment metadata,
    which permanently disqualifies a chargeable run from `flash.server.billing.retry._needs_charge`.

    The settled state is kept for the same reason it is checked at all: `_update` refuses to move a
    run back out of a terminal state, but quarantine writes the envelope directly and bypasses that
    compare-and-set, so a `done` or `deployed` record would otherwise be republished as `failed` --
    which is undeployable (`_UNDEPLOYABLE_STATES`) and no longer settleable.
    """
    values: dict = {"spec": {}, "state": "failed", "remote": None}
    cleanup_remotes = None
    finished_at: float | None = now
    if raw is not None and _describes_run(run_id, raw):
        values.update(
            {
                key: value
                for key, value in raw.items()
                if key in RunStatus.__dataclass_fields__ and key not in _QUARANTINE_CONTROLLED
            }
        )
        for key, fallback in _QUARANTINE_MAPPING_FALLBACKS.items():
            if not isinstance(values.get(key), dict):
                values[key] = dict(fallback) if fallback is not None else None
        values["source_snapshot"] = _salvaged_source_snapshot(raw)
        stored_state = raw.get("state")
        if isinstance(stored_state, str) and stored_state in _QUARANTINE_RETAINED_STATES:
            values["state"] = stored_state
            finished_at = _salvaged_finished_at(raw)
        values["remote"] = _salvage_teardown_handle(raw.get("remote"))
        stored_cleanup = raw.get(state._CLEANUP_REMOTES_KEY)
        if isinstance(stored_cleanup, list):
            cleanup_remotes = list(stored_cleanup)
    failed = RunStatus(
        **values, run_id=run_id, error=detail, finished_at=finished_at, updated_at=now
    )
    return failed, cleanup_remotes


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
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError, OSError):
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
            if (
                key in {"lifecycle_started_attempt", "lifecycle_progressed_attempt"}
                and getattr(status, key) is not None
            ):
                continue
            setattr(status, key, value)
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True
