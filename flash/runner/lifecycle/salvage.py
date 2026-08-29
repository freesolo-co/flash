"""What a failing run still owes, reconstructed from a record that may not decode.

Recovery's hard case is a status record it cannot read: the run is unrecoverable, but something it
rented is still billing, and the only thing that names that resource is the very bytes that failed.
Two questions follow from that, and everything here answers one of them.

*What address survives.* A persisted handle the cleanup drain can act on, or -- when the handle was
lost before it was ever written -- the GPU classes whose derived endpoint names are the only other
way to reach the worker. `has_teardown_address` is the seam between the two, and it decides on the
same key the drain selects by so exactly one of them always applies.

*What record survives.* `_salvage_corrupt_record` rebuilds an envelope that keeps every field the
corrupt bytes still read back, because the field that failed is rarely the field that matters: a run
stripped of its billing context, its artifacts, or its source descriptor is permanently
unsettleable, and that is a worse outcome than the decode error.

These are pure functions over the raw mapping. Locking, the atomic write, and the terminal
transition all live in `status.py`, which owns the durable record itself.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from typing import Any

from flash.core.spec import persisted_gpu_types
from flash.runner.lifecycle import state
from flash.runner.lifecycle.state import RunStatus
from flash.snapshot.archive import SourceSnapshotError


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


def _salvaged_deployment(value: dict) -> dict:
    """The record's own deployment metadata, with a `state` its readers can safely test.

    The mapping fallback above only replaces a `deployment` that is not a dict at all. A dict whose
    nested `state` is unhashable passes that check and is persisted verbatim -- and
    `recover_deployments()` opens by testing exactly that value for membership in two `set`
    literals, which raises `TypeError` on a list or dict. That runs synchronously in the
    `create_app()` lifespan, so it aborts startup readiness rather than failing one record.

    Dropping the key rather than the whole mapping is deliberate: `_deployment_projection` in
    `flash/server/domain/registry/runs.py` still needs `checkpoint_id` and `endpoint` to report a
    live deployment, so discarding them over an unusable sibling would cost a settled run its
    serving provenance. A missing `state` reads back as `None`, which is in neither state set, so
    the record is skipped by recovery exactly as an unrecognized state already is.
    """
    stored = value.get("state")
    if isinstance(stored, str) or "state" not in value:
        return value
    return {key: item for key, item in value.items() if key != "state"}


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
    # json admits an int of unbounded width, and `math.isfinite` converts before it tests, so a
    # 400-digit `finished_at` raises OverflowError rather than answering False. Raising here aborts
    # the whole quarantine (its caller suppresses), leaving a corrupt record that still owns a
    # remote with no terminal state and no teardown intent -- so convert first and refuse what
    # cannot be one.
    try:
        boundary = float(stored)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(boundary) or boundary <= 0:
        return None
    return boundary


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


def has_teardown_address(remote: object) -> bool:
    """Whether a persisted handle names a resource the cleanup drain can actually delete.

    The reclaim marker is the fallback for a run whose endpoint has no handle, so what decides
    between the two mechanisms is whether the handle is *addressable* -- not whether the field is
    truthy. A partial remote such as `{"provider": "runpod"}` is truthy and names nothing: the drain
    rejects it because `_teardown_removal_key` is `None`, so gating on `bool(remote)` would suppress
    the marker in favour of a record that will never be acted on, and the endpoint bills with both
    mechanisms silently declining it. Deciding on the same key the drain selects by means exactly one
    of the two always applies.
    """
    from flash.runner.accounting.reconciliation import _teardown_removal_key

    return _teardown_removal_key(remote) is not None


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


def reclaimable_gpu_types(status: Any) -> tuple[str, ...]:
    """Every GPU class a handle-less run could have created an endpoint under.

    The endpoint is named from the class the deploy actually used, which is the WORKER spec's --
    `_persist_effective_worker_spec` commits that snapshot before `_submit_provider` creates the
    endpoint, so it is on disk whenever a lost handle is possible. Reading only `spec` misses the
    auto-selected run entirely: an unpinned spec persists `gpu.type = ""`, the class is resolved
    during allocation, and `reallocation_spec_from_status` exists precisely because the two halves
    legitimately differ. Both are returned, worker half first, because an attempt that failed over
    can have left an endpoint under either.
    """

    def field(name: str) -> Any:
        return status.get(name) if isinstance(status, dict) else getattr(status, name, None)

    preparation = field("effective_preparation")
    worker_spec = preparation.get("worker_spec") if isinstance(preparation, dict) else None
    return tuple(
        dict.fromkeys(persisted_gpu_types(worker_spec) + persisted_gpu_types(field("spec")))
    )


def _stored_reclaim_types(raw: dict) -> tuple[str, ...]:
    """The frozen classes a stored marker names, ignoring any other shape."""
    stored = raw.get(state._ENDPOINT_RECLAIM_KEY)
    if not isinstance(stored, list):
        return ()
    return tuple(dict.fromkeys(entry for entry in stored if isinstance(entry, str) and entry))


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
        if isinstance(values.get("deployment"), dict):
            values["deployment"] = _salvaged_deployment(values["deployment"])
        if not isinstance(values.get("submitted_instance_providers"), list):
            # the value `_validate_recovery_status` just rejected would be lifted verbatim, so the
            # envelope would fail the same validation on the next boot and quarantine itself again,
            # leaking a `.corrupt-` copy every restart. `None` is the field's own "record predates
            # the feature" value and is what the salvaged record can honestly claim: the submit-time
            # provider set is exactly what was unreadable. it costs this record nothing, because
            # quarantine has just made it terminal and `_confirm_run_clear` only ever runs for a
            # `_RECOVERABLE` one.
            values["submitted_instance_providers"] = None
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


class QuarantineWriteFailed(Exception):
    """The quarantine envelope could not be installed, carrying what it would have addressed.

    Everything needed to reach the run's worker is computed while building the envelope and is
    unreadable afterwards: the terminal record was never written, and the corrupt bytes it was
    salvaged from sit behind the storage that just failed. Without this the caller sees only a
    suppressed exception and skips the row, leaving a worker billing under a name nothing holds.
    """

    def __init__(self, salvaged: RunStatus, reclaim: Sequence[str]) -> None:
        super().__init__("quarantine envelope could not be written")
        self.salvaged = salvaged
        self.reclaim = tuple(reclaim)
