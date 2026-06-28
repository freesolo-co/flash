"""Deploy / cancel / recover state transitions for a run.

Store helpers and the lifecycle functions (``_run_training`` / ``_gc_run_endpoints``) are
pulled in via FUNCTION-LOCAL lazy ``from flash.runner import ...`` imports — never at module
level — for the same two reasons as ``lifecycle.py``: avoid a partially-initialized-package
import cycle, and keep the test monkeypatches (e.g. ``flash.runner._gc_run_endpoints``)
reachable through the package global rather than a statically-bound copy.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

from flash.spec import JobSpec

if TYPE_CHECKING:
    from flash.runner import RunStatus


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run: stop the remote worker and mark it cancelled."""
    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _update,
        get_status,
        mark_deployment_undeployed,
    )

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    # Only a deployed run can have a racing undeploy write `done`; a training `done` is genuine.
    entered_deployed = status.state == "deployed"
    spec = JobSpec.from_dict(status.spec)
    remote = status.remote or {}
    if status.state == "deployed":
        try:
            from flash.serve.deploy import undeploy_adapter

            undeploy_adapter(run_id)
            if status.deployment:
                mark_deployment_undeployed(run_id)
        except Exception:
            pass
    if remote:
        try:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(remote)
            provider = get_provider(handle.provider)
            provider.cancel(handle)
            provider.destroy(handle)
        except Exception:
            pass
    _gc_run_endpoints(spec)
    # Serialize the terminal `cancelled` write + orphan teardown under the per-run deploy lock (the
    # /deploy route holds the same lock around attach_checkpoint_deployment/mark_deployed). Without it,
    # a deliberate post-cancel checkpoint deploy could attach its deployment between the `cancelled`
    # write and the re-read below, and the teardown would tear that fresh, intentional deployment down
    # as if it were a cancel orphan.
    from flash.server._locks import _deploy_lock

    with _deploy_lock(run_id):
        # Override terminal `done` only if we entered `deployed`: a racing mark_undeployed writes
        # `done` as an undeploy artifact (cancel wins), but a genuine training completion must not
        # be clobbered (cancel loses). The two races are mutually exclusive on entry state.
        _update(run_id, "cancelled", allow_from_terminal=entered_deployed)
        # A deploy can race in after the entry snapshot (running -> done -> deployed) and land before
        # this write; `deployed` is non-terminal so the `cancelled` write still wins, but the
        # entry-gated undeploy above never ran. Re-read and tear down any deployment still active so
        # cancel never orphans one. Holding the deploy lock keeps a post-cancel checkpoint deploy from
        # interleaving here: it blocks until we release, then attaches its deployment untouched.
        final = get_status(run_id)
        if (final.deployment or {}).get("state") not in (None, "undeployed", "dry_run"):
            with contextlib.suppress(Exception):
                from flash.serve.deploy import undeploy_adapter

                undeploy_adapter(run_id)
                mark_deployment_undeployed(run_id)
    with contextlib.suppress(Exception):
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(get_status(run_id))
    return get_status(run_id)


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner import (
        FIXED_SEED,
        TERMINAL_STATES,
        _gc_run_endpoints,
        _persist_metrics,
        _run_training,
        _RunCancelled,
        _update,
        artifacts_dir,
        get_status,
    )

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if not status.remote:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")

    spec = JobSpec.from_dict(status.spec)
    remote = dict(status.remote)
    seed = int(remote.pop("seed", FIXED_SEED))
    # The class the run actually provisioned (a policy retry may have walked past the
    # provisional spec.gpu.type). The in-process success path stamps this into metrics;
    # on recovery the worker output carries no such field, so recover it from the handle
    # to cost the right card.
    allocated_gpu = remote.pop("allocated_gpu", None)
    log = log_stream or sys.stderr
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    handle = JobHandle.from_dict(remote)
    print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
    res = get_provider(handle.provider).poll(handle, spec, seed, log=log)
    try:
        if get_status(run_id).state == "cancelled":
            return get_status(run_id)
        if not res.ok:
            # Job ended not-ok — usually because it was abandoned during the redeploy. Resume from
            # the last HF checkpoint (fresh allocation, worker resumes mid-training) instead of
            # failing; _run_training still terminates a genuinely broken run when it re-fails.
            print(
                f"attach: {run_id} ended ({res.failure}); resuming from checkpoint",
                file=log,
            )
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            # Bail if the run was raced to terminal during the long poll above: _update's CAS
            # returns False, and resuming would submit paid work for a dead run.
            if not _update(run_id, "running", remote=None):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return get_status(run_id)
            _run_training(spec, log, prior_cost=float(status.cost_usd or 0.0))
            return get_status(run_id)
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        # Add the recovered run's cost to any already booked before the restart so recovery
        # doesn't underreport spend.
        total = float(status.cost_usd or 0.0) + _persist_metrics(spec, res.metrics)
        # A cancel can land while this thread persists the recovered metrics (after the late-cancel
        # check above). Re-read before the terminal "done" so a late worker success can't resurrect
        # a user-cancelled run. _RunCancelled is caught below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        _update(run_id, "done", cost_usd=total, artifacts_dir=artifacts_dir(spec))
    except _RunCancelled:
        pass  # cancel_run already wrote terminal `cancelled`
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        _gc_run_endpoints(spec)
    return get_status(run_id)


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    from flash.runner import _STATUS_LOCK, _UNDEPLOYABLE_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        # Preserve teardown time for legacy `done` runs (finished_at=None) before deploy bumps updated_at.
        if status.state == "done" and status.finished_at is None and not status.reconciled_at:
            status.finished_at = status.updated_at
        status.deployment = deployment
        status.state = "deployed"
        status.updated_at = time.time()
        _save_status(status)
        return status


def attach_checkpoint_deployment(run_id: str, deployment: dict) -> RunStatus:
    """Attach a serving deployment to a run WITHOUT changing its training state.

    Used for cancelled/failed runs serving a mid-RL checkpoint — preserves terminal state
    while still tracking the deployment so /v1/deployments lists it and undeploy clears it.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live `deployed` runs return to `done`."""
    from flash.runner import _STATUS_LOCK, TERMINAL_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
        if status.state not in TERMINAL_STATES:
            status.state = "done"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by cancel_run — unlike mark_undeployed, never asserts or changes the run state,
    so it works even after a racing mark_undeployed has already written terminal `done`.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
            status.updated_at = time.time()
            _save_status(status)
        return status
