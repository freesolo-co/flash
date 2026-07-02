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

_FINAL_DEPLOYMENT_STATES = frozenset({"done", "deployed"})
_RESTORABLE_DEPLOYMENT_STATES = frozenset({"ready", "deployed"})


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run: stop the remote worker and mark it cancelled."""
    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _update,
        actual_steps_run,
        charge_usd_for_spec,
        get_status,
        mark_deployment_undeployed,
    )

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    # Only a deployed run can have a racing undeploy write `done`; a training `done` is genuine.
    entered_deployed = status.state == "deployed"
    spec = JobSpec.from_dict(status.spec)
    # A run cancelled MID-training is re-priced to how far it got: the same flash.cost estimate, but
    # at the steps it actually ran instead of the planned steps. A `deployed` run already COMPLETED
    # training (its cost_usd is the full quote), so it keeps that and isn't re-priced here. The price
    # is snapshotted AFTER the remote worker is torn down (below), from the freshest persisted
    # heartbeat, so a step the worker finished between this cancel request and teardown isn't
    # undercounted.
    bill_cancel = bool(status.billing_context) and not entered_deployed
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
    # Price the cancel now that the worker is torn down, from the freshest persisted heartbeat.
    cancel_charge_usd: float | None = (
        charge_usd_for_spec(spec, steps=actual_steps_run(get_status(run_id)), fallback=0.0)
        if bill_cancel
        else None
    )
    from flash.server._locks import _deploy_lock

    with _deploy_lock(run_id):
        # Set the cancel charge (estimate at actual steps) when re-pricing a mid-training cancel; a
        # deployed-then-cancelled run keeps its already-quoted cost_usd. The billing_retry sweep
        # charges the run from cost_usd (idempotent by runId).
        cancel_updates = {} if cancel_charge_usd is None else {"cost_usd": cancel_charge_usd}
        _update(run_id, "cancelled", allow_from_terminal=entered_deployed, **cancel_updates)
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
        _resolve_init_from_adapter,
        _run_training,
        _RunCancelled,
        _status_org_id,
        _update,
        artifacts_dir,
        charge_usd_for_spec,
        get_status,
    )

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if not status.remote:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")

    public_spec = JobSpec.from_dict(status.spec)
    log = log_stream or sys.stderr
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    try:
        remote = dict(status.remote)
        seed = int(remote.pop("seed", FIXED_SEED))
        code_prefix = remote.pop("code_prefix", None)
        # The class the run actually provisioned (a policy retry may have walked past the
        # provisional spec.gpu.type). The in-process success path stamps this into metrics;
        # on recovery the worker output carries no such field, so recover it from the handle
        # to cost the right card.
        allocated_gpu = remote.pop("allocated_gpu", None)
        handle = JobHandle.from_dict(remote)
        print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
        res = get_provider(handle.provider).poll(handle, public_spec, seed, log=log)
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
                _gc_run_endpoints(public_spec)
            # Bail if the run was raced to terminal during the long poll above: _update's CAS
            # returns False, and resuming would submit paid work for a dead run.
            if not _update(run_id, "running", remote=None):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return get_status(run_id)
            owner_key_id = None
            with contextlib.suppress(Exception):
                from flash.server import db

                owner_key_id = db.run_owner(run_id)
            worker_spec = _resolve_init_from_adapter(
                public_spec,
                owner_org_id=_status_org_id(status),
                owner_key_id=owner_key_id,
            )
            if code_prefix is None:
                from flash.providers.runpod.train import upload_code
                from flash.runner import flash_code_prefix

                code_prefix = flash_code_prefix()
                upload_code(worker_spec.train.hf_repo, code_prefix=code_prefix)
            _run_training(
                worker_spec,
                log,
                prior_cost=float(status.cost_usd or 0.0),
                code_prefix=code_prefix,
            )
            return get_status(run_id)
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        # Add the recovered run's cost to any already booked before the restart so recovery
        # doesn't underreport spend.
        measured = float(status.cost_usd or 0.0) + _persist_metrics(public_spec, res.metrics)
        # Charge the QUOTE (flash.cost estimate for the planned steps), not the measured wall;
        # recovery doesn't change the quote. Falls back to measured only if the spec can't be priced.
        charge_usd = charge_usd_for_spec(public_spec, fallback=measured)
        # A cancel can land while this thread persists the recovered metrics (after the late-cancel
        # check above). Re-read before the terminal "done" so a late worker success can't resurrect
        # a user-cancelled run. _RunCancelled is caught below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        _update(run_id, "done", cost_usd=charge_usd, artifacts_dir=artifacts_dir(public_spec))
    except _RunCancelled:
        pass  # cancel_run already wrote terminal `cancelled`
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        _gc_run_endpoints(public_spec)
    return get_status(run_id)


def _promote_final_deployment(status: RunStatus, deployment: dict) -> None:
    """Apply the lifecycle state for a final-adapter deployment."""
    # Preserve teardown time for legacy `done` runs (finished_at=None) before deploy bumps updated_at.
    if status.state == "done" and status.finished_at is None and not status.reconciled_at:
        status.finished_at = status.updated_at
    status.deployment = deployment
    status.state = "deployed"


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    from flash.runner import _STATUS_LOCK, _UNDEPLOYABLE_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        _promote_final_deployment(status, deployment)
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_checkpoint_deployed(
    run_id: str, deployment: dict, expect_state: str | None = None
) -> RunStatus:
    """Record a checkpoint deployment using the run's current lifecycle state.

    If training has finished by the time serving registration completes, the run behaves like any
    finished deployed run. Otherwise, keep the training state and only attach the deployment record.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        if status.state in _FINAL_DEPLOYMENT_STATES:
            _promote_final_deployment(status, deployment)
        else:
            status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_pending(
    run_id: str, deployment: dict, expect_state: str | None = None
) -> RunStatus:
    """Attach an in-progress deployment record without changing the run lifecycle state."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    """Record a failed deployment attempt while preserving the run lifecycle state."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        current = status.deployment or {}
        # Don't clobber a newer deployment attempt or an explicit undeploy.
        if current.get("state") == "undeployed":
            return status
        if (
            current.get("requested_at") is not None
            and deployment.get("requested_at") is not None
            and current.get("requested_at") != deployment.get("requested_at")
        ):
            return status
        previous = deployment.get("previous_deployment")
        if (
            isinstance(previous, dict)
            and previous.get("state") in _RESTORABLE_DEPLOYMENT_STATES
        ):
            status.deployment = {
                **previous,
                "last_deploy_error": deployment.get("error") or "deployment failed",
                "last_deploy_failed_at": time.time(),
            }
        else:
            failed = dict(deployment)
            failed.pop("previous_deployment", None)
            status.deployment = {**failed, "state": "failed"}
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live final-adapter deployments return to `done`."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
        if status.state == "deployed":
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
