"""Deploy / cancel / recover state transitions for a run.

Uses function-local imports to avoid import cycles and keep test monkeypatches reachable.
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
    # Override terminal `done` only if we entered `deployed`: a racing mark_undeployed writes
    # `done` as an undeploy artifact (cancel wins), but a genuine training completion must not
    # be clobbered (cancel loses). The two races are mutually exclusive on entry state.
    _update(run_id, "cancelled", allow_from_terminal=entered_deployed)
    with contextlib.suppress(Exception):
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(get_status(run_id))
    return get_status(run_id)


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from any process (after a client crash/restart)."""
    import sys

    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _persist_metrics,
        _run_seed_loop,
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
    seed = int(remote.pop("seed", spec.train.seeds[0]))
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
            try:
                seed_index = list(spec.train.seeds).index(seed)
            except ValueError:
                seed_index = 0
            print(
                f"attach: {run_id} seed {seed} ended ({res.failure}); resuming from checkpoint",
                file=log,
            )
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            if not _update(run_id, "running", remote=None, resume_seed_index=seed_index):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return get_status(run_id)
            _run_seed_loop(
                spec, log, start_index=seed_index, prior_cost=float(status.cost_usd or 0.0)
            )
            return get_status(run_id)
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        total = float(status.cost_usd or 0.0) + _persist_metrics(spec, seed, res.metrics)
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        try:
            resumed_index = list(spec.train.seeds).index(seed) + 1
        except ValueError:
            resumed_index = len(spec.train.seeds)
        more_seeds = resumed_index < len(spec.train.seeds)
        # Clear stale handle before resuming so a restart in the inter-seed gap picks the right seed.
        applied = _update(
            run_id,
            "running",
            cost_usd=total,
            artifacts_dir=artifacts_dir(spec),
            **({"remote": None, "resume_seed_index": resumed_index} if more_seeds else {}),
        )
        if more_seeds:
            if not applied:
                print(
                    f"attach: {run_id} went terminal during recovery; "
                    "not resuming the remaining seeds",
                    file=log,
                )
                return get_status(run_id)
            _run_seed_loop(spec, log, start_index=resumed_index, prior_cost=total)
        else:
            _update(run_id, "done", cost_usd=total, artifacts_dir=artifacts_dir(spec))
    except _RunCancelled:
        pass  # cancel_run already wrote terminal `cancelled`
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        _gc_run_endpoints(spec)
    return get_status(run_id)


def resume_run(run_id: str, log_stream=None) -> RunStatus:
    """Resume remaining seeds of a multi-seed run after a restart in the inter-seed gap."""
    import sys

    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _run_seed_loop,
        _RunCancelled,
        _update,
        get_status,
    )

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if status.resume_seed_index is None:
        raise ValueError(f"run {run_id} has no resume_seed_index; cannot resume")
    spec = JobSpec.from_dict(status.spec)
    log = log_stream or sys.stderr
    print(f"resuming {run_id}: remaining seeds from index {status.resume_seed_index}", file=log)
    try:
        _run_seed_loop(
            spec,
            log,
            start_index=status.resume_seed_index,
            prior_cost=float(status.cost_usd or 0.0),
        )
    except _RunCancelled:
        pass  # cancel_run already set the terminal state
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
