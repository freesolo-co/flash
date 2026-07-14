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

from flash.providers._deadline import deadline_kwargs
from flash.spec import JobSpec

if TYPE_CHECKING:
    from flash.runner import RunStatus

_FINAL_DEPLOYMENT_STATES = frozenset({"done", "deployed"})
_RESTORABLE_DEPLOYMENT_STATES = frozenset({"ready", "deployed"})


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run, confirm remote teardown, and persist any remaining cleanup target."""
    from flash.runner import (
        TERMINAL_STATES,
        _drain_cleanup_remotes,
        _gc_run_endpoints,
        _preserve_cleanup_remote,
        _update,
        actual_steps_run,
        charge_usd_for_spec,
        effective_spec_from_status,
        get_status,
        mark_deployment_undeployed,
    )
    from flash.server._locks import _deploy_lock

    with _deploy_lock(run_id):
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            if status.state == "cancelled":
                with contextlib.suppress(Exception):
                    _drain_cleanup_remotes(run_id)
                return get_status(run_id)
            return status
        entered_deployed = status.state == "deployed"
        public_spec = JobSpec.from_dict(status.spec)
        effective_spec = None
        with contextlib.suppress(Exception):
            effective_spec = effective_spec_from_status(status)
        cleanup_spec = effective_spec or public_spec
        bill_cancel = bool(status.billing_context) and not entered_deployed
        remote = status.remote or {}
        if entered_deployed:
            try:
                from flash.serve.deploy import undeploy_adapter

                undeploy_adapter(run_id)
                if status.deployment:
                    mark_deployment_undeployed(run_id)
            except Exception:
                pass
        teardown_confirmed = False
        if remote:
            try:
                from flash.providers.base import JobHandle
                from flash.runner.lifecycle import _strict_teardown_handle

                _strict_teardown_handle(JobHandle.from_dict(remote))
                teardown_confirmed = True
            except Exception:
                pass
        _gc_run_endpoints(cleanup_spec)
        if remote and not teardown_confirmed and not _preserve_cleanup_remote(run_id, remote):
            raise RuntimeError(
                f"run {run_id} teardown was unconfirmed and its exact cleanup target "
                "could not be preserved"
            )
        cancel_charge_usd: float | None = None
        billing_diagnostic: dict = {}
        if bill_cancel:
            fresh_status = get_status(run_id)
            if effective_spec is not None:
                cancel_charge_usd = charge_usd_for_spec(
                    effective_spec,
                    steps=actual_steps_run(fresh_status),
                    fallback=float(
                        fresh_status.estimated_cost_usd
                        if fresh_status.estimated_cost_usd is not None
                        else fresh_status.cost_usd
                    ),
                )
            else:
                cancel_charge_usd = float(
                    fresh_status.estimated_cost_usd
                    if fresh_status.estimated_cost_usd is not None
                    else fresh_status.cost_usd
                )
                billing_diagnostic = {
                    "billing_state": "retry",
                    "billing_error": (
                        "cancel repricing requires reconciliation because the private preparation "
                        "snapshot was unavailable or invalid"
                    ),
                }
        cancel_updates = {} if cancel_charge_usd is None else {"cost_usd": cancel_charge_usd}
        cancel_updates.update(billing_diagnostic)
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
        _compare_and_clear_remote,
        _gc_run_endpoints,
        _load_run_deadline_at,
        _opd_progress_detected,
        _persist_metrics,
        _run_training,
        _RunCancelled,
        _spec_with_remaining_wall,
        _status_estimated_charge,
        _update,
        artifacts_dir,
        effective_spec_from_status,
        get_status,
    )

    cleanup_terminal = False

    def status_for_return() -> RunStatus:
        nonlocal cleanup_terminal
        current = get_status(run_id)
        cleanup_terminal = current.state in TERMINAL_STATES
        return current

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if not status.remote:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")

    public_spec = JobSpec.from_dict(status.spec)
    worker_spec = public_spec
    log = log_stream or sys.stderr
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.runner.lifecycle import _strict_teardown_handle

    try:
        worker_spec = effective_spec_from_status(status)
        persisted_remote = dict(status.remote)
        remote = dict(persisted_remote)
        seed = int(remote.pop("seed", FIXED_SEED))
        code_prefix = remote.pop("code_prefix", None)
        if (remote.get("provider") or "runpod") == "runpod":
            from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

            recovered_attempt = RunpodJobHandle.from_dict(remote).attempt
        else:
            try:
                recovered_attempt = max(0, int(remote.get("attempt") or 0))
            except (TypeError, ValueError):
                recovered_attempt = 0
        # legacy handles without an attempt predate persisted retry identities and represent attempt 0;
        # resuming at 1 preserves reverse-only recovery while avoiding the known attempt-0 artifact.
        next_attempt = recovered_attempt + 1
        # The class the run actually provisioned (a policy retry may have walked past the
        # provisional spec.gpu.type). The in-process success path stamps this into metrics;
        # on recovery the worker output carries no such field, so recover it from the handle
        # to cost the right card.
        allocated_gpu = remote.pop("allocated_gpu", None)
        handle = JobHandle.from_dict(remote)
        try:
            poll_spec = _spec_with_remaining_wall(worker_spec, require_provider_minimum=False)
        except RuntimeError as exc:
            # expiry is terminal: stop the persisted worker before clearing its handle, and retain the
            # handle when teardown is unconfirmed so later cleanup can still target the paid resource.
            teardown_confirmed = True
            try:
                _strict_teardown_handle(handle)
            except Exception:
                teardown_confirmed = False
            if teardown_confirmed:
                _compare_and_clear_remote(run_id, persisted_remote)
            _update(run_id, "failed", error=str(exc))
            print(f"attach: {run_id} {exc}", file=log)
            return get_status(run_id)
        print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
        res = get_provider(handle.provider).poll(
            handle,
            poll_spec,
            seed,
            log=log,
            _deadline_at=_load_run_deadline_at(run_id),
        )
        if get_status(run_id).state == "cancelled":
            return status_for_return()
        if not res.ok:
            print(f"attach: {run_id} ended ({res.failure}); evaluating recovery", file=log)
            try:
                _strict_teardown_handle(handle)
            except Exception:
                with contextlib.suppress(Exception):
                    _gc_run_endpoints(worker_spec)
                print(
                    f"attach: {run_id} {handle.provider} teardown unconfirmed; "
                    "not resuming over a possibly-live resource",
                    file=log,
                )
                return status_for_return()
            with contextlib.suppress(Exception):
                _gc_run_endpoints(worker_spec)
            opd_progress_detected = public_spec.algorithm == "opd" and _opd_progress_detected(
                run_id
            )
            if not _compare_and_clear_remote(run_id, persisted_remote):
                print(
                    f"attach: {run_id} persisted remote changed before clear; not resuming",
                    file=log,
                )
                return status_for_return()
            if opd_progress_detected:
                detail = (
                    "opd progress was detected; automatic recovery is blocked because opd checkpoint "
                    "resume is not supported"
                )
                _update(run_id, "failed", error=detail)
                print(f"attach: {run_id} {detail}", file=log)
                return status_for_return()
            try:
                _spec_with_remaining_wall(worker_spec, require_provider_minimum=True)
            except RuntimeError as exc:
                _update(run_id, "failed", error=str(exc))
                print(f"attach: {run_id} {exc}", file=log)
                return status_for_return()
            worker_spec = effective_spec_from_status(get_status(run_id), verify_source=True)
            if not _update(run_id, "running"):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return status_for_return()
            print(
                f"attach: {run_id} resubmitting before the run-global wall deadline"
                + ("" if public_spec.algorithm == "opd" else " from the latest checkpoint"),
                file=log,
            )
            if code_prefix is None:
                from flash.providers._worker import upload_code
                from flash.runner import flash_code_prefix

                code_prefix = flash_code_prefix()
                upload_code(
                    worker_spec.train.hf_repo,
                    code_prefix=code_prefix,
                    **deadline_kwargs(upload_code, _load_run_deadline_at(run_id)),
                )
            _run_training(
                worker_spec,
                log,
                prior_cost=float(status.cost_usd or 0.0),
                code_prefix=code_prefix,
                attempt_start=next_attempt,
            )
            return status_for_return()
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        # Add the recovered run's cost to any already booked before the restart so recovery
        # doesn't underreport spend.
        measured = float(status.cost_usd or 0.0) + _persist_metrics(public_spec, res.metrics)
        # Charge the submit-time QUOTE, not measured wall; recovery doesn't change the quote.
        # Legacy runs without a persisted quote are re-priced from the spec.
        charge_usd = _status_estimated_charge(get_status(run_id), public_spec, fallback=measured)
        # A cancel can land while this thread persists the recovered metrics (after the late-cancel
        # check above). Re-read before the terminal "done" so a late worker success can't resurrect
        # a user-cancelled run. _RunCancelled is caught below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            cleanup_terminal = True
            raise _RunCancelled(f"run {run_id} was cancelled")
        _update(run_id, "done", cost_usd=charge_usd, artifacts_dir=artifacts_dir(public_spec))
        cleanup_terminal = True
    except _RunCancelled:
        # this also signals a non-terminal duplicate-supervisor refusal, so only a readable terminal
        # status may authorize cleanup; otherwise retain prior positive terminal knowledge.
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
        cleanup_terminal = True
    finally:
        # only reap a positively observed terminal run. if the final status read is transiently
        # unavailable, retain prior terminal knowledge without risking a live duplicate supervisor.
        with contextlib.suppress(Exception):
            cleanup_terminal = get_status(run_id).state in TERMINAL_STATES
        if cleanup_terminal:
            with contextlib.suppress(Exception):
                _gc_run_endpoints(worker_spec)
    return status_for_return()


def _promote_final_deployment(status: RunStatus, deployment: dict) -> None:
    """Apply the lifecycle state for a final-adapter deployment."""
    # Preserve teardown time for legacy `done` runs (finished_at=None) before deploy bumps updated_at.
    if status.state == "done" and status.finished_at is None and not status.reconciled_at:
        status.finished_at = status.updated_at
    status.deployment = deployment
    status.state = "deployed"


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    from flash.runner import _UNDEPLOYABLE_STATES, _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        _promote_final_deployment(status, deployment)
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_checkpoint_deployed(
    run_id: str, deployment: dict, expect_state: str | None = None
) -> RunStatus:
    """Record a checkpoint deployment using the run's current lifecycle state.

    If training has finished by the time serving registration completes, the run behaves like any
    finished deployed run. Otherwise, keep the training state and only attach the deployment record.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
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
        _save_status_unlocked(status)
        return status


def mark_deployment_pending(
    run_id: str, deployment: dict, expect_state: str | None = None
) -> RunStatus:
    """Attach an in-progress deployment record without changing the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    """Record a failed deployment attempt while preserving the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
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
        if isinstance(previous, dict) and previous.get("state") in _RESTORABLE_DEPLOYMENT_STATES:
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
        _save_status_unlocked(status)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live final-adapter deployments return to `done`."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
        if status.state == "deployed":
            status.state = "done"
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by cancel_run — unlike mark_undeployed, never asserts or changes the run state,
    so it works even after a racing mark_undeployed has already written terminal `done`.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
            status.updated_at = time.time()
            _save_status_unlocked(status)
        return status
