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
_CHECKPOINT_DEPLOYMENT_STATES = frozenset(
    {"queued", "provisioning", "running", "done", "deployed", "failed", "cancelled"}
)
_CHECKPOINT_FORWARD_STATES = {
    "queued": frozenset({"provisioning", "running", "done"}),
    "provisioning": frozenset({"running", "done"}),
    "running": frozenset({"done"}),
}


def deployment_lifecycle_allows_intent(
    current_state: str,
    *,
    expected_state: str,
    is_checkpoint: bool,
) -> bool:
    """Return whether exact queued lifecycle ownership still permits registry intent."""
    allowed_states = _CHECKPOINT_DEPLOYMENT_STATES if is_checkpoint else _FINAL_DEPLOYMENT_STATES
    if expected_state not in allowed_states or current_state not in allowed_states:
        return False
    if current_state == expected_state:
        return True
    if not is_checkpoint:
        return False
    return current_state in _CHECKPOINT_FORWARD_STATES.get(expected_state, frozenset())


def _local_cleanup_intent(run_id: str, deployment: dict) -> dict | None:
    """Best-effort cleanup identity recovered from a local deployment record.

    When the authoritative registry read fails (or is deferred past provider teardown) during cancel,
    the local deployment still carries the immutable ``target_revision``/``mutation_id`` this attempt
    registered, which is enough to persist a cleanup intent for the reconcile pass to retry. Returns
    ``None`` when the local identity is itself malformed and no intent can be formed.
    """
    from flash.serve.deploy import ServingError, persisted_adapter_cleanup

    try:
        return persisted_adapter_cleanup(run_id, deployment)
    except ServingError:
        return None


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run: stop the remote worker and mark it cancelled."""
    from flash.runner import (
        TERMINAL_STATES,
        _gc_run_endpoints,
        _update,
        actual_steps_run,
        charge_usd_for_spec,
        complete_deployment_cleanup,
        get_status,
        mark_deployment_cleanup,
        mark_deployment_undeployed,
        revoke_deployment_attempt,
        revoke_deployment_intent,
    )
    from flash.server._locks import _deploy_lock

    def revoke_current_deployment_ownership(*, remote: bool) -> RunStatus:
        status = get_status(run_id)
        attempt = status.deployment_attempt
        if isinstance(attempt, dict):
            revoke_deployment_attempt(run_id, attempt)
        status = get_status(run_id)
        deployment = status.deployment or {}
        if deployment.get("state") == "deploying":
            mutation_id = str(deployment.get("mutation_id") or "")
            if mutation_id:
                revoke_deployment_intent(run_id, mutation_id)
        status = get_status(run_id)
        cleanup = status.deployment_cleanup
        if not isinstance(cleanup, dict):
            deployment = status.deployment or {}
            # A cancelled run keeps any ready per-step CHECKPOINT deployment it registered (the chat
            # path still serves a cancelled run's checkpoint), so only a FINAL adapter deployment is
            # torn down on cancel.
            is_checkpoint = deployment.get("checkpoint_step") is not None
            if not is_checkpoint and deployment.get("state") in {"ready", "deployed"}:
                ownership: dict | None
                authoritative_absence = False
                if remote:
                    try:
                        from flash.serve.deploy import owned_adapter_cleanup

                        ownership = owned_adapter_cleanup(run_id, deployment)
                        authoritative_absence = ownership is None
                    except Exception:
                        # The registry read (and its persisted-identity fallback) failed, so we cannot
                        # prove the adapter is gone. Persist a cleanup intent from the local
                        # deployment's immutable identity so reconcile retries, rather than leaving the
                        # remote adapter active with the record stuck at `ready`.
                        ownership = _local_cleanup_intent(run_id, deployment)
                else:
                    # Defer the networked registry read until after provider teardown so a serving
                    # outage cannot block GPU destruction; persist the local cleanup intent now.
                    ownership = _local_cleanup_intent(run_id, deployment)
                if authoritative_absence:
                    mark_deployment_undeployed(run_id, expect_deployment=deployment)
                elif ownership is not None:
                    cleanup = {
                        **ownership,
                        "local_deployment": deployment,
                        "requested_at": time.time(),
                    }
                    marked = mark_deployment_cleanup(
                        run_id,
                        cleanup,
                        expect_deployment=deployment,
                    )
                    cleanup = marked.deployment_cleanup
        if remote and isinstance(cleanup, dict):
            try:
                from flash.serve.deploy import reconcile_owned_adapter_cleanup

                reconciled = reconcile_owned_adapter_cleanup(run_id, cleanup)
            except Exception:
                pass
            else:
                if reconciled:
                    complete_deployment_cleanup(run_id, cleanup)
        return get_status(run_id)

    with _deploy_lock(run_id):
        status = get_status(run_id)
        entered_deployed = status.state == "deployed"
        # A deployed or already-terminal run has no live training GPU, so reconcile its serving
        # adapter now: the terminal path returns early below, so deferring would leave the pending
        # cleanup to background recovery. A run still training bills a GPU, so defer the networked
        # adapter reconcile until after teardown below.
        reconcile_now = entered_deployed or status.state in TERMINAL_STATES
        status = revoke_current_deployment_ownership(remote=reconcile_now)
        if status.state in TERMINAL_STATES and not entered_deployed:
            return status
    # Only a deployed run can have a racing undeploy write `done`; a training `done` is genuine.
    spec = JobSpec.from_dict(status.spec)
    # A run cancelled MID-training is re-priced to how far it got: the same flash.cost estimate, but
    # at the steps it actually ran instead of the planned steps. A `deployed` run already COMPLETED
    # training (its cost_usd is the full quote), so it keeps that and isn't re-priced here. The price
    # is snapshotted AFTER the remote worker is torn down (below), from the freshest persisted
    # heartbeat, so a step the worker finished between this cancel request and teardown isn't
    # undercounted.
    bill_cancel = bool(status.billing_context) and not entered_deployed
    remote = status.remote or {}
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
    with _deploy_lock(run_id):
        revoke_current_deployment_ownership(remote=True)
        # Set the cancel charge (estimate at actual steps) when re-pricing a mid-training cancel; a
        # deployed-then-cancelled run keeps its already-quoted cost_usd. The billing_retry sweep
        # charges the run from cost_usd (idempotent by runId).
        cancel_updates = {} if cancel_charge_usd is None else {"cost_usd": cancel_charge_usd}
        _update(run_id, "cancelled", allow_from_terminal=entered_deployed, **cancel_updates)
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
        _status_estimated_charge,
        _status_org_id,
        _update,
        artifacts_dir,
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
            # Before resuming, the in-flight instance MUST be CONFIRMED torn down. Resubmitting while
            # it may still be alive runs TWO workers against this run's shared HF artifacts
            # (DONE/metrics/checkpoints) — double bill AND corrupted state. An instance provider's
            # destroy() raises only on an UNCONFIRMED teardown (Vast: DELETE success:false / network
            # breakdown — a real 404 is now treated as confirmed-gone). The poll loop's own finally
            # already best-effort-destroyed the box; re-confirm here. On an unconfirmed result, GC by
            # label (run-scoped, not orphan-sweep-shielded) and BAIL with the handle intact + the run
            # left non-terminal, so a later recovery/sweep reconciles instead of racing a live box.
            from flash.providers import INSTANCE_PROVIDERS

            teardown_confirmed = True
            if handle.provider in INSTANCE_PROVIDERS:
                try:
                    get_provider(handle.provider).destroy(handle)
                except Exception as exc:
                    teardown_confirmed = False
                    print(
                        f"attach: {run_id} {handle.provider} instance teardown UNCONFIRMED ({exc}); "
                        "not resuming over a possibly-live box",
                        file=log,
                    )
            # GC the dead endpoint / any label-named instances (a second force-reap attempt when the
            # teardown above was unconfirmed), then clear the stale handle.
            with contextlib.suppress(Exception):
                _gc_run_endpoints(public_spec)
            if not teardown_confirmed:
                # Keep ``remote`` so the still-billing box stays reachable for the next recovery/sweep,
                # and leave the run non-terminal (do not _update) so a future re-attach re-polls it.
                return get_status(run_id)
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
                from flash.providers._worker import upload_code
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
        # Charge the submit-time QUOTE, not measured wall; recovery doesn't change the quote.
        # Legacy runs without a persisted quote are re-priced from the spec.
        charge_usd = _status_estimated_charge(get_status(run_id), public_spec, fallback=measured)
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


def mark_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    expect_mutation_id: str | None = None,
    expect_deployment_state: str | None = None,
) -> RunStatus:
    from flash.runner import _STATUS_LOCK, _UNDEPLOYABLE_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        current_deployment = status.deployment or {}
        if (
            expect_mutation_id is not None
            and current_deployment.get("mutation_id") != expect_mutation_id
        ):
            return status
        if (
            expect_deployment_state is not None
            and current_deployment.get("state") != expect_deployment_state
        ):
            return status
        _promote_final_deployment(status, deployment)
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_checkpoint_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    expect_mutation_id: str | None = None,
    expect_deployment_state: str | None = None,
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
        current_deployment = status.deployment or {}
        if (
            expect_mutation_id is not None
            and current_deployment.get("mutation_id") != expect_mutation_id
        ):
            return status
        if (
            expect_deployment_state is not None
            and current_deployment.get("state") != expect_deployment_state
        ):
            return status
        if status.state in _FINAL_DEPLOYMENT_STATES:
            _promote_final_deployment(status, deployment)
        else:
            status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_attempt_queued(
    run_id: str,
    attempt: dict,
    *,
    expect_state: str,
    expect_deployment: dict | None,
    expect_attempt: dict | None,
) -> RunStatus:
    """Persist exact queued ownership without hiding an active ready deployment."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state != expect_state:
            return status
        if status.deployment != expect_deployment or status.deployment_attempt != expect_attempt:
            return status
        phase = attempt.get("phase")
        queued = attempt.get("deployment")
        if (
            phase not in {"initial", "redeploy"}
            or not isinstance(queued, dict)
            or not queued.get("mutation_id")
        ):
            raise ValueError("invalid deployment attempt")
        if phase == "redeploy":
            if attempt.get("active_deployment") != expect_deployment:
                raise ValueError("redeploy attempt active deployment mismatch")
        else:
            status.deployment = queued
        status.deployment_attempt = attempt
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_intent(
    run_id: str,
    intent: dict,
    *,
    expect_attempt: dict,
    expect_state: str,
) -> RunStatus:
    """Replace exact queued ownership with durable forward-only registry intent."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment_attempt != expect_attempt:
            return status
        queued = expect_attempt.get("deployment")
        phase = expect_attempt.get("phase")
        if (
            not isinstance(queued, dict)
            or not queued.get("mutation_id")
            or intent.get("mutation_id") != queued.get("mutation_id")
        ):
            return status
        if not deployment_lifecycle_allows_intent(
            status.state,
            expected_state=expect_state,
            is_checkpoint=queued.get("checkpoint_step") is not None,
        ):
            return status
        if phase == "initial":
            current = status.deployment or {}
            if (
                current != queued
                or current.get("state") != "deploying"
                or current.get("mutation_id") != queued.get("mutation_id")
            ):
                return status
        elif phase == "redeploy":
            if status.deployment != expect_attempt.get("active_deployment"):
                return status
        else:
            return status
        status.deployment = intent
        status.deployment_attempt = None
        status.updated_at = time.time()
        _save_status(status)
        return status


def revoke_deployment_attempt(run_id: str, attempt: dict) -> RunStatus:
    """Revoke exact queued ownership without disturbing a prior active redeploy."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment_attempt != attempt:
            return status
        queued = attempt.get("deployment")
        if attempt.get("phase") == "redeploy":
            if status.deployment != attempt.get("active_deployment"):
                return status
        elif attempt.get("phase") == "initial" and status.deployment == queued:
            status.deployment = {**queued, "state": "undeployed"}
        else:
            return status
        status.deployment_attempt = None
        status.updated_at = time.time()
        _save_status(status)
        return status


def revoke_deployment_intent(run_id: str, mutation_id: str) -> RunStatus:
    """Revoke an exact durable deployment intent before registry mutation."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        deployment = status.deployment or {}
        if deployment.get("state") != "deploying" or deployment.get("mutation_id") != mutation_id:
            return status
        target_revision = deployment.get("target_revision")
        if not isinstance(target_revision, int):
            return status
        prior_revision = deployment.get("prior_revision")
        prior_mutation_id = deployment.get("prior_mutation_id")
        prior = None
        if prior_revision is not None:
            prior = {
                "revision": prior_revision,
                "mutation_id": prior_mutation_id,
            }
        status.deployment_cleanup = {
            "adapter_id": run_id,
            "target": {
                "revision": target_revision,
                "mutation_id": mutation_id,
            },
            "prior": prior,
            "requested_at": time.time(),
        }
        status.deployment = {**deployment, "state": "undeployed"}
        status.deployment_attempt = None
        if status.state == "deployed":
            status.state = "done"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_cleanup(
    run_id: str,
    cleanup: dict,
    *,
    expect_deployment: dict,
) -> RunStatus:
    """Persist exact cleanup ownership for an unchanged active deployment."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment != expect_deployment or status.deployment_cleanup is not None:
            return status
        status.deployment_cleanup = cleanup
        status.updated_at = time.time()
        _save_status(status)
        return status


def complete_deployment_cleanup(run_id: str, cleanup: dict) -> RunStatus:
    """Complete exact cleanup and deactivate only its matching local deployment."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment_cleanup != cleanup:
            return status
        local_deployment = cleanup.get("local_deployment")
        if isinstance(local_deployment, dict) and status.deployment == local_deployment:
            status.deployment = {**local_deployment, "state": "undeployed"}
            # Mirror mark_undeployed / revoke_deployment_intent: a deployed run that loses its active
            # deployment returns to `done`, so a crash-recovery reconcile of this cleanup cannot leave
            # a `deployed` run whose chat path rejects it for having no ready deployment.
            if status.state == "deployed":
                status.state = "done"
        status.deployment_cleanup = None
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_pre_intent_failed(
    run_id: str,
    attempt: dict,
    failed: dict,
) -> RunStatus:
    """Fail an initial attempt or privately clear an exact queued redeploy."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment_attempt != attempt:
            return status
        queued = attempt.get("deployment")
        phase = attempt.get("phase")
        if phase == "redeploy":
            if status.deployment != attempt.get("active_deployment"):
                return status
            status.deployment_attempt = None
        elif phase == "initial" and status.deployment == queued:
            status.deployment = {**failed, "state": "failed"}
            status.deployment_attempt = None
        else:
            return status
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
            current.get("mutation_id") is not None
            and deployment.get("mutation_id") is not None
            and current.get("mutation_id") != deployment.get("mutation_id")
        ):
            return status
        status.deployment = {**deployment, "state": "failed"}
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
        status.deployment_attempt = None
        status.deployment_cleanup = None
        if status.state == "deployed":
            status.state = "done"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_undeployed(
    run_id: str,
    *,
    expect_deployment: dict | None = None,
) -> RunStatus:
    """Flip only an optionally exact deployment field to ``undeployed``."""
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if expect_deployment is not None and status.deployment != expect_deployment:
            return status
        changed = False
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
            changed = True
        if status.deployment_attempt is not None:
            status.deployment_attempt = None
            changed = True
        if status.deployment_cleanup is not None:
            status.deployment_cleanup = None
            changed = True
        if changed:
            status.updated_at = time.time()
            _save_status(status)
        return status
