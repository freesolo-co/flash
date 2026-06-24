"""Deploy / cancel / recover state transitions for a run.

Store helpers and the lifecycle functions (``_run_seed_loop`` / ``_gc_run_endpoints``) are
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
    # RunStatus lives in flash.runner.__init__ and is only referenced here as a return
    # annotation (stringized by `from __future__ import annotations`), so a TYPE_CHECKING
    # import keeps it resolvable for tooling without a runtime import cycle.
    from flash.runner import RunStatus


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run: delete its remote Flash endpoint (stopping the worker), then mark it
    cancelled.

    Uses ``terminate_endpoint`` (reconstructs the run's uniquely-named endpoint and deletes it
    via the RunPod API) so the cancel works **cross-process** — a fresh ``flash cancel`` actually
    stops the GPU worker, instead of leaving it running until the wall cap. Best-effort: any
    teardown error is recorded but still flips the run to ``cancelled``.
    """
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
    # Whether the run was a live `deployed` serving run at cancel entry. This scopes the
    # final `cancelled` transition's terminal override below: only a `deployed` run can have
    # a concurrent undeploy (`mark_undeployed`) race this teardown and write a non-completion
    # terminal `done`. A non-deployed run (running/provisioning/queued) has an in-flight
    # TRAINING thread whose only terminal `done` is a GENUINE completion — which cancel must
    # never clobber. See the final _update call for how this gates the override.
    entered_deployed = status.state == "deployed"
    spec = JobSpec.from_dict(status.spec)
    remote = status.remote or {}
    # A deployed run also owns a serving registration with the freesolo serving
    # app that the training-endpoint GC below does not touch; deregister it too so
    # a cancelled run can't leave a deployment registered as active.
    if status.state == "deployed":
        try:
            from flash.serve.deploy import undeploy_adapter

            undeploy_adapter(run_id)
            # Mark the deployment inactive so /v1/deployments and /chat stop treating the
            # cancelled run as active. Delete is idempotent: an already-absent adapter still
            # means the local deployment record can be cleared.
            if status.deployment:
                # Mark the deployment inactive through the lock-guarded path so this write
                # participates in the same _STATUS_LOCK as the rest of the runner. A bare
                # _save_status here would persist a stale pre-teardown snapshot OUTSIDE the
                # lock, bypassing serialization and potentially clobbering a concurrent field
                # update. We mark ONLY the deployment field and leave the run's state alone
                # (no state re-assert): a concurrent mark_undeployed can move the run to
                # terminal `done` between our get_status read and this write, and _update's
                # compare-and-set rejects ANY transition off a terminal state (even a
                # same-field re-assert of the stale `deployed`), which would silently leave
                # the deployment advertised as `ready`. mark_deployment_undeployed flips the
                # deployment regardless of (and without disturbing) the current state.
                mark_deployment_undeployed(run_id)
        except Exception:
            # Best-effort serving teardown: a failure here must not block the cancel
            # below (the run still flips to cancelled and the training endpoint is GC'd).
            pass
    # Durable path first: stop the exact remote worker via the handle's provider
    # (works from any process); endpoint/instance teardown is shared with the GC.
    # Dispatched generically through the registry — never a hardcoded per-provider branch.
    if remote:
        try:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(remote)
            provider = get_provider(handle.provider)
            provider.cancel(handle)
            # Belt-and-suspenders destroy after cancel; RunPod endpoint GC follows.
            provider.destroy(handle)
        except Exception:
            # Best-effort remote stop; _gc_run_endpoints below still tears the endpoint down.
            pass
    _gc_run_endpoints(spec)
    # Final transition to `cancelled`. The run was NON-terminal at entry, but teardown takes
    # time and a terminal state can race in mid-teardown. We must distinguish two cases:
    #
    #   - A concurrent mark_undeployed() (an external `DELETE /v1/runs/{id}/deploy`) flipped a
    #     `deployed` run to terminal `done`. That `done` is NOT a fresh result — it just
    #     restored the run's pre-deploy completion marker while retiring serving. The user
    #     explicitly asked to cancel, so this must be OVERRIDDEN to `cancelled`.
    #   - A genuine training-COMPLETION `done` from the run's own training thread
    #     (_run_job_inner / attach_run), which persisted real metrics+cost+artifacts. Cancel
    #     must NEVER clobber that — the run finished, so the real result is preserved.
    #
    # These two races are mutually exclusive on the entry state: only a `deployed` run owns a
    # deployment that mark_undeployed can race, and only a non-deployed (running/provisioning/
    # queued) run has an in-flight training thread that can complete mid-teardown. So scope the
    # terminal override to runs that were `deployed` at entry — there a racing `done` is always
    # an undeploy artifact (cancel wins); elsewhere a racing `done` is a genuine completion that
    # _update's CAS correctly protects (cancel loses to a real finish).
    _update(run_id, "cancelled", allow_from_terminal=entered_deployed)
    # A run cancelled mid-RL keeps whatever per-step adapters the worker already streamed to
    # HF; mirror them to the backend store now so the cancelled run is immediately listable +
    # deployable (`flash checkpoints` / `flash deploy --step N`). Best-effort: never let
    # checkpoint bookkeeping fail a cancel.
    with contextlib.suppress(Exception):
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(get_status(run_id))
    return get_status(run_id)


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from ANY process (after a client crash/restart).

    Uses the persisted {endpoint_id, job_id} handle to resume polling; on completion,
    persists metrics exactly like the original client would have, flips the state, and
    GCs the endpoint. Raises if the run has no persisted handle (it failed or was
    cancelled before a worker was provisioned).
    """
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
    # The class the run actually provisioned (a policy retry may have walked past the
    # provisional spec.gpu.type). The in-process success path stamps this into metrics;
    # on recovery the worker output carries no such field, so recover it from the handle
    # to cost the right card.
    allocated_gpu = remote.pop("allocated_gpu", None)
    log = log_stream or sys.stderr
    # Dispatch the poll generically via the handle's provider (the provider owns its
    # heartbeat reader + poll loop); the orchestrator stays provider-agnostic.
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    handle = JobHandle.from_dict(remote)
    print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
    res = get_provider(handle.provider).poll(handle, spec, seed, log=log)
    try:
        # A best-effort cancel deletes the job/instance, which the poller reports as a
        # failure (or a late worker may still succeed) — either way, re-read the state
        # first so a recovery thread can't overwrite the user's terminal `cancelled`.
        if get_status(run_id).state == "cancelled":
            return get_status(run_id)
        if not res.ok:
            # Job ended not-ok — usually because it was abandoned during the redeploy. Resume the
            # in-flight seed from its last HF checkpoint instead of failing; the seed loop
            # (unchanged) still terminates a genuinely broken run when it re-fails.
            try:
                seed_index = list(spec.train.seeds).index(seed)
            except ValueError:
                seed_index = 0
            print(
                f"attach: {run_id} seed {seed} ended ({res.failure}); resuming from checkpoint",
                file=log,
            )
            # GC the dead endpoint, then clear the stale handle and record the seed so a second
            # restart mid-allocation resumes the right one.
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            # Bail if the run was raced to terminal during the long poll above: _update's CAS
            # returns False, and resuming would submit paid work for a dead run.
            if not _update(run_id, "running", remote=None, resume_seed_index=seed_index):
                print(f"attach: {run_id} went terminal during recovery; not resuming", file=log)
                return get_status(run_id)
            _run_seed_loop(
                spec, log, start_index=seed_index, prior_cost=float(status.cost_usd or 0.0)
            )
            return get_status(run_id)
        # Carry the provisioned class into metrics so _persist_metrics costs the card the
        # run actually used (the in-process path stamps this; recovery must restore it).
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
        # Earlier seeds of a multi-seed run already persisted their cost into
        # status.cost_usd; add this seed's so recovery doesn't underreport spend.
        total = float(status.cost_usd or 0.0) + _persist_metrics(spec, seed, res.metrics)
        # A cancel can land while this thread persists the recovered seed's metrics
        # (after the late-cancel check above). Re-read before the post-seed writes so
        # the "running" update and the terminal "done" below can't resurrect a
        # user-cancelled run (mirrors the fresh seed loop). _RunCancelled is caught
        # below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        # The remote handle only identifies the seed that was in flight. For a
        # multi-seed run, resume the remaining seeds instead of terminally
        # completing the whole run after just this one.
        try:
            resumed_index = list(spec.train.seeds).index(seed) + 1
        except ValueError:
            resumed_index = len(spec.train.seeds)
        more_seeds = resumed_index < len(spec.train.seeds)
        # Clear the now-stale completed handle before resuming. In the
        # allocation/provisioning gap before the next seed's on_handle() persists a
        # fresh handle, a server restart must not reattach recovery to this finished
        # job — that would double-count its cost and replay the wrong seed. Record the
        # next seed index so a restart in that gap resumes the remaining seeds rather
        # than failing the run. (The last seed keeps its handle for post-run
        # observability, mirroring the fresh-submit seed loop.)
        applied = _update(
            run_id,
            "running",
            cost_usd=total,
            artifacts_dir=artifacts_dir(spec),
            **({"remote": None, "resume_seed_index": resumed_index} if more_seeds else {}),
        )
        # Same TOCTOU guard as the not-ok recovery path: a concurrent thread can flip this
        # run terminal (e.g. failed/done from another recovery) between the cancel re-check
        # above and here. The sticky CAS rejects the `running` write (applied is False) — so
        # don't resume the remaining seeds and submit paid GPU work for an already-terminal
        # run. (The non-multi-seed arm writes the terminal `done`; the CAS protects a racing
        # terminal there too, so no extra guard is needed.)
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
        # Intentional: cancel_run already wrote the terminal `cancelled` state; leave it.
        pass
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        _gc_run_endpoints(spec)
    return get_status(run_id)


def resume_run(run_id: str, log_stream=None) -> RunStatus:
    """Resume the remaining seeds of a multi-seed run after a restart in the inter-seed gap.

    Between two seeds the completed seed's handle is cleared and ``resume_seed_index`` is
    recorded (see ``_run_seed_loop``). A control-plane restart in that handle-less window
    must RESUME from that index rather than fail the run and discard the finished seeds.
    Unlike ``attach_run`` there is no live job to poll — the prior process already tore the
    seed's endpoint down — so we start a fresh seed loop from the recorded index. The flash
    package was uploaded to HF on the original submit, so the worker can still fetch it; no
    re-upload is needed.
    """
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
        # Mirror _run_job: GC any endpoint a transient destroy left behind rather than
        # leaking a billable RunPod endpoint.
        _gc_run_endpoints(spec)
    return get_status(run_id)


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    from flash.runner import _STATUS_LOCK, _UNDEPLOYABLE_STATES, _save_status, get_status

    # Atomic + terminal-respecting (same guard as _update): a /cancel landing during
    # deployment writes `cancelled`; this must NOT overwrite it with
    # `deployed` and resurrect the run as an active deployment. `done` is deployable
    # though (the common case: deploy a finished run), so only the non-`done` terminal
    # states block here — otherwise a freshly finished run could never be deployed.
    #
    # expect_state is a compare-and-set: the deploy flow passes the state it expects the
    # run to still be in (the pre-deploy snapshot, or "deployed" after the provisional
    # mark). If an undeploy raced finalization — deleting the endpoint and writing `done`
    # with deployment.state="undeployed" mid-warmup — the state no longer matches and we
    # refuse to re-advertise the just-deleted endpoint.
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        status.deployment = deployment
        status.state = "deployed"
        status.updated_at = time.time()
        _save_status(status)
        return status


def attach_checkpoint_deployment(run_id: str, deployment: dict) -> RunStatus:
    """Attach a serving deployment to a run WITHOUT changing its training state.

    Used when deploying a specific intermediate checkpoint of a run that never reached
    ``done`` — e.g. one cancelled or failed mid-RL. The checkpoint adapter exists on HF, so it
    can be served, but the run's terminal training outcome (``cancelled``/``failed``) must be
    preserved: flipping it to ``deployed`` would both erase that outcome and make a later
    undeploy wrongly restore it to ``done`` (``mark_undeployed`` sends non-terminal runs to
    ``done``). The deployment is tracked via the ``deployment`` field exactly like a normal
    deploy, so ``/v1/deployments`` lists it and undeploy clears it. Lock-guarded so it
    serializes with a racing deploy/undeploy on the same run.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy (endpoint torn down -> run back to `done`).

    Lock-guarded so it serializes with a racing deploy finalization: the raw read +
    _save_status the endpoint used to do could interleave with mark_deployed and be
    clobbered. With this under the same lock, mark_deployed's expect_state CAS then sees
    the `done`/undeployed write and won't re-advertise the deleted endpoint.
    """
    from flash.runner import _STATUS_LOCK, TERMINAL_STATES, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
        # Record the teardown but don't resurrect a terminal run: undeploying a
        # cancelled/failed run keeps its terminal state (only a live `deployed` run goes
        # back to `done`). `done` is terminal too, so this naturally no-ops the state.
        if status.state not in TERMINAL_STATES:
            status.state = "done"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by ``cancel_run`` to retire a deployed run's serving record. Unlike
    ``mark_undeployed`` (which is a state transition: a live `deployed` run goes back to
    `done`), this never asserts or changes the run state. That matters under the cancel
    race: a concurrent ``mark_undeployed`` may have already moved the run to terminal
    `done`, and ``_update``'s compare-and-set rejects any transition off a terminal state —
    even re-asserting `deployed` to carry the deployment field — which would leave the
    deployment advertised as `ready`. Marking the field directly (lock-guarded for
    serialization) sidesteps the CAS so the deployment reliably ends `undeployed`, while the
    trailing ``cancelled`` transition is left to ``_update``.
    """
    from flash.runner import _STATUS_LOCK, _save_status, get_status

    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
            status.updated_at = time.time()
            _save_status(status)
        return status
