"""Reconciling a training attempt whose worker is already gone, and settling what it owes.

When a run's process disappears -- preempted, restarted, or torn down mid-flight -- the supervisor
cannot ask it how it finished, so the outcome has to be reconstructed from what the provider and
the output directory still say. `_completed_attempt_metrics` is that reconstruction; the strict
teardown and charge helpers below are what a settled attempt then triggers.

Split out of `flash.runner.supervise.lifecycle` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable

from flash.core.spec import JobSpec


def _lifecycle():
    """The supervisor module, imported lazily because it re-exports this one.

    `_register_checkpoints_best_effort`, `_strict_teardown_handle` and `_runpod_completed_metrics`
    are patched as attributes of `flash.runner.supervise.lifecycle` by the run-management and
    resume tests, and their callers live here. Resolving them through the parent is what keeps
    those patches effective; a direct call would bind this module's own function.
    """
    from flash.runner.supervise import lifecycle

    return lifecycle


_RECOVERY_MARKER_GRACE_S = 120.0
_RECOVERY_METRICS_POLL_S = 5.0
_RUNPOD_STATUS_PROBE_TIMEOUT_S = 10.0


class _CompletedAttemptPending(RuntimeError):
    """A strict success marker exists, but its metrics are not readable yet."""


def _canonical_provider_handle(handle):
    """Validate and canonicalize one complete provider-specific persisted handle."""
    from flash.providers.core.base import JobHandle

    data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    provider = data.get("provider")
    if provider == "runpod":
        from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

        return JobHandle.from_dict(RunpodJobHandle.from_dict(data).to_dict())
    if provider == "lambda":
        from flash.providers.lambda_.jobs.builders import LambdaJobHandle

        return JobHandle.from_dict(LambdaJobHandle.from_dict(data).to_dict())
    if provider == "vast":
        from flash.providers.vast.jobs.builders import VastJobHandle

        return JobHandle.from_dict(VastJobHandle.from_dict(data).to_dict())
    raise ValueError("persisted provider identity is missing or unsupported")


def _runpod_completed_metrics(handle, *, deadline_at: float | None = None) -> dict | None:
    """Return decoded metrics only when the exact RunPod job completed successfully."""
    try:
        original = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
        canonical = _canonical_provider_handle(original)
        data = canonical.to_dict()
        if canonical.provider != "runpod" or not data.get("job_id"):
            return None
        from flash.providers.runpod.client import api as runpod_api
        from flash.providers.runpod.execution.jobs import TERMINAL_OK, decode_output

        # a status probe must fail fast: cap it at a short fresh timeout regardless of how far
        # the run wall deadline is. handing job_status the wall+grace deadline (which can be
        # hours out for a run that just started) lets a runpod api outage burn the full
        # per-request retry budget before returning None, stalling the reconciler each pass.
        # the wall+grace value governs only the pending-output decision below, never the probe.
        probe_deadline_at = (
            time.time() + _RUNPOD_STATUS_PROBE_TIMEOUT_S if deadline_at is not None else None
        )
        job = runpod_api.job_status(
            data["endpoint_id"],
            data["job_id"],
            key_fingerprint=data["key_fingerprint"],
            deadline_at=probe_deadline_at,
        )
        if not isinstance(job, dict) or job.get("status") not in TERMINAL_OK:
            return None
        try:
            metrics = decode_output(job.get("output"))
            output_readable = isinstance(metrics, dict)
        except Exception:
            raw_output = job.get("output")
            if isinstance(raw_output, str):
                # a string-form envelope (json text) is still a READABLE failure if it decodes to one;
                # parse it before falling through to the pending path so a completed-with-failure job
                # is not kept reconciling as if its output were merely lagging.
                try:
                    decoded = json.loads(raw_output)
                except (ValueError, TypeError):
                    decoded = None
                if isinstance(decoded, dict):
                    raw_output = decoded
            if isinstance(raw_output, dict) and (
                raw_output.get("error")
                or ("success" in raw_output and not raw_output.get("success"))
            ):
                # the terminal-ok job's output is a READABLE worker-failure envelope, not
                # lagging success metrics: the attempt definitively completed with a failure.
                # do not raise _CompletedAttemptPending (which would keep reconciling a job
                # that already failed); return None so the caller takes the completed-without-
                # metrics (failed) path.
                return None
            # otherwise the output is present but not yet decodable (unparseable/non-dict):
            # treat it like a missing output below (pending within grace) so a job that
            # already completed is not torn down over a transient output lag.
            metrics = None
            output_readable = False
        if not output_readable:
            # the queue job is terminal-ok but its output metrics are not readable yet
            # (missing, non-dict, or not yet decodable); treat this lag like instance
            # recovery (raise pending) so callers keep reconciling instead of tearing down
            # a job that already completed.
            grace_expired = (
                deadline_at is None or time.time() >= deadline_at + _RECOVERY_MARKER_GRACE_S
            )
            if grace_expired:
                return None
            raise _CompletedAttemptPending(
                "runpod job completed but its output metrics are not readable yet"
            )
        allocated_gpu = original.get("allocated_gpu")
        if allocated_gpu:
            metrics.setdefault("allocated_gpu", allocated_gpu)
        allocated_count = original.get("allocated_gpu_count")
        if allocated_count:
            metrics.setdefault("allocated_gpu_count", int(allocated_count))
        return metrics
    except _CompletedAttemptPending:
        raise
    except Exception:
        return None


def _worker_provably_gone(run_id: str, handle) -> bool:
    """Return true only when the captured attempt cannot still have a live worker."""
    from flash.providers.core.registry import INSTANCE_PROVIDERS, get_provider

    try:
        handle = _canonical_provider_handle(handle)
        data = handle.to_dict()
    except Exception:
        return False
    if handle.provider == "runpod":
        job_id = data.get("job_id")
        if not job_id:
            return False
        try:
            from flash.providers.runpod.client import api as runpod_api
            from flash.providers.runpod.execution.jobs import TERMINAL_FAIL, TERMINAL_OK

            job = runpod_api.job_status(
                data["endpoint_id"],
                job_id,
                key_fingerprint=data["key_fingerprint"],
            )
            return isinstance(job, dict) and job.get("status") in TERMINAL_OK | TERMINAL_FAIL
        except Exception:
            return False
    if handle.provider in INSTANCE_PROVIDERS:
        try:
            check = getattr(get_provider(handle.provider), "run_instances_remaining", None)
            return check is not None and check(run_id) == []
        except Exception:
            return False
    return False


def _delete_runpod_endpoint(data: dict, canonical=None) -> None:
    """Delete one exact RunPod endpoint without trusting the persisted handle's own metadata."""
    from flash.providers.runpod.client import api as runpod_api

    endpoint_id = data.get("endpoint_id")
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise ValueError("persisted RunPod endpoint identity is invalid")

    fingerprint = data.get("key_fingerprint")
    if canonical is not None:
        from flash.providers.core.registry import get_provider

        get_provider("runpod").destroy(canonical)
        return

    owner_resolved = False
    try:
        runpod_api._key_for_fingerprint(fingerprint)
    except runpod_api.RunpodApiError:
        if runpod_api._is_prefix_key_fingerprint(fingerprint):
            try:
                fingerprint = runpod_api.resolve_prefix_key_fingerprint(endpoint_id, fingerprint)
            except runpod_api.RunpodApiError:
                pass
            else:
                owner_resolved = True
    else:
        owner_resolved = True

    if owner_resolved:
        if runpod_api.delete_endpoint_for_fingerprint(endpoint_id, fingerprint):
            return
        if not runpod_api.endpoint_absent_for_fingerprint(endpoint_id, fingerprint):
            raise runpod_api.RunpodApiError(f"runpod endpoint {endpoint_id} deletion unconfirmed")
        return

    by_fingerprint, failed = runpod_api.list_endpoints_by_key(
        deadline_at=time.time() + _RUNPOD_STATUS_PROBE_TIMEOUT_S
    )
    owners = [
        owner_fingerprint
        for owner_fingerprint, endpoints in by_fingerprint.items()
        if any(
            isinstance(endpoint, dict) and endpoint.get("id") == endpoint_id
            for endpoint in endpoints
        )
    ]
    if len(owners) > 1:
        raise runpod_api.RunpodApiError(
            f"runpod endpoint {endpoint_id} appears in multiple accounts; cleanup unconfirmed"
        )
    if not owners:
        # an inventory over the CONFIGURED keys cannot prove absence. this branch is reached only
        # when the persisted fingerprint did not resolve, so the owning credential may simply no
        # longer be in RUNPOD_API_KEY -- "none of my accounts list it" and "it was deleted" are
        # indistinguishable from here. reporting deletion would let the caller drop the cleanup
        # record while the unreachable endpoint stays live and billing, so refuse instead and let
        # the record survive for a later drain that may have the owning key configured again.
        if failed:
            raise runpod_api.RunpodApiError(
                f"runpod endpoint {endpoint_id} owner discovery was incomplete; cleanup unconfirmed"
            )
        raise runpod_api.RunpodApiError(
            f"runpod endpoint {endpoint_id} has no reachable owner account; cleanup unconfirmed"
        )

    if runpod_api.delete_endpoint_for_fingerprint(endpoint_id, owners[0]):
        return
    if not runpod_api.endpoint_absent_for_fingerprint(endpoint_id, owners[0]):
        raise runpod_api.RunpodApiError(f"runpod endpoint {endpoint_id} deletion unconfirmed")


def _strict_teardown_handle(handle, run_id: str) -> bool:
    """Request exact teardown, then prove the captured attempt's worker is gone.

    Returns true when the billable resource deletion itself was confirmed. Returns false only for a
    RunPod job proven terminal while its endpoint deletion remains unconfirmed; callers must persist
    that exact endpoint in cleanup_remotes before clearing the active remote.
    """
    from flash.providers.core.registry import INSTANCE_PROVIDERS, get_provider

    raw = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    if raw.get("provider") == "runpod":
        canonical = None
        with contextlib.suppress(Exception):
            canonical = _canonical_provider_handle(raw)
        if canonical is not None and canonical.to_dict().get("job_id"):
            with contextlib.suppress(Exception):
                get_provider("runpod").cancel(canonical)
        try:
            _delete_runpod_endpoint(raw, canonical)
        except Exception as exc:
            # malformed legacy handles deliberately cannot use the job-status escape hatch: without
            # a strict owner identity, only confirmed endpoint deletion may settle teardown.
            if canonical is not None and _worker_provably_gone(run_id, canonical):
                return False
            raise RuntimeError(
                "runpod endpoint deletion could not be confirmed and its worker may still be live"
            ) from exc
        return True

    handle = _canonical_provider_handle(raw)
    provider = get_provider(handle.provider)
    if handle.provider in INSTANCE_PROVIDERS:
        destroy_error: Exception | None = None
        try:
            provider.destroy(handle)
        except Exception as exc:
            destroy_error = exc
        if _worker_provably_gone(run_id, handle):
            return True
        raise RuntimeError(
            "instance teardown could not be confirmed absent; its worker may still be live"
        ) from destroy_error
    provider.cancel(handle)
    provider.destroy(handle)
    return True


def _completed_attempt_metrics(
    spec: JobSpec,
    *,
    provider: str,
    attempt: int,
    launch_floor: float,
    deadline_at: float,
    log=None,
) -> dict | None:
    """Read a strict successful instance marker plus its run-scoped metrics."""
    if provider not in {"vast", "lambda"} or not spec.train.hf_repo:
        return None
    from flash.providers._lifecycle.instances.poll import make_say
    from flash.providers._lifecycle.instances.poll_instance import (
        _TERMINAL_REREAD_RETRIES,
        _TERMINAL_REREAD_WAIT_S,
    )
    from flash.providers._lifecycle.instances.terminal_artifacts import (
        INVALID_MARKER_DETAIL,
        AttemptIdentity,
        ProbeBudget,
        TerminalKind,
        resolve_terminal_artifacts,
    )
    from flash.providers.artifacts.hf import make_hf_text_reader

    prefix = f"{spec.phase}/{spec.run_id}"
    marker_reader = make_hf_text_reader(
        spec.train.hf_repo,
        f"{prefix}/{provider}_attempt{attempt}.json",
    )
    metrics_reader = make_hf_text_reader(spec.train.hf_repo, f"{prefix}/metrics.json")
    say = make_say(log)
    marker_bound = deadline_at + _RECOVERY_MARKER_GRACE_S
    # ONE observation window for both artifacts. it previously computed a fresh window for the
    # marker and then another fresh one for metrics, so the real ceiling was their sum and moved
    # with however long the marker read took.
    resolution = resolve_terminal_artifacts(
        AttemptIdentity(run_id=spec.run_id, attempt=attempt, launch_floor=launch_floor),
        read_marker=lambda: marker_reader(force=True),
        read_metrics=lambda: metrics_reader(force=True),
        budget=ProbeBudget(
            tries=_TERMINAL_REREAD_RETRIES,
            wait_s=_TERMINAL_REREAD_WAIT_S,
            cutoff_at=time.time() + _TERMINAL_REREAD_RETRIES * _TERMINAL_REREAD_WAIT_S,
        ),
        say=say,
        marker_deadline_at=marker_bound,
        marker_wait_message="recovery deadline reached; waiting for the terminal attempt marker",
        metrics_message="successful recovery marker seen; waiting for metrics.json",
    )
    if resolution.kind is TerminalKind.SUCCESS:
        return resolution.metrics
    if resolution.kind is TerminalKind.UNVERIFIABLE:
        # name it the way live polling does instead of logging it as silence. it is still not
        # completed work, so recovery does not adopt it either way.
        say(f"recovery: {INVALID_MARKER_DETAIL}; not adopting it as completed work")
        return None
    # a success marker landed but its metrics have not. keep reconciling within the grace window
    # rather than tearing down an attempt that already finished its paid work.
    if resolution.kind is TerminalKind.PENDING and time.time() < marker_bound:
        raise _CompletedAttemptPending(
            "successful recovery marker is present but metrics.json is not readable yet"
        )
    return None


def _adopt_completed_attempt(
    run_id: str,
    spec: JobSpec,
    expected_remote: dict | None,
    metrics: dict,
    *,
    log,
) -> bool:
    """Finalize a phantom-completed attempt through the expected-remote CAS."""
    from flash.runner.accounting.reconciliation import _compare_and_complete_remote

    applied = _compare_and_complete_remote(run_id, expected_remote, spec, metrics)
    if applied:
        _charge_completed_run_by_id(spec.run_id, log)
        _lifecycle()._register_checkpoints_best_effort(spec, log)
    return applied


def _await_runpod_completed_metrics(
    last_handle,
    deadline_at,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict | None:
    # a terminal-ok runpod job whose output metrics are not decodable yet raises
    # _CompletedAttemptPending within the grace window; keep reconciling (like attach_run
    # and background reconciliation) instead of letting it escape the supervisor and fail
    # a job that already completed.
    #
    # bound the wait to a short observation window measured from first observation (never past the
    # run wall), not the full run wall deadline. callers pass the run wall deadline, up to
    # max_wall_seconds (default 24h); passing it straight through let a terminal-ok job whose output
    # never became readable pin the supervisor for the remainder of the run instead of failing over
    # to a retry. _runpod_completed_metrics returns None once time >= observation_floor +
    # _RECOVERY_MARKER_GRACE_S, so this synchronous poll is bounded to ~the grace window.
    observation_floor = time.time()
    if deadline_at is not None:
        observation_floor = min(observation_floor, deadline_at)
    while True:
        try:
            return _lifecycle()._runpod_completed_metrics(
                last_handle, deadline_at=observation_floor
            )
        except _CompletedAttemptPending:
            if check_cancelled is not None:
                check_cancelled()
            time.sleep(_RECOVERY_METRICS_POLL_S)


def _register_checkpoints_best_effort(spec: JobSpec, log) -> None:
    """Mirror a finished run's per-step checkpoints to the backend store (best-effort)."""
    from flash.runner.lifecycle.status import get_status

    try:
        from flash.server.domain.registry.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(get_status(spec.run_id), log=log)
    except Exception as exc:  # never let checkpoint bookkeeping disturb a run
        print(
            f"[ckpt] register warn ({spec.run_id}): {type(exc).__name__}",
            file=log,
            flush=True,
        )


def _charge_completed_run_by_id(run_id: str, log) -> None:
    """Bill a completed external run by run id, without changing its training result.

    The charge reads everything it needs from the persisted ``RunStatus`` (``billing_context`` +
    ``cost_usd`` + the raw ``spec`` dict), so a run id is the only input. The retry sweep calls this
    directly so a legacy/stale persisted spec that ``JobSpec.from_dict`` would reject does NOT block
    recovery of a real pending/failed charge."""
    from flash.server.billing.charges import charge_completed_run

    _apply_charge_with_state(
        run_id,
        log,
        # "terminal" not "completed": the retry sweep bills cancelled-mid-training runs through here
        # too, so the not-billed log/error text must read accurately for both.
        noun="terminal",
        charge_call=lambda internal_key, status: charge_completed_run(
            internal_key=internal_key, status=status
        ),
    )


def _apply_charge_with_state(run_id: str, log, *, charge_call, noun: str) -> None:
    """Drive the billing state machine around one charge attempt (charging -> charged/failed).

    ``charge_call(internal_key, status)`` performs the actual backend charge and returns its response
    dict. Reading org/cost from the
    persisted ``RunStatus`` (never a reparsed spec) is what lets a legacy/stale spec still be charged.
    """
    from flash.runner.accounting.costs import record_billing_state
    from flash.runner.lifecycle.status import get_status
    from flash.server.billing.charges import BillingError
    from flash.server.platform.auth import INTERNAL_KEY_ENV, standalone
    from flash.server.platform.internal_client import internal_key as operator_internal_key

    status = get_status(run_id)
    if not status.billing_context or status.billing_state == "charged":
        return

    # The shared gate, not a raw env read: it also returns None in standalone mode. A standalone
    # plane started against an existing state directory can hold a run that still carries a
    # managed-mode `billing_context`; charging it would send the operator's key to
    # FREESOLO_BASE_URL (the hosted default when unset) and bill an organization this plane has no
    # relationship with. Every other backend reporter is gated the same way, so billing turns off
    # as a whole rather than one caller at a time.
    internal_key = operator_internal_key()
    if not internal_key:
        # Name both causes: in standalone the key IS set, and reporting it as missing would send
        # an operator looking for configuration that is already correct.
        cause = (
            "standalone mode has no billing backend"
            if standalone()
            else f"{INTERNAL_KEY_ENV} is not configured"
        )
        detail = f"{cause}; {noun} run was not billed"
        # Field-only billing write that re-reads state under the lock: never overwrite a `deployed`
        # that a concurrent /deploy may have written since we last read the run.
        record_billing_state(run_id, billing_state="failed", billing_error=detail)
        print(f"billing failed: {detail}", file=log, flush=True)
        return

    record_billing_state(run_id, billing_state="charging", billing_error=None)
    status = get_status(run_id)
    try:
        charge = charge_call(internal_key, status)
    except BillingError as exc:
        record_billing_state(run_id, billing_state="failed", billing_error=exc.detail)
        print(f"billing failed: {exc.detail}", file=log, flush=True)
        return

    record_billing_state(
        run_id,
        billing_state="charged",
        billing_error=None,
        billing_charge=charge,
    )
    print(
        f"billing charged: amount_cents={charge.get('amountCents')} "
        f"replay={bool(charge.get('replay'))}",
        file=log,
        flush=True,
    )


def _gc_run_endpoints(spec: JobSpec) -> None:
    """Best-effort teardown of every endpoint a run may have registered."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_confirm_remote_teardown,
        _drain_cleanup_remotes,
        _remote_resource_identity,
    )
    from flash.runner.lifecycle.status import effective_spec_from_status, get_status

    attempted_cleanup = set()
    with contextlib.suppress(Exception):
        attempted_cleanup = _drain_cleanup_remotes(spec.run_id)
    status = None
    with contextlib.suppress(Exception):
        status = get_status(spec.run_id)
    if status is not None:
        with contextlib.suppress(Exception):
            spec = effective_spec_from_status(status)
    if (
        status is not None
        and status.remote
        and _remote_resource_identity(status.remote) not in attempted_cleanup
    ):
        try:
            resource_deleted = _lifecycle()._strict_teardown_handle(status.remote, spec.run_id)
            if resource_deleted:
                _compare_and_confirm_remote_teardown(spec.run_id, status.remote)
            elif status.remote.get("provider") == "runpod":
                from flash.runner.accounting.reconciliation import _record_cleanup_remote

                _record_cleanup_remote(spec.run_id, status.remote)
        except Exception:
            pass
    from flash.providers.core.registry import available_providers, get_provider

    # Sweep every CONFIGURED provider, including RunPod (whose gc also reaps the other attempts'
    # endpoints the persisted handle cannot name). Gating on available_providers() is what makes
    # this work on a self-hosted plane: an unconfigured provider holds nothing of ours, and
    # calling it would only raise against a credential the operator never set.
    for _prov in available_providers():
        with contextlib.suppress(Exception):
            get_provider(_prov).gc(spec)
