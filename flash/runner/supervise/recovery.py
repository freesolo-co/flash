"""reconcile fenced attempt results, exact cleanup, and terminal accounting."""

from __future__ import annotations

import contextlib
import time

from flash.core.spec import JobSpec


def _lifecycle():
    """The supervisor module, imported lazily because it re-exports this one.

    `_register_checkpoints_best_effort` and `_strict_teardown_handle`
    are patched as attributes of `flash.runner.supervise.lifecycle` by the run-management and
    resume tests, and their callers live here. Resolving them through the parent is what keeps
    those patches effective; a direct call would bind this module's own function.
    """
    from flash.runner.supervise import lifecycle

    return lifecycle


_RUNPOD_STATUS_PROBE_TIMEOUT_S = 10.0


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


def _attempt_result(run_id: str, handle=None):
    """return the current verified fenced terminal result, if visible."""
    from flash.providers.artifacts.attempts import (
        persist_attempt_artifacts,
        poll_result_from_manifest,
        read_attempt_artifacts,
    )
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.status import (
        effective_spec_from_status,
        get_status,
        source_snapshot_from_status,
    )

    status = get_status(run_id)
    attempt = AttemptRecord.from_dict(status.attempt)
    if handle is not None:
        data = _canonical_provider_handle(handle).to_dict()
        if data.get("attempt") != attempt.attempt_id or data.get("fence") != attempt.fence:
            raise RuntimeError(
                "persisted provider handle does not match the current fenced attempt"
            )
    spec = effective_spec_from_status(status)
    artifacts = read_attempt_artifacts(
        spec.train.hf_repo,
        phase=spec.phase,
        run_id=run_id,
        attempt_id=attempt.attempt_id,
        fence=attempt.fence,
        source_snapshot=source_snapshot_from_status(status, required=True),
    )
    persist_attempt_artifacts(run_id, artifacts)
    if artifacts.result is None:
        return None
    return poll_result_from_manifest(artifacts.result)


def _attempt_result_metrics(run_id: str, handle=None) -> dict | None:
    """return metrics only from the current verified fenced success result."""
    result = _attempt_result(run_id, handle)
    return result.metrics if result is not None and result.ok else None


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


def _adopt_completed_attempt(
    run_id: str,
    spec: JobSpec,
    expected_remote: dict | None,
    metrics: dict,
    *,
    log,
    expected_attempt: tuple[int, int] | None = None,
    expected_no_attempt: bool = False,
) -> bool:
    """Finalize a phantom-completed attempt through the expected-remote CAS."""
    from flash.runner.accounting.reconciliation import _compare_and_complete_remote

    applied = _compare_and_complete_remote(
        run_id,
        expected_remote,
        spec,
        metrics,
        expected_attempt=expected_attempt,
        expected_no_attempt=expected_no_attempt,
    )
    if applied:
        _charge_completed_run_by_id(spec.run_id, log)
        _lifecycle()._register_checkpoints_best_effort(spec, log)
    return applied


def _shape_key(candidate) -> tuple[str, str, int]:
    """Retry-bookkeeping identity: a class at a CARD COUNT, not a class.

    Providers report several counts for the same class (2x and 4x H100 are distinct rentable
    shapes), so keying on (provider, gpu) alone would mark every count tried the moment one fails
    and skip a wider shape that fits.
    """
    return (candidate.provider, candidate.gpu, int(getattr(candidate, "gpu_count", 1) or 1))


def _select_candidate(
    candidates, failed_providers: set[str], tried_classes: set[tuple[str, str, int]]
):
    """Pick the next (provider, class) from the cross-provider ranked candidate list.

    Escapes a congested/sick provider cross-provider before walking classes within it, then takes
    the allocator's own ranking.

    That third key is the list position, not a re-priced one. ``allocate()`` first applies any
    authored provider preference, then ranks within one preference rank on the dollars one optimizer
    step costs, so preserving the list also preserves both policies. Re-sorting here on total $/hr
    answered a different question and silently overrode it: for Qwen3.5-0.8B OPD the allocator ranks the
    RTX 5090 cheapest per step, while hourly price picks the slower RTX 4090, so the FIRST paid
    attempt ignored the choice the cost model had just made and ran slower for more money.
    Preserving the incoming order keeps one owner of the cost policy. ``min`` returns the FIRST
    minimal element, so dropping the price keys is what preserves it -- no index key needed.
    """
    return min(
        candidates,
        key=lambda c: (
            c.provider in failed_providers,  # 1) escape providers that already failed this run
            _shape_key(c) in tried_classes,  # 2) then prefer a shape not yet tried
            # 3) ties keep the allocator's cheapest-per-step order, via min's first-wins semantics
        ),
    )


def _projected_retry_class(
    candidates, failed_providers, tried_classes, chosen, *, cache_drop: bool
):
    """The class the NEXT attempt is expected to select, given the failure this one records.

    Mirrors the bookkeeping at the bottom of the retry loop: a cache-drop retry leaves both sets
    untouched (same class, cold), any other retry marks this class tried and its provider failed.
    Only valid off the OOM path, where the escalation floor rewrites the candidate list first.

    Expected, not guaranteed: the next attempt calls ``allocate()`` again, and providers that build
    candidates from live capacity can drop this class or surface a cheaper one. Callers must word it
    as a projection.
    """
    if not candidates:
        return None
    if cache_drop:
        return _select_candidate(candidates, failed_providers, tried_classes)
    return _select_candidate(
        candidates,
        failed_providers | {chosen.provider},
        tried_classes | {_shape_key(chosen)},
    )


def _candidate_usable_vram_gb(candidate) -> float:
    """Run-usable VRAM under the allocator's fit model.

    Use ``combined_vram_gb`` on both sides of OOM escalation. Raw card-count multiplication ignores
    replicated floors and shard efficiency, so it can move a retry to a smaller effective shape. SFT
    can launch fewer ranks than it rents; the allocator stamps that run-specific width, while an
    unstamped candidate preserves the historical all-rented-cards behavior.
    """
    from flash.providers.core.sharding import combined_vram_gb

    rented = int(getattr(candidate, "gpu_count", 1) or 1)
    executed = getattr(candidate, "executed_gpu_count", None)
    return combined_vram_gb(candidate.vram_gb, int(executed) if executed is not None else rented)


def _oom_escalated(candidates, oom_vram_floor: float):
    """Candidates strictly LARGER than the VRAM that just OOM'd. ``oom_vram_floor == 0`` (no prior OOM)
    leaves the list unchanged; otherwise an 80GB OOM leaves only the >80GB classes (a same-size retry
    would just OOM again). EMPTY means the run already OOM'd the largest available class.

    Both sides are measured with `_candidate_usable_vram_gb`, and the floor recorded on OOM uses it
    too -- the filter is only meaningful if the floor and the candidates are on one scale."""
    if not oom_vram_floor:
        return list(candidates)
    return [c for c in candidates if _candidate_usable_vram_gb(c) > oom_vram_floor]


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

    # sweep every configured provider, including runpod (whose gc also reaps attempt-suffixed
    # endpoints the persisted handle cannot name). Gating on available_providers() is what makes
    # this work on a self-hosted plane: an unconfigured provider holds nothing of ours, and
    # calling it would only raise against a credential the operator never set.
    for _prov in available_providers():
        with contextlib.suppress(Exception):
            get_provider(_prov).gc(spec)
