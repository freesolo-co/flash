"""Reconciling a training attempt whose worker is already gone, and settling what it owes.

When a run's process disappears -- preempted, restarted, or torn down mid-flight -- the supervisor
cannot ask it how it finished, so the outcome has to be reconstructed from what the provider and
the output directory still say. `_completed_attempt_metrics` is that reconstruction; the strict
teardown and charge helpers below are what a settled attempt then triggers.

Completion evidence and adoption are shared by live supervision and restart recovery.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from flash.core.spec import JobSpec

_RECOVERY_MARKER_GRACE_S = 120.0
_RECOVERY_METRICS_POLL_S = 5.0
_MODAL_STATUS_PROBE_TIMEOUT_S = 10.0


class _CompletedAttemptPending(RuntimeError):
    """A strict success marker exists, but its metrics are not readable yet."""


def _canonical_provider_handle(handle):
    from flash.providers.core.base import JobHandle
    from flash.providers.modal.handles import ModalJobHandle

    data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    return JobHandle.from_dict(ModalJobHandle.from_dict(data).to_dict())


def _completed_worker_metrics(handle, *, deadline_at=None):
    """Recover metrics only from this attempt's verified terminal artifacts."""
    from flash.providers._lifecycle.instances.terminal_artifacts import (
        AttemptIdentity,
        ProbeBudget,
        TerminalKind,
        resolve_terminal_artifacts,
    )
    from flash.providers.artifacts.hf import make_hf_text_reader
    from flash.providers.modal.handles import ModalJobHandle

    try:
        data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
        h = ModalJobHandle.from_dict(data)
        cutoff = (
            deadline_at if deadline_at is not None else time.time()
        ) + _RECOVERY_MARKER_GRACE_S
        read_deadline = min(cutoff, time.time() + _MODAL_STATUS_PROBE_TIMEOUT_S)
        prefix = f"{h.phase}/{h.run_id}"
        marker = make_hf_text_reader(
            h.hf_repo, f"{prefix}/modal_attempt{h.attempt}.json", deadline_at=read_deadline
        )
        metrics = make_hf_text_reader(
            h.hf_repo, f"{prefix}/metrics.json", deadline_at=read_deadline
        )
        resolution = resolve_terminal_artifacts(
            AttemptIdentity(run_id=h.run_id, attempt=h.attempt, launch_floor=h.started_ts),
            read_marker=lambda: marker(force=True),
            read_metrics=lambda: metrics(force=True),
            budget=ProbeBudget(tries=1, wait_s=0, cutoff_at=read_deadline),
        )
        if resolution.kind is TerminalKind.PENDING and time.time() < cutoff:
            raise _CompletedAttemptPending("Modal attempt succeeded; awaiting metrics visibility")
        return resolution.metrics if resolution.kind is TerminalKind.SUCCESS else None
    except _CompletedAttemptPending:
        raise
    except Exception:
        return None


def _worker_provably_gone(run_id: str, handle) -> bool:
    from flash.providers.modal import api

    try:
        h = _canonical_provider_handle(handle)
        from flash.providers.modal.handles import ModalJobHandle, require_scope

        require_scope(ModalJobHandle.from_dict(h.to_dict()))
        instance = api.get_instance(h.to_dict()["instance_id"])
        return instance is None or instance.get("status") == "terminated"
    except Exception:
        return False


def _strict_teardown_handle(handle, run_id: str) -> bool:
    from flash.providers.core.registry import get_provider

    canonical = _canonical_provider_handle(handle)
    get_provider(canonical.provider).destroy(canonical)
    if not _worker_provably_gone(run_id, canonical):
        raise RuntimeError("Modal sandbox termination could not be confirmed")
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
    if provider != "modal" or not spec.train.hf_repo:
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
    from flash.runner.lifecycle.resource_cleanup import _compare_and_complete_remote

    applied = _compare_and_complete_remote(run_id, expected_remote, spec, metrics)
    if applied:
        _charge_completed_run_by_id(spec.run_id, log)
        _register_checkpoints_best_effort(spec, log)
    return applied


def _await_completed_worker_metrics(
    last_handle,
    deadline_at,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> dict | None:
    observation_floor = time.time()
    if deadline_at is not None:
        observation_floor = min(observation_floor, deadline_at)
    while True:
        try:
            return _completed_worker_metrics(last_handle, deadline_at=observation_floor)
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


def _gc_run_sandboxes(spec: JobSpec) -> None:
    """Best-effort teardown of every endpoint a run may have registered."""
    from flash.runner.lifecycle.resource_cleanup import (
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
            resource_deleted = _strict_teardown_handle(status.remote, spec.run_id)
            if resource_deleted:
                _compare_and_confirm_remote_teardown(spec.run_id, status.remote)
            elif status.remote.get("provider") == "modal":
                from flash.runner.lifecycle.resource_cleanup import _record_cleanup_remote

                _record_cleanup_remote(spec.run_id, status.remote)
        except Exception:
            pass
    from flash.providers.core.registry import available_providers, get_provider

    for _prov in available_providers():
        with contextlib.suppress(Exception):
            get_provider(_prov).gc(spec)


def carry_allocation_stamp(metrics: dict, remote: dict | None) -> None:
    """carry the persisted allocation stamp onto adopted metrics."""
    if not isinstance(metrics, dict) or not isinstance(remote, dict):
        return
    allocated_gpu = remote.get("allocated_gpu")
    if allocated_gpu:
        metrics.setdefault("allocated_gpu", allocated_gpu)
    allocated_count = remote.get("allocated_gpu_count")
    if allocated_count:
        metrics.setdefault("allocated_gpu_count", int(allocated_count))
    provider = remote.get("provider")
    if provider:
        metrics.setdefault("allocated_provider", provider)


def completed_metrics_for_remote(
    worker_spec: JobSpec,
    persisted_remote: dict,
    *,
    provider: str,
    attempt: int,
    deadline_at: float,
    log,
) -> dict | None:
    metrics = _completed_worker_metrics(persisted_remote, deadline_at=deadline_at)
    started_ts = persisted_remote.get("started_ts")
    if metrics is None and started_ts is not None:
        metrics = _completed_attempt_metrics(
            worker_spec,
            provider=provider,
            attempt=attempt,
            launch_floor=float(started_ts),
            deadline_at=deadline_at,
            log=log,
        )
    if metrics is not None:
        carry_allocation_stamp(metrics, persisted_remote)
    return metrics
