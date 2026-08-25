"""Reconciling a training attempt whose worker is already gone, and settling what it owes.

When a run's process disappears -- preempted, restarted, or torn down mid-flight -- the supervisor
cannot ask it how it finished, so the outcome has to be reconstructed from what the provider and
the output directory still say. `_completed_attempt_metrics` is that reconstruction; the strict
teardown and charge helpers below are what a settled attempt then triggers.

Split out of `flash.runner.supervise.lifecycle` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
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
    from flash.providers.base import JobHandle

    data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    provider = data.get("provider")
    if provider == "runpod":
        from flash.providers.runpod.pods import RunpodPodHandle as RunpodJobHandle

        return JobHandle.from_dict(RunpodJobHandle.from_dict(data).to_dict())
    if provider == "lambda":
        from flash.providers.lambda_.jobs.builders import LambdaJobHandle

        return JobHandle.from_dict(LambdaJobHandle.from_dict(data).to_dict())
    if provider == "vast":
        from flash.providers.vast.jobs.builders import VastJobHandle

        return JobHandle.from_dict(VastJobHandle.from_dict(data).to_dict())
    raise ValueError("persisted provider identity is missing or unsupported")


def _runpod_completed_metrics(
    handle,
    *,
    spec: JobSpec | None = None,
    deadline_at: float | None = None,
    log=None,
) -> dict | None:
    """Recover one RunPod Pod attempt from the same HF artifacts used by live polling."""
    if spec is None or deadline_at is None:
        return None
    try:
        original = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
        canonical = _canonical_provider_handle(original)
        if canonical.provider != "runpod":
            return None
        data = canonical.to_dict()
        metrics = _completed_attempt_metrics(
            spec,
            provider="runpod",
            attempt=int(data["attempt"]),
            launch_floor=float(data["started_ts"]),
            deadline_at=deadline_at,
            log=log,
        )
        if metrics is None:
            return None
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
    from flash.providers import INSTANCE_PROVIDERS, get_provider

    try:
        handle = _canonical_provider_handle(handle)
    except Exception:
        return False
    if handle.provider in INSTANCE_PROVIDERS:
        try:
            check = getattr(get_provider(handle.provider), "run_instances_remaining", None)
            return check is not None and check(run_id) == []
        except Exception:
            return False
    return False


def _strict_teardown_handle(handle, run_id: str) -> bool:
    """Request exact teardown, then prove the captured attempt's worker is gone.

    Returns true only when the provider confirms the complete resource teardown. A provider error
    remains an error even after the worker is absent because provider-owned payload secrets must also
    be proven absent before the durable cleanup target can be cleared.
    """
    from flash.providers import INSTANCE_PROVIDERS, get_provider

    raw = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    handle = _canonical_provider_handle(raw)
    provider = get_provider(handle.provider)
    if handle.provider in INSTANCE_PROVIDERS:
        destroy_error: Exception | None = None
        try:
            provider.destroy(handle)
        except Exception as exc:
            destroy_error = exc
        if destroy_error is not None and handle.provider == "runpod":
            raise RuntimeError(
                "runpod Pod and payload secret teardown could not be confirmed complete"
            ) from destroy_error
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
    if provider not in {"runpod", "vast", "lambda"} or not spec.train.hf_repo:
        return None
    from flash.providers._lifecycle.poll import make_say
    from flash.providers._lifecycle.poll_instance import (
        _TERMINAL_REREAD_RETRIES,
        _TERMINAL_REREAD_WAIT_S,
    )
    from flash.providers._lifecycle.terminal_artifacts import (
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
    from flash.runner import _compare_and_complete_remote

    applied = _compare_and_complete_remote(run_id, expected_remote, spec, metrics)
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
    from flash.providers.base import combined_vram_gb

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


def _await_runpod_completed_metrics(
    last_handle,
    deadline_at,
    *,
    spec: JobSpec | None = None,
    log=None,
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
            recovery_kwargs = {"deadline_at": observation_floor}
            if spec is not None:
                recovery_kwargs["spec"] = spec
            if log is not None:
                recovery_kwargs["log"] = log
            return _lifecycle()._runpod_completed_metrics(last_handle, **recovery_kwargs)
        except _CompletedAttemptPending:
            if check_cancelled is not None:
                check_cancelled()
            time.sleep(_RECOVERY_METRICS_POLL_S)


def _register_checkpoints_best_effort(spec: JobSpec, log) -> None:
    """Mirror a finished run's per-step checkpoints to the backend store (best-effort)."""
    from flash.runner import get_status

    try:
        from flash.server.domain.checkpoints import register_checkpoints_best_effort

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
    from flash.runner import get_status, record_billing_state
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
    """Best-effort teardown of every provider resource a run may have registered."""
    from flash.runner import (
        _drain_cleanup_remotes,
        _remote_resource_identity,
        effective_spec_from_status,
        get_status,
    )

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
            if status.remote.get("provider") == "runpod" and not resource_deleted:
                from flash.runner import _record_cleanup_remote

                _record_cleanup_remote(spec.run_id, status.remote)
        except Exception:
            pass
    from flash.providers import available_providers, get_provider

    # sweep every configured provider, including runpod, after the exact persisted handle path.
    # gating on available_providers() is what makes
    # this work on a self-hosted plane: an unconfigured provider holds nothing of ours, and
    # calling it would only raise against a credential the operator never set.
    for _prov in available_providers():
        with contextlib.suppress(Exception):
            get_provider(_prov).gc(spec)
